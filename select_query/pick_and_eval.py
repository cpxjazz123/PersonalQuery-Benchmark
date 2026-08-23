#!/usr/bin/env python3
"""Query 选择 + 质量评估 主脚本（10K 流水线）。

阶段 1: pick_best — 从 K 候选选最优（Mahalanobis 距离）
阶段 2: eval_quality — query 质量分析（attrs 覆盖 / hallucination / 多样性）
阶段 3: eval_rank1 — 池内 rank-1 评估

路径全部硬编码，不接受 CLI 参数。
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────
# 硬编码路径
# ──────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

RECORDS_IN = REPO_ROOT / "result/query_records_with_query_inject_strict_10k.json"
RESIDUAL_NPZ = SCRATCH / "residual_hidden_10k.npz"
PCA_OUT = SCRATCH / "pca_components_10k.npz"
BEST_CANDS_OUT = SCRATCH / "best_cands_10k.json"

# ──────────────────────────────────────────────────────────────────────────
# 硬编码超参（Phase 35.G/I-v2 SOTA）
# ──────────────────────────────────────────────────────────────────────────
LAYERS = [26]
PCA_D = 32
TAU = 0.5
QWEN_BATCH = 16
MAX_INPUT_LENGTH = 256


# ══════════════════════════════════════════════════════════════════════════
# 阶段 1: pick_best — 从 K 候选选最优
# ══════════════════════════════════════════════════════════════════════════
def stage1_pick_best() -> list[dict]:
    t0 = time.time()
    print("\n[Stage 1] Pick best candidate from K candidates")
    print("=" * 60)

    # Load residuals
    if not RESIDUAL_NPZ.exists():
        raise FileNotFoundError(f"{RESIDUAL_NPZ} 不存在")
    cache = np.load(RESIDUAL_NPZ, allow_pickle=True)
    LAYER = LAYERS[0]
    residual_key = f"residual_layer_{LAYER}"
    all_resids = cache[residual_key]
    sentences = cache["sentences"].tolist() if "sentences" in cache.files else None
    print(f"[load] residuals shape={all_resids.shape}")

    # Build uid → residuals mapping
    sent_to_uid: dict = {}
    if sentences:
        sents_path = SCRATCH / "sentences_for_rewrite_10k.jsonl"
        if sents_path.exists():
            with open(sents_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        sent_to_uid[d["sentence_text"]] = d["user_id"]
                    except Exception:
                        continue
            print(f"[load] {len(sent_to_uid)} sentence→uid mappings")

    user_resids: dict = {}
    if sentences and sent_to_uid:
        for i, sent in enumerate(sentences):
            uid = sent_to_uid.get(sent)
            if uid:
                user_resids.setdefault(uid, []).append(all_resids[i])
    else:
        user_resids["__legacy__"] = [all_resids[i] for i in range(len(all_resids))]
    print(f"[load] {len(user_resids)} users, total {len(all_resids)} residuals")

    # PCA
    if PCA_OUT.exists():
        pca_data = np.load(PCA_OUT, allow_pickle=True)
        components = {int(k): v for k, v in pca_data["components"].item().items()}
        means = {int(k): v for k, v in pca_data["means"].item().items()}
    else:
        from sklearn.decomposition import PCA
        X = all_resids.astype(np.float32)
        pca = PCA(n_components=PCA_D, random_state=42)
        pca.fit(X)
        components = {PCA_D: pca.components_.astype(np.float32)}
        means = {PCA_D: pca.mean_.astype(np.float32)}
        np.savez_compressed(PCA_OUT,
                            components=np.array([{PCA_D: components[PCA_D]}], dtype=object),
                            means=np.array([{PCA_D: means[PCA_D]}], dtype=object))
        print(f"[PCA] explained_var={pca.explained_variance_ratio_.sum():.3f}")

    # Project users
    user_z = {}
    for uid, vecs in user_resids.items():
        V = np.stack(vecs, axis=0).astype(np.float32)
        Z = (V - means[PCA_D]) @ components[PCA_D].T
        user_z[uid] = Z
    user_mu = {uid: Z.mean(axis=0) for uid, Z in user_z.items()}
    var_global = np.stack([Z.var(axis=0) for Z in user_z.values()]).mean(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)
    print(f"[sigma] mean_var={var_global.mean():.4f}")

    # Load candidates
    if not RECORDS_IN.exists():
        raise FileNotFoundError(f"{RECORDS_IN} 不存在")
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    print(f"[load] {len(records)} records")

    cands_flat = []
    for rec_idx, r in enumerate(records):
        ac = r.get("all_candidates") or [r.get("y_plus_query_inject")] if r.get("y_plus_query_inject") else []
        if not ac:
            continue
        for k_idx, q in enumerate(ac):
            cands_flat.append({
                "rec_idx": rec_idx, "user_id": r["user_id"], "asin": r["asin"],
                "attrs_used": r.get("attrs_used", {}), "query": q, "k": k_idx,
            })
    print(f"[cands] {len(cands_flat)} candidates")

    # Encode queries
    unique_queries = {}
    for c in cands_flat:
        unique_queries.setdefault(c["query"], len(unique_queries))
    print(f"[encode] {len(unique_queries)} unique queries")

    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    query_residual = {}
    queries_list = list(unique_queries.keys())
    for i in range(0, len(queries_list), QWEN_BATCH):
        chunk = queries_list[i:i + QWEN_BATCH]
        hidden_dict = client.get_hidden_states(chunk, layers=LAYERS, max_length=MAX_INPUT_LENGTH, batch_size=QWEN_BATCH)
        h_layer = hidden_dict[LAYERS[0]]
        if hasattr(h_layer, "cpu"):
            h_layer = h_layer.cpu().numpy()
        for q, h in zip(chunk, h_layer):
            query_residual[q] = h
        done = min(i + QWEN_BATCH, len(queries_list))
        print(f"  [encode] {done}/{len(queries_list)}", flush=True)

    query_z = {q: (h.astype(np.float32) - means[PCA_D]) @ components[PCA_D].T for q, h in query_residual.items()}

    def maha_score(z32: np.ndarray, uid: str) -> float:
        mu = user_mu.get(uid)
        if mu is None:
            return -1e9
        diffs = z32 - mu
        d = float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))
        return -d / TAU

    by_record: dict = {}
    for i, c in enumerate(cands_flat):
        by_record.setdefault(c["rec_idx"], []).append(i)

    results = []
    for rec_idx, idxs in by_record.items():
        scores = np.array([maha_score(query_z[cands_flat[i]["query"]], cands_flat[i]["user_id"]) for i in idxs])
        best_i = idxs[int(np.argmax(scores))]
        results.append({
            "user_id": cands_flat[best_i]["user_id"],
            "asin": cands_flat[best_i]["asin"],
            "best_k": cands_flat[best_i]["k"],
            "best_query": cands_flat[best_i]["query"],
            "best_score": float(scores.max()),
            "all_scores": [float(s) for s in scores],
            "n_cands": len(idxs),
            "attrs_used": cands_flat[best_i]["attrs_used"],
        })

    json.dump(results, open(BEST_CANDS_OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[Stage 1] → {BEST_CANDS_OUT} ({len(results)} records, {time.time()-t0:.1f}s)")
    return results


# ══════════════════════════════════════════════════════════════════════════
# 阶段 2: eval_quality — query 质量分析
# ══════════════════════════════════════════════════════════════════════════
HALLUCINATION_WORDS = [
    "bottle", "bottles", "clothing", "clothes", "toy", "toys", "candle", "candles",
    "tumbler", "tumblers", "carrier", "carriers", "basket", "baskets", "socks",
    "shoes", "shirt", "shirts", "pants", "dress", "dresses", "blanket", "blankets",
    "pillow", "pillows", "diaper", "diapers", "wipes", "formula", "pacifier",
    "stroller", "highchair", "swaddle", "swaddles", "romper", "rompers",
    "onesie", "onesies", "mittens", "booties", "teether", "teethers",
    "mobile", "mobiles", "nightlight", "soap", "lotion", "shampoo", "brush", "comb",
    "scented", "scent", "fragrance", "aroma", "decorative", "decoration",
    "friend", "friends", "sister", "brother", "wife", "husband", "mom", "dad",
    "grandma", "grandpa", "aunt", "uncle", "neighbor",
    "lightweight", "heavyweight", "premium", "luxury", "cheap",
    "accessory", "accessories", "tool", "tools", "decor", "gadget",
]


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs:
        return 0
    text_lower = text.lower()
    n = 0
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s and s.lower() in text_lower:
            n += 1
    return n


def jaccard(a: str, b: str) -> float:
    a_s = set(a.lower().split())
    b_s = set(b.lower().split())
    if not a_s and not b_s:
        return 0.0
    return len(a_s & b_s) / len(a_s | b_s)


def stage2_eval_quality(results: list[dict]) -> None:
    t0 = time.time()
    print("\n[Stage 2] Query quality evaluation")
    print("=" * 60)

    n = len(results)
    cov_rates, hallu_counts, query_lens = [], [], []
    all_cands_hallu = []
    opener_counter = Counter()
    broken_outputs = 0

    for r in results:
        q = r.get("best_query", "")
        attrs = r.get("attrs_used", {})
        cov = count_attrs_covered(q, attrs)
        cov_rates.append(cov / max(len(attrs), 1))

        text_lower = q.lower()
        hallu = sum(1 for w in HALLUCINATION_WORDS
                     if re.search(rf"\b{re.escape(w)}\b", text_lower))
        hallu_counts.append(hallu)

        wl = len(q.split())
        query_lens.append(wl)
        if wl:
            hallu_per_word = hallu / wl
        else:
            hallu_per_word = 0
        all_cands_hallu.append(hallu_per_word)

        # Broken output check
        if re.search(r'["\(\)\[""]', q) or len(q.strip()) < 3:
            broken_outputs += 1

        words = q.lower().split()
        if words:
            opener_counter[words[0]] += 1

    # K-candidate diversity
    records_raw = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    diversities = []
    for rec in records_raw:
        cands = rec.get("all_candidates") or []
        if len(cands) >= 2:
            scores = [jaccard(cands[i], cands[j])
                      for i in range(len(cands)) for j in range(i + 1, len(cands))]
            diversities.append(1 - np.mean(scores) if scores else 0)

    # Summary
    print(f"\n  n_records              : {n}")
    print(f"  attrs_coverage_mean    : {np.mean(cov_rates):.3f}")
    print(f"  attrs_coverage_full    : {sum(1 for c in cov_rates if c >= 1.0)} ({sum(1 for c in cov_rates if c >= 1.0)/max(n,1)*100:.1f}%)")
    print(f"  hallucination_mean     : {np.mean(hallu_counts):.2f}")
    print(f"  hallucination_zero    : {sum(1 for h in hallu_counts if h == 0)} ({sum(1 for h in hallu_counts if h == 0)/max(n,1)*100:.1f}%)")
    print(f"  query_len_chars_mean  : {np.mean([len(r.get('best_query','')) for r in results]):.1f}")
    print(f"  query_len_words_mean   : {np.mean(query_lens):.1f}")
    print(f"  broken_outputs         : {broken_outputs} ({broken_outputs/max(n,1)*100:.1f}%)")
    print(f"  cand_diversity_mean    : {np.mean(diversities):.3f}" if diversities else "  cand_diversity_mean: N/A")

    # Top openers
    top_openers = opener_counter.most_common(5)
    print(f"\n  Top openers:")
    for word, cnt in top_openers:
        print(f"    {word!r:20s} {cnt:5d} ({cnt/n*100:.1f}%)")

    print(f"\n[Stage 2] done ({time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# 阶段 3: eval_rank1 — 池内 rank-1 评估
# ══════════════════════════════════════════════════════════════════════════
def stage3_eval_rank1(results: list[dict]) -> None:
    t0 = time.time()
    print("\n[Stage 3] Intra-pool rank-1 evaluation")
    print("=" * 60)

    # Load PCA
    pca_data = np.load(PCA_OUT, allow_pickle=True)
    components = {int(k): v for k, v in pca_data["components"].item().items()}
    means = {int(k): v for k, v in pca_data["means"].item().items()}

    # Build user residual profiles from residuals.npz
    cache = np.load(RESIDUAL_NPZ, allow_pickle=True)
    all_resids = cache[f"residual_layer_{LAYERS[0]}"]
    sentences = cache["sentences"].tolist()

    sent_to_uid: dict = {}
    sents_path = SCRATCH / "sentences_for_rewrite_10k.jsonl"
    if sents_path.exists():
        with open(sents_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    sent_to_uid[d["sentence_text"]] = d["user_id"]
                except Exception:
                    continue

    user_resids: dict = {}
    for i, sent in enumerate(sentences):
        uid = sent_to_uid.get(sent)
        if uid:
            user_resids.setdefault(uid, []).append(all_resids[i])

    user_z = {}
    for uid, vecs in user_resids.items():
        V = np.stack(vecs, axis=0).astype(np.float32)
        Z = (V - means[PCA_D]) @ components[PCA_D].T
        user_z[uid] = Z
    user_mu = {uid: Z.mean(axis=0) for uid, Z in user_z.items()}
    var_global = np.stack([Z.var(axis=0) for Z in user_z.values()]).mean(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)

    uid_ids = sorted(user_mu.keys())
    uid_mat = np.stack([user_mu[u] for u in uid_ids], axis=0)
    norms = np.linalg.norm(uid_mat, axis=1, keepdims=True) + 1e-8
    uid_mat_norm = uid_mat / norms
    uid_to_idx = {u: i for i, u in enumerate(uid_ids)}

    # Encode best queries
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    unique_queries = list(set(r["best_query"] for r in results))
    print(f"[encode] {len(unique_queries)} unique queries")
    query_z = {}
    for i in range(0, len(unique_queries), QWEN_BATCH):
        chunk = unique_queries[i:i + QWEN_BATCH]
        hidden_dict = client.get_hidden_states(chunk, layers=LAYERS, max_length=MAX_INPUT_LENGTH, batch_size=QWEN_BATCH)
        h = hidden_dict[LAYERS[0]]
        if hasattr(h, "cpu"):
            h = h.cpu().numpy()
        for q, hv in zip(chunk, h):
            query_z[q] = ((hv.astype(np.float32) - means[PCA_D]) @ components[PCA_D].T)

    # Retrieval eval: each best_query targets its own user
    ranks = []
    hit10 = 0
    mrr_sum = 0.0

    for r in results:
        q = r["best_query"]
        uid = r["user_id"]
        if q not in query_z or uid not in uid_to_idx:
            continue
        q_norm = query_z[q] / (np.linalg.norm(query_z[q]) + 1e-8)
        scores = (uid_mat_norm * q_norm).sum(axis=1)
        sorted_idx = np.argsort(-scores)
        sorted_uids = [uid_ids[i] for i in sorted_idx]
        rank = sorted_uids.index(uid) + 1
        ranks.append(rank)
        if rank <= 10:
            hit10 += 1
        mrr_sum += 1.0 / rank

    n = len(ranks)
    hit10_rate = hit10 / n if n else 0
    mrr = mrr_sum / n if n else 0
    mean_rank = np.mean(ranks) if ranks else float("inf")
    median_rank = np.median(ranks) if ranks else float("inf")

    print(f"\n  n_eval                 : {n}")
    print(f"  Hit@10                 : {hit10_rate:.3f} ({hit10}/{n})")
    print(f"  MRR                    : {mrr:.4f}")
    print(f"  Mean Rank              : {mean_rank:.1f}")
    print(f"  Median Rank            : {median_rank:.1f}")
    print(f"\n[Stage 3] done ({time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    t0 = time.time()
    results = stage1_pick_best()
    stage2_eval_quality(results)
    stage3_eval_rank1(results)
    print(f"\n[Total] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
