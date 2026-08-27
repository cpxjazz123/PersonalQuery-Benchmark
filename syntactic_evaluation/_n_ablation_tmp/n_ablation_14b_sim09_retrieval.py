"""14B N-ablation Stage 5 retrieval with cosine ≥0.9 cluster selection。

新增约束:同一 ASIN 上评估用的 10 个 unique queries 之间 cosine 相似度必须 ≥0.9。
Greedy:从 seed query 开始,选 cosine ≥0.9 的 queries,直到凑够 K_SELECT=10。
若不足 10,ASIN 被排除。

数据:
- N=5, N=7 sub-set: /home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_full/pool_N{5,7}.json
- N=10 full-pool:    /home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full/pool_N10.json
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

from syntax_subspace_utils import (
    ASIN_TO_DOC_CACHE, MINILM_CORPUS_EMBEDS_CACHE, log,
)

# 配置
SEED = 2024
K_SELECT = 10
SIM_THRESHOLD = 0.9  # 同一 ASIN 内 pairwise cosine ≥ 0.9
N_LIST = [5, 7]  # sub-set
N10_POOL_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full")
SUB_POOL_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_full")


def get_pool_path(N):
    if N == 10:
        return N10_POOL_DIR / f"pool_N{N}.json"
    return SUB_POOL_DIR / f"pool_N{N}.json"


def greedy_select_queries(embeddings, threshold=0.9, k=10, seed_rng=None):
    """从 N 个 query 中选 k 个,要求 cluster 内 **all-pairs** cosine ≥ threshold。

    Greedy with strict all-pairs constraint:
    - 试每个 seed
    - 每次从剩余候选中选一个,该候选必须与已选所有 query 都 ≥ threshold
    - 若候选都无法满足,终止
    - 凑齐 k 后,记 cluster min pairwise cosine
    - 全 seed 取 cluster min sim 最大的 cluster

    Returns: selected indices or None if cannot form k.
    """
    n = len(embeddings)
    if n < k:
        return None

    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T  # [n, n]

    best_set = None
    best_min_sim = -1.0

    for seed_idx in range(n):
        selected = [seed_idx]
        remaining = [i for i in range(n) if i != seed_idx]
        ok = True

        while len(selected) < k:
            # 与已选所有 query 都 ≥ threshold 的候选
            sim_to_sel = sim[selected]  # [len(selected), n]
            mask = (sim_to_sel >= threshold).all(axis=0)  # [n]
            mask[selected] = False
            cands = [i for i in remaining if mask[i]]
            if not cands:
                ok = False
                break
            # 选 mean cosine 最大的
            mean_scores = {c: float(sim_to_sel[:, c].mean()) for c in cands}
            best = max(mean_scores, key=mean_scores.get)
            selected.append(best)
            remaining.remove(best)

        if ok and len(selected) == k:
            sub = sim[np.ix_(selected, selected)]
            min_sim = float(sub[np.triu_indices(k, k=1)].min())
            if min_sim > best_min_sim:
                best_min_sim = min_sim
                best_set = selected

    return best_set


def main():
    rng = random.Random(SEED)
    log("=== 14B Stage 5 with cosine ≥0.9 cluster selection ===")
    corpus = json.load(open(ASIN_TO_DOC_CACHE))
    asin_list = sorted(corpus.keys())
    asin_to_pos = {a: i for i, a in enumerate(asin_list)}

    doc_embeds = np.load(MINILM_CORPUS_EMBEDS_CACHE)
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    doc_gpu = torch.from_numpy(doc_embeds).to(device)

    import bm25s
    log("  building BM25 index...")
    corpus_texts = [corpus[a] for a in asin_list]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens, show_progress=False)

    summary_per_N = {}

    # 收集所有 ASINs 跨所有 N,一次性 encode 全部 strict queries
    asin_to_strict_queries = {}  # asin -> list of (N, query_idx, query_text)
    asin_to_pool_N = {}  # asin -> N

    for N in N_LIST + [10]:
        pool = json.load(open(get_pool_path(N)))
        for asin, qs in pool["pools"].items():
            strict_qs = [q for q in qs if q["strict"]]
            if len(strict_qs) < K_SELECT:
                continue
            for q in strict_qs:
                asin_to_strict_queries.setdefault((N, asin), []).append(
                    (q["k"], q["query"]))
            asin_to_pool_N[(N, asin)] = N

    # 收集所有 unique query strings 去重 encode (节省 GPU)
    all_query_texts = set()
    for (N, asin), qs in asin_to_strict_queries.items():
        for k, q_text in qs:
            all_query_texts.add(q_text)
    all_query_texts = sorted(all_query_texts)
    log(f"  unique query strings: {len(all_query_texts)}")

    # Encode
    qtext_to_emb = {}
    BATCH = 256
    log(f"  MiniLM encoding...")
    for i in range(0, len(all_query_texts), BATCH):
        batch = all_query_texts[i:i+BATCH]
        embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        for t, e in zip(batch, embs):
            qtext_to_emb[t] = e.cpu().numpy()

    # Greedy cluster per (N, asin)
    asin_to_selected_queries = {}  # (N, asin) -> [query_text, ...]
    asin_to_cluster_sim = {}  # 记录 cluster min sim

    n_dropped_sim = 0
    for (N, asin), qs in asin_to_strict_queries.items():
        embs = np.stack([qtext_to_emb[t] for k, t in qs])
        selected_idx = greedy_select_queries(embs, threshold=SIM_THRESHOLD, k=K_SELECT)
        if selected_idx is None:
            n_dropped_sim += 1
            continue
        selected_qs = [qs[i] for i in selected_idx]
        asin_to_selected_queries[(N, asin)] = [t for k, t in selected_qs]
        # 算 cluster min sim
        sel_embs = embs[selected_idx]
        normed = sel_embs / (np.linalg.norm(sel_embs, axis=1, keepdims=True) + 1e-12)
        sim_mx = normed @ normed.T
        np.fill_diagonal(sim_mx, 0)
        asin_to_cluster_sim[(N, asin)] = float(sim_mx.max())  # 任意两 query 的 max cosine

    log(f"  dropped (cos<0.9): {n_dropped_sim} ASINs")

    # 按 N 分组,跑 retrieval + flip
    for N in N_LIST + [10]:
        log(f"\n=== N={N} ===")
        asins_in_N = [a for (n, a) in asin_to_selected_queries if n == N]
        if not asins_in_N:
            log(f"  no ASINs, skip")
            continue

        all_queries = []
        for a in asins_in_N:
            for q_text in asin_to_selected_queries[(N, a)]:
                all_queries.append((a, q_text))
        log(f"  ASINs: {len(asins_in_N)}, queries: {len(all_queries)}")

        if len(all_queries) < 100:
            log(f"  too few, skip")
            continue

        q_texts = [q[1] for q in all_queries]
        q_tokens = bm25s.tokenize(q_texts, stopwords="en", show_progress=False)
        bm_results, _ = bm25_retriever.retrieve(q_tokens, k=200, show_progress=False)

        q_embeds = torch.from_numpy(np.stack([qtext_to_emb[t] for t in q_texts])).to(device)

        log(f"  MiniLM: full sims...")
        bs = 256
        mlm_ranks = []
        for i in range(0, len(q_texts), bs):
            chunk = q_embeds[i:i+bs] @ doc_gpu.T
            chunk_np = chunk.cpu().numpy()
            for r, row in enumerate(chunk_np):
                tp = asin_to_pos.get(all_queries[i+r][0])
                if tp is None:
                    mlm_ranks.append(None); continue
                target_score = row[tp]
                rank = int((row > target_score).sum()) + 1
                mlm_ranks.append(rank)

        per_q_bm_rank = []
        for idx, (asin, text) in enumerate(all_queries):
            tp = asin_to_pos.get(asin)
            if tp is None:
                per_q_bm_rank.append(None); continue
            pos = np.where(bm_results[idx] == tp)[0]
            per_q_bm_rank.append(int(pos[0] + 1) if len(pos) else None)

        asin_groups = collections.defaultdict(list)
        for idx, (asin, text) in enumerate(all_queries):
            asin_groups[asin].append({
                "bm_rank": per_q_bm_rank[idx],
                "mlm_rank": mlm_ranks[idx],
                "n_tok": len(text.split()),
            })

        per_asin_flip = []
        per_asin_rr_std = []
        for asin, qs in asin_groups.items():
            n_pairs = len(qs) * (len(qs) - 1) // 2
            if n_pairs == 0: continue
            bm_dis = 0; ml_dis = 0
            for i in range(len(qs)):
                for j in range(i+1, len(qs)):
                    bm_hi = 1 if (qs[i]["bm_rank"] is not None and qs[i]["bm_rank"] <= 10) else 0
                    bm_hj = 1 if (qs[j]["bm_rank"] is not None and qs[j]["bm_rank"] <= 10) else 0
                    if bm_hi != bm_hj: bm_dis += 1
                    ml_hi = 1 if (qs[i]["mlm_rank"] is not None and qs[i]["mlm_rank"] <= 10) else 0
                    ml_hj = 1 if (qs[j]["mlm_rank"] is not None and qs[j]["mlm_rank"] <= 10) else 0
                    if ml_hi != ml_hj: ml_dis += 1
            # RR Std: std of 1/rank across queries within this ASIN
            bm_rrs = [1.0/r for r in [q["bm_rank"] for q in qs] if r is not None and r > 0]
            ml_rrs = [1.0/r for r in [q["mlm_rank"] for q in qs] if r is not None and r > 0]
            bm_rr_std = float(np.std(bm_rrs, ddof=0)) if len(bm_rrs) >= 2 else None
            ml_rr_std = float(np.std(ml_rrs, ddof=0)) if len(ml_rrs) >= 2 else None
            per_asin_flip.append({
                "asin": asin, "n_q": len(qs), "n_pairs": n_pairs,
                "bm_flip": bm_dis / n_pairs, "mlm_flip": ml_dis / n_pairs,
            })
            if bm_rr_std is not None and ml_rr_std is not None:
                per_asin_rr_std.append({"asin": asin, "bm_rr_std": bm_rr_std, "ml_rr_std": ml_rr_std})

        n_asins_eff = len(per_asin_flip)
        bm_flip_mean = float(np.mean([p["bm_flip"] for p in per_asin_flip]))
        mlm_flip_mean = float(np.mean([p["mlm_flip"] for p in per_asin_flip]))
        bm_hit10 = float(np.mean([1 if (r is not None and r <= 10) else 0 for r in per_q_bm_rank]))
        mlm_hit10 = float(np.mean([1 if (r is not None and r <= 10) else 0 for r in mlm_ranks]))
        # RR Std aggregation
        bm_rr_std_arr = np.array([p["bm_rr_std"] for p in per_asin_rr_std])
        ml_rr_std_arr = np.array([p["ml_rr_std"] for p in per_asin_rr_std])
        bm_rr_std_mean = float(bm_rr_std_arr.mean()) if len(bm_rr_std_arr) else None
        bm_rr_std_median = float(np.median(bm_rr_std_arr)) if len(bm_rr_std_arr) else None
        bm_rr_std_p95 = float(np.percentile(bm_rr_std_arr, 95)) if len(bm_rr_std_arr) else None
        ml_rr_std_mean = float(ml_rr_std_arr.mean()) if len(ml_rr_std_arr) else None
        ml_rr_std_median = float(np.median(ml_rr_std_arr)) if len(ml_rr_std_arr) else None
        ml_rr_std_p95 = float(np.percentile(ml_rr_std_arr, 95)) if len(ml_rr_std_arr) else None

        log(f"  N={N}: n_asins={n_asins_eff}, n_queries={len(all_queries)}")
        log(f"  BM25:   hit@10={bm_hit10:.4f}, flip_mean={bm_flip_mean:.4f}, RR_Std mean={bm_rr_std_mean:.4f} median={bm_rr_std_median:.4f} p95={bm_rr_std_p95:.4f}")
        log(f"  MiniLM: hit@10={mlm_hit10:.4f}, flip_mean={mlm_flip_mean:.4f}, RR_Std mean={ml_rr_std_mean:.4f} median={ml_rr_std_median:.4f} p95={ml_rr_std_p95:.4f}")

        out = {
            "N": N, "model": "Qwen2.5-14B-Instruct",
            "config": {"K_select": K_SELECT, "SEED": SEED,
                       "sim_threshold": SIM_THRESHOLD,
                       "source": "sub-set" if N in N_LIST else "full-pool"},
            "n_asins": n_asins_eff, "n_queries": len(all_queries),
            "n_dropped_sim": n_dropped_sim if N == 10 else None,
            "bm25_hit10": bm_hit10, "minilm_hit10": mlm_hit10,
            "bm25_flip_mean": bm_flip_mean, "minilm_flip_mean": mlm_flip_mean,
            "bm25_rr_std_mean": bm_rr_std_mean, "bm25_rr_std_median": bm_rr_std_median,
            "bm25_rr_std_p95": bm_rr_std_p95,
            "minilm_rr_std_mean": ml_rr_std_mean, "minilm_rr_std_median": ml_rr_std_median,
            "minilm_rr_std_p95": ml_rr_std_p95,
        }
        # 输出到对应 pool dir
        if N == 10:
            out_dir = N10_POOL_DIR
        else:
            out_dir = SUB_POOL_DIR
        json.dump(out, open(out_dir / f"flip_sim09_N{N}.json", "w"), indent=2)
        log(f"  wrote → {out_dir}/flip_sim09_N{N}.json")
        summary_per_N[N] = out

    # 总体 summary
    log(f"\n=== FINAL SUMMARY (cosine ≥ {SIM_THRESHOLD}) ===")
    for N in sorted(summary_per_N.keys()):
        d = summary_per_N[N]
        log(f"  N={N}: n_asins={d['n_asins']} BM25_flip={d['bm25_flip_mean']:.4f} MiniLM_flip={d['minilm_flip_mean']:.4f}")


if __name__ == "__main__":
    main()