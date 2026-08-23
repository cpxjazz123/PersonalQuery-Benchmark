#!/usr/bin/env python3
"""Pick the most user-style-matching candidate for 10K user pipeline.

Adapted from pick_best_cand.py for the 10K user pipeline:
- Reads K candidates per record from generate_strict_nohallu.py output
- Computes PCA on-the-fly from the 10K user Qwen residuals
- Scores each candidate by Mahalanobis distance to user's mean residual in PCA32
- Selects the candidate with the highest -D/τ (closest to user style)

Pipeline:
  cand.query → Qwen2-7B layer 26 mean-pool → 3584d residual → PCA32
             → Mahalanobis distance D = sqrt((z - μ_u)^T Σ^-1 (z - μ_u))
             → score = -D / τ
             → argmax over K candidates per record

Inputs (all hardcoded):
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_records_with_query_inject_strict_10k.json
    (output of gen_query/generate_strict_nohallu.py; uses all_candidates field)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/residual_hidden_10k.npz
    (output of gaussian/extract_residual_hidden.py; user residuals from Stage 3)

Outputs:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/best_cands_10k.json
    [{user_id, asin, best_k, best_query, best_score, all_scores, n_cands}, ...]
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pca_components_10k.npz
    (cached PCA components for downstream eval reuse)
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# === Hardcoded paths (CLAUDE.md rule: no argparse) ===
REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

RECORDS_IN = REPO_ROOT / "result/query_records_with_query_inject_strict_10k.json"
RESIDUAL_NPZ = SCRATCH / "residual_hidden_10k.npz"
PCA_OUT = SCRATCH / "pca_components_10k.npz"
OUTPUT_PATH = SCRATCH / "best_cands_10k.json"

# === Hyperparameters (Phase 35.G/I-v2 SOTA config) ===
LAYERS = [int(x) for x in os.environ.get("PB_LAYERS", "26").split(",")]
PCA_D = int(os.environ.get("PB_PCA_D", "32"))
TAU = float(os.environ.get("PB_TAU", "0.5"))
QWEN_BATCH = int(os.environ.get("PB_QWEN_BATCH", "16"))
MAX_INPUT_LENGTH = int(os.environ.get("PB_MAX_INPUT_LENGTH", "256"))


def main() -> int:
    t0 = time.time()
    # === Load user Qwen residuals (cached) ===
    if not RESIDUAL_NPZ.exists():
        raise FileNotFoundError(
            f"{RESIDUAL_NPZ} 不存在, 先跑 gaussian/extract_residual_hidden.py 生成 10K residuals"
        )
    cache = np.load(RESIDUAL_NPZ, allow_pickle=True)
    LAYER = LAYERS[0]
    residual_key = f"residual_layer_{LAYER}"
    if residual_key not in cache.files:
        raise KeyError(f"{residual_key} not in npz (available: {cache.files})")
    all_resids = cache[residual_key]   # [N, 3584]
    sentences = cache["sentences"].tolist() if "sentences" in cache.files else None
    print(f"[load] residuals shape={all_resids.shape}, dtype={all_resids.dtype}")

    # Group residuals by user (uid extracted from sentence if sentence_text is "{uid}||{text}")
    # Otherwise use sentences_for_rewrite_10k.jsonl mapping
    sent_to_uid: dict[str, str] = {}
    if sentences is not None:
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

    user_resids: dict[str, list[np.ndarray]] = {}
    if sentences is not None and sent_to_uid:
        for i, sent in enumerate(sentences):
            uid = sent_to_uid.get(sent)
            if uid:
                user_resids.setdefault(uid, []).append(all_resids[i])
    else:
        # Fallback: treat all residuals as one user (legacy behavior)
        print("[load] WARNING: no sentence→uid mapping; using all residuals as user")
        user_resids["__legacy__"] = [all_resids[i] for i in range(len(all_resids))]

    print(f"[load] {len(user_resids)} users, total {len(all_resids)} residuals")
    if len(user_resids) < 100:
        print(f"[load] WARNING: only {len(user_resids)} users in residuals — rerun Stage 3 with 10K data!")

    # === Compute / load PCA ===
    if PCA_OUT.exists():
        print(f"[PCA] loading cached: {PCA_OUT}")
        pca_data = np.load(PCA_OUT, allow_pickle=True)
        components = {int(k): v for k, v in pca_data["components"].item().items()}
        means = {int(k): v for k, v in pca_data["means"].item().items()}
    else:
        print(f"[PCA] computing PCA{PCA_D} from {all_resids.shape[0]} residuals...")
        from sklearn.decomposition import PCA
        X = all_resids.astype(np.float32)
        pca = PCA(n_components=PCA_D, random_state=42)
        pca.fit(X)
        components = {PCA_D: pca.components_.astype(np.float32)}
        means = {PCA_D: pca.mean_.astype(np.float32)}
        np.savez_compressed(PCA_OUT, components=np.array([{PCA_D: components[PCA_D]}], dtype=object),
                            means=np.array([{PCA_D: means[PCA_D]}], dtype=object))
        print(f"[PCA] explained_var_ratio sum = {pca.explained_variance_ratio_.sum():.3f}, saved → {PCA_OUT}")

    assert PCA_D in components, f"PCA dim {PCA_D} not in {list(components.keys())}"

    # === Project users to PCA32 ===
    user_z = {}
    for uid, vecs in user_resids.items():
        V = np.stack(vecs, axis=0).astype(np.float32)
        Z = (V - means[PCA_D]) @ components[PCA_D].T
        user_z[uid] = Z

    user_mu = {uid: Z.mean(axis=0) for uid, Z in user_z.items()}
    # Global diagonal covariance (shared)
    var_global = np.stack([Z.var(axis=0) for Z in user_z.values()]).mean(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)
    print(f"[sigma] mean var = {var_global.mean():.4f}, median = {np.median(var_global):.4f}")

    # === Load candidates from strict_nohallu output ===
    if not RECORDS_IN.exists():
        raise FileNotFoundError(f"{RECORDS_IN} 不存在, 先跑 gen_query/generate_strict_nohallu.py")
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    print(f"[load] {len(records)} records from {RECORDS_IN.name}")

    # Flatten: each record has all_candidates: [str, ...] (K candidates)
    cands_flat = []
    for rec_idx, r in enumerate(records):
        ac = r.get("all_candidates") or [r.get("y_plus_query_inject")] if r.get("y_plus_query_inject") else []
        if not ac:
            continue
        for k_idx, q in enumerate(ac):
            cands_flat.append({
                "rec_idx": rec_idx,
                "user_id": r["user_id"],
                "asin": r["asin"],
                "attrs_used": r.get("attrs_used", {}),
                "query": q,
                "k": k_idx,
            })
    print(f"[cands] {len(cands_flat)} candidates total ({len(records)} records)")

    # === Encode all unique cand queries via Qwen ===
    unique_queries = {}
    for c in cands_flat:
        unique_queries.setdefault(c["query"], len(unique_queries))
    print(f"[encode] {len(unique_queries)} unique queries to encode via Qwen")

    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    query_residual = {}
    queries_list = list(unique_queries.keys())
    for i in range(0, len(queries_list), QWEN_BATCH):
        chunk = queries_list[i:i + QWEN_BATCH]
        hidden_dict = client.get_hidden_states(
            chunk, layers=LAYERS, max_length=MAX_INPUT_LENGTH, batch_size=QWEN_BATCH
        )
        h_layer = hidden_dict[LAYERS[0]]
        if hasattr(h_layer, "cpu"):
            h_layer = h_layer.cpu().numpy()
        for q, h in zip(chunk, h_layer):
            query_residual[q] = h
        done = min(i + QWEN_BATCH, len(queries_list))
        print(f"  [encode] {done}/{len(queries_list)}  ({time.time()-t0:.1f}s)", flush=True)

    # Project cand residuals to PCA32
    query_z = {}
    for q, h in query_residual.items():
        query_z[q] = (h.astype(np.float32) - means[PCA_D]) @ components[PCA_D].T

    # === Score each record (argmax over K) ===
    def maha_score(z32: np.ndarray, uid: str) -> float:
        mu = user_mu.get(uid)
        if mu is None:
            return -1e9
        diffs = z32 - mu
        d = float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))
        return -d / TAU

    # Group by record
    by_record: dict[int, list[int]] = {}
    for i, c in enumerate(cands_flat):
        by_record.setdefault(c["rec_idx"], []).append(i)

    results = []
    for rec_idx, idxs in by_record.items():
        scores = np.array([maha_score(query_z[cands_flat[i]["query"]], cands_flat[i]["user_id"])
                           for i in idxs])
        best_i = idxs[int(np.argmax(scores))]
        results.append({
            "user_id": cands_flat[best_i]["user_id"],
            "asin": cands_flat[best_i]["asin"],
            "best_k": cands_flat[best_i]["k"],
            "best_query": cands_flat[best_i]["query"],
            "best_score": float(scores.max()),
            "all_scores": [float(s) for s in scores],
            "n_cands": len(idxs),
        })
        # Periodic save (every 1000 records)
        if len(results) % 1000 == 0:
            json.dump(results, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
            print(f"  [save] checkpoint {len(results)} records")

    json.dump(results, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUTPUT_PATH}  ({len(results)} records)")
    print(f"[time]  {time.time()-t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())