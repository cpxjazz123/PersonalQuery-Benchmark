#!/usr/bin/env python3
"""Pick the most user-style-matching candidate from K generated queries.

For each record (user_id, asin) with K generated query candidates, compute the
Mahalanobis distance from each candidate's Qwen residual (PCA32) to the target
user's residual distribution, then pick the candidate with the smallest distance.

Pipeline:
  cand.query → Qwen2-7B layer 26 mean-pool → 3584d residual
             → PCA32 projection
             → Mahalanobis distance D = sqrt((z - mu_u)^T Σ^-1 (z - mu_u))
             → score = -D / τ
             → argmax over K candidates

Where mu_u is the user's mean residual in PCA32 space and Σ is the global
diagonal covariance (per-dim variance averaged across all users).

Inputs (all hardcoded):
  - /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35i_v2_cands_N7.json
    (or any cands JSON with [{user_id, asin, attrs_used, query, k}, ...])
  - /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35e_user_qwen_residuals.pt
  - /fs04/ar57/wenyu/PersoanlQuery/result/phase35e/pca_components.npz

Output:
  - /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/best_cands.json
    (note: scratch2 dir still uses old name 'syntax_subspace'; this script lives in
    the renamed select_query/ project dir)
    [{record_id, user_id, asin, best_k, best_query, best_score, all_scores}, ...]

Reference baseline (Phase 35.I-v2 SOTA): intra-product Rank-1 = 50.7%
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
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")

CANDS_PATH = SCRATCH / "phase35i_v2_cands_N7.json"  # K cands per (uid, asin)
USER_RESIDUALS_PATH = SCRATCH / "phase35e_user_qwen_residuals.pt"
PCA_PATH = PHASE35E_DIR / "pca_components.npz"
OUTPUT_PATH = SCRATCH / "best_cands.json"

# === Hyperparameters (Phase 35.G SOTA config) ===
LAYERS = [26]                    # Qwen2-7B layer for mean-pool residual
PCA_D = 32                       # PCA dim (Phase 35.C SOTA)
TAU = 0.5                        # softmax temperature
QWEN_BATCH = 16                  # batch size for Qwen forward
MAX_INPUT_LENGTH = 256             # Qwen max input tokens


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def main() -> int:
    t0 = time.time()
    # === Load user Qwen residuals (cached) ===
    cache = torch.load(USER_RESIDUALS_PATH, weights_only=False)
    all_resids = cache["residuals"]   # [N, 3584]
    flat_meta = cache["meta"]          # list of [kind, id, ...]
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])
    print(f"[load] {len(user_resids)} users, total {len(all_resids)} residuals")

    # === Load PCA components ===
    pca_data = np.load(PCA_PATH, allow_pickle=True)
    pca_components = {int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {int(k): v for k, v in pca_data["means"].item().items()}
    assert PCA_D in pca_components, f"PCA dim {PCA_D} not in {list(pca_components.keys())}"

    # === Project users to PCA32 ===
    user_z = {}
    for uid, vecs in user_resids.items():
        V = np.stack(vecs, axis=0).astype(np.float32)
        Z = (V - pca_means[PCA_D]) @ pca_components[PCA_D].T
        user_z[uid] = Z  # [n_user_sents, 32]

    user_mu = {uid: Z.mean(axis=0) for uid, Z in user_z.items()}
    # Global diagonal covariance (shared)
    var_global = np.stack([Z.var(axis=0) for Z in user_z.values()]).mean(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)
    print(f"[sigma] mean var = {var_global.mean():.4f}, median = {np.median(var_global):.4f}")

    # === Load K candidates ===
    cands = json.load(open(CANDS_PATH))
    print(f"[load] {len(cands)} candidates from {CANDS_PATH.name}")

    # Group by (uid, asin) → list of cand indices
    by_record = {}
    for i, c in enumerate(cands):
        key = (c["user_id"], c["asin"])
        by_record.setdefault(key, []).append(i)

    n_records = len(by_record)
    avg_K = sum(len(v) for v in by_record.values()) / max(n_records, 1)
    print(f"[records] {n_records} unique (uid, asin), avg K = {avg_K:.1f}")

    # === Encode all cand queries via Qwen (deduplicated) ===
    unique_queries = {}
    for c in cands:
        unique_queries.setdefault(c["query"], len(unique_queries))
    print(f"[encode] {len(unique_queries)} unique queries to encode via Qwen")

    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    # Map query → residual
    query_residual = {}
    queries_list = list(unique_queries.keys())
    for i in range(0, len(queries_list), QWEN_BATCH):
        chunk = queries_list[i:i + QWEN_BATCH]
        hidden_dict = client.get_hidden_states(
            chunk, layers=LAYERS, max_length=MAX_INPUT_LENGTH, batch_size=QWEN_BATCH
        )
        for q, h in zip(chunk, hidden_dict[LAYERS[0]].cpu().numpy()):
            query_residual[q] = h
        done = min(i + QWEN_BATCH, len(queries_list))
        print(f"  [encode] {done}/{len(queries_list)}  ({time.time()-t0:.1f}s)")

    # Project cand residuals to PCA32 (cache for repeated queries)
    query_z = {}
    for q, h in query_residual.items():
        query_z[q] = (h.astype(np.float32) - pca_means[PCA_D]) @ pca_components[PCA_D].T

    # === Score each record ===
    def maha_score(z32: np.ndarray, uid: str) -> float:
        """Higher score = closer to user. Returns -D/τ."""
        mu = user_mu[uid]
        diffs = z32 - mu
        d = float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))
        return -d / TAU

    results = []
    for (uid, asin), idxs in by_record.items():
        if uid not in user_mu:
            # Fallback: pick first cand
            best_i = idxs[0]
            best_score = -1e9
        else:
            scores = np.array([maha_score(query_z[cands[i]["query"]], uid) for i in idxs])
            best_i = idxs[int(np.argmax(scores))]
            best_score = float(scores.max())
        results.append({
            "user_id": uid,
            "asin": asin,
            "best_k": cands[best_i].get("k", -1),
            "best_query": cands[best_i]["query"],
            "best_score": best_score,
            "all_scores": [
                float(maha_score(query_z[cands[i]["query"]], uid)) if uid in user_mu else -1e9
                for i in idxs
            ],
            "n_cands": len(idxs),
        })

    json.dump(results, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUTPUT_PATH}  ({len(results)} records)")
    print(f"[time]  {time.time()-t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())