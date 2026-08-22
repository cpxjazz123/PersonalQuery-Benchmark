#!/usr/bin/env python3
"""Phase 35.I Score: N_attrs sweep eval.

Inputs: phase35i_cands_N{N}.json for N in [4..10]
  + phase35e_user_qwen_residuals.pt (74 user exemplars)
  + pca_components.npz

For each N:
  - Compute Qwen residual for each candidate (batched)
  - PCA + softmax_g32_τ0.5 scoring
  - Report cov, Rank-1 size>=10, mean_own_rank, mean_softmax_score

Plus aggregate across N.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35i"
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")

N_ATTRS_LIST = [4, 5, 6, 7, 8, 9, 10]
QWEN_BATCH = int(os.environ.get("P35I_BATCH", "16"))
MAX_INPUT_LENGTH = int(os.environ.get("P35I_MAX_INP", "256"))


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs: return 0
    return sum(1 for v in attrs.values() if v and str(v).strip().lower() in text.lower())


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def main() -> int:
    t0 = time.time()
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    syntax_row = {m: i for i, m in enumerate(flat_meta)}

    # Index user residuals by uid
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])
    print(f"[user_resids] {len(user_resids)} users, total {len(all_resids) - sum(len(user_resids[u]) for u in user_resids)} candidates in base")

    # === For each N, load candidates, compute residuals, score ===
    summary_per_n = {}
    all_cands_residuals = []  # to fit PCA on combined pool later
    all_cands_meta = []
    cand_residuals_per_n = {}

    print("[main] loading Qwen...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time() - t0:.1f}s")

    import torch

    # Build texts per N
    cand_texts_per_n = {}
    for N in N_ATTRS_LIST:
        path = SCRATCH / f"phase35i_cands_N{N}.json"
        if not path.exists():
            print(f"[skip N={N}] not found: {path}")
            continue
        cands = json.load(open(path))
        cand_texts_per_n[N] = cands

    if not cand_texts_per_n:
        print("[ERR] no candidates found, run phase35i_attrsweep.py first")
        return 1

    # Qwen forward for all candidates (combined)
    all_texts = []
    text_index = []  # (N, idx)
    for N, cands in cand_texts_per_n.items():
        for i, c in enumerate(cands):
            all_texts.append(c["query"])
            text_index.append((N, i))
    print(f"[resid] computing residuals for {len(all_texts)} candidates...")

    all_cand_residuals = np.zeros((len(all_texts), all_resids.shape[1]), dtype=np.float32)
    LAYERS = [int(os.environ.get("P35I_LAYER", "26"))]
    for i in range(0, len(all_texts), QWEN_BATCH):
        chunk = all_texts[i:i + QWEN_BATCH]
        hidden_dict = client.get_hidden_states(chunk, layers=LAYERS, max_length=MAX_INPUT_LENGTH, batch_size=QWEN_BATCH)
        all_cand_residuals[i:i + QWEN_BATCH] = hidden_dict[LAYERS[0]]
        done = min(i + QWEN_BATCH, len(all_texts))
        if done % (QWEN_BATCH * 4) == 0 or done == len(all_texts):
            print(f"  [resid] {done}/{len(all_texts)} ({time.time() - t0:.1f}s)")

    # Bucket residuals by N
    cand_residuals_per_n = {N: {} for N in cand_texts_per_n}
    for k, (N, i) in enumerate(text_index):
        cand_residuals_per_n[N][i] = all_cand_residuals[k]

    # Save residuals
    torch.save({
        "residuals": all_cand_residuals,
        "text_index": text_index,
    }, SCRATCH / "phase35i_cand_residuals.pt")
    print(f"[save] cand residuals → {SCRATCH}/phase35i_cand_residuals.pt")

    # === PCA fit on combined ===
    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}
    PCA_DIMS = sorted(pca_components.keys())

    # Reuse existing PCA fitted on Phase 35.E residuals (962 vectors)
    # Project all cand residuals
    cand_z_per_n = {N: {} for N in cand_texts_per_n}
    for N in cand_texts_per_n:
        for i, c in enumerate(cand_texts_per_n[N]):
            r = cand_residuals_per_n[N][i]
            for d in PCA_DIMS:
                cand_z_per_n[N].setdefault(d, {})[i] = (r - pca_means[d]) @ pca_components[d].T

    # User Z projection
    user_z_per_d = {}
    for d in PCA_DIMS:
        user_z_per_d[d] = {}
        for uid, vecs in user_resids.items():
            V = np.stack(vecs, axis=0).astype(np.float32)
            Z = (V - pca_means[d]) @ pca_components[d].T
            user_z_per_d[d][uid] = Z

    # User mu + per-user inverse variance (diag) + global sigma
    user_pca_mu = {}
    user_pca_inv_var_diag = {}
    pca_global_inv_sigma = {}
    for d in PCA_DIMS:
        var_global = np.stack([user_z_per_d[d][u].var(axis=0)
                               for u in user_z_per_d[d]]).mean(axis=0)
        pca_global_inv_sigma[d] = 1.0 / np.maximum(var_global, 1e-6)
        for uid in user_z_per_d[d]:
            Z = user_z_per_d[d][uid]
            user_pca_mu[(uid, d)] = Z.mean(axis=0)
            var = Z.var(axis=0) + 1e-3
            user_pca_inv_var_diag[(uid, d)] = 1.0 / np.maximum(var, 1e-6)

    def maha_diag(cand_z, uid, d):
        mu = user_pca_mu[(uid, d)]
        inv_var = user_pca_inv_var_diag[(uid, d)]
        diffs = cand_z - mu
        return float(np.sqrt(np.maximum((diffs * diffs * inv_var).sum(), 1e-8)))

    def maha_global(cand_z, uid, d):
        mu = user_pca_mu[(uid, d)]
        inv_sigma = pca_global_inv_sigma[d]
        diffs = cand_z - mu
        return float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))

    # === Per-N intra-product Rank-1 ===
    summary_per_n = {}
    for N in N_ATTRS_LIST:
        if N not in cand_texts_per_n:
            continue
        cands = cand_texts_per_n[N]
        # Group by asin
        asin_to_recs: dict[str, list[int]] = {}
        for i, c in enumerate(cands):
            asin_to_recs.setdefault(c["asin"], []).append(i)
        asin_pool: dict[str, list[dict]] = {}
        for asin, idxs in asin_to_recs.items():
            pool = []
            for i in idxs:
                c = cands[i]
                cov = count_attrs_covered(c["query"], c["attrs_used"])
                pool.append({
                    "uid": c["user_id"],
                    "rec_i": i,
                    "cov": cov, "n_attr": N,
                    "z32": cand_z_per_n[N][32][i],
                    "z64": cand_z_per_n[N][64][i],
                })
            asin_pool[asin] = pool

        # For each (asin, uid in pool), compute softmax_g32_τ0.5 rank
        intra_results = []
        cov_rates = []
        for asin, pool in asin_pool.items():
            size = len(pool)
            if size < 2: continue
            uids_in_pool = list({p["uid"] for p in pool})
            if len(uids_in_pool) < 2: continue
            n_users = len(uids_in_pool)
            uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}
            D_g32 = np.zeros((size, n_users))
            for j, v in enumerate(uids_in_pool):
                D_g32[:, j] = [maha_global(p["z32"], v, 32) for p in pool]
            for uid in uids_in_pool:
                target_idx = uid_to_idx[uid]
                tau = 0.5
                logits = -D_g32 / tau
                probs = softmax(logits, axis=1)
                scores = probs[:, target_idx]
                order = np.argsort(-scores)
                rank1_uid = pool[int(order[0])]["uid"]
                hit = rank1_uid == uid
                own_ranks = [int(np.where(order == i)[0][0]) for i, p in enumerate(pool) if p["uid"] == uid]
                intra_results.append({
                    "asin": asin, "uid": uid, "n_pool": size,
                    "is_hit": hit, "mean_own_rank": float(np.mean(own_ranks)) if own_ranks else None,
                })
                # cov rate: per-cand
                cov_rates.extend(p["cov"] / max(p["n_attr"], 1) for p in pool if p["uid"] == uid)

        n_hits = sum(1 for it in intra_results if it["is_hit"])
        n_total = len(intra_results)
        ranks = [it["mean_own_rank"] for it in intra_results if it["mean_own_rank"] is not None]
        full_cov_rate = sum(1 for cr in cov_rates if cr >= 1.0) / max(len(cov_rates), 1)
        avg_cov = np.mean(cov_rates) if cov_rates else 0
        s = {
            "n_users": n_total, "n_hit": n_hits,
            "rank1_pct": n_hits / max(n_total, 1) * 100,
            "mean_own_rank": float(np.mean(ranks)) if ranks else None,
            "median_own_rank": float(np.median(ranks)) if ranks else None,
            "p90_own_rank": float(np.percentile(ranks, 90)) if ranks else None,
            "full_cov_rate": full_cov_rate * 100,
            "avg_cov_rate": avg_cov * 100,
            "n_cands": len(cands),
        }
        summary_per_n[N] = s
        print(f"\n=== N={N} ===")
        print(f"  n_records={len(cands)} (×K={len(cands)//sum(1 for r in json.load(open(SCRATCH/'phase35b_records.json')))} records per record))")
        print(f"  n_users (size>=2)={n_total}, rank-1={n_hits}/{n_total} ({s['rank1_pct']:.1f}%)")
        print(f"  mean_own_rank={s['mean_own_rank']:.2f}, p90={s['p90_own_rank']:.2f}")
        print(f"  cov: full={s['full_cov_rate']:.1f}%, avg={s['avg_cov_rate']:.1f}%")

    json.dump(summary_per_n, open(OUT_DIR / "attrsweep_summary.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[save] summary → {OUT_DIR}/attrsweep_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
