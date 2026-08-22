#!/usr/bin/env python3
"""Phase 14.H: distribution-to-distribution rerank on K=30 candidates.

Three rerank metrics:
1. Bhattacharyya distance (per-dim Gaussian BC, sum over dim)
2. 2-Wasserstein (diagonal Gaussian closed-form: (μ1-μ2)² + (σ1-σ2)²)
3. Symmetric KL divergence

Plus baseline: pooled Mahalanobis (Phase 14.F SOTA) — using point estimate
(mean of K candidates) vs per-user Gaussian.

For all methods, best-of-K = min distance across K candidates.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

JSONL_IN = OUT_DIR / "phase14_h_k30_generations.jsonl"
HIDDEN_NPZ = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

N_PAIRS = 30
K_PER_PAIR = 30
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
LW_SHRINK = 0.1
MIN_VAR = 1e-4

EVAL_OUT = OUT_DIR / "phase14_h_dist2dist_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_h_dist2dist_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_h_dist2dist_meta.json"
CONDS = ["D_off", "A14_a1.0"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bhattacharyya_per_dim(mu1, var1, mu2, var2):
    """Per-dim Bhattacharyya distance for diagonal Gaussians.

    BC_d = 1/4 * (μ1-μ2)² / (var1+var2) + 1/2 * log((var1+var2)/(2*sqrt(var1*var2)))
    """
    avg_var = (var1 + var2) / 2
    geo_mean = np.sqrt(var1 * var2)
    bc = 0.25 * (mu1 - mu2) ** 2 / (var1 + var2) + 0.5 * np.log(avg_var / geo_mean + 1e-30)
    return bc.sum()


def w2_diagonal(mu1, var1, mu2, var2):
    """2-Wasserstein distance for diagonal Gaussians (closed-form).

    W2² = ||μ1-μ2||² + ||σ1-σ2||²  (where σ = sqrt(var))
    """
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    return ((mu1 - mu2) ** 2).sum() + ((sigma1 - sigma2) ** 2).sum()


def symmetric_kl(mu1, var1, mu2, var2):
    """Symmetric KL for diagonal Gaussians.

    KL(p||q) = 1/2 * [log(var_q/var_p) + var_p/var_q + (μ_p-μ_q)²/var_q - 1]
    KL_sym = KL(p||q) + KL(q||p)
    """
    kl_pq = 0.5 * (np.log(var2 / var1 + 1e-30) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
    kl_qp = 0.5 * (np.log(var1 / var2 + 1e-30) + var2 / var1 + (mu1 - mu2) ** 2 / var1 - 1)
    return (kl_pq + kl_qp).sum()


def main():
    log("[1] Load JSONL rows ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            all_results.append(json.loads(line))
    log(f"  rows: {len(all_results)} (expect {N_PAIRS * 2 * K_PER_PAIR} = {N_PAIRS * 2 * K_PER_PAIR})")

    log("[2] Load user Gaussians for layer 26 ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in LAYERS_5}
    user_gauss = {}
    for ui, uid in enumerate(cached_user_ids):
        user_gauss[uid] = {}
        for li, layer in enumerate(LAYERS_5):
            layer_h = user_hiddens_arr[ui, :, li, :]
            residual = layer_h - neutral_per_layer[layer][None, :]
            user_gauss[uid][layer] = {"mu": residual.mean(axis=0), "sigma_diag": residual.std(axis=0, ddof=0)}
    log(f"  users: {len(user_gauss)}")

    log("[3] Encode K=30 candidates at layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    cand_qs = [r["q_final_post"] for r in all_results]
    hiddens = client.get_hidden_states(cand_qs, [RERANK_LAYER])
    cand_hiddens = hiddens[RERANK_LAYER]
    log(f"  cand_hiddens shape: {cand_hiddens.shape}")

    log("[4] Compute residuals ...")
    neutral_layer = neutral_per_layer[RERANK_LAYER]
    cand_residuals = cand_hiddens - neutral_layer[None, :]
    log(f"  residual norm: {np.linalg.norm(cand_residuals, axis=-1).mean():.2f}")

    log("[5] Build per-user mu/var for layer 26 ...")
    user_uid_list = list(user_gauss.keys())
    user_mu_26 = np.stack([user_gauss[uid][RERANK_LAYER]["mu"] for uid in user_uid_list])
    user_var_26 = np.stack([user_gauss[uid][RERANK_LAYER]["sigma_diag"] ** 2 for uid in user_uid_list])
    user_var_26 = user_var_26 + LW_SHRINK
    user_var_26 = np.maximum(user_var_26, MIN_VAR)
    pooled_var_26 = user_var_26.mean(axis=0)
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}
    log(f"  user_mu_26 shape: {user_mu_26.shape}")

    log("[6] Group rows by (cond, pair) ...")
    # Each pair has K=30 rows per cond, indexed by cand_local_idx
    by_cond_pair = defaultdict(lambda: defaultdict(list))
    for r_idx, r in enumerate(all_results):
        by_cond_pair[r["condition"]][(r["user_id"], r["asin"])].append((r_idx, r["cand_local_idx"]))
    log(f"  conds: {list(by_cond_pair.keys())}")

    log("[7] Rerank per cond ...")
    eval_summary = {c: {} for c in CONDS}
    per_pair_results = []
    for cond in CONDS:
        pairs_dict = by_cond_pair[cond]
        # For each pair: compute 4 metrics
        bc_ranks = []
        w2_ranks = []
        kl_ranks = []
        maha_ranks = []
        rank1_bc = rank1_w2 = rank1_kl = rank1_maha = 0
        for (uid, asin), cand_list in pairs_dict.items():
            if uid not in uid_to_idx:
                continue
            target_idx = uid_to_idx[uid]
            # Sort by cand_local_idx to ensure order
            cand_list = sorted(cand_list, key=lambda x: x[1])
            cand_indices = [c[0] for c in cand_list]
            K = len(cand_indices)
            # Compute candidate distribution from K hiddens
            cand_resid_K = cand_residuals[cand_indices]  # [K, H]
            cand_mu = cand_resid_K.mean(axis=0)  # [H]
            cand_var = cand_resid_K.var(axis=0, ddof=0) + LW_SHRINK  # [H]
            cand_var = np.maximum(cand_var, MIN_VAR)
            # Per-user distance (vectorized)
            # For each user u, compute 4 distances using cand distribution
            # BC / W2 / KL / Maha against per-user Gaussian (mu_u, var_u)
            # Maha: (cand_mu - mu_u)² / pooled_var
            # Maha uses mean of K (best-of-K by candidate mean? or per-candidate?)
            # For dist2dist, we compare distributions.
            # For Maha, we use point (mean of K)
            # Shape: [n_users]
            diff = cand_mu[None, :] - user_mu_26  # [n_users, H]
            maha_per_user = (diff ** 2 / pooled_var_26[None, :]).sum(axis=-1)  # [n_users]
            # BC, W2, KL per user
            # Need pairwise computation: cand vs user
            # user_mu_26 [n_users, H], cand_mu [H]
            # user_var_26 [n_users, H], cand_var [H]
            mu_diff = cand_mu[None, :] - user_mu_26  # [n_users, H]
            # BC: 1/4 * (μ1-μ2)² / (var1+var2) + 1/2 * log((var1+var2)/(2*sqrt(var1*var2)))
            avg_var = (cand_var[None, :] + user_var_26) / 2  # [n_users, H]
            geo_mean = np.sqrt(cand_var[None, :] * user_var_26 + 1e-30)  # [n_users, H]
            bc_per_dim = 0.25 * mu_diff ** 2 / (cand_var[None, :] + user_var_26) + 0.5 * np.log(avg_var / geo_mean)
            bc_per_user = bc_per_dim.sum(axis=-1)  # [n_users]
            # W2: (μ1-μ2)² + (σ1-σ2)²
            sigma_diff = np.sqrt(cand_var[None, :]) - np.sqrt(user_var_26)
            w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
            # Symmetric KL
            kl_pq = 0.5 * (np.log(user_var_26 / cand_var[None, :] + 1e-30) +
                           cand_var[None, :] / user_var_26 +
                           mu_diff ** 2 / user_var_26 - 1)
            kl_qp = 0.5 * (np.log(cand_var[None, :] / user_var_26 + 1e-30) +
                           user_var_26 / cand_var[None, :] +
                           mu_diff ** 2 / cand_var[None, :] - 1)
            kl_per_user = (kl_pq + kl_qp).sum(axis=-1)
            # Rank (best-of-K = the rank of target in sorted ascending)
            bc_rank = (bc_per_user < bc_per_user[target_idx]).sum()
            w2_rank = (w2_per_user < w2_per_user[target_idx]).sum()
            kl_rank = (kl_per_user < kl_per_user[target_idx]).sum()
            maha_rank = (maha_per_user < maha_per_user[target_idx]).sum()
            bc_ranks.append(bc_rank)
            w2_ranks.append(w2_rank)
            kl_ranks.append(kl_rank)
            maha_ranks.append(maha_rank)
            if bc_rank == 0:
                rank1_bc += 1
            if w2_rank == 0:
                rank1_w2 += 1
            if kl_rank == 0:
                rank1_kl += 1
            if maha_rank == 0:
                rank1_maha += 1
            per_pair_results.append({
                "condition": cond,
                "user_id": uid,
                "asin": asin,
                "K": K,
                "bc_rank": int(bc_rank),
                "w2_rank": int(w2_rank),
                "kl_rank": int(kl_rank),
                "maha_rank": int(maha_rank),
            })
        bc_ranks = np.array(bc_ranks)
        w2_ranks = np.array(w2_ranks)
        kl_ranks = np.array(kl_ranks)
        maha_ranks = np.array(maha_ranks)
        for metric, ranks, rank1 in [
            ("bhattacharyya", bc_ranks, rank1_bc),
            ("w2", w2_ranks, rank1_w2),
            ("symmetric_kl", kl_ranks, rank1_kl),
            ("maha_pooled", maha_ranks, rank1_maha),
        ]:
            eval_summary[cond][metric] = {
                "n_pairs": len(ranks),
                "rank1": int(rank1),
                "rank1_pct": float(rank1 / len(ranks)) if len(ranks) > 0 else 0.0,
                "top10": int((ranks < 10).sum()),
                "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "top100": int((ranks < 100).sum()),
                "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
                "min_rank": int(ranks.min()) if len(ranks) > 0 else 0,
                "max_rank": int(ranks.max()) if len(ranks) > 0 else 0,
            }

    log(f"  eval summary:")
    print(json.dumps(eval_summary, indent=2))
    EVAL_OUT.write_text(json.dumps(eval_summary, indent=2))

    with PER_PAIR_OUT.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.H",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "conds": CONDS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "metrics": ["bhattacharyya", "w2", "symmetric_kl", "maha_pooled"],
        "n_total": len(all_results),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.H DIST2DIST RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()