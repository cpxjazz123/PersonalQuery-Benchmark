#!/usr/bin/env python3
"""Phase 14.I: PCA降维 + dist2dist rerank.

Compare 4 dimensions: 30d / 50d / 100d / 3584d (no PCA)
3 dist2dist metrics: Bhattacharyya, W2, symmetric_kl
+ Maha baseline (single-point)
+ Maha with PCA residual

PCA fitted on pooled (user + candidate) hidden states at layer 26.
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
PCA_DIMS = [30, 50, 100, 3584]  # last = no PCA
CONDS = ["D_off", "A14_a1.0"]

EVAL_OUT = OUT_DIR / "phase14_i_pca_dist2dist_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_i_pca_dist2dist_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_i_pca_dist2dist_meta.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_pca(X, n_components):
    """Fit PCA via SVD. Returns projection matrix [D, n_components]."""
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    # SVD on covariance
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # S: singular values, Vt: [n_components, D]
    explained_var = (S ** 2) / (len(X) - 1)
    total_var = explained_var.sum()
    cumvar = np.cumsum(explained_var) / total_var
    V = Vt[:n_components].T  # [D, n_components]
    return X_mean.squeeze(0), V, explained_var[:n_components], float(cumvar[n_components - 1])


def project(X, mean, V):
    return (X - mean) @ V


def bc_per_dim(mu1, var1, mu2, var2):
    """Bhattacharyya per dim."""
    avg_var = (var1 + var2) / 2
    geo_mean = np.sqrt(var1 * var2 + 1e-30)
    bc = 0.25 * (mu1 - mu2) ** 2 / (var1 + var2) + 0.5 * np.log(avg_var / geo_mean + 1e-30)
    return bc.sum()


def w2_diagonal(mu1, var1, mu2, var2):
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    return ((mu1 - mu2) ** 2).sum() + ((sigma1 - sigma2) ** 2).sum()


def symmetric_kl(mu1, var1, mu2, var2):
    kl_pq = 0.5 * (np.log(var2 / var1 + 1e-30) + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
    kl_qp = 0.5 * (np.log(var1 / var2 + 1e-30) + var2 / var1 + (mu1 - mu2) ** 2 / var1 - 1)
    return (kl_pq + kl_qp).sum()


def maha_pooled(mu1, mu2, pooled_var):
    return ((mu1 - mu2) ** 2 / pooled_var).sum()


def main():
    log("[1] Load JSONL rows ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            all_results.append(json.loads(line))
    log(f"  rows: {len(all_results)}")

    log("[2] Load user Gaussians for layer 26 ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in LAYERS_5}
    user_gauss = {}
    user_residuals = []  # for PCA fitting
    user_resid_uid = []
    for ui, uid in enumerate(cached_user_ids):
        user_gauss[uid] = {}
        for li, layer in enumerate(LAYERS_5):
            layer_h = user_hiddens_arr[ui, :, li, :]
            residual = layer_h - neutral_per_layer[layer][None, :]
            user_gauss[uid][layer] = {
                "mu": residual.mean(axis=0),
                "sigma_diag": residual.std(axis=0, ddof=0),
                "all_resid": residual,
            }
            if layer == RERANK_LAYER:
                user_residuals.append(residual)  # [30, 3584]
                user_resid_uid.extend([uid] * 30)
    user_residuals = np.concatenate(user_residuals, axis=0)  # [198*30, 3584]
    log(f"  users: {len(user_gauss)}, residuals for PCA: {user_residuals.shape}")

    log("[3] Encode K=30 candidates at layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    cand_qs = [r["q_final_post"] for r in all_results]
    hiddens = client.get_hidden_states(cand_qs, [RERANK_LAYER])
    cand_hiddens = hiddens[RERANK_LAYER]
    cand_residuals_full = cand_hiddens - neutral_per_layer[RERANK_LAYER][None, :]
    log(f"  cand_residuals shape: {cand_residuals_full.shape}")

    log("[4] Fit PCA on user residuals (signal directions) ...")
    # Use user residuals as "natural language style" reference for PCA
    # This is the same fitting set as Phase 13.A, 14.B
    n_max = max(PCA_DIMS)
    if n_max < user_residuals.shape[1]:
        log(f"  fitting PCA with {n_max} components...")
        pca_mean, pca_V, expl_var, cumvar_30 = fit_pca(user_residuals, n_max)
        log(f"  cumvar at 30 dims: {cumvar_30:.4f}")
        log(f"  top-30 explained var sum: {expl_var[:30].sum() / expl_var.sum():.4f}")
        log(f"  top-50 explained var sum: {expl_var[:50].sum() / expl_var.sum():.4f}")
        log(f"  top-100 explained var sum: {expl_var[:100].sum() / expl_var.sum():.4f}")
    else:
        pca_mean = np.zeros(cand_residuals_full.shape[1], dtype=np.float32)
        pca_V = np.eye(cand_residuals_full.shape[1], dtype=np.float32)
        cumvar_30 = 1.0
        log("  no PCA needed (n_max >= dim)")

    # Project user residuals to PCA space
    user_resid_pca = project(user_residuals, pca_mean, pca_V)  # [198*30, n_max]
    cand_resid_pca = project(cand_residuals_full, pca_mean, pca_V)  # [1800, n_max]
    log(f"  user_resid_pca: {user_resid_pca.shape}, cand_resid_pca: {cand_resid_pca.shape}")

    log("[5] Recompute per-user Gaussian in PCA space ...")
    user_gauss_pca = {}
    for uid in cached_user_ids:
        idx_start = cached_user_ids.index(uid) * 30
        idx_end = idx_start + 30
        u_resid_pca = user_resid_pca[idx_start:idx_end]  # [30, n_max]
        user_gauss_pca[uid] = {
            "mu": u_resid_pca.mean(axis=0),
            "sigma_diag": u_resid_pca.std(axis=0, ddof=0),
        }
    log(f"  user_gauss_pca: {len(user_gauss_pca)}")

    # Compute pooled variance per dim (using PCA-projected user residuals)
    log("[6] Compute pooled var per dim for each PCA dim setting ...")
    pooled_var_per_pca = {}
    for n_dim in PCA_DIMS:
        # Stack all user sigmas for this dim
        if n_dim == 3584:
            # full dim, use original pooled var
            all_var = np.stack([user_gauss[uid][RERANK_LAYER]["sigma_diag"] ** 2 for uid in cached_user_ids])
        else:
            all_var = np.stack([user_gauss_pca[uid]["sigma_diag"][:n_dim] ** 2 for uid in cached_user_ids])
        all_var = all_var + LW_SHRINK
        all_var = np.maximum(all_var, MIN_VAR)
        pooled_var = all_var.mean(axis=0)  # [n_dim]
        pooled_var_per_pca[n_dim] = pooled_var

    log("[7] Group rows by (cond, pair) ...")
    by_cond_pair = defaultdict(lambda: defaultdict(list))
    for r_idx, r in enumerate(all_results):
        by_cond_pair[r["condition"]][(r["user_id"], r["asin"])].append(r_idx)
    log(f"  conds: {list(by_cond_pair.keys())}")

    log("[8] Rerank per (cond, n_dim, metric) ...")
    eval_summary = {c: {} for c in CONDS}
    per_pair_results = []
    for n_dim in PCA_DIMS:
        for cond in CONDS:
            pairs_dict = by_cond_pair[cond]
            metrics_results = {
                "maha_pooled": [],
                "bhattacharyya": [],
                "w2": [],
                "symmetric_kl": [],
            }
            rank1_counts = {m: 0 for m in metrics_results}
            for (uid, asin), cand_indices in pairs_dict.items():
                if uid not in user_gauss_pca and n_dim < 3584:
                    continue
                if uid not in user_gauss and n_dim == 3584:
                    continue
                # Find target user index
                target_uid_idx = cached_user_ids.index(uid)
                # Get cand distribution
                if n_dim == 3584:
                    cand_resid_K = cand_residuals_full[cand_indices]  # [K, 3584]
                    user_mu_arr = np.stack([user_gauss[u][RERANK_LAYER]["mu"] for u in cached_user_ids])
                    user_var_arr = np.stack([user_gauss[u][RERANK_LAYER]["sigma_diag"] ** 2 for u in cached_user_ids])
                else:
                    cand_resid_K = cand_resid_pca[cand_indices][:, :n_dim]  # [K, n_dim]
                    user_mu_arr = np.stack([user_gauss_pca[u]["mu"][:n_dim] for u in cached_user_ids])
                    user_var_arr = np.stack([user_gauss_pca[u]["sigma_diag"][:n_dim] ** 2 for u in cached_user_ids])
                cand_mu = cand_resid_K.mean(axis=0)  # [n_dim]
                cand_var = cand_resid_K.var(axis=0, ddof=0) + LW_SHRINK  # [n_dim]
                cand_var = np.maximum(cand_var, MIN_VAR)
                user_var_arr = user_var_arr + LW_SHRINK
                user_var_arr = np.maximum(user_var_arr, MIN_VAR)
                pooled_var = pooled_var_per_pca[n_dim]
                # Distance per user [n_users]
                mu_diff = cand_mu[None, :] - user_mu_arr  # [n_users, n_dim]
                # Maha
                maha_per_user = (mu_diff ** 2 / pooled_var[None, :]).sum(axis=-1)
                # BC
                avg_var = (cand_var[None, :] + user_var_arr) / 2
                geo_mean = np.sqrt(cand_var[None, :] * user_var_arr + 1e-30)
                bc_per_dim = 0.25 * mu_diff ** 2 / (cand_var[None, :] + user_var_arr) + 0.5 * np.log(avg_var / geo_mean)
                bc_per_user = bc_per_dim.sum(axis=-1)
                # W2
                sigma_diff = np.sqrt(cand_var[None, :]) - np.sqrt(user_var_arr)
                w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
                # KL
                kl_pq = 0.5 * (np.log(user_var_arr / cand_var[None, :] + 1e-30) +
                               cand_var[None, :] / user_var_arr +
                               mu_diff ** 2 / user_var_arr - 1)
                kl_qp = 0.5 * (np.log(cand_var[None, :] / user_var_arr + 1e-30) +
                               user_var_arr / cand_var[None, :] +
                               mu_diff ** 2 / cand_var[None, :] - 1)
                kl_per_user = (kl_pq + kl_qp).sum(axis=-1)
                # Rank
                target_idx = target_uid_idx
                ranks = {
                    "maha_pooled": int((maha_per_user < maha_per_user[target_idx]).sum()),
                    "bhattacharyya": int((bc_per_user < bc_per_user[target_idx]).sum()),
                    "w2": int((w2_per_user < w2_per_user[target_idx]).sum()),
                    "symmetric_kl": int((kl_per_user < kl_per_user[target_idx]).sum()),
                }
                for m, r in ranks.items():
                    metrics_results[m].append(r)
                    if r == 0:
                        rank1_counts[m] += 1
                per_pair_results.append({
                    "condition": cond,
                    "n_dim": n_dim,
                    "user_id": uid,
                    "asin": asin,
                    **ranks,
                })
            # Aggregate
            for metric in metrics_results:
                ranks = np.array(metrics_results[metric])
                eval_summary[cond].setdefault(f"{metric}_d{n_dim}", {
                    "n_pairs": len(ranks),
                    "rank1": int(rank1_counts[metric]),
                    "rank1_pct": float(rank1_counts[metric] / len(ranks)) if len(ranks) > 0 else 0.0,
                    "top10": int((ranks < 10).sum()),
                    "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                    "top100": int((ranks < 100).sum()),
                    "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                    "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
                })

    log(f"  eval summary:")
    print(json.dumps(eval_summary, indent=2))
    EVAL_OUT.write_text(json.dumps(eval_summary, indent=2))

    with PER_PAIR_OUT.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.I",
        "pca_dims": PCA_DIMS,
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "conds": CONDS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "pca_cumvar_30d": cumvar_30,
        "metrics": ["maha_pooled", "bhattacharyya", "w2", "symmetric_kl"],
        "n_total": len(all_results),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.I PCA + DIST2DIST RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()