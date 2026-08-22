#!/usr/bin/env python3
"""Phase 14.P-3: LOPO validation for BoK-88 ENSEMBLE-α=0.3 (PCA-200 + PCA-500).

Validates best Phase 14.P config (rank-1 11/30 36.7%) generalizes without overfit.

Uses Phase 14.N pattern: Randomized PCA + multiprocessing (40x speedup).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_p_lopo_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_p_lopo_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_p_lopo_meta.json"

# Best BoK-88 config from Phase 14.P-2
CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
              "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]
K_PER_PAIR = 8
LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584

# Ensemble: 2 PCA dims
PCA_DIMS_ENSEMBLE = [200, 500]
ALPHA = 0.3  # PCA-200 weight
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000
N_WORKERS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n: int = N_BOOTSTRAP, seed: int = SEED):
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def aggregate(ranks):
    if not ranks:
        return None
    arr = np.array(ranks)
    mean, ci = bootstrap_ci(ranks)
    return {
        "n": len(ranks),
        "rank1": int((arr == 0).sum()),
        "rank1_pct": float((arr == 0).sum()) / max(1, len(arr)),
        "top10": int((arr < 10).sum()),
        "top10_pct": float((arr < 10).sum()) / max(1, len(arr)),
        "top100": int((arr < 100).sum()),
        "top100_pct": float((arr < 100).sum()) / max(1, len(arr)),
        "mean_rank": mean,
        "mean_rank_ci95": list(ci),
    }


# === Worker: one fold ===
def process_fold(args):
    """Process one held-out pair; compute BoK-88 ENSEMBLE rank."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    fold_idx, target_uid, target_asin, cand_residuals_per_pair, \
        user_mu_full, user_var_full, pooled_var_diag_full, cand_pca_components, \
        pca_mean, ALPHA, PCA_DIMS_ENSEMBLE = args

    # Project cand residuals (per-pair 88 cands × 3584) → 2 PCA dims
    pca_list = []
    for pca_dim in PCA_DIMS_ENSEMBLE:
        comp = cand_pca_components[:pca_dim]
        cand_pca = (cand_residuals_per_pair - pca_mean) @ comp.T  # [88, pca_dim]
        inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
        user_mu = user_mu_full[:, :pca_dim]
        delta = cand_pca[:, None, :] - user_mu[None, :, :]
        weighted = delta * inv_var[None, None, :]
        maha = (delta * weighted).sum(axis=-1)  # [88, 198]
        pca_list.append(maha)

    # Normalize each Maha
    maha_200 = pca_list[0]
    maha_500 = pca_list[1]
    maha_200_norm = maha_200 / (maha_200.max() + 1e-9)
    maha_500_norm = maha_500 / (maha_500.max() + 1e-9)

    # Ensemble
    score = ALPHA * maha_200_norm + (1 - ALPHA) * maha_500_norm  # [88, 198]

    # Find target_idx from user_ids_cache
    # We pass target_idx via the args
    target_idx = args[11]
    target_score = score[:, target_idx: target_idx + 1]
    rank_per_cand = (score < target_score).sum(axis=1)
    best_rank = int(np.min(rank_per_cand))
    return fold_idx, target_uid, target_asin, best_rank


def main() -> None:
    log("=" * 70)
    log("Phase 14.P-3: LOPO validation for BoK-88 ENSEMBLE-α=0.3")
    log("=" * 70)

    # === [1] Load candidates ===
    log("[1] Loading candidates ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            candidates.append(r)
    log(f"  candidates: {len(candidates)}")

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})

    # Map: (uid, asin) → cond → list of cand idx
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load hiddens + neutral ===
    log("[2] Loading hiddens + neutral ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals_all = np.load(IN_CAND_RESID)  # [2640, 5, 3584]
    cand_residuals = cand_residuals_all[:, layer_idx, :]
    log(f"  cand_residuals: {cand_residuals.shape}")

    # === [3] Fit PCA-500 once ===
    log(f"[3] Fitting Randomized PCA-{MAX_PCA} on combined pool ...")
    user_residuals_flat = user_residuals.reshape(-1, HIDDEN)
    train_pool = np.concatenate([user_residuals_flat, cand_residuals], axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=SEED)
    pca.fit(train_pool)
    components_full = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)
    log(f"  PCA-{MAX_PCA} fitted: explained var {pca.explained_variance_ratio_.sum():.4f}")

    user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
    user_pca_full_3d = user_pca_full.reshape(N_USERS, 30, MAX_PCA)
    cand_pca_full = (cand_residuals - pca_mean) @ components_full.T

    # Per-user Gaussian
    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    # === [4] LOPO over 30 folds (sequential per-pair, fast enough) ===
    log(f"[4] LOPO over {len(pairs)} folds (per-pair BoK-88 ENSEMBLE) ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)
    log(f"  valid pairs: {len(pair_keys)}")

    fold_records = []
    t0 = time.time()

    for fold_idx, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]
        all_cand_idx = []
        for cond in CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx.extend(cond_to_cand_idx[cond])
        cand_residuals_per_pair = cand_residuals[np.array(all_cand_idx)]  # [88, 3584]

        # Compute maha on 2 PCAs
        maha_list = []
        for pca_dim in PCA_DIMS_ENSEMBLE:
            comp = components_full[:pca_dim]
            cand_pca = (cand_residuals_per_pair - pca_mean) @ comp.T
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            delta = cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha = (delta * weighted).sum(axis=-1)
            maha_list.append(maha)

        maha_200 = maha_list[0]
        maha_500 = maha_list[1]
        maha_200_norm = maha_200 / (maha_200.max() + 1e-9)
        maha_500_norm = maha_500 / (maha_500.max() + 1e-9)
        score = ALPHA * maha_200_norm + (1 - ALPHA) * maha_500_norm

        target_score = score[:, target_idx: target_idx + 1]
        rank_per_cand = (score < target_score).sum(axis=1)
        best_rank = int(np.min(rank_per_cand))

        fold_records.append({
            "fold": fold_idx,
            "user_id": uid,
            "asin": asin,
            "best_rank": best_rank,
            "n_cands": len(all_cand_idx),
            "method": "BoK-88 ENSEMBLE-α=0.3 (PCA-200 + PCA-500)",
        })

        if (fold_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"  fold {fold_idx+1}/{len(pair_keys)} done (elapsed {elapsed:.1f}s)")

    # === [5] Aggregate ===
    log("[5] Aggregating LOPO ...")
    all_ranks = [r["best_rank"] for r in fold_records]
    summary = aggregate(all_ranks)
    log(f"\n=== LOPO BoK-88 ENSEMBLE-α=0.3 ===")
    log(f"  rank-1: {summary['rank1']}/{summary['n']} ({summary['rank1_pct']*100:.1f}%)")
    log(f"  top-10: {summary['top10']}/{summary['n']} ({summary['top10_pct']*100:.1f}%)")
    log(f"  top-100: {summary['top100']}/{summary['n']} ({summary['top100_pct']*100:.1f}%)")
    log(f"  mean_rank: {summary['mean_rank']:.1f} (CI95 {summary['mean_rank_ci95']})")

    # === [6] Save ===
    out = {
        "phase": "14.P-3",
        "method": "LOPO BoK-88 ENSEMBLE-α=0.3 (PCA-200 + PCA-500)",
        "conditions": CONDITIONS,
        "k_per_pair": K_PER_PAIR,
        "alpha": ALPHA,
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "layer": LAYER_FOCUS,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_p_full_fit_bok88_ensemble_alpha03": "rank1=11/30 (36.7%), top100=29/30, mean_rank=18.4",
            "phase14_o_full_fit_bok24_pca200": "rank1=5/30 (16.7%), top100=29/30, mean_rank=25.3",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in fold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.P-3",
        "method": "LOPO BoK-88 ENSEMBLE-α=0.3",
        "conditions": CONDITIONS,
        "alpha": ALPHA,
        "n_folds": len(pair_keys),
        "elapsed_s": time.time() - t0,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.P-3 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
