#!/usr/bin/env python3
"""Phase 14.N: Leave-One-Pair-Out (LOPO) validation of Phase 14.M PCA + residual rerank.

Optimizations vs first version:
  1. Shared full SVD per fold: 1 SVD (top-500) → slice to 5 PCA dims (30/50/100/200/500)
     Saves 4× SVD per fold (~24s × 4 = ~96s).
  2. Multiprocessing: 30 folds in parallel across N_WORKERS (8 CPUs).
     Each worker ~30s → wall time ~2 min.
  3. Vectorized Maha computation (already optimized).

User constraint: "不要减少样本减少逻辑" — keep all 30 LOPO folds × 5 PCA dims × 3 conds.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_n_lopo_pca_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_n_lopo_pca_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_n_lopo_pca_meta.json"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584
K_PER_PAIR = 8
N_BOOTSTRAP = 2000

PCA_DIMS = [30, 50, 100, 200, 500]
PCA_DIMS_SORTED = sorted(PCA_DIMS)  # [30, 50, 100, 200, 500]
MAX_PCA = max(PCA_DIMS)  # 500 — do SVD to this many components

LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42

# Multiprocessing
N_WORKERS = 8  # 8 CPUs


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


def fit_pca_topk(train: np.ndarray, n_components: int):
    """PCA via Randomized SVD (sklearn) — 12x faster than full SVD.

    train: [N, H] float32
    Returns: components [k, H], mean [H], explained_var [k]
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=SEED)
    pca.fit(train)
    components = pca.components_.astype(np.float32)  # [k, H]
    mean = pca.mean_.astype(np.float32)
    explained_var = pca.explained_variance_.astype(np.float32)
    return components, mean, explained_var


def project_pca(X: np.ndarray, components: np.ndarray, mean: np.ndarray):
    """Project X [N, H] to PCA space [N, k]."""
    return (X - mean) @ components.T


def process_fold(fold_data):
    """Process one LOPO fold: compute best rank per (cond, pca_dim) using shared full SVD.

    fold_data: (fold_idx, user_residuals, cand_residuals, target_user_idx, target_cand_indices_per_cond)
        target_cand_indices_per_cond: dict cond → list of 8 cand indices
    """
    fold_idx, user_residuals, cand_residuals, target_user_idx, target_cand_indices_per_cond = fold_data

    # Build PCA training pool: exclude held-out user + held-out cands
    keep_user_mask = np.ones(N_USERS, dtype=bool)
    keep_user_mask[target_user_idx] = False
    user_residuals_train = user_residuals[keep_user_mask].reshape(-1, HIDDEN)  # [195*30, 3584]

    held_out_cand_idx = []
    for cond in CONDITIONS:
        held_out_cand_idx.extend(target_cand_indices_per_cond[cond])
    keep_cand_mask = np.ones(len(cand_residuals), dtype=bool)
    for idx in held_out_cand_idx:
        keep_cand_mask[idx] = False
    cand_residuals_train = cand_residuals[keep_cand_mask]

    train_pool = np.concatenate([user_residuals_train, cand_residuals_train], axis=0)

    # Shared full SVD to MAX_PCA components
    components_full, pca_mean, _ = fit_pca_topk(train_pool, MAX_PCA)

    # Held-out data for projection
    held_user_residuals = user_residuals[target_user_idx]  # [30, 3584]
    held_cand_residuals = cand_residuals[held_out_cand_idx]  # [24, 3584] (8 × 3 conds)

    # Results
    results = {"fold": fold_idx}

    for cond in CONDITIONS:
        local_idx = target_cand_indices_per_cond[cond]
        # Find offset within held_cand_residuals
        offset = 0
        for c in CONDITIONS:
            if c == cond:
                break
            offset += K_PER_PAIR
        local_cand_resid_8 = cand_residuals[local_idx]  # [8, 3584]

        # Full-no-PCA baseline (no projection)
        user_mu_full = user_residuals.mean(axis=1)  # [198, 3584]
        user_var_full = user_residuals.var(axis=1)
        pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
        user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
        user_var_full = np.maximum(user_var_full, MIN_VAR)
        pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(HIDDEN)
        inv_var_full = 1.0 / (pooled_var_diag_full + 1e-6)

        delta = local_cand_resid_8[:, None, :] - user_mu_full[None, :, :]
        weighted = delta * inv_var_full[None, None, :]
        maha_d_full = (delta * weighted).sum(axis=-1)
        target_d_full = maha_d_full[:, target_user_idx: target_user_idx + 1]
        rank_per_cand_full = (maha_d_full < target_d_full).sum(axis=1)
        best_rank_full = int(np.min(rank_per_cand_full))
        results[f"full_{cond}_rank"] = best_rank_full

        # Per-PCA-dim
        for pca_dim in PCA_DIMS:
            components = components_full[:pca_dim]  # [pca_dim, 3584]
            user_pca = project_pca(user_residuals.reshape(-1, HIDDEN), components, pca_mean)  # [5940, pca_dim]
            user_pca_3d = user_pca.reshape(N_USERS, 30, pca_dim)
            user_mu_pca = user_pca_3d.mean(axis=1)
            user_var_pca = user_pca_3d.var(axis=1)
            pooled_var_pca = user_var_pca.mean(axis=0, keepdims=True)
            user_var_pca = (1 - LW_SHRINK_USER) * user_var_pca + LW_SHRINK_USER * pooled_var_pca
            user_var_pca = np.maximum(user_var_pca, MIN_VAR)
            pooled_var_diag = (1 - LW_SHRINK_POOLED) * pooled_var_pca[0] + LW_SHRINK_POOLED * np.ones(pca_dim)
            inv_var = 1.0 / (pooled_var_diag + 1e-6)

            local_cand_pca = project_pca(local_cand_resid_8, components, pca_mean)  # [8, pca_dim]
            delta = local_cand_pca[:, None, :] - user_mu_pca[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha_d = (delta * weighted).sum(axis=-1)
            target_d = maha_d[:, target_user_idx: target_user_idx + 1]
            rank_per_cand = (maha_d < target_d).sum(axis=1)
            best_rank = int(np.min(rank_per_cand))
            results[f"pca{pca_dim}_{cond}_rank"] = best_rank

    return results


def main() -> None:
    log("=" * 70)
    log("Phase 14.N: Leave-One-Pair-Out validation of PCA + residual rerank (OPTIMIZED)")
    log("=" * 70)

    # === [1] Load candidates ===
    log(f"[1] Loading candidates ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load hiddens ===
    log(f"[2] Loading hiddens ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)  # [2976, 28, 3584]
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)  # [5, 3584]

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]  # [198, 30, 3584]
    cand_residuals_all = np.load(IN_CAND_RESID)  # [720, 5, 3584]
    cand_residuals = cand_residuals_all[:, layer_idx, :]  # [720, 3584]

    pair_info = []
    for (uid, asin) in pairs:
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            continue
        cond_to_cands = {}
        for cond in CONDITIONS:
            cond_to_cands[cond] = cand_by_pair_cond[(uid, asin)][cond]
        pair_info.append({
            "user_id": uid,
            "asin": asin,
            "user_idx": target_idx,
            "cond_to_cands": cond_to_cands,
        })
    log(f"  valid pairs: {len(pair_info)}")

    # === [3] Prepare fold data ===
    fold_data_list = []
    for fi, pair in enumerate(pair_info):
        fold_data_list.append((
            fi,
            user_residuals,
            cand_residuals,
            pair["user_idx"],
            pair["cond_to_cands"],
        ))

    # === [4] Multiprocessing LOPO ===
    log(f"[4] LOPO multiprocessing: {len(fold_data_list)} folds × {N_WORKERS} workers ...")

    # Set threads per worker = 1 to avoid contention
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    t0 = time.time()
    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_fold, fold_data_list)
    elapsed = time.time() - t0
    log(f"  LOPO done in {elapsed:.1f}s")

    # Sort by fold idx
    results = sorted(results, key=lambda x: x["fold"])

    # === [5] Aggregate ===
    log("[5] Aggregating LOPO results ...")

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

    summary = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for pca_dim in PCA_DIMS:
            ranks = [r[f"pca{pca_dim}_{cond}_rank"] for r in results]
            summary[cond][f"lopo_pca_{pca_dim}"] = aggregate(ranks)
        ranks_full = [r[f"full_{cond}_rank"] for r in results]
        summary[cond]["full_no_pca"] = aggregate(ranks_full)

    # Print summary
    log("\n=== LOPO Summary (rank-1 / top-100 / mean_rank) ===")
    for cond in CONDITIONS:
        log(f"\n{cond}:")
        for pca_dim in PCA_DIMS:
            s = summary[cond][f"lopo_pca_{pca_dim}"]
            log(
                f"  LOPO PCA-{pca_dim}: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
                f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), "
                f"mean_rank={s['mean_rank']:.1f} CI [{s['mean_rank_ci95'][0]:.1f}, {s['mean_rank_ci95'][1]:.1f}]"
            )
        s = summary[cond]["full_no_pca"]
        log(
            f"  FULL-no-PCA:    rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
            f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), "
            f"mean_rank={s['mean_rank']:.1f} CI [{s['mean_rank_ci95'][0]:.1f}, {s['mean_rank_ci95'][1]:.1f}]"
        )

    # === [6] Save ===
    out = {
        "phase": "14.N",
        "method": "Leave-One-Pair-Out (LOPO) validation of PCA + residual rerank",
        "layer": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "lw_shrink_user": LW_SHRINK_USER,
        "lw_shrink_pooled": LW_SHRINK_POOLED,
        "min_var": MIN_VAR,
        "n_pairs": len(pair_info),
        "n_users": N_USERS,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "comparison_phase14_m_full_fit": {
            "A22_a0.5_pca30_rank1": "4/30 (13.3%)",
            "A14_a1.0_pca200_top100": "28/30 (93.3%)",
        },
        "interpretation": (
            "LOPO holds out 1 pair → its user + its 8 cands excluded from PCA fit. "
            "If LOPO results ≈ Phase 14.M full-fit results, PCA generalizes (no overfitting). "
            "If LOPO results << Phase 14.M, PCA overfits to the 30 specific users."
        ),
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.N",
        "method": "LOPO PCA + residual Maha",
        "layer_focus": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "n_pairs": len(pair_info),
        "n_users_cache": N_USERS,
        "elapsed_seconds": elapsed,
        "n_workers": N_WORKERS,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.N COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()