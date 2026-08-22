#!/usr/bin/env python3
"""Phase 14.P-5: Fine α sweep for PCA-300+PCA-500 (rank-1 14/30 46.7% NEW SOTA).

Also tests other promising pairs: PCA-200+PCA-400, PCA-300+PCA-400, PCA-100+PCA-300.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_p_alpha_sweep_v2_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_p_alpha_sweep_v2_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_p_alpha_sweep_v2_meta.json"

CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
              "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]
K_PER_PAIR = 8
LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000

MAX_PCA = 500


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


def main() -> None:
    log("=" * 70)
    log("Phase 14.P-5: Fine α sweep for PCA-300+PCA-500")
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

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
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

    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    # === [4] Pre-compute mahas for PCA dims of interest ===
    log("[4] Pre-computing per-pair mahas ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    PCA_DIMS_INTEREST = [50, 100, 150, 200, 250, 300, 350, 400, 500]

    per_pair_mahas = []
    for pi, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]
        all_cand_idx = []
        for cond in CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx.extend(cond_to_cand_idx[cond])
        cand_idx_arr = np.array(all_cand_idx)

        maha_dict = {}
        for pca_dim in PCA_DIMS_INTEREST:
            comp = components_full[:pca_dim]
            cand_pca = cand_pca_full[cand_idx_arr, :pca_dim]
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            delta = cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha = (delta * weighted).sum(axis=-1)
            maha_dict[pca_dim] = maha

        per_pair_mahas.append({
            "uid": uid,
            "asin": asin,
            "target_idx": target_idx,
            "mahas": maha_dict,
        })

    log(f"  pre-computed for {len(per_pair_mahas)} pairs × {len(PCA_DIMS_INTEREST)} PCA dims")

    # === [5] Strategies ===
    log("[5] Evaluating ...")

    def ensemble_2pca(pca_dims, alpha):
        ranks = []
        for p in per_pair_mahas:
            m1 = p["mahas"][pca_dims[0]] / (p["mahas"][pca_dims[0]].max() + 1e-9)
            m2 = p["mahas"][pca_dims[1]] / (p["mahas"][pca_dims[1]].max() + 1e-9)
            score = alpha * m1 + (1 - alpha) * m2
            target_idx = p["target_idx"]
            target_score = score[:, target_idx: target_idx + 1]
            rank_per_cand = (score < target_score).sum(axis=1)
            ranks.append(int(np.min(rank_per_cand)))
        return ranks

    summary = {}

    # PCA-300+PCA-500 α sweep (the new SOTA pair)
    log("\n=== PCA-300+PCA-500 α fine sweep (NEW SOTA 14/30) ===")
    summary["ensemble_300_500"] = {}
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        ranks = ensemble_2pca([300, 500], alpha)
        agg = aggregate(ranks)
        summary["ensemble_300_500"][f"alpha_{alpha}"] = agg
        log(f"  α={alpha}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # PCA-200+PCA-400 α sweep
    log("\n=== PCA-200+PCA-400 α fine sweep ===")
    summary["ensemble_200_400"] = {}
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        ranks = ensemble_2pca([200, 400], alpha)
        agg = aggregate(ranks)
        summary["ensemble_200_400"][f"alpha_{alpha}"] = agg
        log(f"  α={alpha}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # PCA-300+PCA-400 α sweep
    log("\n=== PCA-300+PCA-400 α fine sweep ===")
    summary["ensemble_300_400"] = {}
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        ranks = ensemble_2pca([300, 400], alpha)
        agg = aggregate(ranks)
        summary["ensemble_300_400"][f"alpha_{alpha}"] = agg
        log(f"  α={alpha}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # PCA-200+PCA-350 α sweep
    log("\n=== PCA-200+PCA-350 α fine sweep ===")
    summary["ensemble_200_350"] = {}
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        ranks = ensemble_2pca([200, 350], alpha)
        agg = aggregate(ranks)
        summary["ensemble_200_350"][f"alpha_{alpha}"] = agg
        log(f"  α={alpha}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # === [6] Save ===
    out = {
        "phase": "14.P-5",
        "method": "α fine sweep for PCA-300+PCA-500 (NEW SOTA 14/30) + similar pairs",
        "conditions": CONDITIONS,
        "layer": LAYER_FOCUS,
        "pca_dims_interest": PCA_DIMS_INTEREST,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_p_full_fit_bok88_ensemble_alpha03_pca200_500": "rank1=11/30, top100=29/30, mean_rank=18.4",
            "phase14_p_full_fit_pca300_500_alpha03": "rank1=14/30, top100=29/30, mean_rank=20.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    meta = {
        "phase": "14.P-5",
        "method": "α fine sweep PCA-300+PCA-500",
        "n_pairs": len(pair_keys),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.P-5 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
