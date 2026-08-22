#!/usr/bin/env python3
"""Phase 14.P-4: α fine sweep + 3-PCA ensemble over BoK-88.

Goal: find optimal α (PCA-200 weight) in [0.1, 0.4] and test if 3-PCA ensemble
(PCA-100 + PCA-200 + PCA-500) beats 2-PCA.

Tests:
  1. α sweep [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7] for PCA-200 + PCA-500
  2. 3-PCA ensemble: PCA-100 + PCA-200 + PCA-500 (equal weights)
  3. 3-PCA ensemble: PCA-100 + PCA-300 + PCA-500 (equal weights)
  4. Single-PCA best: PCA-50, PCA-100, PCA-200, PCA-300, PCA-500
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

OUT_EVAL = OUT_DIR / "phase14_p_alpha_sweep_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_p_alpha_sweep_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_p_alpha_sweep_meta.json"

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

# PCA dims to fit (single shot)
PCA_DIMS_ALL = [50, 100, 150, 200, 250, 300, 400, 500]
MAX_PCA = max(PCA_DIMS_ALL)


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
    log("Phase 14.P-4: α fine sweep + 3-PCA ensemble over BoK-88")
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

    # === [4] Pre-compute all PCA-dim mahas per pair ===
    log("[4] Pre-computing per-pair mahas for 8 PCA dims ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    # Per-pair: dict {pca_dim: maha [88, 198]}
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
        for pca_dim in PCA_DIMS_ALL:
            comp = components_full[:pca_dim]
            cand_pca = cand_pca_full[cand_idx_arr, :pca_dim]  # [88, pca_dim]
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            delta = cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha = (delta * weighted).sum(axis=-1)  # [88, 198]
            maha_dict[pca_dim] = maha

        per_pair_mahas.append({
            "uid": uid,
            "asin": asin,
            "target_idx": target_idx,
            "cand_idx_arr": cand_idx_arr,
            "mahas": maha_dict,  # {pca_dim: [88, 198]}
        })

    log(f"  pre-computed mahas for {len(per_pair_mahas)} pairs × {len(PCA_DIMS_ALL)} PCA dims")

    # === [5] Strategies ===
    log("[5] Evaluating strategies ...")

    # Normalize each Maha (across all 88 cands × 198 users in the dataset)
    # Use the global max from all pairs for each PCA dim
    def normalize_mahas(maha_dict):
        return {pca_dim: m / (m.max() + 1e-9) for pca_dim, m in maha_dict.items()}

    # Strategy 1: BoK-N at single PCA dims
    def single_pca_results(pca_dim):
        ranks = []
        for p in per_pair_mahas:
            maha = p["mahas"][pca_dim]
            target_idx = p["target_idx"]
            target_d = maha[:, target_idx: target_idx + 1]
            rank_per_cand = (maha < target_d).sum(axis=1)
            ranks.append(int(np.min(rank_per_cand)))
        return ranks

    # Strategy 2: 2-PCA ensemble α sweep
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

    # Strategy 3: 3-PCA ensemble (equal weights)
    def ensemble_3pca(pca_dims):
        ranks = []
        for p in per_pair_mahas:
            ms = [p["mahas"][d] / (p["mahas"][d].max() + 1e-9) for d in pca_dims]
            score = sum(ms) / len(ms)
            target_idx = p["target_idx"]
            target_score = score[:, target_idx: target_idx + 1]
            rank_per_cand = (score < target_score).sum(axis=1)
            ranks.append(int(np.min(rank_per_cand)))
        return ranks

    # Evaluate
    summary = {}

    # Single PCA baselines
    log("\n=== Single PCA baselines (BoK-88) ===")
    summary["single_pca"] = {}
    for pca_dim in PCA_DIMS_ALL:
        ranks = single_pca_results(pca_dim)
        agg = aggregate(ranks)
        summary["single_pca"][f"PCA-{pca_dim}"] = agg
        log(f"  PCA-{pca_dim}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # 2-PCA ensemble α sweep (PCA-200 + PCA-500)
    log("\n=== 2-PCA ensemble: PCA-200 + PCA-500 (α sweep) ===")
    summary["ensemble_2pca_200_500"] = {}
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        ranks = ensemble_2pca([200, 500], alpha)
        agg = aggregate(ranks)
        summary["ensemble_2pca_200_500"][f"alpha_{alpha}"] = agg
        log(f"  α={alpha}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # 2-PCA ensemble other PCA pairs (PCA-100 + PCA-300, etc.)
    log("\n=== 2-PCA ensemble: other PCA pairs (α=0.3 fixed) ===")
    summary["ensemble_2pca_pairs"] = {}
    for pair in [(100, 300), (100, 500), (200, 300), (200, 400), (300, 500), (50, 500)]:
        ranks = ensemble_2pca(pair, 0.3)
        agg = aggregate(ranks)
        summary["ensemble_2pca_pairs"][f"PCA-{pair[0]}_PCA-{pair[1]}"] = agg
        log(f"  PCA-{pair[0]}+PCA-{pair[1]} α=0.3: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # 3-PCA ensemble
    log("\n=== 3-PCA ensemble (equal weights) ===")
    summary["ensemble_3pca"] = {}
    for triple in [(100, 200, 500), (100, 300, 500), (200, 300, 500), (50, 200, 500)]:
        ranks = ensemble_3pca(list(triple))
        agg = aggregate(ranks)
        summary["ensemble_3pca"][f"PCA-{triple[0]}_PCA-{triple[1]}_PCA-{triple[2]}"] = agg
        log(f"  PCA-{triple[0]}+PCA-{triple[1]}+PCA-{triple[2]}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # === [6] Save ===
    out = {
        "phase": "14.P-4",
        "method": "α fine sweep + 3-PCA ensemble over BoK-88",
        "conditions": CONDITIONS,
        "layer": LAYER_FOCUS,
        "pca_dims": PCA_DIMS_ALL,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_p_full_fit_bok88_ensemble_alpha03": "rank1=11/30, top100=29/30, mean_rank=18.4",
            "phase14_p_lopo_bok88_ensemble_alpha03": "rank1=11/30, top100=29/30, mean_rank=18.4",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    meta = {
        "phase": "14.P-4",
        "method": "α fine sweep + 3-PCA ensemble",
        "pca_dims": PCA_DIMS_ALL,
        "n_pairs": len(pair_keys),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.P-4 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
