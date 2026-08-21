#!/usr/bin/env python3
"""Phase 14.Q-4: Evaluate Q conds vs control with Phase 14.P-5 SOTA rerank.

Compares:
  - D_off (uniform control, 8 cands)
  - A14_a1.0 (StyleVector control, 8 cands)
  - Q_Dynamic (text conditions only, 8 cands)
  - Q_DynamicExemplar (text + 2 exemplars, 8 cands)
  - BoK-32 (combined 4 conds × 8)
  - BoK-120 (4 conds × 8 + 88 cached conds × 8 = 32 + 88 = 120)
  - BoK-88 (11 cached conds only, baseline Phase 14.P-5)

All evaluated with PCA-300+PCA-500 α=0.25 best-of-K Maha.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0042/wenyu/vades_prototype") if False else Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

# Inputs
IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"  # for control conds (cached 11 conds)
IN_PAIRS_Q = OUT_DIR / "phase14_q_generated.jsonl"  # for Q_Dynamic + Q_DynamicExemplar
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"  # 2640 cands
IN_CAND_RESID_Q = OUT_DIR / "phase14_q_cand_residuals_qwen.npy"  # 480 cands

OUT_EVAL = OUT_DIR / "phase14_q_eval_fixed_pca.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q_per_pair_fixed_pca.jsonl"
OUT_META = OUT_DIR / "phase14_q_eval_fixed_pca_meta.json"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584

# Phase 14.P-5 SOTA config
PCA_DIMS_ENSEMBLE = [300, 500]
ALPHA = 0.25
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000

# 11 cached conds (Phase 14.P baseline)
CACHED_CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
                     "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]

# 2 new Q conds
Q_CONDITIONS = ["Q_Dynamic", "Q_DynamicExemplar"]


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
    log("Phase 14.Q-4: Evaluate Q conds vs control with Phase 14.P-5 SOTA")
    log("=" * 70)

    # === [1] Load candidates from BOTH sources ===
    log("[1] Loading candidates ...")
    candidates = []

    # 11 cached conds from phase14_b
    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            candidates.append(r)
    log(f"  from phase14_b: {len(candidates)}")

    # 2 Q conds from phase14_q_generated
    n_q = 0
    with IN_PAIRS_Q.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            candidates.append(r)
            n_q += 1
    log(f"  from phase14_q: {n_q}")
    log(f"  total: {len(candidates)} (2640 + 480 = 3120)")

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})

    # Map: (uid, asin) → cond → list of cand idx (in full array)
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

    # Load cand residuals (combine 2640 + 480 = 3120)
    log("[3] Loading cand residuals ...")
    cand_resid_11 = np.load(IN_CAND_RESID_11)  # [2640, 5, 3584]
    cand_resid_q = np.load(IN_CAND_RESID_Q)  # [480, 5, 3584]
    cand_residuals_all = np.concatenate([cand_resid_11, cand_resid_q], axis=0)
    log(f"  cand_residuals_all shape: {cand_residuals_all.shape}")

    cand_residuals = cand_residuals_all[:, layer_idx, :]  # [3120, 3584]
    cand_residuals_cached_only = cand_resid_11[:, layer_idx, :]  # [2640, 3584]

    # === [4] Fit PCA-500 once ===
    # IMPORTANT: Fit PCA on user + 2640 CACHED cands only (NOT Q cands),
    # so the basis matches Phase 14.P-5. Q cands are projected onto this fixed basis.
    log(f"[4] Fitting Randomized PCA-{MAX_PCA} on (user + 2640 cached cands) ...")
    user_residuals_flat = user_residuals.reshape(-1, HIDDEN)
    train_pool = np.concatenate([user_residuals_flat, cand_residuals_cached_only], axis=0)

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

    # === [5] Pre-compute per-pair 2-PCA maha ===
    log("[5] Pre-computing per-pair mahas (PCA-300 + PCA-500) ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    per_pair_data = []
    for pi, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]

        # Compute maha on all 3120 cands for this pair
        all_cand_idx = []
        for cond in candidates_keys_for_pair(cond_to_cand_idx):
            all_cand_idx.extend(cond_to_cand_idx[cond])
        all_cand_idx = np.array(all_cand_idx)

        maha_list = []
        for pca_dim in PCA_DIMS_ENSEMBLE:
            comp = components_full[:pca_dim]
            cand_pca = cand_pca_full[all_cand_idx, :pca_dim]
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            delta = cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha = (delta * weighted).sum(axis=-1)
            maha_list.append(maha)

        maha_300 = maha_list[0]
        maha_500 = maha_list[1]
        maha_300_norm = maha_300 / (maha_300.max() + 1e-9)
        maha_500_norm = maha_500 / (maha_500.max() + 1e-9)
        score = ALPHA * maha_300_norm + (1 - ALPHA) * maha_500_norm  # [N, 198]

        target_score = score[:, target_idx: target_idx + 1]
        rank_per_cand = (score < target_score).sum(axis=1)  # [N]

        per_pair_data.append({
            "uid": uid,
            "asin": asin,
            "target_idx": target_idx,
            "cond_to_cand_idx": cond_to_cand_idx,
            "all_cand_idx": all_cand_idx,
            "rank_per_cand": rank_per_cand,
            "score": score,
        })

        if (pi + 1) % 10 == 0:
            log(f"  pair {pi+1}/{len(pair_keys)} done")

    # === [6] Strategies ===
    log("[6] Evaluating strategies ...")

    # Define condition subsets
    subsets = {
        "D_off_8c": ["D_off"],  # uniform control
        "A14_a1.0_8c": ["A14_a1.0"],  # StyleVector control (Phase 14.M SOTA)
        "Q_Dynamic_8c": ["Q_Dynamic"],  # text conditions only
        "Q_DynamicExemplar_8c": ["Q_DynamicExemplar"],  # text + exemplars
        "BoK_4_conds_32c": CACHED_CONDITIONS[:3] + Q_CONDITIONS,  # 11+2 = 4 conds = 32 cands
        "BoK_4_control_24c": CACHED_CONDITIONS[:3],  # D_off + A22_a0.5 + A14_a1.0 = 24 cands (Phase 14.O)
        "BoK_4_q_16c": Q_CONDITIONS,  # just 2 Q conds = 16 cands
        "BoK_5_conds_40c": ["A22_a0.5", "A14_a1.0", "D_off"] + Q_CONDITIONS,
        "BoK_88_88c": CACHED_CONDITIONS,  # 11 cached = 88 cands (Phase 14.P SOTA)
        "BoK_120_88plus32c": CACHED_CONDITIONS + Q_CONDITIONS,  # 88 + 32 = 120 cands
        "BoK_88_plus_q_8c": CACHED_CONDITIONS + ["Q_DynamicExemplar"],  # 11 + 1 = 96 cands
    }

    summary = {}
    for subset_name, subset_conds in subsets.items():
        log(f"\n--- {subset_name} ---")
        ranks = []
        for p in per_pair_data:
            # Gather cands for this subset
            sub_cand_idx = []
            for cond in subset_conds:
                if cond in p["cond_to_cand_idx"]:
                    sub_cand_idx.extend(p["cond_to_cand_idx"][cond])
            sub_cand_idx_set = set(sub_cand_idx)
            # Filter rank_per_cand
            local_indices = [i for i, ci in enumerate(p["all_cand_idx"]) if ci in sub_cand_idx_set]
            sub_ranks = p["rank_per_cand"][local_indices]
            best_rank = int(np.min(sub_ranks))
            ranks.append(best_rank)
        agg = aggregate(ranks)
        summary[subset_name] = agg
        log(f"  rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # === [7] Per-pair detail for top configs ===
    log("\n[7] Per-pair detail ...")
    per_pair_records = []
    for p in per_pair_data:
        record = {"user_id": p["uid"], "asin": p["asin"], "n_users": N_USERS}
        for subset_name in ["D_off_8c", "A14_a1.0_8c", "Q_Dynamic_8c", "Q_DynamicExemplar_8c",
                            "BoK_4_conds_32c", "BoK_88_88c", "BoK_120_88plus32c"]:
            subset_conds = subsets[subset_name]
            sub_cand_idx = []
            for cond in subset_conds:
                if cond in p["cond_to_cand_idx"]:
                    sub_cand_idx.extend(p["cond_to_cand_idx"][cond])
            sub_cand_idx_set = set(sub_cand_idx)
            local_indices = [i for i, ci in enumerate(p["all_cand_idx"]) if ci in sub_cand_idx_set]
            sub_ranks = p["rank_per_cand"][local_indices]
            best_rank = int(np.min(sub_ranks))
            record[f"{subset_name}_rank"] = best_rank
            record[f"{subset_name}_n_cands"] = len(sub_cand_idx)
        per_pair_records.append(record)

    with OUT_PER_PAIR.open("w") as f:
        for r in per_pair_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    # === [8] Save summary ===
    log("\n=== Phase 14.Q-4 Summary ===")
    for subset_name, agg in summary.items():
        log(f"  {subset_name}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    out = {
        "phase": "14.Q-4",
        "method": "Phase 14.P-5 SOTA (PCA-300+PCA-500 α=0.25) rerank for 4 conds + subsets",
        "conditions_cached": CACHED_CONDITIONS,
        "conditions_q": Q_CONDITIONS,
        "layer": LAYER_FOCUS,
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "alpha": ALPHA,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_p5_full_fit_bok88_ensemble_alpha025": "rank1=15/30, top100=29/30, mean_rank=20.1",
            "phase14_p_lopo_bok88_ensemble_alpha025": "rank1=15/30, top100=29/30, mean_rank=20.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    meta = {
        "phase": "14.Q-4",
        "method": "Q conds evaluation",
        "n_pairs": len(pair_keys),
        "subsets_evaluated": list(subsets.keys()),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-4 COMPLETE")
    log("=" * 70)


def candidates_keys_for_pair(cond_to_cand_idx):
    """Return conds sorted by stable order."""
    conds = []
    for c in CACHED_CONDITIONS + Q_CONDITIONS:
        if c in cond_to_cand_idx:
            conds.append(c)
    return conds


if __name__ == "__main__":
    main()