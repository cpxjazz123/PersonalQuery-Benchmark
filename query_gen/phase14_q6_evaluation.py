#!/usr/bin/env python3
"""Phase 14.Q-6: Evaluate Q_DynamicExemplar4 vs Phase 14.P-5 SOTA rerank.

Strategy:
  - Fit PCA on (user + 2640 cached) only (same as Phase 14.P-5)
  - Project Q_Dynamic(4) cands onto fixed basis
  - Evaluate:
    - Q_Dynamic_4 (text only, 8 cands)
    - Q_DynamicExemplar4 (4 exemplars, 8 cands)
    - BoK_96 (88 + Q_DynamicExemplar4 = 96)
    - BoK_104 (88 + 16 Q4 cands)
    - BoK_120 (88 + Q_DynamicEx_2 + Q_DynamicEx_4 = 88 + 8 + 8 = 104)
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_Q = OUT_DIR / "phase14_q_generated.jsonl"  # 2-exemplar Q
IN_PAIRS_Q6 = OUT_DIR / "phase14_q6_generated.jsonl"  # 4-exemplar Q
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"
IN_CAND_RESID_Q = OUT_DIR / "phase14_q_cand_residuals_qwen.npy"
IN_CAND_RESID_Q6 = OUT_DIR / "phase14_q6_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_q6_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q6_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_q6_eval_meta.json"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584

PCA_DIMS_ENSEMBLE = [300, 500]
ALPHA = 0.25
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000

CACHED_CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
                     "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=SEED):
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
    log("Phase 14.Q-6: Evaluate Q_DynamicExemplar4 vs Phase 14.P-5 SOTA")
    log("=" * 70)

    # === [1] Load candidates ===
    log("[1] Loading candidates ...")
    candidates = []
    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  phase14_b cached: {len(candidates)}")

    n_q = 0
    with IN_PAIRS_Q.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
            n_q += 1
    log(f"  + Q-2 conds: {n_q}")

    n_q6 = 0
    with IN_PAIRS_Q6.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
            n_q6 += 1
    log(f"  + Q-6 conds: {n_q6}")
    log(f"  total: {len(candidates)} (2640 + 480 + 480 = 3600)")

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

    log("[3] Loading cand residuals ...")
    cand_resid_11 = np.load(IN_CAND_RESID_11)
    cand_resid_q = np.load(IN_CAND_RESID_Q)
    cand_resid_q6 = np.load(IN_CAND_RESID_Q6)
    cand_residuals_all = np.concatenate([cand_resid_11, cand_resid_q, cand_resid_q6], axis=0)
    log(f"  cand_residuals_all shape: {cand_residuals_all.shape}")
    cand_residuals = cand_residuals_all[:, layer_idx, :]
    cand_residuals_cached_only = cand_resid_11[:, layer_idx, :]

    # === [3.5] Build a robust condition label map ===
    # In our concatenation order: first 2640 are cached (idx 0-2639), then Q-2 (idx 2640-3119), then Q-6 (idx 3120-3599)
    # cand_by_pair_cond gives global indices already (idx in [0, 3600))
    log(f"  Total cands: {len(candidates)} = 2640 (cached) + 480 (Q-2) + 480 (Q-6)")

    # === [4] Fit PCA-500 on user + cached only ===
    log(f"[4] Fitting Randomized PCA-{MAX_PCA} on (user + 2640 cached) ...")
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

    # We need to compute score on ALL 3600 cands per pair (for filtering later)
    per_pair_score = {}
    for pi, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        all_cond_pairs = [(c["user_id"], c["asin"], c["condition"]) for c in candidates]
        all_cand_idx = [i for i, (u, a, _) in enumerate(all_cond_pairs) if u == uid and a == asin]
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

        per_pair_score[(uid, asin)] = (all_cand_idx, rank_per_cand)

        if (pi + 1) % 10 == 0:
            log(f"  pair {pi+1}/{len(pair_keys)} done")

    # === [6] Strategies ===
    log("[6] Evaluating strategies ...")

    subsets = {
        # Q only
        "Q_Dynamic_8c": ["Q_Dynamic"],  # 2-exemplar (Phase 14.Q-1)
        "Q_Dynamic4_8c": ["Q_Dynamic"],  # 4-exemplar (Phase 14.Q-6 — same cond name, but different cache)
        "Q_DynamicExemplar_8c": ["Q_DynamicExemplar"],
        "Q_DynamicExemplar4_8c": ["Q_DynamicExemplar4"],
        # Cached baseline
        "BoK_88_88c": CACHED_CONDITIONS,
        # Mix
        "BoK_96_88plus_q2_8c": CACHED_CONDITIONS + ["Q_DynamicExemplar"],
        "BoK_96_88plus_q4_8c": CACHED_CONDITIONS + ["Q_DynamicExemplar4"],
        "BoK_104_88plus_q2_q4_16c": CACHED_CONDITIONS + ["Q_DynamicExemplar", "Q_DynamicExemplar4"],
        "BoK_120_88plus_allQ_32c": CACHED_CONDITIONS + ["Q_Dynamic", "Q_DynamicExemplar", "Q_DynamicExemplar4"],
    }

    summary = {}
    for subset_name, subset_conds in subsets.items():
        log(f"\n--- {subset_name} ---")
        ranks = []
        for (uid, asin) in pair_keys:
            all_cand_idx, rank_per_cand = per_pair_score[(uid, asin)]
            # Filter by condition
            sub_cand_idx_set = set()
            for cond in subset_conds:
                sub_cand_idx_set.update(cand_by_pair_cond[(uid, asin)].get(cond, []))
            local_indices = [i for i, ci in enumerate(all_cand_idx) if ci in sub_cand_idx_set]
            if not local_indices:
                ranks.append(200)  # failed
                continue
            sub_ranks = rank_per_cand[local_indices]
            best_rank = int(np.min(sub_ranks))
            ranks.append(best_rank)
        agg = aggregate(ranks)
        summary[subset_name] = agg
        log(f"  rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    # === [7] Per-pair detail ===
    log("\n[7] Per-pair detail ...")
    per_pair_records = []
    for (uid, asin) in pair_keys:
        all_cand_idx, rank_per_cand = per_pair_score[(uid, asin)]
        record = {"user_id": uid, "asin": asin, "n_users": N_USERS}
        for subset_name in ["Q_Dynamic_8c", "Q_DynamicExemplar_8c", "Q_DynamicExemplar4_8c",
                            "BoK_88_88c", "BoK_96_88plus_q2_8c", "BoK_96_88plus_q4_8c",
                            "BoK_104_88plus_q2_q4_16c"]:
            subset_conds = subsets[subset_name]
            sub_cand_idx_set = set()
            for cond in subset_conds:
                sub_cand_idx_set.update(cand_by_pair_cond[(uid, asin)].get(cond, []))
            local_indices = [i for i, ci in enumerate(all_cand_idx) if ci in sub_cand_idx_set]
            if not local_indices:
                record[f"{subset_name}_rank"] = -1
                record[f"{subset_name}_n_cands"] = 0
                continue
            sub_ranks = rank_per_cand[local_indices]
            best_rank = int(np.min(sub_ranks))
            record[f"{subset_name}_rank"] = best_rank
            record[f"{subset_name}_n_cands"] = len(local_indices)
        per_pair_records.append(record)

    with OUT_PER_PAIR.open("w") as f:
        for r in per_pair_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    # === [8] Save summary ===
    log("\n=== Phase 14.Q-6 Summary ===")
    for subset_name, agg in summary.items():
        log(f"  {subset_name}: rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), mean_rank={agg['mean_rank']:.1f}")

    out = {
        "phase": "14.Q-6",
        "method": "Q_DynamicExemplar4 (4 exemplars) on Phase 14.P-5 SOTA rerank",
        "conditions_cached": CACHED_CONDITIONS,
        "q_conditions": ["Q_Dynamic", "Q_DynamicExemplar", "Q_DynamicExemplar4"],
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "alpha": ALPHA,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_p5_full_fit_bok88_pca300_500_alpha025": "rank1=15/30, top100=29/30, mean_rank=20.1",
            "phase14_q_full_fit_bok96_pca300_500_alpha025": "rank1=16/30 (53.3%), top100=29/30, mean_rank=15.5",
            "phase14_q_lopo_bok96_pca300_500_alpha025": "rank1=15/30 (50.0%), top100=29/30, mean_rank=19.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    meta = {
        "phase": "14.Q-6",
        "method": "Q_DynamicExemplar4 evaluation",
        "n_pairs": len(pair_keys),
        "subsets_evaluated": list(subsets.keys()),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-6 EVAL COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
