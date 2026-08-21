#!/usr/bin/env python3
"""Phase 14.Q9-1: Filter broken (non-ASCII) StyleVector outputs + rerun LOPO.

Test if removing broken cands reduces rerank pool contamination and improves
rank-1 on the 12 hard pairs.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_Q8 = OUT_DIR / "phase14_q8_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"
IN_CAND_RESID_Q8 = OUT_DIR / "phase14_q8_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_q9_filter_lopo_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q9_filter_lopo_per_pair.jsonl"

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

# Filter threshold: drop cands with > N% non-ASCII chars
NON_ASCII_THRESHOLD = 0.1

CACHED_CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
                     "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]
Q8_CONDITIONS = [f"Q_DynamicExemplar4_s{i}" for i in range(1, 9)]
SELECTED_CONDITIONS = CACHED_CONDITIONS + Q8_CONDITIONS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_broken(text: str, threshold: float = NON_ASCII_THRESHOLD) -> bool:
    if not text:
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii > len(text) * threshold


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
    log(f"Phase 14.Q9-1: Filter broken cands (non-ASCII > {NON_ASCII_THRESHOLD*100:.0f}%)")
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
    n_cached = len(candidates)
    cand_resid_11 = np.load(IN_CAND_RESID_11)
    log(f"  cached: {n_cached}")

    with IN_PAIRS_Q8.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    cand_resid_q8 = np.load(IN_CAND_RESID_Q8)
    n_q8 = len(candidates) - n_cached
    log(f"  + Q8: {n_q8}, total: {len(candidates)}")

    cand_residuals_all = np.concatenate([cand_resid_11, cand_resid_q8], axis=0)

    # === [2] Filter broken ===
    log(f"\n[2] Filtering broken (non-ASCII > {NON_ASCII_THRESHOLD*100:.0f}%) ...")
    keep_mask = []
    n_kept = 0
    n_dropped = 0
    drop_by_cond = defaultdict(int)
    for c in candidates:
        text = c.get("q_final_post", "")
        if is_broken(text, NON_ASCII_THRESHOLD):
            keep_mask.append(False)
            n_dropped += 1
            drop_by_cond[c["condition"]] += 1
        else:
            keep_mask.append(True)
            n_kept += 1
    keep_mask = np.array(keep_mask)
    log(f"  kept: {n_kept} / dropped: {n_dropped}")
    for cond in sorted(drop_by_cond.keys()):
        if drop_by_cond[cond] > 0:
            total_for_cond = sum(1 for c in candidates if c["condition"] == cond)
            log(f"    {cond}: {drop_by_cond[cond]}/{total_for_cond} dropped ({drop_by_cond[cond]/total_for_cond*100:.1f}%)")

    cand_residuals_filtered = cand_residuals_all[keep_mask]
    cand_residuals_cached_only = cand_resid_11[:, LAYERS_5.index(LAYER_FOCUS), :]  # always use uncached for PCA basis
    candidates_filtered = [c for c, k in zip(candidates, keep_mask) if k]

    # Re-index
    cand_by_pair_cond_filtered = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates_filtered):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond_filtered[key][c["condition"]].append(ci)

    # === [3] Load hiddens + neutral ===
    log("\n[3] Loading user hiddens ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals = cand_residuals_filtered[:, layer_idx, :]

    # === [4] Fit PCA-500 on user + 2640 cached (unfiltered) ===
    log(f"\n[4] Fitting Randomized PCA-{MAX_PCA} on (user + 2640 cached, unfiltered) ...")
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

    # === [5] LOPO ===
    log(f"\n[5] LOPO over 30 folds ...")
    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates_filtered})
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    fold_records = []
    t0 = time.time()

    for fold_idx, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        cond_to_cand_idx = cand_by_pair_cond_filtered[(uid, asin)]

        all_cand_idx = []
        for cond in SELECTED_CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx.extend(cond_to_cand_idx[cond])

        if not all_cand_idx:
            fold_records.append({"fold": fold_idx, "user_id": uid, "asin": asin,
                                 "best_rank": 200, "n_cands": 0})
            continue

        cand_residuals_per_pair = cand_residuals[np.array(all_cand_idx)]

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

        maha_300 = maha_list[0]
        maha_500 = maha_list[1]
        maha_300_norm = maha_300 / (maha_300.max() + 1e-9)
        maha_500_norm = maha_500 / (maha_500.max() + 1e-9)
        score = ALPHA * maha_300_norm + (1 - ALPHA) * maha_500_norm

        target_score = score[:, target_idx: target_idx + 1]
        rank_per_cand = (score < target_score).sum(axis=1)
        best_rank = int(np.min(rank_per_cand))

        fold_records.append({
            "fold": fold_idx,
            "user_id": uid,
            "asin": asin,
            "best_rank": best_rank,
            "n_cands": len(all_cand_idx),
        })

        if (fold_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"  fold {fold_idx+1}/{len(pair_keys)} done (elapsed {elapsed:.1f}s)")

    # === [6] Aggregate ===
    log("\n[6] Aggregating LOPO ...")
    all_ranks = [r["best_rank"] for r in fold_records]
    summary = aggregate(all_ranks)
    log(f"\n=== LOPO BoK (88+64 Q4 8-subsets, FILTERED) PCA-300+PCA-500 α={ALPHA} ===")
    log(f"  rank-1: {summary['rank1']}/{summary['n']} ({summary['rank1_pct']*100:.1f}%)")
    log(f"  top-10: {summary['top10']}/{summary['n']} ({summary['top10_pct']*100:.1f}%)")
    log(f"  top-100: {summary['top100']}/{summary['n']} ({summary['top100_pct']*100:.1f}%)")
    log(f"  mean_rank: {summary['mean_rank']:.1f} (CI95 {summary['mean_rank_ci95']})")

    out = {
        "phase": "14.Q9-1",
        "method": f"LOPO BoK-152 FILTERED (non-ASCII > {NON_ASCII_THRESHOLD*100:.0f}% dropped)",
        "n_total_cands_pre_filter": len(candidates),
        "n_total_cands_post_filter": n_kept,
        "n_dropped": n_dropped,
        "drop_by_cond": dict(drop_by_cond),
        "selected_conditions": SELECTED_CONDITIONS,
        "alpha": ALPHA,
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "layer": LAYER_FOCUS,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_q8_lopo_bok152_q4_8subsets": "rank1=18/30, mean_rank=12.0",
            "phase14_q7_lopo_bok120_q4_4subsets": "rank1=18/30 (60.0%), mean_rank=11.9",
            "phase14_p6_lopo_bok88": "rank1=15/30 (50.0%), mean_rank=20.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in fold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 14.Q9-1 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()