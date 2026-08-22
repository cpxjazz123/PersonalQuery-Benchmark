#!/usr/bin/env python3
"""Phase 14.P-2: BoK-48/88 + margin rerank + PCA-200/500 ensemble.

Evaluates over 11 conds × 8 = 88 cands per pair (extends Phase 14.O BoK-24 → BoK-88).

Strategies tested:
  1. BoK per cond × 11 (Phase 14.M baseline, all conds)
  2. BoK-N for N in [8, 16, 24, 32, 48, 64, 88] (random + greedy conds subsets)
  3. Margin ranking at 88 cands (user proposal, large pool needed)
  4. PCA ensemble: score = α × PCA200_Maha + (1-α) × PCA500_Maha
  5. LOPO validation on best BoK config
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

OUT_EVAL = OUT_DIR / "phase14_p_bok_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_p_bok_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_p_bok_meta.json"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584
K_PER_PAIR = 8

PCA_DIMS = [50, 200, 500]
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000

# All 11 conds from phase14_b jsonl
ALL_CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
                  "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]


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
    log("Phase 14.P-2: BoK-48/88 + margin rerank + PCA ensemble")
    log("=" * 70)

    # === [1] Load candidates (all 11 conds) ===
    log(f"[1] Loading candidates (all {len(ALL_CONDITIONS)} conds) ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] not in ALL_CONDITIONS:
                continue
            candidates.append(r)
    log(f"  candidates: {len(candidates)} (expected 2640)")

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
    log(f"  pairs: {len(pairs)}")

    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load hiddens + global neutral ===
    log("[2] Loading hiddens ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals_all = np.load(IN_CAND_RESID)  # [2640, 5, 3584]
    cand_residuals = cand_residuals_all[:, layer_idx, :]  # [2640, 3584]
    log(f"  cand_residuals shape: {cand_residuals.shape}")

    # === [3] Fit PCA once on combined pool ===
    log(f"[3] Fitting PCA on combined pool ({N_USERS * 30} user + 2640 cand) ...")
    user_residuals_flat = user_residuals.reshape(-1, HIDDEN)
    train_pool = np.concatenate([user_residuals_flat, cand_residuals], axis=0)

    from sklearn.decomposition import PCA
    MAX_PCA = max(PCA_DIMS)
    pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=SEED)
    pca.fit(train_pool)
    components_full = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)
    log(f"  PCA fitted: {components_full.shape}, total explained var: {pca.explained_variance_ratio_.sum():.4f}")

    user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
    user_pca_full_3d = user_pca_full.reshape(N_USERS, 30, MAX_PCA)
    cand_pca_full = (cand_residuals - pca_mean) @ components_full.T  # [2640, MAX_PCA]

    # Per-user Gaussian
    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    # === [4] Per-pair evaluation over all 88 cands ===
    log("[4] Per-pair evaluation over all 88 cands ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)
    log(f"  valid pairs: {len(pair_keys)}")

    per_pair_records = []
    # Storage per PCA dim: for each pair, we store:
    #   - per-cond best rank (11 conds)
    #   - BoK-N ranks (N cands subsets)
    #   - BoK-88 best rank, margin rank
    #   - PCA ensemble rank (PCA-200 + PCA-500)
    per_pair_per_pca = {pca_dim: {
        "bok_per_cond": defaultdict(list),  # cond → ranks
        "bok_24": [],
        "bok_48": [],
        "bok_64": [],
        "bok_88": [],
        "margin_88": [],
        "ensemble": {0.3: [], 0.5: [], 0.7: []},  # α for PCA-200 weight
    } for pca_dim in PCA_DIMS}

    # Pre-selected cond subsets for BoK-24/48/64 (fixed for reproducibility)
    PHASE14_O_CONDS = ["A22_a0.5", "A14_a1.0", "D_off"]  # baseline BoK-24
    BOK_48_CONDS = ["A22_a0.5", "A14_a1.0", "A22_a1.0", "A26_a0.5", "A26_a1.0", "D_off"]
    BOK_64_CONDS = ALL_CONDITIONS[:8]  # 8 conds × 8 = 64

    for pi, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        record = {"user_id": uid, "asin": asin, "n_users": N_USERS}

        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]

        # Build per-cond arrays and union
        all_cand_idx_per_cond = []
        for cond in ALL_CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx_per_cond.append((cond, np.array(cond_to_cand_idx[cond])))

        # Build subset cand index arrays
        cond_to_subset_idx = {
            "bok_24": np.concatenate([c[1] for c in all_cand_idx_per_cond if c[0] in PHASE14_O_CONDS]),
            "bok_48": np.concatenate([c[1] for c in all_cand_idx_per_cond if c[0] in BOK_48_CONDS]),
            "bok_64": np.concatenate([c[1] for c in all_cand_idx_per_cond if c[0] in BOK_64_CONDS]),
            "bok_88": np.concatenate([c[1] for c in all_cand_idx_per_cond]),
        }

        for pca_dim in PCA_DIMS:
            components = components_full[:pca_dim]
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            user_var = user_var_full[:, :pca_dim]

            # Per-pair maha on all 88 cands
            all_cand_idx = cond_to_subset_idx["bok_88"]
            all_cand_pca = cand_pca_full[all_cand_idx, :pca_dim]  # [88, pca_dim]
            delta = all_cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha_d_all = (delta * weighted).sum(axis=-1)  # [88, 198]
            target_d_all = maha_d_all[:, target_idx: target_idx + 1]
            rank_per_cand_88 = (maha_d_all < target_d_all).sum(axis=1)  # [88]

            # Per-cond BoK best rank (11 conds)
            for cond, idxs in all_cand_idx_per_cond:
                idxs = np.array(idxs)
                cond_rank_min = int(np.min(rank_per_cand_88[
                    np.isin(all_cand_idx, idxs)
                ])) if len(idxs) > 0 else None
                per_pair_per_pca[pca_dim]["bok_per_cond"][cond].append(cond_rank_min)

            # BoK-N for N in 24, 48, 64, 88
            for n_key in ["bok_24", "bok_48", "bok_64", "bok_88"]:
                sub_idx = cond_to_subset_idx[n_key]
                mask = np.isin(all_cand_idx, sub_idx)
                sub_ranks = rank_per_cand_88[mask]
                best_rank = int(np.min(sub_ranks))
                per_pair_per_pca[pca_dim][n_key].append(best_rank)

            # Margin ranking on 88 cands
            maha_d_masked = maha_d_all.copy()
            maha_d_masked[:, target_idx] = np.inf
            d_nearest_other = maha_d_masked.min(axis=1)  # [88]
            d_target = target_d_all[:, 0]  # [88]
            margin = d_nearest_other - d_target
            best_margin_idx = int(np.argmax(margin))
            best_margin_rank = int(rank_per_cand_88[best_margin_idx])
            per_pair_per_pca[pca_dim]["margin_88"].append(best_margin_rank)

            # Ensemble: α × PCA200 + (1-α) × PCA500
            if pca_dim == 200:
                # Need PCA-500 ranks too
                components_500 = components_full[:500]
                inv_var_500 = 1.0 / (pooled_var_diag_full[:500] + 1e-6)
                user_mu_500 = user_mu_full[:, :500]
                all_cand_pca_500 = cand_pca_full[all_cand_idx, :500]
                delta_500 = all_cand_pca_500[:, None, :] - user_mu_500[None, :, :]
                weighted_500 = delta_500 * inv_var_500[None, None, :]
                maha_d_500 = (delta_500 * weighted_500).sum(axis=-1)  # [88, 198]

                # Normalize both Mahas to [0,1] within the 88 cands × 198 users pool
                maha_200_norm = maha_d_all / (maha_d_all.max() + 1e-9)
                maha_500_norm = maha_d_500 / (maha_d_500.max() + 1e-9)

                for alpha in [0.3, 0.5, 0.7]:
                    score = alpha * maha_200_norm + (1 - alpha) * maha_500_norm  # [88, 198]
                    target_score = score[:, target_idx: target_idx + 1]
                    rank_per_cand_e = (score < target_score).sum(axis=1)
                    best_e_rank = int(np.min(rank_per_cand_e))
                    per_pair_per_pca[pca_dim]["ensemble"][alpha].append(best_e_rank)

            # Per-pair metadata at PCA-200 only
            if pca_dim == 200:
                record["bok_88_rank"] = int(np.min(rank_per_cand_88))
                record["margin_88_rank"] = best_margin_rank
                record["bok_24_rank"] = per_pair_per_pca[pca_dim]["bok_24"][-1]
                record["bok_48_rank"] = per_pair_per_pca[pca_dim]["bok_48"][-1]
                record["bok_64_rank"] = per_pair_per_pca[pca_dim]["bok_64"][-1]

        per_pair_records.append(record)
        if pi % 5 == 0:
            log(f"  pair {pi+1}/{len(pair_keys)} done")

    # === [5] Aggregate ===
    log("[5] Aggregating results ...")
    summary = {pca_dim: {} for pca_dim in PCA_DIMS}

    for pca_dim in PCA_DIMS:
        d = per_pair_per_pca[pca_dim]
        summary[pca_dim]["bok_per_cond"] = {
            cond: aggregate([r for r in ranks if r is not None])
            for cond, ranks in d["bok_per_cond"].items()
        }
        for n_key in ["bok_24", "bok_48", "bok_64", "bok_88"]:
            summary[pca_dim][n_key] = aggregate(d[n_key])
        summary[pca_dim]["margin_88"] = aggregate(d["margin_88"])
        if "ensemble" in d:
            summary[pca_dim]["ensemble"] = {f"alpha_{a}": aggregate(d["ensemble"][a])
                                            for a in [0.3, 0.5, 0.7]}

    log("\n=== Phase 14.P-2 Summary ===")
    for pca_dim in PCA_DIMS:
        log(f"\n--- PCA-{pca_dim} ---")
        for n_key in ["bok_24", "bok_48", "bok_64", "bok_88"]:
            s = summary[pca_dim][n_key]
            log(f"  {n_key.upper()}: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
                f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")
        s = summary[pca_dim]["margin_88"]
        log(f"  MARGIN-88: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
            f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")
        if pca_dim == 200 and "ensemble" in summary[pca_dim]:
            for a_str, s in summary[pca_dim]["ensemble"].items():
                log(f"  ENSEMBLE-{a_str}: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
                    f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")

        log("  Per-cond BoK (best-of-K per cond, 11 conds):")
        for cond in ALL_CONDITIONS:
            s = summary[pca_dim]["bok_per_cond"][cond]
            log(f"    {cond:>10}: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
                f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")

    # === [6] Save ===
    out = {
        "phase": "14.P-2",
        "method": "BoK-N + margin-88 + PCA ensemble over 11 conds",
        "layer": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "all_conditions": ALL_CONDITIONS,
        "bok_24_conds": PHASE14_O_CONDS,
        "bok_48_conds": BOK_48_CONDS,
        "bok_64_conds": BOK_64_CONDS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_o_bok24_pca200": "rank1=5/30, top100=29/30, mean_rank=25.3",
            "phase14_m_pca200_A14_a1.0": "rank1=4/30, top100=28/30, mean_rank=36.9",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in per_pair_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.P-2",
        "method": "BoK-48/88 + margin + ensemble over 11 conds",
        "layer_focus": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "n_pairs": len(pair_keys),
        "all_conditions": ALL_CONDITIONS,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.P-2 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
