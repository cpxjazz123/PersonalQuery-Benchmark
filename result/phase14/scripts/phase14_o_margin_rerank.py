#!/usr/bin/env python3
"""Phase 14.O: Margin-based rerank with 24 cands (3 conds × K=8).

User insight: Margin = D_nearest_other_user - D_target.
Selecting by largest margin enforces CLOSENESS to target + DISTANCE from others.

First version: 24 cands/pair (3 conds: A22_a0.5, A14_a1.0, D_off).
If margin shows rank-1 lift, will extend to 88 cands (all 11 conds).
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

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_o_margin_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_o_margin_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_o_margin_rerank_meta.json"

# 3 conds only (cached)
CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584
K_PER_PAIR = 8
N_BOOTSTRAP = 2000

PCA_DIMS = [50, 200, 500]
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42


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


def main() -> None:
    log("=" * 70)
    log("Phase 14.O: Margin-based rerank with 24 cands (3 conds × K=8)")
    log("=" * 70)

    # === [1] Load candidates (3 conds) ===
    log(f"[1] Loading candidates ({len(CONDITIONS)} conds) ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    log(f"  candidates: {len(candidates)}")
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
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals_all = np.load(IN_CAND_RESID)  # [720, 5, 3584]
    cand_residuals = cand_residuals_all[:, layer_idx, :]  # [720, 3584]

    # === [3] Fit PCA once on combined pool ===
    log(f"[3] Fitting PCA on combined pool (5940 user + 720 cand = 6660) ...")
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
    cand_pca_full = (cand_residuals - pca_mean) @ components_full.T  # [720, MAX_PCA]

    # Per-user Gaussian
    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    # === [4] Per-pair evaluation ===
    log(f"[4] Per-pair margin rerank evaluation ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)
    log(f"  valid pairs: {len(pair_keys)}")

    # Methods to evaluate per PCA dim:
    # 1. best-of-K Maha per cond (Phase 14.M baseline)
    # 2. best-of-K Maha overall (24 cands combined)
    # 3. margin ranking overall (24 cands)
    # 4. weighted score = -D_target + λ * margin

    per_pair_records = []
    per_pair_per_pca = {pca_dim: {"bok_per_cond": defaultdict(list), "bok_24": [], "margin_24": [],
                                  "weighted": {0.1: [], 0.5: [], 1.0: [], 2.0: [], 5.0: []}}
                        for pca_dim in PCA_DIMS}

    for (uid, asin) in pair_keys:
        target_idx = uid_to_useridx[uid]
        record = {"user_id": uid, "asin": asin}

        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]
        all_cand_idx = []
        all_cond_label = []
        for cond in CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx.extend(cond_to_cand_idx[cond])
                all_cond_label.extend([cond] * K_PER_PAIR)
        all_cand_idx = np.array(all_cand_idx)
        n_total_cands = len(all_cand_idx)  # 24

        for pca_dim in PCA_DIMS:
            components = components_full[:pca_dim]
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            user_var = user_var_full[:, :pca_dim]

            # All 24 cands per pair
            all_cand_pca = cand_pca_full[all_cand_idx, :pca_dim]  # [24, pca_dim]

            # Maha: [24, 198]
            delta = all_cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha_d_all = (delta * weighted).sum(axis=-1)
            target_d_all = maha_d_all[:, target_idx: target_idx + 1]
            rank_per_cand = (maha_d_all < target_d_all).sum(axis=1)  # [24]

            # Best-of-K overall (24 cands)
            best_rank_24 = int(np.min(rank_per_cand))
            per_pair_per_pca[pca_dim]["bok_24"].append(best_rank_24)

            # Margin: D_nearest_other - D_target
            maha_d_masked = maha_d_all.copy()
            maha_d_masked[:, target_idx] = np.inf
            d_nearest_other = maha_d_masked.min(axis=1)  # [24]
            d_target = target_d_all[:, 0]  # [24]
            margin = d_nearest_other - d_target  # [24]

            # Best-by-margin = largest margin
            best_margin_idx = int(np.argmax(margin))
            best_margin_rank = int(rank_per_cand[best_margin_idx])
            per_pair_per_pca[pca_dim]["margin_24"].append(best_margin_rank)

            # Weighted
            for lambda_margin in [0.1, 0.5, 1.0, 2.0, 5.0]:
                score = -d_target + lambda_margin * margin
                best_score_idx = int(np.argmax(score))
                best_score_rank = int(rank_per_cand[best_score_idx])
                per_pair_per_pca[pca_dim]["weighted"][lambda_margin].append(best_score_rank)

            # Per-cond best-of-K
            for cond_idx, cond in enumerate(CONDITIONS):
                if cond not in cond_to_cand_idx:
                    continue
                local_cand_idx = cond_to_cand_idx[cond]
                local_cand_pca = cand_pca_full[local_cand_idx, :pca_dim]
                delta_c = local_cand_pca[:, None, :] - user_mu[None, :, :]
                weighted_c = delta_c * inv_var[None, None, :]
                maha_d_c = (delta_c * weighted_c).sum(axis=-1)
                target_d_c = maha_d_c[:, target_idx: target_idx + 1]
                rank_per_cand_c = (maha_d_c < target_d_c).sum(axis=1)
                best_rank_c = int(np.min(rank_per_cand_c))
                per_pair_per_pca[pca_dim]["bok_per_cond"][cond].append(best_rank_c)

            if pca_dim == 200:
                record["bok_24_rank"] = best_rank_24
                record["margin_24_rank"] = best_margin_rank
                record["margin_24_value"] = float(margin[best_margin_idx])
                record["margin_24_cond"] = all_cond_label[best_margin_idx]
                for cond in CONDITIONS:
                    record[f"bok_{cond}_rank"] = per_pair_per_pca[pca_dim]["bok_per_cond"][cond][-1]

        per_pair_records.append(record)

    # === [5] Aggregate ===
    log("[5] Aggregating ...")

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

    summary = {pca_dim: {} for pca_dim in PCA_DIMS}

    for pca_dim in PCA_DIMS:
        d = per_pair_per_pca[pca_dim]
        summary[pca_dim]["bok_24"] = aggregate(d["bok_24"])
        summary[pca_dim]["margin_24"] = aggregate(d["margin_24"])
        summary[pca_dim]["weighted"] = {f"lambda_{lam}": aggregate(d["weighted"][lam])
                                        for lam in [0.1, 0.5, 1.0, 2.0, 5.0]}
        summary[pca_dim]["bok_per_cond"] = {cond: aggregate(d["bok_per_cond"][cond])
                                             for cond in CONDITIONS}

    log("\n=== Phase 14.O Summary ===")
    for pca_dim in PCA_DIMS:
        log(f"\n--- PCA-{pca_dim} ---")
        s = summary[pca_dim]["bok_24"]
        log(f"  BoK-24 (no margin): rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
            f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")

        s = summary[pca_dim]["margin_24"]
        log(f"  ★ MARGIN-24: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
            f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%), mean_rank={s['mean_rank']:.1f}")

        for lam_str, s in summary[pca_dim]["weighted"].items():
            log(f"  Weighted {lam_str}: rank1={s['rank1']}/{s['n']} ({s['rank1_pct']*100:.1f}%), "
                f"top100={s['top100']}/{s['n']} ({s['top100_pct']*100:.1f}%)")

    # === [6] Save ===
    out = {
        "phase": "14.O",
        "method": "Margin-based rerank with 24 cands (3 conds × K=8)",
        "layer": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "n_candidates_per_pair": 24,
        "summary": summary,
        "interpretation": (
            "Margin = D_nearest_other_user - D_target. Selecting by margin enforces "
            "CLOSENESS to target + DISTANCE from others."
        ),
        "comparison_phase14_m": {
            "A14_a1.0_pca200_top100": "28/30 (93.3%)",
            "A22_a0.5_pca30_rank1": "4/30 (13.3%)",
            "Phase14_M_LOPO_A22_pca30_rank1": "4/30 (13.3%)",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in per_pair_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.O",
        "method": "Margin-based rerank with 24 cands",
        "layer_focus": LAYER_FOCUS,
        "pca_dims": PCA_DIMS,
        "n_pairs": len(pair_keys),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.O COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()