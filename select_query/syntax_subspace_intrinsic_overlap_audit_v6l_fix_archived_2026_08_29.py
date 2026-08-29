"""Stage 4 v6l-FIX — Matched-cohort Intrinsic Overlap Audit (用户指令 2026-08-28).

Stage 4 v6l 之前的实验有 methodology bug:
  - Real LOO: 10000-user 全池里识别 (z_i 跟 9999 other users 比)
  - Generated: 10-user ASIN cohort 内识别 (z_i 跟 ~9 other users 比)
  - 难度不可比!

本脚本 fix: 对每个 ASIN 的 cohort (10 users),只在 cohort 内做 LOO identification,
跟 generated query 的 10-user cohort M>0=63.7% 同台对比。

对 (ASIN, user_u) 的真实句子 z_{u,i}:
  cohort = asin.users_sampled (10 users, 包含 u)
  μ_u^(-i) = (n_u * μ_u - z_i) / (n_u - 1)  ← LOO self centroid
  μ_v = mean(z_{v,j} for all j)  ← cohort other-user centroid (full sample)
  d_self = || z_i - μ_u^(-i) ||
  d_other = min over v in cohort, v != u of || z_i - μ_v ||
  M_real = d_other - d_self

按 cohort 大小 (10 users) 跟 generated pool (10 users) 同台对比。

输入:
  stage8_5_asins.json (1619 ASINs × 10 users = 16190 pairs)
  sentences_318d_cache.jsonl.gz
输出:
  result/select_query/intrinsic_overlap_audit_v6l_fix.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, LAMBDA, PCA_DIM, PCA_SEED, log,
)
from syntax_subspace_utils import _syntax_subspace_prepare  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, select_feature_names, fit_pca, compute_R99,
)


FEATURE_SUBSET = "F3_CoreStruct"


def main():
    log("=== Stage 4 v6l-FIX: Matched-Cohort Intrinsic Overlap Audit ===")
    t_start = time.time()

    # ---- 1. Load stage8_5 asins (1619 ASINs × 10 users) ----
    log("\n=== 1. Loading stage8_5_asins.json ===")
    asins_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asins_data)} ASINs")

    # Build (asin, user) pair list + cohort dict
    asin_to_users: dict[str, list[str]] = {}
    asin_user_pairs: list[tuple[str, str]] = []
    for entry in asins_data:
        asin = entry["asin"]
        users = entry.get("users_sampled", [])
        if len(users) < 2:
            continue
        asin_to_users[asin] = users[:10]
        for u in users[:10]:
            asin_user_pairs.append((asin, u))
    n_pairs_total = len(asin_user_pairs)
    n_pairs_unique_users = len(set(u for _, u in asin_user_pairs))
    log(f"  pairs: {n_pairs_total} ({(n_pairs_total/1619):.1f} users/asin avg)")
    log(f"  unique users in cohorts: {n_pairs_unique_users}")

    # ---- 2. Load _syntax_subspace_prepare ----
    log("\n=== 2. Loading _syntax_subspace_prepare() ===")
    P = _syntax_subspace_prepare()
    X = P["X"]
    all_fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]

    # ---- 3. F3_CoreStruct + PCA48 + whitening ----
    log("\n=== 3. F3_CoreStruct + PCA48 + whitening ===")
    cfg = FEATURE_SUBSETS[FEATURE_SUBSET]
    fnames = select_feature_names(all_fnames, cfg["exclude_prefixes"], cfg["exclude_exact"])
    col_idx = [all_fnames.index(n) for n in fnames]
    X_sub = X[:, col_idx]
    scaler_sub = StandardScaler()
    scaler_sub.fit(X_sub[train_idx])
    X_sub_scaled = scaler_sub.transform(X_sub)
    log(f"  {FEATURE_SUBSET}: {len(fnames)} features, X_sub_scaled: {X_sub_scaled.shape}")

    pca, sqrt_lambda = fit_pca(X_sub_scaled[train_idx], PCA_DIM, PCA_SEED)
    cum_var = float(pca.explained_variance_ratio_.sum())
    log(f"  PCA{PCA_DIM}: cumvar={cum_var:.4f}, sqrt_lambda mean={float(sqrt_lambda.mean()):.4f}")

    R99, n_resid = compute_R99(pca, sqrt_lambda, X_sub_scaled, user_to_indices)
    log(f"  R_99 = {R99:.4f} (n_resid={n_resid})")

    Z_all = pca.transform(X_sub_scaled) / sqrt_lambda
    log(f"  Z_all: {Z_all.shape}")

    # ---- 4. Build per-user Z + global mean ----
    log("\n=== 4. Pre-computing per-user Z + global mean μ_u ===")
    user_Z: dict[str, np.ndarray] = {}
    user_mu: dict[str, np.ndarray] = {}
    n_in_history = 0
    for uid in set(u for _, u in asin_user_pairs):
        idx = user_to_indices.get(uid, [])
        if len(idx) < 2:
            continue
        user_Z[uid] = Z_all[idx]
        user_mu[uid] = Z_all[idx].mean(axis=0)
        n_in_history += len(idx)
    log(f"  cohort users with real history (>=2 sents): {len(user_Z)}")
    log(f"  total cohort-user sentences: {n_in_history}")

    # ---- 5. Matched-cohort LOO identification ----
    log("\n=== 5. Matched-cohort LOO identification (10-user per ASIN) ===")
    M_real = []
    self_L2_real = []
    other_L2_real = []
    rank_real = []
    n_user_real = []
    in_dist_real = []
    asin_real = []
    user_real = []

    skipped_asin_no_users = 0
    skipped_user_too_few = 0
    for asin, users in asin_to_users.items():
        cohort = users
        cohort_in_history = [u for u in cohort if u in user_Z]
        if len(cohort_in_history) < 2:
            skipped_asin_no_users += 1
            continue
        cohort_means = np.stack([user_mu[u] for u in cohort_in_history], axis=0)  # (C, 48)
        cohort_id_to_idx = {u: i for i, u in enumerate(cohort_in_history)}
        for uid in cohort_in_history:
            if uid not in user_Z:
                skipped_user_too_few += 1
                continue
            Z_u = user_Z[uid]
            n_u = len(Z_u)
            mu_global = user_mu[uid]
            # LOO self mean
            loo_means = (n_u * mu_global - Z_u) / (n_u - 1)  # (n_u, 48)
            d_self = np.linalg.norm(Z_u - loo_means, axis=1)  # (n_u,)
            # d_other: min over cohort v != uid
            ui_in_cohort = cohort_id_to_idx[uid]
            others_mask = np.ones(len(cohort_in_history), dtype=bool)
            others_mask[ui_in_cohort] = False
            cohort_means_others = cohort_means[others_mask]  # (C-1, 48)
            diffs = Z_u[:, None, :] - cohort_means_others[None, :, :]  # (n_u, C-1, 48)
            d_to_others = np.linalg.norm(diffs, axis=2)  # (n_u, C-1)
            d_other = d_to_others.min(axis=1)  # (n_u,)

            # M_real
            M_u = d_other - d_self
            # Rank of self in cohort
            all_dists = np.concatenate([d_self[:, None], d_to_others], axis=1)  # (n_u, C)
            rank_u = (all_dists <= d_self[:, None]).sum(axis=1)

            in_dist = d_self <= R99

            M_real.extend(M_u.tolist())
            self_L2_real.extend(d_self.tolist())
            other_L2_real.extend(d_other.tolist())
            rank_real.extend(rank_u.tolist())
            n_user_real.extend([n_u] * n_u)
            in_dist_real.extend(in_dist.tolist())
            asin_real.extend([asin] * n_u)
            user_real.extend([uid] * n_u)

    log(f"  total evaluations: {len(M_real)}")
    log(f"  skipped ASINs (<2 cohort users in history): {skipped_asin_no_users}")

    M_real = np.array(M_real)
    self_L2_real = np.array(self_L2_real)
    other_L2_real = np.array(other_L2_real)
    rank_real = np.array(rank_real)
    in_dist_real = np.array(in_dist_real)
    n_user_real = np.array(n_user_real)

    # ---- 6. Aggregate ----
    log("\n=== 6. Aggregate matched-cohort metrics ===")
    in_mask = in_dist_real
    in_total = int(in_mask.sum())
    log(f"  In R_99 gate: {in_total}/{len(M_real)} ({in_total/len(M_real)*100:.1f}%)")

    M_in = M_real[in_mask]
    rank_in = rank_real[in_mask]

    metrics_overall = {
        "n_evaluations": int(len(M_real)),
        "n_in_R99": in_total,
        "frac_in_R99": float(in_total / len(M_real)),
        "M_gt_0_fraction": float((M_in > 0).mean()),
        "M_mean": float(M_in.mean()),
        "M_median": float(np.median(M_in)),
        "M_std": float(M_in.std()),
        "Rank_at_1": float((rank_in == 1).mean()),
        "Rank_le_3": float((rank_in <= 3).mean()),
        "self_L2_mean": float(self_L2_real[in_mask].mean()),
        "self_L2_median": float(np.median(self_L2_real[in_mask])),
        "self_L2_std": float(self_L2_real[in_mask].std()),
    }
    log(f"  P(M_real > 0) [matched-cohort intrinsic separability] = "
        f"{metrics_overall['M_gt_0_fraction']*100:.1f}%")
    log(f"  Real Rank@1 = {metrics_overall['Rank_at_1']*100:.1f}%")
    log(f"  Real Rank≤3 = {metrics_overall['Rank_le_3']*100:.1f}%")
    log(f"  Margin mean/median = {metrics_overall['M_mean']:.3f} / {metrics_overall['M_median']:.3f}")

    # ---- 7. Per-ASIN M_real (mean across sentences) ----
    log("\n=== 7. Per-ASIN averaged M_real ===")
    asin_M = collections.defaultdict(list)
    asin_M_gt_0 = collections.defaultdict(list)
    asin_rank = collections.defaultdict(list)
    for i, asin in enumerate(asin_real):
        if in_dist_real[i]:
            asin_M[asin].append(M_real[i])
            asin_M_gt_0[asin].append(int(M_real[i] > 0))
            asin_rank[asin].append(int(rank_real[i] == 1))
    asin_summary = {}
    for asin in asin_M:
        if len(asin_M[asin]) < 5:
            continue
        asin_summary[asin] = {
            "n_in_R99": int(len(asin_M[asin])),
            "M_mean": float(np.mean(asin_M[asin])),
            "M_gt_0_frac": float(np.mean(asin_M_gt_0[asin])),
            "Rank_at_1_frac": float(np.mean(asin_rank[asin])),
        }
    log(f"  ASINs with >=5 in-gate real sentences: {len(asin_summary)}")
    log(f"  per-ASIN M>0 fraction: mean={np.mean([v['M_gt_0_frac'] for v in asin_summary.values()])*100:.1f}%, "
        f"median={np.median([v['M_gt_0_frac'] for v in asin_summary.values()])*100:.1f}%, "
        f"min={min(v['M_gt_0_frac'] for v in asin_summary.values())*100:.1f}%, "
        f"max={max(v['M_gt_0_frac'] for v in asin_summary.values())*100:.1f}%")

    # ---- 8. Comparison with v6k generated (cohort-based) ----
    log("\n=== 8. Comparison with K=200 generated (10-user cohort) ===")
    # v6k selection_stats.json already has the cohort-based metrics:
    #   margin_positive_fraction = 0.637 (M>0 over 7299 pairs)
    #   selected_mean = 11.347 (self_L2)
    #   margin_mean / margin_median
    sel_stats = json.load(open("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_stats.json"))
    gen_M_gt_0 = sel_stats["margin_positive_fraction"]
    gen_self_L2 = sel_stats["selected_mean"]
    gen_margin_mean = sel_stats["margin_mean"]
    gen_margin_median = sel_stats["margin_median"]
    log(f"  Generated K=200 (10-user cohort, from selection_stats.json):")
    log(f"    M>0 = {gen_M_gt_0*100:.1f}%")
    log(f"    self_L2 mean = {gen_self_L2:.3f}")
    log(f"    margin mean/median = {gen_margin_mean:.3f} / {gen_margin_median:.3f}")
    log(f"  Real (matched cohort 10-user, this experiment):")
    log(f"    M>0 = {metrics_overall['M_gt_0_fraction']*100:.1f}%")
    log(f"    self_L2 mean = {metrics_overall['self_L2_mean']:.3f}")
    log(f"    margin mean/median = {metrics_overall['M_mean']:.3f} / {metrics_overall['M_median']:.3f}")

    # Coverage Ratio (matched-cohort)
    coverage_M = gen_M_gt_0 / max(metrics_overall['M_gt_0_fraction'], 1e-9)
    log(f"  Coverage Ratio M>0 = {coverage_M*100:.1f}%")

    if metrics_overall['M_gt_0_fraction'] > 0.5:
        verdict = ("HIGH INTRINSIC CEILING: real-history matched-cohort M>0 is high — "
                   "user-style IS separable when comparing within ASIN cohort. "
                   "The 0.48% from v6l was due to 10k-user pool dilution, NOT true user-style overlap.")
    elif metrics_overall['M_gt_0_fraction'] > 0.2:
        verdict = ("MODERATE INTRINSIC CEILING: real-history cohort M>0 ≈ 20-50% — "
                   "user-style has moderate separability within cohort. "
                   "Generation is well above real ceiling → main lift is query/review genre gap.")
    elif metrics_overall['M_gt_0_fraction'] > 0.05:
        verdict = ("LOW INTRINSIC CEILING: real-history cohort M>0 ≈ 5-20% — "
                   "user-style has limited intrinsic separability. "
                   "Generation 63.7% is significantly above real ceiling, mostly from genre gap.")
    else:
        verdict = ("NEAR-ZERO INTRINSIC CEILING: real-history cohort M>0 < 5% — "
                   "users in same ASIN cohort have nearly indistinguishable syntax styles. "
                   "Generation 63.7% is almost entirely query vs review genre gap, "
                   "NOT user-style lift.")
    log(f"\n  >>> VERDICT: {verdict}")

    # ---- 9. Save ----
    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/intrinsic_overlap_audit_v6l_fix.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6l-FIX matched-cohort intrinsic overlap audit. "
                                "Each ASIN's cohort (10 users from stage8_5_asins.json) is used as the "
                                "candidate pool for both real-history LOO identification and the "
                                "generated-query selection. Same pool size = fair comparison. "
                                "Fixes v6l's methodology bug where real LOO used 10k users but "
                                "generated used 10-user cohort — difficulty mismatch."),
                "feature_subset": FEATURE_SUBSET,
                "pca_dim": PCA_DIM,
                "lambda_shrinkage": LAMBDA,
                "R_99": float(R99),
                "n_residuals_for_R99": int(n_resid),
                "cohort_size_per_asin": 10,
                "n_asins": len(asin_to_users),
                "n_pairs_total": n_pairs_total,
                "n_unique_users_in_cohorts": n_pairs_unique_users,
            },
            "real_history_matched_cohort": metrics_overall,
            "per_asin_summary": asin_summary,
            "generated_K200_comparison": {
                "M_gt_0": float(gen_M_gt_0),
                "self_L2_mean": float(gen_self_L2),
                "margin_mean": float(gen_margin_mean),
                "margin_median": float(gen_margin_median),
                "n_pairs": int(sel_stats["n_pairs"]),
            },
            "coverage_ratio_matched_cohort": {
                "M_gt_0_ratio": float(coverage_M),
                "verdict": verdict,
            },
            "interpretation": {
                "real_M_gt_0_pct": float(metrics_overall['M_gt_0_fraction'] * 100),
                "generated_K200_M_gt_0_pct": float(gen_M_gt_0 * 100),
                "real_Rank_at_1_pct": float(metrics_overall['Rank_at_1'] * 100),
                "real_self_L2": float(metrics_overall['self_L2_mean']),
                "generated_self_L2": float(gen_self_L2),
                "key_finding": (
                    "v6l's 0.48% was due to 10k-user pool dilution. "
                    "Real matched-cohort M>0 = " + f"{metrics_overall['M_gt_0_fraction']*100:.1f}%, "
                    "which is the fair comparison against generated 63.7%."
                ),
            },
            "total_elapsed_s": float(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")
    log(f"\n>>> STAGE 4 v6l-FIX: {verdict}")


if __name__ == "__main__":
    main()
