"""Stage 4 v6l — Intrinsic User Syntax Overlap Audit (用户指令 2026-08-28).

完全不使用生成的 candidate query,直接对真实用户历史评论做 leave-one-out identification。

对用户 u 的第 i 个真实句子 z_{u,i}:
  μ_u^(-i) = mean of z_{u,j} for j != i (LOO self centroid)
  μ_v = mean of z_{v,j} for user v != u (other-user centroid)
  d_self = || z_{u,i} - μ_u^(-i) ||
  d_other = min_{v != u} || z_{u,i} - μ_v ||
  M_real = d_other - d_self
  R_99 gate: 排除 whitened L2 > R_99 (与 main pipeline 同)

统计:
  - P(M_real > 0) = intrinsic real-history Rank@1 (= ceiling of generation M>0)
  - Rank@1 / Rank≤3
  - Margin mean/median/std
  - Per-user 平均 M_real
  - History-size stratification: n_u ∈ [35,50), [50,100), [100+]
  - Coverage Ratio = P(M_query>0) / P(M_real>0) — 量化生成 query 距离真实可分上限多远

输入:
  common/syntax_subspace_utils._syntax_subspace_prepare (X 318d)
  select_query/syntax_subspace_repr_sweep.fit_pca, compute_R99
输出:
  result/select_query/intrinsic_overlap_audit.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    LAMBDA, PCA_DIM, PCA_SEED, log,
)
from syntax_subspace_utils import _syntax_subspace_prepare  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, select_feature_names, fit_pca, compute_R99,
)


FEATURE_SUBSET = "F3_CoreStruct"
# Query vs real gap reference (用户指令: K=200 generated pool)
GENERATED_M_GT_0_REF = 0.637  # K=200, F3+PCA48, sim09 from volatility.json
GENERATED_RANK_AT_1_REF = 0.297  # back-computed from prior selection stats


def main():
    log("=== Stage 4 v6l: Intrinsic User Syntax Overlap Audit ===")
    t_start = time.time()

    # ---- 1. Load _syntax_subspace_prepare ----
    log("\n=== 1. Loading _syntax_subspace_prepare() ===")
    P = _syntax_subspace_prepare()
    X = P["X"]  # raw 318d (will refit StandardScaler on F3 subset)
    all_fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    log(f"  X: {X.shape}, users: {len(user_to_indices)}")

    # ---- 2. F3_CoreStruct subset + StandardScaler + PCA48 + whitening ----
    log("\n=== 2. Building F3_CoreStruct + PCA48 + whitening ===")
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

    # ---- 3. Compute R_99 from real history ----
    log("\n=== 3. Computing R_99 from real history (whitened L2 99th percentile) ===")
    R99, n_resid = compute_R99(pca, sqrt_lambda, X_sub_scaled, user_to_indices)
    log(f"  R_99 = {R99:.4f} (n_resid={n_resid})")

    # ---- 4. Project all real history to whitened 48d ----
    log("\n=== 4. Projecting all real history to whitened 48d ===")
    Z_all = pca.transform(X_sub_scaled) / sqrt_lambda  # (N, 48)
    log(f"  Z_all: {Z_all.shape}")

    # ---- 5. Filter: users with >= 2 sentences (need min 2 for LOO identification) ----
    log("\n=== 5. Filtering users (n_u >= 2) ===")
    eligible_users = []
    for uid in user_id_list:
        idx = user_to_indices[uid]
        if len(idx) >= 2:
            eligible_users.append((uid, idx))
    log(f"  eligible users (n_u >= 2): {len(eligible_users)}")
    log(f"  users with n_u >= 35 (Stage 8.5 asin-users_min): "
        f"{sum(1 for u, idx in eligible_users if len(idx) >= 35)}")

    # ---- 6. Compute per-user global mean μ_u (used for v != u lookup) ----
    log("\n=== 6. Pre-computing per-user global mean μ_u ===")
    user_means = {}
    user_Z = {}
    for uid, idx in eligible_users:
        Z_u = Z_all[idx]
        user_Z[uid] = Z_u
        user_means[uid] = Z_u.mean(axis=0)
    user_ids = [uid for uid, _ in eligible_users]
    user_means_arr = np.stack([user_means[u] for u in user_ids], axis=0)  # (U, 48)
    log(f"  user_means_arr: {user_means_arr.shape}")

    # ---- 7. LOO identification: M_real per real sentence ----
    log("\n=== 7. LOO identification (vectorized per-user) ===")
    log(f"  Computing μ_u^(-i) by leaving out z_{{u,i}}...")
    log(f"  Total real sentences to evaluate: {sum(len(idx) for _, idx in eligible_users)}")

    M_real = []           # margin per (u, i)
    self_L2_real = []     # L2 to LOO self centroid per (u, i)
    other_L2_real = []    # min L2 to other-user centroid per (u, i)
    rank_real = []        # rank of self in [self + all others] (1=closest is self)
    n_user = []           # n_u for stratification
    user_id_record = []
    in_dist_real = []     # True if self_L2 <= R_99 (in gate region)

    for ui, (uid, idx) in enumerate(eligible_users):
        n_u = len(idx)
        Z_u = user_Z[uid]  # (n_u, 48)
        # LOO self mean: leave out z_i → μ_u^(-i) = (n_u * μ_u - z_i) / (n_u - 1)
        mu_global = user_means[uid]
        # LOO means for user u: (n_u * mu_global - Z_u) / (n_u - 1)
        loo_means = (n_u * mu_global - Z_u) / (n_u - 1)  # (n_u, 48)
        # self_L2: || z_i - μ_u^(-i) ||
        d_self = np.linalg.norm(Z_u - loo_means, axis=1)  # (n_u,)

        # other_L2: for each z_i, compute min over v != u of || z_i - μ_v ||
        # = || z_i[None, :] - user_means_arr[others] || axis=1 .min()
        others_mask = np.ones(len(user_ids), dtype=bool)
        others_mask[ui] = False
        diffs = Z_u[:, None, :] - user_means_arr[others_mask][None, :, :]  # (n_u, U-1, 48)
        d_to_others = np.linalg.norm(diffs, axis=2)  # (n_u, U-1)
        d_other = d_to_others.min(axis=1)  # (n_u,)

        # M_real
        M_u = d_other - d_self  # (n_u,)

        # Rank of self: sort all distances (self + others), find rank of self
        all_dists = np.concatenate([d_self[:, None], d_to_others], axis=1)  # (n_u, U)
        rank_u = (all_dists <= d_self[:, None]).sum(axis=1)  # rank of self (1 = self is min)

        in_dist = d_self <= R99

        M_real.extend(M_u.tolist())
        self_L2_real.extend(d_self.tolist())
        other_L2_real.extend(d_other.tolist())
        rank_real.extend(rank_u.tolist())
        n_user.extend([n_u] * n_u)
        user_id_record.extend([uid] * n_u)
        in_dist_real.extend(in_dist.tolist())

    M_real = np.array(M_real)
    self_L2_real = np.array(self_L2_real)
    other_L2_real = np.array(other_L2_real)
    rank_real = np.array(rank_real)
    n_user = np.array(n_user)
    in_dist_real = np.array(in_dist_real)
    log(f"  total evaluations: {len(M_real)}")

    # ---- 8. Aggregate metrics ----
    log("\n=== 8. Aggregate real-history metrics ===")
    in_mask = in_dist_real
    in_total = int(in_mask.sum())
    log(f"  In R_99 gate: {in_total}/{len(M_real)} ({in_total/len(M_real)*100:.1f}%)")

    M_in = M_real[in_mask]
    rank_in = rank_real[in_mask]
    self_L2_in = self_L2_real[in_mask]

    metrics_overall = {
        "n_evaluations": int(len(M_real)),
        "n_in_R99": in_total,
        "frac_in_R99": float(in_total / len(M_real)),
        "M_gt_0_fraction": float((M_in > 0).mean()),
        "M_mean": float(M_in.mean()),
        "M_median": float(np.median(M_in)),
        "M_std": float(M_in.std()),
        "M_min": float(M_in.min()),
        "M_max": float(M_in.max()),
        "Rank_at_1": float((rank_in == 1).mean()),
        "Rank_le_3": float((rank_in <= 3).mean()),
        "Rank_at_1_loose": float((rank_in <= 2).mean()),
        "self_L2_mean": float(self_L2_in.mean()),
        "self_L2_median": float(np.median(self_L2_in)),
        "self_L2_std": float(self_L2_in.std()),
    }
    log(f"  P(M_real > 0) [Intrinsic separability] = {metrics_overall['M_gt_0_fraction']*100:.1f}%")
    log(f"  Real Rank@1 = {metrics_overall['Rank_at_1']*100:.1f}%")
    log(f"  Real Rank≤3 = {metrics_overall['Rank_le_3']*100:.1f}%")
    log(f"  Margin mean/median = {metrics_overall['M_mean']:.3f} / {metrics_overall['M_median']:.3f}")

    # ---- 9. History-size stratification ----
    log("\n=== 9. History-size stratification ===")
    strata = {
        "[2, 10)":   (n_user >= 2) & (n_user < 10),
        "[10, 35)":  (n_user >= 10) & (n_user < 35),
        "[35, 50)":  (n_user >= 35) & (n_user < 50),
        "[50, 100)": (n_user >= 50) & (n_user < 100),
        "[100, ∞)":  (n_user >= 100),
    }
    strata_metrics = {}
    for name, mask in strata.items():
        n_total_stratum = int(mask.sum())
        mask_in = mask & in_mask
        n_in_stratum = int(mask_in.sum())
        if n_in_stratum < 10:
            log(f"  {name}: skipped (n_in={n_in_stratum})")
            continue
        M_stratum = M_real[mask_in]
        rank_stratum = rank_real[mask_in]
        strata_metrics[name] = {
            "n_total": n_total_stratum,
            "n_in_R99": n_in_stratum,
            "M_gt_0_fraction": float((M_stratum > 0).mean()),
            "Rank_at_1": float((rank_stratum == 1).mean()),
            "Rank_le_3": float((rank_stratum <= 3).mean()),
            "M_mean": float(M_stratum.mean()),
            "M_median": float(np.median(M_stratum)),
            "self_L2_mean": float(self_L2_real[mask_in].mean()),
        }
        log(f"  {name}: N_in={n_in_stratum}, M>0={strata_metrics[name]['M_gt_0_fraction']*100:.1f}%, "
            f"Rank@1={strata_metrics[name]['Rank_at_1']*100:.1f}%, "
            f"M_median={strata_metrics[name]['M_median']:.3f}")

    # ---- 10. Coverage Ratio ----
    log("\n=== 10. Coverage Ratio (generated vs real) ===")
    coverage_ratio = GENERATED_M_GT_0_REF / metrics_overall['M_gt_0_fraction']
    rank_coverage = GENERATED_RANK_AT_1_REF / metrics_overall['Rank_at_1']
    log(f"  Generated (K=200) M>0 = {GENERATED_M_GT_0_REF*100:.1f}%")
    log(f"  Real-history M>0 = {metrics_overall['M_gt_0_fraction']*100:.1f}%")
    log(f"  Coverage Ratio (M>0) = {coverage_ratio*100:.1f}%  (close to 100% = generation reaches real ceiling)")
    log(f"  Generated (K=200) Rank@1 = {GENERATED_RANK_AT_1_REF*100:.1f}%")
    log(f"  Real-history Rank@1 = {metrics_overall['Rank_at_1']*100:.1f}%")
    log(f"  Coverage Ratio (Rank@1) = {rank_coverage*100:.1f}%")

    if coverage_ratio >= 0.95:
        verdict = ("SATURATION: Generation 已经达到真实可分上限 (≥95%), 剩下 overlap 几乎完全是"
                   "用户真实句法风格重叠, 不应继续调 K, 应该接受这是 user-style 的 intrinsic ceiling")
    elif coverage_ratio >= 0.85:
        verdict = ("HIGH COVERAGE: Generation 接近真实上限 (85-95%), 进一步 candidate generation "
                   "提升空间 <10pp, 主要 overlap 来自真实用户句法天然重叠")
    elif coverage_ratio >= 0.70:
        verdict = ("MODERATE GAP: Generation 仍距真实上限 5-15pp, candidate generation 仍有空间, "
                   "可以考虑更好的 prompt engineering 或更多 N=5+ attrs 注入")
    else:
        verdict = ("LARGE GAP: Generation 距离真实上限 >15pp, 说明当前 generation pipeline 仍存在"
                   "candidate coverage gap, 应该改 generation 策略而非继续 K-sweep")
    log(f"\n  >>> VERDICT: {verdict}")

    # ---- 11. Per-user separability distribution ----
    log("\n=== 11. Per-user average separability ===")
    user_agg = collections.defaultdict(list)
    for i, uid in enumerate(user_id_record):
        if in_dist_real[i]:
            user_agg[uid].append(M_real[i])
    per_user_M = {uid: float(np.mean(ms)) for uid, ms in user_agg.items() if len(ms) >= 5}
    per_user_M_gt_0 = {uid: float(np.mean([m > 0 for m in ms])) for uid, ms in user_agg.items() if len(ms) >= 5}
    log(f"  users with >=5 in-gate real sentences: {len(per_user_M)}")
    log(f"  per-user M>0 fraction: mean={np.mean(list(per_user_M_gt_0.values()))*100:.1f}%, "
        f"median={np.median(list(per_user_M_gt_0.values()))*100:.1f}%, "
        f"min={min(per_user_M_gt_0.values())*100:.1f}%, max={max(per_user_M_gt_0.values())*100:.1f}%")

    # ---- 12. Save ----
    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/intrinsic_overlap_audit.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6l Intrinsic User Syntax Overlap Audit. "
                                "Leave-one-out identification on real user history in F3_CoreStruct+PCA48 "
                                "whitened 48d space. Quantifies the ceiling of user-level syntax "
                                "separability so we can measure how close K=200 generated queries are to "
                                "the real intrinsic ceiling."),
                "feature_subset": FEATURE_SUBSET,
                "pca_dim": PCA_DIM,
                "lambda_shrinkage": LAMBDA,
                "R_99": float(R99),
                "n_residuals_for_R99": int(n_resid),
                "cumulative_variance_explained": cum_var,
                "reference_generated_K200_M_gt_0": GENERATED_M_GT_0_REF,
                "reference_generated_K200_Rank_at_1": GENERATED_RANK_AT_1_REF,
            },
            "overall_metrics": metrics_overall,
            "history_size_strata": strata_metrics,
            "coverage_ratio": {
                "M_gt_0_ratio": float(coverage_ratio),
                "Rank_at_1_ratio": float(rank_coverage),
                "verdict": verdict,
            },
            "per_user_separability_summary": {
                "n_users_with_5plus_in_gate": len(per_user_M),
                "per_user_M_gt_0_fraction_mean": float(np.mean(list(per_user_M_gt_0.values()))),
                "per_user_M_gt_0_fraction_median": float(np.median(list(per_user_M_gt_0.values()))),
                "per_user_M_gt_0_fraction_min": float(min(per_user_M_gt_0.values())),
                "per_user_M_gt_0_fraction_max": float(max(per_user_M_gt_0.values())),
                "per_user_M_mean": float(np.mean(list(per_user_M.values()))),
                "per_user_M_median": float(np.median(list(per_user_M.values()))),
            },
            "interpretation": {
                "intrinsic_M_gt_0_pct": float(metrics_overall["M_gt_0_fraction"] * 100),
                "generated_K200_M_gt_0_pct": float(GENERATED_M_GT_0_REF * 100),
                "intrinsic_Rank_at_1_pct": float(metrics_overall["Rank_at_1"] * 100),
                "generated_K200_Rank_at_1_pct": float(GENERATED_RANK_AT_1_REF * 100),
                "gap_M_gt_0_pp": float((metrics_overall["M_gt_0_fraction"] - GENERATED_M_GT_0_REF) * 100),
                "gap_Rank_at_1_pp": float((metrics_overall["Rank_at_1"] - GENERATED_RANK_AT_1_REF) * 100),
            },
            "total_elapsed_s": float(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")
    log(f"\n>>> STAGE 4 v6l: {verdict}")


if __name__ == "__main__":
    main()
