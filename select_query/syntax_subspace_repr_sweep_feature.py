"""Stage 4 v6k — Stage B: Feature Set × PCA dim 2D ablation (用户指令 2026-08-28).

基于 Stage A 选定 PCA dim ∈ {32, 48, 64} (sweet spot range),
4 Feature Sets (F1-F4) × 3 PCA dims = 12 cells.

每 cell 4 维指标:
  1. User separability: M>0, Rank@1, Rank≤3, unique ratio
  2. Syntax preservation: cumvar, syntax-probe R² on n_tok/n_clause/nest
  3. Content leakage: top-200 ASIN linear probe R²
  4. σ-stability: ρ(M_L2, log|Σ_u|), PCA-seed variation (3 seeds)

Pareto selection rule:
  - User Rank@1 / M>0 ≥ best_config × 95%
  - Leakage 接近最低
  - σ-deconfounded (|ρ|<0.3)
  - Syntax preservation 不明显下降

输入: stage8_5_asins.json + pool + user_gaussians + sentences_318d_cache
输出: result/select_query/repr_sweep_feature.json
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    log, feat_key,
)
from syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, NGRAM_PREFIXES, SEMANTIC_TAGS,
    select_feature_names, project_pool_to_z, compute_user_means,
    compute_R99, v6k_select_one_config, aggregate_metrics,
    asin_leakage_probe, syntax_preservation, fit_pca,
)


PCA_DIMS_STAGE_B = [32, 48, 64]  # sweet spot from Stage A
SEEDS = [42, 123, 2024]


def main():
    log("=== Stage 4 v6k — Stage B: Feature Set × PCA dim 2D ablation ===")
    t_start = time.time()

    log("\n=== 1. Loading _syntax_subspace_prepare() ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    from sklearn.preprocessing import StandardScaler
    P = _syntax_subspace_prepare()
    all_fnames = P["feature_names_ordered"]
    X = P["X"]  # raw features (P['X_scaled'] is for 182d, can't reuse)
    train_idx = P["train_idx"]
    user_to_indices = P["user_to_indices"]
    leak_train = P["leak_train"]
    leak_test = P["leak_test"]
    label_y = P["label_y"]
    Y_probes = P["Y_probes"]
    probe_train_idx = P["probe_train_idx"]
    probe_test_idx = P["probe_test_idx"]
    PROBE_TARGETS = P["PROBE_TARGETS"]
    log(f"  X: {X.shape}, all features: {len(all_fnames)}")

    log("\n=== 2. Loading pool + Gaussians ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. 2D ablation: Feature Set × PCA dim ===")
    log(f"  Feature subsets: {list(FEATURE_SUBSETS.keys())}")
    log(f"  PCA dims: {PCA_DIMS_STAGE_B}")
    log(f"  Seeds per cell: {SEEDS}")

    results_grid = {}
    cell_count = 0
    for subset_name, cfg in FEATURE_SUBSETS.items():
        fnames_kept = select_feature_names(all_fnames, cfg["exclude_prefixes"],
                                           cfg["exclude_exact"])
        col_idx = [all_fnames.index(n) for n in fnames_kept]
        X_sub = X[:, col_idx]
        # Re-fit StandardScaler on subset's training data
        scaler_sub = StandardScaler()
        scaler_sub.fit(X_sub[train_idx])
        X_sub_scaled = scaler_sub.transform(X_sub)
        log(f"\n--- {subset_name}: {len(fnames_kept)} features (from {len(all_fnames)}) ---")

        for d in PCA_DIMS_STAGE_B:
            cell_key = f"{subset_name}__d={d}"
            cell_count += 1
            log(f"\n--- [{cell_count}/12] {cell_key} ---")
            t_cell = time.time()
            seed_results = []
            for seed in SEEDS:
                pca, sqrt_lambda = fit_pca(X_sub_scaled[train_idx], d, seed)
                R99, n_resid = compute_R99(pca, sqrt_lambda, X_sub_scaled, user_to_indices)
                user_z_means = compute_user_means(pca, sqrt_lambda, X_sub_scaled, user_to_indices)
                pool_z_white, miss = project_pool_to_z(pca, sqrt_lambda, scaler_sub,
                                                       fnames_kept, pools, feat_map)
                entries = v6k_select_one_config(pool_z_white, asin_data, users_gauss,
                                                 user_z_means, R99)
                metrics = aggregate_metrics(entries)
                seed_results.append(metrics)

            ks = [k for k in seed_results[0].keys() if seed_results[0][k] is not None]
            avg = {k: float(np.mean([m[k] for m in seed_results if m[k] is not None]))
                   for k in ks}
            std = {k: float(np.std([m[k] for m in seed_results if m[k] is not None]))
                   for k in ks}

            # leakage + syntax (single seed for speed, seed=42)
            pca_leak, sl_leak = fit_pca(X_sub_scaled[train_idx], d, 42)
            cum_var = float(pca_leak.explained_variance_ratio_.sum())
            leak = asin_leakage_probe(pca_leak, sl_leak, X_sub_scaled, leak_train, leak_test,
                                      label_y)
            probe_R2 = syntax_preservation(pca_leak, sl_leak, X_sub_scaled, Y_probes,
                                          probe_train_idx, probe_test_idx, PROBE_TARGETS)
            # aggregate syntax probe R²
            probe_R2_mean = float(np.mean([v for v in probe_R2.values() if v is not None]))

            results_grid[cell_key] = {
                "feature_subset": subset_name,
                "n_features": len(fnames_kept),
                "pca_dim": d,
                "R99": float(R99),
                "cum_explained_var": cum_var,
                "seed_avg": avg,
                "seed_std": std,
                "leakage_probe": leak,
                "syntax_probe_R2_per_target": probe_R2,
                "syntax_probe_R2_mean": probe_R2_mean,
                "elapsed_s": float(time.time() - t_cell),
            }
            log(f"  M>0={avg.get('M_gt_0_fraction', 0):.3f}, "
                f"Rank@1={avg.get('Rank_at_1', 0):.3f}, "
                f"unique={avg.get('avg_unique_ratio', 0):.3f}, "
                f"ρ={avg.get('rho_spearman_M_vs_logdet', 0):+.3f}, "
                f"cumvar={cum_var:.3f}, "
                f"leak_top1={leak['top1_acc'] if leak else 'NA':.4f}, "
                f"syntax_R²={probe_R2_mean:.3f}, "
                f"elapsed={time.time() - t_cell:.1f}s")

    log("\n=== 4. Pareto analysis ===")
    # rank cells by (M>0, -leak_top1) composite
    cells = []
    for ck, r in results_grid.items():
        cells.append({
            "key": ck,
            "M_gt_0": r["seed_avg"]["M_gt_0_fraction"],
            "Rank_at_1": r["seed_avg"]["Rank_at_1"],
            "unique": r["seed_avg"]["avg_unique_ratio"],
            "rho_M_logdet": r["seed_avg"]["rho_spearman_M_vs_logdet"],
            "cumvar": r["cum_explained_var"],
            "leak_top1": r["leakage_probe"]["top1_acc"] if r["leakage_probe"] else 1.0,
            "syntax_R2": r["syntax_probe_R2_mean"],
            "n_features": r["n_features"],
        })
    # best M>0
    best_M = max(c["M_gt_0"] for c in cells)
    log(f"  best M>0 across all cells: {best_M:.3f}")
    log(f"  95% threshold: {0.95 * best_M:.3f}")
    pareto_candidates = []
    for c in cells:
        score_M_ok = c["M_gt_0"] >= 0.95 * best_M
        score_deconf = abs(c["rho_M_logdet"]) < 0.3
        log(f"  {c['key']}: M>0={c['M_gt_0']:.3f} "
            f"({'✓' if score_M_ok else '✗'}), ρ={c['rho_M_logdet']:+.3f} "
            f"({'✓' if score_deconf else '✗'}), leak={c['leak_top1']:.4f}, "
            f"R²={c['syntax_R2']:.3f}, n_feat={c['n_features']}")
        if score_M_ok and score_deconf:
            pareto_candidates.append(c)
    # among candidates, choose min leak + best syntax
    if pareto_candidates:
        best = min(pareto_candidates, key=lambda c: (c["leak_top1"], -c["syntax_R2"]))
        log(f"\n  ★ Best Pareto config: {best['key']}")
        log(f"    M>0={best['M_gt_0']:.3f}, Rank@1={best['Rank_at_1']:.3f}, "
            f"unique={best['unique']:.3f}")
        log(f"    ρ(M,log|Σ|)={best['rho_M_logdet']:+.3f}, "
            f"cumvar={best['cumvar']:.3f}, leak_top1={best['leak_top1']:.4f}, "
            f"syntax_R²={best['syntax_R2']:.3f}, n_feat={best['n_features']}")
    else:
        best = None
        log("\n  ⚠ No Pareto candidate meets both 95% M>0 + |ρ|<0.3 thresholds")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/repr_sweep_feature.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6k — Stage B 2D ablation. Feature Set × PCA dim. "
                                "Stage B chosen dims {32,48,64} from Stage A plateau. "
                                "Pareto rule: M>0 ≥ 95% of best, |ρ(M,log|Σ|)|<0.3, "
                                "minimize content leakage + maximize syntax R²."),
                "feature_subsets": list(FEATURE_SUBSETS.keys()),
                "pca_dims": PCA_DIMS_STAGE_B,
                "seeds": SEEDS,
                "feature_subset_definitions": {
                    k: {"exclude_prefixes": list(v["exclude_prefixes"]),
                        "exclude_exact": list(v["exclude_exact"])}
                    for k, v in FEATURE_SUBSETS.items()
                },
            },
            "results_grid": results_grid,
            "cells_summary": cells,
            "pareto_best": best,
            "stage_a_recommendation": {
                "pca_dim_plateau": "32-64 (M>0 0.588→0.648, peak at d=64)",
                "overfit_beyond_d64": "d=96 M>0 drops to 0.627",
            },
            "total_elapsed_s": float(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()