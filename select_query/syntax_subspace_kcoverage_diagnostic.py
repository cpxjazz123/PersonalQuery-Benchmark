"""K-coverage diagnostic (用户指令 2026-08-28).

对比 K=50 / K=100 / K=200 pool queries 在 F3_CoreStruct + PCA48 whitened 空间中的
coverage pattern:
  1. Per-ASIN span (max pairwise distance, centroid radius, cov log-det, PCA top-3)
  2. Span growth ratio K=200 / K=50
  3. New query direction: K=200\K=50 新增 query 落在 K=50 query set 的
     99% Mahalanobis 椭球内的比例 (=densification) vs 外 (=span extension)

输入:
  pool_K50_F3pca48_backup.json + pool_K100_F3pca48.json + pool_K200_F3pca48.json
  sentences_318d_cache.jsonl.gz (用于重抽 features)
输出:
  result/select_query/kcoverage_diagnostic.json
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    _syntax_subspace_prepare, FEAT_CACHE, log,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, select_feature_names,
)


POOLS = {
    50:  Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K50_F3pca48_backup.json"),
    100: Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K100_F3pca48.json"),
    200: Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48.json"),
}
PCA_DIM = 48
PCA_SEED = 2024
LAMBDA = 0.1
FEATURE_SUBSET = "F3_CoreStruct"


def main():
    log("=== K-coverage diagnostic: K=50 / K=100 / K=200 ===")
    t_start = time.time()

    # ---- 1. Load 318d feature cache ----
    log("\n=== Loading 318d feature cache ===")
    feat_map: dict = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map: {len(feat_map)} entries")

    # ---- 2. Build F3_CoreStruct + PCA48 + whitening ----
    log("\n=== Building F3_CoreStruct + PCA48 ===")
    P = _syntax_subspace_prepare()
    X = P["X"]
    all_fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    cfg = FEATURE_SUBSETS[FEATURE_SUBSET]
    fnames = select_feature_names(all_fnames, cfg["exclude_prefixes"], cfg["exclude_exact"])
    col_idx = [all_fnames.index(n) for n in fnames]
    X_sub = X[:, col_idx]
    scaler_sub = StandardScaler()
    scaler_sub.fit(X_sub[train_idx])
    X_sub_scaled = scaler_sub.transform(X_sub)
    log(f"  {FEATURE_SUBSET}: {len(fnames)} features, X_sub_scaled: {X_sub_scaled.shape}")

    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(X_sub_scaled[train_idx])
    sqrt_lambda = np.sqrt(pca.explained_variance_ + LAMBDA)
    cum_var = float(pca.explained_variance_ratio_.sum())
    log(f"  PCA48 cumvar={cum_var:.4f}, sqrt_lambda mean={float(sqrt_lambda.mean()):.4f}")

    # ---- 3. Project pool queries to whitened space ----
    log("\n=== Projecting pools to whitened 48d space ===")
    pool_z: dict[int, dict[str, np.ndarray]] = {}
    for K, fp in POOLS.items():
        log(f"  K={K}: loading {fp.name}")
        pools = json.load(open(fp))["pools"]
        asin_z: dict[str, np.ndarray] = {}
        miss = 0
        for asin, qs in pools.items():
            zs = []
            for q in qs:
                import hashlib as _h
                k = _h.sha1(q["query"].strip().lower().encode("utf-8")).hexdigest()
                feats = feat_map.get(k)
                if not feats:
                    miss += 1
                    continue
                vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
                z = pca.transform(scaler_sub.transform(vec[None, :]))[0] / sqrt_lambda
                zs.append(z)
            if zs:
                asin_z[asin] = np.stack(zs)
        pool_z[K] = asin_z
        log(f"    {len(asin_z)} ASINs, miss={miss}")

    # ---- 4. Per-ASIN span metrics ----
    log("\n=== Per-ASIN span metrics (whitened 48d) ===")
    span_metrics: dict[int, dict[str, dict]] = {}
    for K in [50, 100, 200]:
        per_asin: dict[str, dict] = {}
        for asin, Z in pool_z[K].items():
            if len(Z) < 2:
                continue
            diff = Z - Z.mean(0)
            dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            r = np.linalg.norm(diff, axis=1)
            try:
                cov = np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])
                cov_logdet = float(np.linalg.slogdet(cov)[1])
            except Exception:
                cov_logdet = float("nan")
            pca3 = PCA(n_components=min(3, len(Z) - 1, Z.shape[1]))
            pca3.fit(Z)
            per_asin[asin] = {
                "n_queries": int(len(Z)),
                "max_pairwise_dist": float(dists.max()),
                "mean_pairwise_dist": float(dists[dists > 0].mean()) if (dists > 0).any() else 0.0,
                "centroid_radius_mean": float(r.mean()),
                "centroid_radius_max": float(r.max()),
                "cov_logdet": cov_logdet,
                "pca_var_top3": float(pca3.explained_variance_ratio_.sum()),
            }
        span_metrics[K] = per_asin
        n = len(per_asin)
        if n:
            log(f"  K={K}: N={n}")
            log(f"    max_pairwise_dist: mean={np.mean([v['max_pairwise_dist'] for v in per_asin.values()]):.3f}, "
                f"median={np.median([v['max_pairwise_dist'] for v in per_asin.values()]):.3f}")
            log(f"    centroid_radius_max: mean={np.mean([v['centroid_radius_max'] for v in per_asin.values()]):.3f}, "
                f"median={np.median([v['centroid_radius_max'] for v in per_asin.values()]):.3f}")
            log(f"    cov_logdet: mean={np.mean([v['cov_logdet'] for v in per_asin.values()]):.3f}, "
                f"median={np.median([v['cov_logdet'] for v in per_asin.values()]):.3f}")
            log(f"    pca_var_top3: mean={np.mean([v['pca_var_top3'] for v in per_asin.values()]):.3f}")

    # ---- 5. Span growth ratio K=200 / K=50 ----
    log("\n=== Span growth ratio K=200 / K=50 ===")
    growth = []
    for asin in span_metrics[200]:
        if asin in span_metrics[50]:
            s50 = span_metrics[50][asin]
            s200 = span_metrics[200][asin]
            growth.append({
                "asin": asin,
                "max_pairwise_dist_K50": s50["max_pairwise_dist"],
                "max_pairwise_dist_K200": s200["max_pairwise_dist"],
                "max_pairwise_dist_ratio_K200_K50": s200["max_pairwise_dist"] / max(s50["max_pairwise_dist"], 1e-6),
                "centroid_radius_max_ratio": s200["centroid_radius_max"] / max(s50["centroid_radius_max"], 1e-6),
                "cov_logdet_diff": s200["cov_logdet"] - s50["cov_logdet"],
                "cov_logdet_ratio": (s200["cov_logdet"] / max(s50["cov_logdet"], -1000.0)) if s50["cov_logdet"] != 0 else float("nan"),
            })
    log(f"  N={len(growth)}")
    log(f"  median max_pairwise_dist ratio K=200/K=50 = {np.median([g['max_pairwise_dist_ratio_K200_K50'] for g in growth]):.3f}")
    log(f"  median centroid_radius_max ratio = {np.median([g['centroid_radius_max_ratio'] for g in growth]):.3f}")
    log(f"  median cov_logdet diff K=200-K=50 = {np.median([g['cov_logdet_diff'] for g in growth]):.3f}")
    log(f"  fraction ASINs K=200 span >= K=50 (ratio >= 1.0): "
        f"{np.mean([g['max_pairwise_dist_ratio_K200_K50'] >= 1.0 for g in growth]):.3f}")

    # ---- 6. New query direction: K=200 - K=50 sample first 50 ----
    log("\n=== New query direction (K=200 - K=50, K=50 first 50 queries / ASIN) ===")
    from scipy.stats import chi2
    chi2_99_ppf = float(chi2.ppf(0.99, PCA_DIM))
    log(f"  chi2(0.99, 48) = {chi2_99_ppf:.3f}")

    direction_metrics = []
    for asin in pool_z[200]:
        Z50 = pool_z[50].get(asin)
        Z200 = pool_z[200].get(asin)
        if Z50 is None or Z200 is None or len(Z50) < 5:
            continue
        # Assumption: pool queries are stored in sample order — K=50 = first 50 of K=200
        Z_new = Z200[len(Z50):]
        if len(Z_new) == 0:
            continue
        # K=50 query set centroid + cov (Mahalanobis ellipsoid for K=50)
        mu = Z50.mean(0)
        Sigma = np.cov(Z50.T) + 1e-6 * np.eye(Z50.shape[1])
        try:
            inv_Sigma = np.linalg.inv(Sigma)
        except Exception:
            continue
        diffs = Z_new - mu
        m_dist = np.einsum("ni,ij,nj->n", diffs, inv_Sigma, diffs)
        n_inside = int((m_dist < chi2_99_ppf).sum())
        n_outside = int((m_dist >= chi2_99_ppf).sum())
        direction_metrics.append({
            "asin": asin,
            "n_new": int(len(Z_new)),
            "n_inside_K50_99_ellipsoid": n_inside,
            "n_outside_K50_99_ellipsoid": n_outside,
            "frac_inside": n_inside / max(n_inside + n_outside, 1),
            "mean_m_dist": float(m_dist.mean()),
            "median_m_dist": float(np.median(m_dist)),
            "max_m_dist": float(m_dist.max()),
        })

    n_total_new = sum(m["n_new"] for m in direction_metrics)
    n_total_inside = sum(m["n_inside_K50_99_ellipsoid"] for m in direction_metrics)
    n_total_outside = sum(m["n_outside_K50_99_ellipsoid"] for m in direction_metrics)
    frac_inside = n_total_inside / max(n_total_new, 1)
    frac_outside = n_total_outside / max(n_total_new, 1)
    log(f"  total new queries (K=200\K=50): {n_total_new}")
    log(f"  inside K=50 99% Mahalanobis ellipsoid: {n_total_inside} ({frac_inside*100:.1f}%)")
    log(f"  outside K=50 99% Mahalanobis ellipsoid: {n_total_outside} ({frac_outside*100:.1f}%)")

    if frac_inside > 0.7:
        verdict = "MOSTLY DENSIFICATION: K=200 新增 query 主要落在 K=50 已覆盖的 ellipsoid 内 → 再继续增大 K 的边际收益会快速饱和"
    elif frac_outside > 0.5:
        verdict = "MOSTLY SPAN EXTENSION: K=200 新增 query 一半以上落在 K=50 ellipsoid 外 → K 增大确实在扩展 coverage, 继续 K=400 仍可能有显著收益"
    else:
        verdict = "MIXED: K=200 同时在做 densification 和 span extension, 但需要更细粒度诊断"
    log(f"\n  >>> VERDICT: {verdict}")

    # ---- 7. Same-K comparison: K=200 vs K=100 new queries ----
    log("\n=== Bonus: K=200\K=100 new queries direction ===")
    direction_200_100 = []
    for asin in pool_z[200]:
        Z100 = pool_z[100].get(asin)
        Z200 = pool_z[200].get(asin)
        if Z100 is None or Z200 is None or len(Z100) < 5:
            continue
        Z_new = Z200[len(Z100):]
        if len(Z_new) == 0:
            continue
        mu = Z100.mean(0)
        Sigma = np.cov(Z100.T) + 1e-6 * np.eye(Z100.shape[1])
        try:
            inv_Sigma = np.linalg.inv(Sigma)
        except Exception:
            continue
        diffs = Z_new - mu
        m_dist = np.einsum("ni,ij,nj->n", diffs, inv_Sigma, diffs)
        n_inside = int((m_dist < chi2_99_ppf).sum())
        direction_200_100.append({
            "asin": asin,
            "n_new": int(len(Z_new)),
            "frac_inside_K100_99_ellipsoid": n_inside / max(len(Z_new), 1),
        })
    n_total_new_100 = sum(m["n_new"] for m in direction_200_100)
    n_total_inside_100 = sum(int(m["frac_inside_K100_99_ellipsoid"] * m["n_new"]) for m in direction_200_100)
    log(f"  K=200\K=100 total new: {n_total_new_100}")
    log(f"  inside K=100 99% ellipsoid: {n_total_inside_100} ({n_total_inside_100/max(n_total_new_100,1)*100:.1f}%)")

    # ---- 8. Save ----
    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/kcoverage_diagnostic.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("K-coverage diagnostic: K=50/100/200 pool coverage in F3_CoreStruct+PCA48 "
                                "whitened 48d space. Diagnoses whether K growth extends syntax-space span "
                                "or only densifies. Mahalanobis 99% ellipsoid is the K=50 query set's "
                                "fit-Gaussian confidence region; new queries inside = densification, "
                                "outside = span extension."),
                "K_values": [50, 100, 200],
                "feature_subset": FEATURE_SUBSET,
                "pca_dim": PCA_DIM,
                "lambda_shrinkage": LAMBDA,
                "chi2_99_ppf_df48": chi2_99_ppf,
            },
            "per_asin_span": {
                str(K): span_metrics[K] for K in [50, 100, 200]
            },
            "span_growth_K200_vs_K50": growth,
            "new_query_direction_K200_minus_K50": direction_metrics,
            "new_query_direction_K200_minus_K100": direction_200_100,
            "summary": {
                "n_total_new_K200_minus_K50": n_total_new,
                "n_inside_K50_ellipsoid": n_total_inside,
                "frac_inside_K50_ellipsoid": frac_inside,
                "frac_outside_K50_ellipsoid": frac_outside,
                "interpretation_densification_pct": float(frac_inside * 100),
                "interpretation_span_extension_pct": float(frac_outside * 100),
                "verdict": verdict,
                "K200_vs_K100": {
                    "n_total_new": n_total_new_100,
                    "n_inside_K100_ellipsoid": n_total_inside_100,
                    "frac_inside": n_total_inside_100 / max(n_total_new_100, 1),
                },
                "span_growth_K200_vs_K50_summary": {
                    "median_max_pairwise_dist_ratio": float(np.median([g['max_pairwise_dist_ratio_K200_K50'] for g in growth])),
                    "median_cov_logdet_diff": float(np.median([g['cov_logdet_diff'] for g in growth])),
                    "frac_asins_K200_span_ge_K50": float(np.mean([g['max_pairwise_dist_ratio_K200_K50'] >= 1.0 for g in growth])),
                },
            },
            "total_elapsed_s": float(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
