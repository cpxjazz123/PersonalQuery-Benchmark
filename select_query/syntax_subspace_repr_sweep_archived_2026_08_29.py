"""Stage 4 v6k — Representation Sweep (用户指令 2026-08-28).

目标: Feature Set × PCA dim 二维消融,找 user separability + syntax preservation
+ content leakage + σ-stability 之间的最佳 Pareto 配置。

策略 (两阶段):
  Stage A: 固定 F1 Base, 扫描 PCA dim ∈ {8,16,24,32,48,64,96,128}
  Stage B: 固定 best 2-3 PCA dim, 扫描 4 Feature Sets (F1-F4)

每 cell 4 维指标:
  1. User separability: M>0, Rank@1, Rank≤3, unique ratio
  2. Syntax preservation: explained variance (cumsum@95%), syntax ρ (whitened vs raw)
  3. Content leakage: top-200 ASIN linear probe R² on whitened z (val)
  4. σ-confounding stability: ρ(M_L2, log|Σ_u|), PCA seed variation

输入: stage8_5_asins.json + pool + user_gaussians + sentences_318d_cache
输出: result/select_query/repr_sweep.json
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
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    log, feat_key,
)


# ===========================================================================
# Feature subsets (按 prefix 过滤,语义驱动)
# ===========================================================================
# 当前 184d feature inventory:
#   - n-gram 序列 (lexical-sensitive): open_, close_, posbg_, postg_, depbg_
#   - tag counts (类别级, lexical-free): pos_, dep_
#   - structural (lexical-free): nest_, dist_, depth_eq, main_, clpair_, punct_
#   - single tags (混合): n_tok, n_clause, n_coord, n_mod, mean_dist, max_depth,
#                          depth_var, opener, stype, has_passive, is_interrog,
#                          has_cond, acl, advcl, ccomp, xcomp, relcl

NGRAM_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
SEMANTIC_TAGS = ("opener", "stype", "has_passive", "is_interrog", "has_cond",
                 "acl", "advcl", "ccomp", "xcomp", "relcl")

FEATURE_SUBSETS = {
    "F1_Base": dict(exclude_prefixes=(), exclude_exact=()),
    "F2_DropSeq": dict(exclude_prefixes=NGRAM_PREFIXES, exclude_exact=()),
    "F3_CoreStruct": dict(exclude_prefixes=NGRAM_PREFIXES,
                          exclude_exact=SEMANTIC_TAGS + ("opener",)),
    "F4_TagOnly": dict(exclude_prefixes=NGRAM_PREFIXES,
                       exclude_exact=SEMANTIC_TAGS),
}


def select_feature_names(all_names: list[str], exclude_prefixes: tuple,
                         exclude_exact: tuple) -> list[str]:
    """Apply prefix/exact filter, return ordered feature names."""
    keep = []
    for n in all_names:
        if any(n.startswith(p) for p in exclude_prefixes):
            continue
        if n in exclude_exact:
            continue
        keep.append(n)
    return keep


def project_pool_to_z(pca, sqrt_lambda, scaler, fnames_kept,
                      pools, feat_map):
    """Project all pool queries to whitened PCA space.
    Returns: pool_z_white = {asin: [(z_white, query), ...]}
    """
    pool_z_white = {}
    miss = 0
    for asin, qs in pools.items():
        zs = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames_kept], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs.append((z / sqrt_lambda, q))
        pool_z_white[asin] = zs
    return pool_z_white, miss


def compute_user_means(pca, sqrt_lambda, X_scaled_subset, user_to_indices):
    """Compute whitened user means μ̃_u from historical sentences."""
    user_z_white_means = {}
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        if len(idx) < 2:
            continue
        z_user = pca.transform(X_scaled_subset[idx]) / sqrt_lambda
        user_z_white_means[uid] = z_user.mean(axis=0)
    return user_z_white_means


def compute_R99(pca, sqrt_lambda, X_scaled_subset, user_to_indices):
    """R_99 from pooled whitened L2 residuals."""
    residuals = []
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        if len(idx) < 2:
            continue
        z_user = pca.transform(X_scaled_subset[idx]) / sqrt_lambda
        mu = z_user.mean(axis=0)
        residuals.append(np.linalg.norm(z_user - mu, axis=1))
    residuals = np.concatenate(residuals)
    return float(np.percentile(residuals, 99)), len(residuals)


def v6k_select_one_config(pool_z_white, asin_data, users_gauss,
                          user_z_white_means, R99, n_max_asin_users=20):
    """Run v6k selection. Returns list of (asin, user, selected_query, M_L2, self_L2)."""
    entries = []
    for entry in asin_data:
        asin = entry["asin"]
        zqs = pool_z_white.get(asin)
        if not zqs:
            continue
        strict_pairs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_pairs:
            continue

        # collect per-ASIN users
        asin_users = []
        for uid in entry["users_sampled"][:n_max_asin_users]:
            if uid not in users_gauss or uid not in user_z_white_means:
                continue
            asin_users.append((uid, user_z_white_means[uid]))
        if len(asin_users) < 2:
            continue

        n_users = len(asin_users)
        n_queries = len(strict_pairs)
        # L2 matrix
        l2 = np.zeros((n_users, n_queries))
        for ui, (_, mu) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_pairs):
                l2[ui, qi] = float(np.linalg.norm(z - mu))
        # M_L2
        sorted_l2 = np.sort(l2, axis=0)
        argmin = np.argmin(l2, axis=0)
        is_self = (argmin[None, :] == np.arange(n_users)[:, None])
        d_other = np.where(is_self, sorted_l2[1], sorted_l2[0])
        M_L2 = d_other - l2

        # gate
        in_dist = l2 <= R99
        log_det = np.zeros(n_users)
        for ui, (uid, _) in enumerate(asin_users):
            log_det[ui] = float(np.sum(np.log(np.array(users_gauss[uid]["sigma_diag"]))))

        for ui, (uid, _) in enumerate(asin_users):
            cand = in_dist[ui]
            if not cand.any():
                continue
            avail_M = np.where(cand, M_L2[ui], -np.inf)
            best_qi = int(np.argmax(avail_M))
            entries.append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_pairs[best_qi][1]["query"][:80],
                "self_L2": float(l2[ui, best_qi]),
                "M_L2": float(M_L2[ui, best_qi]),
                "rank_M_L2": int((np.sort(M_L2[:, best_qi])[::-1]
                                  >= M_L2[ui, best_qi] - 1e-12).sum()),
                "log_det": float(log_det[ui]),
            })
    return entries


def aggregate_metrics(entries):
    if not entries:
        return None
    M_L2 = np.array([e["M_L2"] for e in entries])
    self_L2 = np.array([e["self_L2"] for e in entries])
    ranks = np.array([e["rank_M_L2"] for e in entries])
    log_dets = np.array([e["log_det"] for e in entries])
    by_asin = collections.defaultdict(set)
    for e in entries:
        by_asin[e["asin"]].add(e["selected_query"])
    n_users_per_asin = collections.Counter(e["asin"] for e in entries)
    unique_ratios = []
    for a, qs in by_asin.items():
        unique_ratios.append(len(qs) / max(n_users_per_asin[a], 1))

    if np.std(M_L2) > 0 and np.std(log_dets) > 0:
        rho_p, _ = pearsonr(M_L2, log_dets)
        rho_s, _ = spearmanr(M_L2, log_dets)
    else:
        rho_p = rho_s = 0.0

    return {
        "n_entries": len(entries),
        "M_L2_mean": float(M_L2.mean()),
        "M_L2_median": float(np.median(M_L2)),
        "M_L2_std": float(M_L2.std()),
        "M_gt_0_fraction": float((M_L2 > 0).mean()),
        "self_L2_mean": float(self_L2.mean()),
        "rank_M_L2_mean": float(ranks.mean()),
        "Rank_at_1": float((ranks == 1).mean()),
        "Rank_le_3": float((ranks <= 3).mean()),
        "avg_unique_ratio": float(np.mean(unique_ratios)),
        "rho_pearson_M_vs_logdet": float(rho_p),
        "rho_spearman_M_vs_logdet": float(rho_s),
    }


def asin_leakage_probe(pca, sqrt_lambda, X_scaled_subset, leak_train, leak_test,
                       label_y, n_classes=200):
    """Train RidgeClassifier on whitened z to predict top-ASIN.
    Returns top-1 acc + macro F1 on leak_test (description of content leak)."""
    Z_train = pca.transform(X_scaled_subset[leak_train]) / sqrt_lambda
    Z_test = pca.transform(X_scaled_subset[leak_test]) / sqrt_lambda
    y_train = label_y[leak_train]
    y_test = label_y[leak_test]
    # filter valid labels
    valid_train = y_train >= 0
    valid_test = y_test >= 0
    Z_train, y_train = Z_train[valid_train], y_train[valid_train]
    Z_test, y_test = Z_test[valid_test], y_test[valid_test]
    if len(np.unique(y_train)) < 2:
        return None
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(Z_train, y_train)
    pred = clf.predict(Z_test)
    acc = float((pred == y_test).mean())
    # top-5 acc (oracle-style) — gives upper bound on leak
    decision = clf.decision_function(Z_test)
    top5_pred = np.argsort(decision, axis=1)[:, -5:]
    top5_acc = float(np.mean([y_test[i] in top5_pred[i] for i in range(len(y_test))]))
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(np.unique(y_train))),
        "top1_acc": acc,
        "top5_acc": top5_acc,
        "chance_top1": 1.0 / max(1, len(np.unique(y_train))),
        "chance_top5": min(5.0, len(np.unique(y_train))) / max(1, len(np.unique(y_train))),
    }


def syntax_preservation(pca, sqrt_lambda, X_scaled_subset, Y_probes,
                        probe_train_idx, probe_test_idx, PROBE_TARGETS):
    """Probe regression on whitened z to predict 6 syntax probes (n_tok etc).
    Returns R² per probe on test."""
    Z_train = pca.transform(X_scaled_subset[probe_train_idx]) / sqrt_lambda
    Z_test = pca.transform(X_scaled_subset[probe_test_idx]) / sqrt_lambda
    Y_train = Y_probes[probe_train_idx]
    Y_test = Y_probes[probe_test_idx]
    out = {}
    for i, tname in enumerate(PROBE_TARGETS):
        y_tr = Y_train[:, i]
        y_te = Y_test[:, i]
        if y_tr.std() < 1e-8:
            out[tname] = None
            continue
        # Standardize y
        y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
        y_tr_s = (y_tr - y_mean) / y_std
        # Ridge regression
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0).fit(Z_train, y_tr_s)
        pred_s = m.predict(Z_test)
        # R² on standardized
        ss_res = ((y_te - pred_s * y_std - y_mean) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum() + 1e-8
        out[tname] = float(1.0 - ss_res / ss_tot)
    return out


def fit_pca(X_train, n_dim, seed):
    """Fit PCA(d) with given seed, return pca, sqrt_lambda."""
    pca = PCA(n_components=n_dim, random_state=seed)
    pca.fit(X_train)
    return pca, np.sqrt(pca.explained_variance_)


def main():
    log("=== Stage 4 v6k — Representation Sweep (Feature × PCA dim) ===")
    t_start = time.time()

    log("\n=== 1. Loading _syntax_subspace_prepare() ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    all_fnames = P["feature_names_ordered"]
    X_scaled = P["X_scaled"]
    train_idx = P["train_idx"]
    user_to_indices = P["user_to_indices"]
    leak_train = P["leak_train"]
    leak_test = P["leak_test"]
    label_y = P["label_y"]
    Y_probes = P["Y_probes"]
    probe_train_idx = P["probe_train_idx"]
    probe_test_idx = P["probe_test_idx"]
    PROBE_TARGETS = P["PROBE_TARGETS"]
    log(f"  X_scaled: {X_scaled.shape}, all features: {len(all_fnames)}")

    log("\n=== 2. Loading pool + Gaussians ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}, users: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Stage A: PCA dim sweep (F1 Base fixed) ===")
    PCA_DIMS = [8, 16, 24, 32, 48, 64, 96]  # 128 skipped: near feature ceiling 184, sqrt_lambda=0
    SEEDS = [42, 123, 2024]
    subset_name = "F1_Base"
    cfg = FEATURE_SUBSETS[subset_name]
    fnames_kept = select_feature_names(all_fnames, cfg["exclude_prefixes"],
                                       cfg["exclude_exact"])
    log(f"  F1_Base: {len(fnames_kept)} features")
    # select X_scaled columns
    col_idx = [all_fnames.index(n) for n in fnames_kept]
    X_sub = X_scaled[:, col_idx]

    stage_a_results = {}
    for d in PCA_DIMS:
        log(f"\n--- d={d} ---")
        t_d = time.time()
        seed_results = []
        for seed in SEEDS:
            pca, sqrt_lambda = fit_pca(X_sub[train_idx], d, seed)
            R99, n_resid = compute_R99(pca, sqrt_lambda, X_sub, user_to_indices)
            user_z_means = compute_user_means(pca, sqrt_lambda, X_sub, user_to_indices)
            pool_z_white, miss = project_pool_to_z(pca, sqrt_lambda, scaler,
                                                   fnames_kept, pools, feat_map)
            entries = v6k_select_one_config(pool_z_white, asin_data, users_gauss,
                                             user_z_means, R99)
            metrics = aggregate_metrics(entries)
            seed_results.append(metrics)
        # aggregate across seeds
        ks = seed_results[0].keys()
        avg = {k: float(np.mean([m[k] for m in seed_results if m[k] is not None]))
               for k in ks if seed_results[0][k] is not None}
        std = {k: float(np.std([m[k] for m in seed_results if m[k] is not None]))
               for k in ks if seed_results[0][k] is not None}
        cum_var = float(pca.explained_variance_ratio_.sum())
        # Leakage probe (single seed for speed)
        pca_leak, sl_leak = fit_pca(X_sub[train_idx], d, 42)
        leak = asin_leakage_probe(pca_leak, sl_leak, X_sub, leak_train, leak_test,
                                  label_y)
        # Syntax probe (single seed)
        probe_R2 = syntax_preservation(pca_leak, sl_leak, X_sub, Y_probes,
                                       probe_train_idx, probe_test_idx, PROBE_TARGETS)
        stage_a_results[f"d={d}"] = {
            "n_features": len(fnames_kept),
            "n_components": d,
            "cum_explained_var": cum_var,
            "R99": float(R99),
            "seed_avg": avg,
            "seed_std": std,
            "leakage_probe": leak,
            "syntax_probe_R2": probe_R2,
            "elapsed_s": float(time.time() - t_d),
        }
        log(f"  d={d}: n_entries={avg.get('n_entries')}, M>0={avg.get('M_gt_0_fraction'):.3f}, "
            f"Rank@1={avg.get('Rank_at_1'):.3f}, unique={avg.get('avg_unique_ratio'):.3f}, "
            f"ρ(M,log|Σ|)={avg.get('rho_spearman_M_vs_logdet'):.3f}, "
            f"cumvar={cum_var:.3f}, leak_top1={leak['top1_acc'] if leak else 'NA'}, "
            f"elapsed={time.time() - t_d:.1f}s")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/repr_sweep_pca.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6k — Stage A PCA dim sweep. F1_Base (184d) "
                                "fixed, PCA dim ∈ {8,16,24,32,48,64,96,128}, "
                                "3 PCA seeds (42,123,2024) for stability."),
                "feature_subset": subset_name,
                "n_features": len(fnames_kept),
                "pca_dims": PCA_DIMS,
                "seeds": SEEDS,
            },
            "results": stage_a_results,
            "total_elapsed_s": float(time.time() - t_start),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()