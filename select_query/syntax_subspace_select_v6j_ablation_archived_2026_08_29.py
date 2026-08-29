"""Stage 4 v6j — Empirical L2-Radius Gate (用户指令 2026-08-28).

v6i G3 L2-radius gate 是首个 GO variant (Rank@1=78.3%, ρ=-0.256 deconfounded),
但 gate pass rate 仅 5.0% — 理论 √χ²(0.95, 48)=8.073 太严。

v6j 实施 empirical pooled L2 threshold (从所有 (user, strict_query) 对
的真实 L2 distance 分布取 percentile),让 gate 更宽松,扩大 pass rate。

策略:
  1. 收集所有 (user, strict_query) L2 distance 到 pool
  2. 算多个 empirical percentile: 50/80/90/95/99
  3. 每个 percentile 跑 G3 (L2 margin score)
  4. 报告 pass rate / Rank@1 / M_L2>0 / ρ / uniqueness
  5. 推荐 best empirical threshold (GO + 合理 pass rate)

GO 条件:
  |ρ(M_L2, log_det)|<0.3 + Rank@1>=25% + M_L2>0>=25%

输入: ASINS_IN / POOL_IN / GAUSSIANS_IN / FEAT_CACHE
输出: result/select_query/v6j_ablation.json
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF,
    PCA_DIM, log, feat_key,
)
from scipy.stats import chi2


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def evaluate_G3_threshold(l2_radius_threshold: float, asin_pool_data):
    """对给定 L2 radius threshold 评估 G3 (L2 margin score).

    asin_pool_data: list of (asin, asin_users_list, strict_zqs, log_dets_per_user)
                    每条 ASIN 包含 user Gaussians + strict pool z
    """
    variant_results = []
    for asin, asin_users, strict_zqs, log_dets in asin_pool_data:
        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # L2 distance matrix
        l2 = np.zeros((n_users, n_queries))
        mahal_per_user = np.zeros((n_users, n_queries))
        for ui, (uid, mu, sigma) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                l2[ui, qi] = float(np.linalg.norm(z - mu))
                diff = z - mu
                mahal_per_user[ui, qi] = float((diff * diff / sigma).sum())

        # M_L2
        sorted_l2 = np.sort(l2, axis=0)
        argmin_l2_per_qi = np.argmin(l2, axis=0)
        is_self_argmin = (argmin_l2_per_qi[None, :] == np.arange(n_users)[:, None])
        d_other_l2 = np.where(is_self_argmin, sorted_l2[1], sorted_l2[0])
        M_L2 = d_other_l2 - l2

        for ui, (uid, mu, sigma) in enumerate(asin_users):
            in_dist = l2[ui] <= l2_radius_threshold
            if not in_dist.any():
                continue
            avail_M = np.where(in_dist, M_L2[ui], -np.inf)
            best_qi = int(np.argmax(avail_M))
            target_M = float(M_L2[ui][best_qi])
            sorted_M = np.sort(M_L2[:, best_qi])[::-1]
            rank = int((sorted_M >= target_M - 1e-12).sum())
            other_M = np.delete(M_L2[:, best_qi], ui)
            margin = target_M - float(other_M.max())
            variant_results.append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi][1]["query"][:80],
                "selected_score": target_M,
                "target_rank": rank,
                "score_margin": margin,
                "log_det": float(log_dets[ui]),
                "selected_mahal_per_user": float(mahal_per_user[ui][best_qi]),
            })
    return variant_results


def aggregate(entries, var_label):
    if not entries:
        return None
    scores = np.array([e["selected_score"] for e in entries])
    log_dets = np.array([e["log_det"] for e in entries])
    ranks = np.array([e["target_rank"] for e in entries])
    margins = np.array([e["score_margin"] for e in entries])
    rank1 = float((ranks == 1).mean())
    margin_pos = float((margins > 0).mean())

    if np.std(scores) > 0 and np.std(log_dets) > 0:
        rho_p, p_p = pearsonr(scores, log_dets)
        rho_s, p_s = spearmanr(scores, log_dets)
    else:
        rho_p = rho_s = 0.0
        p_p = p_s = 1.0

    by_asin_q = collections.defaultdict(list)
    for e in entries:
        by_asin_q[e["asin"]].append(e["selected_query"])
    n_q_total = sum(len(qs) for qs in by_asin_q.values())
    n_unique_total = sum(len(set(qs)) for qs in by_asin_q.values())
    n_all_same = sum(1 for qs in by_asin_q.values() if len(set(qs)) == 1)
    unique_ratio = n_unique_total / max(n_q_total, 1)

    return {
        "label": var_label,
        "n_entries": len(entries),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std": float(scores.std()),
        "target_rank_mean": float(ranks.mean()),
        "target_rank_median": float(np.median(ranks)),
        "rank1_fraction": rank1,
        "rank_le_2_fraction": float((ranks <= 2).mean()),
        "score_margin_mean": float(margins.mean()),
        "score_margin_median": float(np.median(margins)),
        "margin_positive_fraction": margin_pos,
        "rho_pearson_score_vs_logdet": float(rho_p),
        "rho_spearman_score_vs_logdet": float(rho_s),
        "pearson_p": float(p_p),
        "spearman_p": float(p_s),
        "uniqueness": {
            "avg_unique_ratio": float(unique_ratio),
            "n_all_same_asins": n_all_same,
            "n_total_asins": len(by_asin_q),
        },
    }


def main():
    log("=== Stage 4 v6j — Empirical L2-Radius Gate ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    log(f"  PCA{PCA_DIM} ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}, user Gaussians: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    # Per-ASIN 数据准备
    by_asin_users = collections.OrderedDict()
    for entry in asin_data:
        by_asin_users[entry["asin"]] = list(entry["users_sampled"])

    log("\n=== 4. Pre-compute per-ASIN data + collect L2 distribution ===")
    asin_pool_data = []  # (asin, asin_users, strict_zqs, log_dets)
    n_skip = {"no_pool": 0, "no_strict": 0, "lt_2_users": 0}
    all_user_query_l2 = []  # 收集所有 (user, strict_query) L2 distances
    n_total_user_query_pairs = 0

    for asin, user_ids in by_asin_users.items():
        zqs = pool_z.get(asin)
        if zqs is None:
            n_skip["no_pool"] += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_skip["no_strict"] += 1
            continue

        asin_users = []
        for uid in user_ids:
            if uid not in users_gauss:
                continue
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            asin_users.append((uid, mu, sigma))
        if len(asin_users) < 2:
            n_skip["lt_2_users"] += 1
            continue

        n_users = len(asin_users)
        log_dets = np.zeros(n_users)
        for ui, (uid, mu, sigma) in enumerate(asin_users):
            log_dets[ui] = float(np.sum(np.log(sigma)))

        asin_pool_data.append((asin, asin_users, strict_zqs, log_dets))

        # 收集 L2 距离 (per user × per strict_query)
        for ui, (uid, mu, _) in enumerate(asin_users):
            for z, _ in strict_zqs:
                all_user_query_l2.append(float(np.linalg.norm(z - mu)))
                n_total_user_query_pairs += 1

    log(f"  ASINs prepared: {len(asin_pool_data)}, skipped: {n_skip}")
    log(f"  total (user, strict_query) pairs: {n_total_user_query_pairs}")
    log(f"  L2 distance distribution:")
    all_user_query_l2 = np.array(all_user_query_l2)
    for p in [10, 25, 50, 75, 80, 90, 95, 99]:
        log(f"    P{p} = {np.percentile(all_user_query_l2, p):.3f}")

    # Empirical thresholds
    empirical_thresholds = {
        "P50": float(np.percentile(all_user_query_l2, 50)),
        "P80": float(np.percentile(all_user_query_l2, 80)),
        "P90": float(np.percentile(all_user_query_l2, 90)),
        "P95": float(np.percentile(all_user_query_l2, 95)),
        "P99": float(np.percentile(all_user_query_l2, 99)),
        "theoretical_chi48_95": float(np.sqrt(chi2.ppf(0.95, 48))),  # 8.073
    }

    log("\n=== 5. Evaluate G3 at each empirical threshold ===")
    summary = {}
    for label, threshold in empirical_thresholds.items():
        entries = evaluate_G3_threshold(threshold, asin_pool_data)
        agg = aggregate(entries, f"G3_L2_gate_{label}_thresh={threshold:.3f}")
        if agg is None:
            continue
        summary[label] = {
            "threshold": threshold,
            **agg,
        }
        log(f"\n--- G3 L2 gate [{label}] threshold = {threshold:.3f} ---")
        log(f"  n_entries = {agg['n_entries']}")
        log(f"  score: mean={agg['score_mean']:.3f}, median={agg['score_median']:.3f}")
        log(f"  Rank@1 = {agg['rank1_fraction']*100:.1f}%, M_L2>0 = {agg['margin_positive_fraction']*100:.1f}%")
        log(f"  ρ(M_L2, log_det): Spearman = {agg['rho_spearman_score_vs_logdet']:+.3f}")
        log(f"  uniqueness: avg ratio = {agg['uniqueness']['avg_unique_ratio']:.3f}, "
            f"all-same = {agg['uniqueness']['n_all_same_asins']}/{agg['uniqueness']['n_total_asins']}")
        deconfounded = abs(agg["rho_spearman_score_vs_logdet"]) < 0.3
        rank1_ok = agg["rank1_fraction"] >= 0.25
        margin_ok = agg["margin_positive_fraction"] >= 0.25
        go = deconfounded and rank1_ok and margin_ok
        summary[label]["go"] = go
        summary[label]["deconfounded"] = deconfounded
        log(f"  GO verdict: deconfounded={deconfounded}, rank1≥25%={rank1_ok}, "
            f"margin≥25%={margin_ok}, GO={go}")

    log("\n=== 6. Recommendation ===")
    # 找最高 pass rate 的 GO variant
    go_candidates = [(label, s) for label, s in summary.items() if s.get("go")]
    if go_candidates:
        best_label, best = max(go_candidates, key=lambda x: x[1]["n_entries"])
        log(f"  Best GO variant by pass count: {best_label} (threshold={best['threshold']:.3f}, "
            f"n={best['n_entries']})")
    else:
        log("  No GO variant — empirical thresholds all fail")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/v6j_ablation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6j: G3 L2-margin score + empirical pooled L2 radius "
                                "threshold (vs theoretical sqrt(chi2(0.95, 48))=8.073). "
                                "Empirical thresholds from pooled (user, strict_query) L2 "
                                "distance percentile. GO: |ρ|<0.3 + Rank@1>=25% + M>0>=25%."),
                "MAHAL_THRESHOLD_CHI2_PPF": MAHAL_THRESHOLD_CHI2_PPF,
                "MAHAL_THRESHOLD_DF": MAHAL_THRESHOLD_DF,
                "n_total_user_query_pairs": int(n_total_user_query_pairs),
                "L2_distribution_percentiles": {p: float(np.percentile(all_user_query_l2, int(p[1:])))
                                                 for p in ["P10", "P25", "P50", "P75", "P80", "P90", "P95", "P99"]},
            },
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()