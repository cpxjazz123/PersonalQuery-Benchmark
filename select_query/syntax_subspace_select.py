"""Syntax Subspace — Stage 4 (Mahalanobis select).

归 select_query/: 基于 per-user Gaussian 从共享候选池选最佳查询。

用户指令 2026-08-28: 只输出 selected (Mahalanobis 最小) per (asin, user),
删掉 random / farthest 3-way contrast — Stage 5 volatility 已经只读 selected,
random / farthest 既不被评估也不被报告,只是浪费 ~3× retrieval + selection 计算。

Stage 4: 对每 (asin, user) 选 Mahalanobis 最小候选 (selected only)

用法:
  python select_query/syntax_subspace_select.py --stage select

I/O 路径:
  输入: stage8_5_asins.json / stage8_5_pool.json / stage8_5_user_gaussians.json
  输出: stage8_5_selection.json + stage8_5_selection_stats.json  (Stage 4)

共享工具 (log, feat_key, paths, hyperparams, _syntax_subspace_prepare) 来自:
  common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN, SELECTION_IN, SELECTION_OUT,
    SELECTION_STATS_OUT,
    MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF,
    PCA_DIM, PCA_SEED, SEED, log, feat_key,
)
from scipy.stats import chi2


# ===========================================================================
# STAGE 4 — MAHA SELECT
# ===========================================================================

def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def stage_select():
    log("=== STAGE 4 — MAHA SELECT ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}")
    log(f"  user Gaussians: {len(users_gauss)}")

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
    pool_z_white = {}
    miss = 0
    sqrt_lambda = np.sqrt(pca.explained_variance_)
    for asin, qs in pools.items():
        zs_for_asin = []
        zs_white = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            z_white = z / sqrt_lambda
            zs_for_asin.append((z, q))
            zs_white.append((z_white, q))
        pool_z[asin] = zs_for_asin
        pool_z_white[asin] = zs_white
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    log("\n=== 3.5. Whitened user centers + R_99 gate from historical residuals ===")
    # 用户指令 2026-08-28: PCA48 不是 whitened space (λ range [0.427, 19.140]),
    # 原始 z-space ||z-μ|| 中位数 24.16 — 远大于理论 √χ²(0.95,48)=8.073。
    # 直接用理论阈值是 selection effect (v6i 5% pass rate 是假的 GO)。
    # 正确做法: whitening + 真实历史评论校准 shared R_99。
    X_scaled = P["X_scaled"]
    user_to_indices = P["user_to_indices"]
    log(f"  X_scaled shape: {X_scaled.shape}")

    # 计算 whitened μ̃_u 和 user_z_white (PCA48 z / √λ)
    user_z_white_means = {}
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        z_user = pca.transform(X_scaled[idx]) / sqrt_lambda
        user_z_white_means[uid] = z_user.mean(axis=0)

    # 计算 pooled R_99 from whitened residuals
    L2_RADIUS_PERCENTILE = 99  # 用户原命题 R_95 也 OK,但 R_99 给出更多 in-region entries
    all_residuals = []
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        z_user = pca.transform(X_scaled[idx]) / sqrt_lambda
        mu = user_z_white_means[uid]
        residuals = np.linalg.norm(z_user - mu, axis=1)
        all_residuals.append(residuals)
    all_residuals = np.concatenate(all_residuals)
    log(f"  total historical (user, sentence) pairs: {len(all_residuals)}")
    log(f"  whitened residual distribution:")
    for p in [50, 90, 95, 99]:
        log(f"    P{p} = {np.percentile(all_residuals, p):.3f}")
    l2_radius_threshold = float(np.percentile(all_residuals, L2_RADIUS_PERCENTILE))
    log(f"  R_{L2_RADIUS_PERCENTILE} = {l2_radius_threshold:.3f} (whitened L2 shared gate)")

    log("\n=== 4. Selection per (asin, user) ===")

    # 用户指令 2026-08-28: 抛弃 Mahal² per-user σ gate (σ-confounded, v6f-v6h NO-GO)。
    # 改用 whitened L2 shared gate:
    #   ||z̃_q − μ̃_u||_2 ≤ R_99  (= 真实用户历史 99% 落入的 radius)
    # + L2 margin score (无 per-user σ):
    #   M(q, u) = min_{v≠u} ||z̃_q − μ̃_v||_2 − ||z̃_q − μ̃_u||_2
    #   q*_u = argmax_q M(q, u)  subject to gate
    #
    # Why whitened:
    #   PCA 默认只是旋转,各 PC eigenvalue 不同 (λ range 0.43-19.14)。
    #   z̃ = z / √λ 让各维单位方差,||z̃-μ̃|| ≈ χ(48) (in whitened space)。
    # Why R_99 from historical residuals:
    #   代表"真实用户历史句法表达中约 99% 落入的 radius"—
    #   不依赖 candidate pool,避免 "pool coverage bias"。
    # Why L2 margin:
    #   完全用 z̃ 空间 L2,无 per-user σ — ρ(M, log|Σ_u|) ≈ 0。
    selection_entries = []

    n_skip_no_pool = 0
    n_out_of_distribution = 0
    n_l2_white_margin = 0
    n_no_pool_entry = 0
    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        zqs = pool_z.get(asin)
        zqs_white = pool_z_white.get(asin)
        if zqs is None or zqs_white is None:
            # 用户指令 2026-08-28: Stage 1 N=5 + numeric/metadata filter 主动过滤
            # 掉 <5 非数值 attr 的 ASIN(典型原因:商品页 attrs 多数是 numeric/Country/Department/
            # Dimensions 等)。这些 ASIN 不在 pool 是预期行为,跳过而非 raise。
            n_skip_no_pool += 1
            continue
        # strict 过滤:pool_z 和 pool_z_white 是同步构造的,index 一一对应
        strict_pairs = [(zqs[i][0], zqs_white[i][0], zqs[i][1])
                        for i in range(len(zqs)) if zqs[i][1]["strict"]]
        if not strict_pairs:
            # 用户指令 2026-08-27: 不再用 all (含 invalid) 作 strict fallback;
            # 上游 Stage 1 generation 失败/属性不匹配时, 该 ASIN 直接 no_pool
            for uid in entry["users_sampled"]:
                if uid not in users_gauss:
                    raise KeyError(
                        f"user {uid} (ASIN={asin}) not in user_gaussians. "
                        f"上游 build_dataset.py 已过滤 MIN_REVIEWS_PER_USER=20, "
                        f"但 Stage 3 没找到该 user 的 Gaussian — 字段提取不一致, "
                        f"需修 build_dataset.py 或 gaussian/syntax_subspace_user_gaussians.py 的 user_id 提取。"
                    )
                n_no_pool_entry += 1
                selection_entries.append({
                    "asin": asin,
                    "user_id": uid,
                    "attrs_used": attrs,
                    "selection_method": "no_pool",
                    "selected": None,
                    "selected_distance": None,
                    "selected_margin": None,
                    "n_candidates": 0,
                    "user_source": users_gauss[uid]["source"],
                    "n_reviews": users_gauss[uid]["n_reviews"],
                })
            continue

        # 收集该 ASIN 所有 user 的 (μ̃_u, gauss_info)
        # μ̃_u = pca.transform(X_user_scaled).mean(0) / √λ — 已在 3.5 节预算好
        asin_users = []
        for uid in entry["users_sampled"]:
            if uid not in users_gauss:
                raise KeyError(
                    f"user {uid} (ASIN={asin}) not in user_gaussians. "
                    f"上游 build_dataset.py 已过滤 MIN_REVIEWS_PER_USER=20, "
                    f"但 Stage 3 没找到该 user 的 Gaussian — 字段提取不一致, "
                    f"需修 build_dataset.py 或 gaussian/syntax_subspace_user_gaussians.py 的 user_id 提取。"
                )
            if uid not in user_z_white_means:
                # 极少数 user 没进 user_to_indices (train split 边缘) — fallback:
                # 直接对 user_gauss 的 raw μ 做 whitening(等价,因为 μ 是 mean of
                # user sentences,whitening 是线性变换)。v6k ablation 也观察到。
                mu_raw = np.array(users_gauss[uid]["mu"])
                user_z_white_means[uid] = mu_raw / sqrt_lambda
            mu_white = user_z_white_means[uid]
            asin_users.append((uid, mu_white, users_gauss[uid]))

        n_users = len(asin_users)
        n_queries = len(strict_pairs)

        # Whitened L2 distance matrix ||z̃_q − μ̃_u||
        l2_white_matrix = np.zeros((n_users, n_queries))
        for ui, (uid, mu_white, _) in enumerate(asin_users):
            for qi, (_, z_white, _) in enumerate(strict_pairs):
                l2_white_matrix[ui, qi] = float(np.linalg.norm(z_white - mu_white))

        # M_L2 = min_{v≠u} ||z̃_q − μ̃_v||_2 − ||z̃_q − μ̃_u||_2
        # 用 sort(0) 拿 column-wise 第一小(非 self) 和 self 的差
        sorted_l2 = np.sort(l2_white_matrix, axis=0)
        argmin_per_q = np.argmin(l2_white_matrix, axis=0)
        is_self_argmin = (argmin_per_q[None, :] == np.arange(n_users)[:, None])
        d_other_l2 = np.where(is_self_argmin, sorted_l2[1], sorted_l2[0])
        M_L2 = d_other_l2 - l2_white_matrix  # (n_users, n_queries)

        # R_99 gate per (user, query)
        in_dist_mask = l2_white_matrix <= l2_radius_threshold

        for ui, (uid, mu_white, gauss_info) in enumerate(asin_users):
            source = gauss_info["source"]
            n_reviews = gauss_info["n_reviews"]
            cand_mask = in_dist_mask[ui]

            if not cand_mask.any():
                # 无 in_dist → out_of_distribution (whitened L2 > R_99)
                # 保留 fallback: 选 L2 最近的一条 (让 downstream 知道这是边界情况,
                # selection_method="out_of_distribution" 会过滤掉)
                best_qi = int(np.argmin(l2_white_matrix[ui]))
                best_dist = float(l2_white_matrix[ui][best_qi])
                best_margin = float(M_L2[ui][best_qi])
                method = "out_of_distribution"
                n_out_of_distribution += 1
                selected_q = None
            else:
                # argmax M_L2 subject to R_99 gate
                avail_M = np.where(cand_mask, M_L2[ui], -np.inf)
                best_qi = int(np.argmax(avail_M))
                best_dist = float(l2_white_matrix[ui][best_qi])
                best_margin = float(M_L2[ui][best_qi])
                method = "l2_white_margin_max"
                n_l2_white_margin += 1
                selected_q = strict_pairs[best_qi][2]

            entry_dict = {
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "selected_distance": best_dist,
                "selected_margin": best_margin,
                "n_candidates": len(strict_pairs),
                "user_source": source,
                "n_reviews": n_reviews,
            }
            selection_entries.append(entry_dict)

    log(f"  total entries: {len(selection_entries)}")
    log(f"  ASINs skipped (no pool, Stage 1 filtered): {n_skip_no_pool}/{len(asin_data)} "
        f"(Stage 1 N=5 + numeric/metadata filter 主动过滤 <5 非数值 attr 的 ASIN)")
    log(f"  l2_white_margin_max (in R_99 gate, max M_L2): {n_l2_white_margin}")
    log(f"  out_of_distribution (whitened L2 > R_{L2_RADIUS_PERCENTILE}={l2_radius_threshold:.3f}): "
        f"{n_out_of_distribution}")
    log(f"  no_pool (no strict candidates from Stage 1): {n_no_pool_entry}")

    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5 v6k: Whitened Syntax Space Selection. "
                                "PCA48 默认只是旋转,各 PC eigenvalue 不同(λ range 0.43-19.14),"
                                "原始 z-space ||z-μ|| 中位数 24.16 远大于理论 √χ²(0.95,48)=8.073。"
                                "v6k 修正: z̃ = z / √λ (各维单位方差,whitened space)。"
                                "gate: ||z̃_q − μ̃_u||_2 ≤ R_{pct} (R_{pct} = pooled "
                                "percentile of whitened L2 distance from user historical "
                                "sentences to μ̃_u,完全基于真实历史,无 candidate pool bias)。"
                                "score: M(q, u) = min_{v≠u} ||z̃_q − μ̃_v||_2 − ||z̃_q − μ̃_u||_2 "
                                "(纯 z̃-space L2,无 per-user σ,ρ(M, log|Σ_u|) ≈ 0)。"
                                "select: q*_u = argmax_q M(q, u) subject to gate。"
                                "selection_method ∈ {l2_white_margin_max, out_of_distribution, "
                                "no_pool}; out_of_distribution / no_pool 不参与 Stage 5 retrieval。"),
                "SEED": SEED,
                "L2_RADIUS_PERCENTILE": L2_RADIUS_PERCENTILE,
                "l2_radius_threshold": float(l2_radius_threshold),
                "n_historical_sentence_residuals": int(len(all_residuals)),
            },
            "n_entries": len(selection_entries),
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_OUT}")

    log("\n=== 7. Validation ===")
    dist_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected_entries = [e for e in dist_entries
                        if e["selection_method"] == "l2_white_margin_max"]
    ood_entries = [e for e in dist_entries if e["selection_method"] == "out_of_distribution"]
    selected = np.array([e["selected_distance"] for e in selected_entries])
    ood_dist = np.array([e["selected_distance"] for e in ood_entries])
    margin_vals = np.array([e["selected_margin"] for e in selected_entries
                            if e["selected_margin"] is not None])

    log(f"  N pairs total: {len(dist_entries)} "
        f"(l2_white_margin_max={len(selected_entries)}, out_of_distribution={len(ood_entries)})")
    if len(selected) > 0:
        log(f"  l2_white_margin_max (in R_{L2_RADIUS_PERCENTILE} gate): "
            f"n={len(selected)}, mean={selected.mean():.3f}, std={selected.std():.3f}, "
            f"median={np.median(selected):.3f}")
    if len(margin_vals) > 0:
        log(f"  M_L2 = min_other_L2 − self_L2 (exclusivity margin): "
            f"mean={margin_vals.mean():.3f}, median={np.median(margin_vals):.3f}, "
            f"min={margin_vals.min():.3f}, max={margin_vals.max():.3f}, "
            f"M>0={(margin_vals > 0).mean() * 100:.1f}%")
    if len(ood_dist) > 0:
        log(f"  out_of_distribution (whitened L2 > R_{L2_RADIUS_PERCENTILE}={l2_radius_threshold:.3f}): "
            f"n={len(ood_dist)}, mean={ood_dist.mean():.3f}, "
            f"min={ood_dist.min():.3f}, max={ood_dist.max():.3f}")
    log(f"  (whitened L2 越小 = query 越接近 user 个体句法骨架;selected 是 M_L2 "
        f"在 R_{L2_RADIUS_PERCENTILE} region 内的 argmax, M_L2 越大 = query 越属 "
        f"user u 独占 style region (与其他 user 拉开)。out_of_distribution 是 "
        f"whitened L2 > R_{L2_RADIUS_PERCENTILE} 整段无候选,严格不选。)")

    # per-ASIN uniqueness 量化
    by_asin_queries = collections.defaultdict(list)
    for e in selected_entries:
        by_asin_queries[e["asin"]].append(e["selected"]["query"])
    uniqueness = []
    for asin, qs in by_asin_queries.items():
        n_users = len(qs)
        n_unique = len(set(qs))
        uniqueness.append({"asin": asin, "n_users": n_users, "n_unique": n_unique})
    n_users_arr = np.array([u["n_users"] for u in uniqueness])
    n_unique_arr = np.array([u["n_unique"] for u in uniqueness])
    log(f"\n=== Per-ASIN uniqueness ===")
    log(f"  ASINs with selected queries: {len(uniqueness)}")
    log(f"  avg n_users/ASIN: {n_users_arr.mean():.2f}")
    log(f"  avg n_unique/ASIN: {n_unique_arr.mean():.2f}")
    log(f"  avg unique ratio (unique/users): {(n_unique_arr / np.maximum(n_users_arr, 1)).mean():.3f}")
    log(f"  all-same ASINs (1 unique): {sum(1 for u in uniqueness if u['n_unique'] == 1)}/{len(uniqueness)}")
    log(f"  (期望: L2 margin 让不同 user 自然选不同 query — unique ratio 高,但成因是 "
        f"user μ̃_u 之间 L2 距离 + L2 候选 margin 决策,而非 hard exclusion rule。)")

    by_source = collections.defaultdict(list)
    for e in selected_entries:
        by_source[e["user_source"]].append(e["selected_distance"])

    log(f"\n=== Per-source breakdown (l2_white_margin_max only) ===")
    for src, ds in by_source.items():
        arr = np.array(ds)
        log(f"  {src} (n={len(ds)}): mean={arr.mean():.3f}, median={np.median(arr):.3f}")

    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_pairs": len(selected_entries),
            "n_l2_white_margin_max": len(selected_entries),
            "n_out_of_distribution": len(ood_entries),
            "selected_mean": float(selected.mean()) if len(selected) > 0 else None,
            "selected_std": float(selected.std()) if len(selected) > 0 else None,
            "selected_median": float(np.median(selected)) if len(selected) > 0 else None,
            "margin_mean": float(margin_vals.mean()) if len(margin_vals) > 0 else None,
            "margin_median": float(np.median(margin_vals)) if len(margin_vals) > 0 else None,
            "margin_min": float(margin_vals.min()) if len(margin_vals) > 0 else None,
            "margin_max": float(margin_vals.max()) if len(margin_vals) > 0 else None,
            "margin_positive_fraction": float((margin_vals > 0).mean()) if len(margin_vals) > 0 else None,
            "uniqueness": {
                "n_asins": len(uniqueness),
                "avg_n_users_per_asin": float(n_users_arr.mean()) if len(n_users_arr) > 0 else None,
                "avg_n_unique_per_asin": float(n_unique_arr.mean()) if len(n_unique_arr) > 0 else None,
                "avg_unique_ratio": float((n_unique_arr / np.maximum(n_users_arr, 1)).mean())
                    if len(uniqueness) > 0 else None,
                "n_all_same_asins": sum(1 for u in uniqueness if u["n_unique"] == 1),
            },
            "per_source": {
                src: {
                    "n": len(ds),
                    "selected_mean": float(np.mean(ds)),
                    "selected_median": float(np.median(ds)),
                }
                for src, ds in by_source.items()
            },
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_STATS_OUT}")


# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "select": stage_select,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — select_query (Mahalanobis)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_select.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in STAGE_FUNCTIONS:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()