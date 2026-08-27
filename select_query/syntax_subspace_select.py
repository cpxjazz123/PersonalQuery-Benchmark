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

    log("\n=== 4. Selection per (asin, user) ===")

    # 用户指令 2026-08-28: Mahal 阈值 gating — 只选 Mahal² ≤ χ²(0.95, df=48) 的 query。
    # σ_diag 已经 LAMBDA=0.1 收缩到 var.mean(),所以 Mahal² 近似 χ²(PCA_DIM=48) 分布
    # (diagonal Gaussian → Mahalanobis 等价)。χ²(0.95, 48) ≈ 65.22 是 95%
    # confidence region。
    mahal_threshold = chi2.ppf(MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF)
    log(f"  Mahal² gating threshold = χ²({MAHAL_THRESHOLD_CHI2_PPF}, {MAHAL_THRESHOLD_DF}) = {mahal_threshold:.3f}")

    # 用户指令 2026-08-28: Posterior-discriminative Gaussian Selection (替代 greedy unique)
    #
    # 不再用 "别人用过这条 query 我就不能用" 的硬规则,而是从分布本身出发:
    #   P(u|z) = p(z|u) / Σ_v p(z|v)
    # 直觉: 在所有用户 Gaussian 中,这条 query 最属于目标用户 u 的 posterior
    # responsibility 越高,说明它越落在 u 的 user-exclusive style region 而不是
    # 公共重叠区。同时保留 95% Mahal² gate 保证 query 仍在 u 自己合理表达范围。
    #
    # 公式:
    #   log p(z|u) = -0.5 * Mahal²(z,u) - 0.5 * log_det(σ_u)   (常数项省略)
    #   P(u|z) = softmax_v(log p(z|v))                          (数值稳定:max-subtraction)
    #
    # 选:
    #   q*_u = argmax_q  P(u|z_q)    subject to  Mahal²(z_q, u) ≤ mahal_threshold
    #
    # 与 greedy unique 的本质区别: 不同 user 选不同 query 是因为 user distribution
    # 本身不同(posterior 给出的概率责任不同),不是 "代码规定不能重复"。
    selection_entries = []

    n_skip_no_pool = 0
    n_out_of_distribution = 0
    n_posterior_max = 0
    n_no_pool_entry = 0
    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        zqs = pool_z.get(asin)
        if zqs is None:
            # 用户指令 2026-08-28: Stage 1 N=5 + numeric/metadata filter 主动过滤
            # 掉 <5 非数值 attr 的 ASIN(典型原因:商品页 attrs 多数是 numeric/Country/Department/
            # Dimensions 等)。这些 ASIN 不在 pool 是预期行为,跳过而非 raise。
            n_skip_no_pool += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
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
                    "selected_posterior": None,
                    "n_candidates": 0,
                    "user_source": users_gauss[uid]["source"],
                    "n_reviews": users_gauss[uid]["n_reviews"],
                })
            continue

        # 收集该 ASIN 所有 user 的 (μ, σ, gauss_info)
        asin_users = []
        for uid in entry["users_sampled"]:
            if uid not in users_gauss:
                raise KeyError(
                    f"user {uid} (ASIN={asin}) not in user_gaussians. "
                    f"上游 build_dataset.py 已过滤 MIN_REVIEWS_PER_USER=20, "
                    f"但 Stage 3 没找到该 user 的 Gaussian — 字段提取不一致, "
                    f"需修 build_dataset.py 或 gaussian/syntax_subspace_user_gaussians.py 的 user_id 提取。"
                )
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            asin_users.append((uid, mu, sigma, users_gauss[uid]))

        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # Mahal² matrix + log_det (per user)
        mahal_matrix = np.zeros((n_users, n_queries))
        log_det = np.zeros(n_users)
        for ui, (uid, mu, sigma, _) in enumerate(asin_users):
            log_det[ui] = float(np.sum(np.log(sigma)))
            for qi, (z, q) in enumerate(strict_zqs):
                mahal_matrix[ui, qi] = mahalanobis_sq(z, mu, sigma)

        # log p(z|u) = -0.5 * Mahal²(z, u) - 0.5 * log_det(σ_u)   (常数项省略)
        log_p = -0.5 * mahal_matrix - 0.5 * log_det[:, None]  # (n_users, n_queries)

        # P(u|z) = softmax over users (per query) — 数值稳定用 max-subtract
        log_p_max = log_p.max(axis=0, keepdims=True)
        log_p_shifted = log_p - log_p_max
        exp_shifted = np.exp(log_p_shifted)
        log_posterior = log_p_shifted - np.log(exp_shifted.sum(axis=0, keepdims=True))
        posterior = np.exp(log_posterior)  # (n_users, n_queries), 沿 user axis sum=1

        # 95% Mahal² gate per (user, query)
        in_dist_mask = mahal_matrix <= mahal_threshold

        for ui, (uid, mu, sigma, gauss_info) in enumerate(asin_users):
            source = gauss_info["source"]
            n_reviews = gauss_info["n_reviews"]
            cand_mask = in_dist_mask[ui]

            if not cand_mask.any():
                # 无 in_dist → out_of_distribution
                best_qi = int(np.argmin(mahal_matrix[ui]))
                best_dist = float(mahal_matrix[ui][best_qi])
                best_posterior = float(posterior[ui][best_qi])
                method = "out_of_distribution"
                n_out_of_distribution += 1
                selected_q = None
            else:
                # argmax posterior subject to gate
                avail_post = np.where(cand_mask, posterior[ui], -np.inf)
                best_qi = int(np.argmax(avail_post))
                best_dist = float(mahal_matrix[ui][best_qi])
                best_posterior = float(posterior[ui][best_qi])
                method = "posterior_max"
                n_posterior_max += 1
                selected_q = strict_zqs[best_qi][1]

            entry_dict = {
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "selected_distance": best_dist,
                "selected_posterior": best_posterior,  # 新增:P(u|z_q*)
                "n_candidates": len(strict_zqs),
                "user_source": source,
                "n_reviews": n_reviews,
            }
            selection_entries.append(entry_dict)

    log(f"  total entries: {len(selection_entries)}")
    log(f"  ASINs skipped (no pool, Stage 1 filtered): {n_skip_no_pool}/{len(asin_data)} "
        f"(Stage 1 N=5 + numeric/metadata filter 主动过滤 <5 非数值 attr 的 ASIN)")
    log(f"  posterior_max (in-dist, max P(u|z_q)): {n_posterior_max}")
    log(f"  out_of_distribution (Mahal² > χ²(0.95, 48)={mahal_threshold:.3f}): "
        f"{n_out_of_distribution}")
    log(f"  no_pool (no strict candidates from Stage 1): {n_no_pool_entry}")

    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: Posterior-discriminative Gaussian Selection "
                                "with Mahal² gating at χ²(0.95, 48). "
                                "对每 (asin, user) 算 P(u|z_q) = p(z_q|u) / Σ_v p(z_q|v) "
                                "(数值稳定 softmax over users),选 argmax_q P(u|z_q) subject to "
                                "Mahal²(z_q, u) ≤ χ²(0.95, 48)。"
                                "selection_method ∈ {posterior_max, out_of_distribution, no_pool}; "
                                "out_of_distribution / no_pool 不参与 Stage 5 retrieval。"),
                "SEED": SEED,
                "MAHAL_THRESHOLD_CHI2_PPF": MAHAL_THRESHOLD_CHI2_PPF,
                "MAHAL_THRESHOLD_DF": MAHAL_THRESHOLD_DF,
                "mahal_threshold": float(mahal_threshold),
            },
            "n_entries": len(selection_entries),
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_OUT}")

    log("\n=== 7. Validation ===")
    mahal_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected_entries = [e for e in mahal_entries
                        if e["selection_method"] == "posterior_max"]
    ood_entries = [e for e in mahal_entries if e["selection_method"] == "out_of_distribution"]
    selected = np.array([e["selected_distance"] for e in selected_entries])
    ood_dist = np.array([e["selected_distance"] for e in ood_entries])
    posterior_vals = np.array([e["selected_posterior"] for e in selected_entries
                               if e["selected_posterior"] is not None])

    log(f"  N pairs total: {len(mahal_entries)} "
        f"(posterior_max={len(selected_entries)}, out_of_distribution={len(ood_entries)})")
    if len(selected) > 0:
        log(f"  posterior_max (in-distribution): "
            f"n={len(selected)}, mean={selected.mean():.3f}, std={selected.std():.3f}, "
            f"median={np.median(selected):.3f}")
    if len(posterior_vals) > 0:
        log(f"  P(u|z_q*) (posterior responsibility): "
            f"mean={posterior_vals.mean():.3f}, median={np.median(posterior_vals):.3f}, "
            f"min={posterior_vals.min():.3f}, max={posterior_vals.max():.3f}")
    if len(ood_dist) > 0:
        log(f"  out_of_distribution (Mahal² > χ²(0.95, 48)={mahal_threshold:.3f}): "
            f"n={len(ood_dist)}, mean={ood_dist.mean():.3f}, "
            f"min={ood_dist.min():.3f}, max={ood_dist.max():.3f}")
    log(f"  (mahal 越小 = query 越接近 user 个体句法骨架;selected 是 P(u|z_q) 在 95% region "
        f"内的 argmax, posterior 越高 = query 越属 user u 独占 style region。out_of_distribution "
        f"是 Mahal² > χ²(0.95, 48) 整段无候选,严格不选。)")

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
    log(f"  (期望: Posterior 让不同 user 自然选不同 query — unique ratio 高,但成因是 "
        f"user Gaussian posterior responsibility 不同,而非 hard exclusion rule。)")

    by_source = collections.defaultdict(list)
    for e in selected_entries:
        by_source[e["user_source"]].append(e["selected_distance"])

    log(f"\n=== Per-source breakdown (posterior_max only) ===")
    for src, ds in by_source.items():
        arr = np.array(ds)
        log(f"  {src} (n={len(ds)}): mean={arr.mean():.3f}, median={np.median(arr):.3f}")

    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_pairs": len(selected_entries),
            "n_posterior_max": len(selected_entries),
            "n_out_of_distribution": len(ood_entries),
            "selected_mean": float(selected.mean()) if len(selected) > 0 else None,
            "selected_std": float(selected.std()) if len(selected) > 0 else None,
            "selected_median": float(np.median(selected)) if len(selected) > 0 else None,
            "posterior_mean": float(posterior_vals.mean()) if len(posterior_vals) > 0 else None,
            "posterior_median": float(np.median(posterior_vals)) if len(posterior_vals) > 0 else None,
            "posterior_min": float(posterior_vals.min()) if len(posterior_vals) > 0 else None,
            "posterior_max": float(posterior_vals.max()) if len(posterior_vals) > 0 else None,
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