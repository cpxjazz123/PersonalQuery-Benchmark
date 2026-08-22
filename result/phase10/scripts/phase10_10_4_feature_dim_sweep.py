#!/usr/bin/env python3
"""Phase 10.10.4: Feature Dimension Sweep for Rerank Utility.

横向比较不同特征表示 / 距离度量的 rerank utility, 看哪个最稳定 + 泛化最好.

实验矩阵 (8 个方法, 同 N=876 上跑):

| 名称 | 特征 | 预处理 | 距离 |
|------|------|--------|------|
| A_raw_l2 | 20d | raw | L2 |
| B_std_l2 | 20d | z-score per feature | L2 |
| C_l2norm_cos | 20d | L2-normalize per query | L2 (= cosine) |
| D_raw_maha | 20d | raw | Mahalanobis (VADES user_logvar) |
| E_pca10_l2 | PCA 10d | raw | L2 |
| F_pca5_l2 | PCA 5d | raw | L2 |
| G_complex5_l2 | 5d complexity core | raw | L2 |
| H_syntactic10_l2 | 10d full syntactic | raw | L2 |

每个方法都 rerank top-1 by min distance to target_user_centroid_in_that_space,
然后对比 random_pick 平均距离. rerank_utility > 0 表示 rerank 有效.

输出:
  - phase10_10_4_feature_dim_sweep.json: 每个方法 mean + CI + 排名
  - phase10_10_4_per_pair_per_method.jsonl: per-pair metrics for each method

期望:
- 大部分方法 rerank utility > 0 (rerank 效果是 robust 的, 不是 overfit 到单一 metric)
- 标准化 (B/C) 应与 raw (A) 接近 → 说明特征 scale 不是决定性因素
- PCA 降维 (E/F) utility 应下降但仍 > 0 → 关键信息在前 5-10 维
- Subset (G/H) utility 应接近 full → 说明 syntactic 维度足够捕捉 user style
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
CANDIDATES_FILE = OUT_DIR / "phase10_9_4i2_candidates_1000.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
SENTENCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_EVAL = OUT_DIR / "phase10_10_4_feature_dim_sweep.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_4_per_pair.jsonl"
LOG_OUT = OUT_DIR / "phase10_10_4_feature_dim_sweep.log"

SEED = 42
N_BOOTSTRAP = 5000


def log(m):
    print(f"[phase10-10.4] {m}", flush=True)


def load_user_profiles():
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    return user_id_to_idx, user_mu, user_logvar


def load_sentence_data():
    """加载 SENTENCE_FILE, 返回 feat_array + 每行的 user_id + feature_names."""
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    feature_names = list(rows[0]["features"].keys())
    feat_array = np.array(
        [[float(r["features"][k]) for k in feature_names] for r in rows],
        dtype=np.float32,
    )
    user_ids = [r["user_id"] for r in rows]
    return feat_array, user_ids, feature_names


def compute_per_user_centroids_and_pca(feat_array, user_ids, feature_names):
    """每用户 20d 特征均值, + z-score params + PCA on user centroids."""
    by_user: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(user_ids):
        by_user[uid].append(i)
    user_centroids = {}
    for uid, idxs in by_user.items():
        user_centroids[uid] = feat_array[idxs].mean(axis=0)
    centroids_arr = np.stack([user_centroids[uid] for uid in user_centroids.keys()], axis=0)
    centroid_uids = list(user_centroids.keys())
    uid_to_centroid_idx = {uid: i for i, uid in enumerate(centroid_uids)}

    # Z-score params (per feature) computed on user centroids
    c_mean = centroids_arr.mean(axis=0)
    c_std = centroids_arr.std(axis=0) + 1e-9

    # PCA via SVD on user centroids (centered)
    centered = centroids_arr - c_mean
    # SVD: centered = U S Vt; PCA components = Vt[:k]
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt.astype(np.float32)  # [D, D]

    return {
        "user_centroids": user_centroids,
        "uid_to_centroid_idx": uid_to_centroid_idx,
        "c_mean": c_mean,
        "c_std": c_std,
        "pca_components": pca_components,
        "feature_names": feature_names,
    }


def apply_pca(feat_arr, pca_components, c_mean, k):
    """Project to PCA-k space."""
    centered = feat_arr - c_mean
    return centered @ pca_components[:k].T


def apply_zscore(feat_arr, c_mean, c_std):
    return (feat_arr - c_mean) / c_std


def l2_normalize_rows(feat_arr):
    norms = np.linalg.norm(feat_arr, axis=1, keepdims=True) + 1e-9
    return feat_arr / norms


def extract_features(query_text: str, feature_names: List[str]):
    if not query_text or not query_text.strip():
        return None
    try:
        feat_dict = extract_clause_features(query_text)
        return np.array([float(feat_dict[name]) for name in feature_names], dtype=np.float32)
    except Exception:
        return None


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    if not query or not attrs:
        return False
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        tokens = []
        for t in v_clean.split():
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True; break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True; break
            if (t_lower + "s") in q_clean:
                found = True; break
        if not found:
            return False
    return True


def bootstrap_ci(values_per_item: Dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    items = list(values_per_item.keys())
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0)
    obs_vals = np.array([values_per_item[k] for k in items])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals = obs_vals[sample_idx]
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high)


# === Subset indices (by feature name) ===
SYNTAX_FEATURES_10 = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth",
]
COMPLEXITY_CORE_5 = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "clause_nesting_depth",
]


def main():
    log("=" * 70)
    log("Phase 10.10.4: Feature Dimension Sweep (8 methods, n=876)")
    log("=" * 70)

    log("[1] Loading data ...")
    user_id_to_idx, user_mu, user_logvar = load_user_profiles()
    feat_array, user_ids, feature_names = load_sentence_data()
    log(f"  VADES: {len(user_id_to_idx)} users, 20d")
    log(f"  sentences: {len(feat_array)}, features: {feature_names[:3]}...")

    log("[2] Computing per-user centroids + PCA ...")
    ctx = compute_per_user_centroids_and_pca(feat_array, user_ids, feature_names)
    user_centroids = ctx["user_centroids"]
    uid_to_cidx = ctx["uid_to_centroid_idx"]
    c_mean = ctx["c_mean"]
    c_std = ctx["c_std"]
    pca_components = ctx["pca_components"]
    log(f"  centroids: {len(user_centroids)}")

    # Pre-compute per-user representations in all 8 spaces
    log("[3] Pre-computing user representations in 8 spaces ...")
    user_reps = {}
    for uid, c in user_centroids.items():
        c_raw = c
        c_std_z = (c - c_mean) / c_std
        c_l2n = c / (np.linalg.norm(c) + 1e-9)
        c_pca10 = apply_pca(c[None], pca_components, c_mean, 10)[0]
        c_pca5 = apply_pca(c[None], pca_components, c_mean, 5)[0]
        idx_syntactic = [feature_names.index(n) for n in SYNTAX_FEATURES_10]
        idx_complex = [feature_names.index(n) for n in COMPLEXITY_CORE_5]
        c_synt10 = c[idx_syntactic]
        c_complex5 = c[idx_complex]
        user_reps[uid] = {
            "raw": c_raw, "std": c_std_z, "l2n": c_l2n,
            "pca10": c_pca10, "pca5": c_pca5,
            "synt10": c_synt10, "complex5": c_complex5,
        }

    # === Load pairs ===
    log("[4] Loading pairs ...")
    pair_lookup = {}
    with PAIRS_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            pair_lookup[(p["user_id"], p["asin"])] = p
    log(f"  pairs: {len(pair_lookup)}")

    # === Load candidates ===
    log("[5] Loading candidates ...")
    cand_by_pair: Dict[Tuple[str, str], Dict[str, list]] = {}
    with CANDIDATES_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["user_id"], d["asin"])
            cond = d["cond"]
            cand_by_pair.setdefault(key, {}).setdefault(cond, []).append(d)
    log(f"  total pairs (w/ candidates): {len(cand_by_pair)}")

    # === Per-pair evaluation across 8 methods ===
    log("[6] Computing per-pair metrics for 8 methods ...")
    n_feat_fail = 0
    n_attr_fail = 0
    per_pair_metrics = {}

    METHODS = [
        ("A_raw_l2", "raw", "l2"),
        ("B_std_l2", "std", "l2"),
        ("C_l2norm_cos", "l2n", "l2"),
        ("D_raw_maha", "raw", "maha"),
        ("E_pca10_l2", "pca10", "l2"),
        ("F_pca5_l2", "pca5", "l2"),
        ("G_complex5_l2", "complex5", "l2"),
        ("H_synt10_l2", "synt10", "l2"),
    ]
    for (uid, asin), conds_dict in cand_by_pair.items():
        if "D_target_style" not in conds_dict or "C_random_style" not in conds_dict:
            continue
        target_pair = pair_lookup.get((uid, asin))
        if not target_pair:
            continue
        random_uid = target_pair["random_user_id"]
        target_idx = user_id_to_idx.get(uid)
        random_idx = user_id_to_idx.get(random_uid)
        if target_idx is None or random_idx is None:
            continue
        if uid not in user_reps or random_uid not in user_reps:
            continue

        # VADES maha logvar
        logvar_t = user_logvar[target_idx]
        logvar_r = user_logvar[random_idx]
        reps_t = user_reps[uid]
        reps_r = user_reps[random_uid]
        feat_names_subset = [SYNTAX_FEATURES_10, COMPLEXITY_CORE_5]

        # Pool: 5 D_target + 5 C_random, with attr_pass filter
        pool = []
        for cond in ("D_target_style", "C_random_style"):
            for ci, c in enumerate(conds_dict[cond]):
                feat_raw = extract_features(c["candidate_query"], feature_names)
                if feat_raw is None:
                    n_feat_fail += 1
                    feat_n = np.zeros(len(feature_names), dtype=np.float32)
                else:
                    feat_n = feat_raw
                # Pre-compute per-method representations for this candidate
                cand_reps = {}
                cand_reps["raw"] = feat_n
                cand_reps["std"] = (feat_n - c_mean) / c_std
                cand_reps["l2n"] = feat_n / (np.linalg.norm(feat_n) + 1e-9)
                cand_reps["pca10"] = apply_pca(feat_n[None], pca_components, c_mean, 10)[0]
                cand_reps["pca5"] = apply_pca(feat_n[None], pca_components, c_mean, 5)[0]
                cand_reps["synt10"] = feat_n[[feature_names.index(n) for n in SYNTAX_FEATURES_10]]
                cand_reps["complex5"] = feat_n[[feature_names.index(n) for n in COMPLEXITY_CORE_5]]
                # Compute distances for each method
                distances = {}
                for method_name, rep_key, dist_kind in METHODS:
                    ct = reps_t[rep_key]
                    cr = reps_r[rep_key]
                    cq = cand_reps[rep_key]
                    if dist_kind == "l2":
                        d_target = float(np.linalg.norm(cq - ct))
                        d_wrong = float(np.linalg.norm(cq - cr))
                    elif dist_kind == "maha":
                        diff_t = cq - user_mu[target_idx]
                        diff_r = cq - user_mu[random_idx]
                        d_target = float(((diff_t ** 2) * np.exp(-logvar_t)).sum())
                        d_wrong = float(((diff_r ** 2) * np.exp(-logvar_r)).sum())
                    distances[method_name] = (d_target, d_wrong)
                cand_attrs = c["attrs"]
                attr_ok = check_attr_pass(c["candidate_query"], cand_attrs)
                if not attr_ok:
                    n_attr_fail += 1
                pool.append({
                    "cond": cond, "ci": ci,
                    "distances": distances,
                    "attr_pass": attr_ok,
                })

        # Filter pool to attr_pass
        pool_filtered = [p for p in pool if p["attr_pass"]]
        if len(pool_filtered) < 1:
            continue

        # Per-method: rerank_top1 + random_pick avg
        rng = np.random.default_rng(SEED + hash((uid, asin)) % (2**31))
        random_picks = [pool_filtered[rng.integers(0, len(pool_filtered))] for _ in range(100)]
        method_results = {}
        for method_name, _, _ in METHODS:
            # Rerank: top-1 by min d_target
            pool_sorted = sorted(pool_filtered, key=lambda p: p["distances"][method_name][0])
            rerank_top1 = pool_sorted[0]
            d_target_top1 = rerank_top1["distances"][method_name][0]
            d_wrong_top1 = rerank_top1["distances"][method_name][1]
            d_target_random = float(np.mean([p["distances"][method_name][0] for p in random_picks]))
            utility = d_target_random - d_target_top1  # positive = rerank wins
            target_closer = bool(d_target_top1 < d_wrong_top1)
            method_results[method_name] = {
                "rerank_top1_d_target": d_target_top1,
                "rerank_top1_d_wrong": d_wrong_top1,
                "random_pick_d_target": d_target_random,
                "utility": utility,
                "target_closer": float(target_closer),
            }
        per_pair_metrics[(uid, asin)] = {
            "user_id": uid,
            "asin": asin,
            "n_pool_attr_pass": len(pool_filtered),
            "method_results": method_results,
        }

    log(f"  per_pair_metrics: {len(per_pair_metrics)}")
    log(f"  feat fail: {n_feat_fail}, attr fail: {n_attr_fail}")

    # === Aggregate per method ===
    log("[7] Aggregating per method (8 methods) ...")
    method_aggregates = {}
    for method_name, _, _ in METHODS:
        util_per_pair = {k: v["method_results"][method_name]["utility"]
                         for k, v in per_pair_metrics.items()}
        d_target_top1 = {k: v["method_results"][method_name]["rerank_top1_d_target"]
                         for k, v in per_pair_metrics.items()}
        d_target_random = {k: v["method_results"][method_name]["random_pick_d_target"]
                           for k, v in per_pair_metrics.items()}
        target_closer_per_pair = {k: v["method_results"][method_name]["target_closer"]
                                   for k, v in per_pair_metrics.items()}
        mean_util, ci_util = bootstrap_ci(util_per_pair)
        mean_top1, ci_top1 = bootstrap_ci(d_target_top1)
        mean_random, ci_random = bootstrap_ci(d_target_random)
        mean_target_closer, ci_target_closer = bootstrap_ci(target_closer_per_pair)
        method_aggregates[method_name] = {
            "n_pairs": len(per_pair_metrics),
            "rerank_utility": {
                "mean": mean_util,
                "ci_95": list(ci_util),
                "ci_lower_above_zero": bool(ci_util[0] > 0),
            },
            "rerank_top1_distance": {
                "mean": mean_top1,
                "ci_95": list(ci_top1),
            },
            "random_pick_distance": {
                "mean": mean_random,
                "ci_95": list(ci_random),
            },
            "win_rate_target_closer": {
                "mean": mean_target_closer,
                "ci_95": list(ci_target_closer),
                "ci_lower_above_50pct": bool(ci_target_closer[0] > 0.5),
            },
        }

    # Ranking by rerank_utility mean
    log(f"\n=== Per-method rerank utility ranking ===")
    log(f"{'method':<18} {'util_mean':>12} {'util_ci':>30} {'top1_dist':>12} {'rand_dist':>12} {'target_closer':>12}")
    ranked = sorted(method_aggregates.items(), key=lambda kv: -kv[1]["rerank_utility"]["mean"])
    for rank, (name, agg) in enumerate(ranked, 1):
        util = agg["rerank_utility"]
        ci = util["ci_95"]
        log(f"{name:<18} {util['mean']:>12.4f} "
            f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
            f"{agg['rerank_top1_distance']['mean']:>12.4f} "
            f"{agg['random_pick_distance']['mean']:>12.4f} "
            f"{agg['win_rate_target_closer']['mean']:>12.4f}")

    # === Verdict ===
    n_pass = sum(1 for m in method_aggregates.values()
                 if m["rerank_utility"]["ci_lower_above_zero"])
    log(f"\n=== Summary ===")
    log(f"  Methods with rerank utility CI > 0: {n_pass} / {len(method_aggregates)}")
    log(f"  Best method: {ranked[0][0]} (utility={ranked[0][1]['rerank_utility']['mean']:.4f})")
    log(f"  Worst method: {ranked[-1][0]} (utility={ranked[-1][1]['rerank_utility']['mean']:.4f})")

    # === Save ===
    eval_dict = {
        "n_pairs": len(per_pair_metrics),
        "methods": method_aggregates,
        "ranking": [{"rank": i + 1, "method": name,
                     "rerank_utility_mean": agg["rerank_utility"]["mean"]}
                    for i, (name, agg) in enumerate(ranked)],
        "best_method": ranked[0][0],
        "n_methods_with_significant_utility": n_pass,
        "summary": {
            "n_pairs": len(per_pair_metrics),
            "n_feat_fail": n_feat_fail,
            "n_attr_fail": n_attr_fail,
            "methods_tested": [m[0] for m in METHODS],
        },
        "note": "Phase 10.10.4: feature dimension sweep. 8 methods tested on same N=876. "
                "rerank_utility = random_pick_d_target - rerank_top1_d_target. "
                "Higher = rerank wins more. Best method name in 'best_method'.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for v in per_pair_metrics.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")


if __name__ == "__main__":
    main()