"""Stage 4 v6m — Strict Personalized Alignment Selection (用户指令 2026-08-28).

用户原话:
> 选到"在 cohort 里最像目标用户",并且"绝对上也确实落在目标用户正常句法范围内"的 Query。
>
> 严格候选集:
>   Q_strict = {q: M(q,u) > 0 AND d_self(q,u) ≤ R_95}
> 选 M 最大的那条:
>   q*_u = argmax_q∈Q_strict M(q,u)
>
> 如果没有任何 Query 同时满足两个条件,就标记为 no_strict_candidate,
> 而不是硬选一个"相对最好但实际上并不像用户"的句子。

两步 gate + 一步排序:
  1. M(q,u) > 0              (target user Rank@1 in cohort)
  2. d_self(q,u) ≤ R_95       (query 落在真实用户历史 95% 距离内)
  3. argmax M under Q_strict

R_95 = 11.308 from Stage 5C percentile analysis on 298740 real history
sentences (whitened L2 to user-mean).

复用 K=200 pool + F3_CoreStruct + PCA48 + whitening 已有 infrastructure;
不重新 generate, 只重 selection。

输出:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_v6m.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_v6m_stats.json
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    PCA_DIM, PCA_SEED, log,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "select_query"))
from syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, select_feature_names, fit_pca,
)
from syntax_subspace_select import apply_feature_subset  # noqa: E402

# User instruction 2026-08-28: strict alignment threshold
R_95_PERCENTILE = 11.308  # from Stage 5C, real-history whitened L2 P95

POOL_IN_LOCAL = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json"
FEAT_CACHE_LOCAL = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz"
# 主 pipeline 已切到 v6m (用户指令 2026-08-28): 直接覆盖 canonical paths,
# 让 Stage 5 retrieval 默认读取 v6m 数据。备份在 *_v6k_main_backup.json。
SEL_OUT = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json"
STATS_OUT = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_stats.json"


def load_query_features(path: str) -> dict[str, np.ndarray]:
    """Load sha1(text) -> 182d features from sentence features cache.
    用户指令 2026-08-29: 容错 zlib + JSON 错误 (chunked append 模式下
    cache 文件可能被强杀在 gzip block 中间).
    """
    import hashlib
    log(f"  Loading sentence features cache {path} ...")
    feat_lookup: dict[str, np.ndarray] = {}
    n_header = 0
    n_skip = 0
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    n_header += 1
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    # 末尾 line 可能被截断 (chunked append + kill)
                    n_skip += 1
                    continue
                if "k" in e and "v" in e:
                    feat_lookup[e["k"]] = np.array(list(e["v"].values()), dtype=np.float64)
    except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
        log(f"  ⚠ cache gzip stream truncated ({type(exc).__name__}), "
            f"loaded {len(feat_lookup)} entries + skipped {n_skip} corrupt")
    log(f"  feat_lookup (sha1 -> 182d) size: {len(feat_lookup)}, header lines: {n_header}")
    return feat_lookup


def sha1_of(text: str) -> str:
    """SHA1 of lowercased text (matches stage7b_query_features cache key format)."""
    import hashlib
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def main():
    log_start = time.time()
    log("=== Stage 4 v6m: Strict Personalized Alignment Selection ===")
    log(f"  R_95 (hard threshold) = {R_95_PERCENTILE:.3f}")
    log(f"  Gate: M(q,u) > 0 AND d_self(q,u) ≤ R_95")
    log(f"  Selection: argmax M under gate (strict personalized subset)")

    # ---- 1. Load pool K=200 ----
    log("\n=== 1. Loading K=200 pool ===")
    pool_data = json.load(open(POOL_IN_LOCAL))
    asin_pool: dict[str, list[dict]] = pool_data["pools"]
    log(f"  pool ASINs: {len(asin_pool)}")
    log(f"  total queries: {pool_data['n_total_strict']}")

    # ---- 2. Load F3 + StandardScaler + PCA48 + whitening ----
    log("\n=== 2. Loading _syntax_subspace_prepare() for F3_CoreStruct ===")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from syntax_subspace_utils import _syntax_subspace_prepare  # noqa: E402
    P = _syntax_subspace_prepare()
    X = P["X"]; all_fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]; user_to_indices = P["user_to_indices"]

    cfg = FEATURE_SUBSETS["F3_CoreStruct"]
    fnames = select_feature_names(all_fnames, cfg["exclude_prefixes"], cfg["exclude_exact"])
    col_idx = [all_fnames.index(n) for n in fnames]
    ss = StandardScaler()
    ss.fit(X[train_idx][:, col_idx])
    X_sub_scaled = ss.transform(X[:, col_idx])
    pca, sqrt_lambda = fit_pca(X_sub_scaled[train_idx], PCA_DIM, PCA_SEED)
    log(f"  F3_CoreStruct: {len(fnames)} features, PCA{PCA_DIM}: cumvar={pca.explained_variance_ratio_.sum():.4f}")

    # ---- 3. Compute whitened user-means ----
    log("\n=== 3. Whitened user-means μ̃_u ===")
    Z_all = pca.transform(X_sub_scaled) / sqrt_lambda
    user_z_white_means = {}
    for uid, idx in user_to_indices.items():
        if len(idx) < 2:
            continue
        user_z_white_means[uid] = Z_all[idx].mean(axis=0)
    log(f"  users with ≥2 sents: {len(user_z_white_means)}")

    # ---- 4. Load user_gaussians (for source/n_reviews fields) ----
    log("\n=== 4. Loading user_gaussians (source / n_reviews metadata) ===")
    users_gauss_path = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json"
    gauss_data = json.load(open(users_gauss_path))
    users_gauss = gauss_data["users"]
    log(f"  users_gauss: {len(users_gauss)} entries")

    # ---- 5. Load ASINs with cohort users_sampled ----
    log("\n=== 5. Loading stage8_5_asins.json ===")
    asin_data_full = json.load(open(ASINS_IN))["asins"]
    asin_data = [a for a in asin_data_full if a["asin"] in asin_pool]
    log(f"  ASINs with K=200 pool: {len(asin_data)} / {len(asin_data_full)}")

    # ---- 6. Load query features cache (sha1 -> 182d) ----
    log("\n=== 6. Loading sentence features cache (sha1 -> 182d) ===")
    feat_lookup = load_query_features(FEAT_CACHE_LOCAL)

    # ---- 7. Selection loop ----
    log("\n=== 7. Strict personalized selection (per asin, per user) ===")
    selection_entries = []
    n_no_pool = 0
    n_no_pool_for_user = 0
    n_no_user_for_asin = 0          # 用户指令 2026-08-29: 初始化, 否则 n_users==0 分支 raise UnboundLocalError
    n_l2_margin_max_v6k_compat = 0  # argmax M among R_99 gate (v6k legacy)
    n_strict_personalized = 0       # NEW: M>0 AND d_self ≤ R_95
    n_no_strict_candidate = 0      # NEW: no candidate passes both gates
    n_missing_gauss = 0             # user not in user_gauss metadata table

    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        strict_pool = asin_pool[asin]  # already filtered by Stage 1 strict
        if not strict_pool:
            for uid in entry["users_sampled"]:
                if uid not in users_gauss:
                    n_missing_gauss += 1
                    meta_source, meta_n_reviews = None, None
                else:
                    meta_source = users_gauss[uid]["source"]
                    meta_n_reviews = users_gauss[uid]["n_reviews"]
                n_no_pool_for_user += 1
                selection_entries.append({
                    "asin": asin, "user_id": uid, "attrs_used": attrs,
                    "selection_method": "no_pool", "selected": None,
                    "selected_distance": None, "selected_margin": None,
                    "n_candidates": 0,
                    "user_source": meta_source,
                    "n_reviews": meta_n_reviews,
                })
            continue
        # Collect user cohort
        asin_users = []
        asin_users_metadata_missing = []
        for uid in entry["users_sampled"]:
            if uid not in user_z_white_means:
                # user has <2 sentences; skip (not eligible for M computation)
                asin_users_metadata_missing.append(uid)
                continue
            mu_white = user_z_white_means[uid]
            if uid in users_gauss:
                meta_source = users_gauss[uid]["source"]
                meta_n_reviews = users_gauss[uid]["n_reviews"]
            else:
                # user_gauss only has 294 entries (high-n_reviews subset); most users
                # sampled per-ASIN are not in it — derive metadata heuristically.
                meta_source = "non_gaussian_subset"
                meta_n_reviews = None
                n_missing_gauss += 1
            asin_users.append((uid, mu_white, {"source": meta_source, "n_reviews": meta_n_reviews}))

        n_users = len(asin_users)
        if n_users == 0:
            # 边界: 该 ASIN 没有合格用户 (全部 <2 句子)。记录 no_user_for_asin 后 continue
            n_no_user_for_asin += 1
            log(f"  no eligible users for asin {asin}, skip")
            continue
        n_queries = len(strict_pool)

        # ---- 7a. Compute whitened L2 matrix ||z̃_q − μ̃_u|| ----
        # Get query whitened features via sha1(text) lookup
        Z_q_list = []
        valid_q_idx = []
        for qi, q_entry in enumerate(strict_pool):
            text = q_entry.get("query", "")
            sha = sha1_of(text)
            full_feat = feat_lookup.get(sha)
            if full_feat is None or len(full_feat) != 182:
                continue
            # Subset to F3_CoreStruct columns
            x_sub = full_feat[col_idx]
            x_sub_scaled = ss.transform(x_sub.reshape(1, -1))
            z_white = pca.transform(x_sub_scaled) / sqrt_lambda
            Z_q_list.append(z_white[0])
            valid_q_idx.append(qi)
        if not Z_q_list:
            for uid, mu_white, gauss_info in asin_users:
                n_no_pool_for_user += 1
                selection_entries.append({
                    "asin": asin, "user_id": uid, "attrs_used": attrs,
                    "selection_method": "no_query_features", "selected": None,
                    "selected_distance": None, "selected_margin": None,
                    "n_candidates": 0,
                    "user_source": gauss_info["source"],
                    "n_reviews": gauss_info["n_reviews"],
                })
            continue
        Z_q_arr = np.stack(Z_q_list, axis=0)  # (n_valid_q, 48)
        n_valid_q = Z_q_arr.shape[0]
        cohort_mus = np.stack([au[1] for au in asin_users], axis=0)  # (n_users, 48)
        # L2 distance matrix: (n_users, n_valid_q)
        l2_white_matrix = np.linalg.norm(
            cohort_mus[:, None, :] - Z_q_arr[None, :, :], axis=2)

        # ---- 7b. Compute M_L2 per (user, query) ----
        # d_self = l2_white_matrix[ui, qi]
        # d_other = min over v != ui of l2_white_matrix[v, qi]
        sorted_l2 = np.sort(l2_white_matrix, axis=0)
        # is_self_argmin: True if self is the closest cohort user for this query
        is_self_argmin = (np.argmin(l2_white_matrix, axis=0) == np.arange(n_users)[:, None])  # noqa: E501
        # Actually, argmin per column (per query): the cohort user with min L2 for q
        # If self is in cohort and rank-1 → use sorted_l2[1] (second smallest) as d_other
        # If self is NOT rank-1 → use sorted_l2[0] (smallest) as d_other (other user is closer)
        col_argmin = np.argmin(l2_white_matrix, axis=0)  # (n_valid_q,)
        # For each query qi: d_other_l2 = smallest L2 across cohort users that is NOT the self
        # We need to know, per (user, query), the minimum across v != user
        # Simpler: compute per query qi the cohort L2 to all users, then for user ui
        # take min over v != ui.
        # Vectorized: for each (ui, qi), find min over cohort[L][qi] for L != ui
        # Use argpartition per qi for speed
        d_other_l2 = np.zeros((n_users, n_valid_q))
        for qi in range(n_valid_q):
            l2_q = l2_white_matrix[:, qi]  # (n_users,)
            # For each user ui, d_other is min over v != ui of l2_q[v]
            # Use argmin excluding ui: get second-smallest if self is smallest, else smallest
            sorted_for_q = np.sort(l2_q)
            is_self_min = (l2_q == sorted_for_q[0])
            # For each ui: if ui is self-min → sorted_for_q[1], else → sorted_for_q[0]
            # 边界: cohort 只有 1 个 user 时 sorted_for_q[1] 不存在, 用 sorted_for_q[0] (== self 距离) → M=0
            if len(sorted_for_q) >= 2:
                d_other_l2[:, qi] = np.where(is_self_min, sorted_for_q[1], sorted_for_q[0])
            else:
                # 单 user cohort: M=0 (no other user to compare)
                d_other_l2[:, qi] = sorted_for_q[0]
        M_L2 = d_other_l2 - l2_white_matrix  # (n_users, n_valid_q)

        # ---- 7c. NEW GATE: M>0 AND d_self ≤ R_95 ----
        gate_strict = (M_L2 > 0) & (l2_white_matrix <= R_95_PERCENTILE)

        # ---- 7d. Per-user selection ----
        for ui, (uid, mu_white, gauss_info) in enumerate(asin_users):
            source = gauss_info["source"]
            n_reviews = gauss_info["n_reviews"]
            # `gauss_info` is now a small dict with source/n_reviews from local lookup; safe.

            cand_strict_mask = gate_strict[ui]
            if not cand_strict_mask.any():
                # No candidate passes both gates → mark no_strict_candidate
                # DO NOT fallback to L2-margin-max or nearest
                best_qi = int(np.argmin(l2_white_matrix[ui]))
                best_dist = float(l2_white_matrix[ui][best_qi])
                best_margin = float(M_L2[ui][best_qi])
                method = "no_strict_candidate"
                n_no_strict_candidate += 1
                selected_q = None
            else:
                avail_M = np.where(cand_strict_mask, M_L2[ui], -np.inf)
                best_qi = int(np.argmax(avail_M))
                best_dist = float(l2_white_matrix[ui][best_qi])
                best_margin = float(M_L2[ui][best_qi])
                method = "strict_personalized_l2_margin_max"
                n_strict_personalized += 1
                # Map best_qi back to original strict_pool index
                orig_qi = valid_q_idx[best_qi]
                selected_q = strict_pool[orig_qi]

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "selected_distance": best_dist,
                "selected_margin": best_margin,
                "n_candidates": int(cand_strict_mask.sum()),
                "n_pool_candidates": n_queries,
                "user_source": source,
                "n_reviews": n_reviews,
            })

    # ---- 8. Save ----
    log("\n=== 8. Saving selection ===")
    Path(SEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(SEL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6m: Strict Personalized Alignment Selection. "
                                "Two-gate: M(q,u) > 0 (target user Rank@1) AND "
                                "d_self(q,u) ≤ R_95=11.308 (query lies within "
                                "real-history 95% radius). Then argmax M under gate. "
                                "If no candidate passes both, mark 'no_strict_candidate' "
                                "WITHOUT fallback to L2-margin-max."),
                "feature_subset": "F3_CoreStruct",
                "pca_dim": PCA_DIM,
                "R_95_threshold": R_95_PERCENTILE,
                "gate_conditions": ["M > 0", f"d_self <= R_95={R_95_PERCENTILE}"],
                "selection_logic": "argmax M under (M > 0 AND d_self ≤ R_95)",
                "no_strict_candidate_handling": "no fallback — mark as no_strict_candidate",
            },
            "n_entries": len(selection_entries),
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SEL_OUT}")

    # ---- 9. Stats ----
    by_method = collections.Counter(e["selection_method"] for e in selection_entries)
    log(f"\n  Selection method counts:")
    for m, n in by_method.most_common():
        log(f"    {m}: {n}")

    selected_entries = [e for e in selection_entries
                        if e["selection_method"] == "strict_personalized_l2_margin_max"]
    no_strict = [e for e in selection_entries if e["selection_method"] == "no_strict_candidate"]

    log("\n  Stats on strict_personalized selected (n={}):".format(len(selected_entries)))
    if selected_entries:
        sel_d = np.array([e["selected_distance"] for e in selected_entries])
        sel_m = np.array([e["selected_margin"] for e in selected_entries])
        sel_n_cands = np.array([e["n_candidates"] for e in selected_entries])
        log(f"    d_self mean = {sel_d.mean():.3f}, median = {np.median(sel_d):.3f}")
        log(f"    M mean = {sel_m.mean():.3f}, median = {np.median(sel_m):.3f}")
        log(f"    n_candidates (in Q_strict) mean = {sel_n_cands.mean():.1f}, "
            f"median = {np.median(sel_n_cands):.1f}")

    # Save stats
    stats_data = {
        "n_pairs": len(selection_entries),
        "n_missing_user_gauss_metadata": n_missing_gauss,
        "n_no_user_for_asin": n_no_user_for_asin,
        "by_selection_method": dict(by_method),
        "strict_personalized": {
            "n": len(selected_entries),
            "selected_distance_mean": float(np.mean([e["selected_distance"] for e in selected_entries])) if selected_entries else None,
            "selected_distance_median": float(np.median([e["selected_distance"] for e in selected_entries])) if selected_entries else None,
            "margin_mean": float(np.mean([e["selected_margin"] for e in selected_entries])) if selected_entries else None,
            "margin_median": float(np.median([e["selected_margin"] for e in selected_entries])) if selected_entries else None,
            "n_candidates_in_strict_subset_mean": float(np.mean([e["n_candidates"] for e in selected_entries])) if selected_entries else None,
        },
        "no_strict_candidate": {
            "n": len(no_strict),
            "fraction": float(len(no_strict) / len(selection_entries)) if selection_entries else 0,
            "selected_distance_mean_when_no_strict": float(np.mean([e["selected_distance"] for e in no_strict])) if no_strict else None,
        },
    }
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {STATS_OUT}")
    log(f"\n=== Stage 4 v6m complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()