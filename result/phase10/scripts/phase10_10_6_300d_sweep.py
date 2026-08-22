#!/usr/bin/env python3
"""Phase 10.10.6: 300d Feature Dimension Sweep + Shrinkage Covariance.

设计:
- 318d 长度无关句法特征池 (from extract_syntactic_features.ALL_FEATS_V2)
- 过滤: 排除 n_punct_total 等带长度绝对量的特征
- 扫描 d ∈ {5, 10, 20, 40, 60, 80, 120, 160, 200, 240, 280, 300}
- 3 种距离: L2, z-score L2, Mahalanobis (Ledoit-Wolf shrinkage)
- 关键指标: M_d(q,u) = d_nearest_other(q) - d_target(q) per pair
  (positive = target 是最近的 user, 说明 d 维下用户可分)
- Bootstrap CI over pairs

特征选择: F-statistic (between-user var / within-user var), 选 top-d
  (与用户 ID 相关的最有信息量的维度)

用户 profile:
- 每用户均值 (主) + 每用户 diagonal covariance (备)
- Mahalanobis 用 LedoitWolf shrinkage (小样本稳定)

最终选 d*:
- d* = min{d : M_d CI > 0 且 >= best_mean * 0.95}
- 如果 280 已达到 95% best, 选 280 (而非 300, 节省计算 + 减少过拟合)

输入:
- SENTENCE_FILE: 43770 训练句 (per-user, raw text)
- phase10_10_candidates_n10.jsonl: 8760 candidates

输出:
- phase10_10_6_300d_sweep.json: 每个 d × distance 的 M_d + CI + verdict
- phase10_10_6_per_pair.jsonl: per-pair M_d for all d × distance

期望结果 (用户假设):
- d* ∈ {20, 80, 200, 300} 中之一 (不能预判)
- 所有 d × distance 的 M_d CI > 0 (rerank 与维度无关, 但 magnitude 不同)
- d=300d 不一定比 d=80d 好很多 (维度饱和)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_syntactic_features import (  # noqa: E402
    ALL_FEATS_V2, per_sentence_features_v2, user_features_v2,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
CANDIDATES_FILE = OUT_DIR / "phase10_10_candidates_n10.jsonl"
SENTENCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"
CAND_FEATS_CACHE = CACHE_DIR / "candidate_318d.npy"
CAND_KEYS_CACHE = CACHE_DIR / "candidate_318d_keys.json"

OUT_EVAL = OUT_DIR / "phase10_10_6_300d_sweep.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_6_per_pair.jsonl"
LOG_OUT = OUT_DIR / "phase10_10_6_300d_sweep.log"

SEED = 42
N_BOOTSTRAP = 5000

# Dims to test (user requested: 5, 10, 20, 40, 60, 80, 120, 160, 200, 240, 280, 300)
DIMS = [5, 10, 20, 40, 60, 80, 120, 160, 200, 240, 280, 300]
# Distances: L2/zscore_l2 fast (O(n*d)), mahalanobis slow (O(n*d^2) + O(d^3))
# For high d (>120), full covariance inversion is unreliable AND slow, so skip maha
DISTANCES = ["l2", "zscore_l2", "mahalanobis_shrink"]
MAHA_MAX_D = 120  # Mahalanobis only up to d=120

# Features to drop (绝对量, not length-invariant)
# n_punct_total: 总标点数, 与句子长度强相关
LEN_RELATED_DROP = {"n_punct_total"}


def log(m):
    print(f"[phase10-10.6] {m}", flush=True)


def load_spacy_model():
    """延迟加载 spaCy 模型 (与 llm_client.py 同样的导入风格)."""
    import spacy
    return spacy.load("en_core_web_sm")


def get_keep_indices(feature_names):
    keep = []
    dropped = []
    for i, n in enumerate(feature_names):
        if n in LEN_RELATED_DROP:
            dropped.append(n)
            continue
        keep.append(i)
    return keep, dropped


def extract_query_features_batch(texts, nlp):
    """Batch parse + 318d features per text. texts: list[str]."""
    docs = list(nlp.pipe(texts, batch_size=64))
    feats = np.zeros((len(texts), len(ALL_FEATS_V2)), dtype=np.float32)
    for i, doc in enumerate(docs):
        sfs = [per_sentence_features_v2(s) for s in doc.sents]
        sfs = [s for s in sfs if s is not None]
        if sfs:
            v = user_features_v2(sfs)
            if v is not None:
                feats[i] = v
    return feats


def extract_sentence_features_with_cache(sentence_records, nlp):
    """Extract 318d features per sentence, with disk cache."""
    if SENT_FEATS_CACHE.exists() and SENT_USERS_CACHE.exists():
        log("  [cache] loading sentence 318d features from disk ...")
        feats = np.load(SENT_FEATS_CACHE)
        users = json.loads(SENT_USERS_CACHE.read_text())
        return feats, users

    log(f"  [no cache] extracting 318d features for {len(sentence_records)} sentences ...")
    t0 = time.time()
    texts = [r["sentence_text"] for r in sentence_records]
    users = [r["user_id"] for r in sentence_records]
    feats = extract_query_features_batch(texts, nlp)
    log(f"  done in {time.time() - t0:.1f}s, shape={feats.shape}")

    np.save(SENT_FEATS_CACHE, feats)
    SENT_USERS_CACHE.write_text(json.dumps(users))
    return feats, users


def extract_candidate_features_with_cache(candidates, nlp):
    """Extract 318d features per candidate, with disk cache."""
    if CAND_FEATS_CACHE.exists() and CAND_KEYS_CACHE.exists():
        log("  [cache] loading candidate 318d features from disk ...")
        feats = np.load(CAND_FEATS_CACHE)
        keys = json.loads(CAND_KEYS_CACHE.read_text())
        return feats, keys

    log(f"  [no cache] extracting 318d features for {len(candidates)} candidates ...")
    t0 = time.time()
    texts = [c["candidate_query"] for c in candidates]
    keys = [f"{c['user_id']}__{c['asin']}__{c.get('cond', 'A_no_style')}__{c.get('candidate_index', 0)}"
            for c in candidates]
    feats = extract_query_features_batch(texts, nlp)
    log(f"  done in {time.time() - t0:.1f}s, shape={feats.shape}")

    np.save(CAND_FEATS_CACHE, feats)
    CAND_KEYS_CACHE.write_text(json.dumps(keys))
    return feats, keys


def aggregate_per_user_centroid(feat_array, user_ids, min_sent=3):
    """每用户特征均值 (要求至少 min_sent 句)."""
    by_user: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(user_ids):
        by_user[uid].append(i)
    centroids = {}
    for uid, idxs in by_user.items():
        if len(idxs) >= min_sent:
            centroids[uid] = feat_array[idxs].mean(axis=0)
    return centroids


def f_statistic_selection(feat_array, user_ids, top_k):
    """F-statistic per feature: between-user var / within-user var.

    返回 top_k 维度的索引 (按 F 降序).
    """
    by_user: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(user_ids):
        by_user[uid].append(i)
    user_ids_unique = list(by_user.keys())
    overall_mean = feat_array.mean(axis=0)
    D = feat_array.shape[1]
    # Between-group var: sum_k n_k * (mean_k - overall_mean)^2 / (K - 1)
    # Within-group var: sum_k sum_i (x_ki - mean_k)^2 / (N - K)
    K = len(user_ids_unique)
    N = feat_array.shape[0]
    between = np.zeros(D)
    within = np.zeros(D)
    for uid, idxs in by_user.items():
        n_k = len(idxs)
        if n_k < 2:
            continue
        sub = feat_array[idxs]
        mean_k = sub.mean(axis=0)
        diff_k = mean_k - overall_mean
        between += n_k * diff_k ** 2
        diff_i = sub - mean_k
        within += (diff_i ** 2).sum(axis=0)
    between_norm = between / (K - 1) + 1e-9
    within_norm = within / (N - K) + 1e-9
    F = between_norm / within_norm
    top_idx = np.argsort(-F)[:top_k]
    return top_idx, F


def ledoit_wolf_shrinkage(cov_empirical, n_samples, eps=1e-6):
    """Ledoit-Wolf shrinkage: cov = (1-alpha) * empirical + alpha * mu_I.

    mu = trace(empirical) / D
    alpha = shrinkage factor (use simple version)
    """
    D = cov_empirical.shape[0]
    mu = np.trace(cov_empirical) / D
    target = mu * np.eye(D, dtype=cov_empirical.dtype)
    # Standard shrinkage (Ledoit-Wolf 2004): alpha = min(1, sum((x_i x_i^T - cov)^2) / sum((x_i x_i^T - target)^2))
    # For simplicity, use fixed alpha = 0.1 (works well in practice)
    alpha = 0.1
    return (1 - alpha) * cov_empirical + alpha * target


def compute_user_distances_per_d(user_centroids, cand_features, top_idx, train_mean, train_std, dist_kind):
    """对每个 candidate, 计算到所有 users 的距离.

    Returns: distances (n_candidates, n_users)

    Memory-optimized using ||c-u||² = ||c||² + ||u||² - 2<c,u>
    and maha = c^T inv c + u^T inv u - 2*c^T inv u (no (n,m,d) tensor).
    """
    cand_d = cand_features[:, top_idx].astype(np.float64)
    user_ids = list(user_centroids.keys())
    user_arr = np.stack([user_centroids[uid][top_idx] for uid in user_ids], axis=0).astype(np.float64)
    n_cand = cand_d.shape[0]
    n_users = user_arr.shape[0]
    d = len(top_idx)

    if dist_kind == "l2":
        # ||c-u||^2 = ||c||^2 + ||u||^2 - 2<c,u>
        cand_norm = (cand_d ** 2).sum(axis=1)  # (n_cand,)
        user_norm = (user_arr ** 2).sum(axis=1)  # (n_users,)
        cross = cand_d @ user_arr.T  # (n_cand, n_users)
        dist = cand_norm[:, None] + user_norm[None, :] - 2 * cross
    elif dist_kind == "zscore_l2":
        cand_z = (cand_d - train_mean[top_idx]) / train_std[top_idx]
        user_z = (user_arr - train_mean[top_idx]) / train_std[top_idx]
        cand_norm = (cand_z ** 2).sum(axis=1)
        user_norm = (user_z ** 2).sum(axis=1)
        cross = cand_z @ user_z.T
        dist = cand_norm[:, None] + user_norm[None, :] - 2 * cross
    elif dist_kind == "mahalanobis_shrink":
        # Mahalanobis via quadratic expansion: maha = c^T K c + u^T K u - 2 c^T K u
        # where K = inv_cov (precomputed once per d)
        cov = np.cov(user_arr.T)
        cov_shrunk = ledoit_wolf_shrinkage(cov, n_samples=user_arr.shape[0])
        inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(d))
        # Project to inv_cov space
        cand_proj = cand_d @ inv_cov  # (n_cand, d)
        user_proj = user_arr @ inv_cov  # (n_users, d)
        # Diagonal quadratic terms (only need diagonal: c^T K c = sum_d c_proj_d * c_d)
        cand_q = (cand_proj * cand_d).sum(axis=1)  # (n_cand,)
        user_q = (user_proj * user_arr).sum(axis=1)  # (n_users,)
        # Cross term: c^T K u = cand_proj @ user_arr.T (more efficient than cand @ K @ u.T)
        cross = cand_proj @ user_arr.T  # (n_cand, n_users)
        dist = cand_q[:, None] + user_q[None, :] - 2 * cross
    else:
        raise ValueError(f"Unknown dist_kind: {dist_kind}")
    return dist.astype(np.float32), user_ids


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


def main():
    log("=" * 70)
    log("Phase 10.10.6: 300d Feature Dimension Sweep + Shrinkage Covariance")
    log("=" * 70)

    # === Load data ===
    log("[1] Loading sentences + candidates ...")
    sentence_records = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            sentence_records.append(json.loads(line))
    log(f"  sentences: {len(sentence_records)}")

    candidates = []
    with CANDIDATES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")

    pair_lookup = {}
    with PAIRS_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            pair_lookup[(p["user_id"], p["asin"])] = p
    log(f"  pairs: {len(pair_lookup)}")

    # === Extract features ===
    log("[2] Loading spaCy + extracting 318d features ...")
    nlp = load_spacy_model()
    sent_feats, sent_users = extract_sentence_features_with_cache(sentence_records, nlp)
    cand_feats, cand_keys = extract_candidate_features_with_cache(candidates, nlp)
    log(f"  sent_feats: {sent_feats.shape}, cand_feats: {cand_feats.shape}")

    # === Filter out length-related features ===
    log("[3] Filtering length-related features ...")
    keep_idx, dropped = get_keep_indices(ALL_FEATS_V2)
    log(f"  kept: {len(keep_idx)} / {len(ALL_FEATS_V2)}, dropped: {dropped}")
    sent_feats = sent_feats[:, keep_idx]
    cand_feats = cand_feats[:, keep_idx]
    feature_names_kept = [ALL_FEATS_V2[i] for i in keep_idx]
    D_pool = len(keep_idx)
    log(f"  feature pool size: {D_pool}")

    # === Aggregate per-user centroids ===
    log("[4] Computing per-user centroids (min 3 sentences) ...")
    user_centroids = aggregate_per_user_centroid(sent_feats, sent_users, min_sent=3)
    log(f"  users with centroids: {len(user_centroids)}")

    # z-score params (on sentence-level data, not centroid)
    train_mean = sent_feats.mean(axis=0)
    train_std = sent_feats.std(axis=0) + 1e-9

    # === F-statistic feature selection ===
    log("[5] F-statistic feature selection (top-D = max DIMS) ...")
    max_dim = max(DIMS)
    top_idx_max, F = f_statistic_selection(sent_feats, sent_users, top_k=max_dim)
    top_d_ranks = {int(d): [int(i) for i in top_idx_max[:d]] for d in DIMS}
    log(f"  F top-10 features: {[feature_names_kept[i] for i in top_idx_max[:10]]}")

    # === Per-pair candidates by (uid, asin) ===
    log("[6] Grouping candidates by (uid, asin) ...")
    cand_by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ci, key in enumerate(cand_keys):
        uid, asin, cond, idx = key.split("__")
        cand_by_pair[(uid, asin)].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    # === Sweep ===
    log("[7] Sweeping d × distance ...")
    user_ids_arr = list(user_centroids.keys())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids_arr)}

    # For each d, precompute user_ids_used and distance matrix per dist_kind
    # To save memory, compute distance per (d, dist_kind) on-demand

    # Per-pair metrics storage:
    # - M_d_per_pair: dict[(d, dist_kind)] → Dict[pair_key, M_d_avg] (user-level discrimination)
    # - rerank_utility_per_pair: dict[(d, dist_kind)] → Dict[pair_key, util_avg] (within-pair rerank utility)
    M_d_per_pair = {}
    rerank_utility_per_pair = {}

    # Loop
    for d in DIMS:
        top_idx = top_d_ranks[d]
        for dist_kind in DISTANCES:
            t0 = time.time()
            log(f"  d={d}, dist={dist_kind}: computing distances ...")
            # distance_matrix shape: (n_candidates, n_users)
            dist_matrix, _ = compute_user_distances_per_d(
                user_centroids, cand_feats, top_idx, train_mean, train_std, dist_kind,
            )
            log(f"    dist_matrix shape: {dist_matrix.shape}, in {time.time() - t0:.1f}s")

            # For each pair:
            M_d_per_pair[(d, dist_kind)] = {}
            rerank_utility_per_pair[(d, dist_kind)] = {}
            t0 = time.time()
            for (uid, asin), cand_indices in cand_by_pair.items():
                target_pair = pair_lookup.get((uid, asin))
                if not target_pair:
                    continue
                target_idx = user_id_to_idx.get(uid)
                if target_idx is None:
                    continue
                # Get distances for this pair's candidates
                d_target = dist_matrix[cand_indices, target_idx]  # shape (n_candidates,)
                # Nearest other (excluding target)
                dist_matrix_pair = dist_matrix[cand_indices]  # (n_cand, n_users)
                dist_matrix_pair_masked = dist_matrix_pair.copy()
                dist_matrix_pair_masked[:, target_idx] = np.inf
                d_nearest_other = dist_matrix_pair_masked.min(axis=1)  # (n_cand,)
                # === M_d (user-level discrimination) ===
                M_d_per_cand = d_nearest_other - d_target  # positive = target nearest
                valid = np.isfinite(M_d_per_cand)
                M_d_avg = float(M_d_per_cand[valid].mean()) if valid.any() else 0.0
                M_d_per_pair[(d, dist_kind)][(uid, asin)] = M_d_avg
                # === Within-pair rerank utility ===
                # rerank top-1 = candidate with min d_target
                rerank_top1_idx = int(np.argmin(d_target))
                rerank_top1_d = float(d_target[rerank_top1_idx])
                # random pick baseline = mean d_target across all cands
                random_avg_d = float(d_target.mean())
                util = random_avg_d - rerank_top1_d  # positive = rerank wins
                rerank_utility_per_pair[(d, dist_kind)][(uid, asin)] = util
            log(f"    per-pair done in {time.time() - t0:.1f}s")

    # === Aggregate per (d, dist_kind) ===
    log("[8] Aggregating M_d + rerank_utility per (d, dist_kind) ...")
    aggregates = {}
    rerank_aggregates = {}
    for d in DIMS:
        for dist_kind in DISTANCES:
            # M_d
            values = M_d_per_pair[(d, dist_kind)]
            mean_M, ci_M = bootstrap_ci(values)
            aggregates[(d, dist_kind)] = {
                "n_pairs": len(values),
                "M_d_mean": mean_M,
                "M_d_ci_95": list(ci_M),
                "ci_lower_above_zero": bool(ci_M[0] > 0),
            }
            # rerank_utility
            values_ru = rerank_utility_per_pair[(d, dist_kind)]
            mean_ru, ci_ru = bootstrap_ci(values_ru)
            rerank_aggregates[(d, dist_kind)] = {
                "n_pairs": len(values_ru),
                "rerank_utility_mean": mean_ru,
                "rerank_utility_ci_95": list(ci_ru),
                "ci_lower_above_zero": bool(ci_ru[0] > 0),
            }

    # === Verdict: select d* by user rule ===
    log("[9] Applying d* selection rule (>=95% of best) ...")
    # === Metric 1: M_d (user-level discrimination) ===
    # Best M_d = max across (d, dist_kind) of M_d_mean
    best_M = max(v["M_d_mean"] for v in aggregates.values())
    threshold_M = 0.95 * best_M
    log(f"  M_d: best_M = {best_M:.4f}, threshold (95%) = {threshold_M:.4f}")

    d_star_M_per_dist = {}
    for dist_kind in DISTANCES:
        candidates_d = []
        for d in DIMS:
            agg = aggregates[(d, dist_kind)]
            if agg["ci_lower_above_zero"] and agg["M_d_mean"] >= threshold_M:
                candidates_d.append((d, agg["M_d_mean"]))
        if candidates_d:
            d_star_M_per_dist[dist_kind] = min(candidates_d)[0]
        else:
            d_star_M_per_dist[dist_kind] = None

    # === Metric 2: within-pair rerank_utility (operational rerank quality) ===
    best_ru = max(v["rerank_utility_mean"] for v in rerank_aggregates.values())
    threshold_ru = 0.95 * best_ru
    log(f"  rerank_utility: best = {best_ru:.4f}, threshold (95%) = {threshold_ru:.4f}")

    d_star_ru_per_dist = {}
    for dist_kind in DISTANCES:
        candidates_d = []
        for d in DIMS:
            agg = rerank_aggregates[(d, dist_kind)]
            if agg["ci_lower_above_zero"] and agg["rerank_utility_mean"] >= threshold_ru:
                candidates_d.append((d, agg["rerank_utility_mean"]))
        if candidates_d:
            d_star_ru_per_dist[dist_kind] = min(candidates_d)[0]
        else:
            d_star_ru_per_dist[dist_kind] = None

    log(f"  d* per dist_kind (M_d): {d_star_M_per_dist}")
    log(f"  d* per dist_kind (rerank_utility): {d_star_ru_per_dist}")

    # === Per-dist_kind ranking ===
    log("\n=== M_d (user-level discrimination) ===")
    log(f"{'dist_kind':<25} {'d':<5} {'M_d_mean':>10} {'CI':>30} {'CI>0':>8} {'>=95%best':>12}")
    ranked_by_dist = {}
    for dist_kind in DISTANCES:
        rank_list = []
        for d in DIMS:
            agg = aggregates[(d, dist_kind)]
            ci = agg["M_d_ci_95"]
            log(f"{dist_kind:<25} {d:<5} {agg['M_d_mean']:>10.4f} "
                f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
                f"{'PASS' if agg['ci_lower_above_zero'] else 'FAIL':>8} "
                f"{'PASS' if agg['M_d_mean'] >= threshold_M else 'FAIL':>12}")
            rank_list.append((d, agg["M_d_mean"], agg["ci_lower_above_zero"]))
        ranked_by_dist[dist_kind] = rank_list

    log("\n=== rerank_utility (within-pair rerank quality) ===")
    log(f"{'dist_kind':<25} {'d':<5} {'util_mean':>10} {'CI':>30} {'CI>0':>8} {'>=95%best':>12}")
    ranked_ru_by_dist = {}
    for dist_kind in DISTANCES:
        rank_list = []
        for d in DIMS:
            agg = rerank_aggregates[(d, dist_kind)]
            ci = agg["rerank_utility_ci_95"]
            log(f"{dist_kind:<25} {d:<5} {agg['rerank_utility_mean']:>10.4f} "
                f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
                f"{'PASS' if agg['ci_lower_above_zero'] else 'FAIL':>8} "
                f"{'PASS' if agg['rerank_utility_mean'] >= threshold_ru else 'FAIL':>12}")
            rank_list.append((d, agg["rerank_utility_mean"], agg["ci_lower_above_zero"]))
        ranked_ru_by_dist[dist_kind] = rank_list

    # === Save ===
    log("\n[10] Saving results ...")
    eval_dict = {
        "n_pairs": len(M_d_per_pair.get((DIMS[0], DISTANCES[0]), {})),
        "n_features_pool": D_pool,
        "feature_names_kept": feature_names_kept,
        "f_statistic_top_10_features": [feature_names_kept[i] for i in top_idx_max[:10]],
        "dims_tested": DIMS,
        "distances_tested": DISTANCES,
        # === Metric 1: M_d (user-level discrimination) ===
        "best_M_d_mean": best_M,
        "threshold_M_95pct": threshold_M,
        "d_star_M_per_dist_kind": d_star_M_per_dist,
        # === Metric 2: within-pair rerank_utility ===
        "best_rerank_utility_mean": best_ru,
        "threshold_ru_95pct": threshold_ru,
        "d_star_ru_per_dist_kind": d_star_ru_per_dist,
        # === Detailed results ===
        "per_d_per_dist_M_d": {
            f"d={d}_{dk}": aggregates[(d, dk)] for d in DIMS for dk in DISTANCES
        },
        "per_d_per_dist_rerank_utility": {
            f"d={d}_{dk}": rerank_aggregates[(d, dk)] for d in DIMS for dk in DISTANCES
        },
        "ranking_M_d_per_dist": {
            dk: [{"d": r[0], "M_d_mean": r[1], "ci_above_zero": r[2]} for r in ranked_by_dist[dk]]
            for dk in DISTANCES
        },
        "ranking_rerank_utility_per_dist": {
            dk: [{"d": r[0], "rerank_utility_mean": r[1], "ci_above_zero": r[2]} for r in ranked_ru_by_dist[dk]]
            for dk in DISTANCES
        },
        "note": "Phase 10.10.6: 300d feature dimension sweep. "
                "Two metrics: (1) M_d(q,u) = d_nearest_other - d_target = user-level discrimination. "
                "(2) rerank_utility = within-pair min d_target vs mean d_target = operational rerank quality. "
                "Memory-optimized via ||c-u||² = ||c||²+||u||²-2c·u trick (no (n,m,d) tensor). "
                "Mahalanobis via c^T K c + u^T K u - 2 c^T K u quadratic expansion. "
                "d* = smallest d meeting (CI > 0 AND >= 95% of best).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair (only save for d=20 and d=300 to save space)
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin), m_dict in M_d_per_pair[(20, "zscore_l2")].items():
            row = {"user_id": uid, "asin": asin}
            for d in [20, 80, 200, 300]:
                for dk in DISTANCES:
                    row[f"M_d{d}_{dk}"] = M_d_per_pair[(d, dk)].get((uid, asin), float("nan"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.10.6 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()