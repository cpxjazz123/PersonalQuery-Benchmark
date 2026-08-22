#!/usr/bin/env python3
"""Phase 10.10.7: Cross-user separation validation.

用户反馈: 当前验证 (10.10.2/3/4/5/6) 只看了 d_target 是否最小 (within-pair rerank utility),
但没看候选是否同时远离其他用户 (cross-user separation).

设计核心问题:
- rerank utility 选出来的候选, 是否确实比其他候选更远离 wrong users?
- 还是说只是"靠近 target"是局部最优, 但实际对所有 user 都接近 (无区分能力)?

新增指标 (per pair, per (d, dist_kind)):
1. **Gap_target_vs_wrong_per_cand**: 单个候选的 d_target - mean(d_wrong_5_users)
   - 应该 > 0 (候选靠近 target, 远离 wrong)
2. **Gap_target_vs_wrong_mean**: 整对平均 = mean over candidates of (1)
3. **Direction_win_rate**: pair 内 % of candidates where d_target < d_wrong
   - 应该接近 100%
4. **Target_rank_percentile**: target user 在所有用户中的距离排名百分位
   - 应该接近 0% (target 排名靠前)

评估:
- 对每个 (d, dist_kind), 36 个组合, 计算 bootstrap CI
- 验证 Gap > 0 AND direction_win_rate > 50% (实际应该接近 100%)
- PASS 条件: 三个指标的 CI 都跨过有意义边界 (Gap CI > 0, win_rate CI > 50%, rank CI < 50%)

复用 10.10.6 的:
- 候选 features cache
- 用户 centroids
- 距离计算函数 (l2 / zscore_l2 / mahalanobis_shrink)
- F-statistic top-d indices
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

# === Reuse helpers from 10.10.6 ===
sys.path.insert(0, str(Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")))
from phase10_10_6_300d_sweep import (  # noqa: E402
    ALL_FEATS_V2, per_sentence_features_v2, user_features_v2,
    LEN_RELATED_DROP, load_spacy_model, get_keep_indices,
    extract_sentence_features_with_cache, extract_candidate_features_with_cache,
    aggregate_per_user_centroid, f_statistic_selection, ledoit_wolf_shrinkage,
    compute_user_distances_per_d, bootstrap_ci, log,
    SENTENCE_FILE, CANDIDATES_FILE, PAIRS_FILE,
    SENT_FEATS_CACHE, SENT_USERS_CACHE, CAND_FEATS_CACHE, CAND_KEYS_CACHE,
    DIMS, DISTANCES, MAHA_MAX_D,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_EVAL = OUT_DIR / "phase10_10_7_cross_user_separation.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_7_per_pair.jsonl"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5  # K=5 wrong users per pair


def main():
    log("=" * 70)
    log("Phase 10.10.7: Cross-user Separation Validation")
    log("=" * 70)

    # === Load data (same as 10.10.6) ===
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

    # === Extract features (cached from 10.10.6) ===
    log("[2] Loading spaCy + extracting 318d features (cached) ...")
    nlp = load_spacy_model()
    sent_feats, sent_users = extract_sentence_features_with_cache(sentence_records, nlp)
    cand_feats, cand_keys = extract_candidate_features_with_cache(candidates, nlp)
    log(f"  sent_feats: {sent_feats.shape}, cand_feats: {cand_feats.shape}")

    # === Filter length-related ===
    log("[3] Filtering length-related features ...")
    keep_idx, dropped = get_keep_indices(ALL_FEATS_V2)
    sent_feats = sent_feats[:, keep_idx]
    cand_feats = cand_feats[:, keep_idx]
    feature_names_kept = [ALL_FEATS_V2[i] for i in keep_idx]
    D_pool = len(keep_idx)
    log(f"  feature pool size: {D_pool}")

    # === Per-user centroids ===
    log("[4] Computing per-user centroids ...")
    user_centroids = aggregate_per_user_centroid(sent_feats, sent_users, min_sent=3)
    log(f"  users with centroids: {len(user_centroids)}")

    train_mean = sent_feats.mean(axis=0)
    train_std = sent_feats.std(axis=0) + 1e-9

    # === F-statistic feature selection ===
    log("[5] F-statistic feature selection (top-300) ...")
    max_dim = max(DIMS)
    top_idx_max, F = f_statistic_selection(sent_feats, sent_users, top_k=max_dim)
    top_d_ranks = {int(d): [int(i) for i in top_idx_max[:d]] for d in DIMS}

    # === Group candidates by (uid, asin) ===
    log("[6] Grouping candidates by (uid, asin) ...")
    cand_by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ci, key in enumerate(cand_keys):
        uid, asin, cond, idx = key.split("__")
        cand_by_pair[(uid, asin)].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    user_ids_arr = list(user_centroids.keys())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids_arr)}
    rng = np.random.default_rng(SEED)

    # Pre-sample wrong user sets per pair (FIXED seed for reproducibility)
    log(f"[6.5] Pre-sampling {N_WRONG_USERS_PER_PAIR} wrong users per pair ...")
    wrong_users_per_pair = {}
    n_users = len(user_ids_arr)
    for (uid, asin) in cand_by_pair.keys():
        target_idx_global = user_id_to_idx.get(uid)
        if target_idx_global is None:
            continue
        # Sample K wrong users from population (excluding target)
        all_indices = np.arange(n_users)
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx_global] = False
        wrong_idx_global = rng.choice(all_indices[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)
        wrong_users_per_pair[(uid, asin)] = wrong_idx_global

    # === Per-pair metrics storage ===
    # (1) gap_target_vs_wrong_per_pair: mean over candidates of (d_target - mean(d_wrong))
    gap_per_pair = {}
    # (2) direction_win_rate_per_pair: % of candidates where d_target < mean(d_wrong)
    win_rate_per_pair = {}
    # (3) target_rank_percentile_per_pair: target's distance rank / n_users (lower = better)
    target_rank_per_pair = {}

    log("[7] Sweeping d × distance ...")
    for d in DIMS:
        top_idx = top_d_ranks[d]
        for dist_kind in DISTANCES:
            t0 = time.time()
            log(f"  d={d}, dist={dist_kind}: computing distances ...")
            dist_matrix, _ = compute_user_distances_per_d(
                user_centroids, cand_feats, top_idx, train_mean, train_std, dist_kind,
            )
            log(f"    dist_matrix shape: {dist_matrix.shape}, in {time.time() - t0:.1f}s")

            gap_per_pair[(d, dist_kind)] = {}
            win_rate_per_pair[(d, dist_kind)] = {}
            target_rank_per_pair[(d, dist_kind)] = {}

            t0 = time.time()
            for (uid, asin), cand_indices in cand_by_pair.items():
                target_idx = user_id_to_idx.get(uid)
                if target_idx is None:
                    continue
                wrong_idx = wrong_users_per_pair.get((uid, asin))
                if wrong_idx is None:
                    continue
                # Get distances for this pair's candidates to all users
                dist_pair = dist_matrix[cand_indices]  # (n_cand, n_users)
                d_target = dist_pair[:, target_idx]  # (n_cand,)
                d_wrong = dist_pair[:, wrong_idx]  # (n_cand, K=5)
                # === (1) Gap: per-cand (d_target - mean(d_wrong)), then mean over cands ===
                per_cand_gap = d_target - d_wrong.mean(axis=1)  # (n_cand,)
                gap_per_pair[(d, dist_kind)][(uid, asin)] = float(per_cand_gap.mean())
                # === (2) Win rate: % of cands where d_target < mean(d_wrong) ===
                win_rate_per_pair[(d, dist_kind)][(uid, asin)] = float(
                    (d_target < d_wrong.mean(axis=1)).mean()
                )
                # === (3) Target rank: for each cand, rank of target among all users ===
                # Lower rank = target is closer
                ranks = np.argsort(np.argsort(dist_pair, axis=1), axis=1)  # (n_cand, n_users)
                target_ranks = ranks[:, target_idx]  # (n_cand,) in [0, n_users-1]
                # Convert to percentile (lower = better, target ranks high)
                target_rank_percentile = float(target_ranks.mean()) / n_users
                target_rank_per_pair[(d, dist_kind)][(uid, asin)] = target_rank_percentile
            log(f"    per-pair done in {time.time() - t0:.1f}s")

    # === Aggregate ===
    log("[8] Aggregating cross-user metrics ...")
    aggregates = {}
    for d in DIMS:
        for dist_kind in DISTANCES:
            agg = {}
            # (1) Gap
            gap_vals = gap_per_pair[(d, dist_kind)]
            m_gap, ci_gap = bootstrap_ci(gap_vals)
            agg["gap_target_vs_wrong_mean"] = m_gap
            agg["gap_target_vs_wrong_ci_95"] = list(ci_gap)
            agg["gap_ci_above_zero"] = bool(ci_gap[0] > 0)
            # (2) Win rate
            wr_vals = win_rate_per_pair[(d, dist_kind)]
            m_wr, ci_wr = bootstrap_ci(wr_vals)
            agg["win_rate_mean"] = m_wr
            agg["win_rate_ci_95"] = list(ci_wr)
            agg["win_rate_ci_above_50pct"] = bool(ci_wr[0] > 0.5)
            # (3) Target rank percentile
            tr_vals = target_rank_per_pair[(d, dist_kind)]
            m_tr, ci_tr = bootstrap_ci(tr_vals)
            agg["target_rank_percentile_mean"] = m_tr
            agg["target_rank_percentile_ci_95"] = list(ci_tr)
            agg["target_rank_ci_below_50pct"] = bool(ci_tr[1] < 0.5)
            # Overall PASS: gap > 0 AND win_rate > 50% AND target_rank < 50%
            agg["PASS_all"] = (
                agg["gap_ci_above_zero"]
                and agg["win_rate_ci_above_50pct"]
                and agg["target_rank_ci_below_50pct"]
            )
            aggregates[(d, dist_kind)] = agg

    # === Verdict: select d* per metric ===
    log("[9] Selecting d* per metric ...")
    # d* per dist_kind for each metric
    d_star_gap_per_dist = {}
    d_star_wr_per_dist = {}
    d_star_rank_per_dist = {}

    for dist_kind in DISTANCES:
        # Gap: maximize gap_mean (CI > 0)
        candidates_gap = [
            (d, aggregates[(d, dist_kind)]["gap_target_vs_wrong_mean"])
            for d in DIMS
            if aggregates[(d, dist_kind)]["gap_ci_above_zero"]
        ]
        if candidates_gap:
            best_gap = max(g for _, g in candidates_gap)
            threshold = 0.95 * best_gap
            d_star_gap_per_dist[dist_kind] = min(
                d for d, g in candidates_gap if g >= threshold
            )
        else:
            d_star_gap_per_dist[dist_kind] = None

        # Win rate: maximize win_rate_mean (CI > 50%)
        candidates_wr = [
            (d, aggregates[(d, dist_kind)]["win_rate_mean"])
            for d in DIMS
            if aggregates[(d, dist_kind)]["win_rate_ci_above_50pct"]
        ]
        if candidates_wr:
            best_wr = max(w for _, w in candidates_wr)
            threshold = 0.95 * best_wr
            d_star_wr_per_dist[dist_kind] = min(
                d for d, w in candidates_wr if w >= threshold
            )
        else:
            d_star_wr_per_dist[dist_kind] = None

        # Rank: minimize target_rank_percentile (CI < 50%)
        candidates_rank = [
            (d, aggregates[(d, dist_kind)]["target_rank_percentile_mean"])
            for d in DIMS
            if aggregates[(d, dist_kind)]["target_rank_ci_below_50pct"]
        ]
        if candidates_rank:
            best_rank = min(r for _, r in candidates_rank)
            threshold = 1.05 * best_rank  # within 5% of best
            d_star_rank_per_dist[dist_kind] = min(
                d for d, r in candidates_rank if r <= threshold
            )
        else:
            d_star_rank_per_dist[dist_kind] = None

    log(f"  d* per dist (gap > 0, 95% best): {d_star_gap_per_dist}")
    log(f"  d* per dist (win_rate > 50%, 95% best): {d_star_wr_per_dist}")
    log(f"  d* per dist (target_rank < 50%, 5% of best): {d_star_rank_per_dist}")

    # === Per-dist ranking ===
    log("\n=== Gap_target_vs_wrong (target closer than wrong) ===")
    log(f"{'dist_kind':<25} {'d':<5} {'gap_mean':>12} {'CI':>30} {'>0':>8}")
    for dist_kind in DISTANCES:
        for d in DIMS:
            agg = aggregates[(d, dist_kind)]
            ci = agg["gap_target_vs_wrong_ci_95"]
            log(f"{dist_kind:<25} {d:<5} {agg['gap_target_vs_wrong_mean']:>12.4f} "
                f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
                f"{'PASS' if agg['gap_ci_above_zero'] else 'FAIL':>8}")

    log("\n=== Direction_win_rate (% candidates with d_target < d_wrong) ===")
    log(f"{'dist_kind':<25} {'d':<5} {'wr_mean':>12} {'CI':>30} {'>50%':>8}")
    for dist_kind in DISTANCES:
        for d in DIMS:
            agg = aggregates[(d, dist_kind)]
            ci = agg["win_rate_ci_95"]
            log(f"{dist_kind:<25} {d:<5} {agg['win_rate_mean']:>12.4f} "
                f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
                f"{'PASS' if agg['win_rate_ci_above_50pct'] else 'FAIL':>8}")

    log("\n=== Target_rank_percentile (lower = better) ===")
    log(f"{'dist_kind':<25} {'d':<5} {'rank_mean':>12} {'CI':>30} {'<50%':>8}")
    for dist_kind in DISTANCES:
        for d in DIMS:
            agg = aggregates[(d, dist_kind)]
            ci = agg["target_rank_percentile_ci_95"]
            log(f"{dist_kind:<25} {d:<5} {agg['target_rank_percentile_mean']:>12.4f} "
                f"[{ci[0]:>10.4f}, {ci[1]:>10.4f}] "
                f"{'PASS' if agg['target_rank_ci_below_50pct'] else 'FAIL':>8}")

    # === Save ===
    log("\n[10] Saving results ...")
    n_pass = sum(1 for a in aggregates.values() if a["PASS_all"])
    log(f"  PASS_all: {n_pass}/{len(aggregates)}")

    eval_dict = {
        "n_pairs": len(gap_per_pair.get((DIMS[0], DISTANCES[0]), {})),
        "n_wrong_users_per_pair": N_WRONG_USERS_PER_PAIR,
        "n_features_pool": D_pool,
        "dims_tested": DIMS,
        "distances_tested": DISTANCES,
        "d_star_gap_per_dist_kind": d_star_gap_per_dist,
        "d_star_win_rate_per_dist_kind": d_star_wr_per_dist,
        "d_star_rank_per_dist_kind": d_star_rank_per_dist,
        "n_pass_all": n_pass,
        "n_total_combinations": len(aggregates),
        "per_d_per_dist": {
            f"d={d}_{dk}": aggregates[(d, dk)] for d in DIMS for dk in DISTANCES
        },
        "note": "Phase 10.10.7: Cross-user separation validation. "
                "Three metrics: "
                "(1) gap = mean over cands of (d_target - mean(d_wrong_5)). > 0 means target closer. "
                "(2) win_rate = % cands where d_target < mean(d_wrong_5). > 50% means direction correct. "
                "(3) target_rank_percentile = target's distance rank / n_users. < 50% means target in top half. "
                "PASS_all = all three CI conditions hold.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair (only d=20/80/200/300 for size)
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin), _ in gap_per_pair[(20, "zscore_l2")].items():
            row = {"user_id": uid, "asin": asin}
            for d in [20, 80, 200, 300]:
                for dk in DISTANCES:
                    row[f"gap_d{d}_{dk}"] = gap_per_pair[(d, dk)].get((uid, asin), float("nan"))
                    row[f"wr_d{d}_{dk}"] = win_rate_per_pair[(d, dk)].get((uid, asin), float("nan"))
                    row[f"rank_d{d}_{dk}"] = target_rank_per_pair[(d, dk)].get((uid, asin), float("nan"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.10.7 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()