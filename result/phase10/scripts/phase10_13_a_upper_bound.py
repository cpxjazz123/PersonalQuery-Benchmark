#!/usr/bin/env python3
"""Phase 10.13.A: Upper Bound Test.

问题: 当前生成 query 在 VADES 20d 空间 win rate ~67%.
      这是否是天花板? 真实评论本身能否做到 100% target 最近?

如果真实评论也只能 80%, 那 67% 几乎已经逼近上限.
如果真实评论能做到 95%+, 说明生成质量仍有提升空间, 应从 coverage 上突破.

方法:
  1. 加载 VADES user_mu/user_logvar (2918 用户)
  2. 加载每用户 15 句真实评论 (sentences.jsonl)
  3. 对每用户每条评论:
     - d_target = Mahalanobis(自己 user_mu, sentence_features)
     - d_wrong = mean(Mahalanobis(5 错 user_mu, sentence_features))
     - margin = d_wrong - d_target
     - rank = percentile of d_target among all 2918 users
  4. 聚合: win rate, rank, margin, acceptance at various thresholds
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_EVAL = OUT_DIR / "phase10_13_a_upper_bound.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_USER = 5

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"


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
    log = lambda m: print(f"[phase10-13.A] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.13.A: Upper Bound Test - Real Reviews Self-Fit")
    log("=" * 70)

    # === Load VADES user profiles ===
    log("[1] Loading VADES user profiles ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    n_users = user_mu.shape[0]
    log(f"  users: {n_users}")

    # === Load sentences by user ===
    log("[2] Loading sentences ...")
    sentences_by_user = defaultdict(list)
    feature_names = None
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if feature_names is None:
                feature_names = list(r["features"].keys())
            sentences_by_user[r["user_id"]].append(r["features"])
    log(f"  feature dim: {len(feature_names)}")
    log(f"  sentences: {sum(len(v) for v in sentences_by_user.values())}")

    # === Compute population stats (for normalization) ===
    log("[3] Computing population stats ...")
    user_feats = {}
    for uid, feats_list in sentences_by_user.items():
        if not feats_list:
            continue
        arr = np.array([[float(f[k]) for k in feature_names] for f in feats_list], dtype=np.float32)
        user_feats[uid] = arr.mean(axis=0)
    user_arr = np.stack(list(user_feats.values()), axis=0)
    pop_mean = user_arr.mean(axis=0)
    pop_std = user_arr.std(axis=0) + 1e-9

    # === Per-sentence: d_target and d_wrong ===
    log("[4] Computing per-sentence distances ...")

    # Pre-extract sentence features as array (n_total_sentences, 20)
    all_sentences = []
    sentence_user_ids = []
    for uid, feats_list in sentences_by_user.items():
        for f in feats_list:
            all_sentences.append([float(f[k]) for k in feature_names])
            sentence_user_ids.append(uid)
    all_sentences_arr = np.array(all_sentences, dtype=np.float32)  # (43770, 20)
    log(f"  total sentences: {all_sentences_arr.shape}")

    # Normalize sentence features
    sent_norm = (all_sentences_arr - pop_mean) / pop_std  # (43770, 20)

    # Normalize user_mu
    user_mu_norm = (user_mu - pop_mean) / pop_std

    # Compute full distance matrix (this is the heavy step)
    log("[5] Computing full distance matrix (sentences × users) ...")
    t0 = time.time()
    inv_var = np.exp(-user_logvar)  # (n_users, 20)
    sent_norm_sq = (sent_norm ** 2) @ inv_var.T  # (n_sentences, n_users)
    user_norm_sq = (user_mu_norm ** 2 * inv_var).sum(axis=1)  # (n_users,)
    weighted_user = inv_var * user_mu_norm  # (n_users, 20)
    cross = sent_norm @ weighted_user.T  # (n_sentences, n_users)
    dist_matrix = sent_norm_sq + user_norm_sq[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}, computed in {time.time()-t0:.1f}s")

    # === Pre-sample wrong users per user (for K=5 wrong mean) ===
    log(f"[6] Sampling {N_WRONG_USERS_PER_USER} wrong users per sentence-owner ...")
    rng = np.random.default_rng(SEED)
    wrong_users_per_owner = {}
    for uid in sentences_by_user.keys():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx] = False
        wrong_idx = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_USER, replace=False)
        wrong_users_per_owner[uid] = wrong_idx

    # === Per-sentence metrics ===
    log("[7] Computing per-sentence metrics ...")
    per_sent_metrics = []  # list of dicts
    for si in range(len(sentence_user_ids)):
        uid = sentence_user_ids[si]
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_users_per_owner.get(uid)
        if wrong_idx is None:
            continue

        d_target = float(dist_matrix[si, target_idx])
        d_wrong_5 = float(dist_matrix[si, wrong_idx].mean())
        margin = d_wrong_5 - d_target

        # Rank: percentile of d_target among all users
        target_rank_pct = float(np.sum(dist_matrix[si] <= d_target) - 1) / n_users  # excluding self

        per_sent_metrics.append({
            "user_id": uid,
            "si": si,
            "d_target": d_target,
            "d_wrong_5": d_wrong_5,
            "margin": margin,
            "rank_pct": target_rank_pct,
            "win": margin > 0,
            "target_is_nearest": margin > 0,  # because wrong_5 is mean, not nearest
        })

    n_sents = len(per_sent_metrics)
    log(f"  sentences evaluated: {n_sents}")

    # === Per-user aggregation ===
    log("[8] Per-user aggregation ...")
    per_user = defaultdict(list)
    for m in per_sent_metrics:
        per_user[m["user_id"]].append(m)

    user_win_rates = {}  # per user: % of sentences where win=True
    user_mean_margin = {}
    user_mean_rank = {}
    user_worst_rank = {}  # worst rank among sentences

    for uid, metrics in per_user.items():
        wins = [m["win"] for m in metrics]
        margins = [m["margin"] for m in metrics]
        ranks = [m["rank_pct"] for m in metrics]

        user_win_rates[uid] = float(np.mean(wins))
        user_mean_margin[uid] = float(np.mean(margins))
        user_mean_rank[uid] = float(np.mean(ranks))
        user_worst_rank[uid] = float(np.max(ranks))  # worst = highest rank

    # === Global aggregation ===
    log("[9] Global aggregation ...")
    wr_mean, wr_ci = bootstrap_ci(user_win_rates)
    margin_mean, margin_ci = bootstrap_ci(user_mean_margin)
    rank_mean, rank_ci = bootstrap_ci(user_mean_rank)

    log(f"  Per-user mean win rate: {wr_mean:.4f}, CI=[{wr_ci[0]:.4f}, {wr_ci[1]:.4f}]")
    log(f"  Per-user mean margin:   {margin_mean:.4f}, CI=[{margin_ci[0]:.4f}, {margin_ci[1]:.4f}]")
    log(f"  Per-user mean rank:     {rank_mean:.4f}, CI=[{rank_ci[0]:.4f}, {rank_ci[1]:.4f}]")

    # Per-sentence aggregation (more granular)
    sent_wins = {m["si"]: float(m["win"]) for m in per_sent_metrics}
    sent_wr_mean, sent_wr_ci = bootstrap_ci(sent_wins)
    log(f"  Per-sentence win rate:  {sent_wr_mean:.4f}, CI=[{sent_wr_ci[0]:.4f}, {sent_wr_ci[1]:.4f}]")

    # === Coverage analysis: % users with win_rate >= X ===
    log("[10] Coverage analysis (upper bound) ...")
    coverage_at = {}
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        n_users_pass = sum(1 for wr in user_win_rates.values() if wr >= threshold)
        coverage_at[threshold] = {
            "n_users_pass": n_users_pass,
            "coverage_pct": n_users_pass / len(user_win_rates),
        }
        log(f"  threshold >= {threshold}: {n_users_pass}/{len(user_win_rates)} users = {100*n_users_pass/len(user_win_rates):.1f}%")

    # Per-user min win rate distribution
    min_wr = min(user_win_rates.values())
    max_wr = max(user_win_rates.values())
    median_wr = sorted(user_win_rates.values())[len(user_win_rates)//2]
    q25_wr = sorted(user_win_rates.values())[len(user_win_rates)//4]
    q75_wr = sorted(user_win_rates.values())[3*len(user_win_rates)//4]

    log(f"  Per-user win rate distribution: min={min_wr:.3f}, q25={q25_wr:.3f}, median={median_wr:.3f}, q75={q75_wr:.3f}, max={max_wr:.3f}")

    # === Threshold analysis: For acceptance/criteria design ===
    log("[11] Threshold calibration for acceptance ...")
    # Compute d_target percentiles across all sentences (for threshold τ)
    all_d_targets = np.array([m["d_target"] for m in per_sent_metrics])
    all_margins = np.array([m["margin"] for m in per_sent_metrics])

    # P50, P75, P90 of d_target for "typical" review
    p50_d_target = float(np.percentile(all_d_targets, 50))
    p75_d_target = float(np.percentile(all_d_targets, 75))
    p90_d_target = float(np.percentile(all_d_targets, 90))

    p50_margin = float(np.percentile(all_margins, 50))
    p75_margin = float(np.percentile(all_margins, 75))
    p10_margin = float(np.percentile(all_margins, 10))

    log(f"  d_target percentiles: P50={p50_d_target:.2f}, P75={p75_d_target:.2f}, P90={p90_d_target:.2f}")
    log(f"  margin percentiles:   P10={p10_margin:.2f}, P50={p50_margin:.2f}, P75={p75_margin:.2f}")

    # === Save eval ===
    log("\n[12] Saving eval ...")
    eval_dict = {
        "n_users": n_users,
        "n_sentences": n_sents,
        "n_wrong_users_per_user": N_WRONG_USERS_PER_USER,
        "per_user_aggregate": {
            "win_rate_mean": wr_mean,
            "win_rate_ci_95": list(wr_ci),
            "margin_mean": margin_mean,
            "margin_ci_95": list(margin_ci),
            "rank_mean": rank_mean,
            "rank_ci_95": list(rank_ci),
        },
        "per_sentence_aggregate": {
            "win_rate_mean": sent_wr_mean,
            "win_rate_ci_95": list(sent_wr_ci),
        },
        "per_user_win_rate_distribution": {
            "min": min_wr, "q25": q25_wr, "median": median_wr, "q75": q75_wr, "max": max_wr,
        },
        "coverage_at_threshold": coverage_at,
        "thresholds_for_acceptance": {
            "d_target_p50": p50_d_target,
            "d_target_p75": p75_d_target,
            "d_target_p90": p90_d_target,
            "margin_p10": p10_margin,
            "margin_p50": p50_margin,
            "margin_p75": p75_margin,
        },
        "comparison": {
            "Phase10_12_B_max_margin_win_rate": 0.6724,
            "Phase10_12_C_dynamic_control_win_rate": 0.6331,
        },
        "note": "Phase 10.13.A: Upper bound test. "
                "Real reviews of user should achieve high win rate (target user nearest). "
                "This sets the upper bound for generation-based win rate. "
                "If real win rate = 67%, generation is near ceiling. "
                "If real win rate = 90%+, generation has room to improve.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    log("=" * 70)
    log("PHASE 10.13.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()