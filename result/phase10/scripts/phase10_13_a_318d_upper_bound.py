#!/usr/bin/env python3
"""Phase 10.13.A-300d: Upper Bound Test with 318d Length-Invariant Features.

复用 318d 缓存 (phase10_10_6_cache/sentence_318d.npy), 对比 20d 上限 72.68%.

318d = extract_syntactic_features.ALL_FEATS_V2 (length-invariant rates,
POS opener types, punctuation bigrams). 比 20d 多 ~298 维更细粒度信号.

距离: Mahalanobis + Ledoit-Wolf shrinkage covariance from per-user mean.
对比:
  - 20d VADES logvar (Phase 10.13.A): 72.68% WR
  - 318d Ledoit-Wolf shrink: ?
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

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_EVAL = OUT_DIR / "phase10_13_a_318d_upper_bound.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_USER = 5

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"


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
    log = lambda m: print(f"[phase10-13.A-318d] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.13.A (318d): Upper Bound Test with 318d Features")
    log("=" * 70)

    # === Load 318d cache ===
    log("[1] Loading 318d sentence features cache ...")
    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    n_sentences, n_dims = sent_feats.shape
    log(f"  sentences: {n_sentences}, dims: {n_dims}")

    # === Identify zero-variance dims ===
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_keep = int(keep_dims.sum())
    log(f"  zero-variance dims: {n_dims - n_keep}, keeping: {n_keep}")

    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    n_dims_eff = n_keep

    # === Load VADES user profiles for user_id mapping ===
    log("[2] Loading VADES user profiles for user_id mapping ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  VADES users: {n_users}")

    # === Per-user mean features (318d) ===
    log("[3] Computing per-user mean features ...")
    user_ids_unique = list(user_id_to_idx.keys())
    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    sentences_per_user = np.zeros(n_users, dtype=np.int32)
    sent_to_user_idx = np.zeros(n_sentences, dtype=np.int32)  # map sent -> user idx

    for si in range(n_sentences):
        uid = sent_users[si]
        ui = user_id_to_idx.get(uid)
        if ui is None:
            continue
        sent_to_user_idx[si] = ui

    valid_sent_mask = sent_to_user_idx > 0  # exclude unmapped (assigned to -0 init)
    # Actually we init to 0 — need to filter properly
    valid_sent_mask = np.array([sent_users[si] in user_id_to_idx for si in range(n_sentences)])
    log(f"  valid sentences: {valid_sent_mask.sum()}/{n_sentences}")

    sent_to_user_idx = np.array([user_id_to_idx.get(sent_users[si], -1) for si in range(n_sentences)])

    # Compute per-user mean (efficient: scatter add then divide)
    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for si in np.where(valid_sent_mask)[0]:
        ui = sent_to_user_idx[si]
        user_mu_318d[ui] += sent_feats_keep[si]
        counts[ui] += 1

    valid_user_mask = counts > 0
    user_mu_318d[valid_user_mask] /= counts[valid_user_mask, None]
    log(f"  valid users: {valid_user_mask.sum()}/{n_users}")

    # === Ledoit-Wolf shrinkage covariance from per-user mean ===
    log("[4] Computing Ledoit-Wolf shrinkage covariance (from user_mu_318d) ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    log(f"  valid user_mu shape: {valid_user_mu.shape}")
    t0 = time.time()
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage intensity: {shrinkage:.4f}, time: {time.time()-t0:.1f}s")
    log(f"  cov shape: {cov_shrunk.shape}")

    # Invert (with regularization for numerical stability)
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    log(f"  inv_cov computed")

    # === Distance matrix (sentences × users) ===
    log("[5] Computing distance matrix (sentences × users) ...")
    valid_sent_feats = sent_feats_keep[valid_sent_mask]
    valid_sent_user_idx = sent_to_user_idx[valid_sent_mask]
    n_valid_sent = valid_sent_feats.shape[0]
    log(f"  valid sentences for distance: {n_valid_sent}")

    # Vectorized quadratic expansion:
    # d_ij = (s_i - mu_j)' inv_cov (s_i - mu_j)
    # = s_i' inv_cov s_i - 2 s_i' inv_cov mu_j + mu_j' inv_cov mu_j
    # Use all valid users (skip invalid user slots)
    valid_user_mu_all = user_mu_318d  # (n_users, n_dims_eff)
    valid_user_mask_arr = valid_user_mask  # (n_users,)

    t0 = time.time()
    # quad_s[i] = s_i' inv_cov s_i
    quad_s = np.einsum('ij,jk,ik->i', valid_sent_feats, inv_cov, valid_sent_feats)
    # quad_mu[j] = mu_j' inv_cov mu_j
    quad_mu = np.einsum('ij,jk,ik->i', valid_user_mu_all, inv_cov, valid_user_mu_all)
    # cross[i, j] = s_i' inv_cov mu_j
    cross = valid_sent_feats @ inv_cov @ valid_user_mu_all.T
    # dist_matrix
    dist_matrix = quad_s[:, None] + quad_mu[None, :] - 2 * cross  # (n_valid_sent, n_users)
    log(f"  dist_matrix: {dist_matrix.shape}, computed in {time.time()-t0:.1f}s")

    # === Pre-sample wrong users per user ===
    log(f"[6] Sampling {N_WRONG_USERS_PER_USER} wrong users per user ...")
    rng = np.random.default_rng(SEED)
    wrong_users_per_owner = {}
    for ui in np.where(valid_user_mask)[0]:
        mask = np.ones(n_users, dtype=bool)
        mask[ui] = False
        wrong_idx = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_USER, replace=False)
        wrong_users_per_owner[int(ui)] = wrong_idx

    # === Per-sentence metrics ===
    log("[7] Computing per-sentence metrics ...")
    per_sent_metrics = []
    for si_local in range(n_valid_sent):
        ui = int(valid_sent_user_idx[si_local])
        if not valid_user_mask[ui]:
            continue
        wrong_idx = wrong_users_per_owner.get(ui)
        if wrong_idx is None:
            continue

        d_target = float(dist_matrix[si_local, ui])
        d_wrong_5 = float(dist_matrix[si_local, wrong_idx].mean())
        margin = d_wrong_5 - d_target

        # Rank percentile (target rank / n_users, excluding self)
        target_rank_pct = float(np.sum(dist_matrix[si_local] < d_target)) / n_users

        per_sent_metrics.append({
            "user_id": sent_users[np.where(valid_sent_mask)[0][si_local]],
            "user_idx": ui,
            "si_local": si_local,
            "d_target": d_target,
            "d_wrong_5": d_wrong_5,
            "margin": margin,
            "rank_pct": target_rank_pct,
            "win": margin > 0,
        })

    n_sents = len(per_sent_metrics)
    log(f"  sentences evaluated: {n_sents}")

    # === Per-user aggregation ===
    log("[8] Per-user aggregation ...")
    per_user = defaultdict(list)
    for m in per_sent_metrics:
        per_user[m["user_idx"]].append(m)

    user_win_rates = {}
    user_mean_margin = {}
    user_mean_rank = {}
    for uid_key, metrics in per_user.items():
        wins = [m["win"] for m in metrics]
        margins = [m["margin"] for m in metrics]
        ranks = [m["rank_pct"] for m in metrics]
        user_win_rates[uid_key] = float(np.mean(wins))
        user_mean_margin[uid_key] = float(np.mean(margins))
        user_mean_rank[uid_key] = float(np.mean(ranks))

    # === Global aggregation ===
    log("[9] Global aggregation ...")
    wr_mean, wr_ci = bootstrap_ci(user_win_rates)
    margin_mean, margin_ci = bootstrap_ci(user_mean_margin)
    rank_mean, rank_ci = bootstrap_ci(user_mean_rank)

    log(f"  Per-user mean win rate: {wr_mean:.4f}, CI=[{wr_ci[0]:.4f}, {wr_ci[1]:.4f}]")
    log(f"  Per-user mean margin:   {margin_mean:.4f}, CI=[{margin_ci[0]:.4f}, {margin_ci[1]:.4f}]")
    log(f"  Per-user mean rank:     {rank_mean:.4f}, CI=[{rank_ci[0]:.4f}, {rank_ci[1]:.4f}]")

    sent_wins = {m["si_local"]: float(m["win"]) for m in per_sent_metrics}
    sent_wr_mean, sent_wr_ci = bootstrap_ci(sent_wins)
    log(f"  Per-sentence win rate:  {sent_wr_mean:.4f}, CI=[{sent_wr_ci[0]:.4f}, {sent_wr_ci[1]:.4f}]")

    # === Coverage analysis ===
    log("[10] Coverage analysis ...")
    coverage_at = {}
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        n_pass = sum(1 for wr in user_win_rates.values() if wr >= threshold)
        coverage_at[threshold] = {
            "n_users_pass": n_pass,
            "coverage_pct": n_pass / len(user_win_rates),
        }
        log(f"  threshold >= {threshold}: {n_pass}/{len(user_win_rates)} users = {100*n_pass/len(user_win_rates):.1f}%")

    # Distribution
    wr_values = sorted(user_win_rates.values())
    min_wr = wr_values[0]
    max_wr = wr_values[-1]
    median_wr = wr_values[len(wr_values)//2]
    q25_wr = wr_values[len(wr_values)//4]
    q75_wr = wr_values[3*len(wr_values)//4]
    log(f"  Per-user win rate dist: min={min_wr:.3f}, q25={q25_wr:.3f}, median={median_wr:.3f}, q75={q75_wr:.3f}, max={max_wr:.3f}")

    # === Threshold calibration ===
    log("[11] Threshold calibration ...")
    all_d_targets = np.array([m["d_target"] for m in per_sent_metrics])
    all_margins = np.array([m["margin"] for m in per_sent_metrics])

    p50_d = float(np.percentile(all_d_targets, 50))
    p75_d = float(np.percentile(all_d_targets, 75))
    p90_d = float(np.percentile(all_d_targets, 90))
    p50_m = float(np.percentile(all_margins, 50))
    p75_m = float(np.percentile(all_margins, 75))
    p10_m = float(np.percentile(all_margins, 10))

    log(f"  d_target: P50={p50_d:.2f}, P75={p75_d:.2f}, P90={p90_d:.2f}")
    log(f"  margin:   P10={p10_m:.2f}, P50={p50_m:.2f}, P75={p75_m:.2f}")

    # === Save eval ===
    log("\n[12] Saving eval ...")
    eval_dict = {
        "n_users": n_users,
        "n_sentences": n_sents,
        "n_dims_original": n_dims,
        "n_dims_effective": n_dims_eff,
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage (from user_mu_318d)",
        "shrinkage_intensity": float(shrinkage),
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
            "d_target_p50": p50_d,
            "d_target_p75": p75_d,
            "d_target_p90": p90_d,
            "margin_p10": p10_m,
            "margin_p50": p50_m,
            "margin_p75": p75_m,
        },
        "comparison_with_20d": {
            "Phase10_13_A_20d_win_rate": 0.7268,
            "Phase10_13_A_318d_win_rate": wr_mean,
            "improvement_pp": (wr_mean - 0.7268) * 100,
        },
        "note": "Phase 10.13.A 318d version. Uses 318d length-invariant features from "
                "extract_syntactic_features.ALL_FEATS_V2. Distance = Mahalanobis + "
                "Ledoit-Wolf shrinkage covariance estimated from per-user mean vectors. "
                "Compares to Phase 10.13.A 20d version (VADES user_logvar).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    log("=" * 70)
    log("PHASE 10.13.A (318d) COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()