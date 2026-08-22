#!/usr/bin/env python3
"""Phase 10.13.B (318d): Acceptance/Coverage Pipeline in 318d Space.

318d 上限 97.86% WR (vs 20d 72.68%) → acceptance/coverage 应大幅提升.

流程:
  1. 加载 318d 缓存 (sentences + cached candidates 8762 from Phase 10.10.1)
  2. 从真实评论计算 per-user τ (P75 d_target) 和 m 阈值
  3. 对每个 pair × 每个候选:
     - d_target, d_nearest_other, margin (318d Mahalanobis)
     - Accept if: d_target ≤ τ_u AND margin ≥ m
  4. 报告: acceptance rate (per accepted query) + coverage rate (% pairs)

如果 cached 候选不够 → 触发 Phase 10.13.B-318d-ext 重新生成更大候选池.
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
OUT_EVAL = OUT_DIR / "phase10_13_b_318d_acceptance.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"
CAND_FEATS_CACHE = CACHE_DIR / "candidate_318d.npy"
CAND_KEYS_CACHE = CACHE_DIR / "candidate_318d_keys.json"

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
    log = lambda m: print(f"[phase10-13.B-318d] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.13.B (318d): Acceptance/Coverage Pipeline")
    log("=" * 70)

    # === Load all caches ===
    log("[1] Loading caches ...")
    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    cand_feats = np.load(CAND_FEATS_CACHE)
    cand_keys = json.loads(CAND_KEYS_CACHE.read_text())
    log(f"  sentences: {sent_feats.shape}")
    log(f"  candidates: {cand_feats.shape}")

    # Identify effective dims (drop zero-variance)
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    log(f"  effective dims: {n_dims_eff}")

    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    cand_feats_keep = cand_feats[:, keep_dims].astype(np.float64)

    # === Load VADES user profiles for user_id mapping ===
    log("[2] Loading VADES profiles ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  VADES users: {n_users}")

    # === Per-user mean features (318d) ===
    log("[3] Computing per-user mean features ...")
    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)

    for si in range(len(sent_users)):
        ui = user_id_to_idx.get(sent_users[si])
        if ui is None:
            continue
        user_mu_318d[ui] += sent_feats_keep[si]
        counts[ui] += 1
    valid_user_mask = counts > 0
    user_mu_318d[valid_user_mask] /= counts[valid_user_mask, None]
    log(f"  valid users: {valid_user_mask.sum()}/{n_users}")

    # === Ledoit-Wolf covariance ===
    log("[4] Ledoit-Wolf shrinkage covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    t0 = time.time()
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage: {shrinkage:.4f}, time: {time.time()-t0:.1f}s")

    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))

    # === Per-user thresholds from real reviews ===
    log("[5] Computing per-user thresholds from real reviews ...")
    # For each user: P75 of own d_target, P50 of own margin
    log("  computing real review distances per user ...")

    # We need to compute distance from each user's sentences to their own mu
    # Already have dist_matrix in 10.13.A-318d; but let's recompute quickly.
    # Only need per-user d_target values.

    # Build sent → user index mapping
    sent_user_idx = np.array([user_id_to_idx.get(sent_users[si], -1) for si in range(len(sent_users))])

    # Compute per-user d_target (just the diagonal values)
    quad_s = np.einsum('ij,jk,ik->i', sent_feats_keep, inv_cov, sent_feats_keep)
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)
    cross = sent_feats_keep @ inv_cov @ user_mu_318d.T  # (n_sent, n_users)

    # Per-user: collect d_target values
    per_user_d_targets = defaultdict(list)
    per_user_margins = defaultdict(list)

    rng = np.random.default_rng(SEED)
    # Pre-sample wrong users for each valid user
    wrong_per_user = {}
    for ui in np.where(valid_user_mask)[0]:
        mask = np.ones(n_users, dtype=bool)
        mask[ui] = False
        wrong_per_user[int(ui)] = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)

    for si in range(len(sent_users)):
        ui = sent_user_idx[si]
        if ui < 0 or not valid_user_mask[ui]:
            continue
        wrong_idx = wrong_per_user.get(int(ui))
        if wrong_idx is None:
            continue
        d_target = float(quad_s[si] + quad_mu[ui] - 2 * cross[si, ui])
        d_wrong = float((quad_s[si] + quad_mu[wrong_idx] - 2 * cross[si, wrong_idx]).mean())
        per_user_d_targets[int(ui)].append(d_target)
        per_user_margins[int(ui)].append(d_wrong - d_target)

    # Per-user τ (P75 d_target), m (P50 margin)
    user_tau = {}
    user_m = {}
    for ui in per_user_d_targets.keys():
        user_tau[ui] = float(np.percentile(per_user_d_targets[ui], 75))
        user_m[ui] = float(np.percentile(per_user_margins[ui], 50))
    log(f"  per-user thresholds computed for {len(user_tau)} users")

    # Median tau and m
    median_tau = float(np.median(list(user_tau.values())))
    median_m = float(np.median(list(user_m.values())))
    log(f"  median τ (P75 d_target): {median_tau:.2f}")
    log(f"  median m (P50 margin): {median_m:.2f}")

    # === Candidate distances ===
    log("[6] Computing candidate distances ...")
    # For each candidate, compute d_target and margin
    # candidates: keys = "user__asin__cond__seed"
    cand_quad = np.einsum('ij,jk,ik->i', cand_feats_keep, inv_cov, cand_feats_keep)
    cand_cross = cand_feats_keep @ inv_cov @ user_mu_318d.T  # (n_cand, n_users)

    # Group candidates by pair
    cand_by_pair = defaultdict(list)
    for ci, key in enumerate(cand_keys):
        parts = key.split("__")
        uid = parts[0]
        asin = parts[1]
        cand_by_pair[(uid, asin)].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    # Pre-sample wrong users for each pair's owner
    wrong_per_pair_owner = {}
    for (uid, asin) in cand_by_pair.keys():
        ui = user_id_to_idx.get(uid)
        if ui is None:
            continue
        wrong_per_pair_owner[(uid, asin)] = wrong_per_user.get(int(ui))

    # === Per-pair acceptance ===
    log("[7] Per-pair acceptance analysis ...")

    # Try multiple thresholds
    threshold_combos = [
        ("loose", "median_tau", 0.0),
        ("loose", "median_tau", median_m),  # require positive margin at least
        ("medium", "P75", 0.0),
        ("medium", "P75", median_m),
        ("strict", "P50", 0.0),
        ("strict", "P50", median_m),
    ]

    results_by_threshold = {}

    for name, tau_kind, m_val in threshold_combos:
        if tau_kind == "median_tau":
            tau_fn = lambda ui: median_tau
        elif tau_kind == "P75":
            tau_fn = lambda ui: user_tau.get(int(ui), median_tau)
        elif tau_kind == "P50":
            tau_fn = lambda ui: float(np.percentile(per_user_d_targets.get(int(ui), [median_tau]), 50))

        per_pair = {}
        n_pairs_with_cands = 0
        n_pairs_with_accepted = 0
        n_total_accepted = 0
        n_total_cands = 0

        for (uid, asin), cand_indices in cand_by_pair.items():
            ui = user_id_to_idx.get(uid)
            if ui is None or not valid_user_mask[ui]:
                continue
            wrong_idx = wrong_per_pair_owner.get((uid, asin))
            if wrong_idx is None:
                continue
            n_pairs_with_cands += 1
            n_total_cands += len(cand_indices)

            tau_u = tau_fn(ui)

            # Per-cand acceptance
            accepted = []
            for ci in cand_indices:
                d_target = float(cand_quad[ci] + quad_mu[ui] - 2 * cand_cross[ci, ui])
                d_wrong = float((cand_quad[ci] + quad_mu[wrong_idx] - 2 * cand_cross[ci, wrong_idx]).mean())
                margin = d_wrong - d_target
                if d_target <= tau_u and margin >= m_val:
                    accepted.append((ci, d_target, d_wrong, margin))

            if accepted:
                n_pairs_with_accepted += 1
                n_total_accepted += len(accepted)
                # Pick best by max margin
                best = max(accepted, key=lambda x: x[3])
                per_pair[(uid, asin)] = {
                    "n_total": len(cand_indices),
                    "n_accepted": len(accepted),
                    "best_margin": best[3],
                    "best_d_target": best[1],
                    "tau_u": tau_u,
                    "m_threshold": m_val,
                }
            else:
                per_pair[(uid, asin)] = {
                    "n_total": len(cand_indices),
                    "n_accepted": 0,
                    "best_margin": None,
                    "best_d_target": None,
                    "tau_u": tau_u,
                    "m_threshold": m_val,
                }

        coverage = n_pairs_with_accepted / n_pairs_with_cands if n_pairs_with_cands > 0 else 0
        acceptance_per_accepted_query = 1.0  # by definition (we filtered)
        acceptance_per_total_query = n_total_accepted / n_total_cands if n_total_cands > 0 else 0

        results_by_threshold[name] = {
            "tau_kind": tau_kind,
            "m_threshold": m_val,
            "n_pairs_with_cands": n_pairs_with_cands,
            "n_pairs_with_accepted": n_pairs_with_accepted,
            "coverage_pct": coverage,
            "n_total_cands": n_total_cands,
            "n_total_accepted": n_total_accepted,
            "acceptance_rate_per_total_query": acceptance_per_total_query,
            "acceptance_rate_per_accepted_query": acceptance_per_accepted_query,
        }
        log(f"\n  Threshold '{name}' (tau={tau_kind}, m={m_val:.1f}):")
        log(f"    coverage: {n_pairs_with_accepted}/{n_pairs_with_cands} = {100*coverage:.1f}%")
        log(f"    acceptance rate (per total query): {100*acceptance_per_total_query:.1f}%")
        log(f"    acceptance rate (per accepted): 100% (by construction)")

    # === Best threshold summary ===
    log("\n[8] Best threshold selection ...")
    # Pick threshold with best coverage AND reasonable acceptance rate
    best_name = max(results_by_threshold.keys(),
                    key=lambda n: results_by_threshold[n]["coverage_pct"] *
                                  results_by_threshold[n]["acceptance_rate_per_total_query"])
    log(f"  best by coverage×acceptance: {best_name}")

    # === Save eval ===
    log("\n[9] Saving eval ...")
    eval_dict = {
        "n_users": n_users,
        "n_dims_effective": n_dims_eff,
        "n_pairs_with_candidates": results_by_threshold["loose"]["n_pairs_with_cands"],
        "n_candidates_per_pair_avg": results_by_threshold["loose"]["n_total_cands"] // max(1, results_by_threshold["loose"]["n_pairs_with_cands"]),
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage (318d)",
        "shrinkage_intensity": float(shrinkage),
        "results_by_threshold": results_by_threshold,
        "thresholds_summary": {
            "median_tau": median_tau,
            "median_m": median_m,
        },
        "comparison": {
            "Phase10_13_A_318d_upper_bound": 0.9786,
            "Phase10_13_A_20d_upper_bound": 0.7268,
            "Phase10_12_B_20d_generated": 0.6724,
        },
        "note": "Phase 10.13.B (318d): Acceptance/coverage pipeline. "
                "Real review ceiling in 318d space = 97.86% WR. "
                "Acceptance rate per accepted query = 100% by construction. "
                "Coverage rate = % pairs with at least 1 accepted query. "
                "If coverage is high (>80%), 318d space provides strong acceptance signal.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    log("=" * 70)
    log("PHASE 10.13.B (318d) COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()