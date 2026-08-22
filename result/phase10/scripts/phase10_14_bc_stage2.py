#!/usr/bin/env python3
"""Phase 10.14.B + C — Stage 2 only: distance computation + Pareto.

Reads existing candidates jsonl + 318d npy cache, runs fast per-pair distances,
evaluates acceptance at 24/48/96 candidate counts.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_CANDIDATES = OUT_DIR / "phase10_14_bc_candidates_96.jsonl"
OUT_CAND_FEATS_CACHE = OUT_DIR / "phase10_14_bc_candidates_318d.npy"
OUT_EVAL = OUT_DIR / "phase10_14_bc_pareto_eval.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

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
    log = lambda m: print(f"[phase10-14.BC-s2] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.14.B + C — Stage 2: per-pair distance + Pareto")
    log("=" * 70)

    # === Load candidates + features ===
    candidates = []
    with OUT_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")
    cand_feats_keep = np.load(OUT_CAND_FEATS_CACHE)
    log(f"  318d features: {cand_feats_keep.shape}")

    # === Load profiles + sentence features ===
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  VADES users: {n_users}")

    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    log(f"  effective dims: {n_dims_eff}")

    # === Per-user mean features ===
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
    log("[3] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage: {shrinkage:.4f}")

    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)

    # === Group candidates by pair ===
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs: {len(cand_by_pair)}")

    rng = np.random.default_rng(SEED)
    wrong_per_pair = {}
    for (uid, asin) in cand_by_pair.keys():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx] = False
        wrong_per_pair[(uid, asin)] = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)

    # === Per-pair distance (only store d_target + d_wrong, no all_dists for rank) ===
    log("[4] Per-pair distance computation ...")
    pair_dist_data = {}
    t0 = time.time()
    for pi, ((uid, asin), cand_indices) in enumerate(cand_by_pair.items()):
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue

        cands_feats_local = cand_feats_keep[cand_indices]
        cross = cands_feats_local @ inv_cov @ user_mu_318d.T  # (n_cands, n_users)
        quad_cand = np.einsum('ij,jk,ik->i', cands_feats_local, inv_cov, cands_feats_local)
        dist_local = quad_cand[:, None] + quad_mu[None, :] - 2 * cross  # (n_cands, n_users)

        pair_dist_data[(uid, asin)] = {
            "d_target": dist_local[:, target_idx].astype(np.float64),  # (n_cands,)
            "d_wrong": dist_local[:, wrong_idx].astype(np.float64),  # (n_cands, K=5)
            "cand_indices": cand_indices,
        }

        if pi % 100 == 0:
            elapsed = time.time() - t0
            rate = (pi + 1) / max(elapsed, 0.001)
            eta = (len(cand_by_pair) - pi) / max(rate, 0.001)
            log(f"  [{pi}/{len(cand_by_pair)}] elapsed {elapsed:.1f}s, rate={rate:.1f}/s, ETA={eta:.0f}s")
        # Free dist_local explicitly
        del dist_local

    log(f"  done in {time.time()-t0:.1f}s, {len(pair_dist_data)} pairs")

    # === Per-user generated distribution ===
    log("[5] Per-user generated d_target distribution ...")
    user_d_targets = defaultdict(list)
    for (uid, asin), data in pair_dist_data.items():
        for d in data["d_target"]:
            user_d_targets[uid].append(float(d))
    n_users_with_data = len(user_d_targets)
    log(f"  users with data: {n_users_with_data}")

    # === Pareto at multiple candidate counts ===
    log("[6] Pareto at 24/48/96 ...")
    candidate_counts = [24, 48, 96]
    TAU_PERCENTILES = [50, 60, 70, 75, 80, 90, 95]

    pareto_results = {}

    for n_cands in candidate_counts:
        log(f"\n  === N candidates = {n_cands} ===")
        results_by_pct = {}
        for tau_pct in TAU_PERCENTILES:
            user_tau = {}
            for uid, d_list in user_d_targets.items():
                user_tau[uid] = float(np.percentile(d_list, tau_pct))

            n_pairs_total = 0
            n_pairs_with_accepted = 0
            n_total_cands_used = 0
            n_total_accepted = 0

            accepted_margins = {}
            accepted_ranks = {}  # rank only: for d_target among all 96 (full pool)

            for (uid, asin), data in pair_dist_data.items():
                target_idx = user_id_to_idx.get(uid)
                if target_idx is None:
                    continue
                tau_u = user_tau.get(uid)
                if tau_u is None:
                    continue
                n_pairs_total += 1

                cand_indices = data["cand_indices"][:n_cands]
                d_target = data["d_target"][:n_cands]
                d_wrong = data["d_wrong"][:n_cands]
                d_wrong_mean = d_wrong.mean(axis=1)
                margin = d_wrong_mean - d_target
                n_total_cands_used += len(cand_indices)

                accepted_mask = (d_target <= tau_u) & (d_target < d_wrong_mean)
                n_accepted = int(accepted_mask.sum())
                n_total_accepted += n_accepted

                if n_accepted > 0:
                    n_pairs_with_accepted += 1
                    accepted_margins_arr = np.where(accepted_mask, margin, -np.inf)
                    pick_idx = int(np.argmax(accepted_margins_arr))
                    accepted_margins[(uid, asin)] = float(margin[pick_idx])
                    # Compute rank of pick among all 96 (using full d_target)
                    d_target_full = data["d_target"]
                    rank = int((d_target_full < d_target_full[pick_idx]).sum())
                    accepted_ranks[(uid, asin)] = float(rank) / max(1, len(d_target_full))

            coverage = n_pairs_with_accepted / max(1, n_pairs_total)
            acceptance_rate = n_total_accepted / max(1, n_total_cands_used)

            margin_mean, margin_ci = bootstrap_ci(accepted_margins) if accepted_margins else (0.0, (0.0, 0.0))
            rank_mean, rank_ci = bootstrap_ci(accepted_ranks) if accepted_ranks else (0.0, (0.0, 0.0))

            results_by_pct[tau_pct] = {
                "tau_percentile": tau_pct,
                "coverage_pct": coverage,
                "acceptance_rate_per_total_query": acceptance_rate,
                "accepted_query_margin": margin_mean,
                "accepted_query_margin_ci_95": list(margin_ci),
                "accepted_query_rank_pct": rank_mean,
                "accepted_query_rank_pct_ci_95": list(rank_ci),
                "n_pairs_with_accepted": n_pairs_with_accepted,
                "n_pairs_total": n_pairs_total,
            }

        pareto_results[n_cands] = results_by_pct

        for tau_pct in [50, 75, 90, 95]:
            r = results_by_pct[tau_pct]
            log(f"    P{tau_pct}: coverage={100*r['coverage_pct']:.1f}%, "
                f"acc_rate={100*r['acceptance_rate_per_total_query']:.1f}%, "
                f"margin={r['accepted_query_margin']:.1f}, rank={100*r['accepted_query_rank_pct']:.1f}%")

    # === Save eval ===
    log("\n[7] Saving eval ...")
    eval_dict = {
        "n_pairs_total": len(pair_dist_data),
        "n_candidates_per_pair_full": 96,
        "candidate_counts_evaluated": candidate_counts,
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage",
        "shrinkage_intensity": float(shrinkage),
        "results": {str(k): v for k, v in pareto_results.items()},
        "comparison": {
            "Phase10_14_A_P95_coverage": 0.938,
            "Phase10_13_C_B_max_margin_WR": 0.7785,
        },
        "note": "Phase 10.14.B + C: 96 candidates/pool evaluated at 24/48/96. "
                "Acceptance: d_target <= τ_u (P_q of full 96) AND d_target < d_wrong_mean. "
                "Pareto: coverage vs quality (margin, rank, acceptance rate).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    log("=" * 70)
    log("PHASE 10.14.B + C STAGE 2 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()