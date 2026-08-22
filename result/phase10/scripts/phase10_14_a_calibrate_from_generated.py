#!/usr/bin/env python3
"""Phase 10.14.A: 用生成候选分布校准 acceptance 阈值.

用户洞察: 之前用真实评论 P75 作 τ_u 不合理. 生成 query 距离天然 > 真实评论.
应该用每个用户自己生成候选的分布来校准.

τ_u = P_q(d_target_in_generated_pool)  (e.g., P75/P90/P95)
m = 0  (target 比 nearest wrong 更近即可)

评估维度 (Pareto 多目标):
  1. coverage (% pairs 至少 1 个 accepted)
  2. acceptance quality (accepted query 的 win rate vs target)
  3. target rank percentile
  4. margin (d_nearest_other - d_target)
  5. attr exact match (3 attrs in raw query)
  6. semantic similarity (Jaccard 与真实评论)

τ percentile sweep: P50/P60/P70/P75/P80/P90/P95
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
OUT_EVAL = OUT_DIR / "phase10_14_a_calibrate_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase10_14_a_per_pair.jsonl"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

# 318d cache
CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"
CAND_FEATS_CACHE = OUT_DIR / "phase10_12_a_candidates_318d.npy"

# Phase 10.12.A candidates (24/pair)
IN_CANDIDATES = OUT_DIR / "phase10_12_a_candidates_syntactic.jsonl"

# Real reviews for semantic similarity baseline
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SENTENCE_RAW_FILE = OUT_DIR / "vades_prototype_3000u_v6_raw_sentence_text.jsonl"
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


def jaccard_words(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def attr_present(query: str, attrs: dict) -> dict:
    """Check each main attr in query (lowercase substring)."""
    q_lower = query.lower()
    return {k: (str(v).lower() in q_lower) for k, v in attrs.items() if v}


def main():
    log = lambda m: print(f"[phase10-14.A] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.14.A: Calibrate τ from Generated Distribution")
    log("=" * 70)

    # === Load caches ===
    log("[1] Loading 318d caches ...")
    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    cand_feats = np.load(CAND_FEATS_CACHE)

    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    log(f"  effective dims: {n_dims_eff}")

    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    # cand_feats is already pre-filtered to n_dims_eff (saved by phase10_13_c)
    if cand_feats.shape[1] == n_dims_eff:
        cand_feats_keep = cand_feats.astype(np.float64)
    else:
        cand_feats_keep = cand_feats[:, keep_dims].astype(np.float64)

    # === Load VADES profiles ===
    log("[2] Loading VADES profiles ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)

    # === Per-user mean features ===
    log("[3] Per-user mean features (318d) ...")
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

    # === Ledoit-Wolf ===
    log("[4] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))

    # === Load candidates ===
    log(f"[5] Loading candidates from {IN_CANDIDATES} ...")
    candidates = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")

    # === Distance matrix ===
    log("[6] Computing distance matrix ...")
    quad_cand = np.einsum('ij,jk,ik->i', cand_feats_keep, inv_cov, cand_feats_keep)
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)
    cross = cand_feats_keep @ inv_cov @ user_mu_318d.T
    dist_matrix = quad_cand[:, None] + quad_mu[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}")

    # Group candidates by pair
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)

    # Pre-sample wrong users per pair
    rng = np.random.default_rng(SEED)
    wrong_per_pair = {}
    for (uid, asin) in cand_by_pair.keys():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx] = False
        wrong_per_pair[(uid, asin)] = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)

    # === Per-user generated distance distribution ===
    log("[7] Computing per-user generated d_target distribution ...")
    user_d_targets_generated = defaultdict(list)
    for ci, c in enumerate(candidates):
        uid = c["user_id"]
        ui = user_id_to_idx.get(uid)
        if ui is None:
            continue
        user_d_targets_generated[ui].append(float(dist_matrix[ci, ui]))

    log(f"  users with generated candidates: {len(user_d_targets_generated)}")

    # === Real review text for semantic similarity ===
    log("[8] Loading real review texts for semantic similarity ...")
    user_real_texts = defaultdict(list)
    # We'll use sentence text from sentences.jsonl. But sentences.jsonl only has features, not text.
    # Let me check if there's a text source.
    # The sentence file has features but maybe also text? Let's load it and check.

    real_text_by_user = defaultdict(list)
    try:
        SENTENCE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"
        with SENTENCE_FILE.open() as f:
            for line in f:
                r = json.loads(line)
                if "text" in r:
                    real_text_by_user[r["user_id"]].append(r["text"])
        log(f"  loaded {sum(len(v) for v in real_text_by_user.values())} real review texts")
    except Exception as e:
        log(f"  WARN: could not load real texts: {e}")

    has_real_text = sum(len(v) for v in real_text_by_user.values()) > 0

    # === Acceptance analysis: sweep τ percentile ===
    log("[9] Acceptance analysis (sweep τ percentile) ...")
    TAU_PERCENTILES = [50, 60, 70, 75, 80, 90, 95]

    results_by_percentile = {}

    for tau_pct in TAU_PERCENTILES:
        # Per-user τ from generated distribution
        user_tau = {}
        for ui, d_list in user_d_targets_generated.items():
            user_tau[ui] = float(np.percentile(d_list, tau_pct))

        # Per-pair acceptance
        per_pair_metrics = {}
        n_pairs_total = 0
        n_pairs_with_accepted = 0
        n_total_cands = 0
        n_total_accepted = 0

        # Acceptance-quality metrics (over accepted queries)
        accepted_win_rates = {}
        accepted_margins = {}
        accepted_ranks = {}
        accepted_attr_match = {}
        accepted_sem_sim = {}

        for (uid, asin), cand_indices in cand_by_pair.items():
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue
            wrong_idx = wrong_per_pair.get((uid, asin))
            if wrong_idx is None:
                continue
            tau_u = user_tau.get(int(target_idx))
            if tau_u is None:
                continue
            n_pairs_total += 1
            n_total_cands += len(cand_indices)

            dist_pair = dist_matrix[cand_indices]
            d_target = dist_pair[:, target_idx]
            d_wrong = dist_pair[:, wrong_idx]
            d_wrong_mean = d_wrong.mean(axis=1)  # mean over K=5 wrong users
            # margin = d_wrong_mean - d_target (positive if target closer than mean of 5 wrong)
            margin = d_wrong_mean - d_target

            # Per-cand acceptance:
            # d_target <= tau_u AND d_target < d_wrong_mean (i.e., target closer than avg of 5 wrong)
            accepted_mask = (d_target <= tau_u) & (d_target < d_wrong_mean)
            n_accepted = int(accepted_mask.sum())
            n_total_accepted += n_accepted

            if n_accepted > 0:
                n_pairs_with_accepted += 1
                # Best accepted by margin
                accepted_margins_arr = np.where(accepted_mask, margin, -np.inf)
                pick_idx = int(np.argmax(accepted_margins_arr))

                # Compute metrics for picked query
                pair = candidates[cand_indices[pick_idx]]
                picked_query = pair["candidate_query"]
                attrs_3 = {k: (pair.get("attrs_5", {}) or pair.get("attrs", {})).get(k, "") for k in ["Brand", "Color", "Material"]}

                # Attr match (in RAW query, before hard-copy — but hard-copy was applied to candidate_query)
                attr_present_dict = attr_present(picked_query, attrs_3)
                n_attrs_present = sum(attr_present_dict.values())
                attr_match = n_attrs_present == 3  # all 3 attrs present

                # Semantic similarity: max Jaccard with user's real texts
                real_texts = real_text_by_user.get(uid, [])
                if real_texts:
                    sem_sim = max(jaccard_words(picked_query, t) for t in real_texts)
                else:
                    sem_sim = None

                # Win rate
                wr = float(d_target[pick_idx] < d_wrong[pick_idx].mean())
                # Rank
                ranks = np.argsort(np.argsort(dist_pair[pick_idx]))
                rank_pct = float(ranks[target_idx]) / n_users

                accepted_win_rates[(uid, asin)] = wr
                accepted_margins[(uid, asin)] = float(margin[pick_idx])
                accepted_ranks[(uid, asin)] = rank_pct
                accepted_attr_match[(uid, asin)] = float(attr_match)
                if sem_sim is not None:
                    accepted_sem_sim[(uid, asin)] = sem_sim

                per_pair_metrics[(uid, asin)] = {
                    "n_total": len(cand_indices),
                    "n_accepted": n_accepted,
                    "tau_u": tau_u,
                    "best_margin": float(margin[pick_idx]),
                    "best_d_target": float(d_target[pick_idx]),
                    "win_rate": wr,
                    "rank_pct": rank_pct,
                    "attr_match": float(attr_match),
                    "sem_sim": sem_sim,
                }

        # Aggregate
        coverage = n_pairs_with_accepted / max(1, n_pairs_total)
        acceptance_rate = n_total_accepted / max(1, n_total_cands)

        # Bootstrap CI for accepted-query metrics
        wr_mean, wr_ci = bootstrap_ci(accepted_win_rates) if accepted_win_rates else (0.0, (0.0, 0.0))
        margin_mean, margin_ci = bootstrap_ci(accepted_margins) if accepted_margins else (0.0, (0.0, 0.0))
        rank_mean, rank_ci = bootstrap_ci(accepted_ranks) if accepted_ranks else (0.0, (0.0, 0.0))
        attr_match_mean, attr_match_ci = bootstrap_ci(accepted_attr_match) if accepted_attr_match else (0.0, (0.0, 0.0))
        if accepted_sem_sim:
            sem_sim_mean, sem_sim_ci = bootstrap_ci(accepted_sem_sim)
        else:
            sem_sim_mean, sem_sim_ci = 0.0, (0.0, 0.0)

        results_by_percentile[tau_pct] = {
            "tau_percentile": tau_pct,
            "n_pairs_total": n_pairs_total,
            "n_pairs_with_accepted": n_pairs_with_accepted,
            "coverage_pct": coverage,
            "n_total_cands": n_total_cands,
            "n_total_accepted": n_total_accepted,
            "acceptance_rate_per_total_query": acceptance_rate,
            "accepted_query_win_rate": wr_mean,
            "accepted_query_win_rate_ci_95": list(wr_ci),
            "accepted_query_margin": margin_mean,
            "accepted_query_margin_ci_95": list(margin_ci),
            "accepted_query_rank_pct": rank_mean,
            "accepted_query_rank_pct_ci_95": list(rank_ci),
            "accepted_query_attr_match_pct": attr_match_mean,
            "accepted_query_attr_match_ci_95": list(attr_match_ci),
            "accepted_query_sem_sim": sem_sim_mean if has_real_text else None,
            "accepted_query_sem_sim_ci_95": list(sem_sim_ci) if has_real_text else None,
        }
        log(f"\n  τ P{tau_pct}:")
        log(f"    coverage: {n_pairs_with_accepted}/{n_pairs_total} = {100*coverage:.1f}%")
        log(f"    acceptance rate: {100*acceptance_rate:.1f}%")
        log(f"    accepted WR: {wr_mean:.4f} CI [{wr_ci[0]:.4f},{wr_ci[1]:.4f}]")
        log(f"    accepted margin: {margin_mean:.2f}")
        log(f"    accepted rank: {rank_mean:.4f}")
        log(f"    accepted attr_match: {100*attr_match_mean:.1f}%")
        if has_real_text:
            log(f"    accepted sem_sim: {sem_sim_mean:.4f}")

    # === Pareto frontier (simplified: best coverage*quality) ===
    log("\n[10] Pareto frontier analysis ...")
    # Quality score: WR * 100 (so 1.0 = 100)
    # Coverage score: coverage_pct (0-1)
    # Combined: coverage * quality = pairs that get good output
    best_combined = max(results_by_percentile.keys(),
                        key=lambda p: results_by_percentile[p]["coverage_pct"] *
                                      results_by_percentile[p]["accepted_query_win_rate"])
    log(f"  best combined (coverage × WR): P{best_combined}")
    log(f"    coverage={100*results_by_percentile[best_combined]['coverage_pct']:.1f}%, "
        f"WR={results_by_percentile[best_combined]['accepted_query_win_rate']:.4f}")

    # === Save eval ===
    log("\n[11] Saving eval ...")
    eval_dict = {
        "n_pairs_with_cands": len(cand_by_pair),
        "n_candidates_per_pair_avg": len(candidates) // max(1, len(cand_by_pair)),
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage",
        "shrinkage_intensity": float(shrinkage),
        "results_by_percentile": {str(k): v for k, v in results_by_percentile.items()},
        "pareto_best": f"P{best_combined}",
        "comparison": {
            "Phase10_13_C_318d_max_margin_WR": 0.7785,
            "Phase10_13_A_318d_upper_bound": 0.9786,
            "Phase10_13_C_318d_coverage_loose": 0.001,
        },
        "note": "Phase 10.14.A: τ_u calibrated from GENERATED distribution (per user, across all pairs). "
                "Acceptance: d_target <= τ_u (P_q of generated) AND margin > 0. "
                "Pareto: maximize coverage × acceptance quality (WR).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair (best percentile)
    best_p = results_by_percentile[best_combined]
    log(f"\n[12] Saving per-pair data (best P{best_combined}) ...")
    # Need to re-run per-pair for best percentile
    user_tau_best = {}
    for ui, d_list in user_d_targets_generated.items():
        user_tau_best[ui] = float(np.percentile(d_list, best_combined))

    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin), cand_indices in cand_by_pair.items():
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue
            wrong_idx = wrong_per_pair.get((uid, asin))
            if wrong_idx is None:
                continue
            tau_u = user_tau_best.get(int(target_idx))
            if tau_u is None:
                continue

            dist_pair = dist_matrix[cand_indices]
            d_target = dist_pair[:, target_idx]
            d_wrong = dist_pair[:, wrong_idx]
            masked = dist_pair.copy()
            masked[:, target_idx] = np.inf
            d_nearest_other = masked.min(axis=1)
            margin = d_nearest_other - d_target

            accepted_mask = (d_target <= tau_u) & (margin > 0)
            n_accepted = int(accepted_mask.sum())

            row = {
                "user_id": uid,
                "asin": asin,
                "tau_u": float(tau_u),
                "n_total": len(cand_indices),
                "n_accepted": n_accepted,
                "covered": n_accepted > 0,
            }
            if n_accepted > 0:
                accepted_margins_arr = np.where(accepted_mask, margin, -np.inf)
                pick_idx = int(np.argmax(accepted_margins_arr))
                row["best_margin"] = float(margin[pick_idx])
                row["best_d_target"] = float(d_target[pick_idx])
                row["win_rate"] = float(d_target[pick_idx] < d_wrong[pick_idx].mean())
                ranks = np.argsort(np.argsort(dist_pair[pick_idx]))
                row["rank_pct"] = float(ranks[target_idx]) / n_users
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.14.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()