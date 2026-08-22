#!/usr/bin/env python3
"""Phase 10.13.C: 318d Full Rerank with Phase 10.12.A Candidates.

Phase 10.12.A 有 21,024 候选 (12 句法条件 × 2 seeds × 876 pairs).
Phase 10.13.A-318d 上限 97.86% WR.
Phase 10.12.B-20d 67.24% WR (92.5% of 20d ceiling).

目标: 验证 318d rerank with 24 候选/pair → win rate 接近 97.86% 上限.
   + acceptance/coverage with calibrated per-user thresholds.

实施:
  1. 加载 Phase 10.12.A 21,024 候选
  2. 批量提取 318d 特征 (spaCy nlp.pipe, ~3-5 min)
  3. 计算 318d Mahalanobis 距离 (Ledoit-Wolf shrinkage)
  4. Rerank 3 策略 (A_min_d_target / B_max_margin / C_weighted)
  5. 报告 win rate / rank / gap + acceptance/coverage
"""
from __future__ import annotations

import json
import os
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
OUT_CAND_FEATS_CACHE = OUT_DIR / "phase10_12_a_candidates_318d.npy"
OUT_EVAL = OUT_DIR / "phase10_13_c_318d_full_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase10_13_c_318d_per_pair.jsonl"

# Phase 10.12.A candidates
IN_CANDIDATES = OUT_DIR / "phase10_12_a_candidates_syntactic.jsonl"

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
    log = lambda m: print(f"[phase10-13.C-318d] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.13.C (318d): Full Rerank with Phase 10.12.A Candidates")
    log("=" * 70)

    # === Load 318d sentence cache for normalization ===
    log("[1] Loading 318d sentence cache ...")
    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    log(f"  effective dims: {n_dims_eff}")

    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)

    # === Load VADES profiles ===
    log("[2] Loading VADES profiles ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  users: {n_users}")

    # === Per-user mean features ===
    log("[3] Per-user mean features ...")
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

    # === Ledoit-Wolf covariance ===
    log("[4] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage: {shrinkage:.4f}")
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))

    # === Load Phase 10.12.A candidates ===
    log(f"[5] Loading Phase 10.12.A candidates from {IN_CANDIDATES} ...")
    candidates = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")

    cand_queries = [c["candidate_query"] for c in candidates]

    # === Extract 318d features (or load cache) ===
    if OUT_CAND_FEATS_CACHE.exists():
        log(f"[6] Loading cached 318d candidate features from {OUT_CAND_FEATS_CACHE} ...")
        cand_feats_keep = np.load(OUT_CAND_FEATS_CACHE)
        log(f"  shape: {cand_feats_keep.shape}")
    else:
        log(f"[6] Extracting 318d features for {len(cand_queries)} candidates via spaCy batch ...")
        from extract_syntactic_features import (
            ALL_FEATS_V2, per_sentence_features_v2, user_features_v2,
        )
        import spacy

        nlp = spacy.load("en_core_web_sm")

        cand_feats = np.zeros((len(cand_queries), len(ALL_FEATS_V2)), dtype=np.float32)
        t0 = time.time()
        n_success = 0
        BATCH_SIZE = 256
        for start in range(0, len(cand_queries), BATCH_SIZE):
            batch = cand_queries[start:start + BATCH_SIZE]
            for i, doc in enumerate(nlp.pipe(batch)):
                try:
                    sfs = [per_sentence_features_v2(s) for s in doc.sents]
                    v = user_features_v2(sfs)
                    if v is not None:
                        cand_feats[start + i] = v
                        n_success += 1
                except Exception:
                    pass
            if (start // BATCH_SIZE) % 5 == 0:
                log(f"  [{start}/{len(cand_queries)}] elapsed {time.time()-t0:.1f}s")

        log(f"  extracted {n_success}/{len(cand_queries)} in {time.time()-t0:.1f}s")
        log(f"  saving cache to {OUT_CAND_FEATS_CACHE}")
        # Drop zero-variance dims
        cand_feats_keep = cand_feats[:, keep_dims].astype(np.float64)
        np.save(OUT_CAND_FEATS_CACHE, cand_feats_keep)

    # === Distance matrix ===
    log("[7] Computing distance matrix (candidates × users) ...")
    quad_cand = np.einsum('ij,jk,ik->i', cand_feats_keep, inv_cov, cand_feats_keep)
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)
    cross = cand_feats_keep @ inv_cov @ user_mu_318d.T
    dist_matrix = quad_cand[:, None] + quad_mu[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}")

    # Group candidates by pair
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

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

    # === Per-pair: 3 rerank strategies ===
    log("[8] Computing per-pair metrics ...")
    results = {
        "A_min_d_target": {"gap": {}, "wr": {}, "rank": {}},
        "B_max_margin": {"gap": {}, "wr": {}, "rank": {}},
        "C_weighted": {"gap": {}, "wr": {}, "rank": {}},
    }
    coverage_by_threshold = {
        "loose_m0": {"n_pairs_with_accepted": 0, "n_total_cands": 0, "n_total_accepted": 0},
        "medium_m0": {"n_pairs_with_accepted": 0, "n_total_cands": 0, "n_total_accepted": 0},
    }

    # Calibrated per-user thresholds from real reviews
    log("[8.5] Computing per-user thresholds from real reviews ...")
    sent_user_idx = np.array([user_id_to_idx.get(sent_users[si], -1) for si in range(len(sent_users))])
    sent_quad = np.einsum('ij,jk,ik->i', sent_feats_keep, inv_cov, sent_feats_keep)
    sent_cross = sent_feats_keep @ inv_cov @ user_mu_318d.T

    wrong_per_user_real = {}
    for ui in np.where(valid_user_mask)[0]:
        mask = np.ones(n_users, dtype=bool)
        mask[ui] = False
        wrong_per_user_real[int(ui)] = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)

    per_user_d_targets = defaultdict(list)
    for si in range(len(sent_users)):
        ui = sent_user_idx[si]
        if ui < 0 or not valid_user_mask[ui]:
            continue
        wrong_idx = wrong_per_user_real.get(int(ui))
        if wrong_idx is None:
            continue
        d_target = float(sent_quad[si] + quad_mu[ui] - 2 * sent_cross[si, ui])
        per_user_d_targets[int(ui)].append(d_target)

    user_tau_P75 = {ui: float(np.percentile(per_user_d_targets[ui], 75)) for ui in per_user_d_targets.keys()}
    median_tau = float(np.median(list(user_tau_P75.values())))
    log(f"  median τ (P75): {median_tau:.2f}")

    n_pairs_with_cands = 0
    for (uid, asin), cand_indices in cand_by_pair.items():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue
        n_pairs_with_cands += 1
        dist_pair = dist_matrix[cand_indices]
        d_target = dist_pair[:, target_idx]
        d_wrong = dist_pair[:, wrong_idx]

        masked = dist_pair.copy()
        masked[:, target_idx] = np.inf
        d_nearest_other = masked.min(axis=1)
        margin = d_nearest_other - d_target

        for strategy, score_fn in [
            ("A_min_d_target", lambda: np.argmin(d_target)),
            ("B_max_margin", lambda: np.argmax(margin)),
            ("C_weighted", lambda: np.argmin(-margin + d_target)),
        ]:
            pick_idx = score_fn()
            pick_d_target = float(d_target[pick_idx])
            pick_d_wrong = float(d_wrong[pick_idx].mean())
            results[strategy]["gap"][(uid, asin)] = pick_d_target - pick_d_wrong
            results[strategy]["wr"][(uid, asin)] = float(pick_d_target < pick_d_wrong)
            ranks = np.argsort(np.argsort(dist_pair[pick_idx]))
            results[strategy]["rank"][(uid, asin)] = float(ranks[target_idx]) / n_users

        # Acceptance check
        tau_u = user_tau_P75.get(int(target_idx), median_tau)
        n_accepted_loose = sum(1 for ci in cand_indices if d_target[cand_indices.index(ci)] <= tau_u and margin[cand_indices.index(ci)] > 0)
        # Faster vectorized
        d_target_vec = d_target
        margin_vec = margin
        n_accepted_loose = int(np.sum((d_target_vec <= tau_u) & (margin_vec > 0)))
        coverage_by_threshold["loose_m0"]["n_total_cands"] += len(cand_indices)
        coverage_by_threshold["loose_m0"]["n_total_accepted"] += n_accepted_loose
        if n_accepted_loose > 0:
            coverage_by_threshold["loose_m0"]["n_pairs_with_accepted"] += 1

        # Medium: τ = P50 (more strict)
        tau_u_med = float(np.percentile(per_user_d_targets.get(int(target_idx), [median_tau]), 50))
        n_accepted_med = int(np.sum((d_target_vec <= tau_u_med) & (margin_vec > 0)))
        coverage_by_threshold["medium_m0"]["n_total_cands"] += len(cand_indices)
        coverage_by_threshold["medium_m0"]["n_total_accepted"] += n_accepted_med
        if n_accepted_med > 0:
            coverage_by_threshold["medium_m0"]["n_pairs_with_accepted"] += 1

    log(f"  pairs evaluated: {n_pairs_with_cands}")

    # === Aggregate ===
    log("[9] Aggregating ...")
    summary = {}
    for strategy in results:
        m_gap, ci_gap = bootstrap_ci(results[strategy]["gap"])
        m_wr, ci_wr = bootstrap_ci(results[strategy]["wr"])
        m_tr, ci_tr = bootstrap_ci(results[strategy]["rank"])
        summary[strategy] = {
            "n_pairs": len(results[strategy]["gap"]),
            "gap_mean": m_gap,
            "gap_ci_95": list(ci_gap),
            "win_rate_mean": m_wr,
            "win_rate_ci_95": list(ci_wr),
            "target_rank_pct_mean": m_tr,
            "target_rank_pct_ci_95": list(ci_tr),
            "gap_pass": bool(ci_gap[0] > 0),
            "wr_pass_60pct": bool(ci_wr[0] > 0.6),
            "rank_pass_50pct": bool(ci_tr[1] < 0.5),
        }
        log(f"\n  Strategy {strategy}:")
        log(f"    Gap: mean={m_gap:.2f}, CI=[{ci_gap[0]:.2f},{ci_gap[1]:.2f}] {'PASS' if ci_gap[0]>0 else 'FAIL'}")
        log(f"    Win rate: mean={m_wr:.4f}, CI=[{ci_wr[0]:.4f},{ci_wr[1]:.4f}] {'PASS' if ci_wr[0]>0.6 else 'FAIL'}")
        log(f"    Target rank: mean={m_tr:.4f}, CI=[{ci_tr[0]:.4f},{ci_tr[1]:.4f}] {'PASS' if ci_tr[1]<0.5 else 'FAIL'}")

    # Coverage
    coverage_eval = {}
    for name, info in coverage_by_threshold.items():
        n_total = info["n_total_cands"]
        n_accepted = info["n_total_accepted"]
        coverage_pct = info["n_pairs_with_accepted"] / max(1, n_pairs_with_cands)
        acceptance_rate_per_total = n_accepted / max(1, n_total)
        coverage_eval[name] = {
            "n_pairs_with_accepted": info["n_pairs_with_accepted"],
            "n_pairs_total": n_pairs_with_cands,
            "coverage_pct": coverage_pct,
            "n_total_cands": n_total,
            "n_total_accepted": n_accepted,
            "acceptance_rate_per_total": acceptance_rate_per_total,
        }
        log(f"\n  Threshold '{name}':")
        log(f"    coverage: {info['n_pairs_with_accepted']}/{n_pairs_with_cands} = {100*coverage_pct:.1f}%")
        log(f"    acceptance per total query: {100*acceptance_rate_per_total:.1f}%")

    # === Save eval ===
    log("\n[10] Saving eval ...")
    eval_dict = {
        "n_pairs_with_cands": n_pairs_with_cands,
        "n_candidates_per_pair_avg": len(candidates) // max(1, n_pairs_with_cands),
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage",
        "shrinkage_intensity": float(shrinkage),
        "summary_per_strategy": summary,
        "coverage": coverage_eval,
        "comparison": {
            "Phase10_13_A_318d_upper_bound": 0.9786,
            "Phase10_13_A_20d_upper_bound": 0.7268,
            "Phase10_12_B_20d_generated": 0.6724,
            "Phase10_13_B_318d_10cands_coverage_loose": 0.32,
        },
        "note": "Phase 10.13.C: 318d rerank using Phase 10.12.A candidates (24/pair). "
                "Real review ceiling in 318d = 97.86% WR. "
                "Goal: see if 24-cand pool + 318d space approaches ceiling. "
                "Coverage evaluated with per-user thresholds from real reviews.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in results["B_max_margin"]["gap"].keys():
            row = {
                "user_id": uid,
                "asin": asin,
                "A_gap": results["A_min_d_target"]["gap"][(uid, asin)],
                "A_wr": results["A_min_d_target"]["wr"][(uid, asin)],
                "A_rank": results["A_min_d_target"]["rank"][(uid, asin)],
                "B_gap": results["B_max_margin"]["gap"][(uid, asin)],
                "B_wr": results["B_max_margin"]["wr"][(uid, asin)],
                "B_rank": results["B_max_margin"]["rank"][(uid, asin)],
                "C_gap": results["C_weighted"]["gap"][(uid, asin)],
                "C_wr": results["C_weighted"]["wr"][(uid, asin)],
                "C_rank": results["C_weighted"]["rank"][(uid, asin)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.13.C (318d) COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()