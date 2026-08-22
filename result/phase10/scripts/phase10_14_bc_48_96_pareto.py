#!/usr/bin/env python3
"""Phase 10.14.B + C: Generate 96 candidates/pair and evaluate at 24/48/96 levels.

Plan:
  1. Generate 12 conds × 8 seeds × 876 pairs = 84,096 candidates
  2. Extract 318d features (spaCy batch)
  3. Compute per-pair 318d Mahalanobis (chunked to avoid 2GB matrix)
  4. Acceptance at 3 levels (24/48/96 cands/pair):
     - τ from generated distribution
     - acceptance: d_target <= τ AND d_target < d_wrong_mean (K=5)
  5. Pareto frontier: coverage vs quality

Generation: ~7 min at 207 prompts/s.
318d extraction: ~5 min spaCy batch.
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
OUT_CANDIDATES = OUT_DIR / "phase10_14_bc_candidates_96.jsonl"
OUT_CAND_FEATS_CACHE = OUT_DIR / "phase10_14_bc_candidates_318d.npy"
OUT_EVAL = OUT_DIR / "phase10_14_bc_pareto_eval.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

# === Generation params ===
N_CONDITIONS = 12
N_SEEDS = 8  # 8 seeds × 12 conds = 96 candidates/pair
# Total: 12 × 8 × 876 = 84,096 candidates

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

# === Same 12 SYNTAX_CONDITIONS as Phase 10.12.A ===
# (imported to keep consistency)
import phase10_12_a_syntactic_diversity as mod_gen

SYNTAX_CONDITIONS = mod_gen.SYNTAX_CONDITIONS


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
    log = lambda m: print(f"[phase10-14.BC] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.14.B + C: 96 candidates/pair, Pareto at 24/48/96")
    log("=" * 70)

    # === Generation ===
    if OUT_CANDIDATES.exists() and OUT_CAND_FEATS_CACHE.exists():
        log(f"[1] Loading existing candidates and features ...")
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
    else:
        # === Step 1: Generate candidates ===
        log(f"[1] Loading pairs ...")
        pairs = []
        with PAIRS_FILE.open() as f:
            for line in f:
                pairs.append(json.loads(line))
        valid_pairs = []
        for p in pairs:
            attrs = p.get("attrs_5", {}) or p.get("attrs", {})
            if all(k in attrs and attrs[k] for k in ["Brand", "Color", "Material"]):
                valid_pairs.append(p)
        log(f"  valid pairs: {len(valid_pairs)}")

        log(f"[2] Building {len(valid_pairs)} × {N_CONDITIONS} × {N_SEEDS} = {len(valid_pairs)*N_CONDITIONS*N_SEEDS} prompts ...")
        all_prompts = []
        for pair_idx, p in enumerate(valid_pairs):
            attrs = p.get("attrs_5", {}) or p.get("attrs", {})
            for cond_idx, cond in enumerate(SYNTAX_CONDITIONS):
                for seed_idx in range(N_SEEDS):
                    prompt = mod_gen.build_prompt(attrs, attrs, cond)
                    all_prompts.append((pair_idx, cond_idx, seed_idx, prompt))
        total = len(all_prompts)
        log(f"  total prompts: {total}")

        log(f"[3] Generating via vllm ...")
        from llm_client import create_qwen_local_client
        client = create_qwen_local_client()
        from vllm import SamplingParams
        sampling = SamplingParams(
            max_tokens=120,
            temperature=0.9,
            top_p=0.95,
            stop=["\n\n", "Product attributes:", "STYLE REQUIREMENTS:"],
        )
        backend = client._backend
        if backend is None:
            raise RuntimeError("vllm backend not initialized")

        prompts_only = [p[3] for p in all_prompts]
        t0 = time.time()
        outputs = backend.model.generate(prompts_only, sampling)
        log(f"  vllm returned {len(outputs)} in {time.time()-t0:.1f}s")

        log(f"[4] Building candidates + hard-copy ...")
        candidates = []
        n_failed = 0
        n_with_append = 0
        for i, out in enumerate(outputs):
            pair_idx, cond_idx, seed_idx, _ = all_prompts[i]
            pair = valid_pairs[pair_idx]
            if not out.outputs:
                result = ""
                n_failed += 1
            else:
                result = out.outputs[0].text.strip()
                if not result:
                    n_failed += 1

            attrs_3 = {k: (pair.get("attrs_5", {}) or pair.get("attrs", {})).get(k, "") for k in ["Brand", "Color", "Material"]}
            needed_append = False
            if result and mod_gen.needs_hard_copy(result, attrs_3):
                result = mod_gen.hard_copy(result, attrs_3)
                needed_append = True
                n_with_append += 1

            candidates.append({
                "user_id": pair["user_id"],
                "asin": pair["asin"],
                "cond_idx": cond_idx,
                "seed_idx": seed_idx,
                "candidate_query": result,
                "attrs": pair.get("attrs", {}),
                "attrs_5": pair.get("attrs_5") or pair.get("attrs", {}),
                "needed_append": needed_append,
            })
        log(f"  candidates: {len(candidates)}, failed: {n_failed}, hard-copy: {n_with_append}")

        log(f"[5] Saving candidates ...")
        with OUT_CANDIDATES.open("w") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        log(f"[6] Extracting 318d features via spaCy batch ...")
        from extract_syntactic_features import (
            ALL_FEATS_V2, per_sentence_features_v2, user_features_v2,
        )
        import spacy
        nlp = spacy.load("en_core_web_sm")

        cand_queries = [c["candidate_query"] for c in candidates]
        cand_feats = np.zeros((len(cand_queries), len(ALL_FEATS_V2)), dtype=np.float32)
        n_success = 0
        t0 = time.time()
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
            if (start // BATCH_SIZE) % 10 == 0:
                log(f"  [{start}/{len(cand_queries)}] elapsed {time.time()-t0:.1f}s")
        log(f"  extracted {n_success}/{len(cand_queries)} in {time.time()-t0:.1f}s")

        # Apply keep_dims filter
        sent_feats_tmp = np.load(SENT_FEATS_CACHE)
        stds = sent_feats_tmp.std(axis=0)
        keep_dims = stds > 1e-9
        cand_feats_keep = cand_feats[:, keep_dims].astype(np.float64)
        log(f"  effective dims: {cand_feats_keep.shape[1]}")

        np.save(OUT_CAND_FEATS_CACHE, cand_feats_keep)
        log(f"  saved → {OUT_CAND_FEATS_CACHE}")

    # === Step 2: Distance computation (chunked) ===
    log("[*] Computing per-pair distances ...")
    log("[2] Loading caches and profiles ...")
    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    n_dims_eff = cand_feats_keep.shape[1]

    log(f"  n_dims_eff: {n_dims_eff}")

    # Per-user mean features
    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for si in range(len(sent_users)):
        ui = user_id_to_idx.get(sent_users[si])
        if ui is None:
            continue
        user_mu_318d[ui] += sent_feats[:, np.array([s > 1e-9 for s in sent_feats.std(axis=0)])][si].astype(np.float64)
        counts[ui] += 1
    valid_user_mask = counts > 0
    user_mu_318d[valid_user_mask] /= counts[valid_user_mask, None]

    # Actually let me re-do this cleanly
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
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

    # Ledoit-Wolf
    log("[3] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))

    # Per-pair distances (chunked to avoid 2GB matrix)
    log("[4] Computing per-pair distances (chunked) ...")
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs: {len(cand_by_pair)}")

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

    # Pre-compute quad_mu (depends only on user_mu_318d and inv_cov)
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)

    # For each pair, compute distance to all users
    log("[5] Per-pair distance computation ...")
    pair_dist_data = {}  # (uid, asin) -> {d_target: ndarray, d_wrong: ndarray, cand_indices: list}
    t0 = time.time()
    for pi, ((uid, asin), cand_indices) in enumerate(cand_by_pair.items()):
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue

        cands_feats_local = cand_feats_keep[cand_indices]
        quad_cand = np.einsum('ij,jk,ik->i', cands_feats_local, inv_cov, cands_feats_local)
        cross = cands_feats_local @ inv_cov @ user_mu_318d.T  # (n_cands, n_users)
        # Distances (n_cands, n_users)
        dist_local = quad_cand[:, None] + quad_mu[None, :] - 2 * cross

        pair_dist_data[(uid, asin)] = {
            "d_target": dist_local[:, target_idx],  # (n_cands,)
            "d_wrong": dist_local[:, wrong_idx],  # (n_cands, K=5)
            "all_dists": dist_local,  # (n_cands, n_users) for rank
            "cand_indices": cand_indices,
        }

        if pi % 100 == 0:
            log(f"  [{pi}/{len(cand_by_pair)}] elapsed {time.time()-t0:.1f}s")

    log(f"  done in {time.time()-t0:.1f}s")

    # === Per-user generated distribution (using all 96 candidates) ===
    log("[6] Per-user generated d_target distribution ...")
    user_d_targets = defaultdict(list)
    for (uid, asin), data in pair_dist_data.items():
        for d in data["d_target"]:
            user_d_targets[uid].append(float(d))

    # === Pareto analysis at multiple candidate counts ===
    log("[7] Pareto at 24/48/96 ...")
    candidate_counts = [24, 48, 96]
    TAU_PERCENTILES = [50, 60, 70, 75, 80, 90, 95]

    pareto_results = {}

    for n_cands in candidate_counts:
        log(f"\n  === N candidates = {n_cands} ===")

        # Per-user τ from first n_cands of each pair (using generated d_target distribution)
        # Actually for τ, use user's overall generated distribution (all 96)
        # The threshold is per-user, regardless of how many cands we evaluate

        results_by_pct = {}
        for tau_pct in TAU_PERCENTILES:
            user_tau = {}
            for uid, d_list in user_d_targets.items():
                user_tau[uid] = float(np.percentile(d_list, tau_pct))

            n_pairs_total = 0
            n_pairs_with_accepted = 0
            n_total_cands_used = 0
            n_total_accepted = 0

            # Acceptance-quality metrics
            accepted_margins = {}
            accepted_ranks = {}

            for (uid, asin), data in pair_dist_data.items():
                target_idx = user_id_to_idx.get(uid)
                if target_idx is None:
                    continue
                tau_u = user_tau.get(uid)
                if tau_u is None:
                    continue
                n_pairs_total += 1

                # Use first n_cands of pair
                cand_indices = data["cand_indices"][:n_cands]
                d_target = data["d_target"][:n_cands]
                d_wrong = data["d_wrong"][:n_cands]
                d_wrong_mean = d_wrong.mean(axis=1)
                margin = d_wrong_mean - d_target
                all_dists_local = data["all_dists"][:n_cands]
                n_total_cands_used += len(cand_indices)

                accepted_mask = (d_target <= tau_u) & (d_target < d_wrong_mean)
                n_accepted = int(accepted_mask.sum())
                n_total_accepted += n_accepted

                if n_accepted > 0:
                    n_pairs_with_accepted += 1
                    accepted_margins_arr = np.where(accepted_mask, margin, -np.inf)
                    pick_idx = int(np.argmax(accepted_margins_arr))

                    ranks = np.argsort(np.argsort(all_dists_local[pick_idx]))
                    rank_pct = float(ranks[target_idx]) / n_users

                    accepted_margins[(uid, asin)] = float(margin[pick_idx])
                    accepted_ranks[(uid, asin)] = rank_pct

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

        # Per-level summary
        for tau_pct in [50, 75, 90, 95]:
            r = results_by_pct[tau_pct]
            log(f"    P{tau_pct}: coverage={100*r['coverage_pct']:.1f}%, "
                f"acc_rate={100*r['acceptance_rate_per_total_query']:.1f}%, "
                f"margin={r['accepted_query_margin']:.1f}, rank={100*r['accepted_query_rank_pct']:.1f}%")

    # === Save eval ===
    log("\n[8] Saving eval ...")
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
    log("PHASE 10.14.B + C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()