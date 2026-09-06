#!/usr/bin/env python3
"""Stage 10 — Token-Level Typo Injection (main script, v2: single-shot + Mahalanobis).

Loads:
  - result/09_sercl_user_profile/user_sercl_profile.json        (per-user edits + mechanism histograms)
  - result/10_typo_injection/user_mahalanobis_stats.json        (per-user μ_u + Σ_u⁻¹ + D² thresholds)
  - result/08_select_query/selected_queries.json                (uid/asin/query source)

For each (uid, query) in selected_queries:
  1. Look up user's error model + 32d μ + Σ⁻¹ + D² threshold
  2. Single-shot Bernoulli(p_u_err) sampling — NO multi-seed retry
  3. If a token passes Bernoulli, sample one typo at the highest-weighted context
  4. Apply mechanism (typo or surface-form)
  5. Verify full Mahalanobis D²(z_after, μ_u) ≤ user D²_threshold
  6. If gate fails or Bernoulli produces no candidate → no_injection (faithful to
     user's actual error frequency; 81.4% rate from prior version was an artifact
     of best-of-10 selection)

Output:
  - result/10_typo_injection/typo_injection_results.json   (per-query retriever-format)
  - result/10_typo_injection/cohort_summary.json           (per-user aggregate stats,
                                                            mechanism_category breakdown)

Smoke (SMOKE=True):  5 users × ~3 queries = ~15 pairs, <30s
Full  (SMOKE=False): 217 users × ~3 queries = 376 pairs, ~60s
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_typo_injection"))

from error_location import load_sercl_profile
from injection_sampler import sample_injection

# Hardcoded paths (Rule 3)
SERCL_PROFILE = REPO_ROOT / "result/09_sercl_user_profile/user_sercl_profile.json"
MAHAL_STATS = REPO_ROOT / "result/10_typo_injection/user_mahalanobis_stats.json"
SELECTED = REPO_ROOT / "result/08_select_query/selected_queries.json"
COHORT_GATES = REPO_ROOT / "result/10_typo_injection/asin_cohort_gates.json"
OUT_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
OUT_SUMMARY = REPO_ROOT / "result/10_typo_injection/cohort_summary.json"

# Hardcoded hyperparams
SMOKE = False                   # True=5 users smoke, False=full 217 users (Rule 20)
N_SMOKE_USERS = 5
MAX_QUERIES_PER_USER = 3
SEED_BASE = 42

# D² threshold quantile (per-user). We use Q_95 = self-distance at 95th percentile
# of the user's own profile sentences — anything ≤ Q_95 is "within the user's
# style envelope". Tighter than Q_99, looser than Q_50.
D2_THRESHOLD_QUANTILE = "q95"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    log(f"loading SErCL profile: {SERCL_PROFILE}")
    profiles = load_sercl_profile(SERCL_PROFILE)
    log(f"  loaded {len(profiles)} user error models")

    with open(MAHAL_STATS) as f:
        mahal = json.load(f)
    log(f"  loaded {len(mahal)} user Mahalanobis stats")

    with open(SELECTED) as f:
        sel = json.load(f)
    log(f"  loaded {len(sel['selections'])} selections")

    with open(COHORT_GATES) as f:
        cohort = json.load(f)
    n_pairs = sum(len(c) for c in cohort.values())
    log(f"  loaded {len(cohort)} asin cohort gates, {n_pairs} (asin, uid) pairs")

    return profiles, mahal, sel, cohort


def collect_pairs(selections, mahal, max_per_user):
    pairs_by_uid = defaultdict(list)
    for s in selections:
        asin = s["asin"]
        for u in s.get("users", []):
            uid = u["uid"]
            if uid not in mahal:
                continue
            if len(pairs_by_uid[uid]) >= max_per_user:
                continue
            pairs_by_uid[uid].append((asin, u["query"]))
    pairs = []
    for uid, qs in pairs_by_uid.items():
        for asin, q in qs:
            pairs.append((uid, asin, q))
    return pairs


def main():
    t0 = time.time()
    profiles, mahal, sel, cohort = load_inputs()

    pairs = collect_pairs(sel["selections"], mahal, MAX_QUERIES_PER_USER)
    if SMOKE:
        rng = random.Random(SEED_BASE)
        rng.shuffle(pairs)
        seen_uids = set()
        smoke_pairs = []
        for p in pairs:
            if p[0] not in seen_uids:
                smoke_pairs.append(p)
                seen_uids.add(p[0])
            if len(seen_uids) >= N_SMOKE_USERS:
                break
        pairs = smoke_pairs
    log(f"running on {len(pairs)} (uid, asin, query) pairs (SMOKE={SMOKE})")

    results = []
    n_total_processed = 0
    per_user_stats = defaultdict(lambda: {
        "n_total": 0,
        "n_bernoulli_pass": 0,    # Bernoulli fired on at least one token
        "n_injected": 0,          # final injection passed all gates
        "n_gate_fail": 0,         # Bernoulli passed but D² exceeded target threshold
        "n_semantic_fail": 0,     # Gaussian passed but MiniLM sim < 0.9
        "n_exclusive_fail": 0,    # Gaussian + semantic passed but inside competitor core
        "n_no_candidate": 0,      # Bernoulli produced no candidate
        "typo_count": 0,          # keyboard_adjacent / letter_swap / letter_repetition
        "surface_form_count": 0,  # case_error / apostrophe_error
        "mechanisms": defaultdict(int),
        "d2_deltas": [],
        "sem_sims": [],
        "min_d_competitors": [],
    })
    mech_total = defaultdict(int)
    typo_total = 0
    surface_form_total = 0
    gate_fail_total = 0
    semantic_fail_total = 0
    exclusive_fail_total = 0
    bernoulli_pass_total = 0

    for i, (uid, asin, query) in enumerate(pairs):
        if uid not in profiles:
            continue
        user_model = profiles[uid]
        stats = mahal[uid]
        mu_u = np.array(stats["mu"], dtype=np.float32)
        sigma_inv = np.array(stats["sigma_inv"], dtype=np.float32)
        d2_threshold = stats[f"d2_{D2_THRESHOLD_QUANTILE}"]

        # Single-shot injection (no multi-seed retry)
        competitors = cohort.get(asin, {})
        inj, meta = sample_injection(
            query, uid, user_model, mu_u, sigma_inv, d2_threshold,
            competitors=competitors,
            seed=SEED_BASE + i,
        )

        s = per_user_stats[uid]
        s["n_total"] += 1
        n_total_processed += 1
        # Track Bernoulli pass (at least one candidate) via meta.confidence > 0
        # — if meta exists with a position even when injection failed, Bernoulli fired
        if meta is not None and meta.position >= 0:
            s["n_bernoulli_pass"] += 1
            bernoulli_pass_total += 1
        else:
            s["n_no_candidate"] += 1

        if inj is not None and meta is not None and meta.gaussian_pass and meta.semantic_pass and meta.exclusive_pass:
            s["n_injected"] += 1
            s["mechanisms"][meta.mechanism] += 1
            if meta.mechanism_category == "typo":
                s["typo_count"] += 1
                typo_total += 1
            elif meta.mechanism_category == "surface_form":
                s["surface_form_count"] += 1
                surface_form_total += 1
            s["d2_deltas"].append(meta.d_mahalanobis_after - meta.d_mahalanobis_before)
            s["sem_sims"].append(meta.semantic_sim)
            if meta.min_d_competitor > 0:
                s["min_d_competitors"].append(meta.min_d_competitor)
            mech_total[meta.mechanism] += 1
        else:
            if meta is not None and meta.position >= 0:
                if not meta.gaussian_pass:
                    s["n_gate_fail"] += 1
                    gate_fail_total += 1
                elif meta.gaussian_pass and not meta.semantic_pass:
                    s["n_semantic_fail"] += 1
                    semantic_fail_total += 1
                elif meta.gaussian_pass and meta.semantic_pass and not meta.exclusive_pass:
                    s["n_exclusive_fail"] += 1
                    exclusive_fail_total += 1

        # Only keep successfully injected pairs (skip no-candidate / gate-fail)
        if inj is not None and meta is not None and meta.gaussian_pass:
            results.append({
                "uid": uid,
                "asin": asin,
                "original_query": query,
                "typo_query": inj,
                "original_token": meta.original_token,
                "typo_token": meta.typo_token,
            })

        if (i + 1) % 50 == 0:
            log(f"  processed {i+1}/{len(pairs)} pairs")

    log(f"done. {len(results)} injected pairs (from {n_total_processed} attempted) in {time.time()-t0:.1f}s")

    # Summary
    n_injected = len(results)
    summary = {
        "config": {
            "smoke": SMOKE,
            "max_queries_per_user": MAX_QUERIES_PER_USER,
            "single_shot": True,
            "d2_threshold_quantile": D2_THRESHOLD_QUANTILE,
            "mahal_stats_source": str(MAHAL_STATS),
            "cohort_gates_source": str(COHORT_GATES),
            "semantic_threshold": 0.9,
            "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
            "note": "single-shot injection; gates: Bernoulli → Mahalanobis D²(target Q_95) "
                    "→ exclusive cohort (∀comp: d²>comp Q_95) → MiniLM cosine ≥ 0.9",
        },
        "totals": {
            "n_attempted": n_total_processed,
            "n_injected_written": n_injected,
            "n_users_injected": len(set(r["uid"] for r in results)),
            "n_bernoulli_pass": bernoulli_pass_total,
            "n_gate_fail": gate_fail_total,
            "n_semantic_fail": semantic_fail_total,
            "n_exclusive_fail": exclusive_fail_total,
            "injection_rate_among_bernoulli": n_injected / max(1, bernoulli_pass_total),
            "bernoulli_pass_rate": bernoulli_pass_total / max(1, n_total_processed),
            "typo_count": typo_total,
            "surface_form_count": surface_form_total,
            "mechanism_counts": dict(mech_total),
        },
        "per_user": {
            uid: {
                "n_total": s["n_total"],
                "n_bernoulli_pass": s["n_bernoulli_pass"],
                "n_injected": s["n_injected"],
                "n_gate_fail": s["n_gate_fail"],
                "n_semantic_fail": s["n_semantic_fail"],
                "n_exclusive_fail": s["n_exclusive_fail"],
                "n_no_candidate": s["n_no_candidate"],
                "typo_count": s["typo_count"],
                "surface_form_count": s["surface_form_count"],
                "mechanisms": dict(s["mechanisms"]),
                "d2_delta_mean": float(np.mean(s["d2_deltas"])) if s["d2_deltas"] else None,
                "sem_sim_mean": float(np.mean(s["sem_sims"])) if s["sem_sims"] else None,
                "sem_sim_min": float(min(s["sem_sims"])) if s["sem_sims"] else None,
                "min_d_competitor_mean": float(np.mean(s["min_d_competitors"])) if s["min_d_competitors"] else None,
                "min_d_competitor_min": float(min(s["min_d_competitors"])) if s["min_d_competitors"] else None,
            }
            for uid, s in per_user_stats.items()
        },
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w") as f:
        json.dump({
            "config": summary["config"],
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    log(f"wrote → {OUT_RESULTS} ({len(results)} injected pairs)")

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_SUMMARY}")

    t = summary["totals"]
    log(f"Bernoulli pass rate: {t['bernoulli_pass_rate']*100:.1f}% ({t['n_bernoulli_pass']}/{t['n_attempted']})")
    log(f"Injection rate (among Bernoulli pass): {t['injection_rate_among_bernoulli']*100:.1f}%")
    log(f"Overall injection: {t['n_injected_written']}/{t['n_attempted']} = {t['n_injected_written']/t['n_attempted']*100:.1f}%")
    log(f"Mahalanobis gate fail: {t['n_gate_fail']}")
    log(f"Exclusive cohort gate fail: {t['n_exclusive_fail']}")
    log(f"Semantic gate fail (sim<0.9): {t['n_semantic_fail']}")
    log(f"Typo: {t['typo_count']}, Surface-form: {t['surface_form_count']}")
    log(f"Mechanisms: {dict(mech_total)}")


if __name__ == "__main__":
    main()