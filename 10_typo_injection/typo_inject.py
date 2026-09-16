#!/usr/bin/env python3
"""Stage 10 — Token-Level Typo Injection (main script, v2: single-shot + Mahalanobis).

Loads:
  - result/09_sercl_user_profile/user_sercl_profile.json        (per-user edits + mechanism histograms)
  - result/04_gaussian/user_gaussian_stats.json                  (per-user μ_u + Σ_u⁻¹ + cohort gates)
  - result/08_select_query/selected_queries.json                  (uid/asin/query source)

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
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
SELECTED = REPO_ROOT / "result/08_select_query/selected_queries.json"
OUT_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
OUT_SUMMARY = REPO_ROOT / "result/10_typo_injection/cohort_summary.json"

# CONTRASTIVE_5558 ABLATION: 5558 cohort, contrastive encoder + stats
CONTRASTIVE_5558 = False
if CONTRASTIVE_5558:
    SERCL_PROFILE = REPO_ROOT / "result/09_sercl_user_profile/user_sercl_profile_5558.json"
    STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_5558.json"
    SELECTED = REPO_ROOT / "result/08_select_query/selected_queries_5558.json"
    OUT_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results_5558.json"
    OUT_SUMMARY = REPO_ROOT / "result/10_typo_injection/cohort_summary_5558.json"

# Hardcoded hyperparams
SMOKE = False                   # full run over all selected query pairs
N_SMOKE_USERS = 50
SEED_BASE = 42

# D² threshold quantile (per-user). Set dynamically from Stage 04 config.
D2_THRESHOLD_QUANTILE = None  # set in main() after load_inputs()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    log(f"loading SErCL profile: {SERCL_PROFILE}")
    profiles = load_sercl_profile(SERCL_PROFILE)
    log(f"  loaded {len(profiles)} user error models")

    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"missing: {STAGE04_PATH} (run 04_gaussian/fit_per_user_gaussian.py)"
        )
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    if not isinstance(stage04, dict):
        raise ValueError("Stage 04 artifact must be a JSON object")
    mahal = stage04.get("users")
    cohort = stage04.get("cohort_gates")
    config = stage04.get("config")
    if not isinstance(mahal, dict) or not mahal:
        raise ValueError("Stage 04 artifact requires non-empty users")
    if not isinstance(cohort, dict) or not cohort:
        raise ValueError("Stage 04 artifact requires non-empty cohort_gates")
    gate_q = config.get("gate_quantile") if isinstance(config, dict) else None
    if gate_q is None:
        raise ValueError("Stage 04 config missing gate_quantile")
    # 2026-09-15: 读理论 χ²(d, gate_quantile) = d2_q{nn}_theoretical
    # Stage 10 用低侧 rejection: gate_quantile=0.05 → d2_q05_theoretical = χ²(16, 0.05)
    d2_key = f"d2_q{int(gate_q * 100):02d}_theoretical"
    expanded_cohort = {}
    for asin, gates in cohort.items():
        if not isinstance(gates, dict) or len(gates) < 2:
            raise ValueError(f"invalid Stage 04 cohort for ASIN {asin}")
        expanded_cohort[asin] = {}
        for uid, gate in gates.items():
            if uid not in mahal:
                raise ValueError(f"cohort {asin} references missing user {uid}")
            # 2026-09-15: 严格只读 gate_T_theoretical
            if not isinstance(gate, dict) or "gate_T_theoretical" not in gate:
                raise ValueError(
                    f"cohort {asin}/{uid} missing gate_T_theoretical — "
                    "rerun 04_gaussian/rewrite_gaussian_with_theoretical_gate.py")
            user_stats = mahal[uid]
            if d2_key not in user_stats or "mu" not in user_stats \
                    or "sigma_inv" not in user_stats:
                raise ValueError(f"Stage 04 user {uid} is missing Gaussian field {d2_key}")
            if not np.isclose(float(gate["gate_T_low_theoretical"]),
                              float(user_stats[d2_key]),
                              rtol=0.0, atol=1e-5):
                raise ValueError(f"cohort gate mismatch for {asin}/{uid}")
            expanded_cohort[asin][uid] = {
                "mu": user_stats["mu"],
                "sigma_inv": user_stats["sigma_inv"],
                "gate_T": float(gate["gate_T_low_theoretical"]),
                "n": int(gate.get("n_profile", user_stats["n"])),
                "n_val": int(gate.get("n_val", user_stats.get("n_val", 0))),
            }

    with open(SELECTED) as f:
        sel = json.load(f)
    if not isinstance(sel, dict) or not isinstance(sel.get("selections"), list):
        raise ValueError("selected_queries.json requires a selections list")
    for entry in sel["selections"]:
        asin = entry.get("asin")
        if asin not in expanded_cohort:
            raise ValueError(f"selected ASIN {asin} missing from Stage 04 cohort_gates")
        for selected_user in entry.get("users", []):
            uid = selected_user.get("uid")
            if uid not in mahal or uid not in expanded_cohort[asin]:
                raise ValueError(f"selected pair ({uid}, {asin}) missing from Stage 04")
    log(f"  loaded {len(sel['selections'])} selections")
    n_pairs = sum(len(c) for c in expanded_cohort.values())
    log(f"  loaded Stage 04: {len(mahal)} users, {len(expanded_cohort)} ASIN cohort gates, "
        f"{n_pairs} pairs")

    return profiles, mahal, sel, expanded_cohort, gate_q


def collect_pairs(selections, mahal):
    pairs = []
    for s in selections:
        asin = s["asin"]
        for u in s.get("users", []):
            uid = u["uid"]
            if uid not in mahal:
                continue
            pairs.append((uid, asin, u["query"]))
    return pairs


def main():
    global D2_THRESHOLD_QUANTILE
    t0 = time.time()
    profiles, mahal, sel, cohort, gate_q = load_inputs()
    D2_THRESHOLD_QUANTILE = f"q{int(gate_q * 100):02d}_theoretical"

    pairs = collect_pairs(sel["selections"], mahal)
    for uid, asin, _ in pairs:
        if asin not in cohort or uid not in cohort[asin]:
            raise ValueError(f"selected pair ({uid}, {asin}) missing from Stage 04 cohort_gates")
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
    debug_metas = []   # collect meta from each pair for diagnosis (SMOKE only)
    n_total_processed = 0
    per_user_stats = defaultdict(lambda: {
        "n_total": 0,
        "n_bernoulli_pass": 0,    # a candidate position was identified
        "n_injected": 0,          # final injection passed all gates
        "n_gate_fail": 0,         # candidate identified but D² exceeded target
        "n_semantic_fail": 0,     # Gaussian passed but MiniLM sim < 0.9
        "n_exclusive_fail": 0,    # Gaussian + semantic passed but inside competitor core
        "n_no_candidate": 0,      # no candidate (user has zero char-level history)
        "n_surface_form_only_skip": 0,  # user has only case_error/apostrophe history
        "n_minimality_fail": 0,   # edit distance / len delta / valid-word check failed
        "typo_count": 0,          # char-level typos (keyboard_adjacent / letter_swap / ...)
        "mechanisms": defaultdict(int),
        "transformation_sources": defaultdict(int),  # user_historical_full/d3/.../generic_fallback
        "edit_distances": [],
        "d2_deltas": [],
        "sem_sims": [],
        "min_d_competitors": [],
    })
    mech_total = defaultdict(int)
    typo_total = 0
    gate_fail_total = 0
    semantic_fail_total = 0
    exclusive_fail_total = 0
    bernoulli_pass_total = 0
    surface_form_only_skip_total = 0
    minimality_fail_total = 0

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
        # Track char-level Bernoulli pass — meta exists with a position means we
        # identified a candidate position. The new sampler always produces one
        # position (top-scored by char-level rate) so this is essentially 1.0
        # unless the user has zero char-level history.
        if meta is not None and meta.position >= 0:
            s["n_bernoulli_pass"] += 1
            bernoulli_pass_total += 1
        else:
            s["n_no_candidate"] += 1
            if meta is not None and meta.transformation_source == "surface_form_only":
                s["n_surface_form_only_skip"] += 1

        if inj is not None and meta is not None and meta.gaussian_pass and meta.semantic_pass and meta.exclusive_pass:
            s["n_injected"] += 1
            s["mechanisms"][meta.mechanism] += 1
            # mechanism is always char-level (CHAR_LEVEL_MECHANISMS) — surface-form
            # never reaches this point
            s["typo_count"] += 1
            typo_total += 1
            s["d2_deltas"].append(meta.d_mahalanobis_after - meta.d_mahalanobis_before)
            s["sem_sims"].append(meta.semantic_sim)
            if meta.min_d_competitor > 0:
                s["min_d_competitors"].append(meta.min_d_competitor)
            mech_total[meta.mechanism] += 1
            s["transformation_sources"][meta.transformation_source] += 1
            s["edit_distances"].append(meta.edit_distance)
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
                else:
                    # minimality gate failed
                    s["n_minimality_fail"] = s.get("n_minimality_fail", 0) + 1

        # Only keep successfully injected pairs (skip no-candidate / gate-fail)
        # Results now contain only CHAR-LEVEL typos (case_error / apostrophe_error
        # are not emitted by the sampler).
        if inj is not None and meta is not None and meta.gaussian_pass:
            results.append({
                "uid": uid,
                "asin": asin,
                "original_query": query,
                "typo_query": inj,
                "original_token": meta.original_token,
                "typo_token": meta.typo_token,
                "mechanism": meta.mechanism,
                "transformation_source": meta.transformation_source,
                "edit_distance": meta.edit_distance,
                "len_delta": meta.len_delta,
                "context_sig": meta.context_sig,
                "confidence": meta.confidence,
                "semantic_sim": meta.semantic_sim,
            })

        # SMOKE: keep all meta for diagnosis
        if SMOKE and meta is not None:
            debug_metas.append({
                "uid": uid,
                "asin": asin,
                "position": meta.position,
                "original_token": meta.original_token,
                "typo_token": meta.typo_token,
                "mechanism": meta.mechanism,
                "transformation_source": meta.transformation_source,
                "edit_distance": meta.edit_distance,
                "len_delta": meta.len_delta,
                "d2_before": meta.d_mahalanobis_before,
                "d2_after": meta.d_mahalanobis_after,
                "d2_threshold": stats.get(f"d2_{D2_THRESHOLD_QUANTILE}"),
                "gaussian_pass": meta.gaussian_pass,
                "semantic_sim": meta.semantic_sim,
                "semantic_pass": meta.semantic_pass,
                "min_d_competitor": meta.min_d_competitor,
                "n_competitors": meta.n_competitors,
                "exclusive_pass": meta.exclusive_pass,
            })

        if (i + 1) % 50 == 0:
            log(f"  processed {i+1}/{len(pairs)} pairs")

    log(f"done. {len(results)} injected pairs (from {n_total_processed} attempted) in {time.time()-t0:.1f}s")

    # Summary
    n_injected = len(results)
    summary = {
        "config": {
            "smoke": SMOKE,
            "all_selected_queries": True,
            "single_shot": True,
            "d2_threshold_quantile": D2_THRESHOLD_QUANTILE,
            "mahal_stats_source": str(STAGE04_PATH),
            "cohort_gates_source": str(STAGE04_PATH),
            "semantic_threshold": 0.9,
            "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
            "char_level_mechanisms": ["keyboard_adjacent", "letter_swap", "letter_repetition",
                                       "letter_insertion", "letter_deletion"],
            "min_token_len": 3,
            "max_edit_distance": 2,
            "max_len_delta": 1,
            "valid_word_check": True,
            "note": "char-level-only single-shot injection. Position ranked by P_u(char_level_error|context). "
                    "Transformation reuses user-historical char-level typo at the same (orig_token, sig) "
                    "if available, else generic char-level mechanism. Gates: minimality (ed≤2, |Δlen|≤1, "
                    "len≥3, not a different English word) → Mahalanobis D²(target Q_95) → exclusive cohort "
                    "(∀comp: d²>comp Q_95) → MiniLM cosine ≥ 0.9. Surface-form errors (case_error / "
                    "apostrophe_error) are NOT emitted — they are tracked separately via "
                    "n_surface_form_only_skip when user has only surface-form history.",
        },
        "totals": {
            "n_attempted": n_total_processed,
            "n_injected_written": n_injected,
            "n_users_injected": len(set(r["uid"] for r in results)),
            "n_bernoulli_pass": bernoulli_pass_total,
            "n_gate_fail": gate_fail_total,
            "n_semantic_fail": semantic_fail_total,
            "n_exclusive_fail": exclusive_fail_total,
            "n_minimality_fail": minimality_fail_total,
            "n_surface_form_only_skip": surface_form_only_skip_total,
            "injection_rate_among_bernoulli": n_injected / max(1, bernoulli_pass_total),
            "bernoulli_pass_rate": bernoulli_pass_total / max(1, n_total_processed),
            "typo_count": typo_total,
            "mechanism_counts": dict(mech_total),
            "transformation_source_counts": {
                src: sum(s.get("transformation_sources", {}).get(src, 0)
                         for s in per_user_stats.values())
                for src in ["user_historical_full", "user_historical_d3",
                            "user_historical_coarse", "user_historical_any",
                            "generic_fallback"]
            },
        },
        "per_user": {
            uid: {
                "n_total": s["n_total"],
                "n_bernoulli_pass": s["n_bernoulli_pass"],
                "n_injected": s["n_injected"],
                "n_gate_fail": s["n_gate_fail"],
                "n_semantic_fail": s["n_semantic_fail"],
                "n_exclusive_fail": s["n_exclusive_fail"],
                "n_minimality_fail": s.get("n_minimality_fail", 0),
                "n_no_candidate": s["n_no_candidate"],
                "n_surface_form_only_skip": s["n_surface_form_only_skip"],
                "typo_count": s["typo_count"],
                "mechanisms": dict(s["mechanisms"]),
                "transformation_sources": dict(s["transformation_sources"]),
                "edit_distance_mean": float(np.mean(s["edit_distances"])) if s["edit_distances"] else None,
                "edit_distance_max": int(max(s["edit_distances"])) if s["edit_distances"] else None,
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
    log(f"wrote → {OUT_RESULTS} ({len(results)} char-level injected pairs)")

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_SUMMARY}")

    if SMOKE:
        debug_path = OUT_RESULTS.parent / "smoke_debug_metas.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_metas, f, indent=2, ensure_ascii=False)
        log(f"wrote → {debug_path} ({len(debug_metas)} meta records)")

    t = summary["totals"]
    log(f"Bernoulli pass rate: {t['bernoulli_pass_rate']*100:.1f}% ({t['n_bernoulli_pass']}/{t['n_attempted']})")
    log(f"Injection rate (among Bernoulli pass): {t['injection_rate_among_bernoulli']*100:.1f}%")
    log(f"Overall injection: {t['n_injected_written']}/{t['n_attempted']} = {t['n_injected_written']/t['n_attempted']*100:.1f}%")
    log(f"Surface-form-only skip: {t['n_surface_form_only_skip']}")
    log(f"Minimality fail: {t['n_minimality_fail']}")
    log(f"Mahalanobis gate fail: {t['n_gate_fail']}")
    log(f"Exclusive cohort gate fail: {t['n_exclusive_fail']}")
    log(f"Semantic gate fail (sim<0.9): {t['n_semantic_fail']}")
    log(f"Typo (char-level): {t['typo_count']}")
    log(f"Mechanisms: {dict(mech_total)}")
    log(f"Transformation sources: {t['transformation_source_counts']}")


if __name__ == "__main__":
    main()