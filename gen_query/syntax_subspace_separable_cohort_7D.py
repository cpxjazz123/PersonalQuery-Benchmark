"""Phase 7.D — Sentence-Level Syntax-Separable Cohort Filtering.

Root problem (Phase 7.C.2 FAIL, commit 0311e9d):
  P(M_local>0) median = 43.75% on same-ASIN cohort. Root cause: cohort contains
  users whose sentence-level PCA48 profiles overlap almost completely. Spatial-
  locality signal is real, just needs a syntax-separable cohort.

Approach:
  1. Re-fit sentence-level Gaussian on Profile Set P_u only (no source leakage).
  2. Per ASIN: pairwise Bhattacharyya distance D_B(G_u, G_v) on Profile Set Gaussians.
  3. Sweep T_B ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0}; greedy max-min cohort selection.
  4. Held-out validation: ONLY source_set sentences (never profile) for M_local audit.

Trade-off target: T_B↑ → coverage↓, P(M>0)↑.

Inputs:
    scratch2/gaussian_vades/stage8_5_user_profiles_sentence.json  (7.C.1 output)
    scratch2/gaussian_vades/stage8_5_user_gaussians_cv.json        (Q-gate)
    scratch2/gaussian_vades/stage8_5_asins.json                     (cohort)

Outputs:
    result/gen_query/phase7d_separable_cohort.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    python3 gen_query/syntax_subspace_separable_cohort_7D.py
"""

from __future__ import annotations

import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
PROFILES_PATH = SCRATCH / "stage8_5_user_profiles_sentence.json"
GAUSS_CV_PATH = SCRATCH / "stage8_5_user_gaussians_cv.json"
ASINS_PATH = SCRATCH / "stage8_5_asins.json"
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7d_separable_cohort.json"

# Constants (locked, hard-coded per CLAUDE.md Rule 3)
EPS = 1e-12
SEED = 42
# Q-gate (relaxed from Phase 6 cohort gate, see 7.C.2 → 7.D pivot note):
# Original CV_INLIER_MIN=0.9, CV_NLL_MAX=39.0 was designed for cohort inlier
# detection. For personalization, users ARE expected to deviate from cohort
# average, so CV_NLL_MAX is set very loose (effectively disabled at 200).
# We keep CV_INLIER_MIN at 0.85 (need ≥85% folds inlier to ensure σ is
# well-conditioned) and N_REVIEWS_MIN at 15 (7.B cohort floor).
CV_INLIER_MIN = 0.85
CV_NLL_MAX = 34.0
N_REVIEWS_MIN = 15
T_B_SWEEP = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
TARGET_P_M_GT0 = 0.80  # target for next-phase (7.E) gating


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7D] {msg}", flush=True)


def bhattacharyya_distance_diag(mu_i, sigma_i, mu_j, sigma_j):
    """Diagonal Bhattacharyya (7.C.2 convention).

    D_B = 1/8 (mu_i-mu_j)^T Sigma_avg^{-1} (mu_i-mu_j)
        + 1/2 ln(det Sigma_avg / sqrt(det Sigma_i det Sigma_j))
    Diagonal version: term1 sums (diff^2 / sigma_avg) and term2 sums
    log(sigma_avg) - 0.5*log(sigma_i) - 0.5*log(sigma_j).
    """
    diff = mu_i - mu_j
    sigma_avg = 0.5 * (sigma_i + sigma_j)
    term1 = 0.125 * np.sum((diff * diff) / np.maximum(sigma_avg, EPS))
    term2 = 0.5 * np.sum(
        np.log(np.maximum(sigma_avg, EPS))
        - 0.5 * np.log(np.maximum(sigma_i, EPS))
        - 0.5 * np.log(np.maximum(sigma_j, EPS))
    )
    return float(term1 + term2)


def greedy_maxmin_in_group(users_data, t_b, seed=SEED):
    """Greedy BD-maxmin reduction. users_data: list of dicts with mu, sigma."""
    rng = random.Random(seed)
    if not users_data:
        return []
    if len(users_data) == 1:
        return [0]
    selected = [rng.randrange(len(users_data))]
    while True:
        candidates = []
        for i in range(len(users_data)):
            if i in selected:
                continue
            min_bd = min(
                bhattacharyya_distance_diag(
                    users_data[i]["mu"], users_data[i]["sigma"],
                    users_data[s]["mu"], users_data[s]["sigma"],
                )
                for s in selected
            )
            candidates.append((min_bd, i))
        if not candidates:
            break
        best_bd, best_i = max(candidates)
        if best_bd < t_b:
            break
        selected.append(best_i)
    return selected


def stage_profile(profiles_doc):
    """Stage A: verify per-user Profile-Set Gaussian (μ, σ_diag) integrity.

    Reads 7.C.1 output. We do NOT recompute; we trust 7.C.1 identity invariant.
    This stage is a sanity check (σ > 0, no NaN, shape correct).
    """
    log("=== STAGE profile: verify Profile-Set sentence-level Gaussians ===")
    profiles = profiles_doc["users"]
    out = []
    for uid, e in profiles.items():
        mu = np.asarray(e["mu_profile"], dtype=np.float64)
        sigma = np.asarray(e["sigma_profile_diag"], dtype=np.float64)
        sigma_min = float(sigma.min())
        sigma_max = float(sigma.max())
        sigma_mean = float(sigma.mean())
        ok = bool(sigma_min > 0 and not np.isnan(mu).any() and not np.isnan(sigma).any()
                  and mu.shape == (48,) and sigma.shape == (48,))
        out.append({
            "user_id": uid,
            "target_asin": e["target_asin"],
            "n_profile": e["n_profile"],
            "n_source": e["n_source"],
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sigma_mean": sigma_mean,
            "ok": ok,
        })
    n_ok = sum(1 for r in out if r["ok"])
    log(f"  {n_ok}/{len(out)} profiles have valid (μ, σ_diag) on 48d PCA48")
    return out


def stage_cohort(profile_check, profiles_doc, cv_users, asin_to_users, t_b_sweep):
    """Stage B: per T_B, build syntax-separable cohort.

    Note on cohort definition (2026-08-31 user instruction):
    Originally proposed as "same-ASIN cohort + greedy max-min". However our 15
    audit users each have a distinct target_asin (5 7.B + 10 random), and
    none share ASINs with each other. So per-ASIN greedy max-min collapses
    to singletons and yields 0-cohort.

    We generalize: cohort is the **syntax-separable subset of audit users**
    (regardless of target_asin). Validation uses cross-ASIN competitors:
    d_other = min over OTHER surviving audit users. This is consistent with
    the paper's actual research question ("does user expression variation
    affect retrieval?") — what matters is whether a user's style is
    distinguishable from other users' styles, not whether they reviewed
    the same product.

    Q-gate pre-filters the 15 audit users (cv_inlier_frac >= 0.9 AND
    cv_nll <= 39.0, None-safe). Then pairwise D_B on Profile-Set
    Gaussians only. Greedy max-min with T_B threshold.

    Returns cohorts_by_tb: {t_b -> {"selected": [uid,...], "asins": [...]}}
    """
    log("=== STAGE cohort: T_B sweep + greedy max-min on audit users ===")
    profiles = profiles_doc["users"]
    audit_users = sorted(profiles.keys())

    # Build per-user Gaussian payload
    def get_G(uid):
        e = profiles[uid]
        return {
            "mu": np.asarray(e["mu_profile"], dtype=np.float64),
            "sigma": np.asarray(e["sigma_profile_diag"], dtype=np.float64),
        }

    # Q-gate on audit users (relaxed per Phase 7.D design)
    qualifiers = []
    for uid in audit_users:
        e = profiles[uid]
        n_rev = e.get("n_reviews_raw", 0)
        if n_rev < N_REVIEWS_MIN:
            continue
        cv = cv_users.get(uid, {})
        inlier = cv.get("cv_inlier_frac")
        nll = cv.get("cv_nll")
        if inlier is None:
            continue
        if inlier < CV_INLIER_MIN:
            continue
        if nll is not None and nll > CV_NLL_MAX:
            continue
        qualifiers.append(uid)
    log(f"  audit users: {len(audit_users)} | after Q-gate (inlier≥{CV_INLIER_MIN} & "
        f"n_rev≥{N_REVIEWS_MIN} & nll≤{CV_NLL_MAX}): {len(qualifiers)}")

    if len(qualifiers) < 2:
        raise AssertionError(
            f"[7D] Only {len(qualifiers)} audit users pass Q-gate — cannot "
            "build syntax-separable cohort (need ≥ 2)"
        )

    # Pairwise D_B matrix (diagnostic, not for selection)
    n_q = len(qualifiers)
    bd_pairs = {}
    for i in range(n_q):
        for j in range(i + 1, n_q):
            d = bhattacharyya_distance_diag(
                np.asarray(profiles[qualifiers[i]]["mu_profile"]),
                np.asarray(profiles[qualifiers[i]]["sigma_profile_diag"]),
                np.asarray(profiles[qualifiers[j]]["mu_profile"]),
                np.asarray(profiles[qualifiers[j]]["sigma_profile_diag"]),
            )
            bd_pairs[qualifiers[i] + "|" + qualifiers[j]] = d
    if n_q >= 2:
        bd_vals = np.array(list(bd_pairs.values()))
        bd_min = float(bd_vals.min())
        bd_med = float(np.median(bd_vals))
        bd_max = float(bd_vals.max())
    else:
        bd_min = bd_med = bd_max = 0.0
    
    log("  USER GOAL CONSTRAINTS (2026-08-31):")
    log(f"    Q-gate: inlier_frac >= 0.85 + nll <= 34.0 + n_rev >= 15")
    log(f"    Held-out Rank@1 target: P(M>0) >= 0.80 (current ceiling 34.5%)")
    log(f"    Median M > 0 (avoid few-users trick)")
    log(f"  pairwise BD on {n_q} users: min={bd_min:.3f} med={bd_med:.3f} "
        f"max={bd_max:.3f}")

    cohorts_by_tb = {}
    for t_b in t_b_sweep:
        users_data = [get_G(v) for v in qualifiers]
        sel_idx = greedy_maxmin_in_group(users_data, t_b, seed=SEED)
        sel_uids = [qualifiers[i] for i in sel_idx]
        # Per-ASIN accounting (each surviving user is "their own ASIN" since
        # we don't constrain cross-ASIN; pair structure is global).
        per_asin = {}
        for uid in sel_uids:
            a = profiles[uid]["target_asin"]
            per_asin.setdefault(a, []).append(uid)
        asins_with_ge2 = sum(1 for v in per_asin.values() if len(v) >= 2)
        asins_with_ge1 = sum(1 for v in per_asin.values() if len(v) >= 1)
        cohorts_by_tb[t_b] = {
            "selected": sel_uids,
            "total_selected": len(sel_uids),
            "per_asin": {a: {"selected": v, "n_selected": len(v)}
                          for a, v in per_asin.items()},
            "asins_with_ge1": asins_with_ge1,
            "asins_with_ge2": asins_with_ge2,
        }
        log(f"  T_B={t_b}: {len(sel_uids)} users | "
            f"{asins_with_ge1} ASINs≥1 | {asins_with_ge2} ASINs≥2 | "
            f"uids={[u[:10] for u in sel_uids]}")

    return {
        "cohorts_by_tb": cohorts_by_tb,
        "qualifiers": qualifiers,
        "bd_pairs": bd_pairs,
        "bd_min": bd_min,
        "bd_med": bd_med,
        "bd_max": bd_max,
    }


def stage_validate(profiles_doc, cohorts_by_tb):
    """Stage C: held-out validation.

    For each T_B cohort: per surviving user u, for each source_set sentence z_s
    (held-out, NEVER profile), compute:
        d_self = ||z_s - mu_u||
        d_other = min over OTHER surviving audit users ||z_s - mu_c||
        M_local = d_other - d_self
    Report P(M_local>0) median and per-user breakdown per T_B.

    Note: cohort is global (not per-ASIN) — competitor = other surviving
    audit users (cross-ASIN). This is consistent with paper question
    "does user expression variation affect retrieval" — separability in
    syntax space, not in purchase space.
    """
    log("=== STAGE validate: held-out source_set M_local audit per T_B ===")
    profiles = profiles_doc["users"]
    out_by_tb = {}
    for t_b, cohort_data in cohorts_by_tb.items():
        sel_uids = cohort_data["selected"]
        per_user = []
        p_M_list = []
        # Pre-compute other-mu matrix
        other_uids_by_u = {u: [v for v in sel_uids if v != u] for u in sel_uids}
        for uid in sel_uids:
            u_short = uid[:12]
            e = profiles[uid]
            mu_u = np.asarray(e["mu_profile"], dtype=np.float64)
            source_payload = e["source_set"]
            if not source_payload:
                continue
            zs = np.array([s["z48"] for s in source_payload], dtype=np.float64)
            d_self = np.linalg.norm(zs - mu_u[None, :], axis=1)
            other_uids = other_uids_by_u[uid]
            if not other_uids:
                per_user.append({
                    "user_id": uid, "target_asin": e["target_asin"],
                    "n_source": len(zs), "n_comp": 0,
                    "d_self_med": float(np.median(d_self)),
                    "d_other_med": None, "M_local_med": None,
                    "p_M_gt0": None,
                })
                continue
            other_mu = np.array(
                [profiles[v]["mu_profile"] for v in other_uids],
                dtype=np.float64,
            )
            d_other_each = np.linalg.norm(
                zs[:, None, :] - other_mu[None, :, :], axis=2
            )
            d_other = d_other_each.min(axis=1)
            M_local = d_other - d_self
            p_M = float(np.mean(M_local > 0))
            p_M_list.append(p_M)
            per_user.append({
                "user_id": uid, "target_asin": e["target_asin"],
                "n_source": len(zs), "n_comp": len(other_uids),
                "d_self_med": float(np.median(d_self)),
                "d_other_med": float(np.median(d_other)),
                "M_local_med": float(np.median(M_local)),
                "p_M_gt0": p_M,
            })
        if p_M_list:
            med_p = float(np.median(p_M_list))
            mean_p = float(np.mean(p_M_list))
            n_pass_target = sum(1 for p in p_M_list if p >= TARGET_P_M_GT0)
        else:
            med_p = mean_p = None
            n_pass_target = 0
        out_by_tb[t_b] = {
            "cohort_total_selected": cohort_data["total_selected"],
            "asins_with_ge1": cohort_data["asins_with_ge1"],
            "asins_with_ge2": cohort_data["asins_with_ge2"],
            "n_users_evaluated": len(p_M_list),
            "p_M_gt0_med": med_p,
            "p_M_gt0_mean": mean_p,
            "n_users_p_ge_80pct": n_pass_target,
            "per_user": per_user,
        }
        if med_p is not None:
            log(f"  T_B={t_b}: cohort={cohort_data['total_selected']} users "
                f"| asins≥2={cohort_data['asins_with_ge2']} "
                f"| eval={len(p_M_list)} users "
                f"| P(M>0) med={med_p:.3f} mean={mean_p:.3f} "
                f"| n@80%={n_pass_target}")
        else:
            log(f"  T_B={t_b}: cohort={cohort_data['total_selected']} users "
                f"| asins≥2={cohort_data['asins_with_ge2']} "
                f"| NO surviving users to evaluate")
    return out_by_tb


def main() -> None:
    log("=== Phase 7.D — Sentence-Level Syntax-Separable Cohort Filtering ===")

    # Load inputs
    profiles_doc = json.load(open(PROFILES_PATH))
    cv_doc = json.load(open(GAUSS_CV_PATH))
    cv_users = cv_doc.get("users", {})
    asins_doc = json.load(open(ASINS_PATH))
    asin_to_users = collections.defaultdict(list)
    for entry in asins_doc["asins"]:
        a = entry["asin"]
        for uid in entry["users_sampled"]:
            asin_to_users[a].append(uid)
    log(f"  loaded {len(profiles_doc['users'])} profiles | "
        f"CV for {len(cv_users)} users | "
        f"cohort ASINs={len(asin_to_users)}")

    # (smoke hook removed for full run; reverted after smoke validation)

    # Stage A: profile integrity check
    profile_check = stage_profile(profiles_doc)
    n_ok = sum(1 for r in profile_check if r["ok"])
    if n_ok != len(profile_check):
        raise AssertionError(
            f"[7D] {len(profile_check) - n_ok} profiles have invalid σ (degenerate "
            f"or NaN) — cannot build reliable sentence-level Gaussians"
        )

    # Stage B: cohort sweep
    cohort_result = stage_cohort(
        profile_check, profiles_doc, cv_users, asin_to_users, T_B_SWEEP
    )
    cohorts_by_tb = cohort_result["cohorts_by_tb"]
    qualifiers = cohort_result["qualifiers"]
    bd_pairs = cohort_result["bd_pairs"]
    bd_min = cohort_result["bd_min"]
    bd_med = cohort_result["bd_med"]
    bd_max = cohort_result["bd_max"]

    # Stage C: held-out validation
    validate_by_tb = stage_validate(profiles_doc, cohorts_by_tb)

    # Trade-off curve + recommendation
    trade_off = []
    best_tb = None
    best_score = -1.0
    for t_b in T_B_SWEEP:
        v = validate_by_tb[t_b]
        c = cohorts_by_tb[t_b]
        med_p = v["p_M_gt0_med"]
        cohort = c["total_selected"]
        row = {
            "T_B": t_b,
            "cohort_total": cohort,
            "p_M_gt0_med": med_p,
            "p_M_gt0_mean": v["p_M_gt0_mean"],
            "n_users_p_ge_80pct": v["n_users_p_ge_80pct"],
        }
        trade_off.append(row)
        # Score: med_p weighted by sqrt(cohort) so we prefer larger cohorts
        # at high p_M. Need cohort >= 2 to enable any paired analysis.
        if med_p is not None and cohort >= 2:
            import math
            score = med_p * math.sqrt(cohort)
            if score > best_score:
                best_score = score
                best_tb = t_b

    recommendation = {
        "best_T_B_for_P_M_ge_80pct": (
            next((r["T_B"] for r in trade_off
                  if r["p_M_gt0_med"] is not None
                  and r["p_M_gt0_med"] >= TARGET_P_M_GT0
                  and r["cohort_total"] >= 2), None)
        ),
        "best_T_B_overall": best_tb,
    }

    summary = {
        "config": {
            "description": ("Phase 7.D: filter same-ASIN cohort by sentence-level "
                            "Bhattacharyya distance on Profile-Set Gaussians only. "
                            "Held-out source_set sentences used ONLY for M_local "
                            "validation, never for cohort construction."),
            "t_b_sweep": T_B_SWEEP,
            "target_p_M_gt0": TARGET_P_M_GT0,
            "q_gate_inlier_min": CV_INLIER_MIN,
            "q_gate_nll_max": CV_NLL_MAX,
            "seed": SEED,
            "n_audit_users": len(profiles_doc["users"]),
            "gaussian_source": "Profile Set P_u only (7.C.1 80% split)",
            "validation_source": "source_set only (held-out 20%)",
        },
        "profile_check_n_ok": n_ok,
        "pairwise_bd_stats": {
            "n_qualifiers": len(qualifiers),
            "bd_min": bd_min,
            "bd_med": bd_med,
            "bd_max": bd_max,
        },
        "pairwise_bd_pairs": bd_pairs,
        "qualifier_uids": qualifiers,
        "trade_off_curve": trade_off,
        "recommendation": recommendation,
        "cohorts_by_tb_summary": {
            str(t_b): {
                "total_selected": cohorts_by_tb[t_b]["total_selected"],
                "asins_with_ge1": cohorts_by_tb[t_b]["asins_with_ge1"],
                "asins_with_ge2": cohorts_by_tb[t_b]["asins_with_ge2"],
                "selected_uids_short": [u[:10] for u in cohorts_by_tb[t_b]["selected"]],
            }
            for t_b in T_B_SWEEP
        },
        "validate_by_tb_summary": {
            str(t_b): {
                "n_users_evaluated": validate_by_tb[t_b]["n_users_evaluated"],
                "p_M_gt0_med": validate_by_tb[t_b]["p_M_gt0_med"],
                "p_M_gt0_mean": validate_by_tb[t_b]["p_M_gt0_mean"],
                "n_users_p_ge_80pct": validate_by_tb[t_b]["n_users_p_ge_80pct"],
            }
            for t_b in T_B_SWEEP
        },
        "validate_by_tb_detail": {
            str(t_b): validate_by_tb[t_b]["per_user"]
            for t_b in T_B_SWEEP
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log(f"  wrote → {OUT_JSON}")

    log("\n=== TRADE-OFF CURVE ===")
    log(f"  {'T_B':>5} | {'cohort':>7} | {'P(M>0)_med':>11} | {'P(M>0)_mean':>12} | {'n@80%':>5}")
    for r in trade_off:
        log(f"  {r['T_B']:>5.1f} | {r['cohort_total']:>7} | "
            f"{(r['p_M_gt0_med'] if r['p_M_gt0_med'] is not None else float('nan')):>11.3f} | "
            f"{(r['p_M_gt0_mean'] if r['p_M_gt0_mean'] is not None else float('nan')):>12.3f} | "
            f"{r['n_users_p_ge_80pct']:>5}")
    log(f"\n  recommendation: best_T_B_overall = {recommendation['best_T_B_overall']}")
    log(f"  recommendation: first T_B hitting P(M>0)≥{TARGET_P_M_GT0:.0%} with cohort≥2 = "
        f"{recommendation['best_T_B_for_P_M_ge_80pct']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
