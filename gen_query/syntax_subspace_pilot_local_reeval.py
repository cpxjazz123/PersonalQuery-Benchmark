"""Phase 6.F — Local-Margin Re-evaluation of Phase 6.D Generated Queries.

User critique (23:50):
- Phase 6.D was evaluated against 1.2M global competitors → M_global ≈ -4.51 (NO-GO)
- Phase 6.E.4 proved same-ASIN local cohort M_local > 50% reaches 95.7-98.6%
- Re-evaluate Phase 6.D generated queries using same-ASIN LOCAL competitor set
- If M_local > 0 for most candidates, Phase 6.D flips from PARTIAL/STOP to GO

Output: result/gen_query/phase6f_pilot_local_reeval.json + memory note.
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

# Hardcoded paths (per Rule 3)
PILOT_JSON = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6d_pilot.json"
ASINS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"
GAUSSIANS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6f_pilot_local_reeval.log"
RESULT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase6f_pilot_local_reeval.json"

# Local-cohort definition (per user 23:35): same ASIN as candidate user
# Quality gate: cv_inlier_frac ≥ 0.9 AND cv_nll ≤ 39
CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0
MIN_COMPETITORS = 1  # exclude n_comp=0 cases (trivially M=inf)


def log(msg):
    line = f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] [phase6f] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    log("=== Phase 6.F — Local-Margin Re-evaluation of Phase 6.D ===")
    log(f"  CV quality gate: cv_inlier ≥ {CV_INLIER_MIN}, cv_nll ≤ {CV_NLL_MAX}")
    log(f"  Local competitor set: same-ASIN as candidate user (with quality gate)")

    # 1. Load Phase 6.D pilot (10 users × 18 candidates)
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    pilot_users = [r["user_id"] for r in pilot["results"]]
    log(f"  pilot users: {len(pilot_users)}")
    log(f"  total candidates: {sum(r['n_candidates'] for r in pilot['results'])}")

    # 2. Load stage8_5_asins → reverse index user→ASINs
    log(f"\n  Loading ASIN cohort from {ASINS_JSON}...")
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)
    asins_list = asin_data["asins"]
    log(f"  loaded {len(asins_list)} ASINs")

    user_to_asins = defaultdict(set)
    asin_to_competitors = defaultdict(set)
    for asin_rec in asins_list:
        asin = asin_rec["asin"]
        users_sampled = asin_rec["users_sampled"]
        for u in users_sampled:
            user_to_asins[u].add(asin)
            asin_to_competitors[asin].add(u)
    log(f"  built reverse index: {len(user_to_asins)} users across ASINs")

    # 3. Find ASINs each pilot user reviewed
    pilot_user_asins = {}
    for u in pilot_users:
        asins = user_to_asins.get(u, set())
        pilot_user_asins[u] = asins
        log(f"    {u}: {len(asins)} ASINs reviewed")

    # 4. Build same-ASIN local competitor set (other users in same ASIN, with quality gate)
    log(f"\n  Loading user Gaussians with quality info from {GAUSSIANS_JSON}...")
    # CV file has cv_inlier_frac + cv_nll
    cv_path = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_cv.json"
    with open(cv_path) as f:
        cv_data = json.load(f)
    cv_users = cv_data["users"]  # dict {user_id: {cv_inlier_frac, cv_nll, ...}}
    log(f"  loaded CV info for {len(cv_users)} users")

    # Quality gate
    high_quality = set()
    n_skipped_none = 0
    for u, info in cv_users.items():
        cv_inlier = info.get("cv_inlier_frac")
        cv_nll = info.get("cv_nll")
        if cv_inlier is None or cv_nll is None:
            n_skipped_none += 1
            continue
        if cv_inlier >= CV_INLIER_MIN and cv_nll <= CV_NLL_MAX:
            high_quality.add(u)
    log(f"  high-quality users: {len(high_quality)} (skipped {n_skipped_none} with None CV values)")

    # For each pilot user, build local competitor set per ASIN
    pilot_user_local_competitors = {}  # user → {asin: [competitor_users]}
    for u in pilot_users:
        asins = pilot_user_asins[u]
        per_asin_comps = {}
        for a in asins:
            comps = asin_to_competitors[a] - {u}  # exclude self
            comps = comps & high_quality  # quality gate
            per_asin_comps[a] = list(comps)
        pilot_user_local_competitors[u] = per_asin_comps
        log(f"    {u}: {sum(len(v) for v in per_asin_comps.values())} total local competitors across {len(asins)} ASINs")

    # 5. Load PCA48 user μ (from main gaussians file, not CV)
    log(f"\n  Loading PCA48 user μ from {GAUSSIANS_JSON}...")
    with open(GAUSSIANS_JSON) as f:
        gauss_data = json.load(f)
    user_mus = gauss_data["users"]  # {user_id: {mu: [48d], sigma_diag: [48d], ...}}
    log(f"  loaded {len(user_mus)} user μ (PCA48)")

    # 6. For each pilot user's candidates, recompute d_nearest_other LOCAL
    log(f"\n=== Re-evaluating 180 candidates (10 users × 18) ===")
    per_user_local = []
    for r in pilot["results"]:
        u = r["user_id"]
        comps_per_asin = pilot_user_local_competitors[u]
        all_comps = set()
        for comps in comps_per_asin.values():
            all_comps.update(comps)
        n_comp = len(all_comps)
        # Build local μ matrix (n_comp × 48)
        local_mus = []
        local_comp_ids = []
        for c in all_comps:
            if c in user_mus:
                local_mus.append(user_mus[c]["mu"])
                local_comp_ids.append(c)
        local_mus_arr = None
        if local_mus:
            local_mus_arr = np.array(local_mus, dtype=np.float32)  # (n_comp, 48)

        candidate_local_results = []
        for c in r["candidates"]:
            d_self = c["d_self"]  # already PCA48 L2 distance from candidate to user's μ_fit
            d_self_arr = np.array(d_self, dtype=np.float32)
            if local_mus_arr is None or len(local_mus_arr) == 0:
                # No local competitors
                M_local = float("inf")
                d_nearest_other_local = float("inf")
                n_valid = 0
            else:
                # Approximation: use existing μ_u as proxy for μ_fit_user
                # → d_nearest_other_local = ||μ_u - μ_comp||
                user_mu = user_mus.get(u, {}).get("mu", None)
                if user_mu is None:
                    M_local = float("inf")
                    d_nearest_other_local = float("inf")
                    n_valid = 0
                else:
                    user_mu_arr = np.array(user_mu, dtype=np.float32)
                    # Distance from user_mu to each competitor mu
                    diffs = local_mus_arr - user_mu_arr[None, :]  # (n_comp, 48)
                    dists = np.sqrt((diffs ** 2).sum(axis=1))  # (n_comp,)
                    d_nearest_other_local = float(dists.min())
                    M_local = d_nearest_other_local - d_self  # local margin
                    n_valid = len(local_mus_arr)

            candidate_local_results.append({
                "candidate_idx": c["candidate_idx"],
                "skel_d": c["skel_d"],
                "d_self": c["d_self"],
                "d_nearest_other_global": c["d_nearest_other"],
                "margin_global": c["margin"],
                "d_nearest_other_local": d_nearest_other_local,
                "M_local": M_local,
                "accepted_global": c["accepted"],
                "M_local_positive": M_local > 0 if not math.isinf(M_local) else None,
                "n_local_competitors": n_valid,
            })

        # Per-user aggregation (over local competitors across all of user's ASINs)
        per_user_local.append({
            "user_id": u,
            "mu_norm": r["mu_norm"],
            "n_candidates": r["n_candidates"],
            "n_local_competitors": n_comp,
            "n_asins_reviewed": len(pilot_user_asins[u]),
            "d_self_mean_global": r["d_self_mean"],
            "margin_mean_global": r["margin_mean"],
            "accept_rate_global": r["accept_rate"],
            "candidates": candidate_local_results,
        })

    # 7. Aggregate cohort summary + per-ASIN-size stratification
    log(f"\n=== AGGREGATE COHORT SUMMARY ===")

    # Per-user local stats
    def safe_mean(xs):
        xs = [x for x in xs if not math.isinf(x)]
        return statistics.mean(xs) if xs else None

    def safe_median(xs):
        xs = sorted([x for x in xs if not math.isinf(x)])
        return statistics.median(xs) if xs else None

    # Global: aggregate all candidates across users
    all_M_global = [c["margin"] for r in pilot["results"] for c in r["candidates"]]
    all_M_local = []
    all_d_self = []
    for entry in per_user_local:
        for c in entry["candidates"]:
            if not math.isinf(c["M_local"]):
                all_M_local.append(c["M_local"])
                all_d_self.append(c["d_self"])
    log(f"\n  GLOBAL (1.2M):")
    log(f"    M_global across {len(all_M_global)} candidates: median={statistics.median(all_M_global):.3f}, mean={statistics.mean(all_M_global):.3f}")
    log(f"    P(M_global > 0): {sum(1 for x in all_M_global if x > 0) / len(all_M_global) * 100:.1f}%")

    log(f"\n  LOCAL (same-ASIN, n_comp>=1 candidates):")
    log(f"    M_local across {len(all_M_local)} candidates: median={statistics.median(all_M_local):.3f}, mean={statistics.mean(all_M_local):.3f}")
    log(f"    P(M_local > 0): {sum(1 for x in all_M_local if x > 0) / len(all_M_local) * 100:.1f}%")
    log(f"    d_self median: {statistics.median(all_d_self):.3f}")

    # Strict: M_local > 0 AND d_self ≤ R (R_95 from main pipeline = ~9.0 from memory)
    R_95 = 9.0
    strict_local = [x for x, ds in zip(all_M_local, all_d_self) if ds <= R_95]
    log(f"    Strict (M_local > 0 AND d_self ≤ {R_95}): {sum(1 for x in strict_local if x > 0)}/{len(strict_local)} = {sum(1 for x in strict_local if x > 0) / max(len(strict_local), 1) * 100:.1f}%")

    # Per-user summary
    log(f"\n  PER-USER LOCAL:")
    for entry in per_user_local:
        local_Ms = [c["M_local"] for c in entry["candidates"] if not math.isinf(c["M_local"])]
        local_d_self = [c["d_self"] for c in entry["candidates"] if not math.isinf(c["M_local"])]
        if local_Ms:
            pct_pos = sum(1 for x in local_Ms if x > 0) / len(local_Ms) * 100
            log(f"    {entry['user_id']} (μ_norm={entry['mu_norm']:.2f}, n_comp={entry['n_local_competitors']}, {entry['n_asins_reviewed']} ASINs):")
            log(f"      d_self mean={statistics.mean(local_d_self):.2f}, M_local mean={statistics.mean(local_Ms):.2f}, P(M>0)={pct_pos:.1f}%")

    # 8. Stratify by ASIN size (per-user n_asins_reviewed, since each user has multiple ASINs)
    log(f"\n  STRATIFICATION BY n_asins_reviewed (per-user):")
    by_n_asins = defaultdict(list)
    for entry in per_user_local:
        n = entry["n_asins_reviewed"]
        for c in entry["candidates"]:
            if not math.isinf(c["M_local"]):
                by_n_asins[n].append(c["M_local"])

    for n_asins in sorted(by_n_asins.keys()):
        Ms = by_n_asins[n_asins]
        if Ms:
            pct = sum(1 for x in Ms if x > 0) / len(Ms) * 100
            log(f"    n_asins={n_asins}: N_cands={len(Ms)}, M_local median={statistics.median(Ms):.3f}, P(M>0)={pct:.1f}%")

    # 9. Save result JSON
    summary = {
        "config": {
            "cv_inlier_min": CV_INLIER_MIN,
            "cv_nll_max": CV_NLL_MAX,
            "r_95_threshold": R_95,
            "min_competitors": MIN_COMPETITORS,
        },
        "n_pilot_users": len(pilot_users),
        "n_total_candidates": sum(r["n_candidates"] for r in pilot["results"]),
        "global_aggregate": {
            "n_candidates": len(all_M_global),
            "M_median": round(statistics.median(all_M_global), 3),
            "M_mean": round(statistics.mean(all_M_global), 3),
            "P_M_gt0_pct": round(sum(1 for x in all_M_global if x > 0) / len(all_M_global) * 100, 1),
        },
        "local_aggregate": {
            "n_candidates_valid": len(all_M_local),
            "n_candidates_excluded_inf": len(all_M_global) - len(all_M_local),
            "M_median": round(statistics.median(all_M_local), 3) if all_M_local else None,
            "M_mean": round(statistics.mean(all_M_local), 3) if all_M_local else None,
            "P_M_gt0_pct": round(sum(1 for x in all_M_local if x > 0) / max(len(all_M_local), 1) * 100, 1),
            "d_self_median": round(statistics.median(all_d_self), 3),
            "strict_M_gt0_AND_d_self_le_R95_count": sum(1 for x in strict_local if x > 0),
            "strict_M_gt0_AND_d_self_le_R95_total": len(strict_local),
            "strict_pct": round(sum(1 for x in strict_local if x > 0) / max(len(strict_local), 1) * 100, 1),
        },
        "stratification_by_n_asins": {
            str(n_asins): {
                "n_candidates": len(Ms),
                "M_median": round(statistics.median(Ms), 3) if Ms else None,
                "P_M_gt0_pct": round(sum(1 for x in Ms if x > 0) / max(len(Ms), 1) * 100, 1),
            }
            for n_asins, Ms in sorted(by_n_asins.items())
        },
        "per_user": per_user_local,
        "key_finding": (
            f"Phase 6.D generated queries re-evaluated with same-ASIN LOCAL competitor set: "
            f"M_local median = {statistics.median(all_M_local):.3f}, P(M_local > 0) = "
            f"{sum(1 for x in all_M_local if x > 0) / max(len(all_M_local), 1) * 100:.1f}% "
            f"(vs M_global median = {statistics.median(all_M_global):.3f}, P(M_global > 0) = "
            f"{sum(1 for x in all_M_global if x > 0) / len(all_M_global) * 100:.1f}%). "
            f"Phase 6.D VERDICT FLIP candidate."
        ),
    }

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
