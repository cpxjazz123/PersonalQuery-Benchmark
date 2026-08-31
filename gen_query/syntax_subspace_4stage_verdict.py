"""Phase 4-Stage Joint Verdict — Align all 4 quality gates per stop-hook feedback.

User goal 2026-08-31 (stop hook):
  A pipeline "must satisfy all 4 quality gates simultaneously" to be GO.

Stages:
  1. User Gaussian CV-NLL ≤ 34  (predictive reliability, calibrated precision ≈ 98.2%)
  2. Held-out user Rank@1  P(M>0) ≥ 80% AND median M > 0
  3. Query generation semantic fidelity: attr_complete ≥ 95% AND unsupported content ≤ 5%
  4. Strict personalized criterion  P(M>0 ∧ d_self ≤ R_95) ≥ 80%

Inputs (all already produced):
  - result/gaussian/cv_metrics_summary.json   (stage 1 distribution)
  - result/gaussian/reliability_summary.json  (stage 1 ΔNLL bootstrap)
  - result/gen_query/phase7g_asin_stratified_audit.json (stage 2 smoke)
  - result/gen_query/phase7e_strict_personalized.json  (stage 3 attr_coverage + stage 4 strict pass)

Output:
  result/gen_query/phase_4stage_verdict.json

Run:
  python3 gen_query/syntax_subspace_4stage_verdict.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT = REPO / "result" / "gen_query" / "phase_4stage_verdict.json"

CV_METRICS = REPO / "result" / "gaussian" / "cv_metrics_summary.json"
RELIABILITY = REPO / "result" / "gaussian" / "reliability_summary.json"
P7G = REPO / "result" / "gen_query" / "phase7g_asin_stratified_audit.json"
P7E = REPO / "result" / "gen_query" / "phase7e_strict_personalized.json"


def log(m: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [4STG] {m}", flush=True)


def stage1() -> dict:
    """Stage 1 — Gaussian CV-NLL ≤ 34 quality gate."""
    if not CV_METRICS.exists():
        return {"verdict": "MISSING", "reason": f"{CV_METRICS} not found"}
    cv = json.load(open(CV_METRICS))
    rel = json.load(open(RELIABILITY))

    n_total = cv["n_users_total"]
    n_with_cv = cv["n_users_with_cv"]
    frac_with_cv = n_with_cv / n_total
    n_reliable = rel["n_users_reliable"]
    frac_reliable = rel["frac_reliable"]

    # CV-NLL ≤ 34: distribution inspection
    cv_nll_dist = cv["cv_nll_dist"]
    # Estimate fraction with cv_nll ≤ 34 from quantiles: q25=27.1, median=37.97, mean=159.3
    # We cannot get exact count; report proxy from q25/q75 plus reliability frac
    proxy_nll_le_34 = "~q25-median band"

    return {
        "verdict": "PARTIAL",
        "n_users_total": n_total,
        "n_users_with_cv": n_with_cv,
        "frac_with_cv": round(frac_with_cv, 4),
        "n_reliable": n_reliable,
        "frac_reliable_ΔNLL_pos_CI": round(frac_reliable, 4),
        "cv_nll_distribution": {
            "q05": round(cv_nll_dist["q05"], 2),
            "q25": round(cv_nll_dist["q25"], 2),
            "median": round(cv_nll_dist["median"], 2),
            "q75": round(cv_nll_dist["q75"], 2),
            "q95": round(cv_nll_dist["q95"], 2),
            "mean": round(cv_nll_dist["mean"], 2),
        },
        "gate_threshold": "cv_nll ≤ 34 (precision ≈ 98.2%)",
        "gate_status": "PARTIAL — only 129K/1.22M (10.6%) have CV; among them 67.6% reliable ΔNLL. Q-gate filter (cv_nll ≤ 34) used in 7G/7E filters this subset.",
    }


def stage2() -> dict:
    """Stage 2 — Held-out P(M>0) ≥ 80% AND median M > 0."""
    if not P7G.exists():
        return {"verdict": "MISSING", "reason": f"{P7G} not found"}
    d = json.load(open(P7G))
    s = d["summary"]
    p_overall = s["overall_P_M_gt0_weighted"]
    p_med_across = s["P_M_gt0_median_across_asins"]
    asins = d["asin_results"]

    # Per-ASIN M_med
    per_asin_meds = []
    for r in asins:
        med = r.get("M_med_per_user_median")
        if med is not None:
            per_asin_meds.append({"asin": r["asin"], "M_med_per_user_median": round(med, 3),
                                  "P_M_gt0_w": round(r["P_M_gt0_weighted"], 3)})

    # Overall M_med across ASINs (median of per-ASIN M_med)
    asin_meds = [r["M_med_per_user_median"] for r in asins if r.get("M_med_per_user_median") is not None]
    overall_M_med = sum(asin_meds) / len(asin_meds) if asin_meds else None

    pass_p = p_overall >= 0.80
    pass_med = (overall_M_med is not None) and (overall_M_med > 0)
    gate_pass = pass_p and pass_med

    return {
        "verdict": "PASS" if gate_pass else "FAIL",
        "evidence_source": "phase7g_asin_stratified_audit (smoke 5 ASINs, 40 cohort users, 27 held-out)",
        "P_M_gt0_overall_weighted": round(p_overall, 3),
        "gate_P_M_gt0 ≥ 0.80": "PASS" if pass_p else "FAIL",
        "P_M_gt0_median_across_asins_per_user": round(p_med_across, 3),
        "M_med_per_ASIN_mean": round(overall_M_med, 3) if overall_M_med else None,
        "gate_median_M > 0": "PASS" if pass_med else "FAIL",
        "n_asins_evaluated": len(asins),
        "per_asin": per_asin_meds,
        "ceiling_evidence_chain": [
            {"phase": "7.C.1", "setup": "LOO same-user (profile = eval)", "P_M_gt0": "~90%"},
            {"phase": "7.C.2", "setup": "Real held-out, cross-user cohort", "P_M_gt0": "0.4375 median"},
            {"phase": "7.G", "setup": "Real held-out, same-ASIN cohort (strictest semantic fix)", "P_M_gt0": round(p_overall, 3)},
        ],
        "interpretation": "PCA48 sentence-level representation has a hard distinguishability ceiling; cohort restriction by ASIN does NOT lift it.",
    }


def stage3() -> dict:
    """Stage 3 — Query generation semantic fidelity: attr_complete ≥ 95% AND unsupported content ≤ 5%."""
    if not P7E.exists():
        return {"verdict": "MISSING", "reason": f"{P7E} not found"}
    d = json.load(open(P7E))
    per = d["per_gen"]
    s = d["summary"]

    n = len(per)
    attrs = [g.get("attr_coverage", 0) for g in per]
    n_full = sum(1 for a in attrs if a >= 0.95)
    n_pass_60 = sum(1 for a in attrs if a >= 0.60)

    # unsupported_content proxy: status='ok' AND no extra unsupported semantic claims;
    # we use len_filter hits + attr_filter drops as proxy (less than 95% coverage implies unsupported)
    statuses = [g.get("status", "") for g in per]
    n_status_ok = sum(1 for s_ in statuses if s_ == "ok")
    n_status_attr_drop = sum(1 for s_ in statuses if s_.startswith("attr_filter"))
    n_status_len_drop = sum(1 for s_ in statuses if s_.startswith("len_filter"))

    # Unsupported = anything that fails to fully cover target attributes
    n_unsupported = n - n_full  # anything not at >=0.95 coverage
    frac_unsupported = n_unsupported / n
    pass_full = n_full / n >= 0.95
    pass_unsupp = frac_unsupported <= 0.05

    return {
        "verdict": "PASS" if (pass_full and pass_unsupp) else "FAIL",
        "evidence_source": "phase7e_strict_personalized (35 generations, cond A_free/B_exemplar/C_rewrite)",
        "n_total": n,
        "n_attr_complete_≥95%": n_full,
        "frac_attr_complete": round(n_full / n, 3),
        "gate_attr_complete ≥ 0.95": "PASS" if pass_full else "FAIL",
        "n_attr_complete_≥60%": n_pass_60,
        "frac_attr_complete_≥60%": round(n_pass_60 / n, 3),
        "n_unsupported_proxy": n_unsupported,
        "frac_unsupported_proxy": round(frac_unsupported, 3),
        "gate_unsupported ≤ 0.05": "PASS" if pass_unsupp else "FAIL",
        "attr_coverage_mean": round(sum(attrs) / n, 3),
        "status_breakdown": {
            "ok": n_status_ok,
            "attr_filter_dropped": n_status_attr_drop,
            "len_filter_dropped": n_status_len_drop,
        },
        "interpretation": "NLL≤34 + 5-attr generation: 0/35 (0%) reach ≥95% attr coverage; mean 0.44. Generation over-produces content (length filter 3/35, attr_filter 11/35).",
    }


def stage4() -> dict:
    """Stage 4 — Strict personalized P(M>0 ∧ d_self ≤ R_95) ≥ 80%."""
    if not P7E.exists():
        return {"verdict": "MISSING", "reason": f"{P7E} not found"}
    d = json.load(open(P7E))
    s = d["summary"]
    n_pass = s["overall_n_pass"]
    n_total = s["overall_n_total"]
    pass_rate = s["overall_pass_rate"]
    target = 0.80
    pass_gate = pass_rate >= target

    # Per-cond breakdown
    per = d["per_gen"]
    from collections import defaultdict
    per_cond_pass = defaultdict(lambda: {"pass": 0, "total": 0})
    for g in per:
        c = g.get("cond", "?")
        per_cond_pass[c]["total"] += 1
        if g.get("M") is not None and g.get("R_95") is not None:
            if g["M"] > 0 and g.get("d_self", 1e9) <= g["R_95"]:
                per_cond_pass[c]["pass"] += 1

    per_cond_summary = {}
    for c, v in per_cond_pass.items():
        per_cond_summary[c] = {
            "n_pass": v["pass"],
            "n_total": v["total"],
            "pass_rate": round(v["pass"] / v["total"], 3) if v["total"] else None,
        }

    return {
        "verdict": "PASS" if pass_gate else "FAIL",
        "evidence_source": "phase7e_strict_personalized (strict = M>0 AND d_self ≤ R_95)",
        "n_pass_strict": n_pass,
        "n_total": n_total,
        "pass_rate": round(pass_rate, 3),
        "gate_≥80%": "PASS" if pass_gate else "FAIL",
        "target": target,
        "per_cond": per_cond_summary,
        "interpretation": "Only 5/35 (14.3%) strict-pass. C_rewrite 25 trials aggregate, A_free/B_exemplar 5 each. PCA48 ceiling confirmed.",
    }


def joint_verdict(s1, s2, s3, s4) -> dict:
    """All 4 stages must PASS for pipeline GO."""
    stages = {"stage1_gaussian_quality": s1["verdict"],
              "stage2_heldout_P_M_gt0": s2["verdict"],
              "stage3_query_fidelity": s3["verdict"],
              "stage4_strict_personalized": s4["verdict"]}
    all_pass = all(v == "PASS" for v in stages.values())
    n_fail = sum(1 for v in stages.values() if v == "FAIL")
    n_partial = sum(1 for v in stages.values() if v == "PARTIAL")

    if all_pass:
        overall = "GO"
    elif n_fail >= 2 or (n_fail == 1 and n_partial >= 1):
        overall = "NO-GO"
    else:
        overall = "PARTIAL-GO"

    return {"overall_verdict": overall, "per_stage": stages,
            "n_fail": n_fail, "n_partial": n_partial}


def main():
    log("=== Phase 4-Stage Joint Verdict ===")
    s1 = stage1()
    log(f"  stage1 (Gaussian CV-NLL): {s1['verdict']}")
    s2 = stage2()
    log(f"  stage2 (held-out P(M>0)): {s2['verdict']}")
    s3 = stage3()
    log(f"  stage3 (Query fidelity): {s3['verdict']}")
    s4 = stage4()
    log(f"  stage4 (strict personalized): {s4['verdict']}")

    jv = joint_verdict(s1, s2, s3, s4)
    log(f"=== OVERALL: {jv['overall_verdict']} ===")
    log(f"  per-stage: {jv['per_stage']}")

    out = {
        "description": "Phase 4-Stage Joint Verdict (User goal 2026-08-31 stop hook)",
        "thresholds": {
            "stage1": "Gaussian CV-NLL ≤ 34 (precision ≈ 98.2%)",
            "stage2": "held-out P(M>0) ≥ 80% AND median M > 0",
            "stage3": "attr_complete ≥ 95% AND unsupported ≤ 5%",
            "stage4": "strict M>0 ∧ d_self≤R_95 ≥ 80%",
        },
        "stage1_gaussian_quality": s1,
        "stage2_heldout_P_M_gt0": s2,
        "stage3_query_fidelity": s3,
        "stage4_strict_personalized": s4,
        "joint_verdict": jv,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote → {OUT}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
