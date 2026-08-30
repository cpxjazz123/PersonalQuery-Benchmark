#!/usr/bin/env python3
"""Phase 3 PILOT — Audit prototype-guided pool on 50 ASINs.

用户指令 2026-08-30: 在小规模 pilot 上验证 prototype guidance 是否真改善 F3%/min_d_self。
完整版 build_strict34_prototype_audit.py 跑全 2174 ASINs 需要 ~10min Phase 4 + audit;
PILOT 只取 50 ASINs 跑同样的 pipeline, 立即出结论决定是否上全量。

**输入**:
  - prototype_pool_K16_pilot.json
  - stage8_5_asins_strict34_intersect2174_ksweep_K200.json (取前 50 ASINs,
    与 pilot 同样的种子=42 抽样, 保证 audit pool 与 generation pool 用同一批 ASINs)

**输出**:
  - result/gaussian/prototype_pool_audit_pilot.json
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")

PROTO_POOL = SCRATCH / "prototype_pool_K16_pilot.json"
COHORT_ASINS = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"
SEL_OUT = SCRATCH / "stage8_5_selection_proto_K16_pilot.json"
STATS_OUT = SCRATCH / "stage8_5_selection_stats_proto_K16_pilot.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features_proto_K16_pilot.jsonl.gz"
AUDIT_OUT = REPO_ROOT / "result" / "gaussian" / "prototype_pool_audit_pilot.json"

PILOT_N_ASIN = 50
RANDOM_SEED = 42
R_95 = 8.073
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

BASELINE = {
    "F3_pct_fails": 0.869,
    "F3_min_d_self_median": 11.15,
    "F3_max_M_median": -1.99,
    "PASSED_pct_valid": 0.0159,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [proto_audit_pilot] {msg}", flush=True)


def write_subset_cohort() -> int:
    """Write a 50-ASIN cohort subset under a suffix Stage 4 expects to find.

    Stage 4 reads 'stage8_5_asins{ASINS_OUT_SUFFIX}.json'. We suffix
    '_proto_K16_pilot' and write the SAME 50 ASINs (matched to pilot pool
    generation) so Stage 4 user-filter gate accepts them.
    """
    log(f"=== Build 50-ASIN pilot cohort ===")
    with open(COHORT_ASINS, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    # Match the pilot's sampling: canonical-sorted, then seeded shuffle
    rng = random.Random(RANDOM_SEED)
    asins_sorted = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng.shuffle(asins_sorted)
    kept = asins_sorted[:PILOT_N_ASIN]
    n_users = sum(e["n_users_quality"] for e in kept)
    log(f"  cohort subset: {len(kept)} ASINs, {n_users} (user,asin) pairs")

    out_doc = {
        "config": {
            "filter": "strict34_intersect_proto_K16_pilot",
            "cv_nll_threshold": 34,
            "validation_precision": 0.9821,
            "PILOT_N_ASIN": PILOT_N_ASIN,
            "RANDOM_SEED": RANDOM_SEED,
        },
        "n_asins": len(kept),
        "n_users_total": n_users,
        "asins": kept,
    }
    suffix = "_proto_K16_pilot"
    out_path = SCRATCH / f"stage8_5_asins{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {out_path}")
    return len(kept)


def run_stage4() -> dict:
    log(f"=== Stage 4 strict alignment (PILOT) ===")
    if not PROTO_POOL.exists():
        raise FileNotFoundError(f"prototype pilot pool required: {PROTO_POOL}")

    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(PROTO_POOL)
    env["POOL_FEAT_CACHE"] = str(FEAT_CACHE)
    env["ASINS_OUT_SUFFIX"] = "_proto_K16_pilot"
    env["SEL_OUT_SUFFIX"] = "_proto_K16_pilot"
    env["STAGE4_USER_FILTER"] = "1"

    log_path = LOG_DIR / "stage4_select_proto_K16_pilot.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fl:
        rc = subprocess.run(
            [PYTHON, "select_query/syntax_subspace_select_strict_alignment.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fl,
            stderr=subprocess.STDOUT,
        ).returncode
    elapsed = time.time() - t0
    log(f"  Stage 4 done in {elapsed:.1f}s, rc={rc}")
    if rc != 0:
        raise RuntimeError(f"Stage 4 failed rc={rc}, see {log_path}")
    return {"stage4_sec": elapsed}


def classify(min_d_self: float, max_M: float) -> str:
    if max_M > 0 and min_d_self > R_95:
        return "F1_abs_distance_fail"
    if max_M <= 0 and min_d_self <= R_95:
        return "F2_rel_exclusivity_fail"
    if max_M <= 0 and min_d_self > R_95:
        return "F3_double_fail"
    return "PASSED_strict"


def audit() -> dict:
    log(f"=== F1/F2/F3 audit (PILOT) ===")
    if not SEL_OUT.exists():
        raise FileNotFoundError(f"selection required: {SEL_OUT}")
    with open(SEL_OUT, "r", encoding="utf-8") as f:
        sd = json.load(f)
    entries = sd["entries"]

    buckets = {"F1_abs_distance_fail": [], "F2_rel_exclusivity_fail": [],
               "F3_double_fail": [], "PASSED_strict": [], "OTHER": []}
    for e in entries:
        method = e.get("selection_method", "")
        sel_d = e.get("selected_distance")
        sel_m = e.get("selected_margin")
        if sel_d is None or sel_m is None:
            buckets["OTHER"].append({"asin": e["asin"], "user_id": e["user_id"], "reason": "missing"})
            continue
        if method == "strict_personalized_mahal_margin_max":
            buckets["PASSED_strict"].append({
                "asin": e["asin"], "user_id": e["user_id"],
                "min_d_self": float(sel_d), "max_M": float(sel_m),
            })
            continue
        if method != "no_strict_candidate":
            buckets["OTHER"].append({"asin": e["asin"], "user_id": e["user_id"], "method": method})
            continue
        cat = classify(float(sel_d), float(sel_m))
        buckets[cat].append({
            "asin": e["asin"], "user_id": e["user_id"],
            "min_d_self": float(sel_d), "max_M": float(sel_m),
        })

    n_pass = len(buckets["PASSED_strict"])
    n_f1 = len(buckets["F1_abs_distance_fail"])
    n_f2 = len(buckets["F2_rel_exclusivity_fail"])
    n_f3 = len(buckets["F3_double_fail"])
    n_other = len(buckets["OTHER"])
    n_fail = n_f1 + n_f2 + n_f3
    n_valid = n_pass + n_fail

    def stats(arr, key):
        if not arr:
            return {"n": 0}
        s = sorted([x[key] for x in arr if x.get(key) is not None])
        n = len(s)
        return {"n": n, "mean": sum(s)/n, "median": s[n//2],
                "q25": s[n//4], "q75": s[3*n//4], "min": s[0], "max": s[-1]}

    log(f"\n=== Pilot buckets (over {n_valid} valid (user,asin) pairs) ===")
    log(f"  PASSED:    {n_pass} ({100*n_pass/max(1,n_valid):.2f}%)")
    log(f"  F1:        {n_f1} ({100*n_f1/max(1,n_fail):.1f}% of fails)")
    log(f"  F2:        {n_f2} ({100*n_f2/max(1,n_fail):.1f}% of fails)")
    log(f"  F3:        {n_f3} ({100*n_f3/max(1,n_fail):.1f}% of fails)")
    log(f"  OTHER:     {n_other}")

    return {
        "K_pool_kind": "prototype_K16_pilot_50ASIN",
        "n_valid": n_valid,
        "n_other_excluded": n_other,
        "buckets": {
            "PASSED_strict": {"n": n_pass, "pct_valid": n_pass/max(1,n_valid)},
            "F1_abs_distance_fail": {
                "n": n_f1, "pct_fails": n_f1/max(1,n_fail),
                "stats_min_d_self": stats(buckets["F1_abs_distance_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F1_abs_distance_fail"], "max_M"),
            },
            "F2_rel_exclusivity_fail": {
                "n": n_f2, "pct_fails": n_f2/max(1,n_fail),
                "stats_min_d_self": stats(buckets["F2_rel_exclusivity_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F2_rel_exclusivity_fail"], "max_M"),
            },
            "F3_double_fail": {
                "n": n_f3, "pct_fails": n_f3/max(1,n_fail),
                "stats_min_d_self": stats(buckets["F3_double_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F3_double_fail"], "max_M"),
            },
        },
    }


def evaluate(audit_doc: dict) -> dict:
    b = audit_doc["buckets"]
    new_F3 = b["F3_double_fail"]["pct_fails"]
    new_min_d = b["F3_double_fail"]["stats_min_d_self"].get("median")
    new_max_M = b["F3_double_fail"]["stats_max_M"].get("median")
    new_pass = b["PASSED_strict"]["pct_valid"]
    f3_drop_pp = (BASELINE["F3_pct_fails"] - new_F3) * 100
    delta_min_d = new_min_d - BASELINE["F3_min_d_self_median"] if new_min_d else None

    log(f"\n=== Pilot vs baseline ===")
    log(f"  baseline F3%        = {100*BASELINE['F3_pct_fails']:.1f}")
    log(f"  pilot   F3%        = {100*new_F3:.1f}")
    log(f"  ΔF3 (drop pp)      = {f3_drop_pp:+.1f}")
    log(f"  baseline min_d_self = {BASELINE['F3_min_d_self_median']:.2f}")
    log(f"  pilot   min_d_self = {new_min_d}")
    log(f"  Δmin_d_self        = {delta_min_d:+.2f}" if delta_min_d is not None else "")
    log(f"  pilot   max_M      = {new_max_M}")

    if new_F3 <= 0.50 and new_min_d and new_min_d <= 9.0:
        decision = "GO_FULL_SCALE_DISTRIBUTION_CONFIRMED"
    elif f3_drop_pp >= 10.0:
        decision = "GO_FULL_SCALE_PARTIAL_COVERAGE"
    else:
        decision = "NO_GO_FULL_SCALE"

    log(f"\n  DECISION (PILOT): {decision}")
    return {
        "baseline": BASELINE,
        "pilot": {
            "F3_pct_fails": new_F3,
            "F3_min_d_self_median": new_min_d,
            "F3_max_M_median": new_max_M,
            "PASSED_pct_valid": new_pass,
        },
        "delta": {"F3_drop_pp": f3_drop_pp, "min_d_self_delta": delta_min_d},
        "decision": decision,
    }


def main():
    log("=== Phase 3 PILOT — 50 ASINs prototype audit ===")
    if not PROTO_POOL.exists():
        raise FileNotFoundError(f"Phase 2 PILOT must finish first: {PROTO_POOL}")

    n_cohort = write_subset_cohort()
    run_stats = run_stage4()
    audit_doc = audit()
    eval_doc = evaluate(audit_doc)

    out = {
        "description": (
            "Phase 3 PILOT on 50 ASINs — syntax-prototype guided pool K=16 "
            "vs K=200 baseline. Pilot guards 75min full-scale run: if pilot "
            "shows no F3 improvement (decision=NO_GO), skip full run and "
            "switch to user-specific exemplar route."
        ),
        "config": {
            "pool_path": str(PROTO_POOL),
            "cohort_size": n_cohort,
            "R_95": R_95,
            "K_syntax": 16,
        },
        "run_stats_sec": run_stats,
        "audit": audit_doc,
        "evaluation": eval_doc,
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {AUDIT_OUT}")


if __name__ == "__main__":
    main()