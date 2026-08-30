#!/usr/bin/env python3
"""Phase 3 — Prototype-Guided Pool Audit.

用户指令 2026-08-30: Phase 1+2 已完成 (syntax-prototype guided pool generation)。
现在用同样的 Stage 4 strict alignment pipeline (T=34 / F3 + PCA48 / R_95=8.073)
对 prototype_pool_K16.json 跑一遍, 然后做 F1/F2/F3 失败分解,
比较 vs baseline K=200 pool (86.9% F3 / 11.15 min_d_self median / -1.99 max_M median)。

成功判别准则 (用户指定):
  F3: 86.9% → ≤ 50%
  min_d_self median: 11.15 → 8~9
  (max_M 中位数提升辅助判断)
→ 三者都达成 ⇒ DISTRIBUTION_MISALIGNMENT 被证, generation distribution 是根因
→ 只 F3 下降但 min_d_self 仍 > 10 ⇒ 部分覆盖, 仍需更细粒度 prototype
→ F3 不下降 ⇒ distribution 改不动, 转入 user-specific exemplar anchor 路线

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/prototype_pool_K16.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins_strict34.json (cohort)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_quality_strict_users.json

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_proto_K16.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_stats_proto_K16.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/prototype_pool_audit.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/prototype_pool_summary.json

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")

PROTO_POOL = SCRATCH / "prototype_pool_K16.json"
SEL_OUT = SCRATCH / "stage8_5_selection_proto_K16.json"
STATS_OUT = SCRATCH / "stage8_5_selection_stats_proto_K16.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features_proto_K16.jsonl.gz"

AUDIT_OUT = REPO_ROOT / "result" / "gaussian" / "prototype_pool_audit.json"

# Baseline numbers (K=200 pool, no prototype guidance)
BASELINE = {
    "F3_pct_fails": 0.869,
    "F3_min_d_self_median": 11.15,
    "F3_max_M_median": -1.99,
    "PASSED_pct_valid": 0.0159,
}

R_95 = 8.073
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [proto_audit] {msg}", flush=True)


def run_stage4() -> dict:
    """Run Stage 4 strict alignment using prototype_pool_K16.json."""
    log(f"=== Stage 4 strict alignment on prototype_pool_K16.json ===")
    if not PROTO_POOL.exists():
        raise FileNotFoundError(f"prototype pool required: {PROTO_POOL}")

    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(PROTO_POOL)
    env["POOL_FEAT_CACHE"] = str(FEAT_CACHE)
    env["ASINS_OUT_SUFFIX"] = "_proto_K16"
    env["SEL_OUT_SUFFIX"] = "_proto_K16"
    env["STAGE4_USER_FILTER"] = "1"

    log_path = LOG_DIR / "stage4_select_proto_K16.log"
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
    """F1/F2/F3 decomposition."""
    log(f"=== F1/F2/F3 audit on {SEL_OUT} ===")
    with open(SEL_OUT, "r", encoding="utf-8") as f:
        sd = json.load(f)
    entries = sd["entries"]

    buckets: dict[str, list[dict]] = {
        "F1_abs_distance_fail": [],
        "F2_rel_exclusivity_fail": [],
        "F3_double_fail": [],
        "PASSED_strict": [],
        "OTHER": [],
    }

    for e in entries:
        method = e.get("selection_method", "")
        sel_d = e.get("selected_distance")
        sel_m = e.get("selected_margin")
        if sel_d is None or sel_m is None:
            buckets["OTHER"].append({
                "asin": e["asin"], "user_id": e["user_id"],
                "reason": "missing_gaussian_metadata",
            })
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
        return {
            "n": n, "mean": sum(s) / n, "median": s[n // 2],
            "q25": s[n // 4], "q75": s[3 * n // 4],
            "min": s[0], "max": s[-1],
        }

    audit_doc = {
        "K_pool_kind": "prototype_K16",
        "R_95": R_95,
        "n_valid": n_valid,
        "n_other_excluded": n_other,
        "buckets": {
            "PASSED_strict": {"n": n_pass, "pct_valid": n_pass / max(1, n_valid)},
            "F1_abs_distance_fail": {
                "n": n_f1,
                "pct_valid": n_f1 / max(1, n_valid),
                "pct_fails": n_f1 / max(1, n_fail),
                "stats_min_d_self": stats(buckets["F1_abs_distance_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F1_abs_distance_fail"], "max_M"),
            },
            "F2_rel_exclusivity_fail": {
                "n": n_f2,
                "pct_valid": n_f2 / max(1, n_valid),
                "pct_fails": n_f2 / max(1, n_fail),
                "stats_min_d_self": stats(buckets["F2_rel_exclusivity_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F2_rel_exclusivity_fail"], "max_M"),
            },
            "F3_double_fail": {
                "n": n_f3,
                "pct_valid": n_f3 / max(1, n_valid),
                "pct_fails": n_f3 / max(1, n_fail),
                "stats_min_d_self": stats(buckets["F3_double_fail"], "min_d_self"),
                "stats_max_M": stats(buckets["F3_double_fail"], "max_M"),
            },
        },
    }

    log(f"\n=== Bucket counts (over {n_valid} with Gaussian metadata) ===")
    log(f"  PASSED_strict:    {n_pass} ({100*n_pass/max(1,n_valid):.2f}%)")
    log(f"  F1 (abs dist):    {n_f1} ({100*n_f1/max(1,n_fail):.1f}% of fails)")
    log(f"  F2 (rel excl):    {n_f2} ({100*n_f2/max(1,n_fail):.1f}% of fails)")
    log(f"  F3 (double):      {n_f3} ({100*n_f3/max(1,n_fail):.1f}% of fails)")
    log(f"\n  F3 min_d_self median: {audit_doc['buckets']['F3_double_fail']['stats_min_d_self'].get('median')}")
    log(f"  F3 max_M      median: {audit_doc['buckets']['F3_double_fail']['stats_max_M'].get('median')}")

    return audit_doc


def evaluate_vs_baseline(audit_doc: dict) -> dict:
    """Apply user's success criteria."""
    b = audit_doc["buckets"]
    new_F3 = b["F3_double_fail"]["pct_fails"]
    new_min_d = b["F3_double_fail"]["stats_min_d_self"].get("median")
    new_max_M = b["F3_double_fail"]["stats_max_M"].get("median")
    new_pass = b["PASSED_strict"]["pct_valid"]

    # Decision logic (用户指定):
    #   F3 ≤ 50% AND min_d_self ≤ 9 → DISTRIBUTION_MISALIGNMENT_CONFIRMED
    #   F3 drops ≥ 10pp but min_d_self > 9 → PARTIAL_COVERAGE
    #   F3 drops < 5pp → NO_GO
    f3_drop_pp = (BASELINE["F3_pct_fails"] - new_F3) * 100
    delta_min_d = new_min_d - BASELINE["F3_min_d_self_median"] if new_min_d else None

    if new_F3 <= 0.50 and new_min_d and new_min_d <= 9.0:
        decision = "DISTRIBUTION_MISALIGNMENT_CONFIRMED"
        rationale = (
            f"F3 dropped {f3_drop_pp:.1f}pp (86.9% → {100*new_F3:.1f}%) AND "
            f"min_d_self median {new_min_d:.2f} ≤ 9 → candidate syntax coverage "
            f"is the true bottleneck. Prototype-guided generation redistributes "
            f"candidate mass into user-syntax regions."
        )
    elif f3_drop_pp >= 10.0:
        decision = "PARTIAL_COVERAGE"
        rationale = (
            f"F3 dropped {f3_drop_pp:.1f}pp (≥10pp) but min_d_self median "
            f"{new_min_d:.2f} still > 9 → prototype guidance helps but does not "
            f"fully close the gap. Try denser K_syntax (32) or per-user exemplar."
        )
    else:
        decision = "NO_GO"
        rationale = (
            f"F3 dropped only {f3_drop_pp:.1f}pp (<5pp). Generation distribution "
            f"still does not cover user syntactic region even with abstract prototype "
            f"guidance. Switch to user-specific exemplar anchor route."
        )

    log(f"\n=== Decision ===")
    log(f"  ΔF3 pp = {f3_drop_pp:+.1f}")
    log(f"  Δmin_d_self = {delta_min_d:+.2f}" if delta_min_d else "")
    log(f"  Decision: {decision}")
    log(f"  {rationale}")

    return {
        "baseline": BASELINE,
        "prototype_K16": {
            "F3_pct_fails": new_F3,
            "F3_min_d_self_median": new_min_d,
            "F3_max_M_median": new_max_M,
            "PASSED_pct_valid": new_pass,
        },
        "delta": {
            "F3_drop_pp": f3_drop_pp,
            "min_d_self_delta": delta_min_d,
        },
        "decision": decision,
        "rationale": rationale,
    }


def main():
    log(f"=== Phase 3 — Prototype-Guided Pool Audit ===")
    log(f"  pool: {PROTO_POOL}")
    log(f"  baseline: F3%={BASELINE['F3_pct_fails']*100:.1f}, "
        f"min_d_self_med={BASELINE['F3_min_d_self_median']}, "
        f"max_M_med={BASELINE['F3_max_M_median']}")

    if not PROTO_POOL.exists():
        raise FileNotFoundError(
            f"Phase 2 must finish first: {PROTO_POOL} not found"
        )

    # ---- Stage 4 ----
    run_stats = run_stage4()

    # ---- Audit ----
    audit_doc = audit()

    # ---- Decision ----
    eval_doc = evaluate_vs_baseline(audit_doc)

    # ---- Save ----
    out = {
        "description": (
            "Phase 3 audit: syntax-prototype guided pool (K_syntax=16) re-run through "
            "Stage 4 strict alignment (T=34, F3+PCA48, M>0 + d_self≤R_95=8.073). "
            "Compared against baseline K=200 pool (F3=86.9%, min_d_self=11.15, "
            "max_M=-1.99). Goal: prove DISTRIBUTION_MISALIGNMENT by showing F3 "
            "drops and min_d_self approaches R_95."
        ),
        "config": {
            "pool_path": str(PROTO_POOL),
            "R_95": R_95,
            "PCA_DIM": 48,
            "K_syntax": 16,
            "K_per_proto": 4,
            "K_POOL_eff": 64,
        },
        "run_stats_sec": run_stats,
        "audit": audit_doc,
        "evaluation": eval_doc,
        "next_step_hint": {
            "DISTRIBUTION_MISALIGNMENT_CONFIRMED":
                "Move to Stage 5 retrieval (7 retrievers) on prototype_pool_K16 — "
                "expect sim09 RR/BGE/GTE/SPLADE flip > baseline K=200 pool.",
            "PARTIAL_COVERAGE":
                "Try K_syntax=32 (denser), then K_syntax=8 (sparser) to map the "
                "F3 vs K_syntax Pareto.",
            "NO_GO":
                "Switch to user-specific exemplar anchor: pull 1-2 real review "
                "sentences per user, use as LLM style seed (still no verbatim "
                "review into pool).",
        },
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {AUDIT_OUT}")


if __name__ == "__main__":
    main()