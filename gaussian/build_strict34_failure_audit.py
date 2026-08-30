#!/usr/bin/env python3
"""Stage 4 strict34 Failure Decomposition Audit.

用户指令 2026-08-30: 把 30,233 个 no_strict_candidate (98.4%) 拆成三类,
找到 strict personalized Query 找不到的根因。

分类(基于 selection entry 已有字段):
  min_d_self = selected_distance (argmin d_self over full candidate pool)
  max_M      = selected_margin  (argmax M over full candidate pool)
  R_95       = 8.073 (Mahalanobis P95, d=48)

三类桶:
  F1 绝对距离失败 (genre gap):
    max_M > 0 AND min_d_self > R_95
    → 离用户最近 candidate 也超过 R_95, query/review genre gap 主导
  F2 相对排他失败 (Gaussian overlap):
    max_M <= 0 AND min_d_self <= R_95
    → query 风格与用户近, 但 cohort 内其他用户更近, 用户互相重叠主导
  F3 双重失败 (pool coverage):
    max_M <= 0 AND min_d_self > R_95
    → 整个 candidate pool 既不近也不专有, 生成端 pool 不覆盖
  PASSED (strict_personalized_mahal_margin_max):
    走双门的 490 (sanity check)

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_strict34.json

**输出**:
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/strict34_failure_audit.json
    + stdout 摘要

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

SELECTION_IN = SCRATCH / "stage8_5_selection_strict34.json"
SUMMARY_OUT = REPO_ROOT / "result" / "gaussian" / "strict34_failure_audit.json"

R_95 = 8.073  # Mahalanobis P95, d=48 (sqrt(chi2(0.95, 48)))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [failure_audit] {msg}", flush=True)


def classify(min_d_self: float, max_M: float) -> str:
    """Return one of F1/F2/F3."""
    if max_M > 0 and min_d_self > R_95:
        return "F1_abs_distance_fail"
    if max_M <= 0 and min_d_self <= R_95:
        return "F2_rel_exclusivity_fail"
    if max_M <= 0 and min_d_self > R_95:
        return "F3_double_fail"
    # edge case: max_M > 0 AND min_d_self <= R_95 → should be PASSED (selected)
    return "PASSED_strict"


def main():
    log("=== Stage 4 strict34 Failure Decomposition Audit ===")
    if not SELECTION_IN.exists():
        raise FileNotFoundError(f"selection required: {SELECTION_IN}")

    with open(SELECTION_IN, "r", encoding="utf-8") as f:
        sd = json.load(f)
    entries = sd["entries"]
    n_total = len(entries)
    log(f"  loaded {n_total} (user, asin) entries")

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
        n_cands = e.get("n_candidates", 0)
        # users without Gaussian metadata have None → bucket OTHER
        if sel_d is None or sel_m is None:
            buckets["OTHER"].append({
                "asin": e["asin"],
                "user_id": e["user_id"],
                "reason": "missing_gaussian_metadata",
                "user_source": e.get("user_source"),
            })
            continue
        if method == "strict_personalized_mahal_margin_max":
            # Passed strict gate
            buckets["PASSED_strict"].append({
                "asin": e["asin"],
                "user_id": e["user_id"],
                "min_d_self": sel_d,
                "max_M": sel_m,
                "n_candidates": n_cands,
            })
            continue
        if method != "no_strict_candidate":
            buckets["OTHER"].append({
                "asin": e["asin"],
                "user_id": e["user_id"],
                "method": method,
            })
            continue
        # no_strict_candidate → classify by (min_d_self, max_M)
        cat = classify(float(sel_d), float(sel_m))
        buckets[cat].append({
            "asin": e["asin"],
            "user_id": e["user_id"],
            "min_d_self": float(sel_d),
            "max_M": float(sel_m),
            "n_candidates": n_cands,
        })

    # ---- Summary stats ----
    n_f1 = len(buckets["F1_abs_distance_fail"])
    n_f2 = len(buckets["F2_rel_exclusivity_fail"])
    n_f3 = len(buckets["F3_double_fail"])
    n_pass = len(buckets["PASSED_strict"])
    n_other = len(buckets["OTHER"])
    n_fail_total = n_f1 + n_f2 + n_f3
    n_valid = n_fail_total + n_pass  # excludes OTHER

    log(f"\n=== Bucket counts (over {n_valid} with Gaussian metadata) ===")
    log(f"  PASSED_strict:                   {n_pass} ({100*n_pass/max(1,n_valid):.2f}%)")
    log(f"  F1_abs_distance_fail:            {n_f1} ({100*n_f1/max(1,n_valid):.2f}%)")
    log(f"  F2_rel_exclusivity_fail:         {n_f2} ({100*n_f2/max(1,n_valid):.2f}%)")
    log(f"  F3_double_fail:                  {n_f3} ({100*n_f3/max(1,n_valid):.2f}%)")
    log(f"  sum fail (F1+F2+F3):             {n_fail_total} ({100*n_fail_total/max(1,n_valid):.2f}%)")
    log(f"  OTHER (missing Gaussian / other method): {n_other}")
    log(f"\n  Of fail: F1 share = {100*n_f1/max(1,n_fail_total):.1f}% | "
        f"F2 share = {100*n_f2/max(1,n_fail_total):.1f}% | "
        f"F3 share = {100*n_f3/max(1,n_fail_total):.1f}%")

    # ---- Per-bucket distribution stats ----
    def stat_block(b: list[dict], key: str) -> dict:
        if not b:
            return {"n": 0}
        arr = [x[key] for x in b if x.get(key) is not None]
        if not arr:
            return {"n": 0}
        s = sorted(arr)
        n = len(s)
        return {
            "n": n,
            "mean": sum(s) / n,
            "median": s[n // 2],
            "q25": s[n // 4],
            "q75": s[3 * n // 4],
            "min": s[0],
            "max": s[-1],
        }

    log(f"\n=== Distributional stats ===")
    log(f"  F1 (abs dist fail)  min_d_self: {stat_block(buckets['F1_abs_distance_fail'], 'min_d_self')}")
    log(f"  F1 (abs dist fail)  max_M:      {stat_block(buckets['F1_abs_distance_fail'], 'max_M')}")
    log(f"  F2 (rel excl fail)  min_d_self: {stat_block(buckets['F2_rel_exclusivity_fail'], 'min_d_self')}")
    log(f"  F2 (rel excl fail)  max_M:      {stat_block(buckets['F2_rel_exclusivity_fail'], 'max_M')}")
    log(f"  F3 (double fail)    min_d_self: {stat_block(buckets['F3_double_fail'], 'min_d_self')}")
    log(f"  F3 (double fail)    max_M:      {stat_block(buckets['F3_double_fail'], 'max_M')}")
    log(f"  PASSED_strict       min_d_self: {stat_block(buckets['PASSED_strict'], 'min_d_self')}")
    log(f"  PASSED_strict       max_M:      {stat_block(buckets['PASSED_strict'], 'max_M')}")

    # ---- Dominant failure mode ----
    dominant = max(
        [("F1_abs_distance_fail (genre gap)", n_f1),
         ("F2_rel_exclusivity_fail (user overlap)", n_f2),
         ("F3_double_fail (pool coverage)", n_f3)],
        key=lambda x: x[1],
    )
    log(f"\n=== DOMINANT FAILURE MODE: {dominant[0]} ({dominant[1]} / {n_fail_total} = "
        f"{100*dominant[1]/max(1,n_fail_total):.1f}% of fails) ===")

    # ---- Save ----
    out = {
        "description": (
            "Stage 4 strict34 failure decomposition: classify no_strict_candidate into "
            "F1 (max_M>0 & min_d_self>R_95: genre gap / absolute distance), "
            "F2 (max_M≤0 & min_d_self≤R_95: Gaussian overlap), "
            "F3 (max_M≤0 & min_d_self>R_95: pool coverage gap). "
            "min_d_self = argmin d_self over full candidate pool; "
            "max_M = argmax M over full candidate pool."
        ),
        "thresholds": {"R_95": R_95, "PCA_DIM": 48},
        "n_total": n_total,
        "n_valid": n_valid,
        "n_other_excluded": n_other,
        "buckets": {
            "PASSED_strict": {"n": n_pass, "pct_of_valid": n_pass / max(1, n_valid)},
            "F1_abs_distance_fail": {
                "n": n_f1,
                "pct_of_valid": n_f1 / max(1, n_valid),
                "pct_of_fails": n_f1 / max(1, n_fail_total),
                "diagnostic": (
                    "Query closest to target user is still outside R_95 Mahalanobis. "
                    "Dominant cause: query/review genre gap — generated query lexicon "
                    "and structure differ fundamentally from user's historical review style."
                ),
                "stats": {
                    "min_d_self": stat_block(buckets["F1_abs_distance_fail"], "min_d_self"),
                    "max_M": stat_block(buckets["F1_abs_distance_fail"], "max_M"),
                },
            },
            "F2_rel_exclusivity_fail": {
                "n": n_f2,
                "pct_of_valid": n_f2 / max(1, n_valid),
                "pct_of_fails": n_f2 / max(1, n_fail_total),
                "diagnostic": (
                    "Closest query is within R_95 (similar style) BUT some other user in "
                    "cohort is even closer (M<=0). Dominant cause: real Gaussian overlap "
                    "between users — strict personalized query genuinely hard to find."
                ),
                "stats": {
                    "min_d_self": stat_block(buckets["F2_rel_exclusivity_fail"], "min_d_self"),
                    "max_M": stat_block(buckets["F2_rel_exclusivity_fail"], "max_M"),
                },
            },
            "F3_double_fail": {
                "n": n_f3,
                "pct_of_valid": n_f3 / max(1, n_valid),
                "pct_of_fails": n_f3 / max(1, n_fail_total),
                "diagnostic": (
                    "Closest query is both outside R_95 AND beaten by another user. "
                    "Dominant cause: candidate generation pool does not cover this user's "
                    "syntactic region — pool coverage gap (gen_query side issue, not selection)."
                ),
                "stats": {
                    "min_d_self": stat_block(buckets["F3_double_fail"], "min_d_self"),
                    "max_M": stat_block(buckets["F3_double_fail"], "max_M"),
                },
            },
            "OTHER": {
                "n": n_other,
                "diagnostic": "Missing Gaussian metadata or non-strict-method entries (excluded from classification).",
            },
        },
        "dominant_failure_mode": {
            "name": dominant[0],
            "n": dominant[1],
            "pct_of_fails": dominant[1] / max(1, n_fail_total),
        },
        "decision": (
            "T=34 Gaussian quality is FROZEN. "
            "If F1 dominates → relax R_95 (loosen absolute gate, e.g. R_99 = 9.34) or add "
            "review-style generation (style-anchor LLM exemplar). "
            "If F2 dominates → strict personalized is fundamentally infeasible at this T; "
            "consider widening cohort per user (k-NN multi-anchor) or accepting high-overlap "
            "as the ceiling. "
            "If F3 dominates → regen_query with stronger pool diversity / larger K_POOL or "
            "switch to hybrid (review-style + LLM-generated) candidate source."
        ),
        "samples": {
            "F1": buckets["F1_abs_distance_fail"][:5],
            "F2": buckets["F2_rel_exclusivity_fail"][:5],
            "F3": buckets["F3_double_fail"][:5],
        },
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
