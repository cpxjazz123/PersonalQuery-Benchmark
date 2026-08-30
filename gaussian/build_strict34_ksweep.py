#!/usr/bin/env python3
"""Strict34 K_POOL coverage sweep — K ∈ {50, 100, 200}.

用户指令 2026-08-30: F3=86.9% 主因是 candidate pool coverage gap, 但还没证明
"只增加 K 就一定能解决"。这一刀切判别实验固定所有东西 (T=34 / F3 CoreStruct +
PCA48 / whitening / M>0 + d_self≤R_95), 只换 K_POOL, 重新跑 Stage 4 + audit,
看 F3% 是否单调下降:

  K↑ ⇒ F3↓ ?  →  pool 太小是主因, 继续扩 K 即可
  K↑ ⇒ F3 ≈    →  生成端分布不对, 需 review-style exemplar anchor / 改 prompt

ASIN 集合: 固定 cohort ∩ K=200 pool 的 736 ASINs, 保证 3 个 K 在同样的 ASINs
上比 (排除 ASIN coverage 干扰)。

每个 K 用独立 features cache (POOL_FEAT_CACHE) 避免互相污染。

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K50_F3pca48_backup.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K100_F3pca48.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins_strict34.json

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_ksweep_K{50,100,200}.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_stats_ksweep_K{50,100,200}.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/strict34_ksweep_results.json (聚合 summary)
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/strict34_ksweep_audit.json (per-K F1/F2/F3 buckets)

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

ASINS_STRICT = SCRATCH / "stage8_5_asins_strict34.json"
SWEEP_OUT = REPO_ROOT / "result" / "gaussian" / "strict34_ksweep_results.json"
AUDIT_OUT = REPO_ROOT / "result" / "gaussian" / "strict34_ksweep_audit.json"

# Stage 4 reads "stage8_5_asins{ASINS_OUT_SUFFIX}.json" (line 319 of select script).
# Each K run sets ASINS_OUT_SUFFIX=_ksweep_K{K}; write the SAME intersect cohort
# under each per-K filename so the user-filter gate finds it. (No per-K cohort
# variation — K-sweep varies pool only, cohort is fixed intersect=2174 ASINs.)
ASINS_INTERSECT_BASE = "stage8_5_asins"  # Stage 4 builds f"stage8_5_asins{suffix}" → uses this directly

PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

# Pool paths (already exist on disk)
K_VALUES = [50, 100, 200]
POOL_PATHS = {
    50: SCRATCH / "pool_K50_F3pca48_backup.json",
    100: SCRATCH / "pool_K100_F3pca48.json",
    200: SCRATCH / "pool_K200_F3pca48_full.json",
}

# Intersect ASINs: cohort ∩ K=200 pool (most permissive)
ASINS_INTERSECT_OUT = SCRATCH / "stage8_5_asins_strict34_intersect736.json"

R_95 = 8.073  # Mahalanobis P95, d=48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [ksweep] {msg}", flush=True)


def build_intersect_cohort() -> dict:
    """Restrict cohort to ASINs present in K=200 pool (= 736)."""
    log(f"=== Step 0: build intersect cohort (cohort ∩ K=200 pool) ===")
    with open(ASINS_STRICT, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    with open(POOL_PATHS[200], "r", encoding="utf-8") as f:
        pool200 = json.load(f)
    pool200_asins = set(pool200["pools"].keys())
    log(f"  cohort ASINs: {len(cohort['asins'])}, K=200 pool ASINs: {len(pool200_asins)}")
    kept = [e for e in cohort["asins"] if e["asin"] in pool200_asins]
    log(f"  intersect: {len(kept)} ASINs")
    n_users_pre = sum(e["n_users_quality"] for e in cohort["asins"])
    n_users_post = sum(e["n_users_quality"] for e in kept)
    log(f"  (user,asin) pairs: {n_users_pre} → {n_users_post}")

    n_ge1 = sum(1 for e in kept if e["n_users_quality"] >= 1)
    n_ge5 = sum(1 for e in kept if e["n_users_quality"] >= 5)
    n_ge10 = sum(1 for e in kept if e["n_users_quality"] >= 10)
    log(f"  ASIN coverage: ≥1={n_ge1}, ≥5={n_ge5}, ≥10={n_ge10}")

    out_doc = {
        "config": {
            "filter": "strict34_intersect_K200pool",
            "cv_nll_threshold": 34,
            "validation_precision": 0.9821,
        },
        "n_asins": len(kept),
        "n_users_total": n_users_post,
        "n_asins_ge1": n_ge1,
        "n_asins_ge5": n_ge5,
        "n_asins_ge10": n_ge10,
        "asins": kept,
    }
    # Write 3 per-K suffix copies so Stage 4 finds its expected path
    for K in K_VALUES:
        suffix_path = SCRATCH / f"{ASINS_INTERSECT_BASE}_ksweep_K{K}.json"
        with open(suffix_path, "w", encoding="utf-8") as f:
            json.dump(out_doc, f, ensure_ascii=False)
        log(f"  wrote → {suffix_path}")
    # Also write canonical base (for audit / manual inspection)
    base_path = SCRATCH / f"{ASINS_INTERSECT_BASE}_strict34_intersect2174.json"
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {base_path}")
    return {
        "n_asins": len(kept),
        "n_users_total_post": n_users_post,
        "n_asins_ge1": n_ge1,
        "n_asins_ge5": n_ge5,
        "n_asins_ge10": n_ge10,
    }


def run_stage4_for_k(K: int) -> dict:
    """Run Stage 4 strict alignment with a specific pool. Independent feat cache."""
    log(f"\n=== Stage 4 strict alignment K={K} ===")
    suffix = f"_ksweep_K{K}"
    feat_cache = SCRATCH / f"stage7b_query_features_ksweep_K{K}.jsonl.gz"
    log(f"  pool: {POOL_PATHS[K]}")
    log(f"  feat cache: {feat_cache}")
    log(f"  suffix: {suffix}")

    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(POOL_PATHS[K])
    env["POOL_FEAT_CACHE"] = str(feat_cache)
    env["ASINS_OUT_SUFFIX"] = suffix
    env["SEL_OUT_SUFFIX"] = suffix
    env["STAGE4_USER_FILTER"] = "1"

    log_path = LOG_DIR / f"stage4_select_strict34_ksweep_K{K}.log"
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
    log(f"  Stage 4 K={K} done in {elapsed:.1f}s, rc={rc}")
    if rc != 0:
        raise RuntimeError(f"Stage 4 K={K} failed rc={rc}, see {log_path}")
    return {"K": K, "stage4_sec": elapsed}


def classify(min_d_self: float, max_M: float) -> str:
    if max_M > 0 and min_d_self > R_95:
        return "F1_abs_distance_fail"
    if max_M <= 0 and min_d_self <= R_95:
        return "F2_rel_exclusivity_fail"
    if max_M <= 0 and min_d_self > R_95:
        return "F3_double_fail"
    return "PASSED_strict"


def audit_k(K: int) -> dict:
    """Run failure decomposition for one K."""
    sel_path = SCRATCH / f"stage8_5_selection_ksweep_K{K}.json"
    with open(sel_path, "r", encoding="utf-8") as f:
        sd = json.load(f)
    entries = sd["entries"]

    buckets: dict[str, list[dict]] = {
        "F1_abs_distance_fail": [],
        "F2_rel_exclusivity_fail": [],
        "F3_double_fail": [],
        "PASSED_strict": [],
    }

    for e in entries:
        method = e.get("selection_method", "")
        sel_d = e.get("selected_distance")
        sel_m = e.get("selected_margin")
        if sel_d is None or sel_m is None:
            continue
        if method == "strict_personalized_mahal_margin_max":
            buckets["PASSED_strict"].append({
                "asin": e["asin"], "user_id": e["user_id"],
                "min_d_self": float(sel_d), "max_M": float(sel_m),
            })
            continue
        if method != "no_strict_candidate":
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

    audit = {
        "K": K,
        "pool_path": str(POOL_PATHS[K]),
        "n_valid": n_valid,
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
        "samples": {
            "F3": buckets["F3_double_fail"][:3],
        },
    }
    log(f"  K={K} audit: PASSED={n_pass} ({100*n_pass/max(1,n_valid):.2f}%) | "
        f"F1={n_f1} ({100*n_f1/max(1,n_fail):.1f}% fail) | "
        f"F2={n_f2} ({100*n_f2/max(1,n_fail):.1f}% fail) | "
        f"F3={n_f3} ({100*n_f3/max(1,n_fail):.1f}% fail)")
    log(f"    F3 min_d_self median: {audit['buckets']['F3_double_fail']['stats_min_d_self'].get('median')}")
    log(f"    F3 max_M median:      {audit['buckets']['F3_double_fail']['stats_max_M'].get('median')}")
    return audit


def main():
    log(f"=== Strict34 K_POOL coverage sweep ===")
    log(f"  K values: {K_VALUES}")
    log(f"  R_95 = {R_95}")

    # ---- 0. Build intersect cohort (cohort ∩ K=200 pool) ----
    cohort_intersect = build_intersect_cohort()

    # ---- 1. Run Stage 4 for each K ----
    run_stats = []
    for K in K_VALUES:
        s = run_stage4_for_k(K)
        run_stats.append(s)

    # ---- 2. Audit each K ----
    audits = {}
    for K in K_VALUES:
        audits[str(K)] = audit_k(K)

    # ---- 3. Compute trend ----
    f3_pct = [audits[str(K)]["buckets"]["F3_double_fail"]["pct_fails"] for K in K_VALUES]
    pass_pct = [audits[str(K)]["buckets"]["PASSED_strict"]["pct_valid"] for K in K_VALUES]
    f3_min_d_self_med = [
        audits[str(K)]["buckets"]["F3_double_fail"]["stats_min_d_self"].get("median") for K in K_VALUES
    ]
    f3_max_M_med = [
        audits[str(K)]["buckets"]["F3_double_fail"]["stats_max_M"].get("median") for K in K_VALUES
    ]

    log(f"\n=== Trend ===")
    for K, f3, pp in zip(K_VALUES, f3_pct, pass_pct):
        log(f"  K={K:3d}  PASSED={100*pp:.2f}%  F3={100*f3:.2f}%")
    log(f"  F3 min_d_self median: {[f'{x:.2f}' if x else 'N/A' for x in f3_min_d_self_med]}")
    log(f"  F3 max_M      median: {[f'{x:.3f}' if x else 'N/A' for x in f3_max_M_med]}")

    # Decision rule
    f3_drop = f3_pct[-1] - f3_pct[0]
    if f3_drop < -0.10:  # F3 drops > 10pp when K 50→200
        decision = "K_COVERAGE_BOTTLENECK"
        recommendation = (
            "Pool coverage is the bottleneck. Continue expanding K_POOL "
            "(e.g. K=400, K=800) until F3% plateaus."
        )
    elif abs(f3_drop) < 0.05:
        decision = "DISTRIBUTION_MISALIGNMENT"
        recommendation = (
            "K-sweep shows F3% is flat (K does not help). The bottleneck is "
            "not pool size but the generation distribution itself. Need to "
            "change generation (review-style exemplar anchor, prompt redesign, "
            "or hybrid pool)."
        )
    else:
        decision = "AMBIGUOUS_MIXED"
        recommendation = (
            "Mixed signal — F3% drops but not dramatically. Try both: "
            "increase K further AND add review-style exemplar."
        )
    log(f"\n  DECISION: {decision}")
    log(f"  RECOMMENDATION: {recommendation}")

    # ---- 4. Aggregate ----
    out = {
        "description": (
            "Strict34 K_POOL coverage ablation: K ∈ {50, 100, 200}. "
            "Fixed: T=34 (cv_nll ≤ 34, val precision 98.21%), "
            "F3_CoreStruct + PCA48 + Mahalanobis + whitening, "
            "gate M>0 AND d_self≤R_95=8.073. "
            "Only K_POOL changes. Same 736 ASINs (cohort ∩ K=200 pool)."
        ),
        "cohort_intersect": cohort_intersect,
        "thresholds": {"R_95": R_95, "PCA_DIM": 48},
        "K_values": K_VALUES,
        "per_K": audits,
        "trend": {
            "F3_pct_fails": {str(K): audits[str(K)]["buckets"]["F3_double_fail"]["pct_fails"] for K in K_VALUES},
            "PASSED_pct_valid": {str(K): audits[str(K)]["buckets"]["PASSED_strict"]["pct_valid"] for K in K_VALUES},
            "F3_min_d_self_median": {str(K): f3_min_d_self_med[i] for i, K in enumerate(K_VALUES)},
            "F3_max_M_median": {str(K): f3_max_M_med[i] for i, K in enumerate(K_VALUES)},
        },
        "decision": decision,
        "recommendation": recommendation,
        "run_stats_sec": {str(K): run_stats[i]["stage4_sec"] for i, K in enumerate(K_VALUES)},
    }
    SWEEP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {SWEEP_OUT}")
    with open(AUDIT_OUT, "w", encoding="utf-8") as f:
        json.dump(audits, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {AUDIT_OUT}")


if __name__ == "__main__":
    main()
