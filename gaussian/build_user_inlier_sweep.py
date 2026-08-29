#!/usr/bin/env python3
"""inlier_frac threshold sweep (parallel).

用户指令 2026-08-29 (改): 串行 5 轮 ~50 min 太久, 改成 5 个 post-filter 串行 (~7.5 min)
+ 5 个 Stage 4 并行 (~10 min)。每次用 ASINS_OUT_SUFFIX + SEL_OUT_SUFFIX 写
threshold-specific 文件, 不互相覆盖。

输出: result/gaussian/inlier_frac_sweep.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_JSON = SCRATCH / "stage8_5_asins.json"
RESULTS_OUT = REPO_ROOT / "result" / "gaussian" / "inlier_frac_sweep.json"
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"


def log(msg: str) -> None:
    print(f"[inlier_sweep] {msg}", flush=True)


def run_post_filter(threshold: float) -> None:
    """build_user.py SWEEP_ONLY mode, 写 threshold-specific cohort 文件。"""
    env = os.environ.copy()
    env["SWEEP_ONLY"] = "1"
    env["MIN_INLIER_FRAC_OVERRIDE"] = str(threshold)
    env["ASINS_OUT_SUFFIX"] = f"_t{threshold:.1f}"
    log_path = LOG_DIR / f"build_user_sweep_inlier_{threshold:.1f}.log"
    with open(log_path, "w", encoding="utf-8") as fl:
        subprocess.run(
            [PYTHON, "gaussian/build_user.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fl,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_stage4(threshold: float) -> dict:
    """Stage 4 strict alignment with threshold-specific cohort + selection files.

    Returns dict with elapsed time + exit code.
    """
    env = os.environ.copy()
    env["ASINS_OUT_SUFFIX"] = f"_t{threshold:.1f}"
    env["SEL_OUT_SUFFIX"] = f"_t{threshold:.1f}"
    # 用户指令 2026-08-29: STAGE4_USER_FILTER=1 让 Stage 4 只加载 cohort 实际用到的
    # user Gaussian (而不是全 1.22M), peak memory 从 ~12GB → ~1-2GB, cgroup 32GB 内可跑 2 workers。
    env["STAGE4_USER_FILTER"] = "1"
    log_path = LOG_DIR / f"stage4_select_sweep_t{threshold:.1f}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fl:
        rc = subprocess.run(
            [PYTHON, "select_query/syntax_subspace_select_strict_alignment.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fl,
            stderr=subprocess.STDOUT,
        ).returncode
    return {"threshold": threshold, "stage4_sec": time.time() - t0, "returncode": rc}


def collect_metrics(threshold: float) -> dict:
    """读 threshold-specific cohort + selection 文件。"""
    cohort_path = SCRATCH / f"stage8_5_asins_t{threshold:.1f}.json"
    sel_path = SCRATCH / f"stage8_5_selection_t{threshold:.1f}.json"
    stats_path = SCRATCH / f"stage8_5_selection_stats_t{threshold:.1f}.json"

    cohort = json.load(open(cohort_path))
    asins = cohort.get("asins", [])
    n_asins = len(asins)
    n_quality_total = sum(e.get("n_users_quality", 0) for e in asins)
    n_quality_per_asin = [e.get("n_users_quality", 0) for e in asins]
    median_qpa = (
        sorted(n_quality_per_asin)[len(n_quality_per_asin) // 2]
        if n_quality_per_asin else 0
    )

    sel = json.load(open(sel_path))
    entries = sel.get("entries", [])
    n_strict = 0
    n_no_cand = 0
    d_self_list = []
    M_list = []
    for e in entries:
        method = e.get("selection_method", "")
        if method == "strict_personalized_mahal_margin_max":
            n_strict += 1
            d_self_list.append(e.get("d_self", 0.0))
            M_list.append(e.get("margin", 0.0))
        elif method == "no_strict_candidate":
            n_no_cand += 1

    import statistics as stats_mod
    return {
        "threshold": threshold,
        "n_quality_users": n_quality_total,
        "n_quality_per_asin_min": min(n_quality_per_asin) if n_quality_per_asin else 0,
        "n_quality_per_asin_median": median_qpa,
        "n_quality_per_asin_max": max(n_quality_per_asin) if n_quality_per_asin else 0,
        "n_asins": n_asins,
        "n_strict_selected": n_strict,
        "n_no_strict_candidate": n_no_cand,
        "d_self_mean": stats_mod.mean(d_self_list) if d_self_list else None,
        "d_self_median": stats_mod.median(d_self_list) if d_self_list else None,
        "d_self_max": max(d_self_list) if d_self_list else None,
        "M_mean": stats_mod.mean(M_list) if M_list else None,
        "M_median": stats_mod.median(M_list) if M_list else None,
        "M_positive_frac": (sum(1 for m in M_list if m > 0) / max(1, len(M_list))),
        "d_self_le_R95_frac": (
            sum(1 for d in d_self_list if d <= 8.073) / max(1, len(d_self_list))
        ),
    }


def main():
    log(f"=== inlier_frac threshold sweep (parallel) ===")
    log(f"  thresholds: {THRESHOLDS}")

    # Backup canonical cohort
    backup = SCRATCH / "stage8_5_asins.json.canonical"
    if not backup.exists():
        shutil.copy(ASINS_JSON, backup)
        log(f"  backed up canonical cohort → {backup}")
    else:
        log(f"  canonical backup exists: {backup}")

    t_total = time.time()

    # Phase 1: 5 post-filter (sequential, fast)
    log(f"\n=== Phase 1: 5 post-filters (sequential) ===")
    t0 = time.time()
    for th in THRESHOLDS:
        cohort_t = SCRATCH / f"stage8_5_asins_t{th:.1f}.json"
        if cohort_t.exists():
            log(f"  [post-filter] threshold={th} exists, skip")
            continue
        log(f"  [post-filter] threshold={th}")
        run_post_filter(th)
        # 立即 collect (fast, 只读小 cohort + selection.json)
    log(f"  5 post-filters done in {time.time() - t0:.1f}s")

    # Phase 2: 5 Stage 4 (serial)
    log(f"\n=== Phase 2: 5 Stage 4 strict alignments (serial, single worker) ===")
    t1 = time.time()
    # 用户指令 2026-08-29: cgroup 32GB 限制, Stage 4 load cache 12GB peak 即使有 filter
    # 也来不及 (json.load 全加载后才过滤), 并行 2 workers 必 OOM。改成 serial, 5 轮 ~45 min。
    stage4_results = []
    for th in THRESHOLDS:
        log(f"  [stage4 t={th}] running...")
        r = run_stage4(th)
        stage4_results.append(r)
        log(f"  [stage4 t={th}] done in {r['stage4_sec']:.1f}s rc={r['returncode']}")
    log(f"  5 Stage 4 done in {time.time() - t1:.1f}s")

    # Phase 3: collect metrics
    log(f"\n=== Phase 3: collect metrics ===")
    results = []
    for th in THRESHOLDS:
        m = collect_metrics(th)
        results.append(m)
        d_self_str = f"{m['d_self_mean']:.2f}" if m['d_self_mean'] is not None else "N/A"
        m_str = f"{m['M_mean']:.2f}" if m['M_mean'] is not None else "N/A"
        log(f"  t={th}: quality={m['n_quality_users']:>7}, "
            f"asins={m['n_asins']:>5}, selected={m['n_strict_selected']:>5}, "
            f"d_self={d_self_str}, M={m_str}, M>0={m['M_positive_frac']:.2f}")

    # Restore canonical cohort (in case anything wrote to it)
    log(f"\n=== restoring canonical cohort ===")
    shutil.copy(backup, ASINS_JSON)
    log(f"  restored → {ASINS_JSON}")

    # Save results
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": (
            f"inlier_frac threshold sweep across {THRESHOLDS}; 5 post-filters "
            f"sequential + 5 Stage 4 parallel. Per-threshold: n_quality_users, "
            f"n_asins, n_strict_selected, d_self, M, M>0 frac."
        ),
        "thresholds": THRESHOLDS,
        "R_95": 8.073,
        "results": results,
        "stage4_timings": sorted(stage4_results, key=lambda x: x["threshold"]),
        "elapsed_total_sec": time.time() - t_total,
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n=== sweep results → {RESULTS_OUT} ===")
    log(f"=== DONE in {time.time() - t_total:.1f}s ===")


if __name__ == "__main__":
    main()
