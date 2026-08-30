#!/usr/bin/env python3
"""Generate strict cohort from T=34 calibrated users + re-run Stage 4 strict alignment + Stage 5 retrieval.

用户指令 2026-08-30: 用 T=34 strict users (51,352 users) 重新做 Stage 4 + Stage 5 跑 benchmark,
  用 ASINS_OUT_SUFFIX=_strict34 + SEL_OUT_SUFFIX=_strict34 写到独立文件, 不覆盖 canonical.

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json (canonical cohort, 39,784 ASINs)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_quality_strict_users.json (T=34 user_ids, 51,352)
  - select_query/syntax_subspace_select_strict_alignment.py (Stage 4)
  - syntactic_evaluation/syntax_subspace_retrieval.py (Stage 5)

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins_strict34.json (filtered cohort)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_strict34.json (Stage 4)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_stats_strict34.json (Stage 4 stats)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query_strict34.json (Stage 5 per-query)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_volatility_strict34.json (Stage 5 volatility)
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/strict34_benchmark.json (聚合 summary)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")

STRICT_USERS = SCRATCH / "stage8_5_quality_strict_users.json"
ASINS_CANON = SCRATCH / "stage8_5_asins.json"
ASINS_STRICT = SCRATCH / "stage8_5_asins_strict34.json"
SUMMARY_OUT = REPO_ROOT / "result" / "gaussian" / "strict34_benchmark.json"

SUFFIX = "_strict34"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

MIN_USERS_PER_ASIN = 5  # filter out ASINs with <5 strict users


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [strict34] {msg}", flush=True)


def build_cohort() -> dict:
    """Filter canonical cohort by T=34 strict users, write stage8_5_asins_strict34.json."""
    log(f"=== Step 1: build strict34 cohort ===")
    if not STRICT_USERS.exists():
        raise FileNotFoundError(f"strict users cache required: {STRICT_USERS}")
    if not ASINS_CANON.exists():
        raise FileNotFoundError(f"canonical cohort required: {ASINS_CANON}")

    with open(STRICT_USERS, "r", encoding="utf-8") as f:
        sd = json.load(f)
    strict_uid_set = set(sd["users"])
    n_strict = len(strict_uid_set)
    log(f"  strict users: {n_strict} (threshold cv_nll ≤ {sd['threshold_cv_nll']}, "
        f"val_precision={sd['validation_precision']:.4f})")

    with open(ASINS_CANON, "r", encoding="utf-8") as f:
        canon = json.load(f)
    asins_canon = canon.get("asins", [])
    log(f"  canonical cohort: {len(asins_canon)} ASINs")

    n_total_users_pre = 0
    n_total_users_post = 0
    asins_strict = []
    for entry in asins_canon:
        a = entry["asin"]
        us_str = entry.get("users_sampled", "")
        try:
            uid_list = ast.literal_eval(us_str) if isinstance(us_str, str) else us_str
        except (ValueError, SyntaxError):
            uid_list = []
        n_total_users_pre += len(uid_list)
        # Filter to strict users
        uid_strict = [u for u in uid_list if u in strict_uid_set]
        n_strict_in_asin = len(uid_strict)
        n_total_users_post += n_strict_in_asin
        new_entry = dict(entry)
        new_entry["users_sampled"] = uid_strict
        new_entry["n_users_quality"] = n_strict_in_asin
        new_entry["users_raw_count"] = n_strict_in_asin
        new_entry["filter"] = (
            f"strict34: cv_nll <= {sd['threshold_cv_nll']} (val_precision="
            f"{sd['validation_precision']:.4f}, n_strict_users={n_strict})"
        )
        asins_strict.append(new_entry)

    log(f"  pre-filter: {n_total_users_pre} (user, asin) pairs")
    log(f"  post-filter: {n_total_users_post} (user, asin) pairs "
        f"({100*n_total_users_post/max(1,n_total_users_pre):.1f}% retention)")

    # ASIN coverage summary
    n_ge1 = sum(1 for e in asins_strict if e["n_users_quality"] >= 1)
    n_ge2 = sum(1 for e in asins_strict if e["n_users_quality"] >= 2)
    n_ge5 = sum(1 for e in asins_strict if e["n_users_quality"] >= MIN_USERS_PER_ASIN)
    n_ge10 = sum(1 for e in asins_strict if e["n_users_quality"] >= 10)
    log(f"  ASIN coverage: ≥1={n_ge1}, ≥2={n_ge2}, ≥{MIN_USERS_PER_ASIN}={n_ge5}, ≥10={n_ge10}")

    # Write cohort
    out_doc = {
        "config": {
            "filter": "strict34",
            "cv_nll_threshold": sd["threshold_cv_nll"],
            "validation_precision": sd["validation_precision"],
            "n_strict_users": n_strict,
            "min_users_per_asin": MIN_USERS_PER_ASIN,
        },
        "n_asins": len(asins_strict),
        "n_users_total": n_total_users_post,
        "n_asins_ge1": n_ge1,
        "n_asins_ge2": n_ge2,
        f"n_asins_ge{MIN_USERS_PER_ASIN}": n_ge5,
        "n_asins_ge10": n_ge10,
        "asins": asins_strict,
    }
    with open(ASINS_STRICT, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {ASINS_STRICT}")
    return {
        "n_strict_users": n_strict,
        "cv_nll_threshold": sd["threshold_cv_nll"],
        "val_precision": sd["validation_precision"],
        "n_asins_input": len(asins_strict),
        "n_asins_ge1": n_ge1,
        "n_asins_ge2": n_ge2,
        f"n_asins_ge{MIN_USERS_PER_ASIN}": n_ge5,
        "n_asins_ge10": n_ge10,
        "n_users_total_post": n_total_users_post,
    }


def run_stage4() -> dict:
    """Run Stage 4 strict alignment with strict34 cohort."""
    log(f"\n=== Step 2: Stage 4 strict alignment (suffix={SUFFIX}) ===")
    env = os.environ.copy()
    env["ASINS_OUT_SUFFIX"] = SUFFIX
    env["SEL_OUT_SUFFIX"] = SUFFIX
    env["STAGE4_USER_FILTER"] = "1"  # reduce memory by filtering to cohort users
    log_path = LOG_DIR / f"stage4_select_strict34.log"
    log(f"  log → {log_path}")
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
        raise RuntimeError(f"Stage 4 failed with rc={rc}, see {log_path}")
    return {"stage4_sec": elapsed, "log": str(log_path)}


def run_stage5() -> dict:
    """Run Stage 5 retrieval + volatility with strict34 selection."""
    log(f"\n=== Step 3: Stage 5 retrieval (suffix={SUFFIX}) ===")
    env = os.environ.copy()
    env["SEL_OUT_SUFFIX"] = SUFFIX
    log_path = LOG_DIR / f"stage5_retrieval_strict34.log"
    log(f"  log → {log_path}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fl:
        rc = subprocess.run(
            [PYTHON, "syntactic_evaluation/syntax_subspace_retrieval_unified.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fl,
            stderr=subprocess.STDOUT,
        ).returncode
    elapsed = time.time() - t0
    log(f"  Stage 5 done in {elapsed:.1f}s, rc={rc}")
    if rc != 0:
        raise RuntimeError(f"Stage 5 failed with rc={rc}, see {log_path}")
    return {"stage5_sec": elapsed, "log": str(log_path)}


def collect_results(cohort_stats: dict, stage4_stats: dict, stage5_stats: dict) -> None:
    """Aggregate Stage 4 + Stage 5 metrics into summary JSON."""
    log(f"\n=== Step 4: collect results ===")
    sel_path = SCRATCH / f"stage8_5_selection{SUFFIX}.json"
    sel_stats_path = SCRATCH / f"stage8_5_selection_stats{SUFFIX}.json"
    retrieval_path = SCRATCH / f"stage8_5_retrieval_per_query{SUFFIX}.json"
    volatility_path = SCRATCH / f"stage8_5_volatility{SUFFIX}.json"

    if not sel_stats_path.exists():
        log(f"  ⚠ no selection_stats at {sel_stats_path}, skipping aggregation")
        return
    with open(sel_stats_path, "r", encoding="utf-8") as f:
        sel_stats = json.load(f)

    retrieval_summary = {}
    if retrieval_path.exists():
        with open(retrieval_path, "r", encoding="utf-8") as f:
            retrieval_summary = json.load(f)

    volatility_summary = {}
    if volatility_path.exists():
        with open(volatility_path, "r", encoding="utf-8") as f:
            volatility_summary = json.load(f)

    summary = {
        "description": (
            f"Strict34 benchmark: T=34 cohort (cv_nll <= 34, val precision 98.21%). "
            f"Stage 4 strict alignment + Stage 5 BM25 + MiniLM retrieval + V_low/V_user/V_high "
            f"volatility calibration."
        ),
        "cohort": cohort_stats,
        "stage4_metrics": sel_stats,
        "stage5_retrieval": retrieval_summary,
        "stage5_volatility": volatility_summary,
        "elapsed_sec": {
            "stage4": stage4_stats["stage4_sec"],
            "stage5": stage5_stats["stage5_sec"],
        },
        "output_files": {
            "cohort": str(ASINS_STRICT),
            "selection": str(sel_path),
            "selection_stats": str(sel_stats_path),
            "retrieval_per_query": str(retrieval_path),
            "volatility": str(volatility_path),
        },
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")
    log(f"  cohort: {cohort_stats['n_asins_input']} ASINs, "
        f"{cohort_stats[f'n_asins_ge{MIN_USERS_PER_ASIN}']} with ≥{MIN_USERS_PER_ASIN} users")


def main():
    log(f"=== Strict34 benchmark: cohort → Stage 4 → Stage 5 ===")
    t_total = time.time()

    # Step 1: build cohort
    cohort_stats = build_cohort()

    # Step 2: Stage 4
    stage4_stats = run_stage4()

    # Step 3: Stage 5
    stage5_stats = run_stage5()

    # Step 4: aggregate
    collect_results(cohort_stats, stage4_stats, stage5_stats)

    log(f"\n=== DONE in {time.time() - t_total:.1f}s ===")


if __name__ == "__main__":
    main()
