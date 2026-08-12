#!/usr/bin/env python3
"""Re-run 14-stage eval (noisy only) for all 3 cats.

Why: clean caches are unchanged, but we just regenerated the 7-retriever
noisy caches. Re-run only the noisy mode (3 cats) to refresh the noisy
summaries in `retrieval_template_summary_noisy.json`.

Usage (from project root):
  python3 PersoanlQuery/13_query_template/scripts/18_reval_template_noisy_only.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/fs04/ar57/wenyu")
SCRIPT_DIR = PROJECT_ROOT / "PersoanlQuery" / "13_query_template"

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_eval(category: str) -> int:
    script = SCRIPT_DIR / f"14_eval_template_noisy_{category}.py"
    if not script.exists():
        raise FileNotFoundError(f"script not found: {script}")
    log(f"--- {category} [noisy] ---")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"eval failed for {category} [noisy] (exit={result.returncode}, elapsed={elapsed:.1f}s)"
        )
    log(f"  ✓ {category} [noisy] done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return int(elapsed)


def main() -> None:
    log("=" * 80)
    log("14-stage noisy eval re-run (3 cats, 8 retrievers each)")
    log("=" * 80)

    start = time.time()
    elapsed_per_job: list[tuple[str, int]] = []

    for cat in CATEGORIES:
        elapsed = run_eval(cat)
        elapsed_per_job.append((cat, elapsed))

    total_elapsed = time.time() - start
    log("")
    log("=" * 80)
    log("✅ All 3 noisy eval jobs done")
    log("=" * 80)
    log(f"  total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log("  per-job timing:")
    for cat, elapsed in elapsed_per_job:
        log(f"    {cat:<28} {elapsed:>6.1f}s ({elapsed/60:.1f} min)")
    log("")
    log("Next step: re-run 14_analyze_template_vs_llm.py to refresh Table 6 with full 2×2 data.")


if __name__ == "__main__":
    main()
