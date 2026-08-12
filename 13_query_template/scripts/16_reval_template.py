#!/usr/bin/env python3
"""Re-run 14-stage eval (template + template_noisy) for all 3 cats.

Triggered by: ColBERTv2 maxlen=96 cache regeneration. The other 7 retrievers'
results are unchanged, but we re-run all 8 to refresh the summary files
(`retrieval_template_summary.json` and `retrieval_template_noisy_summary.json`)
in one shot.

Usage (from project root):
  python3 PersoanlQuery/13_query_template/scripts/16_reval_template.py
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
MODES = ["clean", "noisy"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def eval_script_path(category: str, mode: str) -> Path:
    suffix = "_noisy" if mode == "noisy" else ""
    return SCRIPT_DIR / f"14_eval_template{suffix}_{category}.py"


def run_eval(category: str, mode: str) -> int:
    script = eval_script_path(category, mode)
    if not script.exists():
        raise FileNotFoundError(f"script not found: {script}")
    log(f"--- {category} [{mode}] ---")
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"eval failed for {category} [{mode}] (exit={result.returncode}, elapsed={elapsed:.1f}s)"
        )
    log(f"  ✓ {category} [{mode}] done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return int(elapsed)


def main() -> None:
    log("=" * 80)
    log("14-stage eval re-run (3 cats × 2 modes, 8 retrievers each)")
    log("=" * 80)

    start = time.time()
    elapsed_per_job: list[tuple[str, str, int]] = []

    for mode in MODES:
        log("")
        log(f">>> Stage: {mode}")
        for cat in CATEGORIES:
            elapsed = run_eval(cat, mode)
            elapsed_per_job.append((cat, mode, elapsed))

    total_elapsed = time.time() - start
    log("")
    log("=" * 80)
    log("✅ All 6 eval jobs done")
    log("=" * 80)
    log(f"  total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log("  per-job timing:")
    for cat, mode, elapsed in elapsed_per_job:
        log(f"    {cat:<28} {mode:<6} {elapsed:>6.1f}s ({elapsed/60:.1f} min)")
    log("")
    log("Next step: re-run 14_analyze_template_vs_llm.py to refresh Table 5.")


if __name__ == "__main__":
    main()
