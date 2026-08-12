#!/usr/bin/env python3
"""Regenerate ColBERTv2 caches for all 3 cats × 2 modes at maxlen=96.

Why this script:
  14-stage template queries average 35.7 subword tokens (median 35, max 139).
  ColBERTv2's default `query_maxlen=32` truncates 69% of these queries
  mid-product-name tokens, costing +11.42 pp H@10 on Baby_Products
  (43.90% → 55.32%). Raising `query_maxlen=96` covers 100% of queries.

What it does:
  For each (category, mode) pair in (Baby_Products, Grocery_and_Gourmet_Food,
  Pet_Supplies) × (clean, noisy), invokes the existing colbertv2-only cache
  generator entry point. The other 7 retrievers' caches are unaffected and
  preserved.

Usage (from project root):
  python3 PersoanlQuery/13_query_template/scripts/15_regen_colbertv2_maxlen96.py
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


def cache_script_path(category: str, mode: str) -> Path:
    suffix = "_noisy" if mode == "noisy" else ""
    return SCRIPT_DIR / f"14_generate_query_cache{suffix}_{category}.py"


def run_cache_generation(category: str, mode: str, query_maxlen: int) -> int:
    script = cache_script_path(category, mode)
    if not script.exists():
        raise FileNotFoundError(f"script not found: {script}")
    log(f"--- {category} [{mode}] (query_maxlen={query_maxlen}) ---")
    start = time.time()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--retrievers", "colbertv2",
            "--colbertv2-query-maxlen", str(query_maxlen),
        ],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"cache generation failed for {category} [{mode}] "
            f"(exit={result.returncode}, elapsed={elapsed:.1f}s)"
        )
    log(f"  ✓ {category} [{mode}] done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return int(elapsed)


def main(query_maxlen: int = 96) -> None:
    log("=" * 80)
    log(f"ColBERTv2 cache regeneration (3 cats × 2 modes) at query_maxlen={query_maxlen}")
    log("=" * 80)

    start = time.time()
    elapsed_per_job: list[tuple[str, str, int]] = []
    failed: list[tuple[str, str]] = []

    for mode in MODES:
        log("")
        log(f">>> Stage: {mode}")
        for cat in CATEGORIES:
            try:
                elapsed = run_cache_generation(cat, mode, query_maxlen)
                elapsed_per_job.append((cat, mode, elapsed))
            except Exception as exc:
                log(f"  [ERROR] {cat} [{mode}] failed: {exc}")
                failed.append((cat, mode))
                raise

    total_elapsed = time.time() - start
    log("")
    log("=" * 80)
    log("✅ All 6 ColBERTv2 caches regenerated")
    log("=" * 80)
    log(f"  query_maxlen: {query_maxlen}")
    log(f"  total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log("  per-job timing:")
    for cat, mode, elapsed in elapsed_per_job:
        log(f"    {cat:<28} {mode:<6} {elapsed:>6.1f}s ({elapsed/60:.1f} min)")
    if failed:
        log(f"  ⚠️  failed: {failed}")
        sys.exit(1)
    log("")
    log("Next step: run the matching eval pipeline (3 cats × 2 modes) to")
    log("produce the new retrieval_template_summary.json and")
    log("retrieval_template_noisy_summary.json, then re-run")
    log("14_analyze_template_vs_llm.py to update Table 5.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-maxlen",
        type=int,
        default=96,
        help="ColBERTv2 query_maxlen (default 96, set 32 to revert)",
    )
    args = parser.parse_args()
    main(query_maxlen=args.query_maxlen)
