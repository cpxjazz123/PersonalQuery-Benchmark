#!/usr/bin/env python3
"""Generate noisy query cache for 7 non-ColBERTv2 retrievers (3 cats).

Why: 14-stage colbertv2 noisy cache was generated separately (maxlen=96 fix).
The other 7 retrievers' noisy cache is missing entirely, blocking the 2×2
analysis (clean×noisy × template×LLM).

This script regenerates the noisy cache for BGE / E5 / MiniLM / STAR / ANCE /
SPLADE / BM25 across all 3 categories. ColBERTv2 is skipped because its
noisy cache is already at maxlen=96.

Usage (from project root):
  python3 PersoanlQuery/13_query_template/scripts/17_regen_noisy_cache_7retrievers.py
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
RETRIEVERS_7 = ["bge", "e5", "minilm", "star", "ance", "splade", "bm25"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_one(category: str, retrievers: list[str]) -> int:
    script = SCRIPT_DIR / f"14_generate_query_cache_noisy_{category}.py"
    if not script.exists():
        raise FileNotFoundError(f"script not found: {script}")
    log(f"--- {category} (noisy, 7 retrievers) ---")
    start = time.time()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--retrievers", *retrievers,
        ],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"cache generation failed for {category} (exit={result.returncode}, elapsed={elapsed:.1f}s)")
    log(f"  ✓ {category} done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return int(elapsed)


def main() -> None:
    log("=" * 80)
    log("Noisy cache regeneration for 7 non-ColBERTv2 retrievers (3 cats)")
    log(f"  Retrievers: {RETRIEVERS_7}")
    log("=" * 80)

    start = time.time()
    elapsed_per_cat: list[tuple[str, int]] = []
    for cat in CATEGORIES:
        elapsed = run_one(cat, RETRIEVERS_7)
        elapsed_per_cat.append((cat, elapsed))

    total_elapsed = time.time() - start
    log("")
    log("=" * 80)
    log("✅ All 7-retriever noisy caches generated")
    log("=" * 80)
    log(f"  total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log("  per-cat timing:")
    for cat, elapsed in elapsed_per_cat:
        log(f"    {cat:<28} {elapsed:>6.1f}s ({elapsed/60:.1f} min)")
    log("")
    log("Next step: re-run scripts/16_reval_template.py to refresh the 14-stage")
    log("noisy summaries, then re-run 14_analyze_template_vs_llm.py for Table 6.")


if __name__ == "__main__":
    main()
