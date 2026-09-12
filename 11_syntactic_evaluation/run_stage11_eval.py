#!/usr/bin/env python3
"""Stage 11 — 7-Retriever evaluation driver.

Loads selected_queries.json (Stage 4 output), runs all 7 retrievers
(BM25 / SPLADE / MiniLM / MPNet / BGE / GTE / ColBERTv2) via the unified
module, persists:

  - result/11_syntactic_evaluation/per_query.json          (per-query retr × rank × hit@k)
  - result/11_syntactic_evaluation/retrieval_summary.json  (volatility headline)
  - result/11_syntactic_evaluation/volatility.json         (BM25 + MiniLM canonical)

Stage 12 imports the same unified module and reuses the multi-retrieval
embed cache + asin_to_doc cache via shared EMBED_CACHE_DIR / ASIN_TO_DOC_CACHE
paths. SMOKE/N_SMOKE flags are read from unified (already hardcoded).

Usage (Rule 3, no args):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 11_syntactic_evaluation/run_stage11_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "11_syntactic_evaluation"))

import syntax_subspace_retrieval_unified as unified


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=== Stage 11 driver: invoking unified.main() ===")
    log(f"  SEL_IN       = {unified.SEL_IN}")
    log(f"  PER_QUERY_OUT = {unified.PER_QUERY_OUT}")
    log(f"  SUMMARY_OUT  = {unified.SUMMARY_OUT}")
    log(f"  VOLATILITY_OUT = {unified.VOLATILITY_OUT}")
    log(f"  SMOKE={unified.SMOKE}  RETR_NAMES={unified.RETR_NAMES}")
    unified.main()
    # Verify outputs landed
    for label, path in [("per_query", unified.PER_QUERY_OUT),
                        ("summary",   unified.SUMMARY_OUT),
                        ("volatility", unified.VOLATILITY_OUT)]:
        if not path.exists():
            raise RuntimeError(f"unified.main() did not produce {label}: {path}")
        log(f"  ✓ {label}: {path} ({path.stat().st_size/1e6:.2f} MB)")
    log("=== Stage 11 driver done ===")


if __name__ == "__main__":
    main()