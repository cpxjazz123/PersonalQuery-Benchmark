#!/usr/bin/env python3
"""Transfer Stage 04 syntax_depth query file to Stage 10 expression_style expected location.

iter #195 finding: Stage 10 train_vades_lite_sentence_latent_threshold.py expects each record to have
`expression_style_query` (selected) and `expression_style_queries` (candidate list), but Stage 04
`syntax_depth_no_depth_check.py` still uses the old field names `syntax_depth_query` and
`syntax_depth_queries` (iter #169 rename was incomplete on the producer side).

This script:
  1. Reads Stage 04 output (04_query/<cat>/query_by_syntax_depth_no_depth_check_10.json).
  2. Validates it is a non-empty list with required schema (user_id, asin, syntax_depth_query).
  3. Renames syntax_depth_query -> expression_style_query, syntax_depth_queries -> expression_style_queries.
  4. Writes to Stage 10 expected location (06_query/<cat>/query_by_expression_style_no_depth_check_10.json).
  5. Raises on any missing/invalid input (CLAUDE.md Rule 7: no fallback).

Usage:
    python3 06_transfer_stage04_to_stage10.py --category Baby_Products
    python3 06_transfer_stage04_to_stage10.py --category Grocery_and_Gourmet_Food
    python3 06_transfer_stage04_to_stage10.py --category Pet_Supplies
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
STAGE_04_DIR = REPO_ROOT / "result" / "personal_query" / "04_query"
STAGE_10_INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "06_query"

# Stage 04 producer actual filename (syntax_depth_no_depth_check.py line 42)
STAGE_04_FILENAME = "query_by_syntax_depth_no_depth_check_10.json"

# Stage 10 expected filename (train_vades_lite_sentence_latent_threshold.py line 35)
STAGE_10_INPUT_FILENAME = "query_by_expression_style_no_depth_check_10.json"

# Required schema: top-level record fields. Old name -> New name.
FIELD_RENAMES = {
    "syntax_depth_query": "expression_style_query",
    "syntax_depth_queries": "expression_style_queries",
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def transfer(category: str) -> Path:
    """Copy Stage 04 query file to Stage 10 expected location. Raises on any failure."""
    src = STAGE_04_DIR / category / STAGE_04_FILENAME
    if not src.exists():
        raise FileNotFoundError(
            f"Stage 04 output missing: {src}. "
            f"Run Stage 04 (04_query/04_generate_by_syntax_depth_no_depth_check_10_<category>.py) first."
        )

    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(
            f"Stage 04 output {src} must be a non-empty list "
            f"(got type={type(raw).__name__}, len={len(raw) if hasattr(raw, '__len__') else 'N/A'})"
        )

    # Schema check + rename per record.
    renamed_rows: list[dict] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Stage 04 record #{idx} must be a dict, got {type(row).__name__}")

        # Required: user_id, asin, syntax_depth_query (selected one)
        if "user_id" not in row:
            raise ValueError(f"Stage 04 record #{idx} missing user_id")
        if "asin" not in row:
            raise ValueError(f"Stage 04 record #{idx} missing asin")
        if "syntax_depth_query" not in row:
            raise ValueError(
                f"Stage 04 record #{idx} missing syntax_depth_query. "
                f"Available keys: {sorted(row.keys())}"
            )

        # Build new dict with renamed keys
        new_row: dict = {}
        for k, v in row.items():
            new_key = FIELD_RENAMES.get(k, k)
            new_row[new_key] = v
        renamed_rows.append(new_row)

    dst = STAGE_10_INPUT_DIR / category / STAGE_10_INPUT_FILENAME
    dst.parent.mkdir(parents=True, exist_ok=True)

    log(f"Transfer Stage 04 -> Stage 10 for {category}")
    log(f"  src: {src} ({len(raw)} records)")
    log(f"  dst: {dst}")

    dst.write_text(
        json.dumps(renamed_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(f"  wrote {dst.stat().st_size} bytes, {len(renamed_rows)} records")
    log(f"  renamed fields: {list(FIELD_RENAMES.keys())} -> {list(FIELD_RENAMES.values())}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer Stage 04 query file to Stage 10 input location with field rename"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="Domain category",
    )
    args = parser.parse_args()
    transfer(args.category)


if __name__ == "__main__":
    main()
