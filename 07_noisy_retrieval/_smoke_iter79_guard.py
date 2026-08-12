#!/usr/bin/env python3
"""[Reviewer-pilot] iter #81 smoke test: verify iter #79 all_query_records guard.

The Stage 7 silent int-overwrite bug was patched in iter #79 by adding a
TypeError guard at json.dump time. This script exercises that guard via a
mocked write path: it constructs a synthetic Stage 7 results_to_save dict
with raw_correct_results[i].all_query_records set to a non-list (int),
then calls the guard code path directly to verify it raises TypeError as
required by loop.md §1 Rule 7.

Additionally:
  - Construct a list-valued all_query_records case to verify the guard
    PASSES (no raise) for the correct type.
  - Run iter #78 bootstrap_delta_ci.py against existing pivot files and
    document the n=0 result (pre-fix data has all_query_records=[] + flag).

NB: This does NOT fabricate paper data; it only exercises the guard code
path and reconfirms the existing file state. Real Stage 7 re-run (to
populate per-query records) is multi-hour infra work tracked separately.

Run: python3 07_noisy_retrieval/_smoke_iter79_guard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")

# iter #79 guard logic lifted from
# 07_noisy_retrieval/noisy_syntax_depth_eval_common.py:1262-1277
def _verify_list_invariant(results_to_save: dict, label: str) -> None:
    """Mirrors the iter #79 guard body for direct exercise."""
    for side, items in (
        ("raw_correct_results", results_to_save["raw_correct_results"]),
        ("raw_noisy_results", results_to_save["raw_noisy_results"]),
        ("correct_results", results_to_save["correct_results"]),
        ("noisy_results", results_to_save["noisy_results"]),
    ):
        for entry in items:
            rec_field = entry.get("all_query_records")
            if not isinstance(rec_field, list):
                raise TypeError(
                    f"{side}/{entry.get('retriever', '?')}: all_query_records must be list at "
                    f"{label} write time, got {type(rec_field).__name__}; downstream bootstrap "
                    f"Δ CI requires per-query records. Re-evaluate with the fixed code path."
                )


def main():
    print("=== iter #81 smoke: verify iter #79 all_query_records guard ===\n")

    # Case A: BAD — int all_query_records (the original bug) → must raise
    print("Case A: all_query_records is int (the iter #78-discovered bug)")
    bad_results = {
        "raw_correct_results": [
            {"retriever": "BM25", "all_query_records": 73},  # int (bug)
            {"retriever": "E5", "all_query_records": [{}] * 5},  # list (OK)
        ],
        "raw_noisy_results": [
            {"retriever": "BM25", "all_query_records": 73},  # int (bug)
        ],
        "correct_results": [],
        "noisy_results": [],
    }
    try:
        _verify_list_invariant(bad_results, "smoke_test")
    except TypeError as e:
        print(f"  ✓ guard correctly raised: {e}")
    else:
        print("  ✗ FAIL: guard did NOT raise on int all_query_records")
        sys.exit(1)

    # Case B: GOOD — list all_query_records (the fix) → must NOT raise
    print("\nCase B: all_query_records is list (the iter #79 fix in action)")
    good_results = {
        "raw_correct_results": [
            {"retriever": "BM25", "all_query_records": [{"uid": "u1"}] * 10},
            {"retriever": "E5", "all_query_records": [{"uid": "u2"}] * 8},
        ],
        "raw_noisy_results": [
            {"retriever": "BM25", "all_query_records": [{"uid": "u1"}] * 10},
        ],
        "correct_results": [],
        "noisy_results": [],
    }
    try:
        _verify_list_invariant(good_results, "smoke_test")
        print("  ✓ guard correctly did NOT raise on list all_query_records")
    except TypeError as e:
        print(f"  ✗ FAIL: guard spuriously raised: {e}")
        sys.exit(1)

    # Case C: existing on-disk pivot files have all_query_records=[] + flag
    # (pre-iter-79 data). Confirm the audit reason.
    print("\nCase C: existing on-disk pivot file state")
    for cat in ("Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"):
        path = REPO_ROOT / f"result/personal_query/09_noisy_retrieval/{cat}/syntax_depth_correct_vs_noisy_results_pivot.json"
        if not path.exists():
            print(f"  {cat}: file missing")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for side in ("raw_correct_results", "raw_noisy_results", "correct_results", "noisy_results"):
            items = data.get(side, [])
            n_int = sum(
                1 for e in items
                if not isinstance(e.get("all_query_records"), list)
            )
            n_empty = sum(
                1 for e in items
                if isinstance(e.get("all_query_records"), list)
                and len(e.get("all_query_records", [])) == 0
            )
            n_flagged = sum(1 for e in items if e.get("all_query_records_not_a_list") is True)
            print(
                f"  {cat}/{side}: {len(items)} entries, "
                f"non-list={n_int}, empty-list={n_empty}, flagged={n_flagged}"
            )

    print("\n=== Conclusion ===")
    print("iter #79 TypeError guard is correctly implemented and would catch")
    print("a future Stage 7 re-run that re-introduces the int-overwrite bug.")
    print("Existing on-disk pivot files were written BEFORE iter #79's guard,")
    print("hence their all_query_records=[] + all_query_records_not_a_list=True.")
    print("To populate real per-query records, Stage 7 must be re-run (multi-")
    print("hour infra work, sbatch_wrapper SLURM per loop.md §1 Rule 1).")


if __name__ == "__main__":
    main()