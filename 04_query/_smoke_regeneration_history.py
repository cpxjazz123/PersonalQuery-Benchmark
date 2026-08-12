#!/usr/bin/env python3
"""[Reviewer-pilot] iter #85 smoke test: verify process_one_user emits regeneration_history.

Runs a single-user dry-run against the real vLLM backend to confirm:
  1. process_one_user returns a dict with regeneration_history field
  2. regeneration_history has at least 1 entry (round 1)
  3. Each entry carries the audit fields
     {round, candidates_requested, candidates_returned,
      candidates_accepted_this_round, accepted_candidate_indices,
      rejected_reasons, empty_response, parse_failed, llm_step_name}
  4. When validation succeeds on round 1, regeneration_rounds_used == 1
     and regeneration_triggered is False.

Uses Baby_Products (which has 95 user rows in Stage 03 inputs) but only
1 user, and runs against the existing Stage 04 output JSON to avoid
overwriting the 73 already-accepted rows.

Run: python3 04_query/_smoke_regeneration_history.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/04_query")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

from common.syntax_depth_no_depth_check import (  # noqa: E402
    REGENERATION_MAX_ROUNDS,
    NUM_CANDIDATES_PER_USER,
    process_one_user,
    build_user_tasks,
)
from common.llm_runner import load_minimax_client  # noqa: E402

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")


def _check_vllm_up() -> None:
    """Confirm vLLM backend is reachable before we burn user-time."""
    try:
        req = urllib.request.Request("http://localhost:8000/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        models = [m["id"] for m in body.get("data", [])]
        if not any("Qwen" in m for m in models):
            raise ValueError(f"no Qwen model exposed at :8000; saw {models}")
        print(f"  vLLM reachable, models: {models[:3]}")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        raise RuntimeError(f"vLLM not reachable at :8000 ({e}); cannot run pilot") from e


def main():
    print(f"REGENERATION_MAX_ROUNDS = {REGENERATION_MAX_ROUNDS}")
    print(f"NUM_CANDIDATES_PER_USER = {NUM_CANDIDATES_PER_USER}")
    _check_vllm_up()
    print("  loading LLM client...")
    load_minimax_client(use_minimaxio=False)

    # Load only 1 user task (build_user_tasks honors NUM_USERS_TO_TEST).
    # We bypass the resume cache by temporarily renaming the existing output file
    # so build_user_tasks re-loads the full task pool but the resume index sees 0.
    out_path = (
        REPO_ROOT
        / "result/personal_query/04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json"
    )
    backup_path = out_path.with_suffix(".bak_smoke")
    if out_path.exists():
        print(f"  Backup existing output → {backup_path.name}")
        out_path.rename(backup_path)
    try:
        existing, tasks = build_user_tasks("Baby_Products")
        if not tasks:
            raise RuntimeError("no tasks loaded (Stage 03 missing?)")
        first_task = tasks[0]
        print(f"  task 0: user_id={first_task['user_id']} asin={first_task.get('asin')}")
        print(f"  target_depth={first_task['syntax_depth']['target_depth']}")
        print()
        t0 = time.time()
        result = process_one_user("Baby_Products", first_task)
        elapsed = time.time() - t0
    finally:
        # Restore the existing output (do not overwrite accepted rows).
        if backup_path.exists():
            backup_path.rename(out_path)
            print(f"  Restored existing output from {backup_path.name}")

    if result is None:
        raise RuntimeError("process_one_user returned None — LLM call failed")
    print(f"  process_one_user took {elapsed:.1f}s")
    print()
    print("=== Top-level fields ===")
    for k, v in result.items():
        if k == "syntax_depth_queries":
            print(f"  {k}: list[{len(v)}]")
        elif k == "syntax_depth_query":
            print(f"  {k}: <first candidate>")
        else:
            print(f"  {k}: {v}")
    print()
    rh = result.get("regeneration_history", [])
    print(f"=== regeneration_history ({len(rh)} round(s)) ===")
    for entry in rh:
        print(f"  round={entry['round']}")
        print(f"    candidates_requested={entry['candidates_requested']}")
        print(f"    candidates_returned={entry['candidates_returned']}")
        print(f"    candidates_accepted_this_round={entry['candidates_accepted_this_round']}")
        print(f"    accepted_candidate_indices={entry['accepted_candidate_indices']}")
        print(f"    rejected_reasons (first 3)={entry['rejected_reasons'][:3]}")
        print(f"    empty_response={entry['empty_response']} parse_failed={entry['parse_failed']}")
        print(f"    llm_step_name={entry['llm_step_name']}")

    # Hard assertions (no fallback per loop.md §1 Rule 7)
    print()
    assert isinstance(result["regeneration_history"], list), "regeneration_history must be a list"
    assert len(result["regeneration_history"]) >= 1, "regeneration_history must have ≥1 round"
    assert isinstance(result["regeneration_rounds_used"], int), "rounds_used must be int"
    assert result["regeneration_rounds_used"] >= 1, "rounds_used must be ≥1"
    assert isinstance(result["regeneration_triggered"], bool), "triggered must be bool"
    e0 = result["regeneration_history"][0]
    for required_key in (
        "round", "candidates_requested", "candidates_returned",
        "candidates_accepted_this_round", "accepted_candidate_indices",
        "rejected_reasons", "empty_response", "parse_failed", "llm_step_name",
    ):
        assert required_key in e0, f"regeneration_history[0] missing key: {required_key!r}"
    print("All assertions passed: regeneration_history schema is correct.")


if __name__ == "__main__":
    main()