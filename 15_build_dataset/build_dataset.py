#!/usr/bin/env python3
"""Stage 15 — Per-ASIN build of clean / typo query pairs.

Source inputs (per category):
  * Stage 08 selected queries: result/08_select_query/<subdir>/selected_queries.json
      - ``selections`` is a list of {asin, users: [{uid, query, logp, d2, margin}, ...]}
  * Stage 10 typo injection: result/10_typo_injection/<subdir>/typo_injection_results.json
      - ``results`` is a flat list of {uid, asin, original_query, typo_query, ...}.

Output (per category): result/15_build_dataset/<subdir>.json
  A dict keyed by ASIN; each ASIN maps to a list of per-sample records. Each
  sample keeps only the four primary fields:
      {asin, uid, syntax_query, typo_query (or null)}

Pairing rule: a Stage 10 entry only joins a Stage 08 entry on the exact triple
(asin, uid, original_query == query). If Stage 10 did not produce a typo for a
Stage 08 query (e.g. all gates failed), the Stage 08 row is still emitted with
typo_query = null so downstream consumers know the clean-only cell exists.
If Stage 10 produced a typo whose original_query is absent from Stage 08, it is
logged as an unmatched typo and skipped (Stage 10's pool is built from Stage 08
selections, so this should not normally happen, but we guard against drift).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

CATEGORY_INPUTS = [
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_for_category(subdir: str) -> dict[str, list[dict]]:
    """Build the per-ASIN dataset JSON for one category."""
    sel_path = REPO_ROOT / f"result/08_select_query/{subdir}/selected_queries.json"
    typo_path = REPO_ROOT / f"result/10_typo_injection/{subdir}/typo_injection_results.json"
    out_path = REPO_ROOT / f"result/15_build_dataset/{subdir}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sel = json.load(open(sel_path))
    typo = json.load(open(typo_path))

    # Stage 08: asin -> list of {uid, query, logp, d2, margin}
    stage08_by_asin: dict[str, list[dict]] = defaultdict(list)
    for entry in sel["selections"]:
        asin = entry["asin"]
        for u in entry["users"]:
            stage08_by_asin[asin].append({
                "uid":    u["uid"],
                "query":  u["query"],
                "logp":   u.get("logp"),
                "d2":     u.get("d2"),
                "margin": u.get("margin"),
            })

    # Stage 10: (asin, uid, original_query) -> typo record
    stage10_index: dict[tuple, dict] = {}
    for pair in typo["results"]:
        key = (pair["asin"], pair["uid"], pair["original_query"])
        if key in stage10_index:
            log(f"  [warn] duplicate Stage 10 entry for {key!r}, keeping first")
            continue
        stage10_index[key] = pair

    # Build the per-ASIN output: a dict keyed by ASIN, each value is a list of
    # per-sample records belonging to that ASIN.
    asins = sorted(stage08_by_asin.keys())
    by_asin: dict[str, list[dict]] = {asin: [] for asin in asins}
    n_queries_total = 0
    n_typo_paired = 0
    n_clean_only = 0

    for asin in asins:
        for u in stage08_by_asin[asin]:
            q_clean = u["query"]
            key = (asin, u["uid"], q_clean)
            typo_rec = stage10_index.get(key)
            if typo_rec is None:
                by_asin[asin].append({
                    "asin":         asin,
                    "uid":          u["uid"],
                    "syntax_query": q_clean,
                    "typo_query":   None,
                })
                n_clean_only += 1
            else:
                by_asin[asin].append({
                    "asin":         asin,
                    "uid":          u["uid"],
                    "syntax_query": typo_rec["original_query"],
                    "typo_query":   typo_rec["typo_query"],
                })
                n_typo_paired += 1
            n_queries_total += 1

    # Sanity check: any Stage 10 typos whose (asin, uid, original_query) never
    # matched a Stage 08 row?  This can only happen if Stage 10 was run on a
    # different selection pool than the saved Stage 08 JSON — surface it.
    matched_stage10_keys = set()
    for asin, users in stage08_by_asin.items():
        for u in users:
            matched_stage10_keys.add((asin, u["uid"], u["query"]))
    n_unmatched_typo = sum(1 for k in stage10_index.keys() if k not in matched_stage10_keys)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(by_asin, f, indent=2, ensure_ascii=False)

    log(f"  [{subdir}] asins={len(asins)} queries={n_queries_total} "
        f"typo_paired={n_typo_paired} clean_only={n_clean_only} "
        f"unmatched_typo={n_unmatched_typo} -> {out_path.relative_to(REPO_ROOT)}")
    return by_asin


def main_task_body() -> None:
    log("=== Stage 15 build per-ASIN clean/typo dataset ===")
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n--- [{category}] (subdir={subdir}) ---")
        build_for_category(subdir)


if __name__ == "__main__":
    main_task_body()
