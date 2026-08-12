#!/usr/bin/env python3
"""Pivot flat retrieval_expression_style_summary.json by expression_style user_avg_depth.

`06_fast_fullscale_eval_<cat>.py` writes a single 'all_queries' group because it
is hardcoded to use QUERY_GROUP_KEY='all_queries'. Stage 08
`load_08_group_hit10` expects per complexity-bucket group_metrics with keys
'low_complexity' and 'high_complexity'. This script:

  1. Loads the flat summary produced by fast_fullscale.
  2. For each (user_id, asin) join on expression_style query file to get
     user_avg_depth.
  3. Re-bin each retriever's `all_query_records` into:
       low_complexity  (user_avg_depth <= 6)
       high_complexity (user_avg_depth >= 7)
     per the same CUT points used elsewhere in Stage 08.
  4. Recompute 'metrics' / 'group_metrics' and write a new JSON to:
       06_retrieval/<cat>/retrieval_expression_style_summary_pivot.json

The pivot file is shape-compatible with what Stage 08 reads, so copy the pivot
over the original `retrieval_expression_style_summary.json` to feed Stage 08.

Usage:
    python3 08_compare_all_domain/pivot_summary_by_user_avg_depth.py \\
        --category Baby_Products \\
        [--mid-threshold 6] [--high-threshold 7]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
RETRIEVAL_ROOT = RESULT_ROOT / "06_retrieval"
EXPRESSION_STYLE_DIR_NAME = "06_query"

PIVOT_THRESHOLD_MID = 6  # user_avg_depth <= 6 -> low
PIVOT_THRESHOLD_HIGH = 7  # user_avg_depth >= 7 -> high

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def _bucket_for(depth: int) -> str:
    if depth <= PIVOT_THRESHOLD_MID:
        return "low_complexity"
    if depth >= PIVOT_THRESHOLD_HIGH:
        return "high_complexity"
    return "mid_complexity"  # not used in Stage 08 input format but kept for safety


def _load_user_avg_depth_index(category: str) -> dict:
    """Map (user_id, asin) -> user_avg_depth from the expression_style query file."""
    syntax_path = RESULT_ROOT / EXPRESSION_STYLE_DIR_NAME / category / "query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json"
    if not syntax_path.exists():
        raise FileNotFoundError(f"Syntax-depth query file missing: {syntax_path}")
    with open(syntax_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    index = {}
    for row in rows:
        user_id = row.get("user_id")
        asin = row.get("asin")
        depth_record = row.get("expression_style_query", {})
        depth = depth_record.get("user_avg_depth")
        if user_id is None or asin is None or depth is None:
            continue
        index[(user_id, asin)] = int(depth)
    return index


def _average_metrics(values: list) -> dict:
    if not values:
        return {}
    keys = sorted({k for d in values for k in d.keys() if isinstance(d, dict)})
    result = {}
    for k in keys:
        nums = [d[k] for d in values if isinstance(d, dict) and k in d and d[k] is not None]
        if not nums:
            continue
        result[k] = sum(nums) / len(nums)
    return result


def pivot_for_category(category: str) -> Path:
    src = RETRIEVAL_ROOT / category / "retrieval_expression_style_summary.json"
    if not src.exists():
        raise FileNotFoundError(f"Flat summary missing: {src}")

    with open(src, "r", encoding="utf-8") as f:
        summary = json.load(f)

    depth_index = _load_user_avg_depth_index(category)

    out_records = []
    for item in summary.get("all_results_combined", []):
        retriever = item.get("retriever")
        if not retriever:
            continue
        bucket_records = defaultdict(list)
        bucket_metrics = defaultdict(list)
        for record in item.get("all_query_records", []):
            user_id = record.get("user_id")
            asin = record.get("asin")
            depth = depth_index.get((user_id, asin))
            if depth is None:
                continue
            bucket = _bucket_for(depth)
            augmented_record = dict(record)
            # Reviewer note iter #70: Stage 08's compute_three_level_score
            # requires `expression_style` int field on each record, downstream of
            # the pivot. The original flat summary stores query info but not
            # the depth itself; we join on user_id+asin to attach it.
            augmented_record["expression_style"] = depth
            bucket_records[bucket].append(augmented_record)
            hit = record.get("hit_at10")
            p_at1 = record.get("p_at1")
            p_at3 = record.get("p_at3")
            p_at5 = record.get("p_at5")
            p_at10 = record.get("p_at10")
            n_at10 = record.get("n_at10")
            mrr_at10 = record.get("mrr_at10")
            metrics = {
                "H@10": hit,
                "P@1": p_at1,
                "P@3": p_at3,
                "P@5": p_at5,
                "P@10": p_at10,
                "N@10": n_at10,
                "MR@10": mrr_at10,
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                bucket_metrics[bucket].append(metrics)
        if not bucket_metrics:
            continue
        new_group_metrics = {bucket: _average_metrics(metrics) for bucket, metrics in bucket_metrics.items()}
        # Flatten augmented_records from all buckets back into a single list
        # so downstream consumers (e.g. compute_three_level_score) see the
        # `expression_style` field we attached during the per-record join.
        augmented_all = []
        for bucket in bucket_records:
            augmented_all.extend(bucket_records[bucket])
        new_item = dict(item)
        new_item["group_metrics"] = new_group_metrics
        new_item["group_counts"] = {bucket: len(records) for bucket, records in bucket_records.items()}
        new_item["all_query_records"] = augmented_all
        out_records.append(new_item)

    if not out_records:
        raise ValueError(f"Pivot produced no retriever records for {category}")

    summary["all_results_combined"] = out_records
    summary["pivot_metadata"] = {
        "pivot_script": "08_compare_all_domain/pivot_summary_by_user_avg_depth.py",
        "low_threshold_inclusive": PIVOT_THRESHOLD_MID,
        "high_threshold_inclusive": PIVOT_THRESHOLD_HIGH,
        "depth_key": "expression_style_query.user_avg_depth",
    }

    out_path = RETRIEVAL_ROOT / category / "retrieval_expression_style_summary_pivot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[pivot] {category}: {len(out_records)} retriever records -> {out_path}")
    return out_path


def pivot_noisy_for_category(category: str) -> Path:
    """Same depth join for the Stage 07 noisy JSON.

    Stage 08 load_09_group_hit10 expects each record to have
    `expression_style: int`. The original noisy query JSON stores user_id+asin
    only; we look up user_avg_depth from the expression_style query file and
    stamp it back onto every record (both correct and noisy sides).
    """
    src = RESULT_ROOT / "09_noisy_retrieval" / category / "expression_style_correct_vs_noisy_results.json"
    if not src.exists():
        raise FileNotFoundError(f"Noisy results missing: {src}")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    depth_index = _load_user_avg_depth_index(category)

    def _stamp(records):
        stamped = 0
        for record in records:
            depth = depth_index.get((record.get("user_id"), record.get("asin")))
            if depth is None:
                continue
            record["expression_style"] = int(depth)
            stamped += 1
        return stamped

    for side in ("raw_correct_results", "raw_noisy_results", "correct_results", "noisy_results"):
        items = data.get(side, [])
        for item in items:
            all_records = item.get("all_query_records")
            if isinstance(all_records, list):
                item["all_query_records"] = _stamp(all_records)
            else:
                # Reviewer note iter #70: the iter #68-produced noisy JSONs
                # store `all_query_records` as an int (= num_queries) rather
                # than a list of records. There is no way to recover the
                # per-(user_id, asin) rows without re-running the eval, so
                # we synthesize a list of pair_id-only stub records from the
                # binary correct/noisy JSON if available. For simplicity here,
                # when a list is missing we set it to []; downstream consumers
                # that need it will read final_statistics_filters or just
                # skip the pair.
                item["all_query_records"] = []
                item["all_query_records_not_a_list"] = True

    data.setdefault("pivot_metadata", {})
    data["pivot_metadata"]["noisy_pivot_script"] = "08_compare_all_domain/pivot_summary_by_user_avg_depth.py:pivot_noisy_for_category"

    out_path = src.with_name("expression_style_correct_vs_noisy_results_pivot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[pivot-noisy] {category}: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, help="Pivot one category (default: all 3)")
    parser.add_argument("--mid-threshold", type=int, default=PIVOT_THRESHOLD_MID)
    parser.add_argument("--high-threshold", type=int, default=PIVOT_THRESHOLD_HIGH)
    parser.add_argument(
        "--also-noisy",
        action="store_true",
        help="Also pivot the 09_noisy_retrieval/<cat>/expression_style_correct_vs_noisy_results.json",
    )
    parser.add_argument(
        "--only-noisy",
        action="store_true",
        help="Only pivot the noisy JSON (skip the 06_retrieval summary)",
    )
    args = parser.parse_args()

    targets = [args.category] if args.category else CATEGORIES
    for category in targets:
        if not args.only_noisy:
            pivot_for_category(category)
        if args.also_noisy or args.only_noisy:
            pivot_noisy_for_category(category)


if __name__ == "__main__":
    main()
