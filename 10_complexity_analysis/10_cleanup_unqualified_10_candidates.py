#!/usr/bin/env python3
"""Remove users whose 10 expression_style candidates all fail the VADES threshold.

Reads the rejected-records JSONL produced by
`10_query_selection_<Cat>.py` (one best row per user whose 10 candidates
all had `passes_abs_threshold=False`), collects those user_ids, and deletes
matching records from
`result/personal_query/04_query/<Cat>/query_by_syntax_depth_no_depth_check_10.json`.

Idempotent: re-running deletes 0 records.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
OUTPUT_TAG = "vades_lite_sentence_user_distribution_train10_holdout10"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _rejected_user_ids(category: str) -> set[str]:
    rejected_file = (
        REPO_ROOT
        / "result" / "personal_query" / "10_complexity_analysis_clause_features"
        / category / f"{OUTPUT_TAG}_rejected_query_records.jsonl"
    )
    if not rejected_file.exists():
        raise FileNotFoundError(f"rejected records file not found: {rejected_file}")
    user_ids: set[str] = set()
    with rejected_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("user_id")
            if not isinstance(uid, str) or not uid:
                raise ValueError(f"rejected record missing user_id: {rejected_file}")
            user_ids.add(uid)
    return user_ids


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def process_category(category: str) -> dict:
    raw_file = (
        REPO_ROOT / "result" / "personal_query" / "04_query" / category
        / "query_by_syntax_depth_no_depth_check_10.json"
    )
    if not raw_file.exists():
        raise FileNotFoundError(f"raw 10-candidate file not found: {raw_file}")

    rejected_ids = _rejected_user_ids(category)
    log(f"  rejected user_ids loaded: {len(rejected_ids)}")

    raw_records = json.loads(raw_file.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise TypeError(f"raw file must contain a JSON list: {raw_file}")
    before_count = len(raw_records)

    kept_records = [row for row in raw_records if row.get("user_id") not in rejected_ids]
    deleted_count = before_count - len(kept_records)

    _write_json_atomic(raw_file, kept_records)

    kept_user_ids = {row["user_id"] for row in kept_records if "user_id" in row}
    return {
        "category": category,
        "before": before_count,
        "deleted": deleted_count,
        "after": len(kept_records),
        "rejected_user_count": len(rejected_ids),
        "kept_user_count": len(kept_user_ids),
        "raw_file": str(raw_file),
    }


def main() -> None:
    log("开始清理不合格用户的 10 候选记录")
    log(f"类别列表: {CATEGORIES}")
    log(f"vades output tag: {OUTPUT_TAG}")

    summaries = []
    for category in CATEGORIES:
        log(f"\n处理类别: {category}")
        summary = process_category(category)
        log(
            f"  before={summary['before']} deleted={summary['deleted']} "
            f"after={summary['after']} rejected_users={summary['rejected_user_count']}"
        )
        log(f"  raw file: {summary['raw_file']}")
        summaries.append(summary)

    log("\n汇总:")
    for s in summaries:
        log(
            f"  {s['category']}: 删除 {s['deleted']} 条 "
            f"(rejected_users={s['rejected_user_count']}, "
            f"after={s['after']})"
        )
    log("清理完成")


if __name__ == "__main__":
    main()
