"""14-stage noisy template query generator.

Reads 14-stage clean `query_template.json`, calls 07-stage
`apply_lambdamart_userbased_noisy.process_batch` (rule-based, not actual
LightGBM) to inject per-user writing errors, and writes the noisy output in
14-stage schema:

    [ {user_id, asin, query (=noisy_query), word_count, attrs_used,
       attrs_source, template_id, template_version, noisy_meta} ]

The `noisy_meta` block carries the per-record noise provenance (selected
token / applied error) for downstream analysis. The 14-stage noisy query file
sits next to the clean file with the `_noisy.json` suffix.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_14_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(STAGE_14_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
STAGE_07_DIR = os.path.join(PROJECT_ROOT, "07_inject_noisy", "common")


from common_utils import log  # 统一 log 函数


def _load_07_modules() -> Dict[str, Any]:
    """Import the 07 noisy injection helpers (no source mutation, no fallback).

    07's `apply_lambdamart_userbased_noisy.py` does `from common import ...`
    which would resolve to 14's `common` package if STAGE_07_DIR is on
    sys.path naively. To avoid this, we load 07's `common.py` and
    `apply_lambdamart_userbased_noisy.py` both via importlib under namespaced
    module slots (`_stage07_*`). The `from common import ...` inside the 07
    module then resolves to the 07 module object we already loaded.

    We also pre-register `_stage07_common` in sys.modules under the bare
    name `common` ONLY for the duration of the 07 module load, then restore
    it. This is a one-time shim, not a mutation of any 07 disk file.
    """
    import importlib.util

    common_path = os.path.join(STAGE_07_DIR, "common.py")
    apply_path = os.path.join(STAGE_07_DIR, "apply_lambdamart_userbased_noisy.py")
    token_path = os.path.join(STAGE_07_DIR, "token_level_lambdamart_user_based.py")

    for path in (common_path, apply_path, token_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"07 module not found: {path}")

    common_spec = importlib.util.spec_from_file_location(
        "_stage07_common", common_path,
    )
    if common_spec is None or common_spec.loader is None:
        raise ImportError(f"Unable to build importlib spec for {common_path}")
    common_mod = importlib.util.module_from_spec(common_spec)
    sys.modules["_stage07_common"] = common_mod
    common_spec.loader.exec_module(common_mod)

    sys.modules["common"] = common_mod
    try:
        token_spec = importlib.util.spec_from_file_location(
            "_stage07_token_lambdamart", token_path,
        )
        token_mod = importlib.util.module_from_spec(token_spec)
        sys.modules["_stage07_token_lambdamart"] = token_mod
        sys.modules["token_level_lambdamart_user_based"] = token_mod
        token_spec.loader.exec_module(token_mod)

        apply_spec = importlib.util.spec_from_file_location(
            "_stage07_apply_lambdamart_userbased_noisy", apply_path,
        )
        apply_mod = importlib.util.module_from_spec(apply_spec)
        sys.modules["_stage07_apply_lambdamart_userbased_noisy"] = apply_mod
        sys.modules["apply_lambdamart_userbased_noisy"] = apply_mod
        apply_spec.loader.exec_module(apply_mod)
    finally:
        sys.modules.pop("common", None)
        sys.modules.pop("apply_lambdamart_userbased_noisy", None)
        sys.modules.pop("token_level_lambdamart_user_based", None)

    return {
        "process_batch": apply_mod.process_batch,
        "load_user_errors": common_mod.load_user_errors,
        "write_json_array": common_mod.write_json_array,
    }


def _load_template_records(query_template_file: str) -> List[Dict[str, Any]]:
    path = Path(query_template_file)
    if not path.exists():
        raise FileNotFoundError(f"clean query_template file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"query_template file must contain a list, got {type(data).__name__}")
    if not data:
        raise ValueError(f"query_template file is empty: {path}")
    return data


def _build_user_error_file_path(category: str) -> str:
    """07-stage noisy injection pulls user writing errors from this file.

    Path must match `apply_lambdamart_userbased_noisy.build_config` (the
    07 stage hard-codes it under 04_writing_analysis).
    """
    return os.path.join(
        REPO_ROOT,
        "result",
        "personal_query",
        "04_writing_analysis",
        category,
        "writing_error.json",
    )


def _build_07_tasks(
    records: List[Dict[str, Any]],
    user_errors: Dict[str, Dict[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    """Reshape 14-stage clean records to the 07 stage `process_batch` task format.

    Note: 04-stage `writing_error.json` covers a strict subset of 14-stage
    template users (e.g., 6167/6961 for Baby_Products). Users without any
    writing error are passed to `process_batch` with empty `errors=[]`; the
    07 stage handles this by emitting `status='skipped'` with
    `noisy_query=clean_query`. This keeps the noisy user set identical to
    the clean user set so H@10 is comparable.
    """
    tasks: List[Dict[str, Any]] = []
    for r in records:
        uid = r["user_id"]
        errors_data = user_errors.get(uid, {})
        if isinstance(errors_data, dict):
            errors = errors_data.get("writing", [])
        elif isinstance(errors_data, list):
            errors = errors_data
        else:
            errors = []
        tasks.append({
            "uid": uid,
            "asin": r["asin"],
            "clean_query": r["query"],
            "query_info": None,
            "errors": errors,
            "attrs_used": r.get("attrs_used"),
        })
    return tasks


def _coerce_noisy_record(
    clean_record: Dict[str, Any],
    noisy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Map 07-stage `process_batch` output back to 14-stage noisy record schema."""
    noisy_query = noisy_result.get("noisy_query", clean_record["query"])
    applied_error = noisy_result.get("applied_error")
    status = noisy_result.get("status", "unknown")

    noisy_meta: Dict[str, Any] = {
        "clean_query": clean_record["query"],
        "noisy_query": noisy_query,
        "query_rewritten": bool(noisy_result.get("query_rewritten", False)),
        "selected_token": noisy_result.get("selected_token"),
        "score": noisy_result.get("score", 0.0),
        "applied_error": applied_error,
        "noise_method": "lambdamart_userbased",
        "status": status,
    }
    if status == "error":
        noisy_meta["error"] = noisy_result.get("error")

    noisy_record: Dict[str, Any] = {
        "user_id": clean_record["user_id"],
        "asin": clean_record["asin"],
        "query": noisy_query,
        "word_count": clean_record.get("word_count"),
        "attrs_used": clean_record.get("attrs_used"),
        "attrs_source": clean_record.get("attrs_source"),
        "template_id": clean_record.get("template_id"),
        "template_version": clean_record.get("template_version"),
        "noisy_meta": noisy_meta,
    }
    return noisy_record


def main(
    category: str,
    clean_query_file: Optional[str] = None,
    noisy_query_file: Optional[str] = None,
) -> Dict[str, int]:
    from .config import get_category_config  # noqa: WPS433

    cat_cfg = get_category_config(category)
    clean_query_file = clean_query_file or cat_cfg["output_query_file"]
    noisy_query_file = noisy_query_file or cat_cfg["output_query_file_noisy"]
    user_error_file = _build_user_error_file_path(category)

    log("=" * 80)
    log(f"14-stage noisy template query generation - {category}")
    log("=" * 80)
    log(f"  clean_query_file:  {clean_query_file}")
    log(f"  noisy_query_file:  {noisy_query_file}")
    log(f"  user_error_file:   {user_error_file}")

    helpers = _load_07_modules()
    log(f"  loaded 07 modules: process_batch, load_user_errors, write_json_array")

    records = _load_template_records(clean_query_file)
    log(f"  loaded {len(records)} clean template records")

    user_errors = helpers["load_user_errors"](user_error_file)
    log(f"  users with writing errors: {len(user_errors)}")

    tasks = _build_07_tasks(records, user_errors)
    tasks_with_errors = sum(1 for t in tasks if t["errors"])
    tasks_without_errors = len(tasks) - tasks_with_errors
    log(f"  tasks: {len(tasks)} total / {tasks_with_errors} with user errors / "
        f"{tasks_without_errors} without (will be skipped by process_batch)")

    log(f"  dispatching {len(tasks)} tasks to 07 process_batch (rule-based)...")
    results = helpers["process_batch"](None, tasks, category)
    log(f"  process_batch returned {len(results)} results")

    rewritten = sum(1 for r in results if r.get("query_rewritten"))
    log(f"  noisy rewrite hit rate: {rewritten}/{len(results)} ({rewritten/len(results)*100:.1f}%)")

    by_key = {(r["uid"], r["asin"]): r for r in results}
    noisy_records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for clean_record in records:
        key = (clean_record["user_id"], clean_record["asin"])
        result = by_key.get(key)
        if result is None:
            skipped.append({"user_id": clean_record["user_id"], "asin": clean_record["asin"]})
            continue
        noisy_records.append(_coerce_noisy_record(clean_record, result))

    if skipped:
        raise RuntimeError(
            f"Rule 7: {len(skipped)} clean records had no matching 07 noisy result "
            f"(first 3: {skipped[:3]}). This should not happen."
        )

    Path(noisy_query_file).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{noisy_query_file}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(noisy_records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, noisy_query_file)
    size_kb = os.path.getsize(noisy_query_file) / 1024
    log(f"  ✓ noisy file saved: {len(noisy_records)} records, {size_kb:.1f} KB")

    stats = {
        "total": len(noisy_records),
        "rewritten": rewritten,
        "skipped": sum(1 for r in results if r.get("status") != "success"),
    }
    log("=" * 80)
    log("✅ 14-stage noisy template query generation complete")
    log(f"  total:    {stats['total']}")
    log(f"  rewritten: {stats['rewritten']} ({stats['rewritten']/max(stats['total'],1)*100:.1f}%)")
    log(f"  skipped:  {stats['skipped']}")
    log("=" * 80)
    return stats


if __name__ == "__main__":
    import sys
    category = sys.argv[1] if len(sys.argv) > 1 else "Baby_Products"
    main(category)
