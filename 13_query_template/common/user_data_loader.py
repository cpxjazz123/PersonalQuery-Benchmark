"""Load query_template.json and reshape to the 08-stage evaluate_* expected format.

The 08-stage `evaluate_dense_retriever` / `evaluate_bm25_retriever` /
`evaluate_cached_result_retriever` / `evaluate_splade_retriever` all consume:

    user_queries: Dict[user_id, List[{query, asin, word_count,
                                       query_group, query_group_ratio}]]
    user_to_group: Dict[user_id, str]

This module reads the 14-stage `query_template.json` (a flat list of records)
and emits those two dicts. There is no depth field, so every user is placed
into a single `all_queries` group with `query_group_ratio=0.0`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


QUERY_GROUP_FIELD = "query_group"
QUERY_GROUP_KEY = "all_queries"
QUERY_GROUP_DISPLAY = {QUERY_GROUP_KEY: "全部查询"}


def load_template_user_queries(
    query_template_file: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, Any]]:
    """Return (user_queries, user_to_group, meta).

    - user_queries: Dict[user_id, List[query_item]]
      Each query_item: {query, asin, word_count, query_group, query_group_ratio}
    - user_to_group: Dict[user_id, "all_queries"]
    - meta: {template_id, template_version, template_str, source_file, user_count, query_count}
    """
    path = Path(query_template_file)
    if not path.exists():
        raise FileNotFoundError(f"query_template file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"query_template file must contain a list, got {type(data).__name__}")

    user_queries: Dict[str, List[Dict[str, Any]]] = {}
    user_to_group: Dict[str, str] = {}
    template_id = None
    template_version = None

    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"row must be dict at index {row_index}, got {type(item).__name__}")
        for field in ("user_id", "asin", "query", "word_count"):
            if field not in item:
                raise ValueError(f"row missing required field '{field}' at index {row_index}")

        user_id = item["user_id"]
        asin = item["asin"]
        query_text = item["query"]
        word_count = item["word_count"]

        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"row has invalid user_id at index {row_index}: {user_id!r}")
        if not isinstance(asin, str) or not asin:
            raise ValueError(f"row has invalid asin for user={user_id} at index {row_index}: {asin!r}")
        if not isinstance(query_text, str):
            raise TypeError(f"query must be str for user={user_id} at index {row_index}")
        query_text = query_text.strip()
        if not query_text:
            raise ValueError(f"query is empty for user={user_id} at index {row_index}")
        if not isinstance(word_count, int):
            raise TypeError(f"word_count must be int for user={user_id} at index {row_index}")

        if template_id is None:
            template_id = item.get("template_id")
            template_version = item.get("template_version")

        if user_id not in user_queries:
            user_queries[user_id] = []
            user_to_group[user_id] = QUERY_GROUP_KEY
        elif user_to_group[user_id] != QUERY_GROUP_KEY:
            raise ValueError(
                f"user {user_id} appears in multiple query groups: "
                f"{user_to_group[user_id]} and {QUERY_GROUP_KEY}"
            )

        user_queries[user_id].append({
            "query": query_text,
            "asin": asin,
            "word_count": word_count,
            f"{QUERY_GROUP_FIELD}_ratio": 0.0,
            QUERY_GROUP_FIELD: QUERY_GROUP_KEY,
            "expression_style": 0,
            "target_depth": 0,
            "user_avg_depth": 0.0,
            "depth_group_label": "n/a",
        })

    if not user_queries:
        raise ValueError(f"No template queries loaded from {path}")

    meta = {
        "source_file": str(path),
        "template_id": template_id,
        "template_version": template_version,
        "user_count": len(user_queries),
        "query_count": sum(len(v) for v in user_queries.values()),
    }
    return user_queries, user_to_group, meta
