#!/usr/bin/env python3
"""Revised query helpers for noisy retrieval evaluation."""

from functools import lru_cache
import json
import os
from typing import Dict, List, Tuple


NOISY_QUERY_BASE = "/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy"


def _parse_packed_json_objects(content: str) -> List[Dict]:
    """解析拼接式 JSON 或标准 JSON 数组。"""
    content = content.strip()
    if not content:
        return []

    if content.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data).__name__}")
        return data

    data: List[Dict] = []
    depth = 0
    start = -1
    for idx, char in enumerate(content):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                data.append(json.loads(content[start:idx + 1]))
                start = -1

    if depth != 0:
        raise ValueError("Malformed packed JSON stream: braces are unbalanced")

    return data


@lru_cache(maxsize=None)
def load_revised_query_map(category_name: str) -> Dict[Tuple[str, str, str], str]:
    """从 07_inject_noisy 的输出中加载 revised_correct_query 映射。"""
    noisy_query_file = os.path.join(NOISY_QUERY_BASE, category_name, "noisy_query.json")
    if not os.path.exists(noisy_query_file):
        raise FileNotFoundError(f"Revised query source not found: {noisy_query_file}")

    with open(noisy_query_file, "r", encoding="utf-8") as f:
        records = _parse_packed_json_objects(f.read())

    if not records:
        raise ValueError(f"Revised query source is empty: {noisy_query_file}")

    revised_map: Dict[Tuple[str, str, str], str] = {}
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise TypeError(
                f"Revised query source must contain objects, got {type(item).__name__} at index {index}"
            )

        user_id = str(item.get("user_id", "")).strip()
        asin = str(item.get("asin", "")).strip()
        query_category = str(item.get("query_category", "")).strip()
        revised_query = str(item.get("ground_truth_query", "")).strip()

        if not user_id or not asin or not query_category:
            raise ValueError(
                f"Missing key fields in revised query source at index {index}: "
                f"user_id={user_id!r}, asin={asin!r}, query_category={query_category!r}"
            )
        if not revised_query:
            raise ValueError(
                f"Missing ground_truth_query in revised source at index {index}: "
                f"user_id={user_id!r}, asin={asin!r}, query_category={query_category!r}"
            )

        key = (user_id, asin, query_category)
        # noisy_query.json 是追加式日志，同一键的后续记录应覆盖前序记录。
        revised_map[key] = revised_query

    if not revised_map:
        raise ValueError(f"No revised queries loaded from: {noisy_query_file}")

    return revised_map
