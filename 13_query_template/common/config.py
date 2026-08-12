"""13_query_template 配置加载模块.

从 13_query_template_config.json 读取模板字符串、必填属性键、各 category 路径.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


_CONFIG_CACHE: Dict[str, Any] = {}


def _config_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "13_query_template_config.json")


def get_config() -> Dict[str, Any]:
    if not _CONFIG_CACHE:
        with open(_config_path(), "r", encoding="utf-8") as f:
            _CONFIG_CACHE.update(json.load(f))
    return _CONFIG_CACHE


def get_template() -> Dict[str, Any]:
    return get_config()["template"]


def get_query_config() -> Dict[str, Any]:
    return get_config()["query_config"]


def resolve_path(path_template: str) -> str:
    config = get_config()
    result = path_template
    for key in ("test_result", "scratch_result", "amazon_reviews", "result", "workspace", "root"):
        placeholder = f"{{{key}}}"
        if placeholder in result:
            result = result.replace(placeholder, config["base_paths"][key])
    return result


def get_category_config(category_name: str) -> Dict[str, Any]:
    cat_config = get_config()["categories"].get(category_name, {})
    resolved: Dict[str, Any] = {}
    for key, value in cat_config.items():
        if isinstance(value, str):
            resolved[key] = resolve_path(value)
        else:
            resolved[key] = value

    clean_query_file = resolved.get("output_query_file", "")
    clean_summary_file = resolved.get("output_summary_file", "")
    resolved["output_query_file_noisy"] = clean_query_file.replace(".json", "_noisy.json")
    resolved["output_summary_file_noisy"] = clean_summary_file.replace(".json", "_noisy.json")
    return resolved


def list_categories() -> List[str]:
    return list(get_config()["categories"].keys())
