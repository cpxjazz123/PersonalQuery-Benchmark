#!/usr/bin/env python3
"""Unified Stage 11 query dataset CLI.

Dataset is built from:
- Stage 06: for attrs_used
- Stage 07: for clean_query, noisy_query, error_type
- Stage 12: for cluster
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/fs04/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
DATASET_ROOT = RESULT_ROOT / "11_query_dataset"
ROOT_DATASET_DIR = Path("/home/wlia0047/ar57/wenyu/dataset")
DEFAULT_SUMMARY_FILE = DATASET_ROOT / "summary.json"

OUTPUT_FIELDS = [
    "category",
    "uuid",
    "asin",
    "cluster",
    "correct_query",
    "noisy_query",
    "error_pattern",
]

FORBIDDEN_DATASET_FIELDS = {
    "sample_id",
    "user_id",
    "query_type",
    "idf",
    "target_depth",
    "user_avg_depth",
    "source_stage",
    "query_category",
    "complexity_level",
    "complexity_group",
    "depth",
    "query_rewritten",
    "selected_token",
    "score",
    "applied_error",
    "injected_errors",
    "status",
    "injection_mode",
    "debug_response",
    "error",
    "attrs_used",
    "error_type",
}

ATTRIBUTE_TYPE_LABELS = {
    "A1": "product_type",
    "A2": "brand",
    "A3": "price",
    "A4": "appearance",
    "A5": "use_case",
    "A6": "detailed",
    "A7": "material",
    "A8": "safety",
    "A9": "durability",
    "A10": "ease_of_use",
    "A11": "temperature_resistance",
    "A12": "surface",
    "A13": "reusability",
    "A14": "size",
    "A15": "weight",
    "A16": "compatibility",
    "A17": "flavor",
    "A18": "quality",
}


@dataclass(frozen=True)
class DatasetPaths:
    category: str
    stage6_query_file: Path
    stage7_noisy_file: Path
    cluster_profile_file: Path
    output_dir: Path
    dataset_file: Path
    summary_file: Path


SUPPORTED_CATEGORIES = (
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def require_item_text(item: dict[str, Any], key: str, label: str) -> str:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def require_text_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def require_item_int(item: dict[str, Any], key: str, label: str) -> int:
    if key not in item:
        raise KeyError(f"{label} is missing required key: {key}")
    value = item[key]
    if not isinstance(value, int):
        raise TypeError(f"{label}.{key} must be an integer")
    return value


def require_int_value(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def require_attr_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def ensure_supported_categories(categories: list[str]) -> list[str]:
    for category in categories:
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")
    return categories


def read_json(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_dict(data, label)


def read_json_array(path: Path, label: str) -> list[Any]:
    require_file(path, label)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return require_list(data, label)


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    require_file(path, label)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"{label}:{line_number} is empty")
            rows.append(require_dict(json.loads(stripped), f"{label}:{line_number}"))
    if not rows:
        raise ValueError(f"{label} contains no rows: {path}")
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            output_row = {field: row[field] for field in OUTPUT_FIELDS}
            f.write(json.dumps(output_row, ensure_ascii=False))
            f.write("\n")


def format_stat_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def format_two_column_stats_table(rows: list[tuple[str, int | float]]) -> str:
    if not rows:
        raise ValueError("Cannot format an empty statistics table")

    string_rows = [(name, format_stat_value(value)) for name, value in rows]
    total_query_index = next((idx for idx, (name, _) in enumerate(string_rows) if name == "Total Query"), None)
    split_index = total_query_index + 1 if total_query_index is not None else (len(string_rows) + 1) // 2
    left_rows = string_rows[:split_index]
    right_rows = string_rows[split_index:]

    stat_width = max(len("Statistics"), *(len(name) for name, _ in string_rows))
    value_width = max(len("Value"), *(len(value) for _, value in string_rows))
    left_header = f"{'Statistics':<{stat_width}} {'Value':>{value_width}}"
    right_header = f"{'Statistics':<{stat_width}} {'Value':>{value_width}}"
    separator = "-" * len(left_header)

    lines = [
        f"{separator}-+-{separator}",
        f"{left_header} | {right_header}",
        f"{separator}-+-{separator}",
    ]
    for row_index, left_row in enumerate(left_rows):
        left_name, left_value = left_row
        if row_index < len(right_rows):
            right_name, right_value = right_rows[row_index]
        else:
            right_name, right_value = "", ""
        if left_name == "Total Category Users":
            lines.append(f"{separator}-+-{separator}")
        left_text = f"{left_name:<{stat_width}} {left_value:>{value_width}}"
        if right_name:
            lines.append(f"{left_text} | {right_name:<{stat_width}} {right_value:>{value_width}}")
        else:
            lines.append(left_text)
    lines.append(f"{separator}-+-{separator}")
    return "\n".join(lines)


def format_dict_table(values: dict[str, Any], key_title: str, value_title: str) -> str:
    if not values:
        raise ValueError("Cannot format an empty table")
    lines = [f"| {key_title} | {value_title} |", "|---|---:|"]
    for key in sorted(values):
        lines.append(f"| `{key}` | {values[key]} |")
    return "\n".join(lines)


def category_label(category: str) -> str:
    labels = {
        "Baby_Products": "Baby",
        "Grocery_and_Gourmet_Food": "Grocery",
        "Pet_Supplies": "Pet",
    }
    if category not in labels:
        raise KeyError(f"Missing statistics label for category: {category}")
    return labels[category]


def normalize_attrs_used(attrs_used: dict[str, Any], label: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for attr_key, attr_value in attrs_used.items():
        if not isinstance(attr_key, str) or not attr_key.strip():
            raise TypeError(f"{label} contains invalid attribute key: {attr_key!r}")
        if attr_key not in ATTRIBUTE_TYPE_LABELS:
            raise KeyError(f"{label} contains unsupported attribute key: {attr_key}")
        semantic_key = ATTRIBUTE_TYPE_LABELS[attr_key]
        if semantic_key in normalized:
            raise ValueError(f"{label} maps multiple slots to the same semantic key: {semantic_key}")
        normalized[semantic_key] = require_attr_value(attr_value, f"{label}[{attr_key}]")
    if len(normalized) != len(attrs_used):
        raise ValueError(f"{label} normalization changed attribute count unexpectedly")
    return normalized


def make_dataset_paths(category: str, output_root: Path) -> DatasetPaths:
    output_dir = output_root / category
    return DatasetPaths(
        category=category,
        stage6_query_file=(
            RESULT_ROOT
            / "06_query"
            / category
            / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
        ),
        stage7_noisy_file=(
            RESULT_ROOT
            / "07_inject_noisy"
            / category
            / "noisy_query.json"
        ),
        cluster_profile_file=(
            RESULT_ROOT
            / "12_complexity_analysis_clause_features"
            / category
            / "strict5550_query_gmm_user_profiles.jsonl"
        ),
        output_dir=output_dir,
        dataset_file=output_dir / "data.jsonl",
        summary_file=output_dir / "summary.json",
    )


def load_stage6_query_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load Stage 06 query file to extract attrs_used.

    Format: [{"user_id": ..., "asin": ..., "syntax_depth_query": {"query": ..., "attrs_used": {...}}}, ...]
    """
    rows = read_json_array(path, "Stage 06 latest query file")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_idx, raw_item in enumerate(rows):
        item = require_dict(raw_item, f"stage6[{row_idx}]")
        user_id = require_item_text(item, "user_id", f"stage6[{row_idx}]")
        asin = require_item_text(item, "asin", f"stage6[{row_idx}]")
        query_info = require_dict(item.get("syntax_depth_query"), f"stage6[{row_idx}].syntax_depth_query")
        query = require_item_text(query_info, "query", f"stage6[{row_idx}].syntax_depth_query")
        attrs_used = require_dict(query_info.get("attrs_used"), f"stage6[{row_idx}].syntax_depth_query.attrs_used")
        normalized_attrs_used = normalize_attrs_used(
            attrs_used,
            f"stage6[{row_idx}].syntax_depth_query.attrs_used",
        )
        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate Stage 06 key: {key}")
        index[key] = {
            "query": query,
            "attrs_used": normalized_attrs_used,
        }
    return index


def compute_error_pattern(clean_query: str, noisy_query: str) -> dict[str, str] | None:
    """Compute error pattern by finding the difference between clean and noisy query.

    Returns error_pattern with:
    - correct_word: the correct word from clean_query that was replaced
    - error_word: the error word that replaced it in noisy_query
    """
    import re

    clean_words = clean_query.split()
    noisy_words = noisy_query.split()

    if len(clean_words) != len(noisy_words):
        return None

    # Find the first position where words differ
    for i, (clean_word, noisy_word) in enumerate(zip(clean_words, noisy_words)):
        if clean_word != noisy_word:
            # Extract only alphanumeric parts for comparison
            clean_alnum = re.sub(r'[^a-zA-Z0-9]', '', clean_word)
            noisy_alnum = re.sub(r'[^a-zA-Z0-9]', '', noisy_word)
            if clean_alnum and noisy_alnum:
                return {"correct_word": clean_word, "error_word": noisy_word}
    return None


def load_stage7_noisy_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load Stage 07 noisy query file.

    Format: [{"uid": ..., "asin": ..., "clean_query": ..., "noisy_query": ...,
              "query_rewritten": ..., "selected_token": ..., "score": ...,
              "applied_error": {"original": ..., "corrected": ..., "error_type": ...},
              "status": ...}, ...]
    """
    rows = read_json_array(path, "Stage 07 noisy query file")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_idx, raw_item in enumerate(rows):
        item = require_dict(raw_item, f"stage7[{row_idx}]")
        user_id = require_item_text(item, "uid", f"stage7[{row_idx}]")
        asin = require_item_text(item, "asin", f"stage7[{row_idx}]")
        clean_query = require_item_text(item, "clean_query", f"stage7[{row_idx}]")
        noisy_query = require_item_text(item, "noisy_query", f"stage7[{row_idx}]")
        error_pattern = compute_error_pattern(clean_query, noisy_query)
        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate Stage 07 key: {key}")
        index[key] = {
            "clean_query": clean_query,
            "noisy_query": noisy_query,
            "error_pattern": error_pattern,
        }
    return index


def load_cluster_profile_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path, "Cluster user profiles")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_idx, item in enumerate(rows):
        user_id = require_item_text(item, "user_id", f"cluster_profiles[{row_idx}]")
        asin = require_item_text(item, "asin", f"cluster_profiles[{row_idx}]")
        cluster_index = require_item_int(item, "cluster_index", f"cluster_profiles[{row_idx}]")
        key = (user_id, asin)
        if key in index:
            raise ValueError(f"Duplicate cluster profile key: {key}")
        index[key] = {
            "cluster": cluster_index,
        }
    return index


def build_dataset_rows(
    category: str,
    stage6_index: dict[tuple[str, str], dict[str, Any]],
    stage7_index: dict[tuple[str, str], dict[str, Any]],
    cluster: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    # Use Stage 06 as primary source, intersect with cluster (records without cluster are excluded)
    common_keys = set(stage6_index) & set(cluster)
    missing_cluster = sorted(set(stage6_index) - set(cluster))
    extra_cluster = sorted(set(cluster) - set(stage6_index))

    rows: list[dict[str, Any]] = []
    for key in sorted(common_keys):
        user_id, asin = key
        stage6_query = stage6_index[key]
        cluster_profile = cluster[key]

        # Get noisy query from stage7 if available, otherwise use correct query
        if key in stage7_index:
            stage7_data = stage7_index[key]
            noisy_query = stage7_data["noisy_query"]
            error_pattern = stage7_data["error_pattern"]
        else:
            noisy_query = stage6_query["query"]
            error_pattern = None

        # When error_pattern is empty, noisy_query must also be empty
        if error_pattern is None:
            noisy_query = ""

        rows.append(
            {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "cluster": cluster_profile["cluster"],
                "correct_query": stage6_query["query"],
                "noisy_query": noisy_query,
                "error_pattern": error_pattern,
            }
        )
    return rows, len(missing_cluster), len(extra_cluster)


def build_one_category(paths: DatasetPaths) -> dict[str, Any]:
    stage6_index = load_stage6_query_index(paths.stage6_query_file)
    stage7_index = load_stage7_noisy_index(paths.stage7_noisy_file)
    cluster = load_cluster_profile_index(paths.cluster_profile_file)

    dataset_rows, missing_cluster, extra_cluster = build_dataset_rows(
        paths.category, stage6_index, stage7_index, cluster
    )

    write_jsonl(paths.dataset_file, dataset_rows)

    unique_users = {row["uuid"] for row in dataset_rows}
    unique_user_product_pairs = {(row["uuid"], row["asin"]) for row in dataset_rows}
    total_query_words = sum(len(row["correct_query"].split()) for row in dataset_rows)
    avg_query_words = total_query_words / len(dataset_rows) if dataset_rows else 0
    cluster_counts = Counter(str(row["cluster"]) for row in dataset_rows)

    # Count how many have noisy_query injected
    noisy_injected = sum(1 for r in dataset_rows if r["error_pattern"] is not None)

    summary = {
        "category": paths.category,
        "stage6_query_file": str(paths.stage6_query_file),
        "stage7_noisy_file": str(paths.stage7_noisy_file),
        "cluster_profile_file": str(paths.cluster_profile_file),
        "dataset_file": str(paths.dataset_file),
        "num_stage6_items": len(stage6_index),
        "num_stage7_items": len(stage7_index),
        "num_cluster_profile_items": len(cluster),
        "num_dataset_rows": len(dataset_rows),
        "num_unique_users": len(unique_users),
        "num_unique_user_product_pairs": len(unique_user_product_pairs),
        "total_query_words": total_query_words,
        "avg_query_words": avg_query_words,
        "rows_by_cluster": dict(sorted(cluster_counts.items(), key=lambda item: int(item[0]))),
        "missing_from_cluster": missing_cluster,
        "extra_in_cluster": extra_cluster,
        "noisy_injected": noisy_injected,
    }
    write_json(paths.summary_file, summary)
    return summary


def print_dataset_statistics_table(aggregate_summary: dict[str, Any]) -> None:
    category_summaries = require_list(aggregate_summary.get("category_summaries"), "aggregate category_summaries")
    cluster_counts: Counter[int] = Counter()
    table_rows: list[tuple[str, int | float]] = []

    for raw_summary in category_summaries:
        summary = require_dict(raw_summary, "category summary")
        label = category_label(require_item_text(summary, "category", "category summary"))
        rows_by_cluster = require_dict(summary.get("rows_by_cluster"), f"{label} rows_by_cluster")
        cluster_counts.update({int(key): int(value) for key, value in rows_by_cluster.items()})
        table_rows.extend(
            [
                (f"{label} Total Query", int(summary["num_dataset_rows"])),
                (f"{label} Users", int(summary["num_unique_users"])),
            ]
        )

    table_rows.extend(
        [
            ("Total Category Users", sum(int(summary["num_unique_users"]) for summary in category_summaries)),
            ("Total Query", int(aggregate_summary["num_dataset_rows"])),
        ]
    )
    for cluster in sorted(cluster_counts):
        table_rows.append((f"cluster_{cluster} Rows", cluster_counts[cluster]))

    print("\n[STATISTICS]\n" + format_two_column_stats_table(table_rows), flush=True)


def run_build(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    categories = ensure_supported_categories(args.categories)
    summaries = []

    for category in categories:
        paths = make_dataset_paths(category, output_root)
        summary = build_one_category(paths)
        summaries.append(summary)
        print(f"[{category}] rows={summary['num_dataset_rows']} output={summary['dataset_file']}", flush=True)

    aggregate_summary = {
        "output_root": str(output_root),
        "categories": [summary["category"] for summary in summaries],
        "num_categories": len(summaries),
        "num_dataset_rows": sum(summary["num_dataset_rows"] for summary in summaries),
        "total_query_words": sum(summary["total_query_words"] for summary in summaries),
        "category_summaries": summaries,
    }
    aggregate_summary["avg_query_words"] = aggregate_summary["total_query_words"] / aggregate_summary["num_dataset_rows"]
    write_json(output_root / "summary.json", aggregate_summary)
    print(
        f"[ALL] categories={aggregate_summary['num_categories']} "
        f"rows={aggregate_summary['num_dataset_rows']} "
        f"summary={output_root / 'summary.json'}",
        flush=True,
    )
    print_dataset_statistics_table(aggregate_summary)


def validate_dataset_file(path: Path, counts: Counter[str]) -> None:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            item = json.loads(line)
            if list(item.keys()) != OUTPUT_FIELDS:
                raise ValueError(f"{path}:{lineno} has unexpected field order: {list(item.keys())}")
            forbidden = FORBIDDEN_DATASET_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(f"{path}:{lineno} contains forbidden fields: {sorted(forbidden)}")

            require_text_value(item["category"], f"{path}:{lineno}.category")
            require_text_value(item["uuid"], f"{path}:{lineno}.uuid")
            require_text_value(item["asin"], f"{path}:{lineno}.asin")
            cluster = require_int_value(item["cluster"], f"{path}:{lineno}.cluster")
            if cluster < 0:
                raise ValueError(f"{path}:{lineno}.cluster must be non-negative")
            require_text_value(item["correct_query"], f"{path}:{lineno}.correct_query")
            # error_pattern and noisy_query are optional; when error_pattern
            # is present, it must be a dict and noisy_query must be non-empty.
            error_pattern = item.get("error_pattern")
            if error_pattern is not None:
                require_dict(error_pattern, f"{path}:{lineno}.error_pattern")
                require_text_value(item["noisy_query"], f"{path}:{lineno}.noisy_query")
            counts[f"cluster:{cluster}"] += 1


def run_validate(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    files = sorted(dataset_root.glob("*/data.jsonl"))
    if not files:
        raise FileNotFoundError(f"No dataset JSONL files found under {dataset_root}")
    counts: Counter[str] = Counter()
    for path in files:
        validate_dataset_file(path, counts)
    print(f"[VALID] files={len(files)} counts={dict(sorted(counts.items()))}", flush=True)


def query_sort_key(query: dict[str, Any]) -> tuple[int, str]:
    return (
        query["cluster"],
        query["correct_query"],
    )


def build_grouped_query(row: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, Any] = {
        "cluster": row["cluster"],
        "correct_query": row["correct_query"],
    }
    if row.get("error_pattern") is not None:
        grouped["noisy_query"] = row["noisy_query"]
        grouped["error_pattern"] = row["error_pattern"]
    return grouped


def build_grouped_rows(category: str, flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row_index, row in enumerate(flat_rows):
        row_label = f"{category} data.jsonl row {row_index + 1}"
        user_id = require_item_text(row, "uuid", row_label)
        asin = require_item_text(row, "asin", row_label)
        cluster = require_item_int(row, "cluster", row_label)
        if cluster < 0:
            raise ValueError(f"{row_label}.cluster must be non-negative")

        key = (user_id, asin)
        if key not in grouped:
            grouped[key] = {
                "category": category,
                "uuid": user_id,
                "asin": asin,
                "queries": [],
            }

        grouped[key]["queries"].append(build_grouped_query(row))

    result = []
    for key, grouped_row in grouped.items():
        grouped_row["queries"].sort(key=query_sort_key)
        seen_query_keys = [
            (item["cluster"], item["correct_query"])
            for item in grouped_row["queries"]
        ]
        if len(seen_query_keys) != len(set(seen_query_keys)):
            raise ValueError(f"Duplicate grouped query under key {key}: {seen_query_keys}")
        result.append(grouped_row)
    return result


def export_one_category(dataset_root: Path, output_dir: Path, category: str) -> dict[str, int]:
    data_file = dataset_root / category / "data.jsonl"
    output_file = output_dir / f"{category}_query.json"
    flat_rows = read_jsonl(data_file, f"{category} data.jsonl")
    grouped_rows = build_grouped_rows(category, flat_rows)
    output_file.write_text(json.dumps(grouped_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "flat_rows": len(flat_rows),
        "grouped_rows": len(grouped_rows),
    }


def run_export_root_grouped(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    categories = ensure_supported_categories(args.categories)
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in categories:
        stats = export_one_category(dataset_root, output_dir, category)
        print(f"[EXPORT] {category}: flat_rows={stats['flat_rows']} grouped_rows={stats['grouped_rows']}", flush=True)


def build_statistics_rows(summary: dict[str, Any]) -> list[tuple[str, int | float]]:
    category_summaries = require_list(summary.get("category_summaries"), "summary.category_summaries")
    cluster_counts: Counter[int] = Counter()
    table_rows: list[tuple[str, int | float]] = []

    for summary_index, raw_category_summary in enumerate(category_summaries):
        category_summary = require_dict(raw_category_summary, f"category_summaries[{summary_index}]")
        category = require_item_text(category_summary, "category", f"category_summaries[{summary_index}]")
        label = category_label(category)
        rows_by_cluster = require_dict(
            category_summary.get("rows_by_cluster"),
            f"{label}.rows_by_cluster",
        )
        cluster_counts.update({int(key): int(value) for key, value in rows_by_cluster.items()})

        table_rows.extend(
            [
                (f"{label} Total Query", require_item_int(category_summary, "num_dataset_rows", label)),
                (f"{label} Users", require_item_int(category_summary, "num_unique_users", label)),
            ]
        )

    table_rows.extend(
        [
            ("Total Category Users", sum(require_item_int(item, "num_unique_users", "category summary") for item in category_summaries)),
            ("Total Query", require_item_int(summary, "num_dataset_rows", "summary")),
        ]
    )
    for cluster in sorted(cluster_counts):
        table_rows.append((f"cluster_{cluster} Rows", cluster_counts[cluster]))
    return table_rows


def run_print_stats(args: argparse.Namespace) -> None:
    summary_file = Path(args.summary_file).expanduser().resolve()
    summary = read_json(summary_file, str(summary_file))
    print("[STATISTICS]")
    print(format_two_column_stats_table(build_statistics_rows(summary)))


def run_obsolete_filter_invalid_complexity_queries(_: argparse.Namespace) -> None:
    raise RuntimeError(
        "11_filter_invalid_complexity_queries.py is obsolete for the current cluster-based Stage 11 dataset."
    )


def run_obsolete_sync_stage7(_: argparse.Namespace) -> None:
    raise RuntimeError(
        "11_sync_dataset_with_stage7_revised_queries.py is obsolete for the current cluster-based Stage 11 dataset."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for Stage 11 query dataset workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build Stage 11 datasets from Stage 06, Stage 07, and Stage 12 inputs.")
    build_parser.add_argument("--categories", nargs="+", required=True)
    build_parser.add_argument("--output-root", default=str(DATASET_ROOT))
    build_parser.set_defaults(func=run_build)

    validate_parser = subparsers.add_parser("validate", help="Validate generated Stage 11 dataset JSONL files.")
    validate_parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    validate_parser.set_defaults(func=run_validate)

    export_parser = subparsers.add_parser("export-root-grouped", help="Export grouped root dataset/*_query.json files.")
    export_parser.add_argument("--categories", nargs="+", required=True)
    export_parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    export_parser.add_argument("--output-dir", default=str(ROOT_DATASET_DIR))
    export_parser.set_defaults(func=run_export_root_grouped)

    stats_parser = subparsers.add_parser("print-stats", help="Print statistics from Stage 11 summary.json.")
    stats_parser.add_argument("--summary-file", default=str(DEFAULT_SUMMARY_FILE))
    stats_parser.set_defaults(func=run_print_stats)

    obsolete_filter_parser = subparsers.add_parser(
        "filter-invalid-complexity-queries",
        help="Obsolete command retained only to fail explicitly.",
    )
    obsolete_filter_parser.set_defaults(func=run_obsolete_filter_invalid_complexity_queries)

    obsolete_sync_parser = subparsers.add_parser(
        "sync-dataset-with-stage7-revised-queries",
        help="Obsolete command retained only to fail explicitly.",
    )
    obsolete_sync_parser.set_defaults(func=run_obsolete_sync_stage7)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
