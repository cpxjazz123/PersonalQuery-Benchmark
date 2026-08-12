#!/usr/bin/env python3
"""Shared expression_style noisy query evaluation logic."""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from json import JSONDecoder
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["XDG_CACHE_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/cache"
os.environ["TORCH_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/torch_cache"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/ar57_scratch/wenyu/triton_cache"
for _cache_dir in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
    os.environ["TRITON_CACHE_DIR"],
):
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
RETRIEVAL_ROOT = CURRENT_DIR.parent / "06_retrieval"
PERSONAL_QUERY_ROOT = RETRIEVAL_ROOT.parent
sys.path.insert(0, str(RETRIEVAL_ROOT))
sys.path.insert(0, str(PERSONAL_QUERY_ROOT))

from config import get_category_config, get_global_paths


SYNTAX_DEPTH_QUERY_CATEGORY = "syntax_depth"
CORRECT_QUERY_TYPE = "correct"
NOISY_QUERY_TYPE = "noisy"
NOISY_INJECTION_SOURCE = "final_strict_query_no_depth_constraint"
CORRECT_QUERY_SOURCE_FIELD = "syntax_depth_query.query"
SYNTAX_DEPTH_QUERY_FILENAME = "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
SUMMARY_EXCLUDE_METRIC = "P@10"

DENSE_RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]
COLBERTV2_RETRIEVERS = ["colbertv2"]
SPARSE_RETRIEVERS = ["splade"]

# Skip retriever families if env var is set — matches Stage 06 build + Stage 07
# query cache behavior (those stages never produce cache for skipped families).
# Reviewer note (iter #68): SKIP_SPLADE unblocks Stage 7 eval when SPLADE index
# build has not yet completed (~90 min on 603K docs); without it the eval hard-
# fails on the first FileNotFoundError instead of producing a partial report.
_skip_colbert = os.environ.get("SKIP_COLBERTV2", "").lower() in {"1", "true", "yes", "on"}
if _skip_colbert:
    COLBERTV2_RETRIEVERS = []
_skip_splade = os.environ.get("SKIP_SPLADE", "").lower() in {"1", "true", "yes", "on"}
if _skip_splade:
    SPARSE_RETRIEVERS = []

RETRIEVERS = DENSE_RETRIEVERS + COLBERTV2_RETRIEVERS + SPARSE_RETRIEVERS + ["bm25"]


class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


COLBERTV2_CUDA_HOME = "/usr/local/cuda-12.5"
COLBERTV2_TORCH_EXTENSIONS_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/torch_extensions"
COLBERTV2_HOST_CC = "/usr/bin/gcc"
COLBERTV2_HOST_CXX = "/usr/bin/g++"


from common_utils import log  # 统一 log 函数


def require_field(item: Dict, field: str, context: str):
    if field not in item:
        raise ValueError(f"{context} missing required field '{field}'")
    return item[field]


def require_str(item: Dict, field: str, context: str) -> str:
    value = require_field(item, field, context)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} field '{field}' must be a non-empty string, got {value!r}")
    return value


def require_dict(item: Dict, field: str, context: str) -> Dict:
    value = require_field(item, field, context)
    if not isinstance(value, dict):
        raise TypeError(f"{context} field '{field}' must be dict, got {type(value).__name__}")
    return value


def require_int(item: Dict, field: str, context: str) -> int:
    value = require_field(item, field, context)
    if not isinstance(value, int):
        raise TypeError(f"{context} field '{field}' must be int, got {type(value).__name__}")
    return value


def require_bool(item: Dict, field: str, context: str) -> bool:
    value = require_field(item, field, context)
    if not isinstance(value, bool):
        raise TypeError(f"{context} field '{field}' must be bool, got {type(value).__name__}")
    return value


def require_number(item: Dict, field: str, context: str) -> float:
    value = require_field(item, field, context)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} field '{field}' must be numeric, got {type(value).__name__}")
    return float(value)


def require_list(item: Dict, field: str, context: str) -> List:
    value = require_field(item, field, context)
    if not isinstance(value, list):
        raise TypeError(f"{context} field '{field}' must be list, got {type(value).__name__}")
    return value


def require_str_or_str_list(item: Dict, field: str, context: str):
    value = require_field(item, field, context)
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, list):
        raise TypeError(f"{context} field '{field}' must be str or list[str], got {type(value).__name__}")
    if not value:
        raise ValueError(f"{context} field '{field}' must not be an empty list")
    for item_index, item_value in enumerate(value):
        if not isinstance(item_value, str) or not item_value:
            raise ValueError(
                f"{context} field '{field}' item {item_index} must be a non-empty string, got {item_value!r}"
            )
    return value


def load_appended_json_objects(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"JSON records file is empty: {file_path}")

    decoder = JSONDecoder()
    records = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break

        decoded, end = decoder.raw_decode(content, index)
        if isinstance(decoded, list):
            if records:
                raise ValueError(f"{file_path} contains JSON records before a JSON array")
            if content[end:].strip():
                raise ValueError(f"{file_path} contains extra content after a JSON array")
            for row_index, row in enumerate(decoded):
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{file_path} JSON array row {row_index} must be dict, got {type(row).__name__}"
                    )
            return decoded
        if not isinstance(decoded, dict):
            raise TypeError(f"{file_path} appended JSON record must be dict, got {type(decoded).__name__}")
        records.append(decoded)
        index = end

    if not records:
        raise ValueError(f"No JSON records parsed from {file_path}")
    return records


def load_syntax_depth_metadata(syntax_depth_query_file: str) -> Dict[Tuple[str, str, str], Dict]:
    with open(syntax_depth_query_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"expression_style query file must contain a list, got {type(data).__name__}: {syntax_depth_query_file}"
        )

    metadata = {}
    for row_index, item in enumerate(data):
        context = f"expression_style row {row_index}"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be dict, got {type(item).__name__}")
        user_id = require_str(item, "user_id", context)
        asin = require_str(item, "asin", context)
        syntax_query = require_dict(item, "syntax_depth_query", context)

        query_text = require_str(syntax_query, "query", context)
        word_count = require_int(syntax_query, "word_count", context)

        attrs_used = syntax_query.get("attrs_used")
        if attrs_used is not None and not isinstance(attrs_used, dict):
            raise TypeError(f"{context} attrs_used must be dict when present, got {type(attrs_used).__name__}")

        key = (user_id, asin, query_text)
        if key in metadata:
            raise ValueError(f"Duplicate expression_style query key: user={user_id}, asin={asin}, query={query_text}")
        metadata[key] = {
            "user_id": user_id,
            "asin": asin,
            "source_query": query_text,
            "word_count": word_count,
            "attrs_used": attrs_used,
            "syntax_depth_row_index": row_index,
        }

    return metadata


def load_noisy_pairs(category_name: str, noisy_query_file: str, syntax_depth_query_file: str) -> List[Dict]:
    """Load noisy query pairs from the new flat noisy_query.json format.

    The noisy_query_file contains records with flat structure:
    - uid, asin, clean_query, noisy_query, query_rewritten, etc.

    The syntax_depth_query_file is no longer used for loading pairs in this format.
    """
    with open(noisy_query_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"noisy query file must contain a list, got {type(data).__name__}: {noisy_query_file}"
        )

    pairs = []
    skipped_invalid_record = 0
    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"noisy query row {row_index} must be dict, got {type(item).__name__}")
        context = f"noisy query row {row_index}"

        user_id = require_str(item, "uid", context)
        asin = require_str(item, "asin", context)
        clean_query = require_str(item, "clean_query", context)
        noisy_query = require_str(item, "noisy_query", context)
        query_rewritten = require_bool(item, "query_rewritten", context)

        applied_error = item.get("applied_error")
        if applied_error is not None and not isinstance(applied_error, dict):
            raise TypeError(f"{context} applied_error must be dict when present, got {type(applied_error).__name__}")

        pair = {
            "pair_id": len(pairs),
            "source_noisy_record_index": row_index,
            "category": category_name,
            "user_id": user_id,
            "asin": asin,
            "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
            "correct_query": clean_query,
            "correct_query_source_field": CORRECT_QUERY_SOURCE_FIELD,
            "source_query": clean_query,
            "ground_truth_query": clean_query,
            "noisy_query": noisy_query,
            "original_query": clean_query,
            "query_rewritten": query_rewritten,
            "injection_mode": "lambdamart_token",
            "injection_source": NOISY_INJECTION_SOURCE,
            "word_count": len(clean_query.split()),
            "attrs_used": {
                "query_attrs_used": None,
                "noise_type": None,
                "correct_text": applied_error.get("corrected") if applied_error else None,
                "noisy_text": applied_error.get("original") if applied_error else None,
                "anchor_replaced_text": None,
            },
        }
        pairs.append(pair)

    if not pairs:
        raise ValueError(f"No noisy records found in {noisy_query_file}")

    rewritten_count = sum(1 for pair in pairs if pair["query_rewritten"] is True)
    log(
        f"加载 noisy expression_style 配对: {len(pairs)} 条，"
        f"跳过无效记录 {skipped_invalid_record} 条，"
        f"其中 query_rewritten=True 为 {rewritten_count} 条"
    )
    log("Correct 侧使用 clean_query；Noisy 侧使用 noisy_query")
    return pairs


def get_query_cache_path(query_cache_base_dir: str, retriever_name: str, query_type: str) -> str:
    if query_type not in (CORRECT_QUERY_TYPE, NOISY_QUERY_TYPE):
        raise ValueError(f"Unsupported query_type: {query_type}")
    return os.path.join(
        query_cache_base_dir,
        f"{SYNTAX_DEPTH_QUERY_CATEGORY}_{query_type}_query",
        f"{retriever_name}__{SYNTAX_DEPTH_QUERY_CATEGORY}_{query_type}_cache.pkl",
    )


def load_query_cache(query_cache_base_dir: str, retriever_name: str, query_type: str) -> Dict:
    cache_path = get_query_cache_path(query_cache_base_dir, retriever_name, query_type)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{retriever_name} {query_type} query cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = NumpyCoreCompatUnpickler(f).load()
    if not isinstance(cache, dict):
        raise TypeError(f"{retriever_name} {query_type} query cache must be dict, got {type(cache).__name__}")
    return cache


def load_dense_retriever_cache(cache_dir: str, retriever_name: str) -> Tuple[np.ndarray, List[str]]:
    embeddings_path = None
    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith(f"{retriever_name}_") and filename.endswith("_embeddings.npy"):
            embeddings_path = os.path.join(cache_dir, filename)
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} document embeddings not found in {cache_dir}")

    embeddings = np.load(embeddings_path, mmap_mode="r")[:].copy()
    doc_ids_path = embeddings_path.replace("_embeddings.npy", "_doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"{retriever_name} doc id cache not found: {doc_ids_path}")
    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)
    if not isinstance(doc_ids, list):
        raise TypeError(f"{retriever_name} doc ids must be list, got {type(doc_ids).__name__}")
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            f"{retriever_name} document cache mismatch: embeddings={embeddings.shape[0]}, doc_ids={len(doc_ids)}"
        )
    return embeddings, doc_ids


def load_splade_retriever(cache_dir: str):
    splade_path = None
    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith("splade_") and filename.endswith(".pkl"):
            splade_path = os.path.join(cache_dir, filename)
            break
    if splade_path is None:
        raise FileNotFoundError(f"SPLADE retriever cache not found in {cache_dir}")
    with open(splade_path, "rb") as f:
        retriever = pickle.load(f)
    return retriever


def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f"P@{k}"] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f"N@{k}"] = 1.0 / np.log2(rank + 1)
            metrics[f"MR@{k}"] = 1.0 / rank
        else:
            metrics[f"N@{k}"] = 0.0
            metrics[f"MR@{k}"] = 0.0
        metrics[f"H@{k}"] = 1.0 if relevant_asin in top_k else 0.0
    return metrics


def compute_average_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    if not all_metrics:
        raise ValueError("Cannot compute average metrics for an empty metric list")
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f"P@{k}"] = float(np.mean([metric[f"P@{k}"] for metric in all_metrics]))
        avg_metrics[f"N@{k}"] = float(np.mean([metric[f"N@{k}"] for metric in all_metrics]))
        avg_metrics[f"MR@{k}"] = float(np.mean([metric[f"MR@{k}"] for metric in all_metrics]))
        avg_metrics[f"H@{k}"] = float(np.mean([metric[f"H@{k}"] for metric in all_metrics]))
    return avg_metrics


def query_record_for_output(pair: Dict, query_type: str, metric: Dict) -> Dict:
    if query_type == CORRECT_QUERY_TYPE:
        query_text = pair["correct_query"]
    elif query_type == NOISY_QUERY_TYPE:
        query_text = pair["noisy_query"]
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")

    return {
        "pair_id": pair["pair_id"],
        "user_id": pair["user_id"],
        "asin": pair["asin"],
        "query_type": query_type,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "query": query_text,
        "correct_query": pair["correct_query"],
        "correct_query_source_field": pair["correct_query_source_field"],
        "source_query": pair["source_query"],
        "ground_truth_query": pair["ground_truth_query"],
        "noisy_query": pair["noisy_query"],
        "query_rewritten": pair["query_rewritten"],
        "injection_mode": pair["injection_mode"],
        "word_count": pair["word_count"],
        "attrs_used": pair["attrs_used"],
        "p_at10": metric.get("P@10", 0.0),
        "metrics": dict(metric),
    }


def build_retriever_result_from_query_records(
    retriever_name: str,
    query_type: str,
    query_records: List[Dict],
    k_values: List[int],
    cache_summary: Dict,
) -> Dict:
    if not query_records:
        raise ValueError(f"{retriever_name} {query_type} produced no query records")

    all_metrics = []
    for record_index, record in enumerate(query_records):
        if "metrics" not in record:
            raise ValueError(f"{retriever_name} {query_type} query record {record_index} missing metrics")
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError(
                f"{retriever_name} {query_type} query record {record_index} metrics must be dict, got {type(metrics).__name__}"
            )
        all_metrics.append(metrics)

    return {
        "retriever": retriever_name,
        "query_type": query_type,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "num_queries": len(query_records),
        "metrics": compute_average_metrics(all_metrics, k_values),
        "cache_summary": cache_summary,
        "all_query_records": query_records,
    }


def build_retriever_result(
    retriever_name: str,
    query_type: str,
    all_metrics: List[Dict],
    records: List[Dict],
    k_values: List[int],
    cache_summary: Dict,
) -> Dict:
    if len(all_metrics) != len(records):
        raise ValueError(
            f"{retriever_name} {query_type} metric/record length mismatch: {len(all_metrics)} vs {len(records)}"
        )
    if not all_metrics:
        raise ValueError(f"{retriever_name} {query_type} produced no metrics")

    query_records = [
        query_record_for_output(pair, query_type, metric)
        for pair, metric in zip(records, all_metrics)
    ]
    return build_retriever_result_from_query_records(
        retriever_name,
        query_type,
        query_records,
        k_values,
        cache_summary,
    )


def filter_retriever_pair_results(
    correct_result: Dict,
    noisy_result: Dict,
    k_values: List[int],
    exclude_metric_name: str = SUMMARY_EXCLUDE_METRIC,
) -> Tuple[Dict, Dict, Dict]:
    correct_records = correct_result["all_query_records"]
    noisy_records = noisy_result["all_query_records"]
    if len(correct_records) != len(noisy_records):
        raise ValueError(
            f"{correct_result['retriever']} query record length mismatch: "
            f"{len(correct_records)} vs {len(noisy_records)}"
        )

    filtered_correct_records = []
    filtered_noisy_records = []
    excluded_records = []
    for record_index, (correct_record, noisy_record) in enumerate(zip(correct_records, noisy_records)):
        if correct_record["pair_id"] != noisy_record["pair_id"]:
            raise ValueError(
                f"{correct_result['retriever']} pair_id mismatch at index {record_index}: "
                f"{correct_record['pair_id']} vs {noisy_record['pair_id']}"
            )
        if exclude_metric_name not in correct_record["metrics"]:
            raise KeyError(
                f"{correct_result['retriever']} correct record pair_id={correct_record['pair_id']} missing metric {exclude_metric_name}"
            )
        if exclude_metric_name not in noisy_record["metrics"]:
            raise KeyError(
                f"{correct_result['retriever']} noisy record pair_id={noisy_record['pair_id']} missing metric {exclude_metric_name}"
            )

        correct_value = correct_record["metrics"][exclude_metric_name]
        noisy_value = noisy_record["metrics"][exclude_metric_name]
        if noisy_value >= correct_value:
            excluded_records.append(
                {
                    "pair_id": correct_record["pair_id"],
                    "user_id": correct_record["user_id"],
                    "asin": correct_record["asin"],
                    "query": noisy_record["query"],
                    "correct_query": correct_record["query"],
                    "noisy_query": noisy_record["query"],
                    "correct_value": correct_value,
                    "noisy_value": noisy_value,
                }
            )
            continue

        filtered_correct_records.append(correct_record)
        filtered_noisy_records.append(noisy_record)

    if not filtered_correct_records:
        # Reviewer note (iter #68): when *every* (clean, noisy) pair has
        # noisy_value >= correct_value, the retriever is "noise-tolerant" on this
        # slice — it has nothing to contribute to Δ Range / BIC / AIC. Do NOT
        # crash the whole eval (that would discard the other retrievers' results);
        # record an empty filter_summary so downstream can iterate it consistently.
        log(
            f"{correct_result['retriever']} 所有样本均为 noisy 不差于 clean，跳过过滤（保留空集）"
        )
        empty_filter_summary = {
            "retriever": correct_result["retriever"],
            "exclude_metric_name": exclude_metric_name,
            "kept_count": 0,
            "excluded_count": len(correct_records),
            "excluded_pair_ids": [
                excluded["pair_id"] for excluded in excluded_records
            ],
            "all_records_excluded": True,
        }
        return (
            {
                **correct_result,
                "all_query_records": [],
                "final_statistics_filter": empty_filter_summary,
            },
            {
                **noisy_result,
                "all_query_records": [],
                "final_statistics_filter": empty_filter_summary,
            },
            empty_filter_summary,
        )

    filtered_correct_result = build_retriever_result_from_query_records(
        correct_result["retriever"],
        correct_result["query_type"],
        filtered_correct_records,
        k_values,
        correct_result["cache_summary"],
    )
    filtered_noisy_result = build_retriever_result_from_query_records(
        noisy_result["retriever"],
        noisy_result["query_type"],
        filtered_noisy_records,
        k_values,
        noisy_result["cache_summary"],
    )

    filter_summary = {
        "retriever": correct_result["retriever"],
        "exclude_metric_name": exclude_metric_name,
        "excluded_count": len(excluded_records),
        "kept_count": len(filtered_correct_records),
        "original_count": len(correct_records),
        "excluded_records": excluded_records,
    }
    filtered_correct_result["final_statistics_filter"] = filter_summary
    filtered_noisy_result["final_statistics_filter"] = filter_summary
    return filtered_correct_result, filtered_noisy_result, filter_summary


def select_pairs_for_user_query_caches(
    pairs: List[Dict],
    correct_cache: Dict,
    noisy_cache: Dict,
) -> Tuple[List[Dict], List, List, Dict]:
    selected_pairs = []
    correct_values = []
    noisy_values = []
    missing_correct_user = 0
    missing_correct_query = 0
    missing_noisy_user = 0
    missing_noisy_query = 0

    for pair in pairs:
        user_id = pair["user_id"]
        correct_query = pair["correct_query"]
        noisy_query = pair["noisy_query"]

        if user_id not in correct_cache:
            missing_correct_user += 1
            continue
        if user_id not in noisy_cache:
            missing_noisy_user += 1
            continue

        correct_user_cache = correct_cache[user_id]
        noisy_user_cache = noisy_cache[user_id]
        if not isinstance(correct_user_cache, dict):
            raise TypeError(f"correct cache for user={user_id} must be dict")
        if not isinstance(noisy_user_cache, dict):
            raise TypeError(f"noisy cache for user={user_id} must be dict")

        if correct_query not in correct_user_cache:
            missing_correct_query += 1
            continue
        if noisy_query not in noisy_user_cache:
            missing_noisy_query += 1
            continue

        selected_pairs.append(pair)
        correct_values.append(correct_user_cache[correct_query])
        noisy_values.append(noisy_user_cache[noisy_query])

    summary = {
        "candidate_pairs": len(pairs),
        "selected_pairs": len(selected_pairs),
        "missing_correct_user": missing_correct_user,
        "missing_correct_query": missing_correct_query,
        "missing_noisy_user": missing_noisy_user,
        "missing_noisy_query": missing_noisy_query,
    }
    if not selected_pairs:
        raise ValueError(f"No pairs have both correct and noisy user-query cache entries: {summary}")
    return selected_pairs, correct_values, noisy_values, summary


def select_pairs_for_bm25_caches(
    pairs: List[Dict],
    correct_cache: Dict,
    noisy_cache: Dict,
) -> Tuple[List[Dict], List, List, Dict]:
    selected_pairs = []
    correct_results = []
    noisy_results = []
    missing_correct_query = 0
    missing_noisy_query = 0

    for pair in pairs:
        correct_query = pair["correct_query"]
        noisy_query = pair["noisy_query"]
        if correct_query not in correct_cache:
            missing_correct_query += 1
            continue
        if noisy_query not in noisy_cache:
            missing_noisy_query += 1
            continue
        selected_pairs.append(pair)
        correct_results.append(correct_cache[correct_query])
        noisy_results.append(noisy_cache[noisy_query])

    summary = {
        "candidate_pairs": len(pairs),
        "selected_pairs": len(selected_pairs),
        "missing_correct_query": missing_correct_query,
        "missing_noisy_query": missing_noisy_query,
    }
    if not selected_pairs:
        raise ValueError(f"No pairs have both correct and noisy BM25 cache entries: {summary}")
    return selected_pairs, correct_results, noisy_results, summary


class DenseSearcher:
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str]):
        self.doc_ids = doc_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            raise ValueError("Dense search received no query embeddings")
        query_array = np.array(query_embeddings)
        query_tensor = torch.from_numpy(query_array).float().to(self.device)
        q_norms = np.linalg.norm(query_array, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
        scores = torch.mm(query_tensor, self.embeddings_tensor.T)

        results = []
        for query_index in range(len(query_embeddings)):
            top_scores, top_indices = torch.topk(scores[query_index], min(top_k, len(self.doc_ids)))
            results.append(
                [
                    (self.doc_ids[idx.item()], float(top_scores[pos].item()))
                    for pos, idx in enumerate(top_indices)
                ]
            )
        return results


def results_to_metrics(results: List[List[Tuple[str, float]]], pairs: List[Dict], k_values: List[int]) -> List[Dict]:
    if len(results) != len(pairs):
        raise ValueError(f"Result/pair length mismatch: {len(results)} vs {len(pairs)}")
    metrics = []
    for retrieved, pair in zip(results, pairs):
        retrieved_asins = [asin for asin, _ in retrieved]
        metrics.append(compute_metrics(pair["asin"], retrieved_asins, k_values))
    return metrics


def evaluate_dense_pair(
    category_config: Dict,
    retriever_name: str,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    log(f"\n{'=' * 60}")
    log(f"评估 {retriever_name.upper()} - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_embeddings, noisy_embeddings, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    embeddings, doc_ids = load_dense_retriever_cache(category_config["retriever_cache_dir"], retriever_name)
    searcher = DenseSearcher(embeddings, doc_ids)

    correct_results = searcher.search_batch(correct_embeddings, top_k=max(k_values))
    noisy_results = searcher.search_batch(noisy_embeddings, top_k=max(k_values))
    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)

    del embeddings
    del searcher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def evaluate_bm25_pair(
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "bm25"
    log(f"\n{'=' * 60}")
    log("评估 BM25 - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_results, noisy_results, cache_summary = select_pairs_for_bm25_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)
    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def splade_search_from_vectors(retriever, query_vectors: List[Dict], top_k: int) -> List[List[Tuple[str, float]]]:
    from scipy import sparse

    inverted_index = retriever._inverted_index
    if not inverted_index:
        raise ValueError("SPLADE inverted index is empty")
    n_docs = len(retriever.doc_ids)

    doc_rows = []
    doc_cols = []
    doc_data = []
    max_doc_term = 0
    for term_id, doc_list in inverted_index.items():
        term_id_int = int(term_id)
        max_doc_term = max(max_doc_term, term_id_int)
        for doc_idx, d_weight in doc_list:
            doc_rows.append(term_id_int)
            doc_cols.append(doc_idx)
            doc_data.append(d_weight)

    q_rows = []
    q_cols = []
    q_data = []
    max_query_term = 0
    for q_idx, q_vec in enumerate(query_vectors):
        if not isinstance(q_vec, dict):
            raise TypeError(f"SPLADE query vector must be dict, got {type(q_vec).__name__}")
        for term_id, q_weight in q_vec.items():
            term_id_int = int(term_id)
            max_query_term = max(max_query_term, term_id_int)
            q_rows.append(q_idx)
            q_cols.append(term_id_int)
            q_data.append(q_weight)

    n_terms = max(max_doc_term, max_query_term) + 1
    doc_matrix = sparse.csr_matrix(
        (doc_data, (doc_rows, doc_cols)),
        shape=(n_terms, n_docs),
        dtype=np.float32,
    )
    query_matrix = sparse.csr_matrix(
        (q_data, (q_rows, q_cols)),
        shape=(len(query_vectors), n_terms),
        dtype=np.float32,
    )

    score_matrix = query_matrix @ doc_matrix
    results = []
    for row_index in range(score_matrix.shape[0]):
        row = score_matrix.getrow(row_index)
        scores_vec = row.toarray().flatten()
        top_indices = np.argsort(scores_vec)[::-1][:top_k]
        results.append([(retriever.doc_ids[idx], float(scores_vec[idx])) for idx in top_indices])
    return results


def evaluate_splade_pair(
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "splade"
    log(f"\n{'=' * 60}")
    log("评估 SPLADE - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_vectors, noisy_vectors, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    retriever = load_splade_retriever(category_config["retriever_cache_dir"])
    retriever.search(["dummy"], top_k=1)

    correct_results = splade_search_from_vectors(retriever, correct_vectors, max(k_values))
    noisy_results = splade_search_from_vectors(retriever, noisy_vectors, max(k_values))
    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)

    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def validate_colbertv2_paths() -> None:
    required_paths = [
        os.path.join(COLBERTV2_CUDA_HOME, "include", "cuda_runtime.h"),
        os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc"),
        COLBERTV2_HOST_CC,
        COLBERTV2_HOST_CXX,
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required ColBERTv2 runtime path not found: {path}")


def configure_colbertv2_env(category_name: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 evaluation requires CUDA")
    validate_colbertv2_paths()

    os.environ["CUDA_HOME"] = COLBERTV2_CUDA_HOME
    os.environ["CUDA_PATH"] = COLBERTV2_CUDA_HOME
    os.environ["CUDACXX"] = os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")
    os.environ["CC"] = COLBERTV2_HOST_CC
    os.environ["CXX"] = COLBERTV2_HOST_CXX
    os.environ["CUDAHOSTCXX"] = COLBERTV2_HOST_CXX
    for env_name, env_value in [
        ("PATH", os.path.join(COLBERTV2_CUDA_HOME, "bin")),
        ("CPATH", os.path.join(COLBERTV2_CUDA_HOME, "include")),
        ("LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
        ("LD_LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
    ]:
        existing_env_value = os.environ.get(env_name)
        os.environ[env_name] = env_value if not existing_env_value else f"{env_value}:{existing_env_value}"

    major, minor = torch.cuda.get_device_capability()
    ext_dir = os.path.join(
        COLBERTV2_TORCH_EXTENSIONS_BASE_DIR,
        f"colbertv2_cuda125_sm{major}{minor}",
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log(f"[ColBERT] shared TORCH_EXTENSIONS_DIR = {ext_dir}")
    return ext_dir


def resolve_colbertv2_output_root(category_name: str, cache_dir: str) -> str:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Retriever cache directory not found: {cache_dir}")

    candidates = []
    for name in sorted(os.listdir(cache_dir)):
        output_root = os.path.join(cache_dir, name)
        if not name.startswith("colbertv2_") or not os.path.isdir(output_root):
            continue
        manifest_path = os.path.join(output_root, "build_manifest.json")
        doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
        index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
        if os.path.isfile(manifest_path) and os.path.isfile(doc_ids_path) and os.path.isdir(index_dir):
            candidates.append(output_root)

    if not candidates:
        raise FileNotFoundError(f"No complete ColBERTv2 cache directory found under: {cache_dir}")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one ColBERTv2 cache directory, found {len(candidates)}: {candidates}")

    output_root = candidates[0]
    manifest_path = os.path.join(output_root, "build_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise TypeError(f"ColBERTv2 build manifest must be dict, got {type(manifest).__name__}: {manifest_path}")
    for required_field in ("category", "output_root"):
        if required_field not in manifest:
            raise ValueError(f"ColBERTv2 build manifest missing required field '{required_field}': {manifest_path}")
    if manifest["category"] != category_name:
        raise ValueError(f"ColBERTv2 cache category mismatch in {manifest_path}: {manifest['category']}")
    if manifest["output_root"] != output_root:
        raise ValueError(f"ColBERTv2 cache output_root mismatch in {manifest_path}: {manifest['output_root']}")

    return output_root


def load_colbertv2_doc_ids(output_root: str) -> List[str]:
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Required ColBERTv2 doc id mapping not found: {doc_ids_path}")
    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)
    if not isinstance(doc_ids, list):
        raise TypeError(f"ColBERTv2 doc_ids must be list, got {type(doc_ids).__name__}")
    if not doc_ids:
        raise ValueError(f"ColBERTv2 doc_ids is empty: {doc_ids_path}")
    return doc_ids


def resolve_local_hf_snapshot(repo_id: str) -> str:
    repo_cache_dir = os.path.join(os.environ["HF_HOME"], "models--" + repo_id.replace("/", "--"))
    ref_file = os.path.join(repo_cache_dir, "refs", "main")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Hugging Face ref file not found for {repo_id}: {ref_file}")
    with open(ref_file, "r", encoding="utf-8") as f:
        snapshot_hash = f.read().strip()
    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", snapshot_hash)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Hugging Face snapshot not found for {repo_id}: {snapshot_dir}")
    return snapshot_dir


def build_colbertv2_searcher(category_name: str, output_root: str, doc_ids: List[str]):
    configure_colbertv2_env(category_name)

    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert import Searcher

    checkpoint_path = resolve_local_hf_snapshot("colbert-ir/colbertv2.0")
    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    log(f"[ColBERT] checkpoint_path = {checkpoint_path}")
    log(f"[ColBERT] output_root = {output_root}")

    start = time.time()
    with Run().context(RunConfig(experiment="colbertv2_index", root=output_root)):
        config = ColBERTConfig(root=output_root)
        searcher = Searcher(
            index="colbertv2_index",
            checkpoint=checkpoint_path,
            collection=collection,
            config=config,
        )
    log(f"[ColBERT] Searcher loaded in {time.time() - start:.1f}s")
    return searcher


def colbertv2_search_from_cached_embedding(searcher, doc_ids: List[str], query_embedding, top_k: int):
    if not isinstance(query_embedding, np.ndarray):
        raise TypeError(f"ColBERTv2 cached query embedding must be numpy.ndarray, got {type(query_embedding).__name__}")
    if query_embedding.ndim != 2:
        raise ValueError(f"ColBERTv2 cached query embedding must be 2D, got shape {query_embedding.shape}")

    query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
    pids, _, scores = searcher.dense_search(query_tensor, k=top_k)

    results = []
    for pid, score in zip(pids, scores):
        pid_int = int(pid)
        if pid_int < 0 or pid_int >= len(doc_ids):
            raise IndexError(f"ColBERTv2 pid {pid_int} is outside doc_ids range 0..{len(doc_ids) - 1}")
        results.append((doc_ids[pid_int], float(score)))
    return results


def evaluate_colbertv2_pair(
    category_name: str,
    category_config: Dict,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    retriever_name = "colbertv2"
    log(f"\n{'=' * 60}")
    log("评估 COLBERTV2 - syntax_depth correct/noisy")
    log(f"{'=' * 60}")

    query_cache_base_dir = category_config["query_cache_dir"]
    correct_cache = load_query_cache(query_cache_base_dir, retriever_name, CORRECT_QUERY_TYPE)
    noisy_cache = load_query_cache(query_cache_base_dir, retriever_name, NOISY_QUERY_TYPE)
    selected_pairs, correct_embeddings, noisy_embeddings, cache_summary = select_pairs_for_user_query_caches(
        pairs,
        correct_cache,
        noisy_cache,
    )
    log(f"  缓存配对命中: {cache_summary}")

    output_root = resolve_colbertv2_output_root(category_name, category_config["retriever_cache_dir"])
    index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")
    doc_ids = load_colbertv2_doc_ids(output_root)
    searcher = build_colbertv2_searcher(category_name, output_root, doc_ids)

    correct_results = []
    noisy_results = []
    for index, (correct_embedding, noisy_embedding) in enumerate(zip(correct_embeddings, noisy_embeddings)):
        correct_results.append(
            colbertv2_search_from_cached_embedding(searcher, doc_ids, correct_embedding, max(k_values))
        )
        noisy_results.append(
            colbertv2_search_from_cached_embedding(searcher, doc_ids, noisy_embedding, max(k_values))
        )
        if (index + 1) % 100 == 0 or index + 1 == len(correct_embeddings):
            log(f"    ColBERTv2 搜索进度: {index + 1}/{len(correct_embeddings)}")

    correct_metrics = results_to_metrics(correct_results, selected_pairs, k_values)
    noisy_metrics = results_to_metrics(noisy_results, selected_pairs, k_values)
    return (
        build_retriever_result(retriever_name, CORRECT_QUERY_TYPE, correct_metrics, selected_pairs, k_values, cache_summary),
        build_retriever_result(retriever_name, NOISY_QUERY_TYPE, noisy_metrics, selected_pairs, k_values, cache_summary),
    )


def metric_diff(correct_result: Dict, noisy_result: Dict) -> Dict:
    metrics = {}
    for metric_name, correct_value in correct_result["metrics"].items():
        metrics[metric_name] = float(noisy_result["metrics"][metric_name] - correct_value)

    return {
        "retriever": correct_result["retriever"],
        "num_queries": correct_result["num_queries"],
        "metrics_noisy_minus_correct": metrics,
    }


def print_results_table(results: List[Dict], title: str) -> None:
    log(f"\n{'=' * 100}")
    log(title)
    log("=" * 100)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    header = f"{'检索器':<12} {'N':>8}"
    for metric_name in metrics_to_show:
        header += f" {metric_name:>10}"
    log(header)
    log("-" * 100)
    for result in results:
        row = f"{result['retriever']:<12} {result['num_queries']:>8}"
        for metric_name in metrics_to_show:
            row += f" {result['metrics'][metric_name]:>10.4f}"
        log(row)
    log("-" * 100)


def print_difference_table(differences: List[Dict]) -> None:
    log(f"\n{'=' * 120}")
    log("CORRECT vs NOISY 差异分析（NOISY - CORRECT）")
    log("=" * 120)
    metrics_to_show = ["P@1", "P@3", "P@5", "P@10", "N@10", "MR@10", "H@10"]
    header = f"{'检索器':<12} {'N':>8}"
    for metric_name in metrics_to_show:
        header += f" {metric_name:>10}"
    log(header)
    log("-" * 120)
    for diff in differences:
        row = f"{diff['retriever']:<12} {diff['num_queries']:>8}"
        for metric_name in metrics_to_show:
            value = diff["metrics_noisy_minus_correct"][metric_name]
            sign = "+" if value > 0 else ""
            row += f" {sign:>1}{value:>9.4f}"
        log(row)
    log("-" * 120)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_retriever_pair(
    category_name: str,
    category_config: Dict,
    retriever_name: str,
    pairs: List[Dict],
    k_values: List[int],
) -> Tuple[Dict, Dict]:
    if retriever_name in DENSE_RETRIEVERS:
        return evaluate_dense_pair(category_config, retriever_name, pairs, k_values)
    if retriever_name == "bm25":
        return evaluate_bm25_pair(category_config, pairs, k_values)
    if retriever_name == "splade":
        return evaluate_splade_pair(category_config, pairs, k_values)
    if retriever_name == "colbertv2":
        return evaluate_colbertv2_pair(category_name, category_config, pairs, k_values)
    raise ValueError(f"Unsupported retriever: {retriever_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate expression_style noisy queries against 08-style caches")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=RETRIEVERS,
        default=RETRIEVERS,
        help="Retrievers to evaluate",
    )
    return parser.parse_args()


def run_category_eval(category_name: str) -> None:
    args = parse_args()
    category_config = get_category_config(category_name)
    for required_key in ("retriever_cache_dir", "query_cache_dir", "corpus_file"):
        if required_key not in category_config:
            raise KeyError(f"Category config for {category_name} missing required key: {required_key}")

    global_paths = get_global_paths()
    if "inject_noisy" not in global_paths:
        raise KeyError("Global paths missing required key: inject_noisy")

    syntax_depth_query_file = (
        f"/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{category_name}/"
        f"{SYNTAX_DEPTH_QUERY_FILENAME}"
    )
    noisy_query_file = os.path.join(global_paths["inject_noisy"], category_name, "noisy_query.json")
    output_root = os.path.join(str(Path(global_paths["inject_noisy"]).parent), "09_noisy_retrieval", category_name)
    os.makedirs(output_root, exist_ok=True)

    log("=" * 80)
    log(f"Syntax-depth noisy query evaluation - {category_name}")
    log("=" * 80)
    log(f"06 expression_style query file: {syntax_depth_query_file}")
    log(f"07 noisy query file (legacy reference only): {noisy_query_file}")
    log(f"Query cache dir: {category_config['query_cache_dir']}")
    log(f"Retriever cache dir: {category_config['retriever_cache_dir']}")
    log(f"Retrievers: {', '.join(args.retrievers)}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]
    pairs = load_noisy_pairs(category_name, noisy_query_file, syntax_depth_query_file)

    correct_results = []
    noisy_results = []
    differences = []
    for retriever_name in args.retrievers:
        correct_result, noisy_result = evaluate_retriever_pair(
            category_name,
            category_config,
            retriever_name,
            pairs,
            k_values,
        )
        correct_results.append(correct_result)
        noisy_results.append(noisy_result)
        differences.append(metric_diff(correct_result, noisy_result))

    filtered_correct_results = []
    filtered_noisy_results = []
    filtered_differences = []
    final_statistics_filters = []
    for correct_result, noisy_result in zip(correct_results, noisy_results):
        filtered_correct_result, filtered_noisy_result, filter_summary = filter_retriever_pair_results(
            correct_result,
            noisy_result,
            k_values,
        )
        filtered_correct_results.append(filtered_correct_result)
        filtered_noisy_results.append(filtered_noisy_result)
        filtered_differences.append(metric_diff(filtered_correct_result, filtered_noisy_result))
        final_statistics_filters.append(filter_summary)
        log(
            f"{correct_result['retriever']} 最终统计剔除 noisy 不差于 clean 的样本 "
            f"{filter_summary['excluded_count']} 条，保留 {filter_summary['kept_count']} 条"
        )

    print_results_table(filtered_correct_results, "CORRECT 检索结果")
    print_results_table(filtered_noisy_results, "NOISY 检索结果")
    print_difference_table(filtered_differences)

    output_file = os.path.join(output_root, "syntax_depth_correct_vs_noisy_results.json")
    results_to_save = {
        "timestamp": datetime.now().isoformat(),
        "category": category_name,
        "query_category": SYNTAX_DEPTH_QUERY_CATEGORY,
        "noisy_injection_source": NOISY_INJECTION_SOURCE,
        "correct_query_source_field": CORRECT_QUERY_SOURCE_FIELD,
        "syntax_depth_query_file": syntax_depth_query_file,
        "noisy_query_file": noisy_query_file,
        "query_cache_dir": category_config["query_cache_dir"],
        "retriever_cache_dir": category_config["retriever_cache_dir"],
        "retrievers": args.retrievers,
        "k_values": k_values,
        "num_noisy_pairs_loaded": len(pairs),
        "raw_correct_results": correct_results,
        "raw_noisy_results": noisy_results,
        "raw_differences": differences,
        "correct_results": filtered_correct_results,
        "noisy_results": filtered_noisy_results,
        "differences": filtered_differences,
        "final_statistics_filters": final_statistics_filters,
    }
    # iter #79: explicit invariant — all_query_records must be a list at JSON
    # write time. Previous runs accidentally serialized this field as an int
    # (lost per-query Δ data), so downstream bootstrap CI failed silently.
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
                    f"write time, got {type(rec_field).__name__}; downstream bootstrap Δ CI "
                    f"(iter #78) requires per-query records. Re-evaluate the pair with the "
                    f"fixed code path."
                )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results_to_save), f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {output_file}")
    log("评估完成")
