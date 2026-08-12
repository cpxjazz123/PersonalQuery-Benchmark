#!/usr/bin/env python3
"""
快速全量评估脚本 - 使用 expression_style 查询缓存进行整体评估。
查询缓存由 08_generate_query_cache_Pet_Supplies.py 生成。
"""

import os
os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
COLBERTV2_CUDA_HOME = "/usr/local/cuda-12.5"
COLBERTV2_TORCH_EXTENSIONS_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/torch_extensions"
COLBERTV2_HOST_CC = "/usr/bin/gcc"
COLBERTV2_HOST_CXX = "/usr/bin/g++"
if not os.path.exists(os.path.join(COLBERTV2_CUDA_HOME, "include", "cuda_runtime.h")):
    raise FileNotFoundError(f"Required CUDA header not found under {COLBERTV2_CUDA_HOME}")
if not os.path.exists(os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")):
    raise FileNotFoundError(f"Required nvcc not found under {COLBERTV2_CUDA_HOME}")
if not os.path.exists(COLBERTV2_HOST_CC):
    raise FileNotFoundError(f"Required host C compiler not found: {COLBERTV2_HOST_CC}")
if not os.path.exists(COLBERTV2_HOST_CXX):
    raise FileNotFoundError(f"Required host C++ compiler not found: {COLBERTV2_HOST_CXX}")
os.environ["CUDA_HOME"] = COLBERTV2_CUDA_HOME
os.environ["CUDA_PATH"] = COLBERTV2_CUDA_HOME
os.environ["CUDACXX"] = os.path.join(COLBERTV2_CUDA_HOME, "bin", "nvcc")
os.environ["CC"] = COLBERTV2_HOST_CC
os.environ["CXX"] = COLBERTV2_HOST_CXX
os.environ["CUDAHOSTCXX"] = COLBERTV2_HOST_CXX
for _env_name, _env_value in [
    ("PATH", os.path.join(COLBERTV2_CUDA_HOME, "bin")),
    ("CPATH", os.path.join(COLBERTV2_CUDA_HOME, "include")),
    ("LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
    ("LD_LIBRARY_PATH", os.path.join(COLBERTV2_CUDA_HOME, "lib64")),
]:
    _existing_env_value = os.environ.get(_env_name)
    os.environ[_env_name] = _env_value if not _existing_env_value else f"{_env_value}:{_existing_env_value}"
import sys
import time
import pickle
import json
import gzip
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
# 设置路径
sys.path.insert(0, '/workspace/PersonalQuery/PersoanlQuery/06_retrieval')

# ============ 配置加载 ============
from config import get_category_config, get_retriever_config
from revised_query_utils import load_revised_query_map

CATEGORY_NAME = "Pet_Supplies"
CAT_CONFIG = get_category_config(CATEGORY_NAME)

CACHE_DIR = CAT_CONFIG['retriever_cache_dir']
QUERY_CACHE_BASE_DIR = CAT_CONFIG['query_cache_dir']
SYNTAX_DEPTH_QUERY_CATEGORY = 'syntax_depth'
SYNTAX_DEPTH_QUERY_TYPE = 'correct'
QUERY_TYPES = [SYNTAX_DEPTH_QUERY_TYPE]
QUERY_CATEGORIES = [SYNTAX_DEPTH_QUERY_CATEGORY]
SYNTAX_DEPTH_QUERY_FILE = "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/Pet_Supplies/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
OUTPUT_DIR = CAT_CONFIG['output_dir']
META_FILE = CAT_CONFIG['corpus_file']

RETRIEVER_CONFIG = get_retriever_config()
DENSE_RETRIEVERS = ['bge', 'e5', 'minilm', 'star', 'ance']
SPARSE_RETRIEVERS = ['splade']
COLBERTV2_RETRIEVERS = ['colbertv2']

# Skip retriever families if env var is set — matches Stage 06 build + Stage 07
# eval behavior (those stages never produce cache for skipped families).
if os.environ.get("SKIP_COLBERTV2", "").lower() in {"1", "true", "yes", "on"}:
    print(f"[{datetime.now().isoformat()}] [SKIP_COLBERTV2] SKIP_COLBERTV2=1 set, excluding colbertv2 from RETRIEVERS", flush=True)
    COLBERTV2_RETRIEVERS = []
if os.environ.get("SKIP_SPLADE", "").lower() in {"1", "true", "yes", "on"}:
    print(f"[{datetime.now().isoformat()}] [SKIP_SPLADE] SKIP_SPLADE=1 set, excluding splade from RETRIEVERS", flush=True)
    SPARSE_RETRIEVERS = []

RETRIEVERS = DENSE_RETRIEVERS + COLBERTV2_RETRIEVERS + SPARSE_RETRIEVERS + ['bm25']
DENSE_SEARCH_BATCH_SIZE = 64
SPLADE_QUERY_BATCH_SIZE = 128

DEPTH_GROUP_FIELD = 'query_group'
DEPTH_GROUPS = [
    ('all_queries', None, None, '全部查询'),
]
DEPTH_GROUP_ORDER = [name for name, _, _, _ in DEPTH_GROUPS]
DEPTH_GROUP_DISPLAY = {name: label for name, _, _, label in DEPTH_GROUPS}

# IDF 分层配置
IDF_BINS = [(2.5, 3.5), (3.5, 4.5), (4.5, 5.0), (5.0, float('inf'))]
IDF_BIN_LABELS = RETRIEVER_CONFIG['idf_bin_labels']

# ============ 日志 ============
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_colbert_torch_extension_arch_tag() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 evaluation requires a CUDA GPU for torch extension arch detection")
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def prepare_colbert_torch_extensions_dir() -> str:
    ext_dir = os.path.join(
        COLBERTV2_TORCH_EXTENSIONS_BASE_DIR,
        f"colbertv2_cuda125_{get_colbert_torch_extension_arch_tag()}"
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log(f"[BOOT] ColBERT TORCH_EXTENSIONS_DIR = {ext_dir}")
    return ext_dir


def log_colbert_extension_dir_state(ext_dir: str, prefix: str):
    active_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    log(f"{prefix} active TORCH_EXTENSIONS_DIR = {active_ext_dir}")
    if active_ext_dir != ext_dir:
        raise RuntimeError(
            f"{prefix} active TORCH_EXTENSIONS_DIR mismatch: expected {ext_dir}, got {active_ext_dir}"
        )
    if not os.path.exists(ext_dir):
        log(f"{prefix} extension dir missing: {ext_dir}")
        return
    entries = sorted(os.listdir(ext_dir))
    preview = entries[:20]
    log(f"{prefix} extension dir entries ({len(entries)}): {preview}")


COLBERTV2_TORCH_EXTENSIONS_DIR = prepare_colbert_torch_extensions_dir()


def restore_colbert_torch_extensions_dir(prefix: str):
    os.makedirs(COLBERTV2_TORCH_EXTENSIONS_DIR, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = COLBERTV2_TORCH_EXTENSIONS_DIR
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log(f"{prefix} ColBERT TORCH_EXTENSIONS_DIR = {COLBERTV2_TORCH_EXTENSIONS_DIR}")

# ============ 缓存完整性检查 ============
def validate_cache() -> bool:
    """检查缓存目录中的文件是否完整且有效"""
    log("\n检查缓存完整性...")

    if not os.path.exists(CACHE_DIR):
        log(f"  错误: 缓存目录不存在: {CACHE_DIR}")
        return False

    issues = []

    # 检查密集检索器文件及数据完整性
    for retriever in DENSE_RETRIEVERS:
        # 查找该检索器的所有版本
        # 文件格式: {retriever}_{hash}_{suffix}.npy/.pkl
        # hash 本身可能包含下划线，所以需要从后往前推断
        retriever_files = {}
        for f in os.listdir(CACHE_DIR):
            if not (f.startswith(f'{retriever}_') and f.endswith(('.npy', '.pkl'))):
                continue

            # 提取 suffix（最后一部分）
            suffix = f.rsplit('_', 1)[-1]

            # 提取 hash（去掉 retriever_ 前缀和 suffix 后缀）
            # 例如: bge_457d1871f380782c05a5d94e656fef2c_embeddings.npy
            # -> 去掉前缀 retriever_ 和后缀 _embeddings.npy
            middle = f[len(f'{retriever}_'):]
            hash_id = middle[:-len(suffix) - 1]  # -1 for the underscore before suffix

            if hash_id not in retriever_files:
                retriever_files[hash_id] = set()
            retriever_files[hash_id].add(suffix)

        if not retriever_files:
            issues.append(f"  缺失: {retriever} 检索器缓存文件")
            continue

        # 检查每个版本是否完整
        for hash_id, suffixes in retriever_files.items():
            required_files = ['embeddings.npy', 'doc_ids.pkl', 'config.pkl', 'metadata.pkl']
            for suffix in required_files:
                full_file = f"{retriever}_{hash_id}_{suffix}"
                file_path = os.path.join(CACHE_DIR, full_file)
                if suffix not in suffixes:
                    issues.append(f"  缺失: {retriever} ({hash_id[:8]}...) - {suffix}")
                elif os.path.getsize(file_path) == 0:
                    issues.append(f"  空文件: {full_file}")

            # 验证 embeddings 和 doc_ids 数量是否匹配
            embeddings_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_embeddings.npy")
            doc_ids_path = os.path.join(CACHE_DIR, f"{retriever}_{hash_id}_doc_ids.pkl")
            if os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
                try:
                    embeddings = np.load(embeddings_path, mmap_mode='r')
                    n_embeddings = embeddings.shape[0]
                    with open(doc_ids_path, 'rb') as f:
                        doc_ids = pickle.load(f)
                    n_doc_ids = len(doc_ids)

                    if n_embeddings != n_doc_ids:
                        issues.append(f"  数据不一致: {retriever} ({hash_id[:8]}...) - embeddings数量({n_embeddings}) != doc_ids数量({n_doc_ids})")

                    # 检查 doc_ids 是否有重复
                    if len(doc_ids) != len(set(doc_ids)):
                        duplicates = len(doc_ids) - len(set(doc_ids))
                        issues.append(f"  数据错误: {retriever} ({hash_id[:8]}...) - doc_ids中有 {duplicates} 个重复项")

                    log(f"  {retriever} ({hash_id[:8]}...): embeddings={n_embeddings}, doc_ids={n_doc_ids}")
                except Exception as e:
                    issues.append(f"  验证失败: {retriever} ({hash_id[:8]}...) - {str(e)}")

    # 检查 BM25 文件
    bm25_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('bm25_') and f.endswith('.pkl')]
    if not bm25_files:
        issues.append("  缺失: bm25 检索器缓存文件")
    else:
        for f in bm25_files:
            file_path = os.path.join(CACHE_DIR, f)
            if os.path.getsize(file_path) == 0:
                issues.append(f"  空文件: {f}")
            else:
                # 验证 BM25 数据可加载
                try:
                    with open(file_path, 'rb') as fp:
                        bm25 = pickle.load(fp)
                    # 检查 BM25 是否有 search 方法
                    if not hasattr(bm25, 'search'):
                        issues.append(f"  数据错误: {f} - BM25对象缺少search方法")
                    else:
                        log(f"  bm25 ({f.split('_')[1][:8]}...): 可正常加载")
                except Exception as e:
                    issues.append(f"  验证失败: {f} - {str(e)}")

    # 检查查询缓存
    log("  检查查询缓存...")
    for query_category in QUERY_CATEGORIES:
        for query_type in QUERY_TYPES:
            query_cache_dir = os.path.join(QUERY_CACHE_BASE_DIR, f'{query_category}_{query_type}_query')
            if not os.path.exists(query_cache_dir):
                issues.append(f"  缺失: 查询缓存目录 {query_category}_{query_type}_query")
                continue

            for retriever in RETRIEVERS:
                cache_file = os.path.join(query_cache_dir, f'{retriever}__{query_category}_{query_type}_cache.pkl')
                if not os.path.exists(cache_file):
                    issues.append(f"  缺失: {retriever} ({query_category}/{query_type}) 查询缓存")
                else:
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        n_users = len(cache_data)
                        log(f"  {retriever} ({query_category}/{query_type}): {n_users} 用户")
                    except Exception as e:
                        issues.append(f"  验证失败: {retriever} ({query_category}/{query_type}) - {str(e)}")

    if issues:
        log("  缓存完整性检查未通过:")
        for issue in issues:
            log(issue)
        return False

    log("  缓存完整性检查通过 ✓")
    return True

# ============ 分组配置 ============
# 模块级变量，当前仅保留统一查询集合分组。
UNIQUE_LEVELS = DEPTH_GROUP_ORDER.copy()
GROUP_FIELD = DEPTH_GROUP_FIELD


def get_group_display(group) -> str:
    if group not in DEPTH_GROUP_DISPLAY:
        raise KeyError(f"Unknown syntax depth group: {group}")
    return DEPTH_GROUP_DISPLAY[group]


def depth_to_complexity_group(depth: int) -> str | None:
    return 'all_queries'


def compute_query_idf_simple(query: str, word_idf: Dict[str, float]) -> float:
    """计算查询的平均IDF"""
    words = query.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]
    return np.mean(idf_values)

# ============ 评估指标计算 ============
def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f'P@{k}'] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f'N@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'MR@{k}'] = 1.0 / rank
        else:
            metrics[f'N@{k}'] = 0.0
            metrics[f'MR@{k}'] = 0.0
        metrics[f'H@{k}'] = 1.0 if relevant_asin in top_k else 0.0
    return metrics

def compute_average_metrics(all_metrics: List[Dict], k_values: List[int]) -> Dict:
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f'P@{k}'] = np.mean([m.get(f'P@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'N@{k}'] = np.mean([m.get(f'N@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'MR@{k}'] = np.mean([m.get(f'MR@{k}', 0.0) for m in all_metrics])
        avg_metrics[f'H@{k}'] = np.mean([m.get(f'H@{k}', 0.0) for m in all_metrics])
    return avg_metrics

# ============ Bootstrap CI ============
def compute_bootstrap_ci(all_metrics: List[Dict], k_values: List[int], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """计算Bootstrap置信区间"""
    np.random.seed(42)
    n_samples = len(all_metrics)
    if n_samples < 2:
        return {}

    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    metric_keys = [f'P@{k}' for k in k_values] + [f'N@{k}' for k in k_values] + [f'MR@{k}' for k in k_values] + [f'H@{k}' for k in k_values]

    bootstrap_results = {}
    for key in metric_keys:
        values = np.array([m.get(key, 0.0) for m in all_metrics])
        bootstrapped_means = []
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_values = values[sample_indices]
            bootstrapped_means.append(np.mean(sample_values))
        bootstrapped_means = np.array(bootstrapped_means)
        bootstrap_results[key] = {
            'mean': np.mean(bootstrapped_means),
            'std': np.std(bootstrapped_means),
            'ci_lower': np.percentile(bootstrapped_means, lower_percentile),
            'ci_upper': np.percentile(bootstrapped_means, upper_percentile),
        }
    return bootstrap_results

def print_bootstrap_ci_table(all_results: List[Dict], k_values: List[int]):
    log("\n" + "=" * 100)
    log("Bootstrap CI (95%) - P@10")
    log("=" * 100)

    header = f"{'检索器':<12} {GROUP_FIELD.upper():<10} {'Mean':<10} {'Std':<10} {'CI Lower':<12} {'CI Upper':<12}"
    log(header)
    log("-" * 100)

    for r in all_results:
        retriever = r['retriever']
        for group in UNIQUE_LEVELS:
            ci = r['bootstrap_ci'].get(group, {}).get('P@10', {})
            if ci:
                row = f"{retriever:<12} {get_group_display(group):<10}   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}"
                log(row)

    log("-" * 100)
    # 总体 CI
    for r in all_results:
        ci = r['bootstrap_ci'].get('overall', {}).get('P@10', {})
        if ci:
            log(f"{r['retriever']:<12} overall   {ci['mean']:.4f}     {ci['std']:.4f}     {ci['ci_lower']:.4f}      {ci['ci_upper']:.4f}")

# ============ 数据加载 ============
def load_dense_retriever(retriever_name: str) -> Tuple[np.ndarray, List[str], int]:
    embeddings_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith(f'{retriever_name}_') and f.endswith('_embeddings.npy'):
            candidate = os.path.join(CACHE_DIR, f)
            # Skip stale empty / 1D placeholder files (Reviewer note iter #69:
            # the rebuild from iter #56a left a 128-byte MD5 hash placeholder
            # bge_d41d8cd98f00b204e9800998ecf8427e_embeddings.npy when there was
            # nothing to encode; this loader used to pick whichever file came
            # first in os.listdir order, sometimes the empty one).
            if os.path.getsize(candidate) <= 1024:
                log(f"  [{retriever_name}] 跳过空缓存: {f}")
                continue
            embeddings_path = candidate
            break
    if embeddings_path is None:
        raise FileNotFoundError(f"{retriever_name} embeddings not found")

    log(f"  [{retriever_name}] 加载: {os.path.getsize(embeddings_path)/1024/1024:.1f} MB")
    mmap_array = np.load(embeddings_path, mmap_mode='r')
    embeddings = mmap_array[:].copy()

    doc_ids_path = embeddings_path.replace('_embeddings.npy', '_doc_ids.pkl')
    with open(doc_ids_path, 'rb') as f:
        doc_ids = pickle.load(f)

    return embeddings, doc_ids, embeddings.shape[1]

def load_bm25_retriever():
    """加载 BM25 检索器"""
    bm25_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('bm25_') and f.endswith('.pkl'):
            bm25_path = os.path.join(CACHE_DIR, f)
            break
    if bm25_path is None:
        raise FileNotFoundError("BM25 cache not found")

    log(f"  [bm25] 加载: {os.path.getsize(bm25_path)/1024/1024:.1f} MB")
    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)
    return bm25


def get_query_cache_path(retriever_name: str, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> str:
    return os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'{retriever_name}__{query_category}_{query_type}_cache.pkl'
    )


def query_cache_exists(retriever_name: str, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> bool:
    return os.path.exists(get_query_cache_path(retriever_name, query_type, query_category))


def list_retrievers_with_query_cache(query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> List[str]:
    return [
        retriever_name
        for retriever_name in RETRIEVERS
        if query_cache_exists(retriever_name, query_type, query_category)
    ]


def load_query_cache(retriever_name: str, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    """加载查询缓存

    Args:
        retriever_name: 检索器名称
        query_type: 查询类型
        query_category: 查询类别
    """
    cache_path = get_query_cache_path(retriever_name, query_type, query_category)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{retriever_name} query cache not found: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"{retriever_name} query cache must be dict, got {type(cache).__name__}: {cache_path}")
    return cache


def load_bm25_query_cache(query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    """加载 BM25 查询缓存

    Args:
        query_type: 查询类型
        query_category: 查询类别

    Returns:
        Dict: 查询缓存，key 为 query 字符串，value 为检索结果 [(asin, score), ...]
    """
    cache_path = get_query_cache_path('bm25', query_type, query_category)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"BM25 query result cache not found: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"BM25 query result cache must be dict, got {type(cache).__name__}: {cache_path}")
    return cache


def load_result_query_cache(retriever_name: str, query_type: str, query_category: str) -> Dict:
    """加载已经生成好的 query -> [(asin, score), ...] 结果缓存"""
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'{retriever_name}__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{retriever_name} query result cache not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    if not isinstance(cache, dict):
        raise TypeError(f"{retriever_name} query result cache must be dict, got {type(cache).__name__}: {cache_path}")

    return cache


def resolve_colbertv2_output_root() -> str:
    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f"Retriever cache directory not found: {CACHE_DIR}")

    candidates = []
    for name in sorted(os.listdir(CACHE_DIR)):
        output_root = os.path.join(CACHE_DIR, name)
        if not name.startswith("colbertv2_") or not os.path.isdir(output_root):
            continue
        manifest_path = os.path.join(output_root, "build_manifest.json")
        doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
        index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
        if os.path.isfile(manifest_path) and os.path.isfile(doc_ids_path) and os.path.isdir(index_dir):
            candidates.append(output_root)

    if not candidates:
        raise FileNotFoundError(f"No complete ColBERTv2 cache directory found under: {CACHE_DIR}")
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
    if manifest["category"] != CATEGORY_NAME:
        raise ValueError(f"ColBERTv2 cache category mismatch in {manifest_path}: {manifest['category']}")
    if manifest["output_root"] != output_root:
        raise ValueError(f"ColBERTv2 cache output_root mismatch in {manifest_path}: {manifest['output_root']}")

    log(f"  直接读取 ColBERTv2 缓存目录: {output_root}")
    return output_root


def load_colbertv2_doc_ids(output_root: str) -> List[str]:
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Required ColBERTv2 doc id mapping not found: {doc_ids_path}")

    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)

    if not isinstance(doc_ids, list):
        raise TypeError(f"doc_ids.pkl must contain a list, got {type(doc_ids).__name__}")
    if not doc_ids:
        raise ValueError(f"doc_ids.pkl is empty: {doc_ids_path}")
    return doc_ids


def resolve_local_hf_snapshot(repo_id: str) -> str:
    repo_cache_dir = os.path.join(
        os.environ["HF_HOME"],
        "models--" + repo_id.replace("/", "--")
    )
    ref_file = os.path.join(repo_cache_dir, "refs", "main")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Hugging Face ref file not found for {repo_id}: {ref_file}")
    with open(ref_file, "r") as f:
        snapshot_hash = f.read().strip()
    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", snapshot_hash)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Hugging Face snapshot not found for {repo_id}: {snapshot_dir}")
    return snapshot_dir


def build_colbertv2_searcher(output_root: str, doc_ids: List[str]):
    restore_colbert_torch_extensions_dir("[ColBERT][before Searcher restore]")

    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher

    collection = [f"pid {pid} asin {asin}" for pid, asin in enumerate(doc_ids)]
    checkpoint_path = resolve_local_hf_snapshot("colbert-ir/colbertv2.0")
    log(f"[ColBERT] checkpoint_path = {checkpoint_path}")
    log(f"[ColBERT] output_root = {output_root}")
    log_colbert_extension_dir_state(COLBERTV2_TORCH_EXTENSIONS_DIR, "[ColBERT][before Searcher]")
    searcher_start = time.time()
    with Run().context(RunConfig(experiment="colbertv2_index", root=output_root)):
        config = ColBERTConfig(root=output_root)
        searcher = Searcher(
            index="colbertv2_index",
            checkpoint=checkpoint_path,
            collection=collection,
            config=config,
        )
    log(f"[ColBERT] Searcher loaded in {time.time() - searcher_start:.1f}s")
    log_colbert_extension_dir_state(COLBERTV2_TORCH_EXTENSIONS_DIR, "[ColBERT][after Searcher]")
    return searcher


def load_colbertv2_query_cache(query_type: str, query_category: str) -> Dict[str, Dict[str, np.ndarray]]:
    cache = load_query_cache("colbertv2", query_type, query_category)
    if not isinstance(cache, dict):
        raise TypeError(f"ColBERTv2 query embedding cache must be dict, got {type(cache).__name__}")
    return cache


def colbertv2_search_from_cached_embedding(searcher, doc_ids: List[str], query_embedding, top_k: int) -> List[Tuple[str, float]]:
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
            raise IndexError(f"ColBERTv2 pid {pid_int} is outside doc_ids range 0..{len(doc_ids)-1}")
        results.append((doc_ids[pid_int], float(score)))
    return results


def load_user_queries(query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY, filter_same_level: bool = False) -> Tuple[Dict[str, List[Dict]], Dict[str, str], List[Tuple[int, int, int]]]:
    """加载 expression_style 查询，并统一归入单一查询集合分组。"""
    global GROUP_FIELD, UNIQUE_LEVELS

    if query_type != SYNTAX_DEPTH_QUERY_TYPE:
        raise ValueError(f"Only query_type={SYNTAX_DEPTH_QUERY_TYPE} is supported, got {query_type}")
    if query_category != SYNTAX_DEPTH_QUERY_CATEGORY:
        raise ValueError(f"Only query_category={SYNTAX_DEPTH_QUERY_CATEGORY} is supported, got {query_category}")
    if filter_same_level:
        raise ValueError("filter_same_level is not supported for expression_style evaluation")
    if not os.path.exists(SYNTAX_DEPTH_QUERY_FILE):
        raise FileNotFoundError(f"expression_style query file not found: {SYNTAX_DEPTH_QUERY_FILE}")

    GROUP_FIELD = DEPTH_GROUP_FIELD
    UNIQUE_LEVELS = DEPTH_GROUP_ORDER.copy()

    with open(SYNTAX_DEPTH_QUERY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"expression_style query file must contain a list, got {type(data).__name__}")

    user_queries: Dict[str, List[Dict]] = {}
    user_to_group: Dict[str, str] = {}
    all_query_metadata: List[Tuple[int, int, int]] = []
    idx = 0

    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"expression_style row must be dict at index {row_index}, got {type(item).__name__}")
        for field in ('user_id', 'asin', 'syntax_depth_query'):
            if field not in item:
                raise ValueError(f"expression_style row missing required field '{field}' at index {row_index}")

        user_id = item['user_id']
        asin = item['asin']
        syntax_depth_query = item['syntax_depth_query']
        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"expression_style row has invalid user_id at index {row_index}: {user_id!r}")
        if not isinstance(asin, str) or not asin:
            raise ValueError(f"expression_style row has invalid asin for user={user_id}, index={row_index}: {asin!r}")
        if not isinstance(syntax_depth_query, dict):
            raise TypeError(f"syntax_depth_query must be dict for user={user_id}, index={row_index}")

        for field in ('query', 'target_depth', 'user_avg_depth', 'word_count'):
            if field not in syntax_depth_query:
                raise ValueError(f"syntax_depth_query missing required field '{field}' for user={user_id}, index={row_index}")

        query_text = syntax_depth_query['query']
        if not isinstance(query_text, str):
            raise TypeError(f"query must be string for user={user_id}, index={row_index}")
        query_text = query_text.strip()
        if not query_text:
            raise ValueError(f"query is empty for user={user_id}, index={row_index}")

        actual_depth = syntax_depth_query['actual_depth']
        target_depth = syntax_depth_query['target_depth']
        word_count = syntax_depth_query['word_count']
        user_avg_depth = syntax_depth_query['user_avg_depth']
        if not isinstance(target_depth, int):
            raise TypeError(f"target_depth must be int for user={user_id}, index={row_index}, got {type(target_depth).__name__}")
        if not isinstance(word_count, int):
            raise TypeError(f"word_count must be int for user={user_id}, index={row_index}, got {type(word_count).__name__}")
        if not isinstance(user_avg_depth, (int, float)):
            raise TypeError(f"user_avg_depth must be numeric for user={user_id}, index={row_index}, got {type(user_avg_depth).__name__}")

        depth_group = 'all_queries'

        if user_id not in user_queries:
            user_queries[user_id] = []
            user_to_group[user_id] = depth_group

        user_queries[user_id].append({
            'query': query_text,
            'asin': asin,
            'word_count': word_count,
            'syntax_depth': actual_depth,
            'target_depth': target_depth,
            'user_avg_depth': float(user_avg_depth),
            'depth_group_label': get_group_display(depth_group),
            f'{GROUP_FIELD}_ratio': 0.0,
            GROUP_FIELD: depth_group,
        })
        all_query_metadata.append((idx, word_count, actual_depth))
        idx += 1

    if not user_queries:
        raise ValueError("No expression_style queries were loaded")

    return user_queries, user_to_group, all_query_metadata

def build_word_idf_dict(meta_file: str, sample_size: int | None = None) -> Dict[str, float]:
    """从商品元数据语料库构建词的IDF字典。

    当 `sample_size` 为 `None` 时，遍历完整商品语料。
    """
    import gzip
    word_doc_freq = defaultdict(int)
    total_sampled = 0

    if sample_size is None:
        log("Building word IDF from full corpus...")
    else:
        log(f"Building word IDF from corpus (sampling {sample_size} docs)...")
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size is not None and i >= sample_size:
                break
            try:
                item = json.loads(line)
                # 从 title + brand + description 提取词
                text = ' '.join(filter(None, [
                    item.get('title', ''),
                    item.get('brand', ''),
                    ' '.join(item.get('description', []))
                ])).lower()
                words = set(text.split())
                for w in words:
                    if len(w) > 1:  # 过滤单字符
                        word_doc_freq[w] += 1
                total_sampled += 1
            except Exception:
                continue

    N = total_sampled
    word_idf = {}
    for w, df in word_doc_freq.items():
        word_idf[w] = np.log(N / (df + 1))  # +1 平滑

    # 也计算字符ngram（用于商品品牌等专有名词）
    for w, df in word_doc_freq.items():
        if len(w) >= 4 and df < 10:
            # 罕见词给高IDF
            word_idf[w] = max(word_idf.get(w, 0), np.log(N / 10))

    log(f"  IDF vocabulary: {len(word_idf)} words, {total_sampled} docs processed")
    return word_idf


def compute_query_idf(query_text: str, word_idf: Dict[str, float]) -> float:
    """计算查询的平均IDF（使用预计算的词IDF）"""
    words = query_text.lower().split()
    if not words:
        return 0.0
    idf_values = [word_idf.get(w, 5.0) for w in words]  # 未知词给中等IDF=5
    return np.mean(idf_values)


def compute_idf(queries: List[str], doc_count: int) -> float:
    """计算查询的平均IDF（简化版：使用词频）"""
    from collections import Counter
    words = ' '.join(queries).lower().split()
    word_freq = Counter(words)
    if not word_freq:
        return 0.0
    # 平均 IDF = log(N / df)，这里简化为平均词频的倒数
    avg_df = sum(word_freq.values()) / len(word_freq) if word_freq else 1
    return np.log(doc_count / avg_df + 1)


def save_word_idf_dict(word_idf: Dict[str, float], sample_size: int, output_dir: str) -> Tuple[str, str]:
    """保存构建好的词级 IDF 字典。

    保存两份文件：
    1. `word_idf.pkl`：完整词典，供后续直接加载复用
    2. `word_idf_summary.json`：概要信息，便于快速检查
    """
    os.makedirs(output_dir, exist_ok=True)

    idf_pickle_path = os.path.join(output_dir, "word_idf.pkl")
    idf_summary_path = os.path.join(output_dir, "word_idf_summary.json")

    with open(idf_pickle_path, "wb") as f:
        pickle.dump(word_idf, f, protocol=pickle.HIGHEST_PROTOCOL)

    sorted_items = sorted(word_idf.items(), key=lambda x: x[1], reverse=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "category_name": CATEGORY_NAME,
        "sample_size": sample_size,
        "vocab_size": len(word_idf),
        "top_100_highest_idf": [
            {"token": token, "idf": float(idf_value)}
            for token, idf_value in sorted_items[:100]
        ]
    }
    with open(idf_summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return idf_pickle_path, idf_summary_path

def compute_oracle_random_baseline(relevant_asin: str, doc_ids: List[str], n_trials: int = 100, seed: int = 42) -> Dict:
    """计算oracle-aware随机基线：给定相关文档在随机位置时的期望性能"""
    np.random.seed(seed)
    n_docs = len(doc_ids)
    if relevant_asin not in doc_ids:
        return {'P@10': 0.0, 'N@10': 0.0}
    rel_idx = doc_ids.index(relevant_asin)
    p10_list = []
    n10_list = []
    for _ in range(n_trials):
        # 随机打乱位置（保留relevant_asin在某个随机位置）
        random_pos = np.random.randint(0, n_docs)
        top10_positions = list(range(random_pos, min(random_pos + 10, n_docs)))
        if rel_idx in top10_positions:
            rank = top10_positions.index(rel_idx) + 1
            p10_list.append(1.0)
            n10_list.append(1.0 / np.log2(rank + 1))
        else:
            p10_list.append(0.0)
            n10_list.append(0.0)
    return {'P@10': np.mean(p10_list), 'N@10': np.mean(n10_list)}

# ============ 搜索器 ============
class DenseSearcher:
    """密集检索器搜索器 (GPU 矩阵乘法 + 余弦相似度)"""
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str], retriever_name: str):
        self.doc_ids = doc_ids
        self.retriever_name = retriever_name
        self.device = torch.device('cuda')
        # 归一化 doc embeddings 以支持余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized_embeddings = embeddings / norms
        self.embeddings_tensor = torch.from_numpy(normalized_embeddings).float().to(self.device)

    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10, batch_size: int = DENSE_SEARCH_BATCH_SIZE) -> List[List[Tuple[str, float]]]:
        if not query_embeddings:
            return []
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0, 实际为 {batch_size}")

        results = []
        n = len(query_embeddings)
        top_k = min(top_k, len(self.doc_ids))

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_embeddings = query_embeddings[start:end]
                query_tensor = torch.from_numpy(np.array(batch_embeddings)).float().to(self.device)
                # 归一化 query embeddings
                q_norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                q_norms = np.where(q_norms == 0, 1, q_norms)
                query_tensor = query_tensor / torch.from_numpy(q_norms).float().to(self.device)
                # 余弦相似度 = 归一化点积
                scores = torch.mm(query_tensor, self.embeddings_tensor.T)
                for i in range(scores.shape[0]):
                    top_scores, top_indices = torch.topk(scores[i], top_k)
                    results.append([(self.doc_ids[idx.item()], top_scores[j].item()) for j, idx in enumerate(top_indices)])
                del scores
                del query_tensor

        if len(results) != n:
            raise RuntimeError(f"批量搜索结果条数不匹配: results={len(results)} queries={n}")
        return results

class BM25Searcher:
    """BM25 搜索器 (文本搜索)"""
    def __init__(self, bm25_retriever):
        self.bm25 = bm25_retriever

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        results = []
        for query in queries:
            # BM25 search returns [(asin, score), ...]
            search_results = self.bm25.search(query, top_k=top_k)
            results.append(search_results)
        return results

# ============ 评估 ============
def evaluate_dense_retriever(retriever_name: str, user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = DEPTH_GROUP_FIELD
    # 清理 CUDA 缓存，避免 cuBLAS 上下文冲突
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    log(f"\n{'='*60}")
    log(f"检索器: {retriever_name.upper()} (密集) - SYNTAX_DEPTH/{query_type.upper()}")
    log(f"{'='*60}")

    embeddings, doc_ids, dim = load_dense_retriever(retriever_name)
    query_cache = load_query_cache(retriever_name, query_type, query_category)
    searcher = DenseSearcher(embeddings, doc_ids, retriever_name)

    matched_users = [uid for uid in user_queries.keys() if uid in query_cache]
    skipped_users = len(user_queries) - len(matched_users)
    log(f"  缓存用户命中: {len(matched_users)}/{len(user_queries)}，跳过无缓存用户: {skipped_users}")

    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []
    eval_start = time.time()

    # 分组统计: word_count bins 和 group_ratio bins
    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']

    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    # IDF × group 交叉分组: {(idf_label, group): [metrics]}
    idf_group_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    # 第一步：收集所有用户的查询信息
    log(f"  收集所有查询信息...")
    all_query_data = []  # [(user_id, query_embedding, relevant_asin, word_count, group_ratio, q_group, q_idf)]

    total_candidate_queries = 0
    skipped_missing_queries = 0
    for user_idx, user_id in enumerate(matched_users):
        queries = user_queries[user_id]
        cached_queries = query_cache[user_id]

        for q in queries:
            total_candidate_queries += 1
            query_text = q['query']
            relevant_asin = q['asin']
            word_count = q['word_count']
            group_ratio = q[f'{GROUP_FIELD}_ratio']
            q_group = q[GROUP_FIELD]
            if query_text not in cached_queries:
                skipped_missing_queries += 1
                continue
            q_idf = 0.0
            all_query_data.append((user_id, cached_queries[query_text], relevant_asin, word_count, group_ratio, q_group, q_idf, q))

    log(
        f"  缓存查询命中: {len(all_query_data)}/{total_candidate_queries}，"
        f"跳过无缓存查询: {skipped_missing_queries}"
    )
    if not all_query_data:
        raise ValueError(f"{retriever_name} has no cached expression_style queries to evaluate")

    # 第二步：一次性批量搜索所有查询
    log(f"  批量搜索所有查询...")
    all_embeddings = [q[1] for q in all_query_data]
    results = searcher.search_batch(all_embeddings, top_k=max(k_values), batch_size=DENSE_SEARCH_BATCH_SIZE)

    # 第三步：遍历搜索结果，进行分组统计
    log(f"  处理结果并分组统计...")
    for i, (result, (user_id, _, relevant_asin, word_count, group_ratio, q_group, q_idf, query_info)) in enumerate(zip(results, all_query_data)):
        retrieved_asins = [r[0] for r in result]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        all_metrics.append(metrics)
        group_groups[q_group].append(metrics)

        # 记录每条 query 的原始数据
        all_query_records.append({
            'user_id': user_id,
            'asin': relevant_asin,
            f'{GROUP_FIELD}': q_group,
            'syntax_depth': query_info['syntax_depth'],
            'target_depth': query_info['target_depth'],
            'user_avg_depth': query_info['user_avg_depth'],
            'depth_group_label': query_info['depth_group_label'],
            'mean_idf': q_idf,
            'query_length': word_count,
            f'{GROUP_FIELD}_ratio': group_ratio,
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # word_count 分组
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= word_count < high:
                word_count_groups[label].append(metrics)
                break

        # group_ratio 分组
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        # IDF 分组
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, q_group)].append(metrics)
                break

        if (i + 1) % 500 == 0:
            log(f"    进度: {i+1}/{len(all_query_data)} ({100*(i+1)/len(all_query_data):.1f}%)")

    log(f"    进度: {len(all_query_data)}/{len(all_query_data)} (100.0%)")

    eval_time = time.time() - eval_start
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 group_ratio 分组统计
    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × group 交叉分组统计
    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    # 显式释放 GPU 内存
    del embeddings
    del searcher
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        'retriever': retriever_name, 'dim': dim, 'type': 'dense', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }

def evaluate_bm25_retriever(user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = DEPTH_GROUP_FIELD
    log(f"\n{'='*60}")
    log(f"检索器: BM25 (稀疏) - SYNTAX_DEPTH/{query_type.upper()}")
    log(f"{'='*60}")

    matched_users = list(user_queries.keys())
    log(f"  用户数: {len(matched_users)}")

    eval_start = time.time()

    # ========== 优化：批量搜索所有查询 ==========
    # 先收集所有查询和元数据
    all_query_texts = []
    all_query_asins = []
    all_query_users = []
    all_query_word_counts = []
    all_query_group_ratios = []
    all_query_groups = []
    all_query_idf_values = []
    all_query_depths = []
    all_query_target_depths = []
    all_query_user_avg_depths = []

    for user_id in matched_users:
        queries = user_queries[user_id]
        for q in queries:
            all_query_texts.append(q['query'])
            all_query_asins.append(q['asin'])
            all_query_users.append(user_id)
            all_query_word_counts.append(q['word_count'])
            all_query_group_ratios.append(q[f'{GROUP_FIELD}_ratio'])
            all_query_groups.append(q[GROUP_FIELD])
            all_query_idf_values.append(0.0)
            all_query_depths.append(q['syntax_depth'])
            all_query_target_depths.append(q['target_depth'])
            all_query_user_avg_depths.append(q['user_avg_depth'])

    log(f"  总查询数: {len(all_query_texts)}")
    if not all_query_texts:
        raise ValueError("BM25 has no expression_style queries to evaluate")

    # 直接加载 08_generate_query_cache_Pet_Supplies.py 生成的 BM25 查询结果缓存
    log(f"  加载 BM25 查询结果缓存...")
    query_cache = load_bm25_query_cache(query_type, query_category)
    log(f"  使用查询缓存 (共 {len(query_cache)} 条记录)...")
    cache_hit_start = time.time()
    all_results = []
    eval_indices = []
    skipped_missing_queries = 0
    for index, query_text in enumerate(all_query_texts):
        if query_text in query_cache:
            eval_indices.append(index)
            all_results.append(query_cache[query_text])
        else:
            skipped_missing_queries += 1
    cache_time = time.time() - cache_hit_start
    log(
        f"  缓存查找完成: {len(eval_indices)}/{len(all_query_texts)} 命中, "
        f"跳过无缓存查询: {skipped_missing_queries}, 耗时: {cache_time:.1f}秒"
    )
    if not all_results:
        raise ValueError("BM25 has no cached expression_style queries to evaluate")

    # 分组统计
    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []

    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_group_cross = defaultdict(list)
    # 收集所有 query 原始数据用于 OLS 回归
    all_query_records = []

    # 处理每个查询的结果
    for result_index, retrieved in enumerate(all_results):
        i = eval_indices[result_index]
        relevant_asin = all_query_asins[i]
        retrieved_asins = [r[0] for r in retrieved]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        group = all_query_groups[i]
        all_metrics.append(metrics)
        group_groups[group].append(metrics)

        # 记录每条 query 的原始数据
        all_query_records.append({
            'user_id': all_query_users[i],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': group,
            'syntax_depth': all_query_depths[i],
            'target_depth': all_query_target_depths[i],
            'user_avg_depth': all_query_user_avg_depths[i],
            'depth_group_label': get_group_display(group),
            'mean_idf': all_query_idf_values[i],
            'query_length': all_query_word_counts[i],
            f'{GROUP_FIELD}_ratio': all_query_group_ratios[i],
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # word_count 分组
        wc = all_query_word_counts[i]
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= wc < high:
                word_count_groups[label].append(metrics)
                break

        # group_ratio 分组
        cr = all_query_group_ratios[i]
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= cr < high:
                group_ratio_groups[label].append(metrics)
                break

        # IDF 分组
        q_idf = all_query_idf_values[i]
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, group)].append(metrics)
                break

        if (result_index + 1) % 100 == 0:
            elapsed = time.time() - eval_start
            log(f"    进度: {result_index+1}/{len(all_results)} ({100*(result_index+1)/len(all_results):.1f}%)")

    eval_time = time.time() - eval_start
    log(f"  评估完成，总耗时: {eval_time:.1f}秒")
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 group_ratio 分组统计
    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × group 交叉分组统计
    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': 'bm25', 'dim': 0, 'type': 'sparse', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }


def evaluate_cached_result_retriever(retriever_name: str, user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float], query_type: str, query_category: str) -> Dict:
    """评估缓存为 user -> query -> token embedding 的 ColBERTv2 查询缓存"""
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = DEPTH_GROUP_FIELD

    required_k_values = {1, 3, 5, 10}
    if not required_k_values.issubset(set(k_values)):
        raise ValueError(f"{retriever_name} evaluation requires k_values to contain {sorted(required_k_values)}")

    log(f"\n{'='*60}")
    if retriever_name != "colbertv2":
        raise ValueError(f"Token-embedding evaluation only supports colbertv2, got {retriever_name}")

    log(f"检索器: {retriever_name.upper()} (query token embedding 缓存) - SYNTAX_DEPTH/{query_type.upper()}")
    log(f"{'='*60}")

    query_cache = load_colbertv2_query_cache(query_type, query_category)
    matched_users = [uid for uid in user_queries.keys() if uid in query_cache]
    skipped_users = len(user_queries) - len(matched_users)
    output_root = resolve_colbertv2_output_root()
    index_dir = os.path.join(output_root, "colbertv2_index", "indexes", "colbertv2_index")
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Required ColBERTv2 index directory not found: {index_dir}")

    log(f"  缓存用户命中: {len(matched_users)}/{len(user_queries)}，跳过无缓存用户: {skipped_users}")
    log(f"  ColBERTv2 索引目录: {index_dir}")
    doc_ids = load_colbertv2_doc_ids(output_root)
    searcher = build_colbertv2_searcher(output_root, doc_ids)

    eval_start = time.time()

    all_query_texts = []
    all_query_embeddings = []
    all_query_asins = []
    all_query_users = []
    all_query_word_counts = []
    all_query_group_ratios = []
    all_query_groups = []
    all_query_idf_values = []
    all_query_depths = []
    all_query_target_depths = []
    all_query_user_avg_depths = []

    total_candidate_queries = 0
    skipped_missing_queries = 0
    for user_id in matched_users:
        cached_queries = query_cache[user_id]
        if not isinstance(cached_queries, dict):
            raise TypeError(f"{retriever_name} query embedding cache for user {user_id} must be dict")
        for q in user_queries[user_id]:
            total_candidate_queries += 1
            query_text = q['query']
            if query_text not in cached_queries:
                skipped_missing_queries += 1
                continue
            all_query_texts.append(query_text)
            all_query_embeddings.append(cached_queries[query_text])
            all_query_asins.append(q['asin'])
            all_query_users.append(user_id)
            all_query_word_counts.append(q['word_count'])
            all_query_group_ratios.append(q[f'{GROUP_FIELD}_ratio'])
            all_query_groups.append(q[GROUP_FIELD])
            all_query_idf_values.append(0.0)
            all_query_depths.append(q['syntax_depth'])
            all_query_target_depths.append(q['target_depth'])
            all_query_user_avg_depths.append(q['user_avg_depth'])

    if not all_query_texts:
        raise ValueError(f"{retriever_name} has no cached expression_style queries to evaluate")

    log(
        f"  缓存查询命中: {len(all_query_texts)}/{total_candidate_queries}，"
        f"跳过无缓存查询: {skipped_missing_queries}"
    )
    log(f"  使用 query token embedding 缓存 (用户数 {len(query_cache)})...")

    all_results = []
    for index, query_embedding in enumerate(all_query_embeddings):
        all_results.append(
            colbertv2_search_from_cached_embedding(
                searcher,
                doc_ids,
                query_embedding,
                max(k_values),
            )
        )
        if (index + 1) % 500 == 0:
            log(f"    ColBERTv2 搜索进度: {index+1}/{len(all_query_embeddings)} ({100*(index+1)/len(all_query_embeddings):.1f}%)")

    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []

    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_group_cross = defaultdict(list)
    all_query_records = []

    for i, (retrieved, relevant_asin) in enumerate(zip(all_results, all_query_asins)):
        retrieved_asins = [r[0] for r in retrieved]
        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        group = all_query_groups[i]
        all_metrics.append(metrics)
        group_groups[group].append(metrics)

        all_query_records.append({
            'user_id': all_query_users[i],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': group,
            'syntax_depth': all_query_depths[i],
            'target_depth': all_query_target_depths[i],
            'user_avg_depth': all_query_user_avg_depths[i],
            'depth_group_label': get_group_display(group),
            'mean_idf': all_query_idf_values[i],
            'query_length': all_query_word_counts[i],
            f'{GROUP_FIELD}_ratio': all_query_group_ratios[i],
            'p_at1': float(metrics['P@1']),
            'p_at3': float(metrics['P@3']),
            'p_at5': float(metrics['P@5']),
            'p_at10': float(metrics['P@10']),
            'n_at10': float(metrics['N@10']),
            'mrr_at10': float(metrics['MR@10']),
            'hit_at10': float(metrics['H@10']),
        })

        wc = all_query_word_counts[i]
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= wc < high:
                word_count_groups[label].append(metrics)
                break

        group_ratio = all_query_group_ratios[i]
        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        q_idf = all_query_idf_values[i]
        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, group)].append(metrics)
                break

        if (i + 1) % 500 == 0:
            log(f"    进度: {i+1}/{len(all_results)} ({100*(i+1)/len(all_results):.1f}%)")

    eval_time = time.time() - eval_start
    log(f"  评估完成，总耗时: {eval_time:.1f}秒")
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': retriever_name, 'dim': 128, 'type': 'late_interaction_query_embedding_cache',
        'num_users': len(matched_users), 'num_queries': len(all_metrics),
        'eval_time_seconds': eval_time, 'metrics': overall_metrics,
        'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }


def load_splade_retriever():
    """加载 SPLADE 检索器（稀疏向量格式）"""
    from utils.retrievers import SPLADERetriever

    splade_path = None
    for f in os.listdir(CACHE_DIR):
        if f.startswith('splade_') and f.endswith('.pkl'):
            splade_path = os.path.join(CACHE_DIR, f)
            break
    if splade_path is None:
        raise FileNotFoundError("SPLADE cache not found")

    log(f"  [splade] 加载: {os.path.getsize(splade_path)/1024/1024:.1f} MB")
    with open(splade_path, 'rb') as f:
        retriever = pickle.load(f)
    return retriever


def load_splade_query_cache(query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    """加载 SPLADE 查询缓存

    Returns:
        Dict: {user_id: {query_text: {doc_idx: score}}} - 稀疏分数格式
    """
    cache_path = os.path.join(
        QUERY_CACHE_BASE_DIR,
        f'{query_category}_{query_type}_query',
        f'splade__{query_category}_{query_type}_cache.pkl'
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"SPLADE query cache not found: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"SPLADE query cache must be dict, got {type(cache).__name__}: {cache_path}")
    return cache


def evaluate_splade_retriever(user_queries: Dict, user_to_group: Dict, k_values: List[int], word_idf: Dict[str, float] = None, query_type: str = SYNTAX_DEPTH_QUERY_TYPE, query_category: str = SYNTAX_DEPTH_QUERY_CATEGORY) -> Dict:
    """评估 SPLADE 检索器

    由于 SPLADE 的 doc_vectors 没有被 pickle 保存（会导致文件过大），
    我们直接调用 search() 方法进行检索，而不是依赖缓存。
    """
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = DEPTH_GROUP_FIELD

    log(f"\n{'='*60}")
    log(f"检索器: SPLADE (稀疏) - SYNTAX_DEPTH/{query_type.upper()}")
    log(f"{'='*60}")

    # 加载 SPLADE 检索器
    retriever = load_splade_retriever()
    doc_ids = retriever.doc_ids

    # SPLADE 的 doc_vectors 没有被 pickle 保存，需要先调用 fit() 重建索引
    if retriever.doc_vectors is None:
        log(f"  [SPLADE] doc_vectors 未保存，正在重建索引...")
        corpus_file = CAT_CONFIG.get('corpus_file', '')
        if os.path.exists(corpus_file):
            documents = []
            with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    if 'asin' in doc and 'title' in doc:
                        documents.append(doc)
                    if len(documents) >= 500000:
                        break
            log(f"  加载了 {len(documents)} 个文档，正在构建 SPLADE 索引...")
            retriever.fit(documents, None)
            log(f"  [SPLADE] 索引构建完成")
        else:
            raise FileNotFoundError(f"SPLADE 索引重建失败: 找不到语料文件 {corpus_file}")

    matched_users = list(user_queries.keys())
    log(f"  用户数: {len(matched_users)}")

    # 加载 SPLADE 查询缓存（预编码的查询向量）
    splade_cache = load_splade_query_cache(query_type, query_category)
    log(f"  [SPLADE] 使用查询缓存 (共 {len(splade_cache)} 条记录)...")

    # 确保倒排索引已构建（触发 lazy initialization）
    retriever.search(["dummy"], top_k=1)

    eval_start = time.time()

    # 分组统计
    group_groups = {g: [] for g in UNIQUE_LEVELS}
    all_metrics = []

    word_bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, float('inf'))]
    word_bin_labels = ['很短(1-15)', '短(15-20)', '中(20-25)', '长(25-30)', '很长(30+)']
    ratio_bins = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
    ratio_bin_labels = ['很低(0-0.05)', '低(0.05-0.1)', '中(0.1-0.2)', '高(0.2-0.5)', '很高(0.5+)']
    word_count_groups = {label: [] for label in word_bin_labels}
    group_ratio_groups = {label: [] for label in ratio_bin_labels}

    # IDF 分层分组
    idf_bin_groups = {label: [] for label in IDF_BIN_LABELS}
    idf_group_cross = defaultdict(list)
    all_query_records = []

    # 获取最大 k 值
    max_k = max(k_values)

    # 收集所有查询用于批量评分
    all_q_data = []
    cached_matched_users = [uid for uid in matched_users if uid in splade_cache]
    skipped_users = len(matched_users) - len(cached_matched_users)
    log(f"  缓存用户命中: {len(cached_matched_users)}/{len(matched_users)}，跳过无缓存用户: {skipped_users}")

    total_candidate_queries = 0
    skipped_missing_queries = 0
    for user_idx, user_id in enumerate(cached_matched_users):
        queries = user_queries[user_id]
        cached_queries = splade_cache[user_id]
        if not isinstance(cached_queries, dict):
            raise TypeError(f"SPLADE query cache for user {user_id} must be dict")
        for q in queries:
            total_candidate_queries += 1
            query_text = q['query']
            if query_text not in cached_queries:
                skipped_missing_queries += 1
                continue
            all_q_data.append({
                'user_id': user_id,
                'query': query_text,
                'relevant_asin': q['asin'],
                'word_count': q['word_count'],
                'group_ratio': q[f'{GROUP_FIELD}_ratio'],
                'q_group': q[GROUP_FIELD],
                'syntax_depth': q['syntax_depth'],
                'target_depth': q['target_depth'],
                'user_avg_depth': q['user_avg_depth'],
                'depth_group_label': q['depth_group_label'],
                'q_vec': cached_queries[query_text]
            })

    log(
        f"  缓存查询命中: {len(all_q_data)}/{total_candidate_queries}，"
        f"跳过无缓存查询: {skipped_missing_queries}"
    )
    log(f"  批量评分: {len(all_q_data)} 个查询")
    if not all_q_data:
        raise ValueError("SPLADE has no cached expression_style queries to evaluate")

    # 预热：确保倒排索引已构建
    inverted_index = retriever._inverted_index
    n_docs = len(retriever.doc_ids)
    log(f"  构建稀疏矩阵...")

    # 构建 term×doc 的稀疏矩阵 (CSR格式)
    from scipy import sparse
    import numpy as np

    # 从倒排索引构建: term_id -> [(doc_idx, weight)]
    # 构建行索引、列索引、数据值
    row_indices = []
    col_indices = []
    data_values = []
    max_doc_term_id = -1
    for term_id, doc_list in inverted_index.items():
        term_id = int(term_id)
        for doc_idx, d_weight in doc_list:
            row_indices.append(term_id)
            col_indices.append(doc_idx)
            data_values.append(d_weight)
            if term_id > max_doc_term_id:
                max_doc_term_id = term_id

    if max_doc_term_id < 0:
        raise ValueError("SPLADE 文档向量为空，无法构建稀疏矩阵")

    max_query_term_id = -1
    for q_data_item in all_q_data:
        for term_id in q_data_item['q_vec'].keys():
            term_id = int(term_id)
            if term_id > max_query_term_id:
                max_query_term_id = term_id

    n_terms = max(max_doc_term_id, max_query_term_id) + 1

    # 构建稀疏矩阵 (term × doc)
    doc_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_terms, n_docs),
        dtype=np.float32
    )
    log(f"  稀疏矩阵构建完成: {n_terms} terms × {n_docs} docs, {len(data_values)} non-zeros")

    n_queries = len(all_q_data)
    log(f"  开始分块矩阵乘法与 top-k 提取...")
    batch_scores = []
    for batch_start in range(0, n_queries, SPLADE_QUERY_BATCH_SIZE):
        batch_end = min(batch_start + SPLADE_QUERY_BATCH_SIZE, n_queries)
        q_rows = []
        q_cols = []
        q_data = []

        for local_q_idx, q_data_item in enumerate(all_q_data[batch_start:batch_end]):
            for term_id, q_weight in q_data_item['q_vec'].items():
                term_id = int(term_id)
                q_rows.append(local_q_idx)
                q_cols.append(term_id)
                q_data.append(q_weight)

        query_matrix = sparse.csr_matrix(
            (q_data, (q_rows, q_cols)),
            shape=(batch_end - batch_start, n_terms),
            dtype=np.float32
        )
        score_matrix = query_matrix @ doc_matrix

        for row_idx in range(score_matrix.shape[0]):
            global_idx = batch_start + row_idx
            if (global_idx + 1) % 500 == 0 or global_idx + 1 == n_queries:
                log(f"    Top-k提取进度: {global_idx + 1}/{n_queries}")

            row = score_matrix.getrow(row_idx)
            if row.nnz == 0:
                batch_scores.append([])
                continue

            if row.nnz <= max_k:
                order = np.argsort(row.data)[::-1]
            else:
                partition = np.argpartition(row.data, -max_k)[-max_k:]
                order = partition[np.argsort(row.data[partition])[::-1]]

            top_doc_indices = row.indices[order]
            top_scores = row.data[order]
            batch_scores.append([
                (retriever.doc_ids[int(doc_idx)], float(score))
                for doc_idx, score in zip(top_doc_indices, top_scores)
            ])

        del query_matrix
        del score_matrix

    if len(batch_scores) != len(all_q_data):
        raise RuntimeError(f"SPLADE result count mismatch: scores={len(batch_scores)}, queries={len(all_q_data)}")

    # 重新组织结果用于后续统计。只处理实际进入批量评分的缓存查询。
    for q_idx, q in enumerate(all_q_data):
        query_text = q['query']
        relevant_asin = q['relevant_asin']
        word_count = q['word_count']
        group_ratio = q['group_ratio']
        q_group = q['q_group']

        retrieved_with_scores = batch_scores[q_idx]

        retrieved_asins = [asin for asin, score in retrieved_with_scores]

        metrics = compute_metrics(relevant_asin, retrieved_asins, k_values)
        all_metrics.append(metrics)
        group_groups[q_group].append(metrics)

        # 计算 IDF
        q_idf = 0.0

        # 记录原始数据
        all_query_records.append({
            'user_id': q['user_id'],
            'asin': relevant_asin,
            f'{GROUP_FIELD}': q_group,
            'syntax_depth': q['syntax_depth'],
            'target_depth': q['target_depth'],
            'user_avg_depth': q['user_avg_depth'],
            'depth_group_label': q['depth_group_label'],
            'mean_idf': q_idf,
            'query_length': word_count,
            f'{GROUP_FIELD}_ratio': group_ratio,
            'p_at1': float(metrics.get('P@1', 0.0)),
            'p_at3': float(metrics.get('P@3', 0.0)),
            'p_at5': float(metrics.get('P@5', 0.0)),
            'p_at10': float(metrics.get('P@10', 0.0)),
            'n_at10': float(metrics.get('N@10', 0.0)),
            'mrr_at10': float(metrics.get('MR@10', 0.0)),
            'hit_at10': float(metrics.get('H@10', 0.0)),
        })

        # 分组统计
        for (low, high), label in zip(word_bins, word_bin_labels):
            if low <= word_count < high:
                word_count_groups[label].append(metrics)
                break

        for (low, high), label in zip(ratio_bins, ratio_bin_labels):
            if low <= group_ratio < high:
                group_ratio_groups[label].append(metrics)
                break

        for (low, high), label in zip(IDF_BINS, IDF_BIN_LABELS):
            if low <= q_idf < high:
                idf_bin_groups[label].append(metrics)
                idf_group_cross[(label, q_group)].append(metrics)
                break

        if (q_idx + 1) % 100 == 0 or q_idx + 1 == len(all_q_data):
            elapsed = time.time() - eval_start
            log(f"    进度: {q_idx+1}/{len(all_q_data)} ({100*(q_idx+1)/len(all_q_data):.1f}%)")

    eval_time = time.time() - eval_start
    log(f"  评估完成，总耗时: {eval_time:.1f}秒")
    overall_metrics = compute_average_metrics(all_metrics, k_values)

    group_metrics = {}
    group_counts = {}
    for group in UNIQUE_LEVELS:
        if group_groups[group]:
            group_metrics[group] = compute_average_metrics(group_groups[group], k_values)
            group_counts[group] = len(group_groups[group])
        else:
            group_metrics[group] = {k: 0.0 for k in [f'P@{i}' for i in k_values] + [f'N@{i}' for i in k_values] + [f'MR@{i}' for i in k_values] + [f'H@{i}' for i in k_values]}
            group_counts[group] = 0

    # 计算 word_count 分组统计
    word_count_analysis = {}
    for label in word_bin_labels:
        if word_count_groups[label]:
            word_count_analysis[label] = {
                'count': len(word_count_groups[label]),
                'metrics': compute_average_metrics(word_count_groups[label], k_values)
            }

    # 计算 group_ratio 分组统计
    group_ratio_analysis = {}
    for label in ratio_bin_labels:
        if group_ratio_groups[label]:
            group_ratio_analysis[label] = {
                'count': len(group_ratio_groups[label]),
                'metrics': compute_average_metrics(group_ratio_groups[label], k_values)
            }

    # 计算 IDF 分组统计
    idf_analysis = {}
    for label in IDF_BIN_LABELS:
        if idf_bin_groups[label]:
            idf_analysis[label] = {
                'count': len(idf_bin_groups[label]),
                'metrics': compute_average_metrics(idf_bin_groups[label], k_values)
            }

    # 计算 IDF × group 交叉分组统计
    idf_group_analysis = {}
    for (idf_label, group_val), metrics_list in idf_group_cross.items():
        if metrics_list:
            idf_group_analysis[(idf_label, group_val)] = {
                'count': len(metrics_list),
                'metrics': compute_average_metrics(metrics_list, k_values)
            }

    return {
        'retriever': 'splade', 'dim': 0, 'type': 'sparse', 'num_users': len(matched_users),
        'num_queries': len(all_metrics), 'eval_time_seconds': eval_time,
        'metrics': overall_metrics, 'group_metrics': group_metrics, 'group_counts': group_counts,
        'raw_metrics_per_group': group_groups, 'all_raw_metrics': all_metrics,
        'word_count_analysis': word_count_analysis,
        'group_ratio_analysis': group_ratio_analysis,
        'idf_analysis': idf_analysis,
        'idf_group_cross': idf_group_analysis,
        'all_query_records': all_query_records
    }


# ============ 表格打印 ============


def print_summary_table_wide(all_results: List[Dict], query_type: str = 'CORRECT'):
    """宽格式汇总表：每行一个检索器，列是 (指标 × 分组) 的所有组合

    格式：
    检索器 | P@1_ACL0 | P@1_ACL1 | ... | P@10_ACL3 | 平均
    """
    global GROUP_FIELD
    log(f"\n{'='*100}")
    log(f"汇总表（宽格式）| {query_type} | 每行一个检索器，列为 (指标×分组)")
    log(f"{'='*100}")

    metrics = ['P@1', 'P@3', 'P@5', 'P@10', 'N@10', 'MR@10', 'H@10']
    groups = [get_group_display(g) for g in UNIQUE_LEVELS]

    # 构建表头 - 使用固定宽度列（每列12字符，右对齐）
    COL_W = 12
    header = f"{'检索器':<10}"
    for m in metrics:
        for g in groups:
            label = f"{m}_{g}"
            header += f" {label:>{COL_W}}"
        header += f" {m:>{COL_W}}"
    header += f" {'总平均':>{COL_W}}"
    log(header)
    log("-" * 100)

    # 构建数据
    retrievers = sorted(set(r['retriever'] for r in all_results))

    for retriever in retrievers:
        r = next((x for x in all_results if x['retriever'] == retriever), None)
        if not r:
            continue

        row = f"{retriever:<10}"
        all_vals = []

        for m in metrics:
            metric_vals = []
            for group in UNIQUE_LEVELS:
                val = r['group_metrics'].get(group, {}).get(m, 0.0)
                row += f" {val:>{COL_W}.4f}"
                metric_vals.append(val)
                all_vals.append(val)
            metric_avg = sum(metric_vals) / len(metric_vals) if metric_vals else 0.0
            row += f" {metric_avg:>{COL_W}.4f}"
            all_vals.append(metric_avg)

        total_avg = sum(all_vals) / len(all_vals) if all_vals else 0.0
        row += f" {total_avg:>{COL_W}.4f}"
        log(row)

    log("-" * 100)


def print_hit10_complexity_table(all_results: List[Dict]):
    """按当前查询集合展示 H@10 对比表。"""
    sep_width = 100
    log(f"\n{'='*sep_width}")
    log("H@10 查询集合对比表")
    log(f"{'='*sep_width}")

    COL_W = 18
    header = f"{'检索器':<12}"
    for group in UNIQUE_LEVELS:
        header += f" {get_group_display(group):>{COL_W}}"
    header += f" {'整体H@10':>{COL_W}}"
    header += f" {'平均差值':>{COL_W}}"
    log(header)
    log("-" * sep_width)

    for result in sorted(all_results, key=lambda item: item['retriever']):
        retriever = result['retriever']
        row = f"{retriever:<12}"
        hit10_values = []
        for group in UNIQUE_LEVELS:
            value = result['group_metrics'].get(group, {}).get('H@10', 0.0)
            hit10_values.append(value)
            row += f" {value:>{COL_W}.4f}"
        overall = result['metrics'].get('H@10', 0.0)
        row += f" {overall:>{COL_W}.4f}"
        mean_abs_diff = 0.0
        row += f" {mean_abs_diff:>{COL_W}.4f}"
        log(row)

    log("-" * sep_width)


def print_hit10_exact_depth_table(all_results: List[Dict], depths: Optional[Tuple[int, ...]] = None):
    return


# ============ 主流程 ============
def print_query_type_comparison(all_results_by_type: Dict[str, List[Dict]], k_values: List[int], category_name: str = ''):
    """打印 correct vs noisy 配对比较表（宽格式）

    对于每个 noisy 查询，找到其同 CCOMP 级别的 correct 查询进行配对
    配对 key: (user_id, asin, ccomp)
    输出宽格式表格，包含所有指标
    """
    log("\n" + "=" * 100)
    log(f"{category_name} CORRECT vs NOISY 配对比较（宽格式 | 基于 (user_id, asin, {GROUP_FIELD.upper()}) 匹配）")
    log("=" * 100)

    # 获取所有检索器
    retrievers = set()
    for qt_results in all_results_by_type.values():
        for r in qt_results:
            retrievers.add(r['retriever'])
    retrievers = sorted(retrievers)

    # 定义所有指标及其对应的字段
    METRICS = [
        ('P@1', 'p_at1'),
        ('P@3', 'p_at3'),
        ('P@5', 'p_at5'),
        ('P@10', 'p_at10'),
        ('N@10', 'n_at10'),
        ('MR@10', 'mrr_at10'),
        ('H@10', 'hit_at10'),
    ]

    group_field = GROUP_FIELD  # 'ccomp' or 'acl'

    # 构建宽格式表头 - 每个指标3个子列，固定宽度
    # 格式: 检索器 | CORR | NOISY | DIFF | CORR | NOISY | DIFF | ...
    header = f"{'检索器':<10}"
    sep = " " * 1
    for metric_name, _ in METRICS:
        header += sep + f"{metric_name:>7} {metric_name:>7} {metric_name:>7}"
    log(header)
    log("-" * 100)

    # 对每个检索器计算所有指标的配对比较
    for retriever in retrievers:
        correct_results = next((x for x in all_results_by_type.get('correct', []) if x['retriever'] == retriever), None)
        noisy_results = next((x for x in all_results_by_type.get('noisy', []) if x['retriever'] == retriever), None)

        if not correct_results or not noisy_results:
            continue

        row = f"{retriever:<10}"

        for metric_name, metric_field in METRICS:
            correct_dict = {}
            noisy_dict = {}

            for rec in correct_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                correct_dict[key] = rec.get(metric_field, 0)

            for rec in noisy_results.get('all_query_records', []):
                key = (rec['user_id'], rec.get('asin', ''), rec.get(group_field, -1))
                noisy_dict[key] = rec.get(metric_field, 0)

            # 找到共同的查询对
            common_keys = set(correct_dict.keys()) & set(noisy_dict.keys())

            if not common_keys:
                row += sep + f"{'N/A':>7} {'N/A':>7} {'N/A':>7}"
                continue

            # 计算配对差异
            correct_vals = [correct_dict[k] for k in common_keys]
            noisy_vals = [noisy_dict[k] for k in common_keys]
            diffs = [noisy_vals[i] - correct_vals[i] for i in range(len(common_keys))]

            mean_correct = sum(correct_vals) / len(correct_vals)
            mean_noisy = sum(noisy_vals) / len(noisy_vals)
            mean_diff = sum(diffs) / len(diffs)

            row += sep + f"{mean_correct:7.4f} {mean_noisy:7.4f} {mean_diff:+7.4f}"
        log(row)

    log("-" * 100)


def main():
    global GROUP_FIELD, UNIQUE_LEVELS
    GROUP_FIELD = DEPTH_GROUP_FIELD
    UNIQUE_LEVELS = DEPTH_GROUP_ORDER.copy()

    log("=" * 60)
    log("快速全量评估 - 最终 expression_style 查询缓存")
    log(f"类别: {CATEGORY_NAME}")
    log(f"查询缓存目录: {os.path.join(QUERY_CACHE_BASE_DIR, 'syntax_depth_correct_query')}")
    log("=" * 60)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    k_values = [1, 3, 5, 10]
    word_idf = None

    query_category = SYNTAX_DEPTH_QUERY_CATEGORY
    query_type = SYNTAX_DEPTH_QUERY_TYPE
    available_retrievers = list_retrievers_with_query_cache(query_type, query_category)
    missing_retrievers = [r for r in RETRIEVERS if r not in available_retrievers]
    log(f"  可用查询缓存检索器: {available_retrievers}")
    if missing_retrievers:
        log(f"  未找到查询缓存，跳过检索器: {missing_retrievers}")
    if not available_retrievers:
        raise FileNotFoundError(
            f"No expression_style query cache files found under "
            f"{os.path.join(QUERY_CACHE_BASE_DIR, 'syntax_depth_correct_query')}"
        )

    user_queries, user_to_group, _ = load_user_queries(query_type, query_category)
    log(f"  expression_style 用户数: {len(user_queries)}")

    all_word_counts = [q['word_count'] for user_qs in user_queries.values() for q in user_qs]
    if not all_word_counts:
        raise ValueError("No expression_style queries loaded for evaluation")
    q25, q50, q75 = np.percentile(all_word_counts, [25, 50, 75])
    log(f"  Query长度分布: min={min(all_word_counts)}, Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}, max={max(all_word_counts)}")

    log("\n" + "#" * 80)
    log("# SYNTAX_DEPTH 查询类型: CORRECT")
    log("#" * 80)

    query_type_results = []

    # 评估密集检索器
    for retriever_name in DENSE_RETRIEVERS:
        if retriever_name not in available_retrievers:
            continue
        result = evaluate_dense_retriever(
            retriever_name, user_queries, user_to_group, k_values, word_idf, query_type, query_category
        )
        result['query_type'] = query_type
        result['query_category'] = query_category
        result['group_field'] = GROUP_FIELD
        query_type_results.append(result)

    # 评估 BM25 查询结果缓存
    if 'bm25' in available_retrievers:
        result = evaluate_bm25_retriever(user_queries, user_to_group, k_values, word_idf, query_type, query_category)
        result['query_type'] = query_type
        result['query_category'] = query_category
        result['group_field'] = GROUP_FIELD
        query_type_results.append(result)

    # 评估 ColBERTv2（使用预生成的 query token embedding 缓存）
    for retriever_name in COLBERTV2_RETRIEVERS:
        if retriever_name not in available_retrievers:
            continue
        result = evaluate_cached_result_retriever(
            retriever_name, user_queries, user_to_group, k_values, word_idf, query_type, query_category
        )
        result['query_type'] = query_type
        result['query_category'] = query_category
        result['group_field'] = GROUP_FIELD
        query_type_results.append(result)

    # 评估 SPLADE 查询向量缓存
    for retriever_name in SPARSE_RETRIEVERS:
        if retriever_name not in available_retrievers:
            continue
        result = evaluate_splade_retriever(user_queries, user_to_group, k_values, word_idf, query_type, query_category)
        result['query_type'] = query_type
        result['query_category'] = query_category
        result['group_field'] = GROUP_FIELD
        query_type_results.append(result)

    if not query_type_results:
        raise ValueError("No retriever produced evaluation results from existing expression_style query caches")

    all_results_by_category_and_type = {
        (query_category, query_type): query_type_results
    }
    all_results_combined = []
    for r in query_type_results:
        r_copy = r.copy()
        r_copy['query_category'] = query_category
        all_results_combined.append(r_copy)

    log("\n")
    log("=" * 100)
    log(f"========== 评估结果汇总 [{CATEGORY_NAME}] ==========")
    log("=" * 100)

    print_summary_table_wide(query_type_results, f'{CATEGORY_NAME} SYNTAX_DEPTH-CORRECT')
    print_hit10_complexity_table(query_type_results)

    # 保存结果（处理 tuple key 等不可 JSON 序列化的问题）
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    output_file = os.path.join(OUTPUT_DIR, "retrieval_syntax_depth_summary.json")
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'category_name': CATEGORY_NAME,
            'query_source_file': SYNTAX_DEPTH_QUERY_FILE,
            'query_types': QUERY_TYPES,
            'query_categories': QUERY_CATEGORIES,
            'group_field': GROUP_FIELD,
            'results_by_category_and_type': sanitize_for_json(all_results_by_category_and_type),
            'all_results_combined': sanitize_for_json(all_results_combined),
        }, f, indent=2, default=str)
    log(f"\n结果已保存到: {output_file}")

    log("\n" + "=" * 60)
    log("评估完成!")
    log("=" * 60)

if __name__ == "__main__":
    main()
