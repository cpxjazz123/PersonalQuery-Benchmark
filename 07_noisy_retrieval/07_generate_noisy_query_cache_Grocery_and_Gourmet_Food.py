#!/usr/bin/env python3
"""
生成并预存储检索器的 noisy 查询缓存

这个脚本将：
1. 从 07_inject_noisy 维护的 query_by_syntax_depth 统一文件加载 expression_style 查询对
2. 将 syntax_depth_query.query 作为 clean query，将 noisy_injection.noisy_query 作为 noisy query
3. 为每个检索器编码 clean/noisy 两套查询
4. 保存缓存到磁盘以加速后续评估

使用方法：
    python3 09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.py > /workspace/logs/09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.log 2> /workspace/logs/09_generate_noisy_query_cache_Grocery_and_Gourmet_Food.err
"""

import os
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
for cache_dir in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
    os.environ["TRITON_CACHE_DIR"],
):
    os.makedirs(cache_dir, exist_ok=True)

import sys
import importlib.util
import json
import pickle
import io
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

current_dir = Path(__file__).parent.resolve()
retrieval_root = current_dir.parent / "08_retrieval"
personquery_root = retrieval_root.parent

sys.path.insert(0, str(retrieval_root))
sys.path.insert(0, str(personquery_root))

from utils.retrievers import (
    E5Retriever, BGERetriever,
    STARRetriever, MiniLMRetriever, BM25,
    ANCERetriever, SPLADERetriever,
    select_cuda_toolkit_for_colbert_extension_build,
    configure_host_compiler_for_colbert_extension_build,
    validate_cuda_toolkit_for_colbert,
    configure_cuda_env_for_colbert_extension_build,
)
from config import get_category_config, get_global_paths

# ============ 配置加载 ============
CATEGORY_NAME = "Grocery_and_Gourmet_Food"
CAT_CONFIG = get_category_config(CATEGORY_NAME)
GLOBAL_PATHS = get_global_paths()

# expression_style 查询文件（07 生成并持续回写 noisy_injection 的统一文件）
SYNTAX_DEPTH_QUERY_FILE = (
    f"/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{CATEGORY_NAME}/"
    "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
)
# LambdaMART 注入的 noisy_query.json 文件（新格式）
NOISY_QUERY_JSON_FILE = os.path.join(
    GLOBAL_PATHS.get("inject_noisy", f"/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy"),
    CATEGORY_NAME,
    "noisy_query.json"
)
CLEAN_MODE = "syntax_depth_correct"
NOISY_MODE = "syntax_depth_noisy"
SYNTAX_DEPTH_INJECTION_SOURCE = "final_strict_query_no_depth_constraint"

# 缓存目录 - 使用与 08 相同的目录结构
CACHE_DIR = CAT_CONFIG['query_cache_dir']
BM25_RETRIEVER_CACHE_DIR = CAT_CONFIG['retriever_cache_dir']

AVAILABLE_RETRIEVERS = {
    # 'GRITLM': GritLMRetriever,
    'BGE': BGERetriever,
    'E5': E5Retriever,
    'MiniLM': MiniLMRetriever,
    'STAR': STARRetriever,
    'ANCE': ANCERetriever,
    'ColBERTv2': None,
    'SPLADE': SPLADERetriever,
    'BM25': None,
}

COLBERTV2_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERTV2_QUERY_BATCH_SIZE = 128


def log_with_timestamp(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_colbertv2_build_module():
    module_path = retrieval_root / "08_build_retriever_indices_Grocery_and_Gourmet_Food.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Required ColBERTv2 build module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("build_retriever_indices_grocery_and_gourmet_food", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ColBERTv2 build module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return module


def configure_colbertv2_runtime() -> None:
    major, minor = torch.cuda.get_device_capability()
    ext_dir = os.path.join(
        "/home/wlia0047/ar57_scratch/wenyu/torch_extensions",
        f"colbertv2_cuda125_sm{major}{minor}",
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log_with_timestamp(f"[BOOT] ColBERT query shared TORCH_EXTENSIONS_DIR = {ext_dir}")
    select_cuda_toolkit_for_colbert_extension_build()
    configure_host_compiler_for_colbert_extension_build()
    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()


def collect_unique_query_texts(queries: List[Dict], mode: str) -> List[str]:
    unique_queries = []
    seen = set()

    for index, item in enumerate(queries):
        query_text = item.get('query')
        if not isinstance(query_text, str):
            raise TypeError(f"{mode} query must be a string at index {index}, got {type(query_text).__name__}")
        if not query_text:
            raise ValueError(f"{mode} query is empty at index {index}")
        if query_text not in seen:
            seen.add(query_text)
            unique_queries.append(query_text)

    if not unique_queries:
        raise ValueError(f"No valid queries collected for ColBERTv2 mode: {mode}")

    return unique_queries


def load_colbertv2_checkpoint_for_query_encoding():
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 query embedding cache generation requires CUDA")

    configure_colbertv2_runtime()

    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    config = ColBERTConfig(
        checkpoint=COLBERTV2_MODEL_NAME,
        query_maxlen=32,
    )
    checkpoint = Checkpoint(COLBERTV2_MODEL_NAME, colbert_config=config, verbose=1)
    return checkpoint.cuda()


def encode_colbertv2_query_texts(checkpoint, query_texts: List[str]) -> Dict[str, np.ndarray]:
    if not query_texts:
        raise ValueError("ColBERTv2 query text list is empty")

    embeddings = checkpoint.queryFromText(
        query_texts,
        bsize=COLBERTV2_QUERY_BATCH_SIZE,
        to_cpu=True,
    )
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"ColBERTv2 query encoder returned {type(embeddings).__name__}, expected torch.Tensor")
    if embeddings.ndim != 3:
        raise ValueError(f"ColBERTv2 query embeddings must be 3D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(query_texts):
        raise RuntimeError(
            f"ColBERTv2 query embedding count mismatch: expected {len(query_texts)}, got {embeddings.shape[0]}"
        )

    return {
        query_text: np.asarray(embeddings[index].detach().cpu().numpy(), dtype=np.float32)
        for index, query_text in enumerate(query_texts)
    }


def save_colbertv2_query_embedding_cache(cache: Dict[str, Dict[str, np.ndarray]], mode: str) -> str:
    subdir = get_cache_subdir(mode)
    os.makedirs(subdir, exist_ok=True)
    cache_path = os.path.join(subdir, f"colbertv2__{mode}_cache.pkl")
    tmp_path = f"{cache_path}.tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp_path, cache_path)

    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    total_cached = sum(len(user_cache) for user_cache in cache.values())
    log_with_timestamp(f"  ✓ ColBERTv2 query embedding 缓存已保存: {cache_path}")
    log_with_timestamp(f"    - 用户数: {len(cache)}")
    log_with_timestamp(f"    - 查询数: {total_cached}")
    log_with_timestamp(f"    - 文件大小: {file_size_mb:.2f} MB")
    return cache_path


def generate_colbertv2_query_embedding_cache_for_mode(
    checkpoint,
    queries: List[Dict],
    queries_by_user: Dict[str, List[Dict]],
    mode: str,
) -> Dict[str, object]:
    query_texts = collect_unique_query_texts(queries, mode)
    log_with_timestamp(
        f"  开始生成 ColBERTv2 query embedding 缓存 ({mode}): "
        f"unique_queries={len(query_texts)}, batch_size={COLBERTV2_QUERY_BATCH_SIZE}"
    )

    start_time = time.time()
    encoded_by_query = encode_colbertv2_query_texts(checkpoint, query_texts)

    result_cache: Dict[str, Dict[str, np.ndarray]] = {uid: {} for uid in queries_by_user.keys()}
    for uid, user_queries in queries_by_user.items():
        for item in user_queries:
            query_text = item.get('query')
            if query_text not in encoded_by_query:
                raise KeyError(f"ColBERTv2 encoded query missing for user={uid}, mode={mode}, query={query_text}")
            result_cache[uid][query_text] = encoded_by_query[query_text]

    cache_path = save_colbertv2_query_embedding_cache(result_cache, mode)
    elapsed = time.time() - start_time
    total_cached = sum(len(user_cache) for user_cache in result_cache.values())

    return {
        'mode': mode,
        'cache_path': cache_path,
        'query_count': total_cached,
        'unique_query_count': len(query_texts),
        'elapsed_seconds': elapsed,
    }


def generate_colbertv2_cache_from_query_types(query_types: List[Tuple[str, List[Dict], Dict[str, List[Dict]], str]]) -> Dict[str, object]:
    checkpoint = load_colbertv2_checkpoint_for_query_encoding()

    summaries = []
    total_cached = 0
    try:
        for query_type, queries, queries_by_user, mode in query_types:
            if not queries:
                log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")
                continue

            summary = generate_colbertv2_query_embedding_cache_for_mode(
                checkpoint,
                queries,
                queries_by_user,
                mode,
            )
            summaries.append(summary)
            total_cached += summary['query_count']
            log_with_timestamp(
                f"  ✓ {query_type} {mode} ColBERTv2 query embedding 缓存: "
                f"{summary['query_count']} 条, unique={summary['unique_query_count']}, "
                f"耗时 {summary['elapsed_seconds']:.1f}s"
            )
    finally:
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return {
        'total_cached': total_cached,
        'summaries': summaries,
    }


def load_syntax_depth_query_records(file_path: str) -> List[Dict]:
    """读取 07 维护的 query_by_syntax_depth 统一记录文件。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"expression_style query 文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"expression_style query 文件顶层必须是 list，实际为 {type(data).__name__}: {file_path}")
    if not data:
        raise ValueError(f"expression_style query 文件为空: {file_path}")
    return data


def _build_query_entry(
    *,
    user_id: str,
    asin: str,
    clean_query: str,
    noisy_query: str,
    query_text: str,
    query_type: str,
    source_query_field: str,
    injection_source: str,
) -> Dict:
    original_query = clean_query

    if not isinstance(user_id, str) or not user_id:
        raise ValueError(f"query item has invalid user_id: {user_id!r}")
    if not isinstance(asin, str) or not asin:
        raise ValueError(f"query item has invalid asin for user={user_id}: {asin!r}")
    if not isinstance(original_query, str) or not original_query:
        raise ValueError(f"query item has invalid original_query for user={user_id}, asin={asin}: {original_query!r}")
    if not isinstance(noisy_query, str) or not noisy_query:
        raise ValueError(f"query item has invalid noisy_query for user={user_id}, asin={asin}: {noisy_query!r}")
    if not isinstance(query_text, str) or not query_text:
        raise ValueError(
            f"query item has invalid {source_query_field} for user={user_id}, asin={asin}: {query_text!r}"
        )
    if not isinstance(injection_source, str) or not injection_source:
        raise ValueError(f"query item has invalid injection_source for user={user_id}, asin={asin}: {injection_source!r}")

    return {
        'user_id': user_id,
        'asin': asin,
        'query': query_text,
        'original_query': original_query,
        'noisy_query': noisy_query,
        'source': 'syntax_depth',
        'query_type': query_type,
        'source_query_field': source_query_field,
        'injection_source': injection_source,
    }


def load_syntax_depth_query_pairs() -> Dict[str, List[Dict]]:
    """从 LambdaMART 生成的 noisy_query.json 文件加载 clean/noisy 查询对。

    新的 noisy_query.json 格式：
    {
        "uid": "...",
        "asin": "...",
        "clean_query": "...",
        "noisy_query": "...",
        "query_rewritten": true,
        ...
    }
    """
    if not os.path.exists(NOISY_QUERY_JSON_FILE):
        raise FileNotFoundError(f"noisy_query.json 文件不存在: {NOISY_QUERY_JSON_FILE}")

    with open(NOISY_QUERY_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"noisy_query.json 顶层必须是 list，实际为 {type(data).__name__}")

    clean_queries = []
    noisy_queries = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"noisy_query.json row must be dict at index {index}, got {type(item).__name__}")

        # 支持 uid 或 user_id 字段
        user_id = item.get('uid') or item.get('user_id')
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"noisy_query.json row has invalid user_id/uid at index {index}: {item}")
        asin = item.get('asin')
        if not isinstance(asin, str) or not asin.strip():
            raise ValueError(f"noisy_query.json row has invalid asin for user={user_id} at index {index}")
        clean_query = item.get('clean_query')
        if not isinstance(clean_query, str) or not clean_query.strip():
            raise ValueError(f"noisy_query.json row has invalid clean_query at index {index}")
        noisy_query = item.get('noisy_query')
        if not isinstance(noisy_query, str) or not noisy_query.strip():
            raise ValueError(f"noisy_query.json row has invalid noisy_query at index {index}")

        clean_query = clean_query.strip()
        noisy_query = noisy_query.strip()

        clean_queries.append(_build_query_entry(
            user_id=user_id,
            asin=asin,
            clean_query=clean_query,
            noisy_query=noisy_query,
            query_text=clean_query,
            query_type='clean',
            source_query_field='clean_query',
            injection_source=SYNTAX_DEPTH_INJECTION_SOURCE,
        ))
        noisy_queries.append(_build_query_entry(
            user_id=user_id,
            asin=asin,
            clean_query=clean_query,
            noisy_query=noisy_query,
            query_text=noisy_query,
            query_type='noisy',
            source_query_field='noisy_query',
            injection_source=SYNTAX_DEPTH_INJECTION_SOURCE,
        ))

    log_with_timestamp(
        f"✓ 从 {NOISY_QUERY_JSON_FILE} 加载了 {len(clean_queries)} 条 clean 查询和 "
        f"{len(noisy_queries)} 条 noisy 查询"
    )
    return {
        'clean_queries': clean_queries,
        'noisy_queries': noisy_queries,
    }


def _build_queries_by_user(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """将查询列表按 user_id 分组。"""
    by_user = defaultdict(list)
    for item in queries:
        for field in ('user_id', 'query', 'asin'):
            if field not in item:
                raise ValueError(f"query item missing required field '{field}': {item}")
        user_id = item['user_id']
        query_text = item['query']
        asin = item['asin']
        if not user_id:
            raise ValueError(f"query item has empty user_id: {item}")
        if not query_text:
            raise ValueError(f"query item has empty query for user={user_id}: {item}")
        if not asin:
            raise ValueError(f"query item has empty asin for user={user_id}: {item}")
        by_user[user_id].append(item)
    return dict(by_user)


def get_retriever(retriever_name: str):
    """获取检索器实例"""
    if retriever_name == 'BM25':
        return BM25()
    retriever_class = AVAILABLE_RETRIEVERS.get(retriever_name)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever: {retriever_name}")
    return retriever_class()


class _StdoutToStderr:
    """上下文管理器，将 stdout 重定向到 stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return self
    def __exit__(self, *args):
        sys.stdout = self._original_stdout


def initialize_cache_dir():
    """初始化缓存目录"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(get_cache_subdir(CLEAN_MODE), exist_ok=True)
    os.makedirs(get_cache_subdir(NOISY_MODE), exist_ok=True)
    log_with_timestamp(f"✓ 缓存目录: {CACHE_DIR}")


def get_cache_subdir(mode: str) -> str:
    if mode == CLEAN_MODE:
        return os.path.join(CACHE_DIR, "syntax_depth_correct_query")
    if mode == NOISY_MODE:
        return os.path.join(CACHE_DIR, "syntax_depth_noisy_query")
    raise ValueError(f"Unsupported noisy cache mode: {mode}")


def get_mode_suffix(mode: str) -> str:
    if mode == CLEAN_MODE:
        return CLEAN_MODE
    if mode == NOISY_MODE:
        return NOISY_MODE
    raise ValueError(f"Unsupported noisy cache mode: {mode}")


def load_retriever_for_query_encoding(retriever_name: str):
    if retriever_name == 'BM25':
        log_with_timestamp("  初始化检索器 BM25...")
        bm25_path = None
        for f in os.listdir(BM25_RETRIEVER_CACHE_DIR):
            if f.startswith('bm25_') and f.endswith('.pkl'):
                bm25_path = os.path.join(BM25_RETRIEVER_CACHE_DIR, f)
                break
        if bm25_path is None:
            raise FileNotFoundError(f"BM25 retriever cache not found in {BM25_RETRIEVER_CACHE_DIR}")

        with open(bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
        log_with_timestamp(f"  ✓ BM25 加载完成")
        return bm25

    retriever_class = AVAILABLE_RETRIEVERS.get(retriever_name)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    log_with_timestamp(f"  初始化检索器 {retriever_name}...")
    with _StdoutToStderr():
        retriever = retriever_class()
    log_with_timestamp(f"  ✓ 检索器初始化完成，模型已加载")
    return retriever


def get_cache_file_path(retriever_name: str, user_id: str, mode: str) -> str:
    """获取缓存文件路径"""
    subdir = get_cache_subdir(mode)
    suffix = get_mode_suffix(mode)
    filename = f"{retriever_name.lower()}_{user_id}_{suffix}_cache.pkl"
    return os.path.join(subdir, filename)


def _encode_and_save_cache(
    retriever_name: str,
    retriever,
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为检索器编码并保存查询缓存"""
    if retriever_name == 'BM25':
        # BM25 使用不同的处理方式
        return _encode_and_save_bm25_cache(retriever, queries, by_user, mode)

    # SPLADE 使用不同的缓存格式
    if retriever_name == 'SPLADE':
        return _encode_and_save_splade_cache(retriever, queries, by_user, mode)

    cache = {}
    total_items = sum(len(user_queries) for user_queries in by_user.values())
    processed_items = 0

    for user_id, user_queries in by_user.items():
        user_cache = {}
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    raise ValueError(f"{mode} query is empty for user={user_id}, asin={q.get('asin', '')}")
                embedding = retriever.encode_query(text)

                if not isinstance(embedding, np.ndarray):
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    else:
                        embedding = np.array(embedding)

                user_cache[text] = embedding
                processed_items += 1
                if processed_items % 100 == 0 or processed_items == total_items:
                    log_with_timestamp(
                        f"    进度 [{retriever_name}] {mode}: {processed_items}/{total_items}"
                    )
            except Exception as e:
                log_with_timestamp(f"      ❌ 编码失败 [{retriever_name}] 查询: {text[:40]}... 错误: {str(e)[:100]}")
                raise

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        cache_file = get_cache_file_path(retriever_name, "", mode)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ 缓存已保存: {cache_file}")

    total_cached = sum(len(user_cache) for user_cache in cache.values())
    return total_cached


def _encode_and_save_splade_cache(
    retriever,
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为 SPLADE 编码并保存查询缓存（稀疏向量格式）"""
    # SPLADE 缓存格式：{user_id: {query_text: sparse_vec_dict}}
    cache = {}
    total_items = sum(len(user_queries) for user_queries in by_user.values())
    processed_items = 0

    for user_id, user_queries in by_user.items():
        user_cache = {}
        for q in user_queries:
            try:
                text = q.get('query', '')
                if not text:
                    raise ValueError(f"{mode} query is empty for user={user_id}, asin={q.get('asin', '')}")
                # SPLADE 返回 Dict[str, float] (sparse vector)
                sparse_vec = retriever.encode_query(text)
                user_cache[text] = sparse_vec
                processed_items += 1
                if processed_items % 100 == 0 or processed_items == total_items:
                    log_with_timestamp(f"    进度 [SPLADE] {mode}: {processed_items}/{total_items}")
            except Exception as e:
                log_with_timestamp(f"      ❌ SPLADE 编码失败: {text[:40]}... 错误: {str(e)[:100]}")
                raise

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        # SPLADE 使用特殊路径格式，与评估代码兼容
        cache_file = os.path.join(get_cache_subdir(mode), f"splade__{mode}_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ SPLADE 缓存已保存: {cache_file}")

    total_cached = sum(len(user_cache) for user_cache in cache.values())
    return total_cached


def _encode_and_save_bm25_cache(
    bm25,
    queries: List[Dict],
    by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    """为 BM25 编码并保存查询缓存"""
    cache = {}
    failed_count = 0
    total_items = len(queries)

    for index, q in enumerate(queries):
        try:
            text = q.get('query', '')
            if not text:
                raise ValueError(f"BM25 {mode} query is empty at index {index}")
            if text in cache:
                continue
            cache[text] = bm25.search(text, top_k=100)
            if (index + 1) % 100 == 0 or (index + 1) == total_items:
                log_with_timestamp(f"    进度 [BM25] {mode}: {index + 1}/{total_items}")
        except Exception as e:
            log_with_timestamp(f"      ❌ BM25 搜索失败: {text[:40]}... 错误: {str(e)[:100]}")
            failed_count += 1
            import sys
            sys.exit(1)

    if cache:
        cache_file = os.path.join(get_cache_subdir(mode), f"bm25__{mode}_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_with_timestamp(f"  ✓ BM25 缓存已保存: {cache_file}")

    if failed_count > 0:
        log_with_timestamp(f"      ⚠️  共有 {failed_count} 个查询编码失败")

    return len(cache)


def clear_noisy_cache() -> int:
    """删除旧的 expression_style clean/noisy 查询缓存文件"""
    if not os.path.exists(CACHE_DIR):
        return 0

    deleted_count = 0
    for mode in (CLEAN_MODE, NOISY_MODE):
        subdir_path = get_cache_subdir(mode)
        if os.path.exists(subdir_path):
            for root, _, files in os.walk(subdir_path):
                for name in files:
                    if name.endswith('.pkl') or name.endswith('.json'):
                        filepath = os.path.join(root, name)
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except Exception as e:
                            log_with_timestamp(f"  ⚠️  删除失败: {filepath} - {e}")

    if deleted_count > 0:
        log_with_timestamp(f"✓ 已清理旧缓存: {deleted_count} 个文件")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='生成 noisy 查询缓存')
    parser.add_argument('--retrievers', type=str, nargs='+',
                        choices=list(AVAILABLE_RETRIEVERS.keys()),
                        default=list(AVAILABLE_RETRIEVERS.keys()),
                        help='指定要处理的检索器')
    parser.add_argument('--clear', action='store_true',
                        help='清理旧缓存后再生成')
    args = parser.parse_args()

    retriever_names = args.retriever_names if hasattr(args, 'retriever_names') else args.retrievers
    clear_cache_before = args.clear

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 生成 expression_style clean/noisy 查询缓存")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"类别: {CATEGORY_NAME}")
    log_with_timestamp(f"Syntax-depth query 文件: {SYNTAX_DEPTH_QUERY_FILE}")
    log_with_timestamp(f"缓存目录: {CACHE_DIR}")
    log_with_timestamp(f"检索器: {', '.join(retriever_names)}")
    log_with_timestamp(
        "HF 离线模式: "
        f"HF_HOME={os.environ.get('HF_HOME')}, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}, "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}, "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}, "
        f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}"
    )
    log_with_timestamp("")

    # 加载 expression_style clean/noisy queries
    query_sets = load_syntax_depth_query_pairs()
    clean_queries = query_sets['clean_queries']
    noisy_queries = query_sets['noisy_queries']

    if not clean_queries or not noisy_queries:
        log_with_timestamp("⚠️  没有加载到任何 expression_style clean/noisy 查询")
        return

    # 按用户分组
    clean_queries_by_user = _build_queries_by_user(clean_queries)
    noisy_queries_by_user = _build_queries_by_user(noisy_queries)
    clean_query_count = sum(len(v) for v in clean_queries_by_user.values())
    noisy_query_count = sum(len(v) for v in noisy_queries_by_user.values())

    log_with_timestamp("")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • 数据源: {SYNTAX_DEPTH_QUERY_FILE}")
    log_with_timestamp(f"  • clean 模式: {CLEAN_MODE}")
    log_with_timestamp(f"  • noisy 模式: {NOISY_MODE}")
    log_with_timestamp(f"  • 目标 injection_source: {SYNTAX_DEPTH_INJECTION_SOURCE}")
    log_with_timestamp(f"  • clean 用户: {len(clean_queries_by_user)} 个, 查询: {clean_query_count} 条")
    log_with_timestamp(f"  • noisy 用户: {len(noisy_queries_by_user)} 个, 查询: {noisy_query_count} 条")
    log_with_timestamp(f"  • clean 缓存目录: {get_cache_subdir(CLEAN_MODE)}")
    log_with_timestamp(f"  • noisy 缓存目录: {get_cache_subdir(NOISY_MODE)}")
    log_with_timestamp("")

    if clear_cache_before:
        clear_noisy_cache()
    initialize_cache_dir()

    start_time = time.time()
    total_cached = 0

    # 定义查询类型
    query_types = [
        ('SYNTAX_DEPTH_CLEAN', clean_queries, clean_queries_by_user, CLEAN_MODE),
        ('SYNTAX_DEPTH_NOISY', noisy_queries, noisy_queries_by_user, NOISY_MODE),
    ]

    for retriever_name in retriever_names:
        if retriever_name not in AVAILABLE_RETRIEVERS:
            log_with_timestamp(f"⚠️  检索器不存在: {retriever_name}")
            continue

        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"正在处理检索器: {retriever_name}")
        log_with_timestamp(f"{'='*80}")

        if retriever_name == 'ColBERTv2':
            summary = generate_colbertv2_cache_from_query_types(query_types)
            total_cached += summary['total_cached']
            log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成: {summary['total_cached']} 条")
            continue

        retriever = load_retriever_for_query_encoding(retriever_name)
        try:
            for query_type, queries, by_user, mode in query_types:
                if queries:
                    total = _encode_and_save_cache(
                        retriever_name,
                        retriever,
                        queries,
                        by_user,
                        mode,
                    )
                    total_cached += total
                    log_with_timestamp(f"  ✓ {query_type} {mode} 缓存: {total} 条")
                else:
                    log_with_timestamp(f"  (无 {query_type} {mode} 查询，跳过)")
        finally:
            del retriever
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_with_timestamp(f"✓ 检索器 {retriever_name} 处理完成")

    elapsed = time.time() - start_time

    # 统计缓存
    cache_files = 0
    cache_dir_size = 0.0
    for mode in (CLEAN_MODE, NOISY_MODE):
        cache_dir = get_cache_subdir(mode)
        if os.path.exists(cache_dir):
            for root, _, files in os.walk(cache_dir):
                for name in files:
                    if name.endswith('.pkl'):
                        cache_files += 1
                        cache_dir_size += os.path.getsize(os.path.join(root, name)) / (1024 * 1024)

    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("✨ 完成!")
    log_with_timestamp(f"  • 处理检索器: {len(retriever_names)} 个")
    log_with_timestamp(f"  • expression_style clean 查询: {clean_query_count} 条")
    log_with_timestamp(f"  • expression_style noisy 查询: {noisy_query_count} 条")
    log_with_timestamp(f"  • 缓存文件: {cache_files} 个")
    log_with_timestamp(f"  • 缓存大小: {cache_dir_size:.1f} MB")
    log_with_timestamp(f"  • 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
