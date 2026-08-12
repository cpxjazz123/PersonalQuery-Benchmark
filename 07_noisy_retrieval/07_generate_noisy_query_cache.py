#!/usr/bin/env python3
"""Stage 07 生成并预存储检索器的 noisy 查询缓存 — 统一入口（3 域）。

使用方法:
    python3 07_generate_noisy_query_cache.py --category Baby_Products
    python3 07_generate_noisy_query_cache.py --category Grocery_and_Gourmet_Food
    python3 07_generate_noisy_query_cache.py --category Pet_Supplies
    python3 07_generate_noisy_query_cache.py --category Baby_Products --retrievers BGE SPLADE --clear
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
# After iter #48's renumber (94cd02b refactor(scripts): align file prefixes
# with new stage numbers), the retrieval utility module moved from
# `08_retrieval/utils/` to `06_retrieval/utils/`. Point at the new location so
# `from utils.retrievers import ...` resolves.
retrieval_root = current_dir.parent / "06_retrieval"
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
CATEGORY_NAME = None  # set by main()
CAT_CONFIG = None
GLOBAL_PATHS = None

CLEAN_MODE = "expression_style_correct"
NOISY_MODE = "expression_style_noisy"
EXPRESSION_STYLE_INJECTION_SOURCE = "final_strict_query_no_depth_constraint"

BM25_RETRIEVER_CACHE_DIR = None

AVAILABLE_RETRIEVERS = {
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


def _get_category_module_name(category: str) -> str:
    return category.lower().replace('_', '')


def load_colbertv2_build_module(category: str):
    module_name = _get_category_module_name(category)
    module_path = retrieval_root / f"06_build_retriever_indices_{category}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Required ColBERTv2 build module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"build_retriever_indices_{module_name}", module_path)
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
        f"nvidia/{torch.version.cuda}",
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    try:
        select_cuda_toolkit_for_colbert_extension_build(major, minor)
        configure_host_compiler_for_colbert_extension_build()
        validate_cuda_toolkit_for_colbert(major, minor)
        configure_cuda_env_for_colbert_extension_build(major, minor)
    except Exception as e:
        log_with_timestamp(f"    ColBERTv2 构建环境配置失败: {e}，跳过 ColBERTv2")


def get_cache_subdir(mode: str) -> str:
    return os.path.join(CAT_CONFIG['query_cache_dir'], f"{mode}_query")


def get_cache_file_path(retriever_name: str, suffix: str, mode: str) -> str:
    return os.path.join(
        get_cache_subdir(mode),
        f"{retriever_name.lower()}{suffix}__{mode}_cache.pkl"
    )


def initialize_cache_dir():
    for mode in (CLEAN_MODE, NOISY_MODE):
        subdir = get_cache_subdir(mode)
        os.makedirs(subdir, exist_ok=True)
    log_with_timestamp(f"  缓存目录已就绪: {CAT_CONFIG['query_cache_dir']}")


def _build_query_entry(
    user_id: str,
    asin: str,
    clean_query: str,
    noisy_query: str,
    query_text: str,
    query_type: str,
    source_query_field: str,
    injection_source: str,
) -> Dict:
    return {
        'user_id': user_id,
        'asin': asin,
        'clean_query': clean_query,
        'noisy_query': noisy_query,
        'query': query_text,
        'query_type': query_type,
        'source_query_field': source_query_field,
        'injection_source': injection_source,
    }


def load_expression_style_query_pairs() -> Dict[str, List[Dict]]:
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
            injection_source=EXPRESSION_STYLE_INJECTION_SOURCE,
        ))
        noisy_queries.append(_build_query_entry(
            user_id=user_id,
            asin=asin,
            clean_query=clean_query,
            noisy_query=noisy_query,
            query_text=noisy_query,
            query_type='noisy',
            source_query_field='noisy_query',
            injection_source=EXPRESSION_STYLE_INJECTION_SOURCE,
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
    by_user = defaultdict(list)
    for q in queries:
        by_user[q['user_id']].append(q)
    return by_user


# ---- BM25 ----
def _build_bm25_index(queries: List[Dict], mode: str):
    corpus = [q['query'] for q in queries]
    bm25 = BM25()
    bm25.fit(corpus)
    return bm25


def _search_bm25(bm25, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
    return bm25.search(query_text, top_k)


# ---- dense / sparse retriever helpers ----
def _encode_and_cache_for_retriever(
    retriever,
    retriever_name: str,
    queries_by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    cache: dict = {}
    total_cached = 0

    for user_id, user_queries in queries_by_user.items():
        user_cache: dict = {}
        total_items = len(user_queries)

        for q in user_queries:
            text = q['query']
            if not text or not text.strip():
                continue

            try:
                if hasattr(retriever, 'encode_queries'):
                    embedding = retriever.encode_queries([text])[0]
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().numpy()
                    else:
                        embedding = np.array(embedding)
                elif hasattr(retriever, 'encode_query'):
                    embedding = retriever.encode_query(text)
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().numpy()
                    else:
                        embedding = np.array(embedding)
                else:
                    embedding = np.array(retriever.encode([text])[0])

                user_cache[text] = embedding
                total_cached += 1
                if total_cached % 100 == 0 or total_cached == total_items:
                    log_with_timestamp(
                        f"    进度 [{retriever_name}] {mode}: {total_cached}/{total_items}"
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

    return total_cached


def _encode_and_save_splade_cache(
    retriever,
    queries_by_user: Dict[str, List[Dict]],
    mode: str,
) -> int:
    cache: dict = {}
    total_cached = 0

    for user_id, user_queries in queries_by_user.items():
        user_cache: dict = {}
        total_items = len(user_queries)

        for idx, q in enumerate(user_queries):
            try:
                text = q.get('query', '')
                if not text:
                    raise ValueError(f"{mode} query is empty for user={user_id}, asin={q.get('asin', '')}")
                sparse_vec = retriever.encode_query(text)
                user_cache[text] = sparse_vec
                total_cached += 1
                if total_cached % 100 == 0 or total_cached == total_items:
                    log_with_timestamp(f"    进度 [SPLADE] {mode}: {total_cached}/{total_items}")
            except Exception as e:
                log_with_timestamp(f"      ❌ SPLADE 编码失败: {text[:40]}... 错误: {str(e)[:100]}")
                raise

        if user_cache:
            cache[user_id] = user_cache

    if cache:
        cache_file = os.path.join(get_cache_subdir(mode), f"splade__{mode}_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        log_with_timestamp(f"  ✓ SPLADE 缓存已保存: {cache_file}")

    return total_cached


def _encode_and_save_bm25_cache(
    bm25,
    queries: List[Dict],
    mode: str,
) -> int:
    cache: dict = {}
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

    return len(cache)


def clear_noisy_cache():
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
                        except OSError as e:
                            log_with_timestamp(f"  ⚠️  删除失败: {filepath} - {e}")

    if deleted_count > 0:
        log_with_timestamp(f"✓ 已清理旧缓存: {deleted_count} 个文件")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(description='生成 noisy 查询缓存（统一入口）')
    parser.add_argument(
        '--category',
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help='域类别',
    )
    parser.add_argument(
        '--retrievers',
        type=str,
        nargs='+',
        choices=list(AVAILABLE_RETRIEVERS.keys()),
        default=list(AVAILABLE_RETRIEVERS.keys()),
        help='指定要处理的检索器',
    )
    parser.add_argument(
        '--clear', action='store_true',
        help='清理旧缓存后再生成',
    )
    args = parser.parse_args()

    global CATEGORY_NAME, CAT_CONFIG, GLOBAL_PATHS, BM25_RETRIEVER_CACHE_DIR, NOISY_QUERY_JSON_FILE, EXPRESSION_STYLE_QUERY_FILE, CACHE_DIR

    CATEGORY_NAME = args.category
    CAT_CONFIG = get_category_config(CATEGORY_NAME)
    GLOBAL_PATHS = get_global_paths()
    CACHE_DIR = CAT_CONFIG['query_cache_dir']
    BM25_RETRIEVER_CACHE_DIR = CAT_CONFIG['retriever_cache_dir']

    EXPRESSION_STYLE_QUERY_FILE = (
        f"/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/06_query/{CATEGORY_NAME}/"
        "query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json"
    )
    NOISY_QUERY_JSON_FILE = os.path.join(
        GLOBAL_PATHS.get("inject_noisy", f"/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/07_inject_noisy"),
        CATEGORY_NAME,
        "noisy_query.json"
    )

    clear_cache_before = args.clear
    retriever_names = args.retrievers

    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 生成 expression_style clean/noisy 查询缓存")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"类别: {CATEGORY_NAME}")
    log_with_timestamp(f"Expression_style query 文件: {EXPRESSION_STYLE_QUERY_FILE}")
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

    query_sets = load_expression_style_query_pairs()
    clean_queries = query_sets['clean_queries']
    noisy_queries = query_sets['noisy_queries']

    if not clean_queries or not noisy_queries:
        log_with_timestamp("⚠️  没有加载到任何 expression_style clean/noisy 查询")
        return

    clean_queries_by_user = _build_queries_by_user(clean_queries)
    noisy_queries_by_user = _build_queries_by_user(noisy_queries)
    clean_query_count = sum(len(v) for v in clean_queries_by_user.values())
    noisy_query_count = sum(len(v) for v in noisy_queries_by_user.values())

    log_with_timestamp("")
    log_with_timestamp(f"📋 任务配置:")
    log_with_timestamp(f"  • 数据源: {EXPRESSION_STYLE_QUERY_FILE}")
    log_with_timestamp(f"  • clean 模式: {CLEAN_MODE}")
    log_with_timestamp(f"  • noisy 模式: {NOISY_MODE}")
    log_with_timestamp(f"  • 目标 injection_source: {EXPRESSION_STYLE_INJECTION_SOURCE}")
    log_with_timestamp(f"  • clean 用户: {len(clean_queries_by_user)} 个, 查询: {clean_query_count} 条")
    log_with_timestamp(f"  • noisy 用户: {len(noisy_queries_by_user)} 个, 查询: {noisy_query_count} 条")
    log_with_timestamp(f"  • clean 缓存目录: {get_cache_subdir(CLEAN_MODE)}")
    log_with_timestamp(f"  • noisy 缓存目录: {get_cache_subdir(NOISY_MODE)}")
    log_with_timestamp("")

    if clear_cache_before:
        clear_noisy_cache()
    initialize_cache_dir()

    # 动态加载 ColBERTv2 构建模块（只做一次）
    colbert_module = None
    if 'ColBERTv2' in retriever_names:
        try:
            colbert_module = load_colbertv2_build_module(CATEGORY_NAME)
            configure_colbertv2_runtime()
        except Exception as e:
            log_with_timestamp(f"  ⚠️  ColBERTv2 不可用: {e}，从可用列表中移除")
            retriever_names = [r for r in retriever_names if r != 'ColBERTv2']

    for retriever_name in retriever_names:
        if retriever_name == 'ColBERTv2' and colbert_module:
            continue
        retriever_cls = AVAILABLE_RETRIEVERS.get(retriever_name)
        if not retriever_cls:
            log_with_timestamp(f"  ⏭  跳过不支持的检索器: {retriever_name}")
            continue

        try:
            retriever = retriever_cls()
        except Exception as e:
            log_with_timestamp(f"  ⚠️  初始化 {retriever_name} 失败: {e}")
            continue

        log_with_timestamp(f"\n  [{retriever_name}] 开始编码 clean 查询...")
        for mode, queries_by_user in (('clean', clean_queries_by_user), ('noisy', noisy_queries_by_user)):
            log_with_timestamp(f"    [{retriever_name}] 模式: {mode}")
            if retriever_name == 'BM25':
                bm25 = _build_bm25_index(queries_by_user.get(list(queries_by_user.keys())[0], []), mode)
                _encode_and_save_bm25_cache(bm25, queries_by_user.get(list(queries_by_user.keys())[0], []), mode)
            elif retriever_name == 'SPLADE':
                _encode_and_save_splade_cache(retriever, queries_by_user, mode)
            else:
                _encode_and_cache_for_retriever(retriever, retriever_name, queries_by_user, mode)

    log_with_timestamp("\n✅ 全部缓存生成完毕")


if __name__ == "__main__":
    main()
