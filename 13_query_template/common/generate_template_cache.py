"""14-stage query cache generator.

Encodes the template queries from `query_template.json` using the same
retrievers as Stage 8 (BGE, E5, MiniLM, STAR, ANCE, SPLADE, ColBERTv2, BM25)
and writes the per-retriever cache to:

    <query_cache_base_dir>/template_query/<retriever>__template_query_cache.pkl

This mirrors the structure that the 14-stage eval driver expects (the 8
`load_*_query_cache` overrides in `cache_path_overrides.py`).
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_14_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(STAGE_14_DIR)
STAGE_08_DIR = os.path.join(REPO_ROOT, "08_retrieval")
STAGE_08_UTILS = os.path.join(STAGE_08_DIR, "utils")
STAGE_08_CACHE_DIR = os.path.join(STAGE_08_DIR)

sys.path.insert(0, STAGE_08_DIR)

from .config import get_category_config  # noqa: E402
from .user_data_loader import load_template_user_queries  # noqa: E402


CACHE_SUBDIR_CLEAN = "template_query"
CACHE_SUBDIR_NOISY = "template_noisy_query"
CACHE_FILE_TEMPLATE = "{retriever}__{subdir}_cache.pkl"


def _cache_subdir_for(mode: str) -> str:
    if mode == "clean":
        return CACHE_SUBDIR_CLEAN
    if mode == "noisy":
        return CACHE_SUBDIR_NOISY
    raise ValueError(f"unknown mode: {mode!r}, expected 'clean' or 'noisy'")

DENSE_RETRIEVER_CLASSES = ("E5Retriever", "BGERetriever", "STARRetriever", "MiniLMRetriever", "ANCERetriever")
SPARSE_RETRIEVER_CLASSES = ("SPLADERetriever",)
COLBERTV2_RETRIEVER = "colbertv2"
BM25_RETRIEVER = "bm25"

RETRIEVER_NAME_TO_CLASS = {
    "bge": "BGERetriever",
    "e5": "E5Retriever",
    "minilm": "MiniLMRetriever",
    "star": "STARRetriever",
    "ance": "ANCERetriever",
    "splade": "SPLADERetriever",
}

DENSE_BATCH_SIZE = 256
SPLADE_BATCH_SIZE = 64
DENSE_MIN_FREE_GB = 1.0

HF_CACHE_ENV = {
    "HF_HOME": "/home/wlia0047/ar57_scratch/wenyu/hf_models",
    "HF_HUB_CACHE": "/home/wlia0047/ar57_scratch/wenyu/hf_models",
    "HF_HUB_OFFLINE": "0",
    "TRANSFORMERS_OFFLINE": "0",
}


def log_with_timestamp(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _free_gpu_memory_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return None


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _set_hf_env() -> None:
    for key, value in HF_CACHE_ENV.items():
        os.environ.setdefault(key, value)


def _get_retriever_class(class_name: str):
    from utils.retrievers import (  # type: ignore  # noqa: WPS433
        E5Retriever, BGERetriever, STARRetriever, MiniLMRetriever, ANCERetriever,
        SPLADERetriever, BM25,
    )
    mapping = {
        "E5Retriever": E5Retriever,
        "BGERetriever": BGERetriever,
        "STARRetriever": STARRetriever,
        "MiniLMRetriever": MiniLMRetriever,
        "ANCERetriever": ANCERetriever,
        "SPLADERetriever": SPLADERetriever,
        "BM25": BM25,
    }
    if class_name not in mapping:
        raise KeyError(f"unknown retriever class: {class_name}")
    return mapping[class_name]


def _cache_path(query_cache_base_dir: str, retriever: str, mode: str = "clean") -> str:
    subdir = _cache_subdir_for(mode)
    return os.path.join(
        query_cache_base_dir,
        subdir,
        CACHE_FILE_TEMPLATE.format(retriever=retriever, subdir=subdir),
    )


def _load_template_queries(query_template_file: str) -> List[Dict[str, Any]]:
    path = Path(query_template_file)
    if not path.exists():
        raise FileNotFoundError(f"query_template file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"query_template file must contain a list: {path}")
    return data


def _build_queries_by_user(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        for field in ("user_id", "asin", "query"):
            if field not in rec:
                raise ValueError(f"record missing field '{field}': {rec}")
        if not isinstance(rec["user_id"], str) or not rec["user_id"]:
            raise ValueError(f"empty user_id: {rec}")
        if not isinstance(rec["query"], str) or not rec["query"]:
            raise ValueError(f"empty query: {rec}")
        if not isinstance(rec["asin"], str) or not rec["asin"]:
            raise ValueError(f"empty asin: {rec}")
        out.setdefault(rec["user_id"], []).append({
            "user_id": rec["user_id"],
            "asin": rec["asin"],
            "query": rec["query"],
        })
    return out


def _flatten_unique_queries(queries_by_user: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for user_queries in queries_by_user.values():
        for q in user_queries:
            key = q["query"]
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    return out


def _encode_dense(retriever, retriever_name: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    if not hasattr(retriever, "encode_queries"):
        for i, q in enumerate(queries):
            cache[q["query"]] = retriever.encode_query(q["query"])
            if (i + 1) % 100 == 0 or (i + 1) == len(queries):
                log_with_timestamp(f"    [progress] {i + 1}/{len(queries)}")
        return cache

    batch_size = SPLADE_BATCH_SIZE if retriever_name == "splade" else DENSE_BATCH_SIZE
    free_gb = _free_gpu_memory_gb()
    if free_gb is not None:
        log_with_timestamp(f"    [gpu] free={free_gb:.2f} GB before encoding")
        if free_gb < DENSE_MIN_FREE_GB:
            raise RuntimeError(
                f"Insufficient free GPU memory for {retriever_name}: "
                f"free={free_gb:.2f} GB < required {DENSE_MIN_FREE_GB:.2f} GB. "
                f"Call _release_gpu_memory() before initializing the retriever."
            )

    texts = [q["query"] for q in queries]
    n = len(texts)
    log_with_timestamp(f"    [dense/sparse batch] encoding {n} queries, batch={batch_size}")
    for batch_start in range(0, n, batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        encs = retriever.encode_queries(batch_texts, batch_size=batch_size)
        for t, e in zip(batch_texts, encs):
            cache[t] = e
        log_with_timestamp(f"    [progress] {min(batch_start + batch_size, n)}/{n}")
    return cache


def _save_per_user_cache(
    query_cache_base_dir: str,
    retriever: str,
    flat_cache: Dict[str, Any],
    queries_by_user: Dict[str, List[Dict[str, Any]]],
    mode: str = "clean",
) -> int:
    path = _cache_path(query_cache_base_dir, retriever, mode=mode)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if retriever == "bm25":
        payload: Dict[str, Any] = flat_cache
    else:
        per_user: Dict[str, Dict[str, Any]] = {uid: {} for uid in queries_by_user.keys()}
        for query_text, embedding in flat_cache.items():
            for uid, user_queries in queries_by_user.items():
                if any(q["query"] == query_text for q in user_queries):
                    per_user[uid][query_text] = embedding
        payload = per_user

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if retriever == "bm25":
        log_with_timestamp(f"  ✓ {retriever} cache saved: {len(flat_cache)} queries, {size_mb:.2f} MB")
        return len(flat_cache)
    total = sum(len(v) for v in payload.values())
    log_with_timestamp(f"  ✓ {retriever} cache saved: {len(payload)} users / {total} queries, {size_mb:.2f} MB")
    return total


def _generate_bm25_cache(
    category: str,
    records: List[Dict[str, Any]],
    top_k: int = 100,
) -> Dict[str, List]:
    from utils.retrievers import BM25  # type: ignore  # noqa: WPS433

    cat_cfg_path = os.path.join(STAGE_08_DIR, "08_retrieval_config.json")
    with open(cat_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if category not in cfg["categories"]:
        raise KeyError(f"unknown category in 08 config: {category!r}")
    cache_dir = cfg["categories"][category]["retriever_cache_dir"]
    cache_dir = cache_dir.replace("{scratch_result}", cfg["base_paths"]["scratch_result"])

    bm25_path = None
    for f in os.listdir(cache_dir):
        if f.startswith("bm25_") and f.endswith(".pkl"):
            bm25_path = os.path.join(cache_dir, f)
            break
    if bm25_path is None:
        raise FileNotFoundError(f"BM25 retriever cache not found in {cache_dir}")
    log_with_timestamp(f"  loading BM25 index for {category}: {os.path.getsize(bm25_path)/1024/1024:.1f} MB")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    cache: Dict[str, List] = {}
    unique_queries = _flatten_unique_queries(_build_queries_by_user(records))
    for i, q in enumerate(unique_queries):
        cache[q["query"]] = bm25.search(q["query"], top_k=top_k)
        if (i + 1) % 100 == 0 or (i + 1) == len(unique_queries):
            log_with_timestamp(f"    [bm25] {i + 1}/{len(unique_queries)}")
    return cache


def _generate_colbertv2_cache(
    query_cache_base_dir: str,
    records: List[Dict[str, Any]],
    mode: str = "clean",
    query_maxlen: int = 96,
) -> int:
    """Encode template queries with ColBERTv2 query encoder and write to disk.

    ColBERTv2 has a separate query encoder (Checkpoint.queryFromText) which
    produces per-token embeddings. The 08 stage stores these in
    `{colbertv2__<subdir>_cache.pkl}` with the same per-user dict shape.
    `mode` selects the cache subdir (`template_query` or `template_noisy_query`).
    `query_maxlen` overrides the ColBERTConfig's `query_maxlen` (default 96).

    Why 96 (not 32): 14-stage template queries average 35.7 subword tokens
    (median 35, max 139). With `query_maxlen=32`, 69% of queries are truncated
    mid-way through product-name tokens, losing the signal ColBERTv2's per-token
    MaxSim needs. 96 covers 100% of Baby_Products and Pet_Supplies queries.
    """
    import torch  # type: ignore  # noqa: WPS433
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 query encoding requires CUDA")

    from colbert.infra import ColBERTConfig  # type: ignore  # noqa: WPS433
    from colbert.modeling.checkpoint import Checkpoint  # type: ignore  # noqa: WPS433

    config = ColBERTConfig(checkpoint="colbert-ir/colbertv2.0", query_maxlen=query_maxlen)
    log_with_timestamp(f"  [colbertv2] using query_maxlen={query_maxlen}")
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config, verbose=1).cuda()

    unique_queries = _flatten_unique_queries(_build_queries_by_user(records))
    query_texts = [q["query"] for q in unique_queries]
    log_with_timestamp(f"  [colbertv2] encoding {len(query_texts)} unique queries (batched)")
    embeddings = checkpoint.queryFromText(query_texts, bsize=128, to_cpu=True)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"ColBERTv2 query encoder returned {type(embeddings).__name__}")
    if embeddings.shape[0] != len(query_texts):
        raise RuntimeError(f"ColBERTv2 embedding count mismatch: {embeddings.shape[0]} vs {len(query_texts)}")

    encoded_by_query = {
        query_texts[i]: embeddings[i].detach().cpu().numpy().astype("float32")
        for i in range(len(query_texts))
    }

    queries_by_user = _build_queries_by_user(records)
    per_user: Dict[str, Dict[str, Any]] = {uid: {} for uid in queries_by_user.keys()}
    for query_text, embedding in encoded_by_query.items():
        for uid, user_queries in queries_by_user.items():
            if any(q["query"] == query_text for q in user_queries):
                per_user[uid][query_text] = embedding

    path = _cache_path(query_cache_base_dir, "colbertv2", mode=mode)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(per_user, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    total = sum(len(v) for v in per_user.values())
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log_with_timestamp(f"  ✓ colbertv2 cache saved: {len(per_user)} users / {total} queries, {size_mb:.2f} MB")
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return total


def generate_template_cache(
    category: str,
    retrievers: Optional[List[str]] = None,
    clear_cache_before: bool = False,
    top_k_bm25: int = 100,
    mode: str = "clean",
    colbertv2_query_maxlen: int = 96,
) -> Dict[str, int]:
    if mode not in ("clean", "noisy"):
        raise ValueError(f"unknown mode: {mode!r}, expected 'clean' or 'noisy'")
    if retrievers is None:
        retrievers = ["bge", "e5", "minilm", "star", "ance", "colbertv2", "splade", "bm25"]

    cat_cfg = get_category_config(category)
    if mode == "noisy":
        query_template_file = cat_cfg["output_query_file_noisy"]
    else:
        query_template_file = cat_cfg["output_query_file"]
    query_cache_base_dir = cat_cfg["query_cache_base_dir"]
    cache_subdir = _cache_subdir_for(mode)

    log_with_timestamp("=" * 80)
    log_with_timestamp(f"🚀 14-stage template cache generation - {category} [{mode}]")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"  mode: {mode}")
    log_with_timestamp(f"  query_template_file: {query_template_file}")
    log_with_timestamp(f"  query_cache_base_dir: {query_cache_base_dir}")
    log_with_timestamp(f"  cache_subdir: {cache_subdir}")
    log_with_timestamp(f"  retrievers: {retrievers}")
    log_with_timestamp(f"  clear_cache_before: {clear_cache_before}")

    _set_hf_env()
    records = _load_template_queries(query_template_file)
    queries_by_user = _build_queries_by_user(records)
    log_with_timestamp(f"  loaded {len(records)} records, {len(queries_by_user)} users")

    cache_dir = os.path.join(query_cache_base_dir, cache_subdir)
    if clear_cache_before and os.path.isdir(cache_dir):
        log_with_timestamp(f"  clearing existing cache dir: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    stats: Dict[str, int] = {
        "total_queries": sum(len(v) for v in queries_by_user.values()),
        "total_cached": 0,
        "retrievers_processed": 0,
    }

    start = time.time()
    for retriever_name in retrievers:
        log_with_timestamp(f"\n--- {retriever_name} ---")
        _release_gpu_memory()
        try:
            if retriever_name == "bm25":
                flat = _generate_bm25_cache(category, records, top_k=top_k_bm25)
                cached = _save_per_user_cache(query_cache_base_dir, "bm25", flat, queries_by_user, mode=mode)
            elif retriever_name == "colbertv2":
                cached = _generate_colbertv2_cache(
                    query_cache_base_dir,
                    records,
                    mode=mode,
                    query_maxlen=colbertv2_query_maxlen,
                )
            else:
                class_name = RETRIEVER_NAME_TO_CLASS.get(retriever_name)
                if class_name is None:
                    raise ValueError(f"unsupported retriever: {retriever_name}")
                cls = _get_retriever_class(class_name)
                log_with_timestamp(f"  initializing {retriever_name} ({class_name}) ...")
                instance = cls()
                try:
                    unique = _flatten_unique_queries(queries_by_user)
                    flat = _encode_dense(instance, retriever_name, unique)
                finally:
                    del instance
                    _release_gpu_memory()
                cached = _save_per_user_cache(query_cache_base_dir, retriever_name, flat, queries_by_user, mode=mode)
            stats["total_cached"] += cached
            stats["retrievers_processed"] += 1
        except Exception as exc:
            log_with_timestamp(f"  [ERROR] {retriever_name} failed: {exc}")
            raise

    elapsed = time.time() - start
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("✅ 14-stage template cache generation complete")
    log_with_timestamp(f"  mode: {mode}")
    log_with_timestamp(f"  total queries: {stats['total_queries']}")
    log_with_timestamp(f"  total cached entries: {stats['total_cached']}")
    log_with_timestamp(f"  retrievers processed: {stats['retrievers_processed']}")
    log_with_timestamp(f"  cache dir: {cache_dir}")
    log_with_timestamp(f"  elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Baby_Products | Grocery_and_Gourmet_Food | Pet_Supplies")
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bge", "e5", "minilm", "star", "ance", "colbertv2", "splade", "bm25"],
        help="subset of retrievers to run (default: all 8)",
    )
    parser.add_argument("--clear-cache-before", action="store_true", help="delete existing template_query cache dir before generation")
    parser.add_argument("--top-k-bm25", type=int, default=100, help="top-k for BM25 precomputation")
    parser.add_argument(
        "--mode",
        choices=["clean", "noisy"],
        default="clean",
        help="clean → template_query/, noisy → template_noisy_query/",
    )
    parser.add_argument(
        "--colbertv2-query-maxlen",
        type=int,
        default=96,
        help="ColBERTv2 query_maxlen override (default 96, prevents 69%% of template queries from being truncated at the default 32)",
    )
    args = parser.parse_args()
    generate_template_cache(
        category=args.category,
        retrievers=args.retrievers,
        clear_cache_before=args.clear_cache_before,
        top_k_bm25=args.top_k_bm25,
        mode=args.mode,
        colbertv2_query_maxlen=args.colbertv2_query_maxlen,
    )


if __name__ == "__main__":
    main()
