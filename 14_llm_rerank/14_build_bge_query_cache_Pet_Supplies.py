#!/usr/bin/env python3
"""Stage 15 辅助脚本：为 Pet_Supplies 一次性生成全量 bge query embedding cache。"""

import os
import pickle
import sys
import time
from datetime import datetime

REPO_ROOT = "/fs04/ar57/wenyu"
os.environ.setdefault("HF_HOME", "/home/wlia0047/ar57_scratch/wenyu/hf_models")
os.environ.setdefault("HF_HUB_CACHE", "/home/wlia0047/ar57_scratch/wenyu/hf_models")

STAGE8_ROOT = os.path.join(REPO_ROOT, "PersoanlQuery/06_retrieval")
for p in (STAGE8_ROOT, REPO_ROOT + "/PersoanlQuery", REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_category_config, get_global_paths  # noqa: E402
from utils.retrievers import BGERetriever  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    CATEGORY = "Pet_Supplies"
    QUERY_TYPE = "correct"

    cat_cfg = get_category_config(CATEGORY)
    global_paths = get_global_paths()
    query_file = os.path.join(
        global_paths["stage6_query"],
        CATEGORY,
        "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json",
    )
    cache_subdir = os.path.join(
        cat_cfg["query_cache_dir"],
        f"syntax_depth_{QUERY_TYPE}_query",
    )
    cache_file = os.path.join(
        cache_subdir,
        f"bge__syntax_depth_{QUERY_TYPE}_cache.pkl",
    )

    log("=" * 80)
    log(f"15_build_bge_query_cache | {CATEGORY} | {QUERY_TYPE}")
    log("=" * 80)
    log(f"  query_file: {query_file}")
    log(f"  cache_file: {cache_file}")

    if not os.path.exists(query_file):
        raise FileNotFoundError(f"expression_style query file not found: {query_file}")
    with open(query_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"query file must be list, got {type(records).__name__}: {query_file}")
    log(f"  loaded {len(records)} records from 06_query")

    seen_user_ids = set()
    queries_by_user: dict = {}
    for idx, r in enumerate(records):
        for field in ("user_id", "asin", "syntax_depth_query"):
            if field not in r:
                raise ValueError(f"record missing '{field}' at index {idx}: {r}")
        user_id = r["user_id"]
        asin = r["asin"]
        query_text = r["syntax_depth_query"]["query"].strip()
        if not user_id or not asin or not query_text:
            raise ValueError(f"empty field at index {idx}: user={user_id} asin={asin}")
        if user_id in seen_user_ids:
            raise ValueError(
                f"duplicate user_id={user_id} at index {idx}; "
                f"06_query 期望每个 user 1 条 record"
            )
        seen_user_ids.add(user_id)
        queries_by_user[user_id] = {"asin": asin, "query": query_text}
    log(f"  unique users: {len(queries_by_user)}")

    user_ids = list(queries_by_user.keys())
    query_texts = [queries_by_user[u]["query"] for u in user_ids]
    log(f"  ready to encode {len(query_texts)} queries with BGE-large")

    log("  loading BGERetriever model...")
    model_load_start = time.time()
    retriever = BGERetriever()
    log(f"  model loaded, elapsed {time.time() - model_load_start:.1f}s")

    log("  encoding queries in batch (batch_size=64)...")
    encode_start = time.time()
    embeddings = retriever.encode_queries(query_texts, batch_size=64)
    log(f"  encoded {len(embeddings)} queries in {time.time() - encode_start:.1f}s, shape={embeddings.shape}")

    cache: dict = {}
    for user_id, emb in zip(user_ids, embeddings):
        cache[user_id] = {queries_by_user[user_id]["query"]: emb}
    log(f"  cache built: {len(cache)} users")

    os.makedirs(cache_subdir, exist_ok=True)
    tmp_path = cache_file + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_file)
    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    log(f"  ✓ cache saved: {cache_file} ({file_size_mb:.2f} MB)")

    log("")
    log("=" * 80)
    log(f"  完成 | users={len(cache)} | cache_file={cache_file}")
    log(f"  Stage 15 rerank 现在可以覆盖全部 {len(cache)} user")
    log("=" * 80)


if __name__ == "__main__":
    import json
    main()
