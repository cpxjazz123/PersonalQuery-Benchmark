#!/usr/bin/env python3
import gzip
import json
import os
import shutil
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def open_possibly_gzipped(file_path: str, mode: str = "rt", encoding: str = "utf-8"):
    """打开文件，自动处理 .gz 压缩和非压缩文件。"""
    if file_path.endswith(".gz"):
        return gzip.open(file_path, mode, encoding=encoding)
    return open(file_path, mode, encoding=encoding)


def count_long_sentences(text: str, min_words: int, max_words: int) -> int:
    """Count how many sentences have min_words <= word count <= max_words."""
    if not text:
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sum(1 for sent in sentences if min_words <= len(sent.split()) <= max_words)


def select_users_by_long_reviews(review_file: str, min_words: int, max_words: int, min_long_sentences: int) -> Dict[str, int]:
    """
    统计每个用户有多少个长句子（词数在 min_words 和 max_words 之间）。
    返回满足 min_long_sentences 条件的用户及其长句子总数。
    """
    user_long_sentence_counts: Dict[str, int] = defaultdict(int)

    log_with_timestamp("Scanning reviews for long-sentence user filtering...")
    with open_possibly_gzipped(review_file, "rt") as f:
        for line in f:
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = review.get("user_id")
            review_text = review.get("text", "")
            if not user_id:
                continue

            # Count how many sentences in this review have min_words <= word count <= max_words
            num_long_sents = count_long_sentences(review_text, min_words, max_words)
            user_long_sentence_counts[user_id] += num_long_sents

    selected = {
        user_id: count
        for user_id, count in user_long_sentence_counts.items()
        if count >= min_long_sentences
    }

    log_with_timestamp(f"Users with >= {min_long_sentences} long sentences ({min_words}-{max_words} words): {len(selected)}")
    return selected


def load_meta_data(meta_file: str) -> Dict[str, dict]:
    """
    加载 meta 文件，建立 asin -> meta_info 的映射。
    支持 2018 版 (asin 字段) 和 2023 版 (parent_asin 字段) 格式。
    2023 数据同时用 asin 和 parent_asin 作为 key，确保评论中的 asin 能匹配到。
    """
    log_with_timestamp(f"Loading meta data from {meta_file}...")
    asin_to_meta: Dict[str, dict] = {}
    count = 0
    with gzip.open(meta_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 2023 版使用 parent_asin，2018 版使用 asin
            asin = meta.get("asin") or meta.get("parent_asin")
            parent_asin = meta.get("parent_asin")
            if asin:
                asin_to_meta[asin] = meta
                count += 1
            # 2023 数据：评论中的 asin 可能是子商品 ID，同时用 asin 字段（如果非空）作为 key
            item_asin = meta.get("asin")
            if item_asin and item_asin != parent_asin:
                asin_to_meta[item_asin] = meta
                count += 1
            if count % 200000 == 0:
                log_with_timestamp(f"  Loaded {count} meta records...")
    log_with_timestamp(f"Loaded {count} meta records")
    return asin_to_meta


def collect_reviews_for_selected_users(
    review_file: str,
    selected_user_ids: Set[str],
    asin_to_meta: Dict[str, dict],
) -> tuple:
    user_products: Dict[str, Set[str]] = defaultdict(set)
    # 优化：直接按用户->商品->评论组织，避免后续重复过滤
    user_asin_reviews: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    log_with_timestamp("Collecting product and review mappings for selected users...")
    with open_possibly_gzipped(review_file, "rt") as f:
        for line in f:
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = review.get("asin")
            reviewer_id = review.get("user_id")
            if not asin or not reviewer_id:
                continue

            # 只收集选中用户的评论，直接按用户和商品组织
            if reviewer_id in selected_user_ids:
                user_asin_reviews[reviewer_id][asin].append(review)
                user_products[reviewer_id].add(asin)

    return user_products, user_asin_reviews


def build_user_output(
    user_id: str,
    user_asins: Set[str],
    user_asin_reviews: Dict[str, Dict[str, List[Dict]]],
    asin_to_meta: Dict[str, dict],
    min_words: int = 25,
    max_words: int = 999999,
) -> dict:
    """
    收集用户的所有长句子（词数在 min_words 和 max_words 之间），按商品组织输出。
    使用 meta 文件中的商品信息。
    返回用户数据字典，不写文件。
    """
    results = []
    user_reviews_by_asin = user_asin_reviews.get(user_id, {})
    for asin in sorted(user_asins):
        target_reviews = []

        for review in user_reviews_by_asin.get(asin, []):
            text = review.get("text", "")
            if not text:
                continue

            # Check if this review has at least one sentence with min_words <= word count <= max_words
            if count_long_sentences(text, min_words, max_words) > 0:
                target_reviews.append(text)

        if not target_reviews:
            continue

        # 从 meta 文件获取商品信息
        meta = asin_to_meta.get(asin, {})
        product_title = meta.get("title", "")
        categories = meta.get("categories", [])
        features = meta.get("features", [])
        price = meta.get("price", "")
        store = meta.get("store", "")
        # 2023 数据的 details 可能是字符串，需要先尝试解析
        details = meta.get("details", {})
        if isinstance(details, dict):
            brand = details.get("Brand", "") or store
        elif isinstance(details, str):
            # details 是字符串格式的 JSON，尝试解析
            try:
                details_dict = json.loads(details)
                brand = details_dict.get("Brand", "") or store
            except:
                brand = store
        else:
            brand = store

        result = {
            "asin": asin,
            "product_title": product_title,
            "categories": categories,
            "features": features,
            "price": price,
            "store": store,
            "brand": brand,
            "target_user_id": user_id,
            "target_reviews_count": len(target_reviews),
            "target_reviews": target_reviews,
            "other_reviews_count": 0,
            "other_reviews": [],
        }
        results.append(result)

    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "total_products": len(results),
        "results": results,
    }
    return output_data


def main() -> None:
    # ============ 硬编码参数 ============
    REVIEW_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/review_categories/Grocery_and_Gourmet_Food.jsonl"
    META_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz"
    OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/00_data_preparation/Grocery_and_Gourmet_Food"
    MIN_WORDS = 15            # 每句话最少词数
    MAX_WORDS = 35            # 每句话最多词数
    MIN_LONG_SENTENCES = 10   # 每个用户最少长句子数（词数在 MIN_WORDS 和 MAX_WORDS 之间）
    MAX_USERS = 0         # 最大用户数（0表示不限制）

    if os.path.isdir(OUTPUT_DIR):
        log_with_timestamp(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0: Batch Data Preparation (Long-Sentence User Filtering)")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"Review file: {REVIEW_FILE}")
    log_with_timestamp(f"Meta file: {META_FILE}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"Min words per sentence: {MIN_WORDS}")
    log_with_timestamp(f"Max words per sentence: {MAX_WORDS}")
    log_with_timestamp(f"Min long-sentence count per user: {MIN_LONG_SENTENCES}")
    log_with_timestamp(f"Max users: {MAX_USERS if MAX_USERS > 0 else 'unlimited'}")

    selected_user_counts = select_users_by_long_reviews(
        REVIEW_FILE,
        MIN_WORDS,
        MAX_WORDS,
        MIN_LONG_SENTENCES,
    )

    selected_user_ids = set(selected_user_counts.keys())
    if not selected_user_ids:
        log_with_timestamp("No users matched criteria. Exiting.")
        return

    # Limit to MAX_USERS if specified
    if MAX_USERS > 0 and len(selected_user_ids) > MAX_USERS:
        selected_user_ids = set(list(sorted(selected_user_ids))[:MAX_USERS])
        log_with_timestamp(f"Limited to {MAX_USERS} users")

    # 加载 meta 数据
    asin_to_meta = load_meta_data(META_FILE)

    user_products, user_asin_reviews = collect_reviews_for_selected_users(
        REVIEW_FILE,
        selected_user_ids,
        asin_to_meta,
    )

    log_with_timestamp(f"Preparing output for {len(selected_user_ids)} users...")
    all_users_data = []
    for idx, user_id in enumerate(sorted(selected_user_ids), start=1):
        log_with_timestamp(f"[{idx}/{len(selected_user_ids)}] Processing user {user_id}...")
        user_data = build_user_output(
            user_id=user_id,
            user_asins=user_products.get(user_id, set()),
            user_asin_reviews=user_asin_reviews,
            asin_to_meta=asin_to_meta,
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
        )
        all_users_data.append(user_data)

    # 写入单个JSON文件
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_users_reviews.json")
    log_with_timestamp(f"Writing all users to single file: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"users": all_users_data}, f, ensure_ascii=False)

    selected_users_file = os.path.join(OUTPUT_DIR, "selected_users.json")
    selected_users = [
        {
            "user_id": user_id,
            "long_sentence_count": selected_user_counts[user_id],
            "product_count": len(user_products.get(user_id, set())),
        }
        for user_id in sorted(selected_user_ids)
    ]
    with open(selected_users_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "selection_criteria": {
                    "min_words": MIN_WORDS,
                    "min_long_sentences": MIN_LONG_SENTENCES,
                    "max_users": MAX_USERS,
                },
                "total_selected": len(selected_users),
                "selected_users": selected_users,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 0 Complete")
    log_with_timestamp(f"Selected users: {len(selected_users)}")
    log_with_timestamp(f"Output directory: {OUTPUT_DIR}")
    log_with_timestamp(f"All users file: {OUTPUT_FILE}")
    log_with_timestamp(f"User list: {selected_users_file}")
    log_with_timestamp("=" * 80)


if __name__ == "__main__":
    main()
