#!/usr/bin/env python3
"""Run parse-only pipeline for Grocery_and_Gourmet_Food (sentence extraction + feature extraction)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Grocery_and_Gourmet_Food"

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module


def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = CATEGORY
    train_module.INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
    train_module.REVIEW_SOURCE_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
    train_module.RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_expression_style_no_depth_check_10.json"
    train_module.CANDIDATE_QUERY_FILE = train_module.INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    train_module.SENTENCE_EXTRACT_CACHE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_extracted_sentences.jsonl"
    train_module.SENTENCE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
    train_module.EXCLUDED_USER_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
    train_module.QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{train_module.OUTPUT_TAG}.json"

    log(f"解析阶段开始 - CATEGORY: {CATEGORY}")
    
    # Step 1: 过滤重复用户
    log("Step 1: 过滤有重复 target_reviews 的用户")
    train_module.filter_users_with_duplicate_reviews()
    
    # Step 2: 加载候选 query 用户
    log("Step 2: 加载候选 query 用户")
    candidate_rows = train_module.load_candidate_query_rows()
    candidate_user_ids = {row["user_id"] for row in candidate_rows}
    user_rows = train_module.load_filtered_user_reviews()
    user_rows = [row for row in user_rows if row.get("user_id") in candidate_user_ids]
    log(f"候选用户数: {len(user_rows)}")
    
    # Step 3: 提取句子
    if train_module.SENTENCE_EXTRACT_CACHE_FILE.exists():
        log(f"从缓存加载句子: {train_module.SENTENCE_EXTRACT_CACHE_FILE}")
        cache_data = train_module.load_json(train_module.SENTENCE_EXTRACT_CACHE_FILE)
        if isinstance(cache_data, dict) and "kept_rows" in cache_data and "excluded_rows" in cache_data:
            sentence_rows = cache_data["kept_rows"]
            excluded_rows = cache_data["excluded_rows"]
            log(f"已从缓存加载 {len(sentence_rows)} 条句子，{len(excluded_rows)} 个排除用户")
        else:
            log("缓存格式错误，将重新提取句子")
            sentence_rows, excluded_rows = train_module.extract_first_twenty_sentences_for_users(user_rows)
            train_module.SENTENCE_EXTRACT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with train_module.SENTENCE_EXTRACT_CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump({"kept_rows": sentence_rows, "excluded_rows": excluded_rows}, f, ensure_ascii=False)
            log(f"已保存句子到缓存: {train_module.SENTENCE_EXTRACT_CACHE_FILE}")
    else:
        log("开始提取句子")
        sentence_rows, excluded_rows = train_module.extract_first_twenty_sentences_for_users(user_rows)
        train_module.SENTENCE_EXTRACT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with train_module.SENTENCE_EXTRACT_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump({"kept_rows": sentence_rows, "excluded_rows": excluded_rows}, f, ensure_ascii=False)
        log(f"已保存句子到缓存: {train_module.SENTENCE_EXTRACT_CACHE_FILE}")
    
    # Step 4: 提取句法特征
    log("Step 4: 提取评论句法特征")
    enriched_sentence_rows, feature_names = train_module.build_sentence_feature_rows(sentence_rows)
    log(f"句法特征提取完成: {len(enriched_sentence_rows)} 条句子, {len(feature_names)} 个特征")
    
    # Step 5: 保存解析结果
    log("Step 5: 保存解析结果")
    train_module.SENTENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_module.write_jsonl(train_module.SENTENCE_FILE, enriched_sentence_rows)
    train_module.write_jsonl(train_module.EXCLUDED_USER_FILE, excluded_rows)
    log(f"解析结果已保存:")
    log(f"  - 句子特征: {train_module.SENTENCE_FILE}")
    log(f"  - 排除用户: {train_module.EXCLUDED_USER_FILE}")
    
    log("解析阶段完成！可以运行训练脚本继续。")


if __name__ == "__main__":
    main()
