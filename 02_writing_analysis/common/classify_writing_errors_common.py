#!/usr/bin/env python3
"""Common functions for filtering simple writing errors from user reviews."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


WRITING_ANALYSIS_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/fs04/ar57/wenyu/result/personal_query/01_preference_extraction")
PROGRESS_INTERVAL_USERS = 100


from common_utils import log  # 统一 log 函数


# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']


def _edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def is_simple_error(original: str, corrected: str) -> Tuple[bool, str]:
    """判断是否是简单错误（应被过滤）
    
    Returns:
        (is_simple, reason): (True, reason) 如果是简单错误
                             (False, "") 如果不是简单错误
    """
    import string
    
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    if not orig or not corr:
        return False, "empty"
    if orig == corr:
        return False, "identity"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "multi_word"

    # 过滤仅相差标点的错误（如 a. -> a）
    PUNCT = set(string.punctuation)
    orig_no_punct = ''.join(c for c in orig if c not in PUNCT)
    corr_no_punct = ''.join(c for c in corr if c not in PUNCT)
    if orig_no_punct == corr_no_punct and orig != corr:
        return True, "punct_only"

    # 过滤仅相差单引号的错误（如 dont -> don't）
    orig_no_quote = orig.replace("'", "")
    corr_no_quote = corr.replace("'", "")
    if orig_no_quote == corr_no_quote and orig != corr:
        return True, "quote_only"

    # 过滤编辑距离 <= 2 的简单词汇错误（如 creat->create, an->a）
    if len(orig) >= 2 and len(corr) >= 2:
        dist = _edit_distance(orig, corr)
        if dist <= 2:
            return True, "edit_dist_le2"

    # 过滤长度相差1且所有字符都在另一个词中的情况（如 a->an）
    if abs(len(orig) - len(corr)) == 1:
        shorter, longer = (orig, corr) if len(orig) < len(corr) else (corr, orig)
        if all(c in longer for c in shorter):
            return True, "char_subset"

    # 过滤常见的主谓不一致/动词形式错误
    COMMON_VERB_VARIATIONS = {
        ('was', 'were'), ('is', 'are'), ('are', 'is'),
        ('do', 'did'), ('does', 'did'),
    }
    if (orig, corr) in COMMON_VERB_VARIATIONS or (corr, orig) in COMMON_VERB_VARIATIONS:
        return True, "verb_variation"

    # 过滤人称代词变化
    COMMON_PRONOUN_VARIATIONS = {
        ('i', 'me'), ('me', 'i'),
        ('he', 'him'), ('him', 'he'),
        ('she', 'her'), ('her', 'she'),
        ('we', 'us'), ('us', 'we'),
        ('they', 'them'), ('them', 'they'),
    }
    if (orig, corr) in COMMON_PRONOUN_VARIATIONS or (corr, orig) in COMMON_PRONOUN_VARIATIONS:
        return True, "pronoun_variation"

    # 过滤词缀变化（如 dog->dogs, jump->jumped）
    if len(orig) > len(corr):
        orig, corr = corr, orig
    if len(corr) - len(orig) >= 1 and len(orig) >= 3:
        for affix in COMMON_AFFIXES:
            if corr == orig + affix:
                return True, "affix_variation"

    return False, ""


def get_category_paths(category: str) -> Tuple[Path, Path]:
    """获取输入输出文件路径"""
    input_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    return input_file, output_file


def process_user(user_data: Dict) -> Dict:
    """处理单个用户的数据，提取并过滤写作错误"""
    user_id = user_data.get("user_id", "")
    results = user_data.get("results", [])
    
    all_errors: List[Dict] = []
    simple_counts: Counter = Counter()
    
    for result in results:
        reviews = result.get("target_reviews", [])
        for review in reviews:
            # 从评论中提取错误（这里需要根据实际的错误提取逻辑）
            # 暂时跳过，因为需要知道错误的格式
            pass
    
    return {
        "user_id": user_id,
        "total_errors": len(all_errors),
        "filtered_errors": 0,
        "simple_error_counts": dict(simple_counts),
    }


def format_elapsed(start_time: float) -> str:
    """格式化耗时"""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m{seconds}s"


def classify_category(category: str) -> None:
    """处理单个类别的错误分类"""
    start_time = time.time()
    input_file, output_file = get_category_paths(category)
    log(f"=== Processing category: {category} ===")
    log(f"[{category}] Reading from: {input_file}")

    if not input_file.exists():
        log(f"[{category}] File not found: {input_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        log(f"[{category}] Created empty output file: {output_file}")
        return

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    users = data.get("users", [])
    log(f"[{category}] Loaded {len(users)} users")

    output_rows = []
    total_filtered = 0
    total_simple = 0
    simple_counts_total: Counter = Counter()

    for idx, user in enumerate(users, start=1):
        output_row = process_user(user)
        output_rows.append(output_row)

        if idx % PROGRESS_INTERVAL_USERS == 0 or idx == len(users):
            log(
                f"[{category}] Processed {idx}/{len(users)} users; "
                f"elapsed={format_elapsed(start_time)}"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{category}] Writing results to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    log(f"[{category}] Total: {len(users)} users")
    log(f"=== Finished: {category}; elapsed={format_elapsed(start_time)} ===")
