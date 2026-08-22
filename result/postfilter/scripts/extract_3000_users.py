#!/usr/bin/env python3
"""从 Baby_Products_2023 选 3000 用户 (≥15 reviews), 提取 15 句 / 用户.

输出: stage1_filtered_users_reviews_3000u.json (符合 VADES 现有格式)

策略:
- 单 pass 流式 scan Baby_Products_2023.jsonl.gz
- 对每个 user 用 rolling buffer 保留前 MAX_REVIEWS_PER_USER 条 review 的 text
- 全文扫描完后, 选 review_count >= 15 的 user 中前 3000 个
- 对每个 user, 取前 15 句 (extract_sentences_from_review_text)
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
DATA_FILE = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
OUTPUT_FILE = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
LOG_FILE = OUT_DIR / "extract_3000u.log"

TARGET_N_USERS = 3000
MIN_REVIEWS_PER_USER = 15
TOTAL_SENTENCES_PER_USER = 15  # 10 train + 5 holdout


def extract_sentences_from_review_text(text: str) -> list[str]:
    """仿 gaussian_vades.py::extract_sentences_from_review_text: 按 ., !, ?, 。 切句 + 长度过滤."""
    if not text:
        return []
    # spaCy 切句太重, 直接用 regex
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        # 去掉全空白
        if not p:
            continue
        # 长度过滤 (与 VADES pipeline 假定一致: 3-30 词)
        wc = len(p.split())
        if wc < 3 or wc > 100:
            continue
        sentences.append(p)
    return sentences


def main():
    log = lambda m: print(f"[3000u] {m}", flush=True)
    log("=" * 70)
    log(f"提取 {TARGET_N_USERS} 用户 × {TOTAL_SENTENCES_PER_USER} 句")
    log(f"  source: {DATA_FILE}")
    log(f"  target: {OUTPUT_FILE}")
    log("=" * 70)

    # 第 1 遍: 流式 scan, 收集每个 user 的前 30 条 review (留 buffer)
    log("Pass 1: 收集每个 user 前 30 条 review ...")
    user_reviews: dict[str, list[dict]] = {}
    user_count: dict[str, int] = {}
    total_lines = 0
    with gzip.open(DATA_FILE, "rt") as f:
        for line in f:
            total_lines += 1
            r = json.loads(line)
            uid = r.get("user_id")
            text = r.get("text", "")
            if not uid or not text:
                continue
            if user_count.get(uid, 0) >= 30:  # 提前截断
                continue
            user_reviews.setdefault(uid, []).append({
                "asin": r.get("parent_asin") or r.get("asin", ""),
                "text": text,
            })
            user_count[uid] = user_count.get(uid, 0) + 1
            if total_lines % 1000000 == 0:
                log(f"  {total_lines} lines, {len(user_reviews)} users")
    log(f"  Pass 1 done: {total_lines} lines, {len(user_reviews)} users")

    # 第 2 步: 选 ≥15 reviews 的 user, 排序取前 3000
    eligible = [(uid, len(reviews)) for uid, reviews in user_reviews.items()
                if len(reviews) >= MIN_REVIEWS_PER_USER]
    eligible.sort(key=lambda x: -x[1])  # 多的优先
    log(f"  eligible users (≥{MIN_REVIEWS_PER_USER} reviews): {len(eligible)}")
    selected = eligible[:TARGET_N_USERS]
    log(f"  selected: {len(selected)}")

    # 第 3 步: 对每个 user 取 15 句
    log("Pass 2: 提取 15 句 / 用户 ...")
    out_users = []
    excluded = []
    for uid, _ in selected:
        reviews = user_reviews[uid]
        collected: list[dict] = []
        seen: set[str] = set()
        for r_idx, review in enumerate(reviews):
            for s_idx, sentence in enumerate(extract_sentences_from_review_text(review["text"])):
                if sentence in seen:
                    continue
                seen.add(sentence)
                collected.append({
                    "user_id": uid,
                    "review_index": r_idx,
                    "sentence_index": s_idx,
                    "sentence_text": sentence,
                    "word_count": len(sentence.split()),
                })
                if len(collected) == TOTAL_SENTENCES_PER_USER:
                    break
            if len(collected) == TOTAL_SENTENCES_PER_USER:
                break
        if len(collected) < MIN_REVIEWS_PER_USER:
            excluded.append({"user_id": uid, "actual": len(collected)})
            continue
        # 构造 stage1 格式
        out_users.append({
            "user_id": uid,
            "asin": reviews[0]["asin"],  # 用首个 review 的 asin
            "reviews": [{"target_reviews": [r["text"]]} for r in reviews],
            "review_count": len(reviews),
        })

    log(f"  最终输出: {len(out_users)} users")
    log(f"  excluded (sentence < 15): {len(excluded)}")

    # 写文件
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(out_users, f, ensure_ascii=False)
    log(f"  已写入 {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
