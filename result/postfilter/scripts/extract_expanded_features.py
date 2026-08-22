#!/usr/bin/env python3
"""扩展 raw 特征到 50-100d: POS / 功能词 / 长度 / 词汇丰富度 / 标点.

输入: vades_disentangled_v2_train10_holdout10_sentences.jsonl (4500 句)
输出: expanded_features_55d.jsonl (4500 句, 每句 55-d 特征)

特征设计 (合计 ~55d):
- POS 分布 (17d): 17 个 Universal POS tag 相对频率
- 功能词 (15d): 常见冠词/介词/连词/助动词的频率
- 长度 (5d): word_count, mean_word_len, max_word_len, char_count, num_unique
- 词汇丰富度 (3d): type-token ratio, hapax ratio, content word density
- 标点 (5d): !, ?, ;, :, ... 计数
- 句子结构 (5d): 复用部分 20-d 特征

总: 50d 新 + 5d 复用 = 55d
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import spacy

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result/personal_query/12_complexity_analysis_clause_features/Baby_Products"
SENTENCE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_sentences.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
OUT_FILE = OUT_DIR / "expanded_features_52d.jsonl"

# 17 Universal POS tags by spaCy
POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

# 17 high-frequency function words (lower-case)
FUNCTION_WORDS = [
    "the", "a", "an",                                   # articles
    "of", "in", "to", "for", "with", "on", "at", "by",  # prepositions
    "and", "but", "or",                                  # conjunctions
    "is", "are", "was",                                  # aux verbs
]

# Reused 20-d features subset (5 dims)
REUSED_FEATURES = [
    "max_dependency_depth",      # 句法复杂度
    "acl_count",                  # 从句密度
    "mean_dependency_distance",  # 依存距离
    "modifier_density",          # 修饰密度
    "max_branching_factor",      # 分支复杂度
]

assert len(POS_TAGS) == 17
assert len(FUNCTION_WORDS) == 17
assert len(REUSED_FEATURES) == 5
# 17 (POS) + 17 (func) + 5 (length) + 3 (richness) + 5 (punct) + 5 (reused) = 52
TOTAL_DIM = len(POS_TAGS) + len(FUNCTION_WORDS) + 5 + 3 + 5 + 5
print(f"Total feature dim: {TOTAL_DIM}")


def extract_features(nlp, doc, sentence_text: str, original_features: dict) -> dict:
    """对单个句子 (spaCy doc) 提取 50-d 特征."""
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {f"f{i}": 0.0 for i in range(TOTAL_DIM)}

    # POS 分布 (17d, 相对频率)
    pos_counts = Counter(t.pos_ for t in tokens)
    pos_freq = {pos: pos_counts.get(pos, 0) / n_tokens for pos in POS_TAGS}

    # 功能词频率 (15d)
    word_lower = Counter(t.text.lower() for t in tokens if t.is_alpha)
    n_words = sum(word_lower.values())
    func_freq = {fw: word_lower.get(fw, 0) / max(1, n_words) for fw in FUNCTION_WORDS}

    # 长度 (5d)
    n_unique = len(set(t.text.lower() for t in tokens if t.is_alpha))
    word_lengths = [len(t.text) for t in tokens if t.is_alpha]
    mean_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
    max_word_len = float(max(word_lengths)) if word_lengths else 0.0
    char_count = len(sentence_text)
    word_count = n_tokens

    # 词汇丰富度 (3d)
    # TTR (type-token ratio)
    ttr = n_unique / max(1, n_words)
    # Hapax ratio (words appearing once)
    hapax_count = sum(1 for w, c in word_lower.items() if c == 1)
    hapax_ratio = hapax_count / max(1, len(word_lower))
    # Content word density (NOUN + VERB + ADJ + ADV)
    content_count = pos_counts.get("NOUN", 0) + pos_counts.get("VERB", 0) + \
                    pos_counts.get("ADJ", 0) + pos_counts.get("ADV", 0)
    content_density = content_count / n_tokens

    # 标点 (5d)
    punct_counts = Counter(t.text for t in tokens if t.pos_ == "PUNCT")
    punct_excl = punct_counts.get("!", 0)
    punct_quest = punct_counts.get("?", 0)
    punct_semi = punct_counts.get(";", 0)
    punct_colon = punct_counts.get(":", 0)
    punct_dash = punct_counts.get("-", 0) + punct_counts.get("—", 0)
    total_punct = sum(punct_counts.values())

    # 复用 5d 句法特征
    reused = {f: float(original_features.get(f, 0.0)) for f in REUSED_FEATURES}

    # 装配为有序 dict
    feats = {}
    for pos in POS_TAGS:
        feats[f"pos_{pos}"] = pos_freq[pos]
    for fw in FUNCTION_WORDS:
        feats[f"func_{fw}"] = func_freq[fw]
    feats["len_word_count"] = float(word_count)
    feats["len_mean_word_len"] = mean_word_len
    feats["len_max_word_len"] = max_word_len
    feats["len_char_count"] = float(char_count)
    feats["len_n_unique"] = float(n_unique)
    feats["rich_ttr"] = ttr
    feats["rich_hapax_ratio"] = hapax_ratio
    feats["rich_content_density"] = content_density
    feats["punct_excl"] = float(punct_excl)
    feats["punct_quest"] = float(punct_quest)
    feats["punct_semi"] = float(punct_semi)
    feats["punct_colon"] = float(punct_colon)
    feats["punct_dash"] = float(punct_dash)
    for k, v in reused.items():
        feats[f"reused_{k}"] = v

    assert len(feats) == TOTAL_DIM, f"expected {TOTAL_DIM}, got {len(feats)}"
    return feats


def main():
    log = lambda m: print(f"[expand] {m}", flush=True)
    log("=" * 70)
    log(f"扩展 raw 特征到 {TOTAL_DIM}d")
    log("=" * 70)

    log("加载句子 ...")
    with SENTENCE_FILE.open() as f:
        rows = [json.loads(line) for line in f]
    log(f"  {len(rows)} 句")

    log("加载 spaCy (only tagger+parser) ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    log(f"批量解析 {len(rows)} 句 ...")
    texts = [r["sentence_text"] for r in rows]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.unlink(missing_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        n_done = 0
        for doc, row in zip(nlp.pipe(texts, batch_size=128), rows):
            feats = extract_features(nlp, doc, row["sentence_text"], row["features"])
            out = {
                "user_id": row["user_id"],
                "review_index": row["review_index"],
                "sentence_index": row["sentence_index"],
                "sentence_text": row["sentence_text"],
                "is_holdout": row.get("is_holdout", False),
                "expanded_features": feats,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 1000 == 0:
                log(f"  {n_done}/{len(rows)} done")
    log(f"  完成 {n_done} 句, 已写入 {OUT_FILE}")

    # 统计
    log("\n样本特征 (first sentence):")
    with OUT_FILE.open() as f:
        first = json.loads(f.readline())
    for k, v in list(first["expanded_features"].items())[:10]:
        log(f"  {k}: {v:.4f}")
    log(f"  ... ({TOTAL_DIM} dims total)")


if __name__ == "__main__":
    main()
