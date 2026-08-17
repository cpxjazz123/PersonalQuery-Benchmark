#!/usr/bin/env python3
"""E25: rewrite 质量综合分析。

输入:
  - result/e23_style_vector/scale_rewrites.jsonl   (13984)
  - result/e23_style_vector/heldout_rewrites.jsonl (800)
  - /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_rewrites.jsonl (10382)

评估维度:
  1. 长度: 原句 vs rewrite 词数分布（diff、ratio）
  2. 失败信号: 空/极短(<3)/超长(>100)/与原文完全相同
  3. 词重叠: Jaccard / word-overlap-ratio（rewrite 应保留核心事实）
  4. 编辑距离: character-level Levenshtein（句子结构变化幅度）
  5. 事实保留: 数字/品牌词保持率
  6. 风格清理: 情感词/人称代词删除率（应该高）
  7. 重写多样性: per-source rewrite 的多样性（同一原句多改写是否多样）
  8. 长度变化分布
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
OUT_E25 = Path("/home/wlia0047/hj82_scratch2/wenyu/e25_rewrite_quality")
OUT_E25.mkdir(parents=True, exist_ok=True)

E23_SCALE = REPO_ROOT / "result" / "e23_style_vector" / "scale_rewrites.jsonl"
E23_HO = REPO_ROOT / "result" / "e23_style_vector" / "heldout_rewrites.jsonl"
E24_TEST = Path("/home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_rewrites.jsonl")
OUT_JSON = OUT_E25 / "rewrite_quality.json"

# sentiment / intensifier / person words that SHOULD be removed by the rewrite
PERSON_WORDS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                "you", "your", "yours", "he", "she", "him", "her", "they", "them"}
SENTIMENT_HINTS = [
    "love", "loved", "loves", "lovely", "loving",
    "hate", "hated", "hates",
    "amazing", "awesome", "wonderful", "fantastic", "great", "perfect",
    "terrible", "horrible", "awful", "bad", "worst",
    "best", "super", "extremely", "absolutely", "totally", "really",
    "very", "quite", "pretty", "much", "many", "sooo", "soo",
    "definitely", "obviously", "seriously",
    "yay", "ugh", "wow", "omg", "lol",
    "cute", "adorable", "beautiful", "gorgeous",
    "disappointed", "frustrated", "annoyed",
    "happy", "sad", "angry", "excited",
    "!!", "!!!", "!?", "!?!", "?!!",
]
SENTIMENT_PATTERNS = [
    r"\b(i|we)\s+(love|liked|hate|hated|adore|adored|recommend|"
    r"disrecommend|want|wanted|need|needed|prefer|think|believe|"
    r"feel|felt)\b",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def levenshtein(a: str, b: str) -> int:
    """Standard O(|a||b|) edit distance (case-sensitive, char-level)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_ratio(a: list[str], b: list[str]) -> float:
    """Fraction of source words that appear in rewrite (recall-style)."""
    sa, sb = set(a), set(b)
    if not sa:
        return 1.0
    return len(sa & sb) / len(sa)


def detect_person_words(s: str) -> int:
    return sum(1 for w in words(s) if w in PERSON_WORDS)


def detect_sentiment_hits(s: str) -> int:
    n = 0
    for w in words(s):
        if w in SENTIMENT_HINTS:
            n += 1
    return n


def detect_exclamations(s: str) -> int:
    return s.count("!") + s.count("?!")


def extract_numbers(s: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", s)


def detect_caps(s: str) -> int:
    """Count uppercased words (>=3 chars, all caps)."""
    return sum(1 for w in re.findall(r"\b[A-Z]{2,}\b", s))


def main() -> None:
    t0 = time.time()
    pairs: list[tuple[str, str, str]] = []  # (source, rewrite, source_label)
    for path, label in [
        (E23_SCALE, "e23_scale"),
        (E23_HO, "e23_heldout"),
        (E24_TEST, "e24_test"),
    ]:
        if not path.exists():
            log(f"WARN missing: {path}")
            continue
        n_before = len(pairs)
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    pairs.append((d["sentence"], d["rewrite"], label))
                except Exception:
                    pass
        log(f"loaded {len(pairs) - n_before} from {label}")
    log(f"total pairs: {len(pairs)}")

    if not pairs:
        raise RuntimeError("no pairs to analyze")

    # ---- per-pair metrics ----
    metrics: list[dict] = []
    n_empty, n_too_short, n_too_long, n_identical = 0, 0, 0, 0
    src_lens, rw_lens = [], []
    src_person, rw_person = [], []
    src_sent, rw_sent = [], []
    src_excl, rw_excl = [], []
    src_caps, rw_caps = [], []
    src_nums, rw_nums = [], []
    jaccards, overlaps, edit_dists = [], [], []
    len_ratios = []
    for src, rw, lbl in pairs:
        rw_stripped = rw.strip()
        if not rw_stripped:
            n_empty += 1
            continue
        sw, ww = words(src), words(rw_stripped)
        if len(ww) < 3:
            n_too_short += 1
        if len(ww) > 100:
            n_too_long += 1
        if src_strip := src.strip():
            if rw_stripped == src_strip:
                n_identical += 1
        src_lens.append(len(sw))
        rw_lens.append(len(ww))
        src_person.append(detect_person_words(src))
        rw_person.append(detect_person_words(rw_stripped))
        src_sent.append(detect_sentiment_hits(src))
        rw_sent.append(detect_sentiment_hits(rw_stripped))
        src_excl.append(detect_exclamations(src))
        rw_excl.append(detect_exclamations(rw_stripped))
        src_caps.append(detect_caps(src))
        rw_caps.append(detect_caps(rw_stripped))
        src_n_list = extract_numbers(src)
        rw_n_list = extract_numbers(rw_stripped)
        src_nums.append(len(src_n_list))
        rw_nums.append(len(rw_n_list))
        jaccards.append(jaccard(sw, ww))
        overlaps.append(overlap_ratio(sw, ww))
        # edit distance on truncated strings (cap at 500 chars for speed)
        ed = levenshtein(src[:500], rw_stripped[:500])
        edit_dists.append(ed / max(len(src), len(rw_stripped), 1))
        len_ratios.append(len(ww) / max(len(sw), 1))

    def stats(arr, name):
        return {
            "mean": round(statistics.mean(arr), 3),
            "median": round(statistics.median(arr), 3),
            "stdev": round(statistics.stdev(arr), 3) if len(arr) > 1 else 0.0,
            "min": min(arr),
            "max": max(arr),
        }

    n_pairs = len(pairs)
    n_analyzed = len(metrics) + len(src_lens)
    log(f"analyzed {n_analyzed}/{n_pairs} pairs (skipped: empty={n_empty})")

    # ---- person-word removal: per-source presence → per-rewrite presence ----
    src_has_person = sum(1 for p in src_person if p > 0)
    src_removed_person = sum(1 for s, r in zip(src_person, rw_person)
                             if s > 0 and r == 0)
    src_has_sent = sum(1 for s in src_sent if s > 0)
    src_removed_sent = sum(1 for s, r in zip(src_sent, rw_sent)
                           if s > 0 and r == 0)
    src_has_excl = sum(1 for s in src_excl if s > 0)
    src_removed_excl = sum(1 for s, r in zip(src_excl, rw_excl)
                           if s > 0 and r == 0)
    src_has_caps = sum(1 for s in src_caps if s > 0)
    src_removed_caps = sum(1 for s, r in zip(src_caps, rw_caps)
                           if s > 0 and r == 0)
    # number preservation
    n_with_src_num = sum(1 for n in src_nums if n > 0)
    n_kept_num = sum(1 for s, r in zip(src_nums, rw_nums)
                     if s > 0 and r >= s)
    n_partial_num = sum(1 for s, r in zip(src_nums, rw_nums)
                        if s > 0 and 0 < r < s)

    summary = {
        "version": "e25_rewrite_quality_v1",
        "n_pairs": n_pairs,
        "n_analyzed": len(src_lens),
        "n_empty": n_empty,
        "n_too_short(<3 words)": n_too_short,
        "n_too_long(>100 words)": n_too_long,
        "n_identical_to_source": n_identical,
        "src_word_len_stats": stats(src_lens, "src"),
        "rw_word_len_stats": stats(rw_lens, "rw"),
        "len_ratio(rw/src)": stats(len_ratios, "len_ratio"),
        "edit_distance_norm_stats": stats(edit_dists, "ed"),
        "word_overlap_ratio(recall)": stats(overlaps, "ovr"),
        "jaccard_stats": stats(jaccards, "jacc"),
        # cleanup effectiveness
        "person_words": {
            "src_with_person_words": src_has_person,
            "rewrite_removed_all_person_words": src_removed_person,
            "removal_rate": round(src_removed_person / max(src_has_person, 1), 3),
        },
        "sentiment_words": {
            "src_with_sentiment_words": src_has_sent,
            "rewrite_removed_all_sentiment_words": src_removed_sent,
            "removal_rate": round(src_removed_sent / max(src_has_sent, 1), 3),
        },
        "exclamations": {
            "src_with_exclamations": src_has_excl,
            "rewrite_removed_all_exclamations": src_removed_excl,
            "removal_rate": round(src_removed_excl / max(src_has_excl, 1), 3),
        },
        "all_caps_words": {
            "src_with_all_caps": src_has_caps,
            "rewrite_removed_all_caps": src_removed_caps,
            "removal_rate": round(src_removed_caps / max(src_has_caps, 1), 3),
        },
        "numbers": {
            "src_with_numbers": n_with_src_num,
            "rewrite_kept_all_numbers": n_kept_num,
            "rewrite_partial_numbers": n_partial_num,
            "preservation_rate": round(n_kept_num / max(n_with_src_num, 1), 3),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }

    # ---- sample qualitative inspection ----
    samples = []
    rng = list(zip(pairs, edit_dists, jaccards, overlaps, src_lens, rw_lens))
    # sort by edit distance to find typical, very-low, very-high
    rng.sort(key=lambda x: x[1])
    log("=== lowest edit (rewrites barely changed) ===")
    for (src, rw, lbl), ed, jc, ov, sl, rl in rng[:5]:
        samples.append({"src": src[:200], "rw": rw[:200], "label": lbl,
                        "ed_norm": round(ed, 3),
                        "jacc": round(jc, 3), "overlap": round(ov, 3),
                        "src_len": sl, "rw_len": rl,
                        "category": "low_edit"})
    log("=== highest edit (rewrites changed a lot) ===")
    for (src, rw, lbl), ed, jc, ov, sl, rl in rng[-5:]:
        samples.append({"src": src[:200], "rw": rw[:200], "label": lbl,
                        "ed_norm": round(ed, 3),
                        "jacc": round(jc, 3), "overlap": round(ov, 3),
                        "src_len": sl, "rw_len": rl,
                        "category": "high_edit"})
    log("=== shortest rewrites ===")
    rng_short = sorted(zip(pairs, rw_lens, src_lens), key=lambda x: x[1])
    for (src, rw, lbl), rl, sl in rng_short[:5]:
        samples.append({"src": src[:200], "rw": rw[:200], "label": lbl,
                        "src_len": sl, "rw_len": rl,
                        "category": "short_rw"})
    log("=== longest rewrites ===")
    for (src, rw, lbl), rl, sl in rng_short[-5:]:
        samples.append({"src": src[:200], "rw": rw[:200], "label": lbl,
                        "src_len": sl, "rw_len": rl,
                        "category": "long_rw"})

    summary["samples"] = samples

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE — wrote {OUT_JSON}")
    log(f"  src_len mean={summary['src_word_len_stats']['mean']:.1f} "
        f"rw_len mean={summary['rw_word_len_stats']['mean']:.1f} "
        f"ratio={summary['len_ratio(rw/src)']['mean']:.2f}")
    log(f"  jacc mean={summary['jaccard_stats']['mean']:.3f} "
        f"overlap mean={summary['word_overlap_ratio(recall)']['mean']:.3f}")
    log(f"  person removal={summary['person_words']['removal_rate']:.3f}, "
        f"sent removal={summary['sentiment_words']['removal_rate']:.3f}, "
        f"excl removal={summary['exclamations']['removal_rate']:.3f}")
    log(f"  number preservation={summary['numbers']['preservation_rate']:.3f}")
    log(f"  identical_to_src={n_identical}, too_short={n_too_short}, "
        f"too_long={n_too_long}, empty={n_empty}")


if __name__ == "__main__":
    main()