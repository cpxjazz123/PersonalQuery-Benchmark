#!/usr/bin/env python3
"""[Ablation] Long-sentence user filter threshold sweep.

P0 reviewer concern from iter #38:
  "User filter thresholds (≥20 reviews, ≥15 words) lack ablation backing."

Hardcoded constants in 00_data_preparation/00_batch_prepare_data_*.py:
  - MIN_WORDS = 15  (min words per sentence)
  - MAX_WORDS = 35  (max words per sentence, "long sentence" window)
  - MIN_LONG_SENTENCES = 10  (min long sentences per user)

This ablation:
  1. Loads Stage 1 filtered_users_reviews.json per domain.
  2. For each user, recomputes long-sentence count (sentences with
     15 <= word_count <= 35).
  3. Sweeps MIN_LONG_SENTENCES ∈ {5, 10, 15, 20} and reports how many
     users would survive each threshold.
  4. Computes mean query generation features (user_avg_depth, word_count)
     for the surviving users, and reports whether downstream query
     distribution is sensitive to the threshold.

Output: JSON + stdout table.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
REVIEW_FILE_TEMPLATE = str(RESULT_ROOT / "01_preference_extraction" / "{cat}" / "stage1_filtered_users_reviews.json")
QUERY_FILE_TEMPLATE = str(RESULT_ROOT / "06_query" / "{cat}" / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json")

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

MIN_WORDS_DEFAULT = 15
MAX_WORDS_DEFAULT = 35
THRESHOLDS = [5, 10, 15, 20]


def _count_long_sentences(text: str, min_words: int, max_words: int) -> int:
    import re
    sentences = re.split(r"[.!?]+", text)
    return sum(1 for s in sentences if min_words <= len(s.split()) <= max_words)


def _build_long_sentence_index(category: str, min_words: int, max_words: int) -> dict:
    review_path = Path(REVIEW_FILE_TEMPLATE.format(cat=category))
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    with open(review_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    index: dict = {}
    for user in d.get("users", []):
        uid = user.get("user_id")
        if not uid:
            continue
        count = 0
        for result in user.get("results", []):
            for review in result.get("target_reviews", []):
                if isinstance(review, str):
                    count += _count_long_sentences(review, min_words, max_words)
        if count > 0:
            index[uid] = count
    return index


def _load_query_features(category: str) -> dict:
    query_path = Path(QUERY_FILE_TEMPLATE.format(cat=category))
    if not query_path.exists():
        return {}
    with open(query_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    per_user = defaultdict(list)
    for row in rows:
        uid = row.get("user_id")
        sq = row.get("syntax_depth_query", {})
        if not uid or not sq:
            continue
        per_user[uid].append(sq)
    features: dict = {}
    for uid, queries in per_user.items():
        words = [q["word_count"] for q in queries if isinstance(q.get("word_count"), (int, float))]
        avgs = [q["user_avg_depth"] for q in queries if isinstance(q.get("user_avg_depth"), (int, float))]
        if not words:
            continue
        features[uid] = {
            "mean_query_words": statistics.mean(words),
            "mean_user_avg_depth": statistics.mean(avgs) if avgs else float("nan"),
            "n_queries": len(words),
        }
    return features


def _ablation_for_category(category: str, thresholds: list[int]) -> dict:
    long_idx = _build_long_sentence_index(category, MIN_WORDS_DEFAULT, MAX_WORDS_DEFAULT)
    query_features = _load_query_features(category)
    # cross-set
    overlap_users = set(long_idx) & set(query_features)
    result = {
        "category": category,
        "n_users_stage1": len(long_idx),
        "n_users_stage6": len(query_features),
        "n_users_overlap": len(overlap_users),
        "min_words": MIN_WORDS_DEFAULT,
        "max_words": MAX_WORDS_DEFAULT,
        "thresholds": thresholds,
        "sweep": [],
    }
    for thr in thresholds:
        surviving_users = {uid for uid, cnt in long_idx.items() if cnt >= thr}
        overlap_surviving = surviving_users & overlap_users
        if overlap_surviving:
            mean_query_words = statistics.mean(query_features[uid]["mean_query_words"] for uid in overlap_surviving)
            mean_user_avg_depth = statistics.mean(query_features[uid]["mean_user_avg_depth"] for uid in overlap_surviving if not math.isnan(query_features[uid]["mean_user_avg_depth"]))
        else:
            mean_query_words = float("nan")
            mean_user_avg_depth = float("nan")
        result["sweep"].append({
            "min_long_sentences": thr,
            "n_surviving_all": len(surviving_users),
            "n_surviving_overlap": len(overlap_surviving),
            "frac_overlap_surviving": len(overlap_surviving) / max(1, len(overlap_users)),
            "mean_query_words_overlap_surviving": mean_query_words,
            "mean_user_avg_depth_overlap_surviving": mean_user_avg_depth,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES)
    args = parser.parse_args()
    targets = [args.category] if args.category else CATEGORIES
    results = []
    for category in targets:
        result = _ablation_for_category(category, THRESHOLDS)
        results.append(result)
        print(f"\n=== {category} ===")
        print(f"  n_users_stage1: {result['n_users_stage1']}, n_users_stage6: {result['n_users_stage6']}, overlap: {result['n_users_overlap']}")
        print(f"  {'MIN_LONG_SENTENCES':>20} {'surviving':>10} {'overlap':>10} {'frac':>8} {'mean_query_words':>18} {'mean_user_avg_depth':>20}")
        for s in result["sweep"]:
            print(
                f"  {s['min_long_sentences']:>20} {s['n_surviving_all']:>10} {s['n_surviving_overlap']:>10} "
                f"{s['frac_overlap_surviving']:>8.3f} {s['mean_query_words_overlap_surviving']:>18.3f} "
                f"{s['mean_user_avg_depth_overlap_surviving']:>20.3f}"
            )

    print("\n=== Cross-domain summary ===")
    print(f"{'Domain':<26} {'@thr=5':>8} {'@thr=10':>8} {'@thr=15':>8} {'@thr=20':>8}")
    print("-" * 70)
    for r in results:
        fracs = [f"{s['frac_overlap_surviving']:.3f}" for s in r["sweep"]]
        print(f"{r['category']:<26} {' '.join(f'{f:>8}' for f in fracs)}")

    out_dir = RESULT_ROOT / "00_data_preparation" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "long_sentence_threshold_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
