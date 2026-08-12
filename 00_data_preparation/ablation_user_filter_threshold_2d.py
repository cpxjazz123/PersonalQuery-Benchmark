#!/usr/bin/env python3
"""[Ablation] Paper §2.1 user filter threshold 2-D sweep (≥20 reviews × ≥15 words).

P0 backlog item: paper_claims_audit UserFilter_20_reviews_15_words.
  paper §2.1 line 48: "PQB retains only users with at least 20 historical
  reviews and requires each review to contain at least 15 words."

Stage 1 code (00_batch_prepare_data_<cat>.py) hardcodes:
  MIN_WORDS = 15, MAX_WORDS = 35, MIN_LONG_SENTENCES = 10
These are per-sentence word-window + per-user long-sentence count thresholds,
NOT user-level (review_count ≥ 20, word_count ≥ 15).

iter #73 ablation (ablation_long_sentence_threshold.py) swept MIN_LONG_SENTENCES,
not review_count or word_count.

iter #185 NEW ablation: 2-D sweep (min_reviews, min_words):
  - For each user, count distinct reviews and per-review word counts
    from stage1_filtered_users_reviews.json.
  - Sweep min_reviews ∈ {5, 10, 15, 20} × min_words ∈ {5, 10, 15} = 12 cells.
  - Report n_surviving per cell.
  - Compare paper's claimed threshold (20, 15) against actual user count.

Output: JSON + stdout 2-D table per domain.
"""

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
REVIEW_FILE_TEMPLATE = str(
    RESULT_ROOT / "01_preference_extraction" / "{cat}" / "stage1_filtered_users_reviews.json"
)

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

MIN_REVIEWS_GRID = [5, 10, 15, 20]
MIN_WORDS_GRID = [5, 10, 15]

PAPER_CLAIM = {"min_reviews": 20, "min_words": 15}


def _count_user_reviews(user: dict) -> tuple[int, list[int]]:
    """Return (n_reviews, per_review_word_counts)."""
    counts: list[int] = []
    for result in user.get("results", []):
        for review in result.get("target_reviews", []):
            if isinstance(review, str):
                counts.append(len(review.split()))
    return len(counts), counts


def _build_user_review_stats(category: str) -> dict:
    review_path = Path(REVIEW_FILE_TEMPLATE.format(cat=category))
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    with open(review_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    stats: dict[str, dict] = {}
    for user in d.get("users", []):
        uid = user.get("user_id")
        if not uid:
            continue
        n_rev, word_counts = _count_user_reviews(user)
        if n_rev == 0:
            continue
        stats[uid] = {
            "n_reviews": n_rev,
            "mean_words": statistics.mean(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }
    return stats


def _ablation_for_category(category: str) -> dict:
    user_stats = _build_user_review_stats(category)
    n_total = len(user_stats)

    n_rev_dist = [s["n_reviews"] for s in user_stats.values()]
    mean_words_dist = [s["mean_words"] for s in user_stats.values()]

    sweep = []
    for mr in MIN_REVIEWS_GRID:
        for mw in MIN_WORDS_GRID:
            surviving = {
                uid for uid, s in user_stats.items()
                if s["n_reviews"] >= mr and s["mean_words"] >= mw
            }
            sweep.append({
                "min_reviews": mr,
                "min_words": mw,
                "n_surviving": len(surviving),
                "frac_surviving": len(surviving) / max(1, n_total),
            })

    paper_match = next(
        s for s in sweep
        if s["min_reviews"] == PAPER_CLAIM["min_reviews"]
        and s["min_words"] == PAPER_CLAIM["min_words"]
    )

    return {
        "category": category,
        "n_users_total": n_total,
        "n_reviews_dist": {
            "min": min(n_rev_dist) if n_rev_dist else None,
            "max": max(n_rev_dist) if n_rev_dist else None,
            "mean": statistics.mean(n_rev_dist) if n_rev_dist else None,
            "median": statistics.median(n_rev_dist) if n_rev_dist else None,
        },
        "mean_words_dist": {
            "min": min(mean_words_dist) if mean_words_dist else None,
            "max": max(mean_words_dist) if mean_words_dist else None,
            "mean": statistics.mean(mean_words_dist) if mean_words_dist else None,
            "median": statistics.median(mean_words_dist) if mean_words_dist else None,
        },
        "paper_claim_threshold": dict(PAPER_CLAIM),
        "paper_claim_n_surviving": paper_match["n_surviving"],
        "paper_claim_frac_surviving": paper_match["frac_surviving"],
        "sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES)
    args = parser.parse_args()
    targets = [args.category] if args.category else CATEGORIES

    results = []
    for category in targets:
        try:
            result = _ablation_for_category(category)
        except FileNotFoundError as e:
            print(f"  SKIP {category}: {e}")
            continue
        results.append(result)

        print(f"\n=== {category} ===")
        print(f"  n_users_total: {result['n_users_total']}")
        nr = result["n_reviews_dist"]
        mw = result["mean_words_dist"]
        print(f"  n_reviews per user: min={nr['min']}, max={nr['max']}, "
              f"mean={nr['mean']:.2f}, median={nr['median']:.1f}")
        print(f"  mean_words per user: min={mw['min']:.1f}, max={mw['max']:.1f}, "
              f"mean={mw['mean']:.2f}, median={mw['median']:.2f}")
        print(f"  paper claim (>=20 reviews, >=15 words): "
              f"{result['paper_claim_n_surviving']} users survive "
              f"({result['paper_claim_frac_surviving']*100:.1f}%)")

        print(f"\n  2-D sweep (n_surviving / frac):")
        print(f"  {'':>14} " + " ".join(f"min_words={mw:>3}" for mw in MIN_WORDS_GRID))
        for mr in MIN_REVIEWS_GRID:
            row = [f"  min_reviews={mr:>2} "]
            for mw in MIN_WORDS_GRID:
                cell = next(s for s in result["sweep"]
                            if s["min_reviews"] == mr and s["min_words"] == mw)
                row.append(f" {cell['n_surviving']:>5} ({cell['frac_surviving']*100:>5.1f}%)")
            print("".join(row))

    if not results:
        print("\nNo results (data lineage gap — see iter #86/96).")
        return

    print("\n=== Cross-domain paper-claim survival ===")
    print(f"  {'Category':<26} {'@paper(20,15)':>15} {'frac':>8}")
    for r in results:
        print(f"  {r['category']:<26} {r['paper_claim_n_surviving']:>15} "
              f"{r['paper_claim_frac_surviving']*100:>7.1f}%")

    out_dir = RESULT_ROOT / "00_data_preparation" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_filter_threshold_2d_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")
    print("\nNOTE: paper §2.1 says ≥20 reviews + ≥15 words. Stage 1 code uses "
          "MIN_LONG_SENTENCES=10 (long-sentence count), not review-count.")
    print("If paper_claim_n_surviving << n_users_total, paper claim would shrink "
          "dataset by >80% — confirm whether Stage 0 already enforces this.")


if __name__ == "__main__":
    main()