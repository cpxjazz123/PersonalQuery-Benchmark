#!/usr/bin/env python3
"""[Reviewer pilot] Review writing style vs query behavior correlation.

Addresses the [Major] reviewer concern from iter #38:
  "Review writing style ≠ Query behavior assumption is unverified — the entire
   pipeline assumes user review-writing complexity correlates with their
   syntactic query complexity, but this is not empirically checked."

This pilot:
  1. Loads Stage 1 review-style index (avg review word count per user).
  2. Loads Stage 6 expression_style query file (per-user query behavior).
  3. Joins on user_id, aggregates queries per user (mean word_count, mean
     user_avg_depth, mean target_depth).
  4. Computes Pearson + Spearman correlation between review-style (X) and
     each query-behavior signal (Y).
  5. Outputs per-domain correlation table.

Output: prints to stdout, saves JSON to result/personal_query/<pilot>/.

Usage:
    python3 02_writing_analysis/pilot_review_vs_query_correlation.py
    # or per-domain:
    python3 02_writing_analysis/pilot_review_vs_query_correlation.py --category Baby_Products

NOTE: This is a deterministic, non-LLM pilot — runs in <1 minute per domain.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"

REVIEW_FILE_TEMPLATE = str(RESULT_ROOT / "01_preference_extraction" / "{cat}" / "stage1_filtered_users_reviews.json")
QUERY_FILE_TEMPLATE = str(RESULT_ROOT / "06_query" / "{cat}" / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json")

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation without scipy, by hand-rolled ranking."""
    if len(xs) < 2:
        return float("nan")

    def _rank(values: list[float]) -> list[float]:
        sorted_unique = sorted(set(values))
        rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
        ranks = [rank_map[v] for v in values]
        # Average ranks for ties
        return ranks

    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Kendall's tau-b rank correlation without scipy.

    τ_b = (n_concordant - n_discordant) / sqrt((n0 - n1) * (n0 - n2))
    where n0 = n(n-1)/2, n1 = sum ties_x * (ties_x - 1) / 2,
          n2 = sum ties_y * (ties_y - 1) / 2.
    """
    n = len(xs)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue  # ties on either → neither concordant nor discordant
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    # tie counts
    def _n_ties(values: list[float]) -> int:
        from collections import Counter
        c = Counter(values)
        return sum(v * (v - 1) // 2 for v in c.values())
    n0 = n * (n - 1) // 2
    n1 = _n_ties(xs)
    n2 = _n_ties(ys)
    denom_sq = (n0 - n1) * (n0 - n2)
    if denom_sq <= 0:
        return float("nan")
    return (concordant - discordant) / (denom_sq ** 0.5)


def _bootstrap_ci(xs: list[float], ys: list[float], fn, n_boot: int = 1000,
                  ci: float = 0.95, rng_seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap percentile CI for any correlation fn (pearson/spearman/kendall).

    Returns (point_estimate, ci_low, ci_high).
    """
    import random
    point = fn(xs, ys)
    if point != point:  # NaN
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(rng_seed)
    n = len(xs)
    boot_stats: list[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idxs]
        by = [ys[i] for i in idxs]
        b = fn(bx, by)
        if b == b:  # not NaN
            boot_stats.append(b)
    if not boot_stats:
        return point, float("nan"), float("nan")
    boot_stats.sort()
    alpha = 1.0 - ci
    lo_idx = int(round(alpha / 2 * len(boot_stats)))
    hi_idx = int(round((1.0 - alpha / 2) * len(boot_stats))) - 1
    lo_idx = max(0, min(lo_idx, len(boot_stats) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_stats) - 1))
    return point, boot_stats[lo_idx], boot_stats[hi_idx]


def _permutation_test(xs: list[float], ys: list[float], fn,
                      n_perm: int = 1000, rng_seed: int = 42) -> float:
    """Two-sided permutation test p-value for H0: corr(X, Y) = 0.

    Returns p-value: fraction of permuted correlations at least as extreme
    as the observed correlation.
    """
    import random
    obs = fn(xs, ys)
    if obs != obs:
        return float("nan")
    rng = random.Random(rng_seed)
    n = len(xs)
    ys_perm = list(ys)
    extreme_count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_perm)
        perm = fn(xs, ys_perm)
        if perm != perm:
            continue
        if abs(perm) >= abs(obs):
            extreme_count += 1
    return extreme_count / n_perm


def _build_review_style_index(review_path: Path) -> dict[str, dict]:
    """Returns {user_id: {'mean_words': float, 'n_reviews': int, 'mean_lens_std': float}}."""
    if not review_path.exists():
        raise FileNotFoundError(f"Review file missing: {review_path}")
    with open(review_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    index: dict[str, dict] = {}
    for user in d.get("users", []):
        uid = user.get("user_id")
        if not uid:
            continue
        review_word_counts = []
        for result in user.get("results", []):
            for review in result.get("target_reviews", []):
                if isinstance(review, str):
                    review_word_counts.append(len(review.split()))
        if not review_word_counts:
            continue
        index[uid] = {
            "mean_words": float(statistics.mean(review_word_counts)),
            "n_reviews": int(len(review_word_counts)),
            "std_words": float(statistics.pstdev(review_word_counts)) if len(review_word_counts) > 1 else 0.0,
        }
    return index


def _build_query_behavior_index(query_path: Path) -> dict[str, dict]:
    """Returns {user_id: {'mean_query_words': float, 'mean_user_avg_depth': float, 'mean_target_depth': float, 'n_queries': int}}."""
    if not query_path.exists():
        raise FileNotFoundError(f"Query file missing: {query_path}")
    with open(query_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    per_user = defaultdict(list)
    for row in rows:
        uid = row.get("user_id")
        if not uid:
            continue
        sq = row.get("syntax_depth_query", row.get("expression_style_query", {}))
        if not sq:
            continue
        per_user[uid].append(
            {
                "query_words": sq.get("word_count", sq.get("query_word_count")),
                "user_avg_depth": sq.get("user_avg_depth", sq.get("avg_depth")),
                "target_depth": sq.get("target_depth", sq.get("target_depth_value")),
            }
        )
    index: dict[str, dict] = {}
    for uid, queries in per_user.items():
        words = [q["query_words"] for q in queries if isinstance(q["query_words"], (int, float))]
        avgs = [q["user_avg_depth"] for q in queries if isinstance(q["user_avg_depth"], (int, float))]
        targets = [q["target_depth"] for q in queries if isinstance(q["target_depth"], (int, float))]
        if not (words and avgs):
            continue
        index[uid] = {
            "mean_query_words": float(statistics.mean(words)),
            "mean_user_avg_depth": float(statistics.mean(avgs)),
            "mean_target_depth": float(statistics.mean(targets)) if targets else float("nan"),
            "n_queries": int(len(words)),
        }
    return index


def _correlate_for_category(category: str) -> dict:
    review_path = Path(REVIEW_FILE_TEMPLATE.format(cat=category))
    query_path = Path(QUERY_FILE_TEMPLATE.format(cat=category))
    review_idx = _build_review_style_index(review_path)
    query_idx = _build_query_behavior_index(query_path)

    user_ids = sorted(set(review_idx) & set(query_idx))
    if not user_ids:
        raise ValueError(f"{category}: no overlap between review and query indices")

    review_words = [review_idx[uid]["mean_words"] for uid in user_ids]
    query_words = [query_idx[uid]["mean_query_words"] for uid in user_ids]
    avg_depths = [query_idx[uid]["mean_user_avg_depth"] for uid in user_ids]
    target_depths = [query_idx[uid]["mean_target_depth"] for uid in user_ids]

    # iter #184: bootstrap CI + permutation test for each (X=review, Y=query signal) pair
    pairs = {
        "review_vs_query_words": (review_words, query_words),
        "review_vs_user_avg_depth": (review_words, avg_depths),
        "review_vs_target_depth": (review_words, target_depths),
    }
    correlation_metrics: dict = {}
    for pair_name, (x, y) in pairs.items():
        pe, pe_lo, pe_hi = _bootstrap_ci(x, y, _pearson)
        sp, sp_lo, sp_hi = _bootstrap_ci(x, y, _spearman)
        kt, kt_lo, kt_hi = _bootstrap_ci(x, y, _kendall_tau)
        correlation_metrics[f"pearson_{pair_name}"] = {
            "point": pe, "ci95_low": pe_lo, "ci95_high": pe_hi,
            "perm_p": _permutation_test(x, y, _pearson),
        }
        correlation_metrics[f"spearman_{pair_name}"] = {
            "point": sp, "ci95_low": sp_lo, "ci95_high": sp_hi,
            "perm_p": _permutation_test(x, y, _spearman),
        }
        correlation_metrics[f"kendall_{pair_name}"] = {
            "point": kt, "ci95_low": kt_lo, "ci95_high": kt_hi,
            "perm_p": _permutation_test(x, y, _kendall_tau),
        }

    return {
        "category": category,
        "n_users_overlap": len(user_ids),
        "n_users_review": len(review_idx),
        "n_users_query": len(query_idx),
        "review_word_stats": {
            "min": float(min(review_words)) if review_words else None,
            "max": float(max(review_words)) if review_words else None,
            "mean": float(statistics.mean(review_words)) if review_words else None,
            "median": float(statistics.median(review_words)) if review_words else None,
        },
        **correlation_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, help="Limit to one category (default: all 3)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="iter #184: bootstrap iterations for CI")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="iter #184: permutation iterations for H0 test")
    args = parser.parse_args()

    targets = [args.category] if args.category else CATEGORIES
    results = []
    for category in targets:
        print(f"\n=== {category} ===")
        try:
            result = _correlate_for_category(category)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue
        results.append(result)
        # iter #184: print nested correlation metrics
        for key, value in result.items():
            if key == "review_word_stats":
                continue
            if isinstance(value, dict):
                # correlation metric with point + CI + p-value
                pt = value.get("point", float("nan"))
                lo = value.get("ci95_low", float("nan"))
                hi = value.get("ci95_high", float("nan"))
                pp = value.get("perm_p", float("nan"))
                sig = "*" if pp < 0.05 else " "
                print(f"  {key:<40} point={pt:+.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]  "
                      f"perm_p={pp:.3f} {sig}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if not results:
        print("\nNo results to summarize (all categories skipped — data lineage gap, see iter #86).")
        return

    print("\n=== Summary table (iter #184: Pearson / Spearman / Kendall + 95% CI + perm p) ===")
    print(
        f"{'Category':<26} {'n':>6} "
        f"{'Prs(Q)':>9} {'Spr(Q)':>9} {'Kt(Q)':>9} "
        f"{'Prs(D)':>9} {'Spr(D)':>9} {'Kt(D)':>9} "
        f"{'Prs(T)':>9} {'Spr(T)':>9} {'Kt(T)':>9}"
    )
    print("-" * 130)
    for r in results:
        n = r["n_users_overlap"]
        def _pt(metric_name):
            m = r.get(metric_name, {})
            return m.get("point", float("nan")) if isinstance(m, dict) else float("nan")
        print(
            f"{r['category']:<26} {n:>6} "
            f"{_pt('pearson_review_vs_query_words'):>+9.4f} "
            f"{_pt('spearman_review_vs_query_words'):>+9.4f} "
            f"{_pt('kendall_review_vs_query_words'):>+9.4f} "
            f"{_pt('pearson_review_vs_user_avg_depth'):>+9.4f} "
            f"{_pt('spearman_review_vs_user_avg_depth'):>+9.4f} "
            f"{_pt('kendall_review_vs_user_avg_depth'):>+9.4f} "
            f"{_pt('pearson_review_vs_target_depth'):>+9.4f} "
            f"{_pt('spearman_review_vs_target_depth'):>+9.4f} "
            f"{_pt('kendall_review_vs_target_depth'):>+9.4f}"
        )
    print()
    print("Legend: Prs=pearson, Spr=spearman, Kt=Kendall tau-b; "
          "Q=query_words, D=user_avg_depth, T=target_depth")
    print("Interpretation: |r| > 0.3 = moderate; > 0.5 = strong. "
          "perm_p<0.05 → reject H0 (correlation ≠ 0).")

    out_dir = RESULT_ROOT / "02_writing_analysis" / "review_vs_query_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "review_vs_query_correlation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
