#!/usr/bin/env python3
"""Smoke test for iter #185 user filter threshold 2-D sweep.

Tests the sweep logic on synthetic data:
  - 100 users with n_reviews ~ U(5, 30) and mean_words ~ U(8, 25).
  - Verify sweep[paper_claim (20, 15)] count is plausible (< total).
  - Verify monotonicity: more restrictive threshold ⇒ fewer survivors.

Tests by monkey-patching REVIEW_FILE_TEMPLATE to a tmp dir.
"""

import sys
import json
import tempfile
from pathlib import Path

_SCRIPT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/00_data_preparation/ablation_user_filter_threshold_2d.py")
import importlib.util
spec = importlib.util.spec_from_file_location("abl185", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    print("=" * 60)
    print("iter #185 user filter threshold 2-D smoke test")
    print("=" * 60)

    # Synthetic data: 100 users, mixed n_reviews + mean_words
    users = []
    for i in range(100):
        n_rev = 5 + (i * 7 % 26)  # 5..30 range
        # Make reviews text with varying length
        reviews = [
            " ".join(["word"] * (10 + (i + j) % 16)) for j in range(n_rev)
        ]
        users.append({
            "user_id": f"u{i}",
            "results": [{"target_reviews": reviews}],
        })
    synth = {"users": users}

    with tempfile.TemporaryDirectory() as tmp:
        result_root = Path(tmp)
        # Create matching subdirs
        for cat in mod.CATEGORIES:
            d = result_root / "01_preference_extraction" / cat
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "stage1_filtered_users_reviews.json", "w") as f:
                json.dump(synth, f)

        # Monkey-patch the paths
        orig_root = mod.RESULT_ROOT
        mod.RESULT_ROOT = result_root
        mod.REVIEW_FILE_TEMPLATE = str(
            result_root / "01_preference_extraction" / "{cat}" / "stage1_filtered_users_reviews.json"
        )
        try:
            result = mod._ablation_for_category("Baby_Products")
        finally:
            mod.RESULT_ROOT = orig_root

    print(f"\n  n_users_total: {result['n_users_total']}")
    assert result["n_users_total"] == 100
    nr = result["n_reviews_dist"]
    mw = result["mean_words_dist"]
    print(f"  n_reviews dist: min={nr['min']}, max={nr['max']}, mean={nr['mean']:.1f}")
    print(f"  mean_words dist: min={mw['min']:.1f}, max={mw['max']:.1f}, mean={mw['mean']:.1f}")
    assert 5 <= nr["min"] and nr["max"] <= 30
    assert 10 <= mw["min"] and mw["max"] <= 25

    # Paper claim check
    pc = result["paper_claim_n_surviving"]
    print(f"\n  paper claim (>=20 reviews, >=15 words): {pc} survive "
          f"({result['paper_claim_frac_surviving']*100:.1f}%)")
    assert pc < 100, "paper claim should filter out some users"
    assert pc > 0, "should have at least some survivors in synthetic"
    assert pc < result["n_users_total"]

    # Monotonicity: more restrictive threshold (HIGHER min_reviews) ⇒ fewer survivors
    # Iterate from smallest to largest min_reviews: count must be NON-INCREASING
    print(f"\n  Monotonicity check (min_reviews axis, min_words=15 fixed):")
    prev = 101
    for mr in sorted(mod.MIN_REVIEWS_GRID):
        cell = next(s for s in result["sweep"]
                    if s["min_reviews"] == mr and s["min_words"] == 15)
        print(f"    min_reviews={mr:>2}: {cell['n_surviving']:>3}")
        assert cell["n_surviving"] <= prev, \
            f"monotonicity violation at min_reviews={mr}: {cell['n_surviving']} > {prev}"
        prev = cell["n_surviving"]
    print("  ✓ monotonicity holds on min_reviews axis")

    print(f"\n  Monotonicity check (min_words axis, min_reviews=20 fixed):")
    prev = 101
    for mw in sorted(mod.MIN_WORDS_GRID):
        cell = next(s for s in result["sweep"]
                    if s["min_reviews"] == 20 and s["min_words"] == mw)
        print(f"    min_words={mw:>2}: {cell['n_surviving']:>3}")
        assert cell["n_surviving"] <= prev, \
            f"monotonicity violation at min_words={mw}: {cell['n_surviving']} > {prev}"
        prev = cell["n_surviving"]
    print("  ✓ monotonicity holds on min_words axis")

    print("\n" + "=" * 60)
    print("ALL CASES PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()