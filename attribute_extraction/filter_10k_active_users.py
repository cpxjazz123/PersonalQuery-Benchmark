#!/usr/bin/env python3
"""Stage 0: Filter 10K most-active users from Baby_Products_2023.jsonl.gz.

For each user with ≥ MIN_USER_REVIEWS total reviews (across all products),
pick their most-reviewed asin as the "primary asin", bundle up to
MAX_REVIEWS_PER_PAIR of their reviews into that asin entry.

Reviews are NOT required to match the asin — we just need the user's
writing style, which is consistent across products (same person).

Output (matches stage1_filtered_users_reviews_3000u.json format):
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage1_filtered_users_reviews_10000u.json
    [{user_id, asin, reviews: [{target_reviews: [str]}, ...], review_count}, ...]

CLAUDE.md rules: hardcoded paths, no argparse; scratch2 for outputs; pq_env.
"""
from __future__ import annotations
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SRC = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage1_filtered_users_reviews_10000u.json")

TARGET_USERS = 10_000
MIN_USER_REVIEWS = 15            # user must have ≥ 15 total reviews (≥15 gives 10,566 candidates)
MIN_REVIEW_TEXT_CHARS = 50       # skip ultra-short reviews
MAX_REVIEWS_PER_USER = 30        # bundle at most 30 reviews per user


def main() -> int:
    t0 = time.time()
    print(f"[stage0] reading {SRC}", flush=True)
    user_count: Counter[str] = Counter()                    # uid → total review count
    user_asin: dict[str, Counter[str]] = defaultdict(Counter)  # uid → asin → count
    review_text: dict[tuple[str, str], list[str]] = {}

    n_lines = 0
    n_skipped_short = 0
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            uid = d.get("user_id")
            asin = d.get("asin")
            text = (d.get("text") or "").strip()
            if not uid or not asin or len(text) < MIN_REVIEW_TEXT_CHARS:
                n_skipped_short += 1
                continue
            user_count[uid] += 1
            user_asin[uid][asin] += 1
            review_text.setdefault((uid, asin), []).append(text)
            if n_lines % 1_000_000 == 0:
                print(f"  ... {n_lines/1e6:.2f}M lines, "
                      f"{len(user_count):,} unique users, "
                      f"{time.time()-t0:.1f}s", flush=True)

    print(f"[stage0] scanned {n_lines:,} reviews, "
          f"{len(user_count):,} unique users, "
          f"{n_skipped_short:,} skipped (short/no id)", flush=True)

    # Filter: users with ≥ MIN_USER_REVIEWS total reviews
    qualified = [(u, c) for u, c in user_count.most_common() if c >= MIN_USER_REVIEWS]
    print(f"[stage0] users with ≥ {MIN_USER_REVIEWS} total reviews: "
          f"{len(qualified):,}", flush=True)
    selected = qualified[:TARGET_USERS]
    print(f"[stage0] selected {len(selected):,} top users "
          f"(review-count range: {selected[-1][1]}–{selected[0][1]})", flush=True)

    # For each user, pick their most-reviewed asin, gather up to MAX_REVIEWS_PER_USER reviews
    # across ALL their products (style is consistent across products for the same user)
    out_items = []
    n_total_reviews = 0
    for uid, total_count in selected:
        asin_counter = user_asin[uid]
        top_asin, _ = asin_counter.most_common(1)[0]
        # Gather all their reviews (across asins), sorted by asin count desc (best asin first)
        all_texts: list[str] = []
        for asin, _ in asin_counter.most_common():
            all_texts.extend(review_text.get((uid, asin), []))
            if len(all_texts) >= MAX_REVIEWS_PER_USER:
                break
        kept_texts = all_texts[:MAX_REVIEWS_PER_USER]
        reviews = [{"target_reviews": [t]} for t in kept_texts]
        # alt_asins: backup asins (top-50), used by build_query_records when
        # primary asin isn't in product_attributes.json
        alt_asins = [a for a, _ in asin_counter.most_common(50) if a != top_asin]
        out_items.append({
            "user_id": uid,
            "asin": top_asin,
            "alt_asins": alt_asins,
            "reviews": reviews,
            "review_count": len(reviews),
        })
        n_total_reviews += len(reviews)

    print(f"[stage0] wrote {len(out_items):,} (uid, asin) records, "
          f"{n_total_reviews:,} total reviews", flush=True)

    # Atomic save: tmp + rename
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out_items, fh, ensure_ascii=False)
    os.replace(tmp, OUT)
    print(f"[stage0] saved → {OUT}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"[stage0] size = {OUT.stat().st_size/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())