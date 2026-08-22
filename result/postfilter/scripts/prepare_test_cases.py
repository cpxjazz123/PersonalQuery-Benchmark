#!/usr/bin/env python3
"""为 generate-then-filter 实验准备 (user, asin, 商品属性) 测试用例.

从 cluster_smoke 用的 300 用户里选 30 个，每个用户挑 1 个他评论过的 asin，
join meta_Baby_Products_2023.jsonl.gz 拿商品 title/categories，作为 prompt
输入。Ground truth 风格：用户对该 asin 的真实评论（多句）。

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/test_cases.jsonl
  每行: {user_id, asin, title, categories, gt_reviews: [text...]}
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
DATA_REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
DATA_META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
USER_PROFILE = REPO_ROOT / "result/personal_query/12_complexity_analysis_clause_features/Baby_Products/cluster_smoke_user_profiles.jsonl"
OUT_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter/test_cases.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
N_USERS = 30
MIN_GT_REVIEWS = 1
RANDOM_SEED = 42


def main() -> None:
    log = lambda msg: print(f"[prepare] {msg}", flush=True)
    rng = random.Random(RANDOM_SEED)
    log(f"读用户 profiles: {USER_PROFILE}")
    target_users: list[str] = []
    with USER_PROFILE.open() as f:
        for line in f:
            target_users.append(json.loads(line)["user_id"])
    rng.shuffle(target_users)
    target_users = target_users[:N_USERS]
    log(f"候选用户 {len(target_users)}")

    log(f"扫 reviews: {DATA_REVIEWS}")
    user_to_reviews: dict[str, list[dict]] = {}
    with gzip.open(DATA_REVIEWS, "rt") as f:
        for line in f:
            row = json.loads(line)
            uid = row.get("user_id")
            if uid in target_users and row.get("text"):
                user_to_reviews.setdefault(uid, []).append(row)
    log(f"用户评论覆盖: {len(user_to_reviews)}/{N_USERS}")

    log(f"读 meta: {DATA_META}")
    asin_to_meta: dict[str, dict] = {}
    with gzip.open(DATA_META, "rt") as f:
        for line in f:
            row = json.loads(line)
            asin = row.get("parent_asin")
            if asin:
                asin_to_meta[asin] = {
                    "title": row.get("title", ""),
                    "store": row.get("store", ""),
                    "categories": row.get("categories", []),
                    "main_category": row.get("main_category", ""),
                    "features": row.get("features", []) or [],
                    "average_rating": row.get("average_rating"),
                }
    log(f"商品元数据覆盖: {len(asin_to_meta)}")

    test_cases = []
    skipped = 0
    for uid in target_users:
        reviews = user_to_reviews.get(uid, [])
        if len(reviews) < MIN_GT_REVIEWS:
            skipped += 1
            continue
        asin_groups: dict[str, list[str]] = {}
        for r in reviews:
            a = r.get("parent_asin")
            t = (r.get("text") or "").strip()
            if a and t:
                asin_groups.setdefault(a, []).append(t)
        candidate_asins = [a for a, rs in asin_groups.items() if len(rs) >= MIN_GT_REVIEWS and a in asin_to_meta]
        if not candidate_asins:
            skipped += 1
            continue
        chosen_asin = rng.choice(candidate_asins)
        meta = asin_to_meta[chosen_asin]
        gt_texts = [t[:300] for t in asin_groups[chosen_asin][:5]]
        if len(gt_texts) == 0:
            skipped += 1
            continue
        test_cases.append({
            "user_id": uid,
            "asin": chosen_asin,
            "title": meta["title"],
            "categories": meta["categories"][:3] if meta["categories"] else [],
            "store": meta["store"],
            "main_category": meta.get("main_category", ""),
            "description": " ".join(meta.get("features", []) or [])[:200],
            "gt_reviews": gt_texts,
        })

    log(f"完成: test_cases={len(test_cases)}, skipped={skipped}")
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in test_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
