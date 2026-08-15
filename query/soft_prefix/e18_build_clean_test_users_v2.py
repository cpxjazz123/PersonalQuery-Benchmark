#!/usr/bin/env python3
"""E18 Task 2c: 基于新 4692 style vector pool 重建 test users。

改动 vs e18_build_clean_test_users.py：
  - 起点池：e18_style_vectors.json 4692 user（vs e17 的 400）
  - 已固定 train/dev/test split（n_train=3866, n_dev=50, n_test=776）
  - 直接使用 test_users 名单（776 user）
  - 过滤条件：每个 test user 在 2023 meta 里有 ≥3 个 clean product
  - 目标：≥100 test users × ≥3 clean product

输出：
  - e18_test_users_v2.json：list of test user_ids
  - e18_clean_pairs_v2.jsonl：每个 (user_id, asin, attrs) pair
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E18 = REPO_ROOT / "result" / "personal_query" / "e18"
STAGE1_E18 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_e18_clean.json"
META_2023 = Path("/tmp/meta_Baby_Products_2023.jsonl")

ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
MIN_CLEAN_PER_USER = 3


def has_clean_attrs(meta: dict) -> bool:
    details = meta.get("details")
    if not details:
        return False
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            return False
    if not isinstance(details, dict):
        return False
    for k in ATTR_FIELDS:
        v = details.get(k)
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
    return True


def extract_attrs(meta: dict) -> dict:
    details = meta.get("details")
    if isinstance(details, str):
        details = json.loads(details)
    return {k: str(details[k]).strip()[:64] for k in ATTR_FIELDS}


def main() -> None:
    # 1) 加载 e18 test user 候选池
    sv = json.load(open(E18 / "e18_style_vectors.json"))
    test_users = sorted(sv["splits"]["test"])
    print(f"[pool] e18 test users: {len(test_users)}", flush=True)

    # 2) 加载 stage1_e18_clean（已按 5 字段过滤 products）
    s1 = json.load(open(STAGE1_E18))
    user_to_clean_asins = {}
    for u in s1["users"]:
        uid = u["user_id"]
        if uid in test_users:
            user_to_clean_asins[uid] = [r["asin"] for r in u.get("results", [])]
    print(f"[stage1] e18 test users with stage1 products: {len(user_to_clean_asins)}", flush=True)

    # 3) 验证：stage1_e18_clean 里的 products 都已通过 5 字段筛选（无需再扫 meta）
    clean_pairs = []
    user_clean_count = {}
    for uid in test_users:
        asins = user_to_clean_asins.get(uid, [])
        if len(asins) >= MIN_CLEAN_PER_USER:
            user_clean_count[uid] = len(asins)
            # 把 attrs_5 字段也带出来
            asin_to_attrs = {}
            for u in s1["users"]:
                if u["user_id"] == uid:
                    for r in u.get("results", []):
                        if "attrs_5" in r:
                            asin_to_attrs[r["asin"]] = r["attrs_5"]
                    break
            for a in asins:
                if a in asin_to_attrs:
                    clean_pairs.append({
                        "user_id": uid, "asin": a, "attrs": asin_to_attrs[a],
                    })

    final_test_users = sorted(user_clean_count.keys())
    print(f"\n[result] test users with ≥{MIN_CLEAN_PER_USER} clean products: {len(final_test_users)}", flush=True)
    print(f"[result] total clean pairs: {len(clean_pairs)}", flush=True)
    if user_clean_count:
        vals = sorted(user_clean_count.values())
        print(f"[result] clean products per user: min={vals[0]}, max={vals[-1]}, median={vals[len(vals)//2]}", flush=True)

    # 4) 保存
    json.dump({
        "version": "e18-clean-test-users-v2",
        "pool_source": "e18_style_vectors.json",
        "n_candidates": len(test_users),
        "n_test_users": len(final_test_users),
        "min_clean_per_user": MIN_CLEAN_PER_USER,
        "attr_fields": ATTR_FIELDS,
        "test_users": final_test_users,
        "user_clean_count": user_clean_count,
    }, open(E18 / "e18_test_users_v2.json", "w"), indent=2)
    print(f"[saved] {E18 / 'e18_test_users_v2.json'}", flush=True)

    with open(E18 / "e18_clean_pairs_v2.jsonl", "w") as f:
        for p in clean_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[saved] {E18 / 'e18_clean_pairs_v2.jsonl'} ({len(clean_pairs)} pairs)", flush=True)


if __name__ == "__main__":
    main()
