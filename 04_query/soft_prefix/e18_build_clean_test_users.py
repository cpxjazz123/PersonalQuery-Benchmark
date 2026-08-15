#!/usr/bin/env python3
"""E18 Task 1: 构建干净 test user 候选集（基于 Amazon Reviews 2023 details 5 字段）。

约束（CLAUDE.md rule 9 — 硬编码）：
  - 数据源：/tmp/meta_Baby_Products_2023.jsonl（217724 Baby products）
  - 字段源：details dict 5 个 schema 字段
    * Brand
    * Item Weight
    * Product Dimensions
    * Color
    * Material
  - 起点池：e17_style_vectors (400) − train (164 ∩ 5) − dev (41 ∩ 2) = 393 candidates
  - 筛选条件：candidate 的 stage1 products 中，至少 N 个 product 的 5 字段**全部存在且非空**
  - N = MIN_CLEAN_PER_USER（默认 3，按 issue 18 "100 user × 3 product" 目标）
  - 无 fallback：5 字段任何一个缺失或空字符串 → 该 product 不算 clean
  - 输出：
    * e18_test_users.json：list of user_id，每个 user 至少有 MIN_CLEAN_PER_USER 个 clean product
    * e18_clean_pairs.jsonl：每行 {user_id, asin, attrs={Brand, Item Weight, Product Dimensions, Color, Material}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
E18 = REPO_ROOT / "result" / "personal_query" / "e18"
STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews.json"
META_2023 = Path("/tmp/meta_Baby_Products_2023.jsonl")

# 5 个 schema 字段（Amazon Reviews 2023 details dict）
ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
MIN_CLEAN_PER_USER = 3  # issue 18 目标：≥100 user × ≥3 product

E18.mkdir(parents=True, exist_ok=True)


def has_clean_attrs(meta: dict) -> bool:
    """所有 5 个字段在 details 里存在且非空。"""
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
    """从 details 抽 5 个字段，无 fallback（不存在的字段 raise KeyError）。"""
    details = meta.get("details")
    if isinstance(details, str):
        details = json.loads(details)
    out = {}
    for k in ATTR_FIELDS:
        v = details[k]
        out[k] = str(v).strip()[:64]
    return out


def main() -> None:
    # 1. 加载 e17 candidates pool = style_vectors - train - dev
    style = json.load(open(E17 / "e17_style_vectors.json"))
    split = json.load(open(E17 / "e17_train_dev_split.json"))
    train_users = set(split["train"]["users"])
    dev_users = set(split["dev"]["users"])
    style_users = set(style["vectors"].keys())
    candidates = sorted(style_users - train_users - dev_users)
    print(f"[candidates] style={len(style_users)} train∩style={len(train_users & style_users)} "
          f"dev∩style={len(dev_users & style_users)} → candidates={len(candidates)}", flush=True)

    # 2. 收集 candidates 的 stage1 products
    s1 = json.load(open(STAGE1))
    user_to_asins = {}
    for u in s1["users"]:
        uid = u["user_id"]
        if uid not in candidates:
            continue
        user_to_asins[uid] = [r["asin"] for r in u.get("results", []) if r.get("asin")]
    print(f"[stage1] candidates with stage1 products: {len(user_to_asins)}", flush=True)

    all_asins = set()
    for v in user_to_asins.values():
        all_asins.update(v)
    print(f"[stage1] unique ASINs: {len(all_asins)}", flush=True)

    # 3. 加载 Amazon Reviews 2023 meta（只在 candidate ASINs 子集上）
    print(f"[load] scanning 2023 meta for {len(all_asins)} candidate ASINs...", flush=True)
    asin_meta = {}
    with open(META_2023) as f:
        for line in f:
            d = json.loads(line)
            pa = d.get("parent_asin")
            if pa in all_asins:
                asin_meta[pa] = d
    print(f"[load] matched ASINs in 2023 meta: {len(asin_meta)} / {len(all_asins)} "
          f"({100*len(asin_meta)/max(len(all_asins),1):.2f}%)", flush=True)

    # 4. 过滤：每 user 至少有 MIN_CLEAN_PER_USER 个 clean product
    clean_pairs = []  # list of {user_id, asin, attrs}
    user_clean_count = {}
    for uid in candidates:
        asins = user_to_asins.get(uid, [])
        clean_asins = [a for a in asins if a in asin_meta and has_clean_attrs(asin_meta[a])]
        if len(clean_asins) >= MIN_CLEAN_PER_USER:
            user_clean_count[uid] = len(clean_asins)
            # 写所有 clean pairs（不止 MIN_CLEAN_PER_USER 个，方便后续选用）
            for a in clean_asins:
                clean_pairs.append({
                    "user_id": uid,
                    "asin": a,
                    "attrs": extract_attrs(asin_meta[a]),
                })

    test_users = sorted(user_clean_count.keys())
    print(f"\n[result] test users with ≥{MIN_CLEAN_PER_USER} clean products: {len(test_users)}", flush=True)
    print(f"[result] total clean pairs: {len(clean_pairs)}", flush=True)
    print(f"[result] clean products per user: min={min(user_clean_count.values()) if user_clean_count else 0}, "
          f"max={max(user_clean_count.values()) if user_clean_count else 0}, "
          f"median={sorted(user_clean_count.values())[len(user_clean_count)//2] if user_clean_count else 0}",
          flush=True)

    # 5. 保存
    json.dump({
        "version": "e18-clean-test-users-v1",
        "n_candidates": len(candidates),
        "n_test_users": len(test_users),
        "min_clean_per_user": MIN_CLEAN_PER_USER,
        "attr_fields": ATTR_FIELDS,
        "meta_source": "Amazon-Reviews-2023 (McAuley-Lab) /tmp/meta_Baby_Products_2023.jsonl",
        "test_users": test_users,
        "user_clean_count": user_clean_count,
    }, open(E18 / "e18_test_users.json", "w"), indent=2)
    print(f"[saved] {E18 / 'e18_test_users.json'}", flush=True)

    with open(E18 / "e18_clean_pairs.jsonl", "w") as f:
        for p in clean_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[saved] {E18 / 'e18_clean_pairs.jsonl'} ({len(clean_pairs)} pairs)", flush=True)


if __name__ == "__main__":
    main()
