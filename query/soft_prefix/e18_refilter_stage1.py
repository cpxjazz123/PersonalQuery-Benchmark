#!/usr/bin/env python3
"""E18 Stage1 Re-filter：用 Amazon Reviews 2023 details 5 字段过滤 stage1 商品。

输入：
  - /tmp/meta_Baby_Products_2023.jsonl (217724 Baby products)
  - stage1_filtered_users_reviews.json (7681 users)

输出：
  - result/personal_query/01_preference_extraction/Baby_Products/stage1_e18_clean.json
    与 stage1 同样 schema，但每个 user 的 results 只保留
    "在 2023 details 里有 Brand+Item Weight+Product Dimensions+Color+Material 全字段" 的 product

每个保留 product 注入 attrs_5 字段（5 字段 dict），供后续 build_attrs 直接读取。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
STAGE1_DIR = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products"
STAGE1_IN = STAGE1_DIR / "stage1_filtered_users_reviews.json"
STAGE1_OUT = STAGE1_DIR / "stage1_e18_clean.json"
META_2023 = Path("/tmp/meta_Baby_Products_2023.jsonl")

ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]


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
    print(f"[load] scanning 2023 meta...")
    asin_meta = {}
    with open(META_2023) as f:
        for line in f:
            d = json.loads(line)
            pa = d.get("parent_asin")
            if pa:
                asin_meta[pa] = d
    print(f"[load] 2023 asins: {len(asin_meta)}")

    print(f"[load] stage1...")
    s1 = json.load(open(STAGE1_IN))
    n_users_in = len(s1["users"])
    print(f"[load] stage1 users in: {n_users_in}")

    n_products_total = 0
    n_products_kept = 0
    users_out = []
    n_users_kept_any = 0
    n_users_kept_3plus = 0

    for u in s1["users"]:
        uid = u["user_id"]
        ts = u.get("timestamp")
        total = u.get("total_products")
        kept = []
        for r in u.get("results", []):
            n_products_total += 1
            asin = r.get("asin")
            if not asin:
                continue
            m = asin_meta.get(asin)
            if not m or not has_clean_attrs(m):
                continue
            r2 = dict(r)
            r2["attrs_5"] = extract_attrs(m)
            kept.append(r2)
            n_products_kept += 1

        if len(kept) >= 1:
            n_users_kept_any += 1
        if len(kept) >= 3:
            n_users_kept_3plus += 1

        users_out.append({
            "user_id": uid,
            "timestamp": ts,
            "total_products": total,
            "results": kept,
            "n_kept": len(kept),
        })

    out = {
        "version": "e18-stage1-clean-v1",
        "category": "Baby_Products",
        "attr_fields": ATTR_FIELDS,
        "meta_source": "Amazon-Reviews-2023 (McAuley-Lab)",
        "filter_rule": "keep only products with all 5 details fields present and non-empty",
        "stats": {
            "users_in": n_users_in,
            "users_with_≥1_kept": n_users_kept_any,
            "users_with_≥3_kept": n_users_kept_3plus,
            "products_in": n_products_total,
            "products_kept": n_products_kept,
            "kept_rate": round(100 * n_products_kept / max(n_products_total, 1), 2),
        },
        "users": users_out,
    }
    json.dump(out, open(STAGE1_OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n[result]")
    for k, v in out["stats"].items():
        print(f"  {k}: {v}")
    print(f"[saved] {STAGE1_OUT} ({STAGE1_OUT.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
