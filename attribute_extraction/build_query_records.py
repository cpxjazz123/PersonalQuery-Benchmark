#!/usr/bin/env python3
"""Build query evaluation records from stage1 (user, asin) pairs.

合并 stage1_filtered_users_reviews_3000u.json 的 (user_id, asin) 关联和
product_attributes.json (802 字段版) 的结构化属性, 生成
copy_aware_generate.py / copy_aware_train.py 需要的 records json:

    [
      {"user_id": ..., "asin": ..., "attrs_used": {Brand: ..., Color: ..., ...},
       "n_attrs": K, "n_product_attrs": N},
      ...
    ]

字段选择策略 (复用 copy_aware_generate.py 的 ATTR_PRIORITY + select_top_attrs):
- 优先级字段优先 (Brand / Color / Material / Item Weight / Size / Style ...)
- 跳过长值 (Care instructions 等非结构化说明)
- 上限 MAX_ATTRS (默认 8, 避免 prompt 过长)

产出 (硬编码, Rule 3):
  result/personal_query/attribute_extraction/Baby_Products/query_records.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# 硬编码输入/输出路径 (Rule 3)
STAGE1_REVIEWS = REPO_ROOT / "result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"
PRODUCT_ATTRS_JSON = REPO_ROOT / "result/personal_query/attribute_extraction/Baby_Products/product_attributes.json"
OUT_RECORDS = REPO_ROOT / "result/personal_query/attribute_extraction/Baby_Products/query_records.json"

# === 属性选择参数 (与 copy_aware_generate.py 保持一致) ===
ATTR_PRIORITY = [
    "Brand", "Main Category", "Item model number", "Manufacturer",
    "Color", "Material", "Material Type", "Fabric Type", "Frame Material",
    "Item Weight", "Product Dimensions", "Size", "Style", "Pattern", "Theme",
    "Age Range (Description)", "Special Feature", "Target gender",
    "Batteries required", "Number Of Items", "Is Discontinued By Manufacturer",
    "Date First Available", "Country/Region of origin", "Country of Origin",
    "Price", "Average Rating", "Rating Number",
]
MAX_ATTRS = 8
MAX_ATTR_VALUE_LEN = 100

# 评测规模上限 (避免一次性产出过大 record)
MAX_RECORDS = int(__import__("os").environ.get("BUILD_QUERY_RECORDS_MAX", "500"))


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS,
                     priority: list[str] = ATTR_PRIORITY,
                     max_val_len: int = MAX_ATTR_VALUE_LEN) -> dict:
    """与 copy_aware_generate.py::select_top_attrs 一致的字段选择逻辑。"""
    out: dict = {}
    used: set[str] = set()
    for k in priority:
        v = asin_attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if not s or len(s) > max_val_len:
            continue
        out[k] = s
        used.add(k)
        if len(out) >= max_n:
            return out
    for k, v in asin_attrs.items():
        if k in used:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s or len(s) > max_val_len:
            continue
        out[k] = s
        if len(out) >= max_n:
            break
    return out


def main() -> None:
    print(f"[build] loading stage1 reviews: {STAGE1_REVIEWS}")
    reviews = json.load(open(STAGE1_REVIEWS, "r", encoding="utf-8"))
    print(f"[build]   {len(reviews)} user-asin pairs")

    print(f"[build] loading product_attrs: {PRODUCT_ATTRS_JSON}")
    product_attrs = json.load(open(PRODUCT_ATTRS_JSON, "r", encoding="utf-8"))
    print(f"[build]   {len(product_attrs)} products")

    records: list[dict] = []
    n_with_attrs = 0
    n_no_product_attrs = 0
    for rec in reviews:
        uid = rec.get("user_id")
        asin = rec.get("asin")
        if not uid or not asin:
            continue
        asin_attrs = product_attrs.get(asin)
        if not asin_attrs:
            n_no_product_attrs += 1
            continue
        attrs = select_top_attrs(asin_attrs)
        if not attrs:
            n_no_product_attrs += 1
            continue
        n_with_attrs += 1
        records.append({
            "user_id": uid,
            "asin": asin,
            "attrs_used": attrs,
            "n_attrs": len(attrs),
            "n_product_attrs": len(asin_attrs),
        })
        if len(records) >= MAX_RECORDS:
            break

    OUT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RECORDS, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[build] ✓ wrote {len(records)} records -> {OUT_RECORDS}")
    print(f"[build]   with attrs: {n_with_attrs}, skipped (no product attrs): {n_no_product_attrs}")
    if records:
        sample = records[0]
        print(f"[build]   sample: asin={sample['asin']} "
              f"({sample['n_attrs']}/{sample['n_product_attrs']} attrs)")
        for k, v in sample["attrs_used"].items():
            print(f"             {k}: {v}")


if __name__ == "__main__":
    main()