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
  result/query_records.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# 硬编码输入/输出路径 (Rule 3)
# 默认 10K 路径 (Stage 0 输出); 也可用 BUILD_STAGE1_REVIEWS env-var 覆盖到 3000u baseline
STAGE1_REVIEWS = Path(
    __import__("os").environ.get(
        "BUILD_STAGE1_REVIEWS",
        "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage1_filtered_users_reviews_10000u.json"
    )
)
PRODUCT_ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"
OUT_RECORDS = REPO_ROOT / "result/query_records_10k.json"

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
MAX_ATTRS = 5   # 用户指令 2026-08-22: 统一使用 5 个属性词
MAX_ATTR_VALUE_LEN = 100
MAX_RECORDS = int(__import__("os").environ.get("BUILD_QUERY_RECORDS_MAX", "10000"))

# 用户指令 2026-08-22: 排除带数字的属性值 (避免 Item Weight / Item model
# number "BWL001" / Price / Date First Available 出现在 query 里)
EXCLUDE_NUMERIC_ATTRS = True
_NUMERIC_KEYWORDS = {"price", "average rating", "rating number", "item weight",
                     "item model number", "date first available",
                     "package dimensions", "product dimensions",
                     "minimum weight recommendation",
                     "maximum weight recommendation",
                     "batteries required", "is discontinued by manufacturer"}


def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS,
                     priority: list[str] = ATTR_PRIORITY,
                     max_val_len: int = MAX_ATTR_VALUE_LEN,
                     exclude_numeric: bool = EXCLUDE_NUMERIC_ATTRS) -> dict:
    """与 copy_aware_generate.py::select_top_attrs 一致的字段选择逻辑。

    exclude_numeric=True 时跳过: 1) 含数字的 value, 2) 字段名本身就是
    数值类的 keyword (price/weight/model number/dimensions/date/batteries/
    is discontinued/rating)。
    """
    def _skip(k: str, s: str) -> bool:
        if not s or len(s) > max_val_len:
            return True
        if exclude_numeric:
            k_low = k.lower()
            if any(nk in k_low for nk in _NUMERIC_KEYWORDS):
                return True
            if has_digit(s):
                return True
        return False

    out: dict = {}
    used: set[str] = set()
    for k in priority:
        v = asin_attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if _skip(k, s):
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
        if _skip(k, s):
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
        primary_asin = rec.get("asin")
        if not uid or not primary_asin:
            continue
        # Try primary asin first, then fall back to alt_asins (if provided by Stage 0)
        candidate_asins = [primary_asin] + list(rec.get("alt_asins", []))
        attrs = None
        chosen_asin = None
        for asin in candidate_asins:
            asin_attrs = product_attrs.get(asin)
            if not asin_attrs:
                continue
            a = select_top_attrs(asin_attrs)
            if a:
                attrs = a
                chosen_asin = asin
                n_product_attrs = len(asin_attrs)
                break
        if attrs is None:
            n_no_product_attrs += 1
            continue
        n_with_attrs += 1
        records.append({
            "user_id": uid,
            "asin": chosen_asin,
            "attrs_used": attrs,
            "n_attrs": len(attrs),
            "n_product_attrs": n_product_attrs,
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