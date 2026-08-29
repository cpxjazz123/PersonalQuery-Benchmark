#!/usr/bin/env python3
"""Product attribute extraction from meta_Baby_Products_2023.jsonl.gz.

Reads product metadata, extracts structured attributes per ASIN (Brand,
Main Category, Color, Material, Size, ..., no numeric values), writes to
result/product_attributes.json.

Also exports select_top_attrs() utility used by gaussian/build_user.py
to pick top-N attrs per ASIN with numeric value exclusion.

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
META_GZ = DATA / "meta_Baby_Products_2023.jsonl.gz"

# === Outputs ===
PRODUCT_ATTRS_JSON = REPO_ROOT / "result" / "product_attributes.json"

# === Step 1 — extract_attrs 参数 ===
MAX_STR_LEN = 200
TOP_LEVEL_NUMERIC_FIELDS = ("average_rating", "rating_number", "price")

# === select_top_attrs 参数 (供下游 cohort construction 共用) ===
MAX_ATTRS = 5
MAX_ATTR_VALUE_LEN = 100
EXCLUDE_NUMERIC_ATTRS = True
_NUMERIC_KEYWORDS = {"price", "average rating", "rating number", "item weight",
                     "item model number", "date first available",
                     "package dimensions", "product dimensions",
                     "minimum weight recommendation",
                     "maximum weight recommendation",
                     "batteries required", "is discontinued by manufacturer"}
ATTR_PRIORITY = [
    "Brand", "Main Category", "Item model number", "Manufacturer",
    "Color", "Material", "Material Type", "Fabric Type", "Frame Material",
    "Item Weight", "Product Dimensions", "Size", "Style", "Pattern", "Theme",
    "Age Range (Description)", "Special Feature", "Target gender",
    "Batteries required", "Number Of Items", "Is Discontinued By Manufacturer",
    "Date First Available", "Country/Region of origin", "Country of Origin",
    "Price", "Average Rating", "Rating Number",
]


def log(msg: str) -> None:
    print(f"[extract_product_attrs] {msg}", flush=True)


def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def extract_attrs(d: dict) -> dict:
    """Extract structured attributes from a single product meta dict.

    Returns dict {field_name: string_value} containing all short-string
    fields from details + top-level structured fields (main_category,
    average_rating, rating_number, price). No numeric value filtering
    here — that is select_top_attrs()'s job (called downstream).
    """
    details = d.get("details")
    if not isinstance(details, dict):
        details = {}
    out: dict = {}

    # details 中的所有短字符串字段
    for k, v in details.items():
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or len(s) > MAX_STR_LEN:
            continue
        out[k] = s

    # 用户指令 2026-08-27: 删除 Brand fallback (Rule 7 禁止降级)
    # 原代码: details 无 Brand → store → title[0], 三层降级
    # 现: 仅从 details 取; Brand 缺失则 out 无 Brand 键,
    # 下游 select_top_attrs 按 ATTR_PRIORITY 跳过, 不影响其他字段

    # top-level 结构化字段
    main_cat = d.get("main_category")
    if isinstance(main_cat, str) and main_cat.strip():
        out["Main Category"] = main_cat.strip()

    for field in TOP_LEVEL_NUMERIC_FIELDS:
        v = d.get(field)
        if v is None or v == "":
            continue
        out[field.replace("_", " ").title().replace(" ", " ")] = v

    return out


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS) -> dict:
    """Select top-N attributes for an ASIN, excluding numeric values.

    Used by both Stage 1 vLLM prompt construction and downstream cohort
    selection. Returns dict {field_name: string_value}, with priority
    order ATTR_PRIORITY first, then any remaining fields.

    Numeric exclusion: any value with digits, or any key in _NUMERIC_KEYWORDS,
    is dropped. Implements 用户指令 2026-08-28: "属性数值过滤".

    Args:
        asin_attrs: dict of field_name -> string_value from extract_attrs()
        max_n: maximum number of attrs to return (Stage 1 = 5, Stage 4 cohort = 4)
    """
    def _skip(k: str, s: str) -> bool:
        if not s or len(s) > MAX_ATTR_VALUE_LEN:
            return True
        if EXCLUDE_NUMERIC_ATTRS:
            k_low = k.lower()
            if any(nk in k_low for nk in _NUMERIC_KEYWORDS):
                return True
            if has_digit(s):
                return True
        return False

    out: dict = {}
    used: set[str] = set()
    for k in ATTR_PRIORITY:
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


def step1_extract_attrs() -> dict[str, dict]:
    log("=== Step 1: extract_attrs ===")
    product_attrs: dict[str, dict] = {}
    n_total = 0
    n_with_attrs = 0
    n_with_details = 0
    n_top_level_only = 0
    field_counter: dict[str, int] = {}

    with gzip.open(META_GZ, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            asin = d.get("parent_asin")
            if not asin:
                continue
            attrs = extract_attrs(d)
            if attrs:
                product_attrs[asin] = attrs
                n_with_attrs += 1
                has_details = any(
                    k not in ("Brand", "Main Category", "Average Rating",
                              "Rating Number", "Price")
                    for k in attrs
                )
                if has_details:
                    n_with_details += 1
                else:
                    n_top_level_only += 1
                for k in attrs:
                    field_counter[k] = field_counter.get(k, 0) + 1

    PRODUCT_ATTRS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PRODUCT_ATTRS_JSON.write_text(
        json.dumps(product_attrs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  total metadata products={n_total}, with attrs={n_with_attrs}")
    log(f"    details-based={n_with_details}, top-level only={n_top_level_only}")
    log(f"    distinct attr fields={len(field_counter)}")
    log(f"  wrote → {PRODUCT_ATTRS_JSON}")
    return product_attrs


def main() -> None:
    log("=== extract_product_attrs.py — Step 1 only ===")
    step1_extract_attrs()
    log("=== DONE ===")


if __name__ == "__main__":
    main()
