#!/usr/bin/env python3
"""Fresh per-product structured attribute extraction from product METADATA.

不依赖旧的 01_preference_extraction 产物; 直接从 meta_Baby_Products_2023
(jsonl.gz) 的 details / store / title / main_category / price / rating
抽取结构化属性。

抽取规则 (用户指令: 不规定字段, 只要是结构化的属性值都需要提取):
  - `details` dict 中所有 value 是 short string (len > 0, len <= MAX_STR_LEN)
    的字段都视为结构化属性 (k = v 形式写入 out)
  - `details` 中的 dict / list / 长字符串 (care instructions / safety warning
    类) 跳过, 不视为结构化属性
  - top-level 结构化字段补全:
      Brand: 优先 details.Brand; 否则 store; 否则 title 首词
      Main Category: d["main_category"]
      Price: d["price"]
      Average Rating: d["average_rating"]
      Rating Number: d["rating_number"]

产出 (全部硬编码, 不接受 CLI 参数, Rule 3):
  1) product_attributes.json
       {asin: {attr_name: attr_value, ...}, ...}
     缺失字段不填; 字段集合因 asin 而异。

属性以 parent_asin 为 join key (元数据无 asin 字段, review 里的 asin 即 parent_asin)。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# 硬编码输入/输出路径 (Rule 3)
META_GZ = Path("/fs04/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz")
OUT_DIR = REPO_ROOT / "result/personal_query/attribute_extraction/Baby_Products"
PRODUCT_ATTRS_JSON = OUT_DIR / "product_attributes.json"

# 字符串值长度上限: 超过此长度的 details 字段 (典型如 Care instructions /
# Safety warning / 从 description 解析来的长句子) 视为非结构化, 跳过。
MAX_STR_LEN = 200

# top-level 字段是否需要提取 (结构性 boolean / numeric / 短 string)
TOP_LEVEL_NUMERIC_FIELDS = ("average_rating", "rating_number", "price")


def extract_attrs(d: dict) -> dict:
    details = d.get("details")
    if not isinstance(details, dict):
        details = {}
    out: dict = {}

    # === 1) details 中所有短字符串字段 ===
    for k, v in details.items():
        if not isinstance(v, str):
            continue  # 跳过 dict / list / 其他非字符串值
        s = v.strip()
        if not s:
            continue
        if len(s) > MAX_STR_LEN:
            continue  # 跳过长文本 (Care instructions 等非结构化说明)
        out[k] = s

    # === 2) Brand fallback: details 无 Brand 时用 store / title ===
    if "Brand" not in out:
        store = d.get("store")
        if isinstance(store, str) and store.strip():
            out["Brand"] = store.strip()
        else:
            title = d.get("title")
            if isinstance(title, str):
                tok = title.split()
                if tok:
                    cand = tok[0].strip()
                    if cand:
                        out["Brand"] = cand

    # === 3) top-level 结构化字段 ===
    main_cat = d.get("main_category")
    if isinstance(main_cat, str) and main_cat.strip():
        out["Main Category"] = main_cat.strip()

    for field in TOP_LEVEL_NUMERIC_FIELDS:
        v = d.get(field)
        if v is None or v == "":
            continue
        out[field.replace("_", " ").title().replace(" ", " ")] = v

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1) 逐商品抽取 ---
    product_attrs: dict[str, dict] = {}
    n_total = 0
    n_with_attrs = 0
    n_with_only_details = 0
    n_with_top_level_only = 0
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
            if attrs:  # 至少抽到 1 个属性才保留
                product_attrs[asin] = attrs
                n_with_attrs += 1
                has_details = any(
                    k not in ("Brand", "Main Category", "Average Rating",
                              "Rating Number", "Price")
                    for k in attrs
                )
                if has_details:
                    n_with_only_details += 1
                else:
                    n_with_top_level_only += 1
                for k in attrs:
                    field_counter[k] = field_counter.get(k, 0) + 1

    PRODUCT_ATTRS_JSON.write_text(
        json.dumps(product_attrs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[1] 元数据商品总数={n_total}, 抽到属性商品数={n_with_attrs}")
    print(f"    含 details 字段的商品={n_with_only_details}, "
          f"仅 top-level 兜底={n_with_top_level_only}")
    print(f"    distinct attribute 字段数={len(field_counter)}")
    top_fields = sorted(field_counter.items(), key=lambda kv: -kv[1])[:25]
    print(f"    top-25 字段覆盖: {top_fields}")
    sample_asin = next(iter(product_attrs))
    print(f"    示例 asin={sample_asin} -> {product_attrs[sample_asin]}")


if __name__ == "__main__":
    main()