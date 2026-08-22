#!/usr/bin/env python3
"""Build enlarged copy head training set (~2000+ records) via prompt augmentation.

源数据:
  - preference_pairs_60.jsonl (60 records) — chosen_query + attrs (5 fields)
  - phase10_9_counterfactual_pairs.jsonl (94 records) — chosen_q_u/v + attrs (5 fields)

增强策略 (3维笛卡尔积):
  1. prompt 模板: 4 选 1
     (a) "Write a short shopping query that includes every attribute."
     (b) "Generate a brief product query mentioning all listed attributes."
     (c) "Compose a concise search query that mentions every product detail."
     (d) "Write a natural shopping query that mentions every attribute."
  2. attr field set: 3 选 1
     (a) Brand/Color/Material (基础)
     (b) Brand/Color/Material + Item Weight (5 fields)
     (c) Brand/Color/Material + Product Dimensions (5 fields)
  3. chosen_query 引用: 每条原 chosen_query 用一次 (no swap)
     加 2 个 negative samples:
       - attr_swap: 用随机其他 user 的 attr value 替换 → 期望 gate 不 copy (negative)
       - attr_drop: 删除 1 个 attr → 期望 copy 剩余 2 attr

总计: ~154 records × 4 templates × 3 attr_sets × 3 (pos/swap/drop) ≈ 5544 records
筛选 Brand/Color/Material 至少 2 个有值后: ~5000 records
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PREF_FILE = OUT_DIR / "preference_pairs_60.jsonl"
CF_FILE = OUT_DIR / "phase10_9_counterfactual_pairs.jsonl"
OUT_TRAIN = OUT_DIR / "phase10_9_4h_large_train_set.jsonl"
LOG_OUT = OUT_DIR / "build_large_train_set.log"

CORE_FIELDS = ["Brand", "Color", "Material"]
TEMPLATES = [
    "Write a short shopping query that includes every attribute.",
    "Generate a brief product query mentioning all listed attributes.",
    "Compose a concise search query that mentions every product detail.",
    "Write a natural shopping query that mentions every attribute.",
]
ATTR_FIELD_SETS = [
    ["Brand", "Color", "Material"],
    ["Brand", "Color", "Material", "Item Weight"],
    ["Brand", "Color", "Material", "Product Dimensions"],
]

SEED = 42


def log(m):
    print(f"[build_large] {m}", flush=True)


def filter_attrs(attrs: Dict[str, str], fields: List[str]) -> Dict[str, str]:
    return {k: v for k, v in attrs.items() if k in fields and v}


def make_prompt_body(attrs: Dict[str, str], template: str) -> str:
    lines = ["Product attributes:"]
    for k in sorted(attrs):
        lines.append(f"{k}: {attrs[k]}")
    lines.append(template)
    return "\n".join(lines)


def make_attr_swap(attrs: Dict[str, str], rng: random.Random) -> dict:
    """随机选 1 个 attr, 用随机其他 user 的 value 替换 → negative sample."""
    out = dict(attrs)
    if not out:
        return out
    key = rng.choice(list(out.keys()))
    # 随机 fake value (e.g., 5 letters)
    fake = "Xyz" + str(rng.randint(100, 999))
    out[key] = fake
    return out


def make_attr_drop(attrs: Dict[str, str], rng: random.Random) -> dict:
    """随机删除 1 个 attr (保持剩余 attribute_complete)."""
    out = dict(attrs)
    if len(out) < 3:
        return out
    key = rng.choice(list(out.keys()))
    out.pop(key, None)
    return out


def main():
    rng = random.Random(SEED)

    # 1. 加载源数据
    log("[1] Loading source data ...")
    sources = []
    with PREF_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            r["chosen_query"] = r.get("chosen_query", "").strip()
            r["attrs"] = r.get("attrs", {})
            r["user_id"] = r.get("user_id", "")
            r["asin"] = r.get("asin", "")
            if r["chosen_query"] and len(filter_attrs(r["attrs"], CORE_FIELDS)) >= 2:
                sources.append({"user_id": r["user_id"], "asin": r["asin"],
                                "attrs": r["attrs"], "chosen_query": r["chosen_query"],
                                "src": "pref"})
    log(f"  pref source: {len(sources)} records")

    with CF_FILE.open() as f:
        cf_n = 0
        for line in f:
            r = json.loads(line)
            # counterfactual has 2 sides (u, v)
            for side, qk, ak in [("u", "chosen_q_u", "attrs_u"), ("v", "chosen_q_v", "attrs_v")]:
                q = r.get(qk, "").strip()
                attrs = r.get(ak, {})
                uid = r.get(f"user_{side}", "")
                asin = r.get(f"asin_{side}", "")
                if q and len(filter_attrs(attrs, CORE_FIELDS)) >= 2:
                    sources.append({"user_id": uid, "asin": asin,
                                    "attrs": attrs, "chosen_query": q,
                                    "src": "cf"})
                    cf_n += 1
        log(f"  cf source: {cf_n} records (u+v)")

    log(f"  total source: {len(sources)} records")

    # 2. 构建增强 dataset
    log("[2] Building augmented dataset ...")
    out_rows = []
    for src in sources:
        base_attrs = filter_attrs(src["attrs"], CORE_FIELDS)
        for fields in ATTR_FIELD_SETS:
            attrs_subset = filter_attrs(src["attrs"], fields)
            if len(attrs_subset) < 2:
                continue
            for template in TEMPLATES:
                # Positive: exact attrs
                pos_attrs = attrs_subset
                pos_prompt = make_prompt_body(pos_attrs, template)
                out_rows.append({
                    "user_id": src["user_id"], "asin": src["asin"],
                    "attrs": pos_attrs, "prompt": pos_prompt,
                    "chosen_query": src["chosen_query"],
                    "type": "positive",
                    "src": src["src"],
                })

                # Negative: attr swap (replace one with fake)
                neg_swap_attrs = make_attr_swap(pos_attrs, rng)
                if neg_swap_attrs != pos_attrs:
                    neg_swap_prompt = make_prompt_body(neg_swap_attrs, template)
                    out_rows.append({
                        "user_id": src["user_id"], "asin": src["asin"],
                        "attrs": neg_swap_attrs, "prompt": neg_swap_prompt,
                        "chosen_query": src["chosen_query"],
                        "type": "negative_swap",
                        "src": src["src"],
                    })

                # Negative: attr drop (remove one, keep ≥ 2)
                if len(pos_attrs) >= 3:
                    neg_drop_attrs = make_attr_drop(pos_attrs, rng)
                    if neg_drop_attrs != pos_attrs and len(neg_drop_attrs) >= 2:
                        neg_drop_prompt = make_prompt_body(neg_drop_attrs, template)
                        out_rows.append({
                            "user_id": src["user_id"], "asin": src["asin"],
                            "attrs": neg_drop_attrs, "prompt": neg_drop_prompt,
                            "chosen_query": src["chosen_query"],
                            "type": "negative_drop",
                            "src": src["src"],
                        })

    log(f"  augmented: {len(out_rows)} records")
    # type counts
    from collections import Counter
    type_counts = Counter(r["type"] for r in out_rows)
    log(f"  type breakdown: {dict(type_counts)}")
    src_counts = Counter(r["src"] for r in out_rows)
    log(f"  source breakdown: {dict(src_counts)}")

    # 3. 写文件
    log(f"[3] Writing to {OUT_TRAIN} ...")
    with OUT_TRAIN.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  done. {len(out_rows)} records.")


if __name__ == "__main__":
    main()