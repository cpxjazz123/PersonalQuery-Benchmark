#!/usr/bin/env python3
"""phase10_9_diag_missing_attr_source.py — 区分"数据无" vs "生成失败".

对每个 needed_append=True 的 candidate:
  - 检查 attrs_5 (源 pair 5 字段) 是否有该 attr 的值
  - 检查 attrs (filter 后的 3 字段) 是否有该 attr 的值
  - 如果 attrs 有值 → generator 失败 (COPY HEAD FAIL)
  - 如果 attrs 无值但 attrs_5 有 → 数据 filter 漏 (BUG)
  - 如果 attrs_5 也无 → 数据稀疏 (asin 本身没此 attr)

源数据来源: phase10_pairs_60.jsonl (原始 attrs 5 字段)
            Amazon Reviews 2023 schema (Brand, Color, Material, Item Weight, Product Dimensions)
            注: 5 字段 attrs 在源 pair jsonl 里就是指定字段
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES_FILE = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_9_4e_candidates_60.jsonl")
PAIRS_FILE = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_pairs_60.jsonl")
ATTR_FIELDS = ["Brand", "Color", "Material"]

ATTR_RE = re.compile(r"[^a-zA-Z0-9./\- ]")


def quick_substring_check(query: str, value: str) -> bool:
    q = ATTR_RE.sub(" ", query.lower())
    v_clean = ATTR_RE.sub(" ", str(value).lower()).strip()
    if not v_clean:
        return False
    return v_clean in q or all(
        t in q for t in v_clean.split() if len(t) >= 3
    )


def main():
    # 1. 加载 pairs (源 attrs_5)
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    pairs_by_key = {(p["user_id"], p["asin"]): p for p in pairs}

    # 2. 加载 candidates
    rows = []
    with CANDIDATES_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))

    needed = [r for r in rows if r["needed_append"]]
    print(f"Total: {len(rows)}")
    print(f"needed_append: {len(needed)} ({len(needed) / len(rows):.2%})")

    # 3. 对每个 needed, 判定每个 missing attr 的来源
    source_cat = Counter()
    by_attr = Counter()
    examples = defaultdict(list)
    missing_per_pair = defaultdict(list)  # (user, asin) → [missing attrs]

    for r in needed:
        key = (r["user_id"], r["asin"])
        pair = pairs_by_key.get(key)
        if pair is None:
            print(f"  [WARN] pair not found: {key}")
            continue
        attrs_5 = pair["attrs"]  # 5 字段源
        attrs_used = r["attrs"]  # 3 字段 filter 后
        raw = r["candidate_query_raw"]

        for k, v_expected in attrs_used.items():
            if not v_expected:
                continue  # 源 attrs_used 已经是 None, 没有要检查的
            if quick_substring_check(raw, v_expected):
                continue  # 这个 attr 实际在 raw 里
            # 该 attr 缺失
            src_value = attrs_5.get(k, "")
            if src_value and src_value == v_expected:
                # attr 值在源数据里, 也传给了 generator, 但 generator 没输出
                source_cat["GEN_FAIL"] += 1
                by_attr[k] += 1
                if len(examples["GEN_FAIL"]) < 8:
                    examples["GEN_FAIL"].append({
                        "user": r["user_id"][:10], "asin": r["asin"], "cond": r["cond"],
                        "raw": raw, "attr": k, "value": v_expected,
                    })
            elif src_value:
                # filter 后 v_expected 与 src_value 不同 (可能是 normalize)
                source_cat["FILTER_MISMATCH"] += 1
                by_attr[k] += 1
                if len(examples["FILTER_MISMATCH"]) < 4:
                    examples["FILTER_MISMATCH"].append({
                        "user": r["user_id"][:10], "asin": r["asin"], "cond": r["cond"],
                        "raw": raw, "attr": k, "value": v_expected, "src_value": src_value,
                    })
            else:
                # 源数据里没这个 attr
                source_cat["DATA_SPARSE"] += 1
                by_attr[k] += 1
                if len(examples["DATA_SPARSE"]) < 8:
                    examples["DATA_SPARSE"].append({
                        "user": r["user_id"][:10], "asin": r["asin"], "cond": r["cond"],
                        "raw": raw, "attr": k, "value": v_expected, "src_attrs_5": attrs_5,
                    })

    print(f"\n=== Source categorization ===")
    for cat in ["GEN_FAIL", "FILTER_MISMATCH", "DATA_SPARSE"]:
        n = source_cat[cat]
        print(f"  {cat:<20s}: {n:>3d} ({n / sum(source_cat.values()):.2%})")
    print(f"  total missing attrs: {sum(source_cat.values())}")

    print(f"\n=== By attribute ===")
    for k, c in by_attr.most_common():
        print(f"  {k}: {c}")

    print(f"\n=== Distinct (user, asin) pairs with missing attrs ===")
    n_pairs = len(missing_per_pair)
    print(f"  {n_pairs} unique pairs")

    print(f"\n=== Examples ===")
    for cat in ["GEN_FAIL", "FILTER_MISMATCH", "DATA_SPARSE"]:
        print(f"\n  --- {cat} ---")
        for ex in examples[cat]:
            print(f"  user={ex['user']} asin={ex['asin']} cond={ex['cond']}")
            print(f"    raw: {ex['raw']!r}")
            print(f"    attr: {ex['attr']} = {ex['value']!r}")
            if 'src_value' in ex:
                print(f"    src_value (5 字段): {ex['src_value']!r}")
            if 'src_attrs_5' in ex:
                print(f"    src_attrs_5: {ex['src_attrs_5']}")

    # 4. 同 (user, asin) 多 cond 都缺同一个 attr 吗?
    print(f"\n=== Pair-level missing attr distribution ===")
    pair_missing = defaultdict(set)
    for r in needed:
        key = (r["user_id"], r["asin"])
        attrs_used = r["attrs"]
        raw = r["candidate_query_raw"]
        for k, v in attrs_used.items():
            if v and not quick_substring_check(raw, v):
                pair_missing[key].add(k)
    pair_missing_count = Counter(len(v) for v in pair_missing.values())
    for n_attr, n_pairs in sorted(pair_missing_count.items()):
        print(f"  pairs with {n_attr} missing attrs: {n_pairs}")


if __name__ == "__main__":
    main()
