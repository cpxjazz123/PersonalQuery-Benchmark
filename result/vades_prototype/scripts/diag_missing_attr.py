#!/usr/bin/env python3
"""phase10_9_diag_missing_attr.py — 诊断 v5 (CJK mask) 中 48 个 needed_append 候选的根因.

对每个 needed_append=True 的 candidate:
  1. 哪个 attr 缺失 (Brand / Color / Material / 多于 1 个)
  2. raw query 中是否有该 attr 的 partial (e.g., "Stainless" 但没 "Steel")
  3. check function 是否漏检 (false negative) — 用放宽后的 substring 检查
  4. LLM 是否真的没生成 (no substring at all)

分类:
  FN_CHECK:        check function 漏检 — 实际 raw 含 attr 但 check 没匹配
  PARTIAL_MATCH:    raw 含 attr 子串 (e.g., "Stainless") 但不全 (e.g., 缺 "Steel")
  REALLY_MISSING:   raw 完全不含 attr 任何 substring
  GENERATION_TRUNC: raw 长度异常短 (< 5 word), 疑似 max_new_tokens 截断
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

CANDIDATES_FILE = Path(
    "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_9_4e_candidates_60.jsonl"
)

ATTR_RE = re.compile(r"[^a-zA-Z0-9./\- ]")


def quick_substring_check(query: str, value: str) -> bool:
    """用最宽松的 substring 检查 (不依赖 check function 的 negatives)."""
    q = ATTR_RE.sub(" ", query.lower())
    v_clean = ATTR_RE.sub(" ", str(value).lower()).strip()
    if not v_clean:
        return False
    # 1. 完整 substring
    if v_clean in q:
        return True
    # 2. 拆 token 后每个 token 都在 q
    tokens = [t for t in v_clean.split() if len(t) >= 3]
    if tokens and all(t in q for t in tokens):
        return True
    # 3. 任意一个 token 在 q (partial)
    return any(t in q for t in tokens)


def main():
    rows = []
    with CANDIDATES_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))

    needed_append_rows = [r for r in rows if r["needed_append"]]
    print(f"Total: {len(rows)}")
    print(f"needed_append: {len(needed_append_rows)} ({len(needed_append_rows) / len(rows):.2%})")

    # 1. 哪个 attr 缺失
    miss_per_attr = Counter()
    for r in needed_append_rows:
        attrs = r["attrs"]
        raw = r["candidate_query_raw"]
        for k, v in attrs.items():
            if not quick_substring_check(raw, v):
                miss_per_attr[k] += 1
    print("\n=== Missing attr breakdown (quick_substring_check) ===")
    for k, c in miss_per_attr.most_common():
        print(f"  {k}: {c} / {len(needed_append_rows)}")

    # 2. 每个 candidate 分类
    cat_global = Counter()
    cat_by_miss_count = defaultdict(Counter)
    fn_examples = []
    partial_examples = []
    really_missing_examples = []
    short_examples = []

    for r in needed_append_rows:
        attrs = r["attrs"]
        raw = r["candidate_query_raw"]
        missing = []
        partial = []
        for k, v in attrs.items():
            if quick_substring_check(raw, v):
                continue
            # 拆分 token, 检查任何 token 是否在 query
            v_clean = ATTR_RE.sub(" ", str(v).lower()).strip()
            tokens = [t for t in v_clean.split() if len(t) >= 3]
            q_lower = ATTR_RE.sub(" ", raw.lower())
            found_any = [t for t in tokens if t in q_lower]
            if found_any:
                partial.append((k, v, found_any))
            else:
                missing.append((k, v))

        n_word = len(raw.split())
        if n_word < 5:
            cat = "GENERATION_TRUNC"
            short_examples.append((r, attrs, missing, partial))
        elif missing and not partial:
            cat = "REALLY_MISSING"
            really_missing_examples.append((r, attrs, missing, partial))
        elif partial and not missing:
            cat = "PARTIAL_MATCH"
            partial_examples.append((r, attrs, missing, partial))
        elif missing and partial:
            cat = "MIXED"  # 一些 attr 真缺失, 一些 partial
            really_missing_examples.append((r, attrs, missing, partial))
        else:
            cat = "FN_CHECK"  # 没有 missing 也没有 partial — check function 漏检
            fn_examples.append((r, attrs, missing, partial))

        cat_global[cat] += 1
        cat_by_miss_count[len(missing)][cat] += 1

    print("\n=== Categorization ===")
    for cat in ["FN_CHECK", "PARTIAL_MATCH", "REALLY_MISSING", "MIXED", "GENERATION_TRUNC"]:
        n = cat_global[cat]
        print(f"  {cat:<20s}: {n:>3d} ({n / len(needed_append_rows):.2%})")

    print("\n=== By n_missing_attrs ===")
    for n_miss, sub in sorted(cat_by_miss_count.items()):
        total = sum(sub.values())
        print(f"  n_missing={n_miss} (total={total}):")
        for cat, c in sub.items():
            print(f"    {cat}: {c}")

    # 3. 看 FN_CHECK (漏检) 详情 — 是否能通过改 check function 修
    print("\n=== FN_CHECK examples (check function 漏检) ===")
    for ex in fn_examples[:5]:
        r, attrs, missing, partial = ex
        print(f"  raw: {r['candidate_query_raw']!r}")
        print(f"  attrs: {attrs}")
        print(f"  → 实际 raw 含全部 attrs, 但 check_attr_pass 返回 False")

    print("\n=== PARTIAL_MATCH examples (raw 含 partial token) ===")
    for ex in partial_examples[:5]:
        r, attrs, missing, partial = ex
        print(f"  raw: {r['candidate_query_raw']!r}")
        print(f"  attrs: {attrs}")
        for k, v, found in partial:
            print(f"    {k}={v!r}: 部分匹配 {found}")

    print("\n=== REALLY_MISSING examples (raw 完全不含 attr) ===")
    for ex in really_missing_examples[:5]:
        r, attrs, missing, partial = ex
        print(f"  raw: {r['candidate_query_raw']!r}")
        print(f"  attrs: {attrs}")
        for k, v in missing:
            print(f"    {k}={v!r}: 完全缺失")

    print("\n=== GENERATION_TRUNC examples (raw < 5 word) ===")
    for ex in short_examples[:5]:
        r, attrs, missing, partial = ex
        print(f"  raw: {r['candidate_query_raw']!r}")
        print(f"  attrs: {attrs}")


if __name__ == "__main__":
    main()
