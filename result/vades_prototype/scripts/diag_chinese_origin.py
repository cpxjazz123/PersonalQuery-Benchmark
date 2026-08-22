#!/usr/bin/env python3
"""Phase 10.9.4d v4 — Chinese origin diagnostic.

分类每个 candidate 的中文来源:
  CLEAN:              raw=False, final=False
  RAW_ONLY:           raw=True,  final=False    (raw 有中文, post 清洗掉 — 我们不做清洗, 应当为 0)
  RAW_AND_FINAL:      raw=True,  final=True     (raw 有中文, append 后仍然有 — 绝大多数)
  APPEND_INTRODUCED:  raw=False, final=True     (raw 是英文, append 阶段被插入中文)

并分 cond (D_target vs C_random) / needed_append / attr_pass_raw 分别统计.
另外检查: attrs 字段本身是否含中文 (source 5).

输入: phase10_9_4d_candidates_60.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

CANDIDATES_FILE = Path(
    "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_9_4d_candidates_60.jsonl"
)

# CJK Unified Ideographs + Hiragana + Katakana + Hangul
CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def has_cjk(s: str) -> bool:
    return bool(s) and bool(CJK_RE.search(s))


def main():
    rows = []
    with CANDIDATES_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Total candidates: {len(rows)}")

    cat_global = Counter()
    cat_by_cond = defaultdict(Counter)
    cat_by_pass_raw = defaultdict(Counter)
    cat_by_needed_append = defaultdict(Counter)
    examples = defaultdict(list)

    # 来源标签
    n_zh_in_attrs = 0
    n_zh_in_prompt = 0
    n_zh_in_attrs_only = 0  # attrs 含中文但 raw 不含 (=append 引入)

    for r in rows:
        raw = r.get("candidate_query_raw", "") or ""
        final = r.get("candidate_query", "") or ""
        zh_raw = has_cjk(raw)
        zh_final = has_cjk(final)
        cond = r["cond"]
        attr_pass_raw = r["attr_pass_raw"]
        needed_append = r["needed_append"]
        attrs = r["attrs"]
        attrs_zh = any(has_cjk(str(v)) for v in attrs.values() if v)
        prompt_used = r.get("prompt_used", "") or ""
        if attrs_zh:
            n_zh_in_attrs += 1
        if has_cjk(prompt_used):
            n_zh_in_prompt += 1
        if attrs_zh and not zh_raw:
            n_zh_in_attrs_only += 1

        if not zh_raw and not zh_final:
            cat = "CLEAN"
        elif zh_raw and not zh_final:
            cat = "RAW_ONLY"
        elif zh_raw and zh_final:
            cat = "RAW_AND_FINAL"
        else:
            cat = "APPEND_INTRODUCED"

        cat_global[cat] += 1
        cat_by_cond[cond][cat] += 1
        cat_by_pass_raw[attr_pass_raw][cat] += 1
        cat_by_needed_append[needed_append][cat] += 1

        if len(examples[cat]) < 6:
            examples[cat].append({
                "user": r["user_id"][:10],
                "asin": r["asin"],
                "cond": cond,
                "raw": raw,
                "final": final,
                "needed_append": needed_append,
                "attr_pass_raw": attr_pass_raw,
                "attrs": attrs,
                "prompt_used": prompt_used[:160],
            })

    print("\n=== Global (n=%d) ===" % len(rows))
    for cat in ["CLEAN", "RAW_ONLY", "RAW_AND_FINAL", "APPEND_INTRODUCED"]:
        n = cat_global[cat]
        print(f"  {cat:<20s}: {n:>4d} ({n / len(rows):.2%})")

    print("\n=== By cond ===")
    for cond in ["D_target_style", "C_random_style"]:
        sub = cat_by_cond[cond]
        total = sum(sub.values())
        print(f"  {cond} (n={total}):")
        for cat in ["CLEAN", "RAW_ONLY", "RAW_AND_FINAL", "APPEND_INTRODUCED"]:
            n = sub[cat]
            print(f"    {cat:<20s}: {n:>4d} ({n / total:.2%})")

    print("\n=== By attr_pass_raw ===")
    for apr in [True, False]:
        sub = cat_by_pass_raw[apr]
        total = sum(sub.values())
        print(f"  attr_pass_raw={apr} (n={total}):")
        for cat in ["CLEAN", "RAW_ONLY", "RAW_AND_FINAL", "APPEND_INTRODUCED"]:
            n = sub[cat]
            print(f"    {cat:<20s}: {n:>4d} ({n / total:.2%})")

    print("\n=== By needed_append ===")
    for na in [True, False]:
        sub = cat_by_needed_append[na]
        total = sum(sub.values())
        print(f"  needed_append={na} (n={total}):")
        for cat in ["CLEAN", "RAW_ONLY", "RAW_AND_FINAL", "APPEND_INTRODUCED"]:
            n = sub[cat]
            print(f"    {cat:<20s}: {n:>4d} ({n / total:.2%})")

    print("\n=== Upstream sources ===")
    print(f"  candidates with Chinese in attrs: {n_zh_in_attrs}")
    print(f"  candidates with Chinese in prompt: {n_zh_in_prompt}")
    print(f"  candidates attrs=zh but raw=clean (attrs NOT propagated to LLM output): {n_zh_in_attrs_only}")

    print("\n=== Examples per category ===")
    for cat in ["CLEAN", "RAW_ONLY", "RAW_AND_FINAL", "APPEND_INTRODUCED"]:
        print(f"\n  --- {cat} ---")
        for it in examples[cat]:
            print(f"  user={it['user']} cond={it['cond']} needed_append={it['needed_append']} "
                  f"attr_pass_raw={it['attr_pass_raw']}")
            print(f"    raw:   {it['raw']!r}")
            print(f"    final: {it['final']!r}")
            print(f"    attrs: {it['attrs']}")

    # 进一步: 看 RAW_AND_FINAL 中, raw 是否有一段"原本是英文+attribute 周围夹中文"
    print("\n=== RAW_AND_FINAL: 中文 token 位置 (raw vs final 是否完全相同?) ===")
    n_raw_eq_final = 0
    n_raw_diff_final = 0
    for r in rows:
        if not has_cjk(r.get("candidate_query_raw", "")):
            continue
        if r.get("candidate_query_raw") == r.get("candidate_query"):
            n_raw_eq_final += 1
        else:
            n_raw_diff_final += 1
    print(f"  raw == final (append 没改): {n_raw_eq_final}")
    print(f"  raw != final (append 改了但仍有中文): {n_raw_diff_final}")


if __name__ == "__main__":
    main()
