#!/usr/bin/env python3
"""聚合 α × layer sweep 7 个 output 的对比表。

读 7 个 query_records_with_query_inject_*.json, 逐 record 对比 y_plus_query_inject,
输出 markdown 格式:
  - prompt-解读副作用 (含 "assuming", "should be", "to fit" 等)
  - 模板套话 (含 "looking for", "I want")
  - 长度分布
  - attrs coverage

硬编码路径 (Rule 3), 不接受 CLI 参数。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
RESULT_DIR = REPO_ROOT / "result"

# === 7 个 sweep runs ===
RUNS = [
    ("baseline (α=0, L16)", "query_records_with_query_inject_baseline_a0_L16.json"),
    ("B L16 α=0.05", "query_records_with_query_inject_b_L16_a0.05.json"),
    ("B L16 α=0.10", "query_records_with_query_inject_b_L16_a0.10.json"),
    ("B L16 α=0.20", "query_records_with_query_inject_b_L16_a0.20.json"),
    ("B L20 α=0.05", "query_records_with_query_inject_b_L20_a0.05.json"),
    ("B L20 α=0.10", "query_records_with_query_inject_b_L20_a0.10.json"),
    ("B L20 α=0.20", "query_records_with_query_inject_b_L20_a0.20.json"),
    ("B L26 α=0.50 + maskCJK", "query_records_with_query_inject_b_L26_a0.5_maskcjk.json"),
]


def is_prompt_interpretation(text: str) -> bool:
    """检测 prompt-解读副作用 (模型开始 '自言自语' 解读 prompt 内容)。"""
    indicators = [
        r"\bassuming\b",
        r"\bshould be\b",
        r"\bto fit\b",
        r"\btypo\b",
        r'\(.*"',
        r"\".*\"",  # 引号包裹的解释
        r"^\s*\(",  # 以括号开头
    ]
    for p in indicators:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def is_template(text: str) -> bool:
    """检测 'I'm looking for' 等模板套话。"""
    bad_patterns = [
        r"i'm looking for",
        r"i want",
        r"can you recommend",
        r"i need",
        r"i'm searching for",
        r"looking for:",
        r"search for:",
    ]
    t = text.lower()
    for p in bad_patterns:
        if re.search(p, t):
            return True
    return False


def has_chinese(text: str) -> bool:
    """检测中文字符。"""
    return any('一' <= ch <= '鿿' for ch in text)


def attrs_coverage(text: str, attrs: dict) -> float:
    """attrs 中出现在 query 里的比例 (0~1)。"""
    if not attrs:
        return 0.0
    text_lower = text.lower()
    n_hit = 0
    for v in attrs.values():
        if v and str(v).strip() and str(v).strip().lower() in text_lower:
            n_hit += 1
    return n_hit / len(attrs)


def main() -> int:
    # 加载所有 run
    data: dict[str, list[dict]] = {}
    for label, fname in RUNS:
        p = RESULT_DIR / fname
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data[label] = json.load(f)

    if not data:
        print("[err] no sweep outputs found")
        return 1

    print(f"loaded {len(data)} runs, {len(next(iter(data.values())))} records each")
    print()

    # 拿第一条 records 的 (user_id, asin) 索引
    first_run_records = next(iter(data.values()))
    n_records = len(first_run_records)

    # === 按 record 对比 ===
    print("=" * 120)
    print("Per-record 对比 (挑 5 条典型 record, 显示所有 7+1 run 的输出)")
    print("=" * 120)

    # 找 5 个有不同 style 的 user
    seen_users: set[str] = set()
    sample_idxs: list[int] = []
    for i, r in enumerate(first_run_records):
        uid = r["user_id"]
        if uid not in seen_users and len(sample_idxs) < 5:
            seen_users.add(uid)
            sample_idxs.append(i)

    for idx in sample_idxs:
        rec = first_run_records[idx]
        print(f"\n## record[{idx}] asin={rec['asin']} user={rec['user_id'][:12]}...")
        print(f"   attrs: {rec['attrs_used']}")
        print()
        for label in data:
            recs = data[label]
            if idx >= len(recs):
                continue
            q = recs[idx].get("y_plus_query_inject", "")
            flag_int = "🔴 INT" if is_prompt_interpretation(q) else "  ok "
            flag_tpl = "🔴 TPL" if is_template(q) else "      "
            flag_zh = "🔴 ZH " if has_chinese(q) else "      "
            cov = attrs_coverage(q, rec.get("attrs_used", {}))
            display = q.replace("\n", "\\n")[:100]
            print(f"  [{label:>32}] {flag_int}{flag_tpl}{flag_zh} "
                  f"len={len(q):3d} cov={cov:.0%}  '{display}'")

    # === 聚合指标 ===
    print()
    print("=" * 120)
    print("聚合指标 (按 run)")
    print("=" * 120)
    print(f"{'run':<32}  {'INT':>4} {'TPL':>4} {'ZH':>4}  {'avg_len':>7} {'avg_cov':>7}  {'n_records':>9}")
    print("-" * 120)
    for label, recs in data.items():
        n_int = sum(1 for r in recs if is_prompt_interpretation(r.get("y_plus_query_inject", "")))
        n_tpl = sum(1 for r in recs if is_template(r.get("y_plus_query_inject", "")))
        n_zh = sum(1 for r in recs if has_chinese(r.get("y_plus_query_inject", "")))
        avg_len = sum(len(r.get("y_plus_query_inject", "")) for r in recs) / max(len(recs), 1)
        avg_cov = sum(
            attrs_coverage(r.get("y_plus_query_inject", ""), r.get("attrs_used", {}))
            for r in recs
        ) / max(len(recs), 1)
        print(f"{label:<32}  {n_int:>4} {n_tpl:>4} {n_zh:>4}  {avg_len:>7.1f} {avg_cov:>7.1%}  {len(recs):>9}")

    print()
    print("图例: 🔴 INT = 含 prompt-解读 (assuming/should be/引号/括号), "
          "🔴 TPL = 模板套话 (looking for/I want), 🔴 ZH = 含中文")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
