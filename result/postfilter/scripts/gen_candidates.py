#!/usr/bin/env python3
"""为每个 (user, asin) 用 Qwen 8B 生成 100 条候选 query.

== 风格：StyleR（query/style_embed_query_generate.py）==
- 纯文本 prompt：
    Product attributes:
    - A1 (product_type): {cat0}
    - A2 (brand): {store}
    - ...
    Write one natural shopping query that uses every attribute exactly once.
- 一次 1 query（不要 JSON schema）
- 重复 N_ROUNDS=100 次 (不同 temperature, 目的: 风格多样性)
- 解析: 直接取 raw 输出，去首尾空白
- attrs 与 syntax_depth 一样的 5 个 (A1..A5)

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/candidates.jsonl
  每行: {user_id, asin, round_idx, attrs_used (5 items), candidates: [single str], raw_response}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
from llm_client import QwenLocalClient

EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
TEST_CASES = EXPERIMENT_DIR / "test_cases.jsonl"
OUT_PATH = EXPERIMENT_DIR / "candidates.jsonl"

N_ROUNDS = 30
TEMPERATURE = 0.85
TOP_P = 0.95
MAX_TOKENS = 256


def build_attrs(title: str, main_category: str, store: str, categories: list[str], description: str) -> dict[str, str]:
    """把 meta 字段拼成 5 个 attrs (项目 A1..A5 schema)."""
    cat0 = (categories[0] if len(categories) > 0 else main_category) or "Baby product"
    cat1 = (categories[1] if len(categories) > 1 else "everyday use") or "everyday use"
    desc = (description or "").strip().split(".")[0]
    if not desc:
        desc = "everyday baby essential"
    return {
        "A1 (product_type)": cat0,
        "A2 (brand)": store or "Generic",
        "A3 (use_case)": cat1,
        "A4 (appearance)": title[:80],
        "A5 (detailed)": desc[:120],
    }


def build_styler_prompt(attrs: dict[str, str]) -> str:
    """StyleR style: plain text prompt, no JSON, no system prompt."""
    lines = ["Product attributes:"]
    for key in sorted(attrs.keys()):
        lines.append(f"- {key}: {attrs[key]}")
    lines.append("Write one natural shopping query that uses every attribute exactly once.")
    return "\n".join(lines)


def load_test_cases() -> list[dict]:
    cases = []
    with TEST_CASES.open() as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def main() -> None:
    log = lambda m: print(f"[gen] {m}", flush=True)
    log("加载 test_cases")
    cases = load_test_cases()
    log(f"cases={len(cases)}")

    log("初始化 QwenLocalClient (with_vllm=True)")
    client = QwenLocalClient(with_vllm=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_PATH.open("w", encoding="utf-8")
    total_kept = 0
    for round_idx in range(N_ROUNDS):
        if round_idx == 0 or (round_idx + 1) % 10 == 0:
            log(f"=== Round {round_idx + 1}/{N_ROUNDS} ===")
        for case_idx, case in enumerate(cases):
            attrs = build_attrs(
                case.get("title", ""),
                case.get("main_category", ""),
                case.get("store", ""),
                case.get("categories", []),
                case.get("description", ""),
            )
            prompt = build_styler_prompt(attrs)
            try:
                raw = client.call(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            except Exception as exc:
                if round_idx == 0:
                    log(f"  case={case_idx} ({case['user_id'][:8]}...): call failed: {exc!r}")
                continue
            query = raw.strip()
            if not query or len(query.split()) < 3:
                continue
            record = {
                "user_id": case["user_id"],
                "asin": case["asin"],
                "round_idx": round_idx,
                "attrs_used": attrs,
                "candidates": [query],
                "raw_response": raw,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            total_kept += 1
            if round_idx == 0 and case_idx < 3:
                log(f"  case={case_idx+1}/{len(cases)} ({case['user_id'][:8]}...): query={query[:80]!r}")

    out_f.close()
    log(f"完成: total_queries={total_kept}")


if __name__ == "__main__":
    main()
