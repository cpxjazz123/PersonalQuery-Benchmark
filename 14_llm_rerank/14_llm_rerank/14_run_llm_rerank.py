#!/usr/bin/env python3
"""Stage 15 LLM 重排序 — 统一入口（3 域 × 多 query_type）。

用法:
    python 14_run_llm_rerank.py --category Baby_Products
    python 14_run_llm_rerank.py --category Baby_Products --query-type noisy
    python 14_run_llm_rerank.py --category Baby_Products --query-type template --smoke-limit 250
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))

from rerank_runner import run_for_category


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 LLM 重排序统一入口")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="域类别",
    )
    parser.add_argument(
        "--query-type",
        default=None,
        help="单 query_type（如 noisy / preprocessed_noisy / prf_noisy / template）；"
             "省略则运行全部类型（clean + noisy + preprocessed_noisy + prf_noisy + template + template_noisy）",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="smoke test 条数限制；省略则按各类型默认配置",
    )
    args = parser.parse_args()

    query_types = None  # None → run_for_category 内部走默认 QUERY_TYPES
    if args.query_type is not None:
        # map shorthand alias
        qt = args.query_type
        if qt == "clean":
            query_types = ["correct"]
        elif qt == "correct":
            query_types = ["correct"]
        elif qt == "noisy":
            query_types = ["noisy"]
        elif qt == "preprocessed":
            query_types = ["preprocessed_noisy"]
        elif qt == "prf":
            query_types = ["prf_noisy"]
        elif qt == "template":
            query_types = ["template", "template_noisy"]
        else:
            query_types = [qt]

    run_for_category(args.category, query_types=query_types, smoke_limit=args.smoke_limit)


if __name__ == "__main__":
    main()
