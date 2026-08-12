#!/usr/bin/env python3
"""Stage 15 LLM 重排序评估 — 统一入口（3 域）。

用法:
    python 14_eval_rerank.py --category Baby_Products
    python 14_eval_rerank.py --category Baby_Products --query-types noisy,template
"""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))

from eval_runner import evaluate_for_category


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 LLM 重排序评估统一入口")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="域类别",
    )
    parser.add_argument(
        "--query-types",
        default=None,
        help="逗号分隔的 query_type 列表（如 correct,noisy,template）；省略则默认 correct,noisy",
    )
    args = parser.parse_args()

    query_types = None
    if args.query_types is not None:
        query_types = [qt.strip() for qt in args.query_types.split(",") if qt.strip()]

    evaluate_for_category(args.category, query_types=query_types)


if __name__ == "__main__":
    main()
