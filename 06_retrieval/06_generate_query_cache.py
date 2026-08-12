#!/usr/bin/env python3
"""生成 query embedding cache — 统一入口（3 域）。

使用方法:
    python 06_generate_query_cache.py --category Baby_Products
    python 06_generate_query_cache.py --category Grocery_and_Gourmet_Food
    python 06_generate_query_cache.py --category Pet_Supplies
"""

import argparse
import runpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 query embedding cache（统一入口）")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="域类别",
    )
    args = parser.parse_args()
    runpy.run_path(
        SCRIPT_DIR / f"06_generate_query_cache_{args.category}.py",
        run_name="__main__",
    )

if __name__ == "__main__":
    main()
