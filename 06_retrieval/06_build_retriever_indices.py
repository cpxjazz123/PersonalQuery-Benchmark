#!/usr/bin/env python3
"""Build all retriever indices for full-scale evaluation — 统一入口（3 域）。

Only builds indices, does NOT evaluate.

使用方法:
    python 06_build_retriever_indices.py --category Baby_Products
    python 06_build_retriever_indices.py --category Grocery_and_Gourmet_Food
    python 06_build_retriever_indices.py --category Pet_Supplies
"""

import argparse
import runpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser(description="Build retriever indices（统一入口）")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="域类别",
    )
    args = parser.parse_args()
    runpy.run_path(
        SCRIPT_DIR / f"06_build_retriever_indices_{args.category}.py",
        run_name="__main__",
    )

if __name__ == "__main__":
    main()
