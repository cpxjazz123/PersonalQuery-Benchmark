#!/usr/bin/env python3
"""Extract writing errors — 统一入口（3 域）。

使用方法:
    python 02_extract_errors.py --category Baby_Products
    python 02_extract_errors.py --category Grocery_and_Gourmet_Food
    python 02_extract_errors.py --category Pet_Supplies
"""

import argparse
from pathlib import Path
import sys
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))

from extract_errors_common import extract_and_filter_errors
from config import get_category_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract writing errors（统一入口）")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
        help="域类别",
    )
    args = parser.parse_args()
    config = get_category_config(args.category)
    extract_and_filter_errors(args.category, config)

if __name__ == "__main__":
    main()
