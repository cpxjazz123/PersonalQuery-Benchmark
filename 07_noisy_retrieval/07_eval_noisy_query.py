#!/usr/bin/env python3
"""Evaluate expression_style noisy queries — 统一入口（3 域）"""
import argparse
from pathlib import Path
import sys
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))
from noisy_syntax_depth_eval_common import run_category_eval

def main():
    parser = argparse.ArgumentParser(description="Evaluate noisy queries（统一入口）")
    parser.add_argument(
        "--category",
        required=True,
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
    )
    args = parser.parse_args()
    run_category_eval(args.category)

if __name__ == "__main__":
    main()
