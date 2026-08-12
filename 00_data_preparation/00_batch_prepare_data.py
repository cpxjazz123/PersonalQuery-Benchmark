#!/usr/bin/env python3
"""Batch prepare data — 统一入口（3 域）"""
import argparse, runpy
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["Baby_Products","Grocery_and_Gourmet_Food","Pet_Supplies"])
    args = parser.parse_args()
    runpy.run_path(SCRIPT_DIR / f"00_batch_prepare_data_{args.category}.py", run_name="__main__")
if __name__ == "__main__":
    main()
