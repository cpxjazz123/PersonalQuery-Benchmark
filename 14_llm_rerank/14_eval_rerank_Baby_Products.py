#!/usr/bin/env python3
"""Stage 15 LLM 重排序评估 — Baby_Products."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))

from eval_runner import evaluate_for_category

CATEGORY_NAME = "Baby_Products"


if __name__ == "__main__":
    evaluate_for_category(CATEGORY_NAME)
