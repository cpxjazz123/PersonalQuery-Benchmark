#!/usr/bin/env python3
"""Stage 15 LLM 重排序 — Baby_Products noisy query 专用。"""

import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "common"))

from rerank_runner import run_for_category  # noqa: E402

CATEGORY_NAME = "Baby_Products"


if __name__ == "__main__":
    run_for_category(CATEGORY_NAME, query_types=["noisy"])
