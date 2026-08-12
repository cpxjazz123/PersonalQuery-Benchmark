#!/usr/bin/env python3
"""Run retrieval evaluation on noisy template queries for Grocery_and_Gourmet_Food."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.eval_template_driver import run_template_eval


CATEGORY = "Grocery_and_Gourmet_Food"


if __name__ == "__main__":
    run_template_eval(CATEGORY, mode="noisy")
