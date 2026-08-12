#!/usr/bin/env python3
"""Run retrieval evaluation on fixed-template queries for Baby_Products."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.eval_template_driver import run_template_eval


CATEGORY = "Baby_Products"


if __name__ == "__main__":
    run_template_eval(CATEGORY)
