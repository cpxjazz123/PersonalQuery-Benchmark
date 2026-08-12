#!/usr/bin/env python3
"""Run review_query_alignment task for Grocery_and_Gourmet_Food."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PQ_CATEGORY"] = "Grocery_and_Gourmet_Food"

_COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(_COMMON))

from evaluate_review_query_alignment import main as alignment_main

if __name__ == "__main__":
    alignment_main()
