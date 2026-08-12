#!/usr/bin/env python3
"""Generate 10 Pet_Supplies expression_style queries per user (no depth check)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.expression_style_no_depth_check import main


CATEGORY = "Pet_Supplies"


if __name__ == "__main__":
    main(CATEGORY)
