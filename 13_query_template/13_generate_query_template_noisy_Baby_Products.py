#!/usr/bin/env python3
"""Noisy template query generation - Baby_Products (07 lambdamart_userbased)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.template_noisy_query import main as noisy_main


CATEGORY = "Baby_Products"


if __name__ == "__main__":
    noisy_main(CATEGORY)
