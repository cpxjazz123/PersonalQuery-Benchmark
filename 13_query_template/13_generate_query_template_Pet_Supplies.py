#!/usr/bin/env python3
"""Generate fixed-template queries for Pet_Supplies (1 per user)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.template_query import main


CATEGORY = "Pet_Supplies"


if __name__ == "__main__":
    main(CATEGORY)
