#!/usr/bin/env python3
"""Generate per-retriever cache for noisy template queries (Baby_Products)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.generate_template_cache import main as cache_main


CATEGORY = "Baby_Products"


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--category", CATEGORY, "--mode", "noisy"] + sys.argv[1:]
    cache_main()
