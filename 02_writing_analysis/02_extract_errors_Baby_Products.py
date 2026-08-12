#!/usr/bin/env python3
"""Extract writing errors for Baby_Products."""

import argparse
from pathlib import Path

_COMMON = Path(__file__).resolve().parent / "common"
import sys
sys.path.insert(0, str(_COMMON))
from extract_errors_common import extract_and_filter_errors


def main():
    parser = argparse.ArgumentParser(description="Extract writing errors for Baby_Products")
    parser.add_argument("--max-users", type=int, default=None, help="Max users to process")
    parser.add_argument("--max-reviews", type=int, default=None, help="Max reviews per user")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent workers")
    args = parser.parse_args()
    
    config = {}
    if args.max_users is not None:
        config["max_users"] = args.max_users
    if args.max_reviews is not None:
        config["max_reviews_per_user"] = args.max_reviews
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers
    
    extract_and_filter_errors("Baby_Products", config)


if __name__ == "__main__":
    main()
