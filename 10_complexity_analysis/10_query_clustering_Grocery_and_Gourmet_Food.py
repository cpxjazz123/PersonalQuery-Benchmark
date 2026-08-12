#!/usr/bin/env python3
"""Run query_clustering task for Grocery_and_Gourmet_Food."""

from __future__ import annotations

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(_COMMON))

from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline

if __name__ == "__main__":
    run_query_gmm_pipeline(
        category="Grocery_and_Gourmet_Food",
        query_file=None,
        write_back_to_query_file=False,
        attach_retrieval=True,
    )
