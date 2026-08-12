#!/usr/bin/env python3
"""Run style_vector_probe SVR 10-fold task for Baby_Products."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PQ_CATEGORY"] = "Baby_Products"

_COMMON = Path(__file__).resolve().parent / "common"
sys.path.insert(0, str(_COMMON))

from evaluate_vades_style_vector_probe_svr10fold import main as probe_main

if __name__ == "__main__":
    probe_main()
