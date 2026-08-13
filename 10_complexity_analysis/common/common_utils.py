"""Shared utilities for 10_complexity_analysis scripts."""
from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)
