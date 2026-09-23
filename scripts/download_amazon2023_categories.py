#!/usr/bin/env python3
"""Download Amazon Reviews 2023 raw review + meta jsonl for two categories.

Uses huggingface_hub hf_hub_download with resume + retry, multi-connection aria2c fallback.
Saves to /home/wlia0047/hj82_scratch2/wenyu/amazon2023_dl/ then caller moves to data/.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

DL_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/amazon2023_dl")
DL_DIR.mkdir(parents=True, exist_ok=True)

# 用户指令 2026-09-23: data 目录从 ar57 迁移到 hj82.
DST_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

TARGETS = [
    # (subdir, filename, expected_size_bytes)
    # 用户指令 2026-09-23: 删 Pet_Supplies + Grocery + Handmade, 只保留 Baby / Musical / Video_Games.
    # 用户指令 2026-09-23: 新增 Musical_Instruments (~215k asins, 与 Baby 同量级).
    ("raw/review_categories", "Musical_Instruments.jsonl", 1557845107),
    ("raw/meta_categories", "meta_Musical_Instruments.jsonl", 631877970),
    # 用户指令 2026-09-23: 用 Video_Games 替代 Handmade_Products (~144k asins, 数据密度高).
    ("raw/review_categories", "Video_Games.jsonl", 2682948518),
    ("raw/meta_categories", "meta_Video_Games.jsonl", 437375785),
]


def download_with_aria2c(url: str, out_path: Path, expected_size: int,
                          final_dst: Path | None = None):
    """Download url → out_path (scratch staging).

    Skip conditions (idempotent):
      1. final_dst already exists with expected_size (already installed)
      2. out_path already exists with expected_size (downloaded, not yet moved)
    """
    if final_dst is not None and final_dst.exists() \
            and final_dst.stat().st_size == expected_size:
        print(f"[skip-installed] {final_dst.name}: already at {final_dst} "
              f"({expected_size} bytes)")
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size == expected_size:
        print(f"[skip] {out_path.name}: already complete ({expected_size} bytes)")
        return True
    log = out_path.with_suffix(out_path.suffix + ".aria2.log")
    cmd = [
        "aria2c",
        "-x", "8", "-s", "8", "-k", "1M",
        "--max-tries=20", "--retry-wait=5",
        "--console-log-level=warn", "--summary-interval=60",
        "--auto-file-renaming=false",
        "--check-certificate=true",
        "--file-allocation=none",
        # 用户指令 2026-09-23: 禁用 IPv6 (CDN IPv6 mirror 不可达, 强制 IPv4).
        "--disable-ipv6=true",
        "-d", str(out_path.parent),
        "-o", out_path.name,
        "--log", str(log),
        url,
    ]
    print(f"[start] {out_path.name} ({expected_size/1e9:.2f} GB) -> {out_path}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0:
        print(f"[fail] {out_path.name}: aria2c rc={rc} after {dt:.1f}s", file=sys.stderr)
        return False
    actual = out_path.stat().st_size
    if actual != expected_size:
        print(f"[size-mismatch] {out_path.name}: got {actual} vs expected {expected_size}", file=sys.stderr)
        return False
    print(f"[done] {out_path.name} {actual/1e9:.2f} GB in {dt:.1f}s")
    return True


def main():
    ok = True
    for subdir, fname, size in TARGETS:
        url = f"{BASE}/{subdir}/{fname}"
        out = DL_DIR / fname
        final = DST_DIR / fname
        if not download_with_aria2c(url, out, size, final_dst=final):
            ok = False
            print(f"[abort] {fname} failed", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
