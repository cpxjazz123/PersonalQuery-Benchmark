#!/usr/bin/env python3
"""Verify Amazon Reviews 2023 downloaded files (size + line count) and move
them from /home/wlia0047/hj82_scratch2/wenyu/amazon2023_dl/ into
/home/wlia0047/hj82/wenyu/PersoanlQuery/data/, where the existing
Baby_Products_2023.jsonl + meta_Baby_Products_2023.jsonl live.

2026-09-23: data 目录从 ar57 整体迁移到 hj82.
"""

import json
import os
import shutil
import sys
from pathlib import Path

SRC_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/amazon2023_dl")
# 用户指令 2026-09-23: data 目录从 ar57 迁移到 hj82.
DST_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")

# (filename, expected_size_bytes)
TARGETS = [
    # 用户指令 2026-09-23: 删 Pet_Supplies + Grocery + Handmade, 只保留 Baby / Musical / Video_Games.
    ("Musical_Instruments.jsonl", 1557845107),
    ("meta_Musical_Instruments.jsonl", 631877970),
    ("Video_Games.jsonl", 2682948518),
    ("meta_Video_Games.jsonl", 437375785),
]


def verify_one(path: Path, expected_size: int, dst: Path | None = None) -> bool:
    # 用户指令 2026-09-23: 文件已经在 dst (data/) 装好, 跳过下载源检查.
    if dst is not None and dst.exists() and dst.stat().st_size == expected_size:
        print(f"[skip-installed] {dst.name}: already at {dst} ({expected_size} bytes)")
        return True
    if not path.exists():
        print(f"[missing] {path}")
        return False
    sz = path.stat().st_size
    if sz != expected_size:
        print(f"[size-mismatch] {path}: got {sz} vs expected {expected_size}")
        return False
    # Quick schema sanity: ensure first line is valid JSON and has expected fields
    with path.open("rb") as f:
        first = f.readline()
    rec = json.loads(first)
    if path.name.startswith("meta_"):
        needed = {"main_category", "title"}
    else:
        needed = {"rating", "text", "asin", "user_id"}
    missing = needed - set(rec.keys())
    if missing:
        print(f"[schema-mismatch] {path}: missing fields {missing}")
        return False
    # Count lines (cheap streaming)
    n = 1
    with path.open("rb") as f:
        while f.readline():
            n += 1
    print(f"[ok] {path.name}: {sz/1e9:.2f} GB, {n:,} reviews, fields={sorted(rec.keys())[:6]}")
    return True


def move_one(src: Path, dst: Path) -> bool:
    if dst.exists():
        print(f"[skip-move] {dst} already exists")
        return True
    print(f"[move] {src.name} -> {dst}")
    # Cross-filesystem move: copy then unlink for safety
    shutil.copyfile(str(src), str(dst))
    sz_src = src.stat().st_size
    sz_dst = dst.stat().st_size
    if sz_src != sz_dst:
        print(f"[copy-mismatch] {sz_src} vs {sz_dst}")
        return False
    src.unlink()
    print(f"[moved+unlinked] {dst.name}")
    return True


def main():
    ok = True
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for fname, size in TARGETS:
        src = SRC_DIR / fname
        dst = DST_DIR / fname
        if not verify_one(src, size, dst=dst):
            ok = False
            continue
        if not move_one(src, dst):
            ok = False
    # Cleanup aria2 sidecars
    for f in SRC_DIR.glob("*.aria2"):
        f.unlink()
    for f in SRC_DIR.glob("*.aria2.log"):
        f.unlink()
    if not ok:
        print("[FAIL]", file=sys.stderr)
        return 1
    print("[ALL-DONE]")
    print("Final data dir:")
    for p in sorted(DST_DIR.iterdir()):
        print(f"  {p.name}: {p.stat().st_size/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
