#!/usr/bin/env python3
"""E22 T3 — grid count: products whose top-X users each have >= Y sentences
of >= Z words each ON THE SAME PRODUCT (est: split by [.!?]+, word-count the
fragments). Full matrix for X, Y, Z grids.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))

from query_gen_main import REVIEWS, load_meta, attrs_for_n

MAX_X = 20
MAX_Y = 20
Z_GRID = [1, 2, 3, 5, 8, 10]

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_topxyz_count.json"


def est_sents_minlen(texts, z):
    """Count sentences (split by .!?) with >= z words."""
    n = 0
    for t in texts:
        for frag in re.split(r"[.!?]+", t):
            if len(frag.split()) >= z:
                n += 1
    return n


def main() -> None:
    t0 = time.time()
    prod_users: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            t = (d.get("text") or "").strip()
            if u and a and t:
                prod_users[a][u].append(t)
            if i > 20000000:
                break
    print(f"products with >=1 review: {len(prod_users)}", flush=True)

    meta = load_meta()
    counts: dict[int, np.ndarray] = {z: np.zeros((MAX_X, MAX_Y), dtype=int)
                                     for z in Z_GRID}
    n_all = 0
    for a, uc in prod_users.items():
        if len(uc) < 1 or a not in meta or attrs_for_n(meta[a], 10) is None:
            continue
        n_all += 1
        # per z: top-MAX_X users' est counts (desc)
        top_by_z: dict[int, list[int]] = {}
        for z in Z_GRID:
            top = sorted((est_sents_minlen(v, z) for v in uc.values()),
                         reverse=True)[:MAX_X]
            if top:
                top_by_z[z] = top
        for z, top in top_by_z.items():
            pref_min = np.minimum.accumulate(np.asarray(top))
            for x in range(len(top)):
                m = pref_min[x]
                for y in range(1, min(MAX_Y, m) + 1):
                    counts[z][x, y - 1] += 1
    print(f"products with 10 attrs: {n_all}", flush=True)

    def show(z):
        hdr = f"Z>={z}: X\\Y | " + " ".join(f"{y:>5}"
                                             for y in range(1, MAX_Y + 1))
        print(hdr)
        print("-" * len(hdr))
        for x in range(1, MAX_X + 1):
            row = " ".join(f"{counts[z][x - 1, y - 1]:>5}"
                           for y in range(1, MAX_Y + 1))
            print(f"{x:>3} | {row}", flush=True)

    for z in Z_GRID:
        show(z)
        print(flush=True)

    # compact slice view: (X, Y) at selected points x each Z
    print("compact: rows=Z, cols=(X,Y)", flush=True)
    cols = [(1, 1), (5, 5), (10, 3), (10, 5), (10, 10), (15, 5), (20, 10)]
    hdr = "Z   | " + " ".join(f"X{x}Y{y:<6}" for x, y in cols)
    print(hdr)
    for z in Z_GRID:
        row = " ".join(f"{counts[z][x - 1, y - 1]:<9}"
                       for x, y in cols)
        print(f"z={z:>2} | {row}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"max_x": MAX_X, "max_y": MAX_Y, "z_grid": Z_GRID,
                   "counts": {str(z): counts[z].tolist() for z in Z_GRID},
                   "n_products_with_10attrs": n_all,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
