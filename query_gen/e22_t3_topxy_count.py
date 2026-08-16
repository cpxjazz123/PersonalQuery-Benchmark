#!/usr/bin/env python3
"""E22 T3 — grid count: products whose top-X users each have >= Y sentences
on the SAME product (sentence count estimated via [.!?]+ — fast full scan).

Outputs a full matrix for X in 1..MAX_X, Y in 1..MAX_Y.
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

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_topxy_count.json"


def est_sents(texts):
    return sum(len(re.findall(r"[.!?]+", t)) for t in texts)


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
    # per product: top-MAX_X users' est sentence counts (desc), then
    # prefix-min array m[x] = min(counts[:x]); product qualifies (X, Y) iff
    # m[X] >= Y.
    counts_matrix = np.zeros((MAX_X, MAX_Y), dtype=int)
    n_all = 0
    for a, uc in prod_users.items():
        if len(uc) < 1 or a not in meta or attrs_for_n(meta[a], 10) is None:
            continue
        n_all += 1
        top = sorted((est_sents(v) for v in uc.values()), reverse=True)[
            :MAX_X]
        if not top:
            continue
        pref_min = np.minimum.accumulate(np.asarray(top))
        for x in range(len(top)):
            m = pref_min[x]
            for y in range(1, min(MAX_Y, m) + 1):
                counts_matrix[x, y - 1] += 1

    print(f"products with 10 attrs: {n_all}", flush=True)
    hdr = "X\\Y | " + " ".join(f"{y:>5}" for y in range(1, MAX_Y + 1))
    print(hdr)
    print("-" * len(hdr))
    for x in range(1, MAX_X + 1):
        row = " ".join(f"{counts_matrix[x - 1, y - 1]:>5}"
                       for y in range(1, MAX_Y + 1))
        print(f"{x:>3} | {row}", flush=True)

    # representative slices for quick reading
    for y in (1, 2, 3, 5, 10, 15, 20):
        row = {x: int(counts_matrix[x - 1, y - 1])
               for x in (1, 2, 3, 5, 8, 10, 15, 20)}
        print(f"\nY>={y:>2}: " + "  ".join(f"X{x}={v}" for x, v in row.items()))

    with open(OUT, "w") as f:
        json.dump({"max_x": MAX_X, "max_y": MAX_Y,
                   "counts_matrix": counts_matrix.tolist(),
                   "n_products_with_10attrs": n_all,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
