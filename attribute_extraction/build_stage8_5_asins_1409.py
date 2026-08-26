#!/usr/bin/env python3
"""Build stage8_5_asins_1409.json — top 1409 ASINs × top-10 heavy-review users.

逻辑 (复用 stage8_5_scan 流程):
  Step 1: filter users with ≥35 TOTAL reviews (across all Baby_Products)
  Step 2: ASINs with ≥10 high-review users
  Step 3: top 1409 ASINs by eligible user count
  Step 4: pick top-10 users per ASIN + verify attrs (Brand+Color+Item Weight+Material)

输入:
  data/Baby_Products_2023.jsonl.gz     (6,028,884 reviews)
  result/product_attributes.json        (217,724 ASINs with structured attrs)

输出:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins_1409.json

时间预估: ~1min CPU only (扫描 6M records)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

REVIEWS_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"
OUT_JSON = SCRATCH / "stage8_5_asins_1409.json"

MIN_REVIEWS_PER_USER = 35
MIN_USERS_PER_ASIN = 10
MAX_USERS_PER_ASIN = 10
TOP_N_ASINS = 1409

PREFERRED_ATTRS = [
    "Brand", "Color", "Material", "Style", "Size",
    "Age Range (Description)", "Special Feature", "Pattern",
    "Item Weight", "Main Category",
]


def log(msg: str) -> None:
    print(f"[stage8_5_asins_1409] {msg}", flush=True)


def main() -> None:
    log(f"=== Stage 8.5 strict scan: top {TOP_N_ASINS} ASINs × top-{MAX_USERS_PER_ASIN} users ===")

    log("loading attrs ...")
    with ATTRS_JSON.open() as f:
        attrs_full = json.load(f)
    asin_to_attrs: dict[str, dict] = attrs_full if isinstance(attrs_full, dict) else {x["asin"]: x for x in attrs_full}
    log(f"  {len(asin_to_attrs)} ASINs with attrs")

    log("scanning reviews ...")
    user_total: Counter[str] = Counter()
    asin_users: dict[str, set[str]] = defaultdict(set)
    user_per_asin_count: dict[str, Counter[str]] = defaultdict(Counter)

    with gzip.open(REVIEWS_GZ, "rt") as f:
        n = 0
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("reviewerID") or r.get("user_id")
            asin = r.get("asin")
            if not uid or not asin:
                continue
            user_total[uid] += 1
            asin_users[asin].add(uid)
            user_per_asin_count[asin][uid] += 1
            n += 1
            if n % 1_000_000 == 0:
                log(f"  {n/1e6:.1f}M records")
    log(f"  done: {n} records, {len(asin_users)} products, {len(user_total)} users")

    heavy_users = {u for u, c in user_total.items() if c >= MIN_REVIEWS_PER_USER}
    log(f"Step 1: users with ≥{MIN_REVIEWS_PER_USER} total reviews: {len(heavy_users)}")

    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        c = len([u for u in uset if u in heavy_users])
        if c >= MIN_USERS_PER_ASIN:
            eligible_count[asin] = c
    log(f"Step 2: ASINs with ≥{MIN_USERS_PER_ASIN} heavy-review users: {len(eligible_count)}")

    ranked = sorted(eligible_count.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:TOP_N_ASINS]
    counts_only = [c for _, c in ranked]
    log(f"Step 3: top {TOP_N_ASINS} ASINs by eligible count: min={min(counts_only)}, "
        f"median={sorted(counts_only)[len(counts_only)//2]}, max={max(counts_only)}")

    asins_out: list[dict] = []
    skipped_no_attrs = 0
    for asin, _ in ranked:
        adoc = asin_to_attrs.get(asin) or {}
        attrs_used = {}
        for a in PREFERRED_ATTRS:
            v = adoc.get(a)
            if isinstance(v, str) and v.strip():
                attrs_used[a] = v.strip()
        if len(attrs_used) < 1:
            skipped_no_attrs += 1
            continue

        top_users = [u for u, _ in user_per_asin_count[asin].most_common(MAX_USERS_PER_ASIN)]
        asins_out.append({
            "asin": asin,
            "n_users_eligible": eligible_count[asin],
            "users_sampled": top_users,
            "attrs_used": attrs_used,
        })

    log(f"Step 4: final ASINs: {len(asins_out)}, skipped (no attrs): {skipped_no_attrs}")

    config = {
        "description": f"Stage 8.5 strict: top {TOP_N_ASINS} ASINs × top-{MAX_USERS_PER_ASIN} users "
                       f"(each ≥{MIN_REVIEWS_PER_USER} reviews)",
        "MIN_REVIEWS_PER_USER": MIN_REVIEWS_PER_USER,
        "MIN_USERS_PER_ASIN": MIN_USERS_PER_ASIN,
        "MAX_USERS_PER_ASIN": MAX_USERS_PER_ASIN,
        "TOP_N_ASINS": TOP_N_ASINS,
        "PREFERRED_ATTRS": PREFERRED_ATTRS,
    }

    out = {
        "config": config,
        "n_asins": len(asins_out),
        "asins": asins_out,
    }
    with OUT_JSON.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote → {OUT_JSON}")


if __name__ == "__main__":
    main()
