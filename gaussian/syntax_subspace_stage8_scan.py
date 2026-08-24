"""Stage 8: Scan review corpus → top 100 ASINs (each ≥10 users) + sample 10 users + extract attrs.

Reads:
  - data/Baby_Products_2023.jsonl.gz: review corpus (parent_asin, user_id)
  - result/product_attributes.json: structured product attrs

Writes:
  - scratch2/.../stage8_asins.json

Strategy:
  1. Scan all review records, aggregate (parent_asin → set of user_id)
  2. Sort by user count desc, take top 100 ASINs (all have ≥10 users)
  3. For each ASIN, sample 10 user_ids (seed-controlled)
  4. From product_attributes.json, pick top 4 preferred attrs per ASIN

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_scan.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_scan.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import json
import random
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"
OUT = SCRATCH / "stage8_asins.json"

TOP_N_ASINS = 100
USERS_PER_ASIN = 10
SEED = 2024
PREFERRED_ATTRS = [
    "Brand", "Color", "Material", "Style", "Size",
    "Age Range (Description)", "Special Feature", "Pattern",
    "Item Weight", "Main Category",
]


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pick_attrs(attrs_dict: dict, n: int = 4) -> dict:
    """Pick top-n preferred attrs from product attrs dict."""
    picked = {}
    for k in PREFERRED_ATTRS:
        if k in attrs_dict and attrs_dict[k]:
            v = str(attrs_dict[k]).strip()
            if v and len(v) <= 100:  # skip very long values
                picked[k] = v
        if len(picked) >= n:
            break
    # If still short, fill from remaining attrs (any)
    if len(picked) < n:
        for k, v in attrs_dict.items():
            if k in picked:
                continue
            sv = str(v).strip() if v else ""
            if sv and len(sv) <= 100:
                picked[k] = sv
                if len(picked) >= n:
                    break
    return picked


def main():
    log("=== Stage 8: scan review corpus for top 100 ASINs (≥10 users) ===")

    log(f"loading attrs from {ATTRS_JSON}")
    attrs_db = json.load(open(ATTRS_JSON))
    log(f"  {len(attrs_db)} ASINs with attrs")

    log(f"scanning {REVIEW_GZ}")
    parent_users = collections.defaultdict(set)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        pa = r.get("parent_asin", "") or r.get("asin", "")
        uid = r.get("user_id", "")
        if not pa or not uid:
            continue
        parent_users[pa].add(uid)
        n_records += 1
        if n_records % 1_000_000 == 0:
            log(f"  {n_records / 1e6:.1f}M records, {len(parent_users)} products")
    log(f"  done: {n_records} records, {len(parent_users)} products")

    # Sort by user count desc
    asins_by_users = sorted(parent_users.items(), key=lambda x: -len(x[1]))
    top100 = asins_by_users[:TOP_N_ASINS]
    log(f"  top 100 user counts: min={len(top100[-1][1])}, max={len(top100[0][1])}")

    # Verify attrs coverage
    n_no_attrs = sum(1 for a, _ in top100 if a not in attrs_db)
    log(f"  ASINs missing from attrs_db: {n_no_attrs}/100")
    if n_no_attrs > 0:
        log(f"  → will skip these ASINs and take next ones")

    # Build output: filter ASINs with usable attrs
    rng = random.Random(SEED)
    out_asins = []
    skipped = 0
    for a, users in asins_by_users:
        if len(out_asins) >= TOP_N_ASINS:
            break
        if a not in attrs_db:
            skipped += 1
            continue
        attrs = pick_attrs(attrs_db[a], n=4)
        if len(attrs) < 2:
            skipped += 1
            continue
        # Sample 10 users
        user_list = sorted(users)  # deterministic order
        sampled = rng.sample(user_list, min(USERS_PER_ASIN, len(user_list)))
        out_asins.append({
            "asin": a,
            "n_users_total": len(user_list),
            "users_sampled": sampled,
            "attrs_used": attrs,
        })
    log(f"  final ASINs: {len(out_asins)}, skipped: {skipped}")

    # Stats
    users_per_asin = [a["n_users_total"] for a in out_asins]
    log(f"  users/ASIN: min={min(users_per_asin)}, median={sorted(users_per_asin)[len(users_per_asin)//2]}, max={max(users_per_asin)}")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8: top 100 ASINs (each ≥10 users) sampled 10 users + canonical attrs from product_attributes.json",
                "TOP_N_ASINS": TOP_N_ASINS,
                "USERS_PER_ASIN": USERS_PER_ASIN,
                "PREFERRED_ATTRS": PREFERRED_ATTRS,
                "SEED": SEED,
            },
            "n_asins": len(out_asins),
            "asins": out_asins,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {OUT}")

    # Print sample
    log(f"\nSample 5 ASINs:")
    for a in out_asins[:5]:
        log(f"  {a['asin']} users={a['n_users_total']} attrs={a['attrs_used']}")


if __name__ == "__main__":
    main()