"""Stage 8.5 strict scan: top 100 ASINs (each ≥10 users) + 10 users/ASIN (each ≥35 reviews).

Strategy:
  1. Scan all reviews → build (parent_asin → user_id → review_count) mapping
  2. Filter: users with ≥35 reviews
  3. Filter: ASINs with ≥10 such users
  4. Sort ASINs by user count desc, take top 100
  5. Per ASIN: take top-10 users by review count
  6. Pick top-4 canonical attrs per ASIN

Output: stage8_5_asins.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_scan.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_scan.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"
OUT = SCRATCH / "stage8_5_asins.json"

# === User requirements (hard) ===
MIN_REVIEWS_PER_USER = 35     # each user must have ≥35 reviews
MIN_USERS_PER_ASIN = 10       # each ASIN must have ≥10 such users
MAX_USERS_PER_ASIN = 10       # take top-10 by review count (deterministic)
TOP_N_ASINS = 100
SEED = 2024   # unused (deterministic sort), but kept for config

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
    """Pick top-n preferred attrs. Strict: ≤30 chars, no commas, ≤5 words."""
    picked = {}
    for k in PREFERRED_ATTRS:
        if k in attrs_dict and attrs_dict[k]:
            v = str(attrs_dict[k]).strip()
            if v and len(v) <= 30 and "," not in v and len(v.split()) <= 5:
                picked[k] = v
        if len(picked) >= n:
            break
    if len(picked) < n:
        for k, v in attrs_dict.items():
            if k in picked:
                continue
            sv = str(v).strip() if v else ""
            if sv and len(sv) <= 30 and "," not in sv and len(sv.split()) <= 5:
                picked[k] = sv
                if len(picked) >= n:
                    break
    return picked


def main():
    log(f"=== Stage 8.5 strict scan: top 100 ASINs × top-10 users (each ≥{MIN_REVIEWS_PER_USER} reviews) ===")

    log(f"loading attrs from {ATTRS_JSON}")
    attrs_db = json.load(open(ATTRS_JSON))
    log(f"  {len(attrs_db)} ASINs with attrs")

    log(f"scanning {REVIEW_GZ}")
    # parent_asin → user_id → review_count
    asin_user_count = collections.defaultdict(lambda: collections.defaultdict(int))
    # user_id → total review count (across all ASINs)
    user_total = collections.Counter()
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        pa = r.get("parent_asin", "") or r.get("asin", "")
        uid = r.get("user_id", "")
        if not pa or not uid:
            continue
        asin_user_count[pa][uid] += 1
        user_total[uid] += 1
        n_records += 1
        if n_records % 1_000_000 == 0:
            log(f"  {n_records / 1e6:.1f}M records")
    log(f"  done: {n_records} records, {len(asin_user_count)} products, "
        f"{sum(len(u) for u in asin_user_count.values())} (product, user) pairs")
    log(f"  unique users: {len(user_total)}")

    # Step 1: filter users by TOTAL review count ≥ MIN_REVIEWS_PER_USER
    log(f"\n=== Step 1: filter users with ≥{MIN_REVIEWS_PER_USER} TOTAL reviews ===")
    high_review_users = {uid for uid, cnt in user_total.items() if cnt >= MIN_REVIEWS_PER_USER}
    log(f"  users with ≥{MIN_REVIEWS_PER_USER} total reviews: {len(high_review_users)}")
    if high_review_users:
        sample = list(high_review_users)[:5]
        for uid in sample:
            log(f"    {uid}: {user_total[uid]} reviews")

    # Step 2: ASINs with ≥10 such users (users need not have ≥35 reviews ON THIS ASIN,
    # only in total; per-ASIN count is whatever the user contributed)
    log(f"\n=== Step 2: ASINs with ≥{MIN_USERS_PER_ASIN} high-review users ===")
    asin_eligible_users = {}
    for pa, uc in asin_user_count.items():
        eligible = [uid for uid in uc.keys() if uid in high_review_users]
        if len(eligible) >= MIN_USERS_PER_ASIN:
            asin_eligible_users[pa] = eligible
    log(f"  ASINs with ≥{MIN_USERS_PER_ASIN} high-review users: {len(asin_eligible_users)}")

    # Step 3: sort by eligible count desc, top 100
    asins_sorted = sorted(asin_eligible_users.items(),
                          key=lambda x: (-len(x[1]), x[0]))  # deterministic
    log(f"\n=== Step 3: top 100 ASINs by eligible user count ===")
    log(f"  eligible counts: "
        f"min={len(asins_sorted[0][1])}, "
        f"median={len(asins_sorted[len(asins_sorted)//2][1])}, "
        f"max={len(asins_sorted[0][1])}")

    # Step 4: filter ASINs with usable attrs + pick top-10 users by review count
    log(f"\n=== Step 4: pick top-10 users per ASIN + verify attrs ===")
    out_asins = []
    skipped = 0
    for a, eligible_users in asins_sorted:
        if len(out_asins) >= TOP_N_ASINS:
            break
        if a not in attrs_db:
            skipped += 1
            continue
        attrs = pick_attrs(attrs_db[a], n=4)
        if len(attrs) < 2:
            skipped += 1
            continue
        # Top-10 users by TOTAL review count (descending), then by uid asc (deterministic)
        user_review_counts = [(uid, user_total[uid]) for uid in eligible_users]
        user_review_counts.sort(key=lambda x: (-x[1], x[0]))
        top_users = [uid for uid, _ in user_review_counts[:MAX_USERS_PER_ASIN]]
        out_asins.append({
            "asin": a,
            "n_users_eligible": len(eligible_users),
            "users_sampled": top_users,
            "attrs_used": attrs,
        })
    log(f"  final ASINs: {len(out_asins)}, skipped (no attrs): {skipped}")

    # Stats
    if out_asins:
        nu = [a["n_users_eligible"] for a in out_asins]
        log(f"  eligible users/ASIN: min={min(nu)}, median={sorted(nu)[len(nu)//2]}, max={max(nu)}")
        log(f"  sampled users/ASIN: always={MAX_USERS_PER_ASIN} (top-{MAX_USERS_PER_ASIN})")

    # Verify sampled users all have ≥35 TOTAL reviews
    all_ok = True
    for a in out_asins:
        for uid in a["users_sampled"]:
            cnt = user_total[uid]
            if cnt < MIN_REVIEWS_PER_USER:
                log(f"  ERROR: {a['asin']} user {uid} has only {cnt} total reviews")
                all_ok = False
    if all_ok:
        log(f"  ✓ all sampled users have ≥{MIN_REVIEWS_PER_USER} TOTAL reviews")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": f"Stage 8.5 strict: top 100 ASINs × top-{MAX_USERS_PER_ASIN} users (each ≥{MIN_REVIEWS_PER_USER} reviews)",
                "MIN_REVIEWS_PER_USER": MIN_REVIEWS_PER_USER,
                "MIN_USERS_PER_ASIN": MIN_USERS_PER_ASIN,
                "MAX_USERS_PER_ASIN": MAX_USERS_PER_ASIN,
                "TOP_N_ASINS": TOP_N_ASINS,
                "PREFERRED_ATTRS": PREFERRED_ATTRS,
            },
            "n_asins": len(out_asins),
            "asins": out_asins,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {OUT}")

    log(f"\nSample 5 ASINs:")
    for a in out_asins[:5]:
        sample_uid = a["users_sampled"][0]
        sample_cnt = user_total[sample_uid]
        log(f"  {a['asin']} eligible={a['n_users_eligible']} sample[0]={sample_uid} "
            f"total_reviews={sample_cnt} attrs={a['attrs_used']}")


if __name__ == "__main__":
    main()