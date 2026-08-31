"""Build a 100-user cohort = the users with the MOST REVIEWS globally.

Strategy (user request 2026-08-31):
  - Scan the entire Baby_Products_2023 corpus, count reviews per
    reviewerID (no ASIN restriction, no stage8_5 quality filter).
  - Take the top 100 users by review count.
  - Second scan: collect each top user's 8-60 token sentences
    (unique, lowercase) for the sentence cache + stats.

Output (same schema as phase8c_cohort_100user.json):
  result/gaussian/phase8e_cohort_top100.json
    {
      "config": {...},
      "n_users": 100,
      "users": [{"uid", "n_reviews", "n_sents", "n_asins", "top_asin"}],
      "per_asin_count": {"<asin>": n}
    }
"""
import collections, gzip, json, os, re, time

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
DATA_PATH = f"{REPO}/data/Baby_Products_2023.jsonl.gz"
OUT_PATH = f"{REPO}/result/gaussian/phase8e_cohort_top100.json"

TARGET_N_USERS = 100
MAX_WORD = 60
MIN_WORD = 8


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


print("[1] scan #1 — count reviews per user across the whole corpus...", flush=True)
t0 = time.time()
review_counts = collections.Counter()
n_scanned = 0
with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid:
            review_counts[uid] += 1
        n_scanned += 1
        if n_scanned % 2_000_000 == 0:
            print(f"  {n_scanned/1e6:.1f}M rows, unique users={len(review_counts):,} "
                  f"t={time.time()-t0:.0f}s", flush=True)
print(f"  done: {n_scanned/1e6:.1f}M rows, {len(review_counts):,} unique users, "
      f"t={time.time()-t0:.0f}s")

top = review_counts.most_common(TARGET_N_USERS)
print(f"\n[2] top-{TARGET_N_USERS} by review count:")
print(f"  min reviews = {top[-1][1]}, max = {top[0][1]}, "
      f"median = {sorted(c for _, c in top)[TARGET_N_USERS//2]}")

top_set = set(u for u, _ in top)

print("\n[3] scan #2 — collect 8-60 token sentences + ASINs for top users...", flush=True)
user_sents = {u: set() for u in top_set}
user_asins = collections.defaultdict(set)
n_scanned = 0
with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in top_set:
            if r.get("asin"):
                user_asins[uid].add(r["asin"])
            if r.get("text"):
                for s in split_sents(r["text"]):
                    if MIN_WORD <= len(s.split()) <= MAX_WORD:
                        user_sents[uid].add(s.lower())
        n_scanned += 1
        if n_scanned % 2_000_000 == 0:
            print(f"  {n_scanned/1e6:.1f}M rows, collected "
                  f"{sum(len(v) for v in user_sents.values()):,} sents t={time.time()-t0:.0f}s",
                  flush=True)
print(f"  done, t={time.time()-t0:.0f}s")

users = []
per_asin_count = collections.Counter()
for uid, n_rev in top:
    n_sents = len(user_sents[uid])
    asins = user_asins[uid]
    users.append({
        "uid": uid,
        "n_reviews": n_rev,
        "n_sents": n_sents,
        "n_asins": len(asins),
        "top_asin": max(asins, key=lambda a: len(asins)) if asins else None,
    })
    for a in asins:
        per_asin_count[a] += 1

print(f"\n[4] RESULT")
print(f"  n_users={len(users)}")
print(f"  n_unique_asins (any top user wrote on): {len(per_asin_count)}")
print(f"  n_sents/user: min={min(u['n_sents'] for u in users)} "
      f"max={max(u['n_sents'] for u in users)} "
      f"median={sorted(u['n_sents'] for u in users)[len(users)//2]}")
print(f"  total sents={sum(u['n_sents'] for u in users):,}")
print(f"  n_reviews/user: min={min(u['n_reviews'] for u in users)} "
      f"max={max(u['n_reviews'] for u in users)}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump({
        "config": {
            "TARGET_N_USERS": TARGET_N_USERS,
            "selection": "global_top_review_count",
            "MIN_WORD": MIN_WORD,
            "MAX_WORD": MAX_WORD,
        },
        "n_users": len(users),
        "users": users,
        "per_asin_count": dict(collections.Counter(per_asin_count)),
    }, f, indent=2)
print(f"\nsaved → {OUT_PATH}")
