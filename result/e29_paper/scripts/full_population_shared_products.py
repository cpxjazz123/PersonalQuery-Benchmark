"""Full population: per-(X,Y) count of products with >=2 distinct eligible users."""
import json
import gzip
import re
import time
import pickle
import numpy as np
from collections import defaultdict

FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz'
SENT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r'\s+')
GRID_X = [3, 5, 8, 15, 20, 30, 40, 50]
GRID_Y = [20, 40, 80, 200]

# ============ Pass 1: per-user sent count per X ============
print(f"[{time.strftime('%H:%M:%S')}] Pass 1: per-user sent counts per X ...")
user_sents = {}  # user_id -> [sX3, sX5, ..., sX50]
n_rec = 0
t0 = time.time()
last_t = t0
with gzip.open(FILE, 'rt') as f:
    for line in f:
        n_rec += 1
        if n_rec % 500000 == 0:
            now = time.time()
            print(f"  [{time.strftime('%H:%M:%S')}] {n_rec/1e6:.1f}M records, "
                  f"{len(user_sents):,} users, {now-last_t:.0f}s/500k", flush=True)
            last_t = now
        o = json.loads(line)
        u = o['user_id']
        text = o.get('text', '') or ''
        if not text:
            continue
        sents = SENT_RE.split(text)
        if u not in user_sents:
            user_sents[u] = [0] * len(GRID_X)
        cnt = user_sents[u]
        for s in sents:
            wc = len(WORD_RE.split(s.strip())) - 1
            if wc < GRID_X[0]:
                continue
            for i, X in enumerate(GRID_X):
                if wc >= X:
                    cnt[i] += 1

print(f"[{time.strftime('%H:%M:%S')}] Pass 1 done: {n_rec:,} records, "
      f"{len(user_sents):,} users in {time.time()-t0:.0f}s")

# ============ Pass 2: per-product reviewer set ============
print(f"\n[{time.strftime('%H:%M:%S')}] Pass 2: product -> reviewer set ...")
prod_reviewers = {}  # parent_asin -> set(user_id)
n_rec = 0
t0 = time.time()
last_t = t0
with gzip.open(FILE, 'rt') as f:
    for line in f:
        n_rec += 1
        if n_rec % 1000000 == 0:
            now = time.time()
            print(f"  [{time.strftime('%H:%M:%S')}] {n_rec/1e6:.1f}M records, "
                  f"{len(prod_reviewers):,} products, {now-last_t:.0f}s/M", flush=True)
            last_t = now
        o = json.loads(line)
        a = o['parent_asin']
        u = o['user_id']
        if a not in prod_reviewers:
            prod_reviewers[a] = set()
        prod_reviewers[a].add(u)

print(f"[{time.strftime('%H:%M:%S')}] Pass 2 done: {len(prod_reviewers):,} products in "
      f"{time.time()-t0:.0f}s")

# ============ Compute per-(X,Y) shared product count ============
print(f"\n=== Per-(X, Y) shared products: products with >=2 distinct users, "
      f"each having >=Y sents of >=X words ===")
print(f"Total products: {len(prod_reviewers):,}")
print(f"Total users:    {len(user_sents):,}")
print()
print(f"{'X':>3} {'Y':>3} | {'n_shared_products':>16} | {'%':>6}")
results = {}
for X in GRID_X:
    xi = GRID_X.index(X)
    for Y in GRID_Y:
        n_shared = 0
        for a, users in prod_reviewers.items():
            eligible = 0
            for u in users:
                if user_sents.get(u, [0]*len(GRID_X))[xi] >= Y:
                    eligible += 1
                    if eligible >= 2:
                        break
            if eligible >= 2:
                n_shared += 1
        results[f"{X}_{Y}"] = n_shared
        pct = n_shared / len(prod_reviewers) * 100
        print(f"{X:>3} {Y:>3} | {n_shared:>16,} | {pct:>5.2f}%")

# Save results
with open('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/full_pop_shared.json', 'w') as f:
    json.dump({"results": results,
               "n_users": len(user_sents),
               "n_products": len(prod_reviewers)}, f, indent=2)
print(f"\nsaved /home/wlia0047/hj82_scratch2/wenyu/e29_paper/full_pop_shared.json")
