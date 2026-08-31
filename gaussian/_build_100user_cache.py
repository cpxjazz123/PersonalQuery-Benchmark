"""Build user-sentence cache for the 100-user cohort (8.C.1).

Reuses _user_sentence_cache.build_cache for arbitrary cohort. The cache
path is cohort-hash-keyed so this is safe to run alongside the original
12-user cache.

Usage:
  python _build_100user_cache.py
"""
import json
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
from _user_sentence_cache import build_cache

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
COHORT_PATH = f"{REPO}/result/gaussian/phase8c_cohort_100user.json"

with open(COHORT_PATH) as f:
    cohort_data = json.load(f)
COHORT = [u["uid"] for u in cohort_data["users"]]
print(f"loaded {len(COHORT)} users from {COHORT_PATH}")

# build_cache auto-skips if file exists. Force=False.
sents = build_cache(COHORT, REPO)
total = sum(len(v) for v in sents.values())
print(f"\nFinal: {len(sents)}/{len(COHORT)} users, {total} unique sentences")
print(f"min={min(len(v) for v in sents.values())} "
      f"max={max(len(v) for v in sents.values())} "
      f"median={sorted(len(v) for v in sents.values())[len(sents)//2]}")
