"""Build a 100-user cross-ASIN high-quality cohort.

Strategy:
  - Source: stage8_5_asins.json (39784 ASINs with ≥2 quality users).
    Each ASIN's users already passed 3-class quality filter
    (Q1 σ_mean ≥ 0.01, Q2 r_floor ≤ 0.3, Q3 inlier_frac ≥ 0.5).
  - Walk ASINs in order; per ASIN, take at most K_MAX_PER_ASIN=3 users.
    Adding ≤3 per ASIN keeps the cohort cross-ASIN diverse.
  - Skip duplicate users globally.
  - For each candidate, count their 8-60 token sentences in the review
    corpus to verify ≥MIN_SENTS=8 (per-sentence feature gate).
  - Stop once 100 users accumulated.

Output:
  result/gaussian/phase8c_cohort_100user.json
    {
      "config": {MIN_SENTS, K_MAX_PER_ASIN, ...},
      "n_users": int,
      "users": [
        {"uid": "...", "asin": "...", "n_sents": int}, ...
      ],
      "per_asin_count": {"<asin>": n, ...}
    }
"""
import collections, gzip, hashlib, json, os, re, sys, time

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
STAGE_ASINS = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8c_cohort_100user.json"

TARGET_N_USERS = int(os.environ.get("PHASE8C_TARGET_N", "100"))
K_MAX_PER_ASIN = int(os.environ.get("PHASE8C_K_MAX_PER_ASIN", "3"))
MIN_SENTS = int(os.environ.get("PHASE8C_MIN_SENTS", "8"))
MAX_WORD = 60
MIN_WORD = 8
LOG_EVERY_N_ASIN = 500


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def load_user_sentence_counts(target_users):
    """Single-pass scan: count 8-60-word sentences per user.

    Returns dict[uid] -> int (number of unique 8-60 word sents).
    For target_users only, since this is the entire corpus.
    """
    target_set = set(target_users)
    counts = collections.Counter()
    n_scanned = 0
    t0 = time.time()
    seen_per_user = {u: set() for u in target_users}
    with gzip.open(f"{REPO}/data/Baby_Products_2023.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            if uid not in target_set:
                n_scanned += 1
                if n_scanned % 2_000_000 == 0:
                    print(f"  scan {n_scanned/1e6:.1f}M t={time.time()-t0:.0f}s "
                          f"hits={sum(len(v) for v in seen_per_user.values()):,}",
                          flush=True)
                continue
            if r.get("text"):
                for s in split_sents(r["text"]):
                    if MIN_WORD <= len(s.split()) <= MAX_WORD:
                        seen_per_user[uid].add(s.lower())
            n_scanned += 1
            if n_scanned % 2_000_000 == 0:
                print(f"  scan {n_scanned/1e6:.1f}M t={time.time()-t0:.0f}s "
                      f"hits={sum(len(v) for v in seen_per_user.values()):,}",
                      flush=True)
    return {u: len(s) for u, s in seen_per_user.items()}


def main():
    print(f"[1] loading stage8_5_asins.json ({STAGE_ASINS})...")
    with open(STAGE_ASINS) as f:
        d = json.load(f)
    asins = d["asins"]
    print(f"  n_asins={d['n_asins']}")

    print(f"\n[2] walking ASINs to collect {TARGET_N_USERS} users "
          f"(≤{K_MAX_PER_ASIN}/ASIN, ≥{MIN_SENTS} sents each)...")

    # Collect candidate users from each ASIN (dedup globally).
    # candidates[uid] -> asin (first ASIN they appeared in for traceability)
    candidates_per_asin = []  # [(asin, [uid1, uid2, ...]), ...]
    for a in asins:
        uids = a.get("users_sampled", []) or []
        # only first K_MAX_PER_ASIN
        candidates_per_asin.append((a["asin"], uids[:K_MAX_PER_ASIN]))

    # Stage 1: collect candidate set globally (≤ TARGET_N_USERS * 5 to leave room
    # after sentence-count filter).
    seen = set()
    candidate_list = []  # (uid, asin)
    OVERSHOOT = 5
    for asin, uids in candidates_per_asin:
        if len(candidate_list) >= TARGET_N_USERS * OVERSHOOT:
            break
        for u in uids:
            if u not in seen:
                seen.add(u)
                candidate_list.append((u, asin))
    print(f"  collected {len(candidate_list)} candidates (≤{TARGET_N_USERS * OVERSHOOT})")

    # Stage 2: count sentences for these candidates in single scan.
    print(f"\n[3] single-pass scan to count 8-60 word sentences per candidate...")
    cand_uids = [u for u, _ in candidate_list]
    sents_count = load_user_sentence_counts(cand_uids)

    # Stage 3: build final cohort (in walk order) up to TARGET_N_USERS
    final_users = []
    per_asin_count = collections.Counter()
    for uid, asin in candidate_list:
        if len(final_users) >= TARGET_N_USERS:
            break
        if sents_count.get(uid, 0) < MIN_SENTS:
            continue
        if per_asin_count[asin] >= K_MAX_PER_ASIN:
            continue
        final_users.append({"uid": uid, "asin": asin, "n_sents": sents_count[uid]})
        per_asin_count[asin] += 1

    n_unique_asin = len(per_asin_count)
    n_users = len(final_users)
    print(f"\n[4] RESULT")
    print(f"  n_users={n_users} (target={TARGET_N_USERS})")
    print(f"  n_unique_asins={n_unique_asin}")
    print(f"  sents per user: min={min(u['n_sents'] for u in final_users)} "
          f"max={max(u['n_sents'] for u in final_users)} "
          f"median={sorted(u['n_sents'] for u in final_users)[n_users//2]}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "config": {
                "TARGET_N_USERS": TARGET_N_USERS,
                "K_MAX_PER_ASIN": K_MAX_PER_ASIN,
                "MIN_SENTS": MIN_SENTS,
                "MIN_WORD": MIN_WORD,
                "MAX_WORD": MAX_WORD,
            },
            "n_users": n_users,
            "users": final_users,
            "per_asin_count": dict(per_asin_count),
        }, f, indent=2)
    print(f"\nsaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
