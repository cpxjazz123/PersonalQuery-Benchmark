#!/usr/bin/env python3
"""Phase 35.B Setup: 选 split=0 size≥10 subset records + 扩展 user exemplars.

输出:
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35b_records.json
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/user_sents_phase35b.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")

# === Inputs ===
PHASE15_7_PER_PAIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase15_7_intra_product_per_pair.jsonl")
PHASE14_Q10L_PAIRS = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase14_q10l_pairs.jsonl")
PHASE14_Q10L_USER_REVIEWS = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase14_q10l_user_reviews.pkl")  # 56MB pickle, 2041 users
USER_SENTS_OLD = SCRATCH / "user_sents_per_user.json"
ATTRS_JSON = REPO_ROOT / "result/query_records.json"

# === Outputs ===
PHASE35B_RECORDS = SCRATCH / "phase35b_records.json"
PHASE35B_USER_SENTS = SCRATCH / "user_sents_phase35b.json"

# === Settings ===
SPLIT = 0
MIN_POOL_SIZE = 10
MAX_RECORDS = 100  # cap at 100 to keep Qwen forward tractable

ATTRS_MAP_FILE = SCRATCH / "phase35b_attrs_map.json"


def main():
    # 1) Load Phase 15.7 per_pair to know which (uid, asin) have size>=N pool
    print("[load] Phase 15.7 per_pair...")
    recs_15_7 = [json.loads(l) for l in open(PHASE15_7_PER_PAIR)]
    # (uid, asin) → pool_size from split=0
    pool_size: dict[tuple[str, str], int] = {}
    for r in recs_15_7:
        if r["split"] == SPLIT and r["method"] == "15.1_D_off":
            pool_size[(r["user_id"], r["asin"])] = r["intra_product_size"]

    # 2) Load phase14_q10l_pairs.jsonl (2041 user records with attrs)
    print("[load] phase14_q10l_pairs.jsonl (2041 records with attrs)...")
    pairs = [json.loads(l) for l in open(PHASE14_Q10L_PAIRS)]
    print(f"[load] {len(pairs)} pairs")

    # 3) Filter by pool_size >= MIN_POOL_SIZE, cap at MAX_RECORDS
    phase35b_records = []
    for r in pairs:
        uid, asin = r["user_id"], r["asin"]
        ps = pool_size.get((uid, asin), 0)
        if ps >= MIN_POOL_SIZE:
            phase35b_records.append({
                "user_id": uid,
                "asin": asin,
                "attrs_used": r["attrs"],
                "n_attrs": r["n_attrs"],
                "intra_product_size": ps,
            })
        if len(phase35b_records) >= MAX_RECORDS:
            break
    print(f"[select] split={SPLIT}, size>={MIN_POOL_SIZE}: {len(phase35b_records)} records")

    # 4) Save records
    json.dump(phase35b_records, open(PHASE35B_RECORDS, "w"), indent=2, ensure_ascii=False)
    print(f"[save] {len(phase35b_records)} records → {PHASE35B_RECORDS}")

    # 5) Build user exemplars for needed users
    needed_uids = set(r["user_id"] for r in phase35b_records)
    print(f"[exemplars] need {len(needed_uids)} user exemplars")

    user_sents = {}
    if USER_SENTS_OLD.exists():
        old = json.load(open(USER_SENTS_OLD))
        for uid in needed_uids:
            if uid in old:
                user_sents[uid] = old[uid]
    print(f"[exemplars] {len(user_sents)} already cached")

    missing_uids = needed_uids - set(user_sents.keys())
    print(f"[exemplars] need to extract {len(missing_uids)} more from phase14_q10l_user_reviews.pkl")

    if missing_uids:
        import pickle
        with open(PHASE14_Q10L_USER_REVIEWS, "rb") as f:
            user_reviews = pickle.load(f)
        print(f"[exemplars] loaded pickle: {len(user_reviews)} users")

        sent_re = re.compile(r"(?<=[.!?])\s+")
        for uid in missing_uids:
            reviews = user_reviews.get(uid, [])
            # reviews is list of {sentence_text, ...} or raw strings — inspect first
            all_sents = []
            for rev in reviews:
                if isinstance(rev, dict):
                    txt = rev.get("sentence_text") or rev.get("review_text") or rev.get("text") or ""
                else:
                    txt = str(rev)
                if not txt:
                    continue
                sents = [s.strip() for s in sent_re.split(txt) if 30 <= len(s.strip()) <= 300]
                all_sents.extend(sents)
            # keep top 5 by length (proxy for informativeness)
            all_sents.sort(key=len, reverse=True)
            user_sents[uid] = all_sents[:5]
        print(f"[exemplars] after extract: {len(user_sents)} users cached")

    json.dump(user_sents, open(PHASE35B_USER_SENTS, "w"), ensure_ascii=False)
    print(f"[save] {len(user_sents)} user exemplars → {PHASE35B_USER_SENTS}")
    print(f"[done] {len(phase35b_records)} records, {len(user_sents)} user exemplars ready")


if __name__ == "__main__":
    main()
