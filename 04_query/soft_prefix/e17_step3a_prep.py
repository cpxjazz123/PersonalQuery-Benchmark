#!/usr/bin/env python3
"""E17 Step 3a: training-user style vectors + user-level train/dev split.

For each of the 900 training users: profile = product groups whose ASIN is
NOT in the user's training target ASINs; vector from profile sentences only
(same 20-feature + 10-opener contract as Step 1). Then split users 80/20
into train/dev (user-clustered, so no leakage across the split).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features_from_doc  # noqa: E402

STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews.json"
TRAIN_DS = REPO_ROOT / "result" / "personal_query" / "04_query" / "Baby_Products" / "query_by_syntax_depth_no_depth_check_10.json"
OUT = REPO_ROOT / "result" / "personal_query" / "e17"
SEED = 42
FEATS = FEATURES20
MAX_SENTENCES = 12


def build_vec(sent_feats: list[np.ndarray], openers: list[int]) -> np.ndarray:
    fmean = np.mean(sent_feats, axis=0) if sent_feats else np.zeros(len(FEATS), dtype=np.float32)
    n = float(np.linalg.norm(fmean))
    if n > 1e-12:
        fmean = fmean / n
    ohist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
    if openers:
        for c in openers:
            ohist[c] += 1
        ohist = ohist / len(openers)
    return np.concatenate([fmean, ohist]).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    ds = json.load(open(TRAIN_DS))
    train_asins_by_user: dict[str, set[str]] = defaultdict(set)
    rows = []
    for rec in ds:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        candidates = rec.get("syntax_depth_queries", [])
        if not attrs:
            for cand in candidates:
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        for cand in candidates:
            q = cand.get("query", "")
            if q:
                rows.append({"user_id": rec["user_id"], "asin": rec["asin"],
                             "attrs_used": attrs, "y_plus_query": q})
        train_asins_by_user[rec["user_id"]].add(rec["asin"])
    train_users = sorted(train_asins_by_user)
    print(f"candidate rows: {len(rows)} users: {len(train_users)}")
    nlp = load_spacy_model()

    data = json.load(open(STAGE1))
    by_user = {u["user_id"]: u for u in data["users"]}
    profile_texts: dict[str, list[str]] = {}
    for uid in train_users:
        u = by_user.get(uid)
        if not u:
            continue
        ex = set(train_asins_by_user[uid])
        groups = [r for r in u.get("results", []) if r.get("target_reviews") and r["asin"] not in ex]
        texts = [t for r in groups for t in r["target_reviews"] if t.strip()]
        if texts:
            profile_texts[uid] = texts
    print(f"train users with non-target profile reviews: {len(profile_texts)} / {len(train_users)}")

    flat = [(uid, t[:1000]) for uid in profile_texts for t in profile_texts[uid][: MAX_SENTENCES * 3]]
    sent_feats: dict[str, list[np.ndarray]] = defaultdict(list)
    openers: dict[str, list[int]] = defaultdict(list)
    for (uid, t), doc in zip(flat, nlp.pipe([t for _, t in flat], batch_size=256)):
        for sent in doc.sents:
            toks = [tok for tok in sent if not tok.is_punct and not tok.is_space]
            if len(toks) < 3:
                continue
            try:
                ex = extract_clause_features_from_doc(sent, sent.text)
                sent_feats[uid].append(np.asarray([float(ex.get(k, 0.0)) for k in FEATS], dtype=np.float32))
            except Exception:
                continue
            openers[uid].append(OPENER_CLASSES.index(opener_class_of(toks[0].text)))
    vecs = {}
    for uid in profile_texts:
        sf, op = sent_feats.get(uid, []), openers.get(uid, [])
        if len(sf) >= 2:
            vecs[uid] = build_vec(sf, op).tolist()
    print(f"users with vectors: {len(vecs)}")

    # user-clustered 80/20 split (dev users never train; all rows of a user
    # stay in one fold)
    rng.shuffle(train_users)
    n_dev = max(1, int(round(len(train_users) * 0.2)))
    dev_users = set(train_users[:n_dev])
    train_users_ = set(train_users[n_dev:])
    dev_rows = [r for r in rows if r["user_id"] in dev_users]
    train_rows = [r for r in rows if r["user_id"] in train_users_]
    print(f"split: train users={len(train_users_)} rows={len(train_rows)} | dev users={len(dev_users)} rows={len(dev_rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_users": len(vecs), "dim": len(FEATS) + len(OPENER_CLASSES),
               "vectors": vecs}, open(OUT / "e17_train_vectors.json", "w"))
    json.dump({"seed": SEED,
               "train": {"users": sorted(train_users_), "rows": train_rows},
               "dev": {"users": sorted(dev_users), "rows": dev_rows}},
              open(OUT / "e17_train_dev_split.json", "w"))
    print(f"wrote {OUT}/e17_train_vectors.json + e17_train_dev_split.json")


if __name__ == "__main__":
    main()
