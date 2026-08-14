"""Compute per-user 20-dim syntactic feature means from review sentences.

E14 statistic-style skeletons need per-user statistics for the multi-product
dataset users (the VADES sentence file only covers the 204 legacy candidate
users). This script parses each target user's review sentences with spacy,
extracts the 20 clause features, and stores the per-user means (JSON) +
per-user skeletons (JSONL). Only SYNTACTIC statistics are kept — no review
text is stored or used downstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features_from_doc,
)
from stat_skeleton import (  # noqa: E402
    FEATURES20,
    stat_to_skeleton,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--dataset", required=True, help="multi-product dataset json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = json.load(open(args.dataset))
    user_ids = sorted({r["user_id"] for r in ds})
    print(f"dataset users: {len(user_ids)}", flush=True)

    reviews = json.load(open(
        REPO_ROOT / "result" / "personal_query" / "01_preference_extraction"
        / args.category / "stage1_filtered_users_reviews.json"
    ))
    sentences_by_user: dict = defaultdict(list)
    for u in reviews["users"]:
        if u["user_id"] not in set(user_ids):
            continue
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if txt and isinstance(txt, str) and txt.strip():
                    sentences_by_user[u["user_id"]].append(txt[:1000])

    nlp = load_spacy_model()
    means: dict = {}
    n_sent = {}
    for uid in user_ids:
        texts = sentences_by_user.get(uid, [])
        if not texts:
            continue
        feats = []
        for doc in nlp.pipe(texts, batch_size=128):
            try:
                extracted = extract_clause_features_from_doc(doc, doc.text)
                feats.append([float(extracted.get(k, 0.0)) for k in FEATURES20])
            except Exception:
                continue
        if not feats:
            continue
        m = np.mean(feats, axis=0)
        means[uid] = m.tolist()
        n_sent[uid] = len(feats)

    out = {
        "category": args.category,
        "feature_names": FEATURES20,
        "n_users": len(means),
        "means": means,
        "n_sentences_per_user": n_sent,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}: {len(means)} users with stats", flush=True)

    # skeletons for those users
    sk_rows = []
    for uid, m in means.items():
        sk = stat_to_skeleton(np.asarray(m, dtype=np.float32), uid)
        sk_rows.append({"user_id": uid, "skeleton": sk})
    sk_out = str(Path(args.out).with_suffix("")) + "_skeletons.jsonl"
    with open(sk_out, "w") as f:
        for row in sk_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {sk_out}: {len(sk_rows)} skeletons", flush=True)


if __name__ == "__main__":
    main()
