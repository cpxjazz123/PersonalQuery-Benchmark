#!/usr/bin/env python3
"""Protocol-ceiling check on the RICH single product B0BQ1QK14T.

Top-10 users have 17-208 sentences ON this product. Question: with how many
query-side sentences (k in {4, 8, 16}) can a random k-sentence subset of the
user's OWN reviews be matched back to the full-review z7 (rank-1 rate)?
This is the noise ceiling for the generation protocol (N_QUERIES x ~1.5
sents each). No GPU needed.
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import TASK1_VECTORS, REVIEWS
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2

ASIN = "B0BQ1QK14T"
N_SEEDS = 30
SIZES = [4, 8, 16]


def main() -> None:
    t0 = time.time()
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    fidx = vd["feature_indices"].astype(int)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    prod_revs: dict[str, list[str]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            t = (d.get("text") or "").strip()
            if u and a == ASIN and t:
                prod_revs[u].append(t)
            if i > 20000000:
                break
    top10 = sorted(prod_revs, key=lambda u: -len(prod_revs[u]))[:10]
    user_sfs: dict[str, list[dict]] = {}
    for u in top10:
        sfs = []
        for doc in nlp.pipe(prod_revs[u][:200], batch_size=64):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        if sfs:
            user_sfs[u] = sfs
    users = list(user_sfs)
    print(f"users: {len(users)}  sfs: {[len(user_sfs[u]) for u in users]}",
          flush=True)

    z7 = {u: ((user_features_v2(user_sfs[u]) - tm) / ts)[fidx]
          for u in users}

    stats = {k: [] for k in SIZES}
    for sz in SIZES:
        for seed in range(N_SEEDS):
            rs = random.Random(seed * 1000 + 1)
            ranks = []
            for u in users:
                sfs = user_sfs[u]
                if len(sfs) < sz:
                    continue
                rs.shuffle(sfs)
                v = user_features_v2(sfs[:sz])
                if v is None:
                    continue
                q = ((v - tm) / ts)[fidx]
                dists = {o: float(np.linalg.norm(q - z7[o]))
                         for o in users}
                rank = 1 + sum(1 for o in users if dists[o] < dists[u])
                ranks.append(rank)
            stats[sz].extend(ranks)
    for sz in SIZES:
        ranks = stats[sz]
        n = len(ranks)
        r1 = sum(1 for r in ranks if r == 1)
        print(f"k={sz:>2}: rank-1 = {r1}/{n} ({r1 / max(1, n):.3f}, "
              f"chance 1/{len(users)} = {1 / len(users):.3f})  "
              f"hist={sorted(ranks)}", flush=True)
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
