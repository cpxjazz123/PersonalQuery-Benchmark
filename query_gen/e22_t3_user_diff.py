#!/usr/bin/env python3
"""E22 T3 — simple between-user style distance on ONE product.

No split-half, no AUC. For each (min_sents, n_users) cell on the product:
  - per user: syntax vector from ALL their reviews on this product
    (281 dims: syntactic, no punctuation, no length-correlated features)
  - pairwise L2 distances between DIFFERENT users (real)
  - baseline: same number of random user-pair distances after permuting
    labels (shuffled pairing)
Report: real mean/std vs shuffled mean/std and their overlap.
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
from extract_syntactic_features import (
    per_sentence_features_v2, user_features_v2, ALL_FEATS_V2,
)

ASIN = "B0BQ1QK14T"
MIN_SENTS_GRID = [5, 10, 20, 30]
N_USERS_GRID = [5, 10, 20]
N_SHUFFLE = 200
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_user_diff.json"


def main() -> None:
    t0 = time.time()
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # syntactic-only, no punctuation
    PUNCT_PREFIXES = ("punct", "n_punct_total")
    SYN = [i for i, n in enumerate(ALL_FEATS_V2)
           if not n.startswith(PUNCT_PREFIXES)]

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
    print(f"product {ASIN}: {len(prod_revs)} users", flush=True)

    user_sfs: dict[str, list[dict]] = {}
    all_sfs: list[dict] = []
    for u, texts in prod_revs.items():
        sfs = []
        for doc in nlp.pipe(texts, batch_size=64):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        if sfs:
            user_sfs[u] = sfs
            all_sfs.extend(sfs)

    # ---- per-sentence vector matrix (cached) for fast length-correlation
    cache = REPO_ROOT / "result" / "cache" / "e22_t3_sentvecs.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cc = np.load(cache, allow_pickle=True)
        P = cc["P"]
        toklens = cc["toklens"]
        if P.shape[0] != len(all_sfs):
            P = None
    else:
        P = None
    if P is None:
        t1 = time.time()
        P = np.stack([user_features_v2([sf]) for sf in all_sfs])
        toklens = np.asarray([sf["n_tok"] for sf in all_sfs],
                             dtype=np.float64)
        np.savez_compressed(cache, P=P, toklens=toklens)
        print(f"per-sentence matrix {P.shape} built "
              f"({time.time() - t1:.0f}s), cached", flush=True)

    # ---- length-correlated screen (|r(n_tok)| >= 0.30) — vectorized
    # NOTE: sentence-level constant features (std<1e-12) are KEPT with r=0
    # (they vary across users after aggregation; matching style_scan_dim).
    kept = []
    for j in SYN:
        x = P[:, j]
        if x.std() < 1e-12:
            kept.append(j)
            continue
        r = abs(float(np.corrcoef(x, toklens)[0, 1]))
        if r < 0.30:
            kept.append(j)
    print(f"syntax dims: {len(SYN)} -> kept (no punct, |r|<0.30): {len(kept)}",
          flush=True)

    zs = {u: ((user_features_v2(user_sfs[u]) - tm) / ts)[kept]
          for u in user_sfs}

    rows = []
    for min_s in MIN_SENTS_GRID:
        elig = [u for u in user_sfs if len(user_sfs[u]) >= min_s]
        elig.sort(key=lambda u: -len(user_sfs[u]))
        print(f"\nmin_sents>={min_s}: eligible={len(elig)}", flush=True)
        for n_use in N_USERS_GRID:
            if len(elig) < n_use:
                continue
            users = elig[:n_use]
            Z = np.stack([zs[u] for u in users])
            # real pairwise distances (different users)
            D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
            real = D[np.triu_indices(len(users), 1)]
            # shuffled baseline: random pairings (same count)
            rng = random.Random(SEED + min_s * 100 + n_use)
            shuf = []
            for _ in range(N_SHUFFLE):
                idx = rng.sample(range(len(users)), len(users))
                d2 = np.linalg.norm(Z[idx, None, :] - Z[None, :, :],
                                    axis=-1)
                shuf.append(float(np.mean(d2[np.triu_indices(len(users), 1)])))
            shuf = np.asarray(shuf)
            sep = (float(real.mean()) - float(shuf.mean())) / \
                max(float(real.std()), 1e-9)
            rows.append({
                "min_sents": min_s, "n_users": n_use,
                "n_eligible": len(elig),
                "real_pairwise_mean": round(float(real.mean()), 4),
                "real_pairwise_std": round(float(real.std()), 4),
                "real_pairwise_min": round(float(real.min()), 4),
                "real_pairwise_max": round(float(real.max()), 4),
                "shuffled_pairwise_mean": round(float(shuf.mean()), 4),
                "shuffled_pairwise_std": round(float(shuf.std()), 4),
                "sep_vs_shuffle_sd": round(sep, 3),
            })
            print(f"  n={n_use:>2}: real {real.mean():.3f}±{real.std():.3f} "
                  f"(min {real.min():.3f}) vs shuffled {shuf.mean():.3f}"
                  f"±{shuf.std():.3f} -> sep {sep:.2f} sd", flush=True)

    with open(OUT, "w") as f:
        json.dump({"asin": ASIN, "dim": len(kept),
                   "results": rows,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
