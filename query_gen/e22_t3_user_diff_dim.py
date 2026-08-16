#!/usr/bin/env python3
"""E22 T3 — user-difference scan over DIMENSIONALITY (simple protocol).

Same product, same simple protocol (real pairwise user distance vs shuffled
pairing baseline, no split-half/AUC), but D is now a variable: syntactic
non-length dims are ranked by between-user/within-user variance ratio, then
the top-D subset is evaluated for D in a grid. Reports which D separates
users best and which individual dims rank highest.
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
D_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 281]
N_SHUFFLE = 200
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_user_diff_dim.json"


def main() -> None:
    t0 = time.time()
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

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

    cache = REPO_ROOT / "result" / "cache" / "e22_t3_sentvecs.npz"
    cc = np.load(cache, allow_pickle=True)
    P = cc["P"]
    print(f"sentence matrix: {P.shape}", flush=True)
    if P.shape[0] != len(all_sfs):
        print("cache size mismatch — rebuild needed")
        return

    # length-correlated screen (kept: constant-at-sentence-level or |r|<0.30)
    toklens = np.asarray([sf["n_tok"] for sf in all_sfs], dtype=np.float64)
    kept = []
    for j in SYN:
        x = P[:, j]
        if x.std() < 1e-12:
            kept.append(j)
            continue
        if abs(float(np.corrcoef(x, toklens)[0, 1])) < 0.30:
            kept.append(j)
    print(f"kept syntactic non-length dims: {len(kept)}", flush=True)

    # ---- per-user aggregated standardized vectors (kept dims)
    zs = {u: ((user_features_v2(user_sfs[u]) - tm) / ts)[kept]
          for u in user_sfs}

    # ---- dimension ranking: between/within variance ratio (per-user mean
    # variance of sentence-level values as within; variance of user means as
    # between). Sentence rows per user via re-index (cache lacks mapping, so
    # recompute a lightweight sentence->user map from all_sfs order: we can't
    # rely on order, so do it via user_features per sentence is too slow —
    # instead approximate within-variance using user-level aggregation of
    # sentence values: recompute sentence rows per user by re-parsing user
    # sentence lists into P-index is unavailable; use mean-variance proxy:
    # within_j = mean over users of var(user's sentence values) requires
    # per-sentence rows per user. We stored them in the cache as flat P, so
    # rebuild the mapping by re-running the per-user sentence parse ONLY for
    # the aggregate call; sentence rows are unavailable -> use a fast proxy:
    #   within_j ~ var of user means x residual from per-sentence P grouped
    # by re-splitting all_sfs back per user is impossible (order mixed).
    # Proxy: treat sentence rows as grouped by user in all_sfs order (we
    # appended per user sequentially, so the ORDER IS per-user grouped!).
    # Recover per-user row ranges by counting sentences per user in the
    # same order.
    offsets = []
    i0 = 0
    for u in user_sfs:
        n = len(user_sfs[u])
        offsets.append((u, i0, i0 + n))
        i0 += n
    ratios = np.zeros(len(kept))
    for k, j in enumerate(kept):
        col = P[:, j]
        u_means = []
        u_vars = []
        for u, a, b in offsets:
            seg = col[a:b]
            if seg.size < 2:
                continue
            u_means.append(float(seg.mean()))
            u_vars.append(float(seg.var(ddof=1)))
        if len(u_means) < 2:
            ratios[k] = 0.0
            continue
        between = float(np.var(u_means))
        within = float(np.mean(u_vars)) if u_vars else 1e-9
        ratios[k] = between / max(within, 1e-9)
    order = np.argsort(-ratios)
    print("top-15 dims:", [ALL_FEATS_V2[kept[j]] for j in order[:15]],
          flush=True)

    results = {}
    for min_s in MIN_SENTS_GRID:
        elig = [u for u in user_sfs if len(user_sfs[u]) >= min_s]
        elig.sort(key=lambda u: -len(user_sfs[u]))
        print(f"\nmin_sents>={min_s}: eligible={len(elig)}", flush=True)
        for n_use in N_USERS_GRID:
            if len(elig) < n_use:
                continue
            users = elig[:n_use]
            Z = np.stack([zs[u] for u in users])
            rows_out = []
            for Dk in D_GRID:
                dims = order[:Dk]
                Dfull = np.linalg.norm(Z[:, None, :] - Z[None, :, :],
                                       axis=-1)
                Dsub = np.linalg.norm(Z[:, dims][:, None, :] -
                                      Z[:, dims][None, :, :], axis=-1)
                real = Dsub[np.triu_indices(len(users), 1)]
                rng = random.Random(SEED + min_s * 100 + n_use + Dk)
                shuf = []
                for _ in range(N_SHUFFLE):
                    idx = rng.sample(range(len(users)), len(users))
                    d2 = np.linalg.norm(Z[idx][:, dims][:, None, :] -
                                        Z[:, dims][None, :, :], axis=-1)
                    shuf.append(float(
                        np.mean(d2[np.triu_indices(len(users), 1)])))
                shuf = np.asarray(shuf)
                sep = (float(real.mean()) - float(shuf.mean())) / \
                    max(float(real.std()), 1e-9)
                rows_out.append({
                    "dim": Dk,
                    "real_mean": round(float(real.mean()), 4),
                    "real_std": round(float(real.std()), 4),
                    "shuf_mean": round(float(shuf.mean()), 4),
                    "shuf_std": round(float(shuf.std()), 4),
                    "sep_sd": round(sep, 3),
                })
                results[f"({min_s},{n_use},{Dk})"] = rows_out[-1]
            line = "  ".join(
                f"D{r['dim']}:{r['sep_sd']:+.2f}" for r in rows_out)
            print(f"  n={n_use:>2}: {line}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"asin": ASIN, "dim_order": [ALL_FEATS_V2[kept[j]]
                                               for j in order],
                   "results": results,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
