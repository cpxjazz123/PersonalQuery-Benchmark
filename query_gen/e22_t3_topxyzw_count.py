#!/usr/bin/env python3
"""E22 T3 — grid count with 4th axis: top-w most-differentiating dims.

For each (X, Y, Z) combination: count products (sampled) whose top-X users
each have >= Y sentences of >= Z words, AND whose style vectors are
separable in the top-w between/within-ranked dims (sep = real pairwise user
distance vs shuffled-pairing baseline, >= SEP_THRESH).

Sampled: per (X,Y,Z) cell we parse at most N_SAMPLE products (spaCy is the
expensive part); the count is reported as "k of n sampled" plus a
population estimate via the full cheap (X,Y,Z) count.
"""
from __future__ import annotations

import gzip
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import TASK1_VECTORS, REVIEWS, load_meta, attrs_for_n
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import (
    per_sentence_features_v2, user_features_v2, ALL_FEATS_V2,
)

X_GRID = [5, 8, 10]
Y_GRID = [3, 5, 10]
Z_GRID = [1, 3, 5, 8, 10, 15, 20]
W_GRID = [1, 3, 5, 10, 20, 50, 100, 281]
SEP_THRESH = 0.5
N_SAMPLE = 100
N_SHUFFLE = 50
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_topxyzw_count.json"


def est_sents_minlen(texts, z):
    n = 0
    for t in texts:
        for frag in re.split(r"[.!?]+", t):
            if len(frag.split()) >= z:
                n += 1
    return n


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

    prod_users: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            t = (d.get("text") or "").strip()
            if u and a and t:
                prod_users[a][u].append(t)
            if i > 20000000:
                break
    meta = load_meta()
    print(f"products: {len(prod_users)}", flush=True)

    # ---- full cheap (X,Y,Z) counts (est sentences)
    full_counts: dict[tuple, int] = defaultdict(int)
    eligible_pool: dict[tuple, list[str]] = defaultdict(list)
    for a, uc in prod_users.items():
        if a not in meta or attrs_for_n(meta[a], 10) is None:
            continue
        for x in X_GRID:
            top = sorted((est_sents_minlen(v, min(Z_GRID))
                          for v in uc.values()), reverse=True)[:x]
            if len(top) < x:
                break
            m_all = min(top)
            for y in Y_GRID:
                if m_all < y:
                    continue
                for z in Z_GRID:
                    top_z = sorted(
                        (est_sents_minlen(v, z) for v in uc.values()),
                        reverse=True)[:x]
                    if min(top_z) >= y:
                        full_counts[(x, y, z)] += 1
                        eligible_pool[(x, y, z)].append(a)
    print(f"full (X,Y,Z) counts computed", flush=True)

    # ---- sample per cell and evaluate with spaCy features
    rng = random.Random(SEED)
    cell_results = {}
    for key, pool in eligible_pool.items():
        x, y, z = key
        # order pool by (desc) min est sents over top-x users so spaCy
        # validation has the highest pass rate (est overcounts: dots in
        # numbers/abbreviations split fragments)
        def pool_key(a):
            uc = prod_users[a]
            return min(est_sents_minlen(v, z)
                       for v in
                       [uc[u] for u, _ in sorted(
                           uc.items(), key=lambda kv: -len(kv[1]))[:x]])
        pool.sort(key=pool_key, reverse=True)
        sample = pool[:N_SAMPLE]
        sep_by_w = {w: [] for w in W_GRID}
        n_ok_feats = 0
        n_valid = 0
        for a in sample:
            uc = prod_users[a]
            top = [u for u, _ in
                   sorted(uc.items(), key=lambda kv: -len(kv[1]))[:x]]
            user_sfs = {}
            all_sfs = []
            for u in top:
                sfs = []
                for doc in nlp.pipe(uc[u], batch_size=64):
                    for s in doc.sents:
                        sf = per_sentence_features_v2(s)
                        if sf is None:
                            continue
                        if sf["n_tok"] >= z:
                            sfs.append(sf)
                if len(sfs) >= y:
                    user_sfs[u] = sfs
                    all_sfs.extend(sfs)
            if len(user_sfs) < x:
                continue
            n_valid += 1
            # per-sentence matrix + length screen
            P = np.stack([user_features_v2([sf]) for sf in all_sfs])
            toklens = np.asarray([sf["n_tok"] for sf in all_sfs],
                                 dtype=np.float64)
            kept = []
            for j in SYN:
                col = P[:, j]
                if col.std() < 1e-12:
                    kept.append(j)
                    continue
                if abs(float(np.corrcoef(col, toklens)[0, 1])) < 0.30:
                    kept.append(j)
            offsets = []
            i0 = 0
            for u in top:
                if u in user_sfs:
                    offsets.append((u, i0, i0 + len(user_sfs[u])))
                    i0 += len(user_sfs[u])
            # dimension ranking (between/within variance ratio)
            ratios = np.zeros(len(kept))
            for k, j in enumerate(kept):
                col = P[:, j]
                u_means, u_vars = [], []
                for u, a2, b2 in offsets:
                    seg = col[a2:b2]
                    if seg.size < 2:
                        continue
                    u_means.append(float(seg.mean()))
                    u_vars.append(float(seg.var(ddof=1)))
                if len(u_means) < 2:
                    continue
                between = float(np.var(u_means))
                within = float(np.mean(u_vars)) if u_vars else 1e-9
                ratios[k] = between / max(within, 1e-9)
            order = np.argsort(-ratios)
            zs = {u: ((user_features_v2(user_sfs[u]) - tm) / ts)[kept]
                  for u in user_sfs}
            users = [u for u, _, _ in offsets]
            Z = np.stack([zs[u] for u in users])
            for w in W_GRID:
                dims = order[:w]
                Dreal = np.linalg.norm(
                    Z[:, dims][:, None, :] - Z[:, dims][None, :, :],
                    axis=-1)
                real = Dreal[np.triu_indices(len(users), 1)]
                rs = random.Random(SEED + hash(a) % 10000)
                shuf = []
                for _ in range(N_SHUFFLE):
                    idx = rs.sample(range(len(users)), len(users))
                    d2 = np.linalg.norm(
                        Z[idx][:, dims][:, None, :] -
                        Z[:, dims][None, :, :], axis=-1)
                    shuf.append(float(
                        np.mean(d2[np.triu_indices(len(users), 1)])))
                shuf = np.asarray(shuf)
                sep = (float(real.mean()) - float(shuf.mean())) / \
                    max(float(real.std()), 1e-9)
                sep_by_w[w].append(sep)
        n_sampled = n_valid
        cell_results[key] = {
            "n_eligible_full": full_counts[key],
            "n_sampled": n_sampled,
            "sep_by_w": {str(w): {
                "n_sep": sum(1 for s in sep_by_w[w] if s >= SEP_THRESH),
                "mean_sep": round(float(np.mean(sep_by_w[w])), 3) if
                sep_by_w[w] else None,
                "max_sep": round(float(np.max(sep_by_w[w])), 3) if
                sep_by_w[w] else None,
            } for w in W_GRID},
        }
        line = "  ".join(
            f"w{w}:{cell_results[key]['sep_by_w'][str(w)]['n_sep']}"
            f"/{n_sampled}" for w in W_GRID)
        print(f"(X{x},Y{y},Z{z}): eligible={full_counts[key]} "
              f"sampled={n_sampled} | {line}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"x_grid": X_GRID, "y_grid": Y_GRID, "z_grid": Z_GRID,
                   "w_grid": W_GRID, "sep_thresh": SEP_THRESH,
                   "cells": {f"({x},{y},{z})": v
                             for (x, y, z), v in cell_results.items()},
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
