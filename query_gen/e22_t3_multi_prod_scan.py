#!/usr/bin/env python3
"""E22 T3 — per-product (n_users x dims) optimal-combination scan.

For each of many products: within the product, scan
  n_users in {5..10} x top-D syntactic non-length dims (D grid),
measure sep = (real pairwise user distance - shuffled baseline) / real std,
record the OPTIMAL (n_users, D) per product. Then compare across products:
does the optimal combination vary per product, or is it stable?

Kept dims per product: syntactic (no punct) minus |r(n_tok)|>=0.30,
computed on that product's own sentences.
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

from query_gen_main import TASK1_VECTORS, REVIEWS, load_meta, attrs_for_n
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import (
    per_sentence_features_v2, user_features_v2, ALL_FEATS_V2,
)

N_PRODUCTS = 30
MIN_EST_SENTS = 3
N_USERS_GRID = [5, 6, 7, 8, 9, 10]
D_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 281]
N_SHUFFLE = 100
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_multi_prod_scan.json"


def est_sents(texts):
    import re
    return sum(len(re.findall(r"[.!?]+", t)) for t in texts)


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
    cands = [a for a, uc in prod_users.items()
             if len(uc) >= 15 and a in meta
             and attrs_for_n(meta[a], 10) is not None]
    rng = random.Random(SEED)
    rng.shuffle(cands)
    # prefilter: top-10 users each >= MIN_EST_SENTS est sents
    pool = [a for a in cands
            if all(est_sents(prod_users[a][u]) >= MIN_EST_SENTS
                   for u in [x for x, _ in sorted(
                       prod_users[a].items(),
                       key=lambda kv: -len(kv[1]))[:10]])]
    print(f"candidate products: {len(pool)}", flush=True)
    pool = pool[:N_PRODUCTS]

    per_product = []
    for pi, asin in enumerate(pool):
        p0 = time.time()
        uc = prod_users[asin]
        top = [u for u, _ in
               sorted(uc.items(), key=lambda kv: -len(kv[1]))[:15]]
        user_sfs: dict[str, list[dict]] = {}
        all_sfs: list[dict] = []
        for u in top:
            sfs = []
            for doc in nlp.pipe(uc[u], batch_size=64):
                for s in doc.sents:
                    sf = per_sentence_features_v2(s)
                    if sf is not None:
                        sfs.append(sf)
            if sfs:
                user_sfs[u] = sfs
                all_sfs.extend(sfs)
        if len(user_sfs) < max(N_USERS_GRID):
            continue
        # per-sentence matrix + length screen on THIS product's sentences
        P = np.stack([user_features_v2([sf]) for sf in all_sfs])
        toklens = np.asarray([sf["n_tok"] for sf in all_sfs],
                             dtype=np.float64)
        kept = []
        for j in SYN:
            x = P[:, j]
            if x.std() < 1e-12:
                kept.append(j)
                continue
            if abs(float(np.corrcoef(x, toklens)[0, 1])) < 0.30:
                kept.append(j)
        # sentence row ranges per user (all_sfs appended per user in order)
        offsets = []
        i0 = 0
        for u in top:
            if u in user_sfs:
                n = len(user_sfs[u])
                offsets.append((u, i0, i0 + n))
                i0 += n
        # per-user vectors on kept dims
        zs = {u: ((user_features_v2(user_sfs[u]) - tm) / ts)[kept]
              for u in user_sfs}
        # dimension ranking: between/within variance ratio
        ratios = np.zeros(len(kept))
        for k, j in enumerate(kept):
            col = P[:, j]
            u_means, u_vars = [], []
            for u, a, b in offsets:
                seg = col[a:b]
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
        # grid scan
        best = None
        grid_rows = []
        for n_use in N_USERS_GRID:
            users = [u for u, _, _ in offsets[:n_use]]
            Z = np.stack([zs[u] for u in users])
            for Dk in D_GRID:
                dims = order[:Dk]
                Dreal = np.linalg.norm(
                    Z[:, dims][:, None, :] - Z[:, dims][None, :, :],
                    axis=-1)
                real = Dreal[np.triu_indices(len(users), 1)]
                rs = random.Random(SEED + pi * 1000 + n_use * 100 + Dk)
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
                grid_rows.append({"n_users": n_use, "dim": Dk,
                                  "sep_sd": round(sep, 3)})
                if best is None or sep > best["sep_sd"]:
                    best = {"n_users": n_use, "dim": Dk,
                            "sep_sd": round(sep, 3),
                            "n_eligible": len(user_sfs),
                            "top_dims": [ALL_FEATS_V2[kept[j]]
                                         for j in order[:5]]}
        per_product.append({"asin": asin, "best": best,
                            "n_sfs": [len(user_sfs[u])
                                      for u, _, _ in offsets]})
        print(f"[{pi + 1}/{len(pool)}] {asin}: best n={best['n_users']} "
              f"D={best['dim']} sep={best['sep_sd']:+.2f} "
              f"top={best['top_dims'][:3]} "
              f"({time.time() - p0:.0f}s)", flush=True)

    # ---- cross-product summary
    print("\n=== cross-product optimal combinations ===", flush=True)
    from collections import Counter
    cn = Counter(p["best"]["n_users"] for p in per_product)
    cd = Counter(p["best"]["dim"] for p in per_product)
    print(f"best n_users: {dict(sorted(cn.items()))}", flush=True)
    print(f"best dim: {dict(sorted(cd.items()))}", flush=True)
    seps = [p["best"]["sep_sd"] for p in per_product]
    print(f"best sep: mean={np.mean(seps):.2f} min={min(seps):.2f} "
          f"max={max(seps):.2f}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "n_users_hist": dict(sorted(cn.items())),
                   "dim_hist": dict(sorted(cd.items())),
                   "sep_stats": {"mean": round(float(np.mean(seps)), 3),
                                 "min": round(float(min(seps)), 3),
                                 "max": round(float(max(seps)), 3)},
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"wrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
