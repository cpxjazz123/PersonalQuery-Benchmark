#!/usr/bin/env python3
"""E22 T3 — style-vector distinguishability scan on ONE product.

For a single product and each (min_sents threshold, n_users used) cell,
scan the STYLE VECTOR DIMENSIONALITY: dimensions are ranked by between-user
variance / within-user variance (split-half), then AUC / Cohen's d / rank-1
are computed on the top-D subset for D in a grid (1..318).

Answers: how many users with at least how many sentences, and how many
style-vector dimensions, are needed for significant between-user style
differences. No GPU needed.
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
from extract_syntactic_features import per_sentence_features_v2, user_features_v2, ALL_FEATS_V2

ASIN = "B0BQ1QK14T"
MIN_SENTS_GRID = [5, 10, 20, 30]
N_USERS_GRID = [5, 10, 20]
D_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 296]
N_SPLIT_REP = 12
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_style_scan_dim_syntax.json"

# ---- SYNTACTIC-ONLY dimensions: drop all punctuation features
# (punct_*, punctbg_*, n_punct_total); keep POS/dep/clause/nest/depth/opener/
# close/main/senttype/posbg/postg/depbg/clpair etc. No semantic features
# exist in the 318-dim set (all are syntactic by construction), so the only
# exclusion is punctuation.
PUNCT_PREFIXES = ("punct", "n_punct_total")
SYNTACTIC_DIMS = [i for i, n in enumerate(ALL_FEATS_V2)
                  if not n.startswith(PUNCT_PREFIXES)]
LEN_CORR_THRESH = 0.30   # |corr(feature, n_tok)| above this = length-driven
print(f"syntactic dims: {len(SYNTACTIC_DIMS)}/{len(ALL_FEATS_V2)} "
      f"(dropped {len(ALL_FEATS_V2) - len(SYNTACTIC_DIMS)} punctuation)",
      flush=True)


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
    print(f"users with sfs: {len(user_sfs)}", flush=True)

    # ---- data-driven length-correlation screen on this product's sentences
    # per-sentence value of every syntactic dim vs n_tok
    n_len = len(all_sfs)
    toklens = np.asarray([sf["n_tok"] for sf in all_sfs], dtype=np.float64)
    corr = {}
    for j in SYNTACTIC_DIMS:
        x = np.asarray([user_features_v2([sf])[j] for sf in all_sfs],
                       dtype=np.float64)
        if x.std() < 1e-12:
            corr[j] = 0.0
            continue
        corr[j] = float(np.corrcoef(x, toklens)[0, 1])
    dropped_len = [j for j in SYNTACTIC_DIMS
                   if abs(corr[j]) >= LEN_CORR_THRESH]
    kept = [j for j in SYNTACTIC_DIMS if j not in dropped_len]
    print(f"length-correlated (|r|>={LEN_CORR_THRESH}, {len(dropped_len)}): "
          f"{[ALL_FEATS_V2[j] for j in sorted(dropped_len, key=lambda j: -abs(corr[j]))][:15]}...",
          flush=True)
    print(f"kept syntactic dims (no punct, no length): {len(kept)}", flush=True)

    def zvec(u, subset=None):
        v = user_features_v2(user_sfs[u])
        if v is None:
            return None
        z = (v - tm) / ts
        return z[subset] if subset is not None else z

    z318 = {u: zvec(u) for u in user_sfs}
    z318 = {u: v for u, v in z318.items() if v is not None}
    D = len(kept)
    # keep only the surviving syntactic columns (in `kept` order)
    zsyn = {u: v[kept] for u, v in z318.items()}

    results = {}
    for min_s in MIN_SENTS_GRID:
        elig = [u for u in user_sfs if len(user_sfs[u]) >= min_s]
        elig.sort(key=lambda u: -len(user_sfs[u]))
        print(f"\nmin_sents>={min_s}: eligible={len(elig)}", flush=True)
        if not elig:
            continue
        # ---- dimension ranking on the eligible pool (between/within ratio)
        n_pool = min(25, len(elig))
        pool = elig[:n_pool]
        ratios = np.zeros(D)
        rng = random.Random(SEED + min_s)
        for j in range(D):
            between = np.var([zsyn[u][j] for u in pool])
            within_s = []
            for u in pool:
                sfs = user_sfs[u]
                vals = []
                for _ in range(6):
                    rng.shuffle(sfs)
                    h = sfs[:len(sfs) // 2]
                    v = user_features_v2(h)
                    if v is None:
                        continue
                    vals.append(((v - tm) / ts)[kept][j])
                if len(vals) >= 2:
                    within_s.append(float(np.std(vals, ddof=1)))
            within = float(np.mean(np.asarray(within_s) ** 2)) \
                if within_s else 1e-9
            ratios[j] = between / max(within, 1e-9)
        order = np.argsort(-ratios)
        top_names = [ALL_FEATS_V2[kept[j]] for j in order[:10]]
        print(f"  top-10 syntax dims: {top_names}", flush=True)

        for n_use in N_USERS_GRID:
            if len(elig) < n_use:
                continue
            users = elig[:n_use]
            us = users
            zu = np.stack([zsyn[u] for u in us])
            # split-half self vectors per user (for self-distance + ranking)
            split1: dict[str, np.ndarray] = {}
            split2: dict[str, np.ndarray] = {}
            for u in us:
                sfs = user_sfs[u]
                rng.shuffle(sfs)
                h1, h2 = sfs[:len(sfs) // 2], sfs[len(sfs) // 2:]
                v1, v2 = user_features_v2(h1), user_features_v2(h2)
                if v1 is None or v2 is None:
                    continue
                split1[u] = ((v1 - tm) / ts)[kept]
                split2[u] = ((v2 - tm) / ts)[kept]
            for Dk in D_GRID:
                dims = order[:Dk]
                cross = []
                for i in range(len(us)):
                    for j in range(i + 1, len(us)):
                        cross.append(float(np.linalg.norm(
                            zu[i, dims] - zu[j, dims])))
                self_d = []
                rank1 = 0
                n_rank = 0
                for u in us:
                    if u not in split1 or u not in split2:
                        continue
                    z1 = split1[u][dims]
                    z2 = split2[u][dims]
                    self_d.append(float(np.linalg.norm(z1 - z2)))
                    d_u = float(np.linalg.norm(z1 - zu[us.index(u), dims]))
                    d_o = min(float(np.linalg.norm(z1 - zu[i2, dims]))
                              for i2 in range(len(us))
                              if us[i2] != u)
                    rank1 += int(d_u < d_o)
                    n_rank += 1
                if not self_d or not cross:
                    continue
                self_d = np.asarray(self_d)
                cross = np.asarray(cross)
                auc = sum(float((cross > x).sum()) for x in self_d) / \
                    (len(self_d) * len(cross))
                delta = float(cross.mean() - self_d.mean())
                pooled = float(np.sqrt(
                    (cross.std(ddof=1) ** 2 + self_d.std(ddof=1) ** 2) / 2))
                cohen = delta / pooled if pooled > 0 else 0.0
                results[f"({min_s},{n_use},{Dk})"] = {
                    "min_sents": min_s, "n_users": n_use,
                    "n_eligible": len(elig), "dim": Dk,
                    "auc": round(auc, 4), "cohen_d": round(cohen, 4),
                    "rank1": round(rank1 / max(1, n_rank), 4),
                    "cross_mean": round(float(cross.mean()), 4),
                    "self_mean": round(float(self_d.mean()), 4),
                }
            row = []
            for Dk in D_GRID:
                r = results[f"({min_s},{n_use},{Dk})"]
                row.append(f"D{Dk}:A{r['auc']:.2f}")
            print(f"  n_users={n_use:>2}: " + "  ".join(row), flush=True)

    with open(OUT, "w") as f:
        json.dump({"asin": ASIN, "dim_order": order.tolist(),
                   "results": results,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
