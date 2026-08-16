#!/usr/bin/env python3
"""E22 T3 — style-vector distinguishability scan on ONE product.

For a single product: for each min-sentence threshold, how many users have
that many review sentences ON THIS PRODUCT, and do their style vectors form
significant between-user differences?

Metrics per (min_sents, n_users_used):
  - cross distance: full-vector L2 between different users
  - self distance:  split-half L2 within the same user (repeated)
  - AUC(self vs cross), Cohen's d, and rank-1 (own half matched back)
Reported for the 7-dim Query-compatible subset AND the full 318-dim.
No GPU needed.
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
MIN_SENTS_GRID = [5, 10, 15, 20, 30, 40]
N_USERS_GRID = [5, 8, 10, 15, 20]
N_SPLIT_REP = 12
SEED = 1234

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_style_scan.json"


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
    for u, texts in prod_revs.items():
        sfs = []
        for doc in nlp.pipe(texts, batch_size=64):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        if sfs:
            user_sfs[u] = sfs
    print(f"users with sfs: {len(user_sfs)}; "
          f"top sfs: {sorted((len(v) for v in user_sfs.values()), reverse=True)[:10]}",
          flush=True)

    def zvec(u, subset=None):
        v = user_features_v2(user_sfs[u])
        if v is None:
            return None
        z = (v - tm) / ts
        return z[subset] if subset is not None else z

    # cache full 7-dim and 318-dim vectors for all users
    z7_full = {u: zvec(u, fidx) for u in user_sfs}
    z318_full = {u: zvec(u, None) for u in user_sfs}
    z7_full = {u: v for u, v in z7_full.items() if v is not None}
    z318_full = {u: v for u, v in z318_full.items() if v is not None}

    results = {}
    for min_s in MIN_SENTS_GRID:
        elig = [u for u in user_sfs if len(user_sfs[u]) >= min_s]
        elig.sort(key=lambda u: -len(user_sfs[u]))
        print(f"\nmin_sents>={min_s}: eligible={len(elig)}", flush=True)
        for n_use in N_USERS_GRID:
            if len(elig) < n_use:
                continue
            users = elig[:n_use]
            for dim_name, zfull in (("dim7", z7_full),
                                    ("dim318", z318_full)):
                zu = {u: zfull[u] for u in users if u in zfull}
                if len(zu) < 4:
                    continue
                us = list(zu)
                # cross distances (full vs full, standardized dims)
                cross = []
                for i in range(len(us)):
                    for j in range(i + 1, len(us)):
                        cross.append(float(np.linalg.norm(zu[us[i]] - zu[us[j]])))
                # self distances (split-half, repeated)
                self_d = []
                rank1 = 0
                n_rank = 0
                rng = random.Random(SEED + min_s * 100 + n_use)
                for u in us:
                    sfs = user_sfs[u]
                    for _ in range(N_SPLIT_REP):
                        rng.shuffle(sfs)
                        h1 = sfs[:len(sfs) // 2]
                        h2 = sfs[len(sfs) // 2:]
                        v1 = user_features_v2(h1)
                        v2 = user_features_v2(h2)
                        if v1 is None or v2 is None:
                            continue
                        z1 = ((v1 - tm) / ts)
                        z2 = ((v2 - tm) / ts)
                        if dim_name == "dim7":
                            z1, z2 = z1[fidx], z2[fidx]
                        self_d.append(float(np.linalg.norm(z1 - z2)))
                        # rank-1: is z1 closer to z_u(full) than other users'?
                        d_u = float(np.linalg.norm(z1 - zu[u]))
                        d_o = min(float(np.linalg.norm(z1 - zu[o]))
                                  for o in us if o != u)
                        if d_u < d_o:
                            rank1 += 1
                        n_rank += 1
                if not self_d or not cross:
                    continue
                self_d = np.asarray(self_d)
                cross = np.asarray(cross)
                auc = 0.0
                for x in self_d:
                    auc += float((cross > x).sum())
                auc = auc / (len(self_d) * len(cross))
                delta = float(cross.mean() - self_d.mean())
                pooled = float(np.sqrt(
                    (cross.std(ddof=1) ** 2 + self_d.std(ddof=1) ** 2) / 2))
                cohen = delta / pooled if pooled > 0 else 0.0
                key = (min_s, n_use, dim_name)
                results[f"{key}"] = {
                    "min_sents": min_s, "n_users_used": n_use,
                    "n_eligible": len(elig), "dim": dim_name,
                    "n_cross": len(cross), "n_self": len(self_d),
                    "auc": round(auc, 4), "cohen_d": round(cohen, 4),
                    "rank1": round(rank1 / max(1, n_rank), 4),
                    "cross_mean": round(float(cross.mean()), 4),
                    "self_mean": round(float(self_d.mean()), 4),
                }
                print(f"  n={n_use:>2} {dim_name:>6}: "
                      f"AUC={auc:.3f} d={cohen:.3f} rank1={rank1 / max(1, n_rank):.3f}",
                      flush=True)

    with open(OUT, "w") as f:
        json.dump({"asin": ASIN, "results": results,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"\nwrote {OUT}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
