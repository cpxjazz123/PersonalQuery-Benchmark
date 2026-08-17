#!/usr/bin/env python3
"""query_gen_same_prod_style — 同商品多用户表达风格差异分析.

For each of the 4,206 products bought by >=10 users (with all 10 attributes),
take the 10 users with the most reviews ON THAT PRODUCT; compute each user's
syntactic style vector (318-dim) from their reviews of THAT product; measure
the BETWEEN-user style difference.

Metrics (per product, over its top-10 users):
  - mean pairwise L2 distance (318-dim, standardized space)
  - mean pairwise cosine distance
  - coefficient of variation (std/mean) of style norms
  - whether user style differs MORE than repeated draws of the same user
    (stability baseline: same-user split-half cosine)
  - style entropy / variance decomposition: within-user vs between-user

Reports aggregate over all products + a sample table.

Outputs: result/e22_t3/e22_t3_same_prod_style.json
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

from query_gen_main import (  # noqa: E402
    load_meta, attrs_for_n, neutralize_content, spacy318, TOP_ATTR_KEYS,
    EXT_ATTR_KEYS,
)
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_same_prod_style.json"
SEED = 4242
N_TOP_USERS = 10
MIN_REVIEWS_PER_USER = 1
MAX_PRODUCTS = 4206
N_SAMPLE_PRINT = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    meta = load_meta()
    log("indexing reviews (user, product)...")
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
            text = d.get("text") or ""
            if u and a and text.strip():
                prod_users[a][u].append(text.strip())
            if i > 20000000:
                break
    log(f"indexed {len(prod_users)} products")

    # filter: products with >=10 users AND all 10 attributes
    cands = []
    for a, uc in prod_users.items():
        if len(uc) >= N_TOP_USERS and a in meta and \
                attrs_for_n(meta[a], 10) is not None:
            cands.append(a)
    log(f"eligible products (>=10 users, 10 attrs): {len(cands)}")
    if not cands:
        return
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands = cands[:MAX_PRODUCTS]

    agg = {
        "pairwise_l2": [], "pairwise_cos": [], "within_cos": [],
        "n_users_per_prod": [], "norm_cv": [],
        "style_span": [],   # max pairwise L2 (10 users)
    }
    per_prod = []
    n_prod_done = 0
    for a in cands:
        uc = prod_users[a]
        top_users = [u for u, _ in
                     sorted(uc.items(), key=lambda kv: -len(kv[1]))[:N_TOP_USERS]]
        # per-user 318-dim vector from their reviews of THIS product
        user_vecs = []
        user_sf = []
        for u in top_users:
            texts = uc[u][:30]
            sfs = []
            for t in texts:
                doc = nlp(t)
                for s in doc.sents:
                    sf = per_sentence_features_v2(s)
                    if sf is not None:
                        sfs.append(sf)
            if not sfs:
                continue
            v = user_features_v2(sfs)
            if v is None:
                continue
            user_vecs.append(((v - tm) / ts).astype(np.float64))
            user_sf.append(sfs)
        if len(user_vecs) < 5:
            continue
        V = np.stack(user_vecs)
        # pairwise L2
        D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
        tri = D[np.triu_indices(len(V), 1)]
        # pairwise cosine
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        C = (V @ V.T) / (norms @ norms.T)
        ctri = C[np.triu_indices(len(V), 1)]
        # within-user stability: split each user's sentences in half, cosine
        within = []
        for sfs in user_sf:
            if len(sfs) < 6:
                continue
            half = len(sfs) // 2
            v1 = user_features_v2(sfs[:half])
            v2 = user_features_v2(sfs[half:2 * half])
            if v1 is None or v2 is None:
                continue
            z1 = (v1 - tm) / ts
            z2 = (v2 - tm) / ts
            within.append(float((z1 @ z2) / (
                np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-9)))
        agg["pairwise_l2"].extend(tri.tolist())
        agg["pairwise_cos"].extend(ctri.tolist())
        agg["within_cos"].extend(within)
        agg["n_users_per_prod"].append(len(V))
        agg["norm_cv"].append(float(norms[:, 0].std() /
                                    (norms[:, 0].mean() + 1e-9)))
        agg["style_span"].append(float(tri.max()))
        per_prod.append({
            "asin": a, "n_users": len(V),
            "mean_pairwise_l2": round(float(tri.mean()), 4),
            "mean_pairwise_cos": round(float(ctri.mean()), 4),
            "span_l2": round(float(tri.max()), 4),
            "within_cos_mean": round(float(np.mean(within)), 4)
            if within else None,
        })
        n_prod_done += 1
        if n_prod_done % 500 == 0:
            log(f"  {n_prod_done}/{len(cands)} products done")

    result = {
        "version": "query_gen_same_prod_style_v1",
        "seed": SEED,
        "n_products_analyzed": n_prod_done,
        "n_top_users": N_TOP_USERS,
        "aggregate": {
            "mean_pairwise_l2": round(float(np.mean(agg["pairwise_l2"])), 4),
            "median_pairwise_l2": round(float(np.median(agg["pairwise_l2"])), 4),
            "mean_pairwise_cos": round(float(np.mean(agg["pairwise_cos"])), 4),
            "median_pairwise_cos": round(float(np.median(agg["pairwise_cos"])), 4),
            "within_user_cos_mean": round(float(np.mean(agg["within_cos"])), 4)
            if agg["within_cos"] else None,
            "n_users_per_prod_mean": round(float(np.mean(agg["n_users_per_prod"])), 2),
            "norm_cv_mean": round(float(np.mean(agg["norm_cv"])), 4),
            "style_span_mean": round(float(np.mean(agg["style_span"])), 4),
        },
        "sample_products": per_prod[:N_SAMPLE_PRINT],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    log("done")
    a = result["aggregate"]
    log(f"AGG: pairwise_l2={a['mean_pairwise_l2']:.3f} "
        f"cos={a['mean_pairwise_cos']:.3f} "
        f"within_cos={a['within_user_cos_mean']} "
        f"norm_cv={a['norm_cv_mean']:.3f} span={a['style_span_mean']:.3f}")


if __name__ == "__main__":
    main()
