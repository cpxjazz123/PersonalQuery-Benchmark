#!/usr/bin/env python3
"""E22 T3 — controlled pure-syntax retrieval test: SAME content words + SAME
attribute values, only syntactic arrangement differs (word order / relative
clause / passive / fronting). Isolates the effect of syntax on hit@10 with
no generation confounds.

Per product (X=5/Y=5/Z=5/w=20): 5 users -> 5 syntax variants built from
templates that preserve all 9 attr values verbatim; BGE retrieval over the
same-category nearest-neighbour corpus; hit@10 per variant and diff across
the 5 syntax variants.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))

from query_gen_main import REVIEWS, load_meta, attrs_for_n

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_pure_syntax.json"
SEED = 7777
N_PRODUCTS = 15
N_ATTRS = 9
N_USERS = 5
TOP_K = 10
N_CORPUS = 200
BGE_PATH = ("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
            "models--BAAI--bge-base-en-v1.5/snapshots/" +
            os.listdir("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
                       "models--BAAI--bge-base-en-v1.5/snapshots/")[0])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def est_sents_minlen(texts, z):
    n = 0
    for t in texts:
        for frag in re.split(r"[.!?]+", t):
            if len(frag.split()) >= z:
                n += 1
    return n


def make_variants(attrs):
    """Five syntax variants with identical attribute values/content words.
    Only syntactic arrangement differs."""
    B = attrs.get("Brand", "")
    C = attrs.get("Category", "")
    Co = attrs.get("Color", "")
    M = attrs.get("Material", "")
    P = attrs.get("Price", "")
    S = attrs.get("Style", "")
    A = attrs.get("Age Range (Description)", "")
    N = attrs.get("Number Of Items", "")
    Sz = attrs.get("Size", "")
    G = attrs.get("Target gender", "")
    Care = attrs.get("Product Care Instructions", "")
    v = [x for x in (B, C, Co, M, P, S, A, N, Sz, G, Care) if x]
    def j(*parts):
        return " ".join(p for p in parts if p)
    return [
        # 1. declarative SVO
        j("I want", "a", B, C, "in", Co, "made of", M, "at", P, ",", S,
          "for", A, ",", N, "items", ",", Sz, ",", G, ",", Care, "."),
        # 2. fronted attribute focus
        j("The", C, "with", B, "brand,", Co, "color and", M, "material",
          "is what I need, priced at", P, ",", S, "style,", A, "age,",
          N, "items,", Sz, ",", G, ",", Care, "."),
        # 3. relative-clause apposition
        j("The product that I want is", "a", B, C, "which comes in", Co,
          "and", M, ", costs", P, ", has", S, "style, suits", A, ",",
          "includes", N, "items,", Sz, ",", G, ", and requires", Care, "."),
        # 4. passive/impersonal
        j("It should be", "a", B, C, "that is", Co, "and made from", M,
          "with a price of", P, "; the", S, "style is preferred for", A,
          ", with", N, "items,", Sz, ",", G, "and", Care, "."),
        # 5. interrogative
        j("Can you find me", "a", B, C, "in", Co, "with", M, "material,",
          "priced under", P, ", in", S, "style, suitable for", A, ",",
          "with", N, "pieces,", Sz, ",", G, "and", Care, "?"),
    ]


def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    meta = load_meta()
    meta_full: dict[str, dict] = {}
    with gzip.open(REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz",
                   "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pa = d.get("parent_asin")
            if pa and d.get("title"):
                meta_full.setdefault(pa, d)
    def title_of(a):
        d = meta_full.get(a)
        return d.get("title") if d else None

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
    cands = []
    for a, uc in prod_users.items():
        if a not in meta or attrs_for_n(meta[a], N_ATTRS) is None:
            continue
        if not title_of(a):
            continue
        top = [u for u, _ in
               sorted(uc.items(), key=lambda kv: -len(kv[1]))[:5]]
        if len(top) < 5:
            continue
        if all(est_sents_minlen(uc[u], 5) >= 5 for u in top):
            cands.append(a)
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands = cands[:N_PRODUCTS]
    log(f"products: {len(cands)}")

    btok = AutoTokenizer.from_pretrained(BGE_PATH)
    bmodel = AutoModel.from_pretrained(BGE_PATH)
    bmodel.eval()
    bmodel.to("cuda:0")

    def encode(texts, bs=64):
        vecs = []
        for i in range(0, len(texts), bs):
            e = btok(texts[i:i + bs], return_tensors="pt", padding=True,
                     truncation=True, max_length=128).to("cuda:0")
            with torch.no_grad():
                o = bmodel(**e)
            m = e["attention_mask"].unsqueeze(-1)
            v = (o.last_hidden_state * m).sum(1) / m.sum(1)
            vecs.append(F.normalize(v, dim=1).cpu().numpy())
        return np.concatenate(vecs, axis=0)

    per_product = []
    for pi, asin in enumerate(cands):
        p0 = time.time()
        attrs = attrs_for_n(meta[asin], N_ATTRS)
        variants = make_variants(attrs)
        # corpus
        same_cat = [a2 for a2, rec in meta.items()
                    if a2 != asin and
                    any(k in (rec.get("category_leaf") or "")
                        for k in ("Cover", "Diaper", "Protection", "Incont",
                                  "Crib", "Pad", "Changing"))
                    and attrs_for_n(rec, N_ATTRS) is not None
                    and title_of(a2)]
        tgt_title = title_of(asin)
        if len(same_cat) >= N_CORPUS - 1:
            cvecs = encode([tgt_title] + [title_of(a) for a in same_cat])
            sims = cvecs[0] @ cvecs[1:].T
            nn = np.argsort(-sims)[:N_CORPUS - 1]
            decoys = [same_cat[i] for i in nn]
        else:
            rng2 = random.Random(SEED + pi)
            rng2.shuffle(same_cat)
            decoys = same_cat[:N_CORPUS - 1]
        corpus = [asin] + decoys
        doc_vecs = encode([title_of(a) for a in corpus])
        target_idx = 0

        def retr(qtext):
            qv = encode([qtext])
            sims = qv @ doc_vecs.T
            order = np.argsort(-sims[0])
            rk = int(np.where(order == target_idx)[0][0]) + 1
            return rk

        hits = []
        ranks = []
        for vq in variants:
            rk = retr(vq)
            hits.append(1 if rk <= TOP_K else 0)
            ranks.append(rk)
        diff = float(np.max(hits) - np.min(hits))
        rdiff = float(np.max(ranks) - np.min(ranks))
        per_product.append({
            "asin": asin,
            "variants": variants,
            "hits": hits, "ranks": ranks,
            "hit10_diff": round(diff, 4),
            "rank_diff": round(rdiff, 4),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: hit10={hits} ranks={ranks} "
            f"hit_diff={diff:.2f} rank_diff={rdiff:.1f} "
            f"({time.time() - p0:.0f}s)")

    diffs = [p["hit10_diff"] for p in per_product]
    rdiffs = [p["rank_diff"] for p in per_product]
    log("\n=== aggregate ===")
    log(f"pure-syntax hit@10 diff: mean={np.mean(diffs):.3f} "
        f"min={np.min(diffs):.3f} max={np.max(diffs):.3f}")
    log(f"pure-syntax rank diff: mean={np.mean(rdiffs):.2f} "
        f"max={np.max(rdiffs):.2f}")
    log(f"nonzero hit diff products: "
        f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}")

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "aggregate": {
                       "hit10_diff_mean": round(float(np.mean(diffs)), 4),
                       "hit10_diff_min": round(float(np.min(diffs)), 4),
                       "hit10_diff_max": round(float(np.max(diffs)), 4),
                       "rank_diff_mean": round(float(np.mean(rdiffs)), 2),
                       "n_nonzero": sum(1 for d in diffs if d > 0),
                       "n_products": len(diffs),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
