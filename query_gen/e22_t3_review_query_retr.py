#!/usr/bin/env python3
"""E22 T3 — review-syntax-as-query: hit@10 diff across users' REAL review
sentences (perfect style transfer by construction, no generation).

For each product (X=5,Y=5,Z=5): use each of the top-5 users' real review
sentences as retrieval queries (same product -> same attrs; reviews are the
user's own syntax). BGE retrieval over same-category nearest-neighbour
corpus; hit@10 = target product in top-10. Report per-user hit@10 and the
max-min diff across the 5 users' review styles. Also compare review-as-query
vs template-as-query vs generated-query hit@10.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import (
    REVIEWS, load_meta, attrs_for_n, TASK1_VECTORS, TEMPLATES,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_review_query_retr.json"
SEED = 7777
N_PRODUCTS = 15
N_ATTRS = 9
N_USERS = 5
N_SENTS_PER_USER = 8
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


def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
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
        uc = prod_users[asin]
        top = [u for u, _ in
               sorted(uc.items(), key=lambda kv: -len(kv[1]))[:15]]
        user_sents: dict[str, list[str]] = {}
        for u in top:
            sents = []
            for doc in nlp.pipe(uc[u][:200], batch_size=64):
                for s in doc.sents:
                    if len([t for t in s if not t.is_space]) >= 5:
                        sents.append(s.text.strip())
            if len(sents) >= 5:
                user_sents[u] = sents
        users = [u for u in top if u in user_sents][:N_USERS]
        if len(users) < N_USERS:
            continue
        # corpus: target + BGE nearest-neighbour same-category decoys
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

        # review sentences as queries
        hits_by_user = {}
        coverage_by_user = {}
        per_sent = []  # (hit, n_attrs_covered, rank)
        for u in users:
            sents = user_sents[u][:N_SENTS_PER_USER]
            hs = []
            covs = []
            for s in sents:
                rk = retr(s)
                h = 1 if rk <= TOP_K else 0
                hs.append(h)
                cov = sum(1 for a in attrs.values()
                          if str(a).lower() in s.lower())
                covs.append(cov)
                per_sent.append((h, cov, rk))
            hits_by_user[u] = hs
            coverage_by_user[u] = covs
        means = [float(np.mean(h)) for h in hits_by_user.values()]
        diff = float(np.max(means) - np.min(means))
        # coverage-stratified diff: within review queries of >=k attr coverage
        strat = {}
        for k in (0, 1, 2, 3):
            sel = [x for x in per_sent if x[1] >= k]
            if len(sel) >= 3:
                strat[k] = {
                    "n": len(sel),
                    "hit_rate": round(float(np.mean([x[0] for x in sel])), 3),
                    "mean_coverage": round(float(np.mean([x[1] for x in sel])), 2),
                }
        corr = float(np.corrcoef([x[1] for x in per_sent],
                                 [x[0] for x in per_sent])[0, 1]) \
            if len(per_sent) >= 3 else None
        # template query as control
        sv = [attrs[k] for k in ("Brand", "Color", "Material", "Category",
                                 "Price")]
        tmpl_hits = []
        for tpl in TEMPLATES:
            q = tpl.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])
            tmpl_hits.append(1 if retr(q) <= TOP_K else 0)
        per_product.append({
            "asin": asin,
            "review_hit10_by_user": {u: hits_by_user[u] for u in users},
            "review_hit10_mean_by_user": {u: round(float(np.mean(h)), 3)
                                          for u, h in hits_by_user.items()},
            "review_hit10_diff": round(diff, 4),
            "coverage_by_user": {u: coverage_by_user[u] for u in users},
            "coverage_stratified": strat,
            "hit_cov_corr": round(corr, 3) if corr is not None else None,
            "template_hit10": tmpl_hits,
            "template_hit10_mean": round(float(np.mean(tmpl_hits)), 3),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: review-hit@10 diff="
            f"{diff:.3f} means={[round(m, 2) for m in means]} "
            f"cov_corr={corr:.2f} strat={strat} "
            f"template={np.mean(tmpl_hits):.2f} "
            f"({time.time() - p0:.0f}s)")

    diffs = [p["review_hit10_diff"] for p in per_product]
    log("\n=== aggregate ===")
    log(f"review-as-query hit@10 diff: mean={np.mean(diffs):.3f} "
        f"min={np.min(diffs):.3f} max={np.max(diffs):.3f}")
    log(f"nonzero-diff products: "
        f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}")

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "aggregate": {
                       "review_hit10_diff_mean": round(float(np.mean(diffs)), 4),
                       "review_hit10_diff_min": round(float(np.min(diffs)), 4),
                       "review_hit10_diff_max": round(float(np.max(diffs)), 4),
                       "n_nonzero": sum(1 for d in diffs if d > 0),
                       "n_products": len(diffs),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
