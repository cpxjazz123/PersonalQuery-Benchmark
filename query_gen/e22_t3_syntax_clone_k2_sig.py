#!/usr/bin/env python3
"""E22 T3 — SYNTAX-CLONE generation: build the query by cloning a user's
review sentence skeleton (function words / clause boundaries / punctuation /
POS scaffolding kept), swapping content slots for attribute values. The query
syntax IS the user's review syntax BY CONSTRUCTION.

Pipeline per user:
  1. pick the review sentence closest to the user's 20-dim syntax centroid
  2. spaCy-parse it; keep scaffolding tokens (function/POS anchors), blank
     content spans
  3. fill content slots with attribute values (nouns -> Category/Brand,
     adjectives -> Color/Material/Style, numerals -> Price/Items/Size)
  4. evaluate self-match (query 20-dim vs user review centroid) and
     hit@10 diff across the 5 users' cloned queries.
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
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import (
    REVIEWS, load_meta, attrs_for_n, TASK1_VECTORS,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import (
    per_sentence_features_v2, user_features_v2, ALL_FEATS_V2,
)

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_syntax_clone_k2_sig.json"
SEED = 7777
N_PRODUCTS = 20
N_ATTRS = 9
N_USERS = 5
TOP_K = 10
N_CORPUS = 200
PUNCT_PREFIXES = ("punct", "n_punct_total")
SYN = [i for i, n in enumerate(ALL_FEATS_V2)
       if not n.startswith(PUNCT_PREFIXES)]
BGE_PATH = ("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
            "models--BAAI--bge-base-en-v1.5/snapshots/" +
            os.listdir("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
                       "models--BAAI--bge-base-en-v1.5/snapshots/")[0])

# content POS tags that get swapped for attribute values
SWAP_POS = {"NOUN", "PROPN", "ADJ", "NUM", "ADV"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def est_sents_minlen(texts, z):
    n = 0
    for t in texts:
        for frag in re.split(r"[.!?]+", t):
            if len(frag.split()) >= z:
                n += 1
    return n


def clone_syntax(sent, attrs, nlp):
    """Clone sent's syntax: keep scaffold tokens, swap content tokens with
    attribute values (greedy, longest values first)."""
    vals = sorted([v for v in attrs.values() if v],
                  key=lambda v: -len(str(v)))
    doc = nlp(sent)
    out = []
    used = set()
    # map value -> insertion (as a single token-ish chunk)
    for tok in doc:
        low = tok.text.lower()
        # prefer exact value present in the sentence
        matched = None
        for v in vals:
            if v.lower() in low or low in v.lower():
                if v not in used:
                    matched = v
                    break
        if matched is not None:
            out.append(matched)
            used.add(matched)
            continue
        if tok.pos_ in SWAP_POS and tok.dep_ not in ("root", "nsubj", "obj",
                                                    "dobj"):
            # non-core content word -> drop (keep scaffolding only)
            continue
        out.append(tok.text)
    text = re.sub(r"\s+", " ", " ".join(out)).strip()
    # if some attribute values still missing, append them
    missing = [v for v in attrs.values()
               if v and str(v).lower() not in text.lower()]
    if missing:
        text = text.rstrip(".") + " " + ", ".join(missing) + "."
    return text


def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
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

    def q20(text):
        sfs = []
        for doc in nlp.pipe([text], batch_size=1):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        v = user_features_v2(sfs)
        if v is None:
            return None
        return ((v - tm) / ts)[SYN]

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
        user_sfs: dict[str, list[dict]] = {}
        user_sents: dict[str, list[str]] = {}
        for u in top:
            sfs = []
            sents = []
            for doc in nlp.pipe(uc[u][:200], batch_size=64):
                for s in doc.sents:
                    if len([t for t in s if not t.is_space]) >= 5:
                        sf = per_sentence_features_v2(s)
                        if sf is not None:
                            sfs.append(sf)
                            sents.append(s.text.strip())
            if len(sfs) >= 5:
                user_sfs[u] = sfs
                user_sents[u] = sents
        users = [u for u in top if u in user_sfs][:N_USERS]
        if len(users) < N_USERS:
            continue
        zc = {}
        for u in users:
            v = user_features_v2(user_sfs[u])
            if v is not None:
                zc[u] = ((v - tm) / ts)[SYN]
        if len(zc) < N_USERS:
            continue
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

        # generate cloned queries per user (pick centroid-closest sentence)
        zq = {}
        hits = {}
        queries = {}
        for u in users:
            best_s, best_d = None, float("inf")
            for s in user_sents[u]:
                sv = q20(s)
                if sv is None:
                    continue
                d = float(np.linalg.norm(sv - zc[u]))
                if d < best_d:
                    best_d, best_s = d, s
            sub_attrs = {k: attrs[k] for k in list(attrs)[:2]}
            q = clone_syntax(best_s, sub_attrs, nlp)
            queries[u] = q
            vq = q20(q)
            if vq is not None:
                zq[u] = vq
            rk = retr(q)
            hits[u] = 1 if rk <= TOP_K else 0
        ranks = []
        for u in users:
            if u not in zq:
                continue
            dists = {o: float(np.linalg.norm(zq[u] - zc[o]))
                     for o in users if o in zc}
            rank = 1 + sum(1 for o in dists if dists[o] < dists[u])
            ranks.append(rank)
        r1 = sum(1 for r in ranks if r == 1)
        diff = float(np.max(list(hits.values())) -
                     np.min(list(hits.values())))
        per_product.append({
            "asin": asin,
            "queries": {u: queries[u] for u in users},
            "self_rank": ranks, "rank1": r1, "n_rank": len(ranks),
            "hit10": {u: hits[u] for u in users},
            "hit10_diff": round(diff, 4),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: self-rank={ranks} "
            f"(r1={r1}/{len(ranks)}) hit10_diff={diff:.3f} "
            f"({time.time() - p0:.0f}s)")

    r1_all = sum(p["rank1"] for p in per_product)
    n_all = sum(p["n_rank"] for p in per_product)
    hds = [p["hit10_diff"] for p in per_product]
    all_ranks = [r for p in per_product for r in p["self_rank"]]
    obs = r1_all / max(1, n_all)
    chance = 1.0 / N_USERS
    from math import comb
    p_binom = 0.0
    for k in range(r1_all, n_all + 1):
        p_binom += comb(n_all, k) * (chance ** k) * ((1 - chance) ** (n_all - k))
    rngb = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        idx = rngb.choice(n_all, n_all, replace=True)
        arr = np.asarray(all_ranks)[idx]
        boot.append(float((arr == 1).mean()))
    boot = np.asarray(boot)
    ci_r1 = (round(float(np.percentile(boot, 2.5)), 4),
             round(float(np.percentile(boot, 97.5)), 4))
    boot_h = []
    for _ in range(2000):
        idx = rngb.choice(len(hds), len(hds), replace=True)
        boot_h.append(float(np.asarray(hds)[idx].mean()))
    boot_h = np.asarray(boot_h)
    ci_h = (round(float(np.percentile(boot_h, 2.5)), 4),
            round(float(np.percentile(boot_h, 97.5)), 4))
    log("\n=== aggregate ===")
    log(f"self-match rank-1: {r1_all}/{n_all} ({obs:.3f}) vs chance {chance} "
        f"| binomial p={p_binom:.5f} | CI95={ci_r1}")
    log(f"hit@10 diff mean: {np.mean(hds):.3f} CI95={ci_h}")

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "aggregate": {
                       "rank1": r1_all, "n_rank": n_all,
                       "rank1_frac": round(obs, 4),
                       "chance": chance,
                       "binomial_p": round(p_binom, 6),
                       "rank1_ci95": list(ci_r1),
                       "hit10_diff_mean": round(float(np.mean(hds)), 4),
                       "hit10_diff_ci95": list(ci_h),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
