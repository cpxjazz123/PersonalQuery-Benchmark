#!/usr/bin/env python3
"""E22 T3 — few-shot style-exemplar transfer: does conditioning the prompt on
a user's real review sentence make the generated query carry THAT user's
syntax? Compare against PACS steering and no-style baseline on:
  1. self-match: 20-dim syntax of query vs 20-dim syntax of each user's
     reviews (rank-1 rate)
  2. retrieval hit@10 diff across the 5 style queries (same product/attrs)
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

import query_gen_main as _qgm
from query_gen_main import (
    DTYPE, DEVICE, MODEL_PATH, TASK1_VECTORS, REVIEWS, load_meta,
    attrs_for_n, generate_batch, user_prompt_tokens,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import (
    per_sentence_features_v2, user_features_v2, ALL_FEATS_V2,
)
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_fewshot_match.json"
SEED = 7777
N_PRODUCTS = 10
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 16
MAX_NEW = 128
TOP_K = 10
N_CORPUS = 200
MODE = "fewshot"   # "fewshot" (exemplar) | "pacs" | "none"
EXEMPLAR_ROLE = "system"  # put the style exemplar in the system prompt
PUNCT_PREFIXES = ("punct", "n_punct_total")
SYN = [i for i, n in enumerate(ALL_FEATS_V2)
       if not n.startswith(PUNCT_PREFIXES)]
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

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

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()

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
    bmodel.to(DEVICE)

    def encode(texts, bs=64):
        vecs = []
        for i in range(0, len(texts), bs):
            e = btok(texts[i:i + bs], return_tensors="pt", padding=True,
                     truncation=True, max_length=128).to(DEVICE)
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

        # corpus (same-category nearest neighbours)
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
        gen_keys = list(attrs)[:5]
        gen_attrs = {k: attrs[k] for k in gen_keys}

        def retr(qtext):
            qv = encode([qtext])
            sims = qv @ doc_vecs.T
            order = np.argsort(-sims[0])
            rk = int(np.where(order == target_idx)[0][0]) + 1
            return rk

        def run_prompt(exemplar_for):
            """Generate per user with an optional style exemplar in prompt."""
            zq = {}
            hits = []
            for u in users:
                attr_prompt = ("Product attributes:\n" +
                               "\n".join(f"{k}: {gen_attrs[k]}"
                                         for k in gen_attrs) +
                               "\nWrite a natural shopping query that "
                               "mentions every attribute.")
                if MODE == "fewshot" and exemplar_for:
                    ex = exemplar_for[u]
                    # STRONG style anchor: transform THIS review into the
                    # query, keeping its sentence structure/clauses/tone.
                    sys_msg = ("You are a copywriter. Rewrite the user's "
                               "review below into a shopping query that "
                               "mentions every product attribute. Keep the "
                               "review's sentence structure, clause order, "
                               "punctuation and tone exactly — only swap "
                               "the descriptive content for the attribute "
                               "values. Output ONLY the rewritten sentence.")
                    attr_prompt = ("Review to transform:\n" + ex +
                                   "\n\nProduct attributes to mention:\n" +
                                   "\n".join(f"{k}: {gen_attrs[k]}"
                                             for k in gen_attrs))
                else:
                    sys_msg = "You write shopping queries."
                p = tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": attr_prompt}],
                    tokenize=True, add_generation_prompt=True)
                texts = generate_batch(
                    model, tok, None, [None] * N_QUERIES,
                    [p] * N_QUERIES, [gen_attrs] * N_QUERIES, MAX_NEW,
                    strict_spans=True, clean_free=False,
                    steer_vecs=None, steer_alpha=0.0)
                vs = [q20(q) for q in texts]
                vs = [v for v in vs if v is not None]
                if vs:
                    zq[u] = np.mean(np.stack(vs), axis=0)
                for q in texts:
                    hits.append(1 if retr(q) <= TOP_K else 0)
            return zq, hits

        if MODE == "fewshot":
            ex = {u: random.Random(SEED + pi).choice(user_sents[u])
                  for u in users}
            zq, hits = run_prompt(ex)
        else:
            zq, hits = run_prompt(None)

        # self-match
        ranks = []
        for u in users:
            if u not in zq:
                continue
            dists = {o: float(np.linalg.norm(zq[u] - zc[o]))
                     for o in users if o in zc}
            rank = 1 + sum(1 for o in dists if dists[o] < dists[u])
            ranks.append(rank)
        r1 = sum(1 for r in ranks if r == 1)
        arr = np.asarray(hits).reshape(N_USERS, N_QUERIES)
        m = arr.mean(1)
        hd = float(m.max() - m.min())
        per_product.append({
            "asin": asin,
            "self_rank": ranks, "rank1": r1, "n_rank": len(ranks),
            "hit10_diff": round(hd, 4),
            "exemplars": {u: (ex[u] if MODE == "fewshot" else None)
                          for u in users},
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: self-rank={ranks} "
            f"(r1={r1}/{len(ranks)}) hit10_diff={hd:.3f} "
            f"({time.time() - p0:.0f}s)")

    r1_all = sum(p["rank1"] for p in per_product)
    n_all = sum(p["n_rank"] for p in per_product)
    hds = [p["hit10_diff"] for p in per_product]
    log("\n=== aggregate ===")
    log(f"[{MODE}] SELF-MATCH rank-1: {r1_all}/{n_all} "
        f"({r1_all / max(1, n_all):.3f}) vs chance 1/{N_USERS}")
    log(f"[{MODE}] hit@10 diff mean: {np.mean(hds):.3f}")

    with open(OUT, "w") as f:
        json.dump({"mode": MODE, "products": per_product,
                   "aggregate": {
                       "rank1": r1_all, "n_rank": n_all,
                       "rank1_frac": round(r1_all / max(1, n_all), 4),
                       "chance": 1.0 / N_USERS,
                       "hit10_diff_mean": round(float(np.mean(hds)), 4),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
