#!/usr/bin/env python3
"""E22 T3 — PACS layer x alpha sweep: find the layer+strength where the
style vector (review vs plain-LLM rewrite contrast) actually transfers to
generated queries (self-match > chance).

PACS pipeline (per user):
  1. user's review sentences x_i
  2. plain-LLM rewrite x'_i (same content, neutral style)
  3. d_i = h_l(x_i) - h_l(x'_i) per sentence (layer l, mean-pooled)
  4. v = mean(d_i) (common direction; also PCA-1 variant)
  5. generate with h' = h + alpha*v at layer l
Evaluate self-match (query 20-dim syntax vs user review centroid) per
(layer, alpha).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_pacs_layer.json"
SEED = 7777
N_PRODUCTS = 10
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 8
MAX_NEW = 96
N_PAIRS_PER_USER = 20
LAYER_GRID = [-2, 24, 16, 8]
ALPHA_GRID = [0.5, 1.0, 2.0, 4.0]
PUNCT_PREFIXES = ("punct", "n_punct_total")
SYN = [i for i, n in enumerate(ALL_FEATS_V2)
       if not n.startswith(PUNCT_PREFIXES)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def est_sents_minlen(texts, z):
    import re
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
    n_layers = model.config.num_hidden_layers

    def plain_rewrite(sents, bs=8):
        outs = []
        for i in range(0, len(sents), bs):
            chunk = sents[i:i + bs]
            msgs = [[{"role": "user",
                      "content": "Rewrite the following review sentence in a "
                                 "plain, neutral style, keeping the exact "
                                 "meaning:\n" + s}] for s in chunk]
            encs = [tok.apply_chat_template(m, tokenize=True,
                                            add_generation_prompt=True)
                    for m in msgs]
            mx = max(len(e) for e in encs)
            ids = torch.tensor([e + [tok.pad_token_id] * (mx - len(e))
                                for e in encs], dtype=torch.long,
                               device=DEVICE)
            att = torch.tensor([[1] * len(e) + [0] * (mx - len(e))
                                 for e in encs], dtype=torch.long,
                                device=DEVICE)
            with torch.no_grad():
                o = model.generate(ids, attention_mask=att,
                                   max_new_tokens=32, do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
            for j, e in enumerate(encs):
                gen = o[j, len(e):]
                if tok.eos_token_id in gen:
                    gen = gen[:gen.tolist().index(tok.eos_token_id)]
                t = tok.decode(gen, skip_special_tokens=True).strip()
                if t:
                    outs.append(t)
        return outs

    def pooled_hidden(texts, layer, bs=8):
        outs = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[layer]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1)
            outs.append(pooled.float().cpu())
        return torch.cat(outs, dim=0)

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
        # per-user contrast pairs (fixed across layers)
        pairs: dict[str, tuple[list[str], list[str]]] = {}
        for u in users:
            sents = user_sents[u][:N_PAIRS_PER_USER]
            plain = plain_rewrite(sents)
            keep = min(len(sents), len(plain))
            pairs[u] = (sents[:keep], plain[:keep])
        # prompt for generation
        gen_keys = list(attrs)[:5]
        gen_attrs = {k: attrs[k] for k in gen_keys}
        gen_prompt_tok = [user_prompt_tokens(gen_attrs, tok)]

        cells = {}
        for layer in LAYER_GRID:
            lid = layer if layer >= 0 else n_layers + layer
            svs = {}
            for u in users:
                sents, plain = pairs[u]
                hu = pooled_hidden(sents, lid)
                hp = pooled_hidden(plain, lid)
                v = (hu - hp).mean(0)
                svs[u] = F.normalize(v, dim=0)
            for alpha in ALPHA_GRID:
                zq = {}
                for u in users:
                    steer = [svs[u]] * N_QUERIES
                    texts = generate_batch(
                        model, tok, None, [None] * N_QUERIES,
                        gen_prompt_tok * N_QUERIES,
                        [gen_attrs] * N_QUERIES, MAX_NEW,
                        strict_spans=True, clean_free=False,
                        steer_vecs=steer, steer_layer=layer,
                        steer_alpha=alpha)
                    vs = [q20(q) for q in texts]
                    vs = [v for v in vs if v is not None]
                    if vs:
                        zq[u] = np.mean(np.stack(vs), axis=0)
                ranks = []
                for u in users:
                    if u not in zq:
                        continue
                    dists = {o: float(np.linalg.norm(zq[u] - zc[o]))
                             for o in users if o in zc}
                    rank = 1 + sum(1 for o in dists if dists[o] < dists[u])
                    ranks.append(rank)
                r1 = sum(1 for r in ranks if r == 1)
                cells[f"L{layer}_a{alpha}"] = {
                    "self_rank": ranks, "rank1": r1,
                    "n_rank": len(ranks),
                    "rank1_frac": round(r1 / max(1, len(ranks)), 3),
                }
                log(f"[{pi + 1}/{len(cands)}] {asin} L{layer} a{alpha}: "
                    f"r1={r1}/{len(ranks)} "
                    f"({r1 / max(1, len(ranks)):.3f})")
        per_product.append({"asin": asin, "cells": cells})
        log(f"  ({time.time() - p0:.0f}s)")

    # aggregate by (layer, alpha)
    log("\n=== aggregate by L x alpha (rank-1 frac, chance 0.20) ===")
    agg = {}
    for layer in LAYER_GRID:
        for alpha in ALPHA_GRID:
            key = f"L{layer}_a{alpha}"
            r1s = sum(p["cells"][key]["rank1"] for p in per_product)
            ns = sum(p["cells"][key]["n_rank"] for p in per_product)
            agg[key] = round(r1s / max(1, ns), 3)
            log(f"{key}: {r1s}/{ns} ({r1s / max(1, ns):.3f})")

    with open(OUT, "w") as f:
        json.dump({"products": per_product, "aggregate": agg,
                   "chance": 0.2,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
