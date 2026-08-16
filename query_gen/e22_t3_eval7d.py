#!/usr/bin/env python3
"""E22 T3 — 7-dim Query-compatible vector evaluation (post strict-decoding).

Goal: verify "review style transfers to query" using the 7 Query-compatible
syntactic features (e22_t1_user_vectors.npz feature_indices):
  nest_max, n_punct_total, punct_,, punct_., punct_!, punct_', punctbg_._.

Protocol (single product B0B5JP4MGX, top-9 users, same as one_prod):
  per user:
    - z7  = user's review-aggregated 7-dim vector (standardized, frozen)
    - generate N_QUERIES queries under strict-span decoding (all 10 attrs
      verbatim) → per-query 7-dim → averaged q7
  metrics:
    1. self-rank: rank of own z7 among all users' z7 by L2 to q7
    2. style-delta: |q7 - z_u| vs |template7 - z_u| (template = attr template
       sentence; lower q7 distance = style transferred beyond template)
    3. zero-z control:  z=0 prefix → self-rank (expect ~chance)
    4. shuffled-z control: z permuted → self-rank (expect ~chance)
    5. content exact: 10/10 attrs present per query (strict decoding)
  report: rank-1 count, rank histogram, mean style-delta, controls.
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

import query_gen_main as _qgm
from query_gen_main import (
    SoftPrefixProjector, CopyAwareHead, generate_batch, user_prompt_tokens,
    attrs_for_n, load_meta, DTYPE, DEVICE, NUM_TOKENS, PROJ_HIDDEN, Z_DIM,
    GATE_INIT, MODEL_PATH, INJECTOR_PT, TASK1_VECTORS, REVIEWS, TEMPLATES,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_eval7d.json"
SEED = 6666
# Rich single-product protocol: B0BQ1QK14T top-10 users have 17-208
# sentences ON this product (the only product where same-product reviews are
# abundant enough for stable z); 9 attrs available.
ASIN = "B0BQ1QK14T"
N_ATTRS = 9
N_QUERIES = 8
MAX_NEW = 128
STRICT_SPANS = True
BATCH = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sentence_vec7(text: str, nlp, tm, ts, fidx) -> np.ndarray | None:
    sfs = []
    for doc in nlp.pipe([text], batch_size=1):
        for s in doc.sents:
            sf = per_sentence_features_v2(s)
            if sf is not None:
                sfs.append(sf)
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    z = (v - tm) / ts
    return z[fidx].astype(np.float64)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    fidx = vd["feature_indices"].astype(int)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    meta = load_meta()
    attrs = attrs_for_n(meta[ASIN], N_ATTRS)
    log(f"product {ASIN}, {N_ATTRS} attrs: {list(attrs)[:4]}...")

    # user reviews on this product → z7
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
    top10 = sorted(prod_revs, key=lambda u: -len(prod_revs[u]))[:10]
    # full 318 z per user (needed for the projector) + 7-dim subset for eval
    user_z318: dict[str, np.ndarray] = {}
    for u in top10:
        sfs = []
        for doc in nlp.pipe(prod_revs[u][:50], batch_size=64):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        if len(sfs) < 3:
            continue
        v = user_features_v2(sfs)
        if v is None:
            continue
        user_z318[u] = ((v - tm) / ts).astype(np.float64)
    users = [u for u in top10 if u in user_z318]
    user_z7 = {u: user_z318[u][fidx] for u in users}
    log(f"users with z318/z7: {len(users)}")

    # template 7-dim baseline (content-correct, no style)
    tmpl7 = {}
    sv = [attrs[k] for k in ("Brand", "Color", "Material", "Category", "Price")]
    for i, tpl in enumerate(TEMPLATES):
        q = tpl.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])
        v = sentence_vec7(q, nlp, tm, ts, fidx)
        if v is not None:
            tmpl7[i] = v
    log(f"template7: {len(tmpl7)} templates")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(INJECTOR_PT, map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector loaded")

    prompt = [user_prompt_tokens(attrs, tok)]

    def gen(z_list, tag):
        with torch.no_grad():
            pes = [proj(torch.tensor(z, dtype=torch.float32,
                                     device=DEVICE).to(DTYPE).unsqueeze(0))[0]
                   for z in z_list]
        qs = []
        for s in range(0, len(users), BATCH):
            sl = users[s:s + BATCH]
            texts = generate_batch(
                model, tok, copy_head, pes[s:s + len(sl)],
                prompt * len(sl), [attrs] * len(sl), MAX_NEW,
                strict_spans=STRICT_SPANS)
            for u, t in zip(sl, texts):
                qs.append((u, t))
        return qs

    all_q7: dict[str, list[np.ndarray]] = defaultdict(list)
    texts_by_user: dict[str, list[str]] = defaultdict(list)
    content_ok = 0
    n_total = 0
    for r in range(N_QUERIES):
        pairs = gen([user_z318[u] for u in users], "real")
        for u, t in pairs:
            texts_by_user[u].append(t)
            v = sentence_vec7(t, nlp, tm, ts, fidx)
            if v is not None:
                all_q7[u].append(v)
            n_total += 1
            if all(str(a).lower() in t.lower() for a in attrs.values()):
                content_ok += 1
    log(f"content exact: {content_ok}/{n_total} (per-query)")

    # controls
    rng = random.Random(SEED)
    zs_shuf = [user_z318[u] for u in users]
    rng.shuffle(zs_shuf)
    zs_zero = [np.zeros_like(user_z318[u]) for u in users]
    q7_shuf: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in range(1):
        for u, t in gen(zs_shuf, "shuf"):
            v = sentence_vec7(t, nlp, tm, ts, fidx)
            if v is not None:
                q7_shuf[u].append(v)
    q7_zero: dict[str, list[np.ndarray]] = defaultdict(list)
    for u, t in gen(zs_zero, "zero"):
        v = sentence_vec7(t, nlp, tm, ts, fidx)
        if v is not None:
            q7_zero[u].append(v)

    def ranks_of(q7_map):
        ranks = []
        for u in users:
            qs = q7_map.get(u)
            if not qs:
                continue
            q = np.mean(np.stack(qs), axis=0)
            dists = {o: float(np.linalg.norm(q - user_z7[o]))
                     for o in users}
            rank = 1 + sum(1 for o in users if dists[o] < dists[u])
            ranks.append(rank)
        return ranks

    r_real = ranks_of(all_q7)
    r_shuf = ranks_of(q7_shuf)
    r_zero = ranks_of(q7_zero)
    log(f"self-rank (real):    {sorted(r_real)}")
    log(f"self-rank (shuffled):{sorted(r_shuf)}")
    log(f"self-rank (zero):    {sorted(r_zero)}")
    log(f"rank-1 real/shuf/zero: "
        f"{sum(1 for r in r_real if r==1)}/"
        f"{sum(1 for r in r_shuf if r==1)}/"
        f"{sum(1 for r in r_zero if r==1)} "
        f"(chance 1/{len(users)})")

    # style-delta: q7 distance to own z vs template7 distance to own z
    t7 = np.mean(np.stack(list(tmpl7.values())), axis=0) if tmpl7 else None
    if t7 is not None:
        d_q = []
        d_t = []
        for u in users:
            qs = all_q7.get(u)
            if not qs:
                continue
            q = np.mean(np.stack(qs), axis=0)
            d_q.append(float(np.linalg.norm(q - user_z7[u])))
            d_t.append(float(np.linalg.norm(t7 - user_z7[u])))
        log(f"style-delta mean: query={np.mean(d_q):.3f} "
            f"template={np.mean(d_t):.3f} "
            f"(lower query = style transferred)")

    result = {
        "version": "e22_t3_eval7d_v1", "seed": SEED, "asin": ASIN,
        "attrs": attrs, "users": users,
        "user_z7": {u: user_z7[u].tolist() for u in users},
        "queries": {u: texts_by_user.get(u, []) for u in users},
        "self_rank_real": r_real,
        "self_rank_shuffled": r_shuf,
        "self_rank_zero": r_zero,
        "rank1": {"real": sum(1 for r in r_real if r == 1),
                  "shuffled": sum(1 for r in r_shuf if r == 1),
                  "zero": sum(1 for r in r_zero if r == 1),
                  "chance": 1.0 / max(1, len(users))},
        "content_exact_per_query": round(content_ok / max(1, n_total), 4),
        "style_delta": {"query_mean": round(float(np.mean(d_q)), 4),
                        "template_mean": round(float(np.mean(d_t)), 4)}
                        if t7 is not None else None,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
