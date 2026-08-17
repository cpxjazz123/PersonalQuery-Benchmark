#!/usr/bin/env python3
"""query_gen_one_prod — 1 product x 10 users x 1 query each: same-content
style probe.

Take ONE product with 10 attrs bought by >=10 users; take its top-10 users
(by review count); compute each user's 318-dim z_user; inject z_u -> generate
ONE query per user (same product/attrs). Report:
  - generated query 318-dim vectors (aggregated per user)
  - pairwise distance among the 10 users' generated-query syntax
  - whether query_u is closer to z_u than to other z's (self-matching rank)
  - content preservation (10 attrs exact)
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

import query_gen_main as _qgm  # noqa: E402
from query_gen_main import (  # noqa: E402
    SoftPrefixProjector, CopyAwareHead, generate_batch, user_prompt_tokens,
    spacy318_aggregate, attrs_for_n, load_meta, EXT_ATTR_KEYS, TOP_ATTR_KEYS,
    DTYPE, DEVICE, NUM_TOKENS, PROJ_HIDDEN, Z_DIM, GATE_INIT, MODEL_PATH,
    INJECTOR_PT, TASK1_VECTORS, REVIEWS,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_one_prod.json"
SEED = 5555
MAX_NEW = 128   # 10 strict attr spans + connector tokens need headroom
STRICT_SPANS = True  # token/span-level constraint: all attrs appear verbatim


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    random.seed(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    meta = load_meta()
    # find ONE product with >=10 users and 10 attrs (deterministic: first in
    # sorted order with the most users)
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
    best = None
    for a, uc in prod_users.items():
        if len(uc) >= 10 and a in meta and \
                attrs_for_n(meta[a], 10) is not None:
            if best is None or len(uc) > len(prod_users[best]):
                best = a
    if best is None:
        log("no eligible product")
        return
    asin = best
    attrs = attrs_for_n(meta[asin], 10)
    top10 = [u for u, _ in
             sorted(prod_users[asin].items(), key=lambda kv: -len(kv[1]))[:10]]
    log(f"product {asin}: {len(prod_users[asin])} users, "
        f"top10 review counts = {[len(prod_users[asin][u]) for u in top10]}")
    log(f"attrs: {list(attrs.items())[:3]} ...")

    # user z from THEIR reviews on this product only
    user_z = {}
    user_sfs = {}
    for u in top10:
        texts = prod_users[asin][u][:50]
        docs = nlp.pipe(texts, batch_size=64)
        sfs = []
        for doc in docs:
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        if len(sfs) < 3:
            continue
        v = user_features_v2(sfs)
        if v is not None:
            user_z[u] = ((v - tm) / ts).astype(np.float64)
            user_sfs[u] = sfs
    log(f"users with z (from this product's reviews): {len(user_z)}")
    if len(user_z) < 2:
        return

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

    users = list(user_z.keys())
    prompt = [user_prompt_tokens(attrs, tok)]
    with torch.no_grad():
        pes = [proj(torch.tensor(user_z[u], dtype=torch.float32,
                                 device=DEVICE).to(DTYPE).unsqueeze(0))[0]
               for u in users]
    texts = generate_batch(model, tok, copy_head, pes, prompt * len(users),
                           [attrs] * len(users), MAX_NEW,
                           strict_spans=STRICT_SPANS)
    # per-user generated 318 aggregate (1 query each)
    gen_z = {}
    content_ok = 0
    for u, q in zip(users, texts):
        agg = spacy318_aggregate([(q, attrs)], nlp, tm, ts)
        gen_z[u] = agg
        if all(str(v).lower() in q.lower() for v in attrs.values()):
            content_ok += 1
        log(f"  {u[:10]}: [{len(q.split())}w] {q[:90]}")
    log(f"content exact: {content_ok}/{len(users)}")

    # self-matching: rank of own z among all users' z by distance to gen_z[u]
    ranks = []
    for u in users:
        if gen_z[u] is None:
            continue
        dists = {o: float(np.linalg.norm(gen_z[u] - user_z[o]))
                 for o in users}
        rank = 1 + sum(1 for o in users if dists[o] < dists[u])
        ranks.append(rank)
    n_ok_rank1 = sum(1 for r in ranks if r == 1)
    log(f"self-rank distribution: {sorted(ranks)}")
    log(f"rank-1 (own z closest): {n_ok_rank1}/{len(ranks)}")

    # pairwise distance among generated queries
    vecs = [gen_z[u] for u in users if gen_z[u] is not None]
    if len(vecs) >= 2:
        V = np.stack(vecs)
        D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)
        tri = D[np.triu_indices(len(V), 1)]
        log(f"generated-query pairwise L2: mean={tri.mean():.2f} "
            f"max={tri.max():.2f}")

    result = {
        "version": "query_gen_one_prod_v1", "seed": SEED,
        "asin": asin, "attrs": attrs,
        "users": users,
        "user_z": {u: user_z[u].tolist() for u in users},
        "generated_queries": {u: t for u, t in zip(users, texts)},
        "generated_z": {u: (gen_z[u].tolist() if gen_z[u] is not None else None)
                        for u in users},
        "self_match_ranks": ranks,
        "rank1_count": int(n_ok_rank1),
        "content_exact": round(content_ok / len(users), 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
