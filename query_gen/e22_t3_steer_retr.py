#!/usr/bin/env python3
"""E22 T3 — Contrastive Activation Steering for user syntax style transfer.

PACS (ACL 2025) idea without training: a user's writing style lives in the
LLM hidden space as a direction. For user u:
    v_u = mean(h_l(user u's review sentences)) - mean(h_l(neutral baseline))
Then during generation, steer hidden states: h' = h + alpha * v_u.

Protocol (X=5, Y=5, Z=5, w=20 config, product B0BQ1QK14T):
  - 5 users (>=5 review sentences of >=5 words on this product)
  - neutral baseline: (a) other users' reviews on the same product (content
    topic matches, style differs), (b) template queries (content-only)
  - steer the last-layer hidden states during strict-span + clean-free
    decoding -> queries keep ALL attribute values (semantically identical)
    while syntax differs per user
  - BGE retrieval over same-category corpus: hit@10 per style query + diff

Steering layer is configurable; alpha sweep reports hit@10 diff vs style
distance (20-dim syntactic).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

import query_gen_main as _qgm
from query_gen_main import (
    DTYPE, DEVICE, MODEL_PATH, TASK1_VECTORS, REVIEWS, load_meta,
    attrs_for_n, CLEAN_CONNECTORS,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_steer_retr.json"
SEED = 7777
ASIN = "B0BQ1QK14T"
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 4
MAX_NEW = 128
STEER_LAYER = -2          # layer index (0-based from last): -2 = layer 26
ALPHA = 1.0               # steering strength (times unit style direction)
STYLE_METHOD = "mean"     # "mean" or "pca" common-direction extraction
N_PAIRS_PER_USER = 30     # max user sentences used for contrast pairs
NEUTRAL_MODE = "pacs"     # "pacs" = plain-LLM rewrite of same content;
                          # "other" / "template" fallbacks
TOP_K = 10
N_CORPUS = 200
BGE_PATH = ("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
            "models--BAAI--bge-base-en-v1.5/snapshots/" +
            os.listdir("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
                       "models--BAAI--bge-base-en-v1.5/snapshots/")[0])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def attrs_to_text(attrs):
    return " ".join(f"{k}: {v}" for k, v in attrs.items())


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
    attrs = attrs_for_n(meta[ASIN], N_ATTRS)
    log(f"product {ASIN}, {len(attrs)} attrs, steering alpha={ALPHA} "
        f"layer={STEER_LAYER}")

    # ---- reviews on this product per user (top-15 by count)
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
    top = sorted(prod_revs, key=lambda u: -len(prod_revs[u]))[:15]
    # sentence filter: n_tok >= 5 (Z=5); keep users with >=5 sents (Y=5)
    user_sents: dict[str, list[str]] = {}
    for u in top:
        sents = []
        for doc in nlp.pipe(prod_revs[u][:200], batch_size=64):
            for s in doc.sents:
                if len([t for t in s if not t.is_space]) >= 5:
                    sents.append(s.text.strip())
        if len(sents) >= 5:
            user_sents[u] = sents
    users = [u for u in top if u in user_sents][:N_USERS]
    log(f"users: {len(users)} (sents: "
        f"{[len(user_sents[u]) for u in users]})")

    # ---- model
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    layer_idx = STEER_LAYER if STEER_LAYER >= 0 else n_layers + STEER_LAYER
    log(f"model {n_layers} layers, steering at layer {layer_idx}")

    # ---- neutral baseline sentences (fallback modes only; PACS rewrites
    # are generated per sentence inside the style-vector block)
    neutral_sents: list[str] = []
    if NEUTRAL_MODE == "other":
        for u, sents in user_sents.items():
            if u not in users:
                neutral_sents.extend(sents[:20])
        for u in top:
            if u not in users and u in user_sents:
                neutral_sents.extend(user_sents[u][:10])
        if not neutral_sents:
            neutral_sents = [s for u in users for s in user_sents[u]]
    elif NEUTRAL_MODE == "template":
        from query_gen_main import TEMPLATES
        sv = [attrs[k] for k in ("Brand", "Color", "Material", "Category",
                                 "Price")]
        neutral_sents = [TEMPLATES[i].format(A1=sv[0], A2=sv[1], A3=sv[2],
                                             A4=sv[3], A5=sv[4])
                         for i in range(len(TEMPLATES))]
    if neutral_sents:
        log(f"neutral baseline: {len(neutral_sents)} sentences "
            f"({NEUTRAL_MODE})")
    else:
        log(f"style extraction mode: {NEUTRAL_MODE} "
            f"(per-sentence plain-LLM contrast)")

    def pooled_hidden(texts, bs=8):
        """Mean-pooled hidden state at layer_idx for a list of sentences."""
        outs = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[layer_idx]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1)
            outs.append(pooled.float().cpu())
        return torch.cat(outs, dim=0)

    # ---- style vectors via PACS: per-sentence contrast with the plain-LLM
    # version of the SAME content. For each user sentence x we ask the base
    # (style-neutral) Qwen to rewrite x in a plain/neutral style -> x'; then
    # d_i = h(x) - h(x') isolates style (content aligned by construction);
    # v_style = mean(d_i) (or PCA-1 direction).
    neutral_h = None  # unused in PACS mode

    def plain_rewrite(sents, bs=8):
        """Base Qwen rewrite of each sentence in a plain neutral style."""
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
                o = model.generate(
                    ids, attention_mask=att, max_new_tokens=32,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id)
            for j, e in enumerate(encs):
                gen = o[j, len(e):]
                if tok.eos_token_id in gen:
                    gen = gen[:gen.tolist().index(tok.eos_token_id)]
                t = tok.decode(gen, skip_special_tokens=True).strip()
                if t:
                    outs.append(t)
        return outs

    def pooled_hidden(texts, bs=8):
        outs = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[layer_idx]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1)
            outs.append(pooled.float().cpu())
        return torch.cat(outs, dim=0)

    style_vectors: dict[str, torch.Tensor] = {}
    for u in users:
        sents = user_sents[u][:N_PAIRS_PER_USER]
        plain = plain_rewrite(sents)
        keep = min(len(sents), len(plain))
        sents, plain = sents[:keep], plain[:keep]
        if keep < 3:
            log(f"  {u[:10]}: only {keep} plain rewrites, skipping")
            continue
        hu = pooled_hidden(sents)
        hp = pooled_hidden(plain)
        diffs = hu - hp                     # per-sentence style difference
        v = diffs.mean(0)                   # common direction (mean)
        if STYLE_METHOD == "pca":
            X = diffs.numpy()
            X = X - X.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            v = torch.tensor(vt[0], dtype=torch.float32)
        # store as a UNIT direction; alpha is the steering scale (PACS-style)
        style_vectors[u] = F.normalize(v, dim=0)
        log(f"  {u[:10]}: {keep} contrast pairs, |v|={float(v.norm()):.2f} "
            f"(unit dir)")
    # check pairwise style-vector separation
    ids = list(users)
    sv_mat = torch.stack([style_vectors[u] for u in ids])
    cos = F.cosine_similarity(sv_mat[:, None, :], sv_mat[None, :, :], dim=-1)
    off = cos[~torch.eye(len(ids), dtype=bool)].abs()
    log(f"style-vector pairwise |cos|: mean={off.mean():.3f} "
        f"max={off.max():.3f}")

    # ---- steer-decoding via generate_batch (strict spans + clean free +
    # PACS activation steering hook)
    from query_gen_main import generate_batch, user_prompt_tokens, TEMPLATES
    prompt_tok = [user_prompt_tokens(attrs, tok)]

    queries_by_user: dict[str, list[str]] = {}
    for u in users:
        v = style_vectors[u]
        steer = [v] * N_QUERIES
        texts = generate_batch(
            model, tok, None, [None] * N_QUERIES,
            prompt_tok * N_QUERIES, [attrs] * N_QUERIES, MAX_NEW,
            strict_spans=True, clean_free=False,
            steer_vecs=steer, steer_layer=STEER_LAYER,
            steer_alpha=ALPHA)
        queries_by_user[u] = texts
        log(f"  {u[:10]}: {[q[:60] for q in texts]}")

    # ---- content audit
    n_ok = 0
    n_total = 0
    for u in users:
        for q in queries_by_user[u]:
            n_total += 1
            missing = [k for k, a in attrs.items()
                       if str(a).lower() not in q.lower()]
            non_ascii = sum(1 for c in q if ord(c) >= 128) / max(1, len(q))
            if not missing and non_ascii == 0:
                n_ok += 1
            if missing or non_ascii > 0:
                log(f"  [dirty] {u[:10]}: missing={missing} "
                    f"na={non_ascii:.2f} | {q[:80]}")
    log(f"content purity: {n_ok}/{n_total}")

    # ---- retrieval
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
    same_cat = []
    for asin, rec in meta.items():
        if asin == ASIN:
            continue
        cat = rec.get("category_leaf") or ""
        if any(k in cat for k in ("Cover", "Diaper", "Protection", "Incont",
                                  "Crib", "Pad", "Changing")):
            if attrs_for_n(rec, N_ATTRS) is not None:
                same_cat.append(asin)
    rng2 = random.Random(SEED)
    rng2.shuffle(same_cat)
    decoys = [a for a in same_cat if title_of(a)][:N_CORPUS - 1]
    corpus = [ASIN] + decoys
    doc_texts = [title_of(a) or attrs_to_text(attrs_for_n(meta[a], N_ATTRS))
                 for a in corpus]
    target_idx = 0
    log(f"corpus: {len(corpus)} products")

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

    doc_vecs = encode(doc_texts)

    def hit10(qtext):
        qv = encode([qtext])
        sims = qv @ doc_vecs.T
        order = np.argsort(-sims[0])[:TOP_K]
        rank = int(np.where(order == target_idx)[0][0]) + 1 \
            if target_idx in order else None
        return (1 if rank is not None else 0), rank

    results = {}
    user_means = []
    for u in users:
        hits, ranks = [], []
        for q in queries_by_user[u]:
            h, rk = hit10(q)
            hits.append(h)
            ranks.append(rk)
        results[u] = {"queries": queries_by_user[u], "hit10": hits,
                      "mean_hit10": round(float(np.mean(hits)), 4),
                      "ranks": ranks}
        user_means.append(float(np.mean(hits)))
        log(f"  {u[:10]}: hit@10={hits} mean={np.mean(hits):.2f} "
            f"ranks={ranks}")
    diff = float(np.max(user_means) - np.min(user_means))
    log(f"hit@10 diff across {len(users)} style queries: {diff:.3f}")

    with open(OUT, "w") as f:
        json.dump({"alpha": ALPHA, "layer": STEER_LAYER,
                   "neutral": NEUTRAL_MODE, "users": results,
                   "hit10_diff": round(diff, 4),
                   "content_purity": round(n_ok / max(1, n_total), 4),
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
