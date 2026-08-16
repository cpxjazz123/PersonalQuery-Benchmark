#!/usr/bin/env python3
"""E22 T3 — alpha sweep for PACS activation steering.

Reuses e22_t3_steer_retr's style extraction; for each alpha in grid:
  - generate 5 users x N queries (strict spans, no clean_free)
  - content purity, pairwise style distance (20-dim syntax), BGE hit@10 diff
Reports the alpha with best tradeoff: content purity high AND style distance
> 0 AND interpretable hit@10 diff.
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
    attrs_for_n, generate_batch, user_prompt_tokens,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2, ALL_FEATS_V2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_steer_alpha.json"
SEED = 7777
ASIN = "B0BQ1QK14T"
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 8
MAX_NEW = 128
STEER_LAYER = -2
ALPHA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_PAIRS_PER_USER = 30
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
    log(f"product {ASIN}, {len(attrs)} attrs")

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
    log(f"users: {len(users)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer_idx = STEER_LAYER if STEER_LAYER >= 0 else n_layers + STEER_LAYER

    # ---- style vectors (PACS contrast with plain-LLM rewrite)
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
            continue
        hu = pooled_hidden(sents)
        hp = pooled_hidden(plain)
        diffs = hu - hp
        v = diffs.mean(0)
        style_vectors[u] = F.normalize(v, dim=0)
    log(f"style vectors: {len(style_vectors)}")
    ids = list(users)
    sv_mat = torch.stack([style_vectors[u] for u in ids])
    cos = F.cosine_similarity(sv_mat[:, None, :], sv_mat[None, :, :], dim=-1)
    off = cos[~torch.eye(len(ids), dtype=bool)].abs()
    log(f"style-vector pairwise |cos|: mean={off.mean():.3f}")

    # ---- corpus + BGE (fixed across alphas)
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

    # ---- per-alpha run
    prompt_tok = [user_prompt_tokens(attrs, tok)]
    results = {}
    for alpha in ALPHA_GRID:
        log(f"--- alpha={alpha} ---")
        queries_by_user = {}
        n_ok = n_total = 0
        for u in users:
            if alpha == 0.0:
                steer = None
            else:
                steer = [style_vectors[u]] * N_QUERIES
            texts = generate_batch(
                model, tok, None, [None] * N_QUERIES,
                prompt_tok * N_QUERIES, [attrs] * N_QUERIES, MAX_NEW,
                strict_spans=True, clean_free=False,
                steer_vecs=steer, steer_layer=STEER_LAYER,
                steer_alpha=alpha)
            queries_by_user[u] = texts
            for q in texts:
                n_total += 1
                missing = [k for k, a in attrs.items()
                           if str(a).lower() not in q.lower()]
                na = sum(1 for c in q if ord(c) >= 128) / max(1, len(q))
                if not missing and na == 0:
                    n_ok += 1
        # 20-dim style distance among generated queries per user
        def q20(text):
            sfs = []
            for doc in nlp.pipe([text], batch_size=1):
                for s in doc.sents:
                    sf = per_sentence_features_v2(s)
                    if sf is not None:
                        sfs.append(sf)
            v = user_features_v2(sfs)
            return None if v is None else ((v - tm) / ts)
        import re as _re
        PUNCT = ("punct", "n_punct_total")
        SYN = [i for i, n in enumerate(ALL_FEATS_V2)
               if not n.startswith(PUNCT)]
        uq = {}
        for u in users:
            vs = [q20(q) for q in queries_by_user[u]]
            vs = [v for v in vs if v is not None]
            if vs:
                uq[u] = np.mean(np.stack(vs), axis=0)
        # top-20 by between/within proxy (simple: std of user means)
        if len(uq) >= 2:
            Z = np.stack([uq[u] for u in users if u in uq])
            ratios = Z.std(0)
            order20 = np.argsort(-ratios)[:20]
            D = np.linalg.norm(Z[:, order20][:, None, :] -
                               Z[:, order20][None, :, :], axis=-1)
            tri = D[np.triu_indices(len(Z), 1)]
            style_dist = float(tri.mean())
        else:
            style_dist = float("nan")
        hits_by_user = {}
        for u in users:
            hits = []
            for q in queries_by_user[u]:
                h, _ = hit10(q)
                hits.append(h)
            hits_by_user[u] = hits
        means = [float(np.mean(h)) for h in hits_by_user.values()]
        diff = float(np.max(means) - np.min(means))
        log(f"  purity={n_ok}/{n_total} style_dist={style_dist:.3f} "
            f"hit10={means} diff={diff:.3f}")
        results[str(alpha)] = {
            "content_purity": round(n_ok / max(1, n_total), 4),
            "style_dist_20d": round(style_dist, 4),
            "hit10": {u: hits_by_user[u] for u in users},
            "hit10_mean": {u: round(float(np.mean(h)), 4)
                           for u, h in hits_by_user.items()},
            "hit10_diff": round(diff, 4),
            "queries": queries_by_user,
        }

    with open(OUT, "w") as f:
        json.dump({"layer": STEER_LAYER, "alphas": results,
                   "style_vector_cos": round(float(off.mean()), 4),
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
