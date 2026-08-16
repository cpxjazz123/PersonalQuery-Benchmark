#!/usr/bin/env python3
"""E22 T3 — FINAL: prove style differs, then measure retrieval hit@10 diff.

Per product (X=5,Y=5,Z=5,w=20):
  1. PACS steering: generate 16 queries/user (alpha=1) and unsteered baseline
  2. VERIFY style actually differs: 20-dim syntax vectors per user (query
     aggregate), between-user vs within-user distance ratio
  3. Retrieval over nearest-neighbour decoy corpus: hit@10 + target rank per
     user; diff across 5 styles; bootstrap CI for the diff
  4. Retrieval-set Jaccard between style pairs (how much syntax changes what
     is retrieved)
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

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_steer_final.json"
SEED = 7777
N_PRODUCTS = 15
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 16
MAX_NEW = 128
STEER_LAYER = -2
ALPHA = 1.0
N_PAIRS_PER_USER = 20
TOP_K = 10
N_CORPUS = 200
N_BOOT = 1000
# syntax-only, no punct, no length features (w=20 proxy: top-20 by std)
PUNCT_PREFIXES = ("punct", "n_punct_total")
SYN = [i for i, n in enumerate(ALL_FEATS_V2)
       if not n.startswith(PUNCT_PREFIXES)]
BGE_PATH = ("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
            "models--BAAI--bge-base-en-v1.5/snapshots/" +
            os.listdir("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
                       "models--BAAI--bge-base-en-v1.5/snapshots/")[0])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def attrs_to_text(attrs):
    return " ".join(f"{k}: {v}" for k, v in attrs.items())


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
    n_layers = model.config.num_hidden_layers
    layer_idx = STEER_LAYER if STEER_LAYER >= 0 else n_layers + STEER_LAYER

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
        z = ((v - tm) / ts)[SYN]
        return z

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
        svs = {}
        for u in users:
            sents = user_sents[u][:N_PAIRS_PER_USER]
            plain = plain_rewrite(sents)
            keep = min(len(sents), len(plain))
            if keep < 3:
                continue
            hu = pooled_hidden(sents[:keep])
            hp = pooled_hidden(plain[:keep])
            v = (hu - hp).mean(0)
            svs[u] = F.normalize(v, dim=0)
        if len(svs) < N_USERS:
            continue
        # nearest-neighbour decoy corpus
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
        doc_texts = [title_of(a) for a in corpus]
        doc_vecs = encode(doc_texts)
        target_idx = 0
        gen_keys = list(attrs)[:5]
        gen_attrs = {k: attrs[k] for k in gen_keys}
        gen_prompt_tok = [user_prompt_tokens(gen_attrs, tok)]

        def retr(qtext):
            qv = encode([qtext])
            sims = qv @ doc_vecs.T
            order = np.argsort(-sims[0])
            rk = int(np.where(order == target_idx)[0][0]) + 1
            return rk, order[:TOP_K]

        def run(alpha):
            hits = {}
            ranks = {}
            topk_sets = {}
            pur = [0, 0]
            for u in users:
                steer = None if alpha == 0 else [svs[u]] * N_QUERIES
                texts = generate_batch(
                    model, tok, None, [None] * N_QUERIES,
                    gen_prompt_tok * N_QUERIES, [gen_attrs] * N_QUERIES,
                    MAX_NEW, strict_spans=True, clean_free=False,
                    steer_vecs=steer, steer_layer=STEER_LAYER,
                    steer_alpha=alpha)
                hs, rs, ts = [], [], []
                for q in texts:
                    rk, order = retr(q)
                    hs.append(1 if rk <= TOP_K else 0)
                    rs.append(rk)
                    ts.append(set(order.tolist()))
                    pur[1] += 1
                    missing = [k for k, a in gen_attrs.items()
                               if str(a).lower() not in q.lower()]
                    na = sum(1 for c in q if ord(c) >= 128) / max(1, len(q))
                    if not missing and na == 0:
                        pur[0] += 1
                hits[u] = hs
                ranks[u] = rs
                topk_sets[u] = ts
            means = [float(np.mean(h)) for h in hits.values()]
            rmeans = [float(np.mean(r)) for r in ranks.values()]
            return {"hits": hits, "ranks": ranks, "topk": topk_sets,
                    "hit_diff": float(np.max(means) - np.min(means)),
                    "rank_diff": float(np.max(rmeans) - np.min(rmeans)),
                    "purity": round(pur[0] / max(1, pur[1]), 4)}

        steer_res = run(ALPHA)
        base_res = run(0.0)
        # recompute per-user query aggregate 20-dim from steer hits
        style_dist = None
        try:
            uq = {}
            for u in users:
                steer = [svs[u]] * 4
                texts = generate_batch(
                    model, tok, None, [None] * 4,
                    gen_prompt_tok * 4, [gen_attrs] * 4, MAX_NEW,
                    strict_spans=True, clean_free=False,
                    steer_vecs=steer, steer_layer=STEER_LAYER,
                    steer_alpha=ALPHA)
                vs = [q20(q) for q in texts]
                vs = [v for v in vs if v is not None]
                if vs:
                    uq[u] = np.mean(np.stack(vs), axis=0)
            if len(uq) >= 2:
                Z = np.stack([uq[u] for u in users if u in uq])
                D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
                tri = D[np.triu_indices(len(Z), 1)]
                style_dist = float(tri.mean())
        except Exception as e:
            log(f"  style check failed: {e}")

        # retrieval-set Jaccard between style pairs (steer)
        sj = []
        us = list(users)
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a_sets = steer_res["topk"][us[i]]
                b_sets = steer_res["topk"][us[j]]
                for s1 in a_sets:
                    for s2 in b_sets:
                        inter = len(s1 & s2)
                        uni = len(s1 | s2)
                        sj.append(inter / max(1, uni))
        steer_jac = float(np.mean(sj)) if sj else None

        # bootstrap CI for hit diff (steer)
        h_all = [h for u in users for h in steer_res["hits"][u]]
        boot = []
        rngb = np.random.default_rng(SEED + pi)
        for _ in range(N_BOOT):
            ids = rngb.choice(len(users) * N_QUERIES,
                              len(users) * N_QUERIES, replace=True)
            arr = np.asarray(h_all)[ids].reshape(N_USERS, N_QUERIES)
            m = arr.mean(1)
            boot.append(float(m.max() - m.min()))
        boot = np.asarray(boot)
        ci = (round(float(np.percentile(boot, 2.5)), 4),
              round(float(np.percentile(boot, 97.5)), 4))

        per_product.append({
            "asin": asin,
            "steer": steer_res, "baseline": base_res,
            "style_dist_20d": round(style_dist, 3) if style_dist else None,
            "steer_jaccard_topk": round(steer_jac, 3) if steer_jac else None,
            "hit_diff_ci95": list(ci),
            "steer_minus_baseline": round(
                steer_res["hit_diff"] - base_res["hit_diff"], 4),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: steer_hit_diff="
            f"{steer_res['hit_diff']:.3f} base={base_res['hit_diff']:.3f} "
            f"style_d={style_dist:.2f} jac={steer_jac:.2f} "
            f"purity={steer_res['purity']} "
            f"ci95={ci} ({time.time() - p0:.0f}s)")

    # aggregate
    sd = [p["steer"]["hit_diff"] for p in per_product]
    bd = [p["baseline"]["hit_diff"] for p in per_product]
    sd20 = [p["style_dist_20d"] for p in per_product if p["style_dist_20d"]]
    sj = [p["steer_jaccard_topk"] for p in per_product
          if p["steer_jaccard_topk"]]
    log("\n=== aggregate ===")
    log(f"steer hit diff mean={np.mean(sd):.3f} | "
        f"baseline mean={np.mean(bd):.3f}")
    log(f"steer style_dist(20d) mean={np.mean(sd20):.2f}")
    log(f"steer topk Jaccard mean={np.mean(sj):.3f}")
    log(f"delta mean={np.mean([p['steer_minus_baseline'] for p in per_product]):+.3f}")

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "aggregate": {
                       "steer_hit_diff_mean": round(float(np.mean(sd)), 4),
                       "baseline_hit_diff_mean": round(float(np.mean(bd)), 4),
                       "style_dist_20d_mean": round(float(np.mean(sd20)), 3),
                       "steer_jaccard_mean": round(float(np.mean(sj)), 3),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1,
                  default=lambda o: sorted(o) if isinstance(o, set) else
                  str(o))
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
