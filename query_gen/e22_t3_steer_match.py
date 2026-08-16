#!/usr/bin/env python3
"""E22 T3 — decisive check: does the generated query carry ITS OWN user's
review syntax (style transfer) rather than just being different across users?

Per product (X=5,Y=5,Z=5,w=20):
  - z_comment[u] = 20-dim syntax vector of user u's reviews
  - steer: generate N queries under PACS steering -> z_query_steer[u]
  - baseline: same without steering -> z_query_base[u]
  - SELF-MATCH rank: among the 5 users' z_comment, where does z_query[u]
    land? rank-1 rate == comment style transferred to the query.
  - also report hit@10 diff (steer vs baseline) for completeness.
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

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_steer_match.json"
SEED = 7777
N_PRODUCTS = 10
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 16
MAX_NEW = 128
STEER_LAYER = -2
ALPHA = 1.0
N_PAIRS_PER_USER = 20
TOP_K = 10
N_CORPUS = 200
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

    def syn20_of_sfs(sfs):
        v = user_features_v2(sfs)
        if v is None:
            return None
        return ((v - tm) / ts)[SYN]

    def q20(text):
        sfs = []
        for doc in nlp.pipe([text], batch_size=1):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    sfs.append(sf)
        return syn20_of_sfs(sfs)

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
                        sfs.append(per_sentence_features_v2(s))
                        sents.append(s.text.strip())
            sfs = [sf for sf in sfs if sf is not None]
            if len(sfs) >= 5:
                user_sfs[u] = sfs
                user_sents[u] = sents
        users = [u for u in top if u in user_sfs][:N_USERS]
        if len(users) < N_USERS:
            continue
        # comment 20-dim per user
        zc = {}
        for u in users:
            v = syn20_of_sfs(user_sfs[u])
            if v is not None:
                zc[u] = v
        if len(zc) < N_USERS:
            continue
        # style vectors
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
        # retrieval corpus
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
            return rk

        def run(alpha, n_q):
            zq = {}
            hits = []
            for u in users:
                steer = None if alpha == 0 else [svs[u]] * n_q
                texts = generate_batch(
                    model, tok, None, [None] * n_q,
                    gen_prompt_tok * n_q, [gen_attrs] * n_q, MAX_NEW,
                    strict_spans=True, clean_free=False,
                    steer_vecs=steer, steer_layer=STEER_LAYER,
                    steer_alpha=alpha)
                vs = [q20(q) for q in texts]
                vs = [v for v in vs if v is not None]
                if vs:
                    zq[u] = np.mean(np.stack(vs), axis=0)
                for q in texts:
                    hits.append(1 if retr(q) <= TOP_K else 0)
            return zq, hits

        # self-match ranks for steer and baseline
        def self_rank(zq):
            if len(zq) < 2:
                return None
            ranks = []
            for u in users:
                if u not in zq:
                    continue
                dists = {o: float(np.linalg.norm(zq[u] - zc[o]))
                         for o in users if o in zc}
                if len(dists) < 2:
                    continue
                rank = 1 + sum(1 for o in dists if dists[o] < dists[u])
                ranks.append(rank)
            return ranks

        zq_steer, hit_steer = run(ALPHA, N_QUERIES)
        zq_base, hit_base = run(0.0, N_QUERIES)
        rs_steer = self_rank(zq_steer)
        rs_base = self_rank(zq_base)
        # chance = 1/N_USERS
        r1s = sum(1 for r in rs_steer if r == 1) if rs_steer else 0
        r1b = sum(1 for r in rs_base if r == 1) if rs_base else 0
        n_rank = len(rs_steer) if rs_steer else 0
        # hit@10 diff
        arr = np.asarray(hit_steer).reshape(N_USERS, N_QUERIES)
        m = arr.mean(1)
        hd_steer = float(m.max() - m.min())
        arr = np.asarray(hit_base).reshape(N_USERS, N_QUERIES)
        m = arr.mean(1)
        hd_base = float(m.max() - m.min())
        per_product.append({
            "asin": asin,
            "self_rank_steer": rs_steer, "self_rank_baseline": rs_base,
            "rank1_steer": r1s, "rank1_baseline": r1b,
            "n_rank": n_rank,
            "hit10_diff_steer": round(hd_steer, 4),
            "hit10_diff_baseline": round(hd_base, 4),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: self-rank steer={rs_steer} "
            f"(r1={r1s}/{n_rank}) base={rs_base} (r1={r1b}/{n_rank}) "
            f"hit10_diff={hd_steer:.3f}/{hd_base:.3f} "
            f"({time.time() - p0:.0f}s)")

    # aggregate
    r1s_all = sum(p["rank1_steer"] for p in per_product)
    r1b_all = sum(p["rank1_baseline"] for p in per_product)
    n_all = sum(p["n_rank"] for p in per_product)
    hds = [p["hit10_diff_steer"] for p in per_product]
    hdb = [p["hit10_diff_baseline"] for p in per_product]
    log("\n=== aggregate ===")
    log(f"SELF-MATCH rank-1: steer={r1s_all}/{n_all} "
        f"({r1s_all / max(1, n_all):.3f}) vs baseline={r1b_all}/{n_all} "
        f"({r1b_all / max(1, n_all):.3f}) vs chance 1/{N_USERS}")
    log(f"hit@10 diff mean: steer={np.mean(hds):.3f} "
        f"baseline={np.mean(hdb):.3f}")

    with open(OUT, "w") as f:
        json.dump({"products": per_product,
                   "aggregate": {
                       "rank1_steer": r1s_all,
                       "rank1_baseline": r1b_all,
                       "n_rank": n_all,
                       "rank1_steer_frac": round(r1s_all / max(1, n_all), 4),
                       "rank1_baseline_frac": round(r1b_all / max(1, n_all), 4),
                       "chance": 1.0 / N_USERS,
                       "hit10_diff_steer_mean": round(float(np.mean(hds)), 4),
                       "hit10_diff_baseline_mean": round(float(np.mean(hdb)), 4),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
