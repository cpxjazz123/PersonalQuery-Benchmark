#!/usr/bin/env python3
"""E22 T3 — multi-product PACS steering hit@10-diff eval (X=5,Y=5,Z=5).

For each of N products with top-5 users having >=5 review sentences of >=5
words (Z=5 filter), on the SAME product and SAME attribute values:
  - extract per-user style vectors (PACS: h(review) - h(plain rewrite))
  - generate 8 queries/user under steering (alpha=1.0) AND unsteered
    (alpha=0) baseline
  - BGE retrieval over same-category corpus (200): hit@10 diff across the
    5 syntax styles
Report: per-product diff, mean/max/min diff for steer vs baseline, paired
difference, and style-vector separability.
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
from extract_syntactic_features import per_sentence_features_v2, user_features_v2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_steer_multi.json"
SEED = 7777
N_PRODUCTS = 15
N_ATTRS = 9
N_USERS = 5
N_QUERIES = 8
MAX_NEW = 128
STEER_LAYER = -2
ALPHA = 1.0
N_PAIRS_PER_USER = 20
TOP_K = 10
N_CORPUS = 200
RETR_N_ATTRS = 5          # generate queries over this many attrs (partial)
                          # so retrieval is non-trivial and syntax matters
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
    # candidate products: top-5 users each >=5 est sentences of >=5 words,
    # has 9 attrs, has a title
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
    log(f"candidate products: {len(cands)}")

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
            log(f"  skip {asin}: only {len(users)} users")
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
            log(f"  skip {asin}: style vectors {len(svs)}")
            continue
        # corpus: target + its BGE NEAREST-NEIGHBOR decoys within the same
        # category, so retrieval is genuinely hard (decoys are semantically
        # close to the target) instead of trivially saturated (target always
        # rank1 under random same-category decoys).
        same_cat = []
        for asin2, rec in meta.items():
            if asin2 == asin:
                continue
            cat = rec.get("category_leaf") or ""
            if any(k in cat for k in ("Cover", "Diaper", "Protection",
                                      "Incont", "Crib", "Pad", "Changing")):
                if attrs_for_n(rec, N_ATTRS) is not None:
                    same_cat.append(asin2)
        same_cat = [a for a in same_cat if title_of(a)]
        if len(same_cat) < N_CORPUS - 1:
            rng2 = random.Random(SEED + pi)
            rng2.shuffle(same_cat)
            decoys = same_cat[:N_CORPUS - 1]
        else:
            # nearest-neighbour selection via BGE title embeddings
            tgt_title = title_of(asin)
            cand_titles = [tgt_title] + [title_of(a) for a in same_cat]
            cand_vecs = encode(cand_titles)
            sims = cand_vecs[0] @ cand_vecs[1:].T
            nn_order = np.argsort(-sims)[:N_CORPUS - 1]
            decoys = [same_cat[i] for i in nn_order]
        corpus = [asin] + decoys
        doc_texts = [title_of(a) for a in corpus]
        doc_vecs = encode(doc_texts)
        target_idx = 0

        prompt_tok = [user_prompt_tokens(attrs, tok)]

        # generation uses a REDUCED attribute set (partial) so the query does
        # not trivially match the target by containing every attribute value;
        # syntax organization then influences retrieval.
        gen_keys = list(attrs)[:RETR_N_ATTRS]
        gen_attrs = {k: attrs[k] for k in gen_keys}
        gen_prompt_tok = [user_prompt_tokens(gen_attrs, tok)]

        def retr_hit10(qtext):
            qv = encode([qtext])
            sims = qv @ doc_vecs.T
            order = np.argsort(-sims[0])[:TOP_K]
            rank = int(np.where(order == target_idx)[0][0]) + 1 \
                if target_idx in order else None
            return (1 if rank is not None else 0), rank

        def run(alpha):
            hits_by_user = {}
            ranks_by_user = {}
            n_ok = n_tot = 0
            for u in users:
                steer = None if alpha == 0 else [svs[u]] * N_QUERIES
                texts = generate_batch(
                    model, tok, None, [None] * N_QUERIES,
                    gen_prompt_tok * N_QUERIES, [gen_attrs] * N_QUERIES,
                    MAX_NEW, strict_spans=True, clean_free=False,
                    steer_vecs=steer, steer_layer=STEER_LAYER,
                    steer_alpha=alpha)
                hits = []
                ranks = []
                for q in texts:
                    h, rk = retr_hit10(q)
                    hits.append(h)
                    ranks.append(rk if rk is not None else TOP_K + 1)
                    n_tot += 1
                    missing = [k for k, a in gen_attrs.items()
                               if str(a).lower() not in q.lower()]
                    na = sum(1 for c in q if ord(c) >= 128) / max(1, len(q))
                    if not missing and na == 0:
                        n_ok += 1
                hits_by_user[u] = hits
                ranks_by_user[u] = ranks
            means = [float(np.mean(h)) for h in hits_by_user.values()]
            rmeans = [float(np.mean(r)) for r in ranks_by_user.values()]
            return {"hit10": {u: hits_by_user[u] for u in users},
                    "ranks": {u: ranks_by_user[u] for u in users},
                    "means": means, "diff": float(np.max(means) -
                                                  np.min(means)),
                    "rank_means": rmeans,
                    "rank_diff": float(np.max(rmeans) - np.min(rmeans)),
                    "purity": round(n_ok / max(1, n_tot), 4)}

        steer_res = run(ALPHA)
        base_res = run(0.0)
        per_product.append({
            "asin": asin,
            "attrs": attrs,
            "users": users,
            "steer": steer_res, "baseline": base_res,
            "steer_diff_minus_baseline": round(
                steer_res["diff"] - base_res["diff"], 4),
            "steer_rankdiff_minus_baseline": round(
                steer_res["rank_diff"] - base_res["rank_diff"], 4),
            "runtime_sec": round(time.time() - p0, 1),
        })
        log(f"[{pi + 1}/{len(cands)}] {asin}: steer_diff="
            f"{steer_res['diff']:.3f} base_diff={base_res['diff']:.3f} "
            f"rankdiff={steer_res['rank_diff']:.1f}/"
            f"{base_res['rank_diff']:.1f} purity={steer_res['purity']} "
            f"({time.time() - p0:.0f}s)")

    # ---- aggregate
    s_diffs = [p["steer"]["diff"] for p in per_product]
    b_diffs = [p["baseline"]["diff"] for p in per_product]
    deltas = [p["steer_diff_minus_baseline"] for p in per_product]
    sr_diffs = [p["steer"]["rank_diff"] for p in per_product]
    br_diffs = [p["baseline"]["rank_diff"] for p in per_product]
    log("\n=== aggregate ===")
    log(f"steer diff:   mean={np.mean(s_diffs):.3f} "
        f"min={np.min(s_diffs):.3f} max={np.max(s_diffs):.3f}")
    log(f"baseline diff:mean={np.mean(b_diffs):.3f} "
        f"min={np.min(b_diffs):.3f} max={np.max(b_diffs):.3f}")
    log(f"steer rank_diff mean={np.mean(sr_diffs):.2f} "
        f"baseline rank_diff mean={np.mean(br_diffs):.2f}")
    log(f"delta(steer-base): mean={np.mean(deltas):+.3f} "
        f"per-product={[round(d, 2) for d in deltas]}")
    pos = sum(1 for d in deltas if d > 0)
    log(f"products where steer diff > baseline: {pos}/{len(deltas)}")

    with open(OUT, "w") as f:
        json.dump({"layer": STEER_LAYER, "alpha": ALPHA,
                   "products": per_product,
                   "aggregate": {
                       "steer_diff_mean": round(float(np.mean(s_diffs)), 4),
                       "steer_diff_min": round(float(np.min(s_diffs)), 4),
                       "steer_diff_max": round(float(np.max(s_diffs)), 4),
                       "baseline_diff_mean": round(float(np.mean(b_diffs)), 4),
                       "delta_mean": round(float(np.mean(deltas)), 4),
                       "n_pos_delta": pos, "n_products": len(deltas),
                   },
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
