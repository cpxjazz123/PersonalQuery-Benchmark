#!/usr/bin/env python3
"""E22 T3 — retrieval hit@10 diff across SYNTAX-STYLE query variants.

Config: X=5 (top-5 users), Y=5, Z=5, w=20 (separable style dims).

Protocol (product B0BQ1QK14T, the rich single-product used before):
  - same product, SAME attribute values -> generate 5 queries, one per user's
    syntax style (v6 injector, strict-span decoding => content identical)
  - also generate a template query (no style) as control
  - BGE-base retrieves over a corpus of N_CORPUS products (target included)
  - per query: hit@10 = target in top-10? and target rank
  - report hit@10 for each style query and the DIFF (max-min) across styles,
    plus diff vs template control.
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
    SoftPrefixProjector, CopyAwareHead, generate_batch, user_prompt_tokens,
    attrs_for_n, load_meta, DTYPE, DEVICE, NUM_TOKENS, PROJ_HIDDEN, Z_DIM,
    GATE_INIT, MODEL_PATH, INJECTOR_PT, TASK1_VECTORS, REVIEWS, TEMPLATES,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_retr_style.json"
SEED = 7777
ASIN = "B0BQ1QK14T"
N_ATTRS = 9
N_USERS = 5
N_QUERIES_PER_USER = 4
MAX_NEW = 128
STRICT_SPANS = True
TOP_K = 10
N_CORPUS = 200          # target product + 199 decoys
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

    # top-N_USERS users by review count on this product; their z_user
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
    top = sorted(prod_revs, key=lambda u: -len(prod_revs[u]))[:N_USERS]
    user_z318: dict[str, np.ndarray] = {}
    for u in top:
        sfs = []
        for doc in nlp.pipe(prod_revs[u][:200], batch_size=64):
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None and sf["n_tok"] >= 5:
                    sfs.append(sf)
        if len(sfs) < 5:
            continue
        v = user_features_v2(sfs)
        if v is not None:
            user_z318[u] = ((v - tm) / ts).astype(np.float64)
    users = [u for u in top if u in user_z318][:N_USERS]
    log(f"users with z: {len(users)} (n_tok>=5, >=5 sents)")

    # ---- load v6 injector + Qwen
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

    def gen_for(z_list):
        with torch.no_grad():
            pes = [proj(torch.tensor(z, dtype=torch.float32,
                                     device=DEVICE).to(DTYPE).unsqueeze(0))[0]
                   for z in z_list]
        return generate_batch(model, tok, copy_head, pes,
                              prompt * len(z_list),
                              [attrs] * len(z_list), MAX_NEW,
                              strict_spans=STRICT_SPANS,
                              clean_free=True)

    queries_by_user = {}
    for r in range(N_QUERIES_PER_USER):
        texts = gen_for([user_z318[u] for u in users])
        for u, t in zip(users, texts):
            queries_by_user.setdefault(u, []).append(t)
    # template control (content-correct, no user style)
    sv = [attrs[k] for k in ("Brand", "Color", "Material", "Category",
                             "Price")]
    tmpl_qs = [TEMPLATES[i].format(A1=sv[0], A2=sv[1], A3=sv[2],
                                   A4=sv[3], A5=sv[4])
               for i in range(min(4, len(TEMPLATES)))]
    log(f"generated {len(users)} users x {N_QUERIES_PER_USER} "
        f"style queries + {len(tmpl_qs)} template controls")

    # ---- content purity audit: all attrs present, no non-ASCII leak
    def content_audit(q):
        missing = [k for k, a in attrs.items()
                   if str(a).lower() not in q.lower()]
        non_ascii = sum(1 for c in q if ord(c) >= 128) / max(1, len(q))
        return missing, round(non_ascii, 4)

    n_content_ok = 0
    n_total_q = 0
    for u in users:
        for q in queries_by_user[u]:
            n_total_q += 1
            missing, na = content_audit(q)
            if not missing and na == 0:
                n_content_ok += 1
            if missing or na > 0:
                log(f"  [dirty] user {u[:10]}: missing={missing} "
                    f"non_ascii={na} | {q[:90]}")
    log(f"content purity: {n_content_ok}/{n_total_q} "
        f"(all attrs + ascii-only)")

    # ---- retrieval corpus: target product + decoys (SAME-CATEGORY so the
    # task is nontrivial). Documents are the product TITLES (natural text),
    # NOT attr strings — otherwise BGE trivially matches all attr values.
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
    def title_of(asin):
        d = meta_full.get(asin)
        return d.get("title") if d else None
    # same-category candidate pool
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
    decoys = []
    for a in same_cat:
        if len(decoys) >= N_CORPUS - 1:
            break
        if title_of(a):
            decoys.append(a)
    corpus_asins = [ASIN] + decoys
    doc_texts = [title_of(a) or attrs_to_text(attrs_for_n(meta[a], N_ATTRS))
                 for a in corpus_asins]
    target_idx = 0
    log(f"corpus: {len(corpus_asins)} products (target idx {target_idx}, "
        f"same-category decoys)")

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
            mask = e["attention_mask"].unsqueeze(-1)
            v = (o.last_hidden_state * mask).sum(1) / mask.sum(1)
            vecs.append(F.normalize(v, dim=1).cpu().numpy())
        return np.concatenate(vecs, axis=0)

    doc_vecs = encode(doc_texts)

    def hit_at10(qtext):
        qv = encode([qtext])
        sims = qv @ doc_vecs.T
        order = np.argsort(-sims[0])[:TOP_K]
        rank = int(np.where(order == target_idx)[0][0]) + 1 \
            if target_idx in order else None
        return (1 if rank is not None else 0), rank

    results = {"users": {}, "template": {}}
    for u in users:
        hits = []
        ranks = []
        for q in queries_by_user[u]:
            h, rk = hit_at10(q)
            hits.append(h)
            ranks.append(rk)
        results["users"][u] = {
            "queries": queries_by_user[u],
            "hit10": hits,
            "mean_hit10": round(float(np.mean(hits)), 4),
            "ranks": ranks,
            "mean_rank": round(float(np.mean([x for x in ranks
                                              if x is not None])), 2) if
            any(x is not None for x in ranks) else None,
        }
        log(f"  user {u[:10]}: hit@10={hits} mean={np.mean(hits):.2f} "
            f"ranks={ranks}")
    tmpl_hits = []
    tmpl_ranks = []
    for q in tmpl_qs:
        h, rk = hit_at10(q)
        tmpl_hits.append(h)
        tmpl_ranks.append(rk)
    results["template"]["queries"] = tmpl_qs
    results["template"]["hit10"] = tmpl_hits
    results["template"]["mean_hit10"] = round(float(np.mean(tmpl_hits)), 4)
    results["template"]["ranks"] = tmpl_ranks
    log(f"  template: hit@10={tmpl_hits} mean={np.mean(tmpl_hits):.2f} "
        f"ranks={tmpl_ranks}")

    # ---- diff across style queries
    user_means = [results["users"][u]["mean_hit10"] for u in users]
    tmean = results["template"]["mean_hit10"]
    diff_style = float(np.max(user_means) - np.min(user_means))
    diff_vs_tmpl = float(np.mean(user_means) - tmean)
    log(f"hit@10 diff across {len(users)} style queries: "
        f"max={np.max(user_means):.3f} min={np.min(user_means):.3f} "
        f"diff={diff_style:.3f}")
    log(f"hit@10 mean-style vs template: {np.mean(user_means):.3f} - "
        f"{tmean:.3f} = {diff_vs_tmpl:+.3f}")

    result = {
        "version": "e22_t3_retr_style_v1", "seed": SEED,
        "asin": ASIN, "attrs": attrs,
        "config": {"X": N_USERS, "Y": 5, "Z": 5, "w": 20},
        "corpus_size": len(corpus_asins), "top_k": TOP_K,
        "target_idx": target_idx,
        "content_purity": round(n_content_ok / max(1, n_total_q), 4),
        "users": results["users"], "template": results["template"],
        "hit10_diff_across_styles": round(diff_style, 4),
        "hit10_mean_style": round(float(np.mean(user_means)), 4),
        "hit10_template": tmean,
        "hit10_style_minus_template": round(diff_vs_tmpl, 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log(f"wrote {OUT}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
