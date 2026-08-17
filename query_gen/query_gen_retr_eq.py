#!/usr/bin/env python3
"""query_gen_retr_eq — Retrieval-based attribute-VALUE equivalence discovery.

User directive: find attribute keys k such that two users BOTH use k but with
DIFFERENT values, yet the retrieval results are nearly the same ("大家都使用
这个属性，但结果差不多" => the attribute's VALUE is retrieval-insensitive /
"平权").

Method:
  For each attribute key k (top-10):
    for ~P pairs of products (A, B):
      - queryA = A's attributes, but with k's value replaced by A's own k value
      - queryB = A's attributes, but with k's value replaced by B's k value
        (everything else identical to A)
      - retrieve top-k over the product document collection with BGE
      - Jaccard(queryA_results, queryB_results)
  If mean Jaccard high (>= threshold) -> attribute k's values are
  retrieval-equivalent: users' choice of value for k barely affects what is
  retrieved. If low -> k's value is retrieval-sensitive (semantically load-
  bearing).

Also report pairwise (k1,k2): swap BOTH values -> still equivalent?

Outputs: result/e22_t3/e22_t3_retr_eq.json
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
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import load_meta, attrs_for_n, TOP_ATTR_KEYS, EXT_ATTR_KEYS  # noqa: E402

BGE_PATH = ("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
            "models--BAAI--bge-base-en-v1.5/snapshots/" +
            os.listdir("/home/wlia0047/ar57_scratch/wenyu/hf_models/"
                       "models--BAAI--bge-base-en-v1.5/snapshots/")[0])
OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_retr_eq.json"
SEED = 9999
N_PRODUCTS = 300
N_ATTR = 10
TOP_K = 20
JACCARD_EQ = 0.8
N_PAIRS = 60          # value-swap pairs per attribute


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_products() -> list[dict]:
    meta = load_meta()
    cands = []
    for asin, rec in meta.items():
        attrs = attrs_for_n(rec, N_ATTR)
        if attrs is not None:
            cands.append(attrs)
        if len(cands) >= N_PRODUCTS * 3:
            break
    rng = random.Random(SEED)
    rng.shuffle(cands)
    return cands[:N_PRODUCTS]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    random.seed(SEED)
    products = load_products()
    log(f"products: {len(products)}")
    keys = list(TOP_ATTR_KEYS) + EXT_ATTR_KEYS[:N_ATTR - 5]

    tok = AutoTokenizer.from_pretrained(BGE_PATH)
    model = AutoModel.from_pretrained(BGE_PATH)
    model.eval()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    def encode(texts: list[str], bs: int = 64) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), bs):
            e = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                o = model(**e)
            mask = e["attention_mask"].unsqueeze(-1)
            v = (o.last_hidden_state * mask).sum(1) / mask.sum(1)
            vecs.append(F.normalize(v, dim=1).cpu().numpy())
        return np.concatenate(vecs, axis=0)

    def attrs_to_text(attrs: dict) -> str:
        return " ".join(f"{k}: {v}" for k, v in attrs.items())

    # document collection: each product's attribute text
    doc_texts = [attrs_to_text(a) for a in products]
    doc_vecs = encode(doc_texts)

    def retrieve(texts: list[str]) -> list[set]:
        qv = encode(texts)
        sims = qv @ doc_vecs.T
        topk = np.argsort(-sims, axis=1)[:, :TOP_K]
        return [set(r.tolist()) for r in topk]

    rng = random.Random(SEED)
    # ---- per-attribute VALUE equivalence ----
    value_eq: dict[str, dict] = {}
    for k in keys:
        ov = []
        n_done = 0
        for _ in range(N_PAIRS * 4):
            if n_done >= N_PAIRS:
                break
            A = rng.choice(products)
            B = rng.choice(products)
            if A is B or k not in A or k not in B:
                continue
            if str(A[k]).lower() == str(B[k]).lower():
                continue
            qA = dict(A)
            qB = dict(A)
            qB[k] = B[k]          # only k's value differs
            resA, resB = retrieve([attrs_to_text(qA), attrs_to_text(qB)])
            ov.append(jaccard(resA, resB))
            n_done += 1
        if not ov:
            value_eq[k] = {"run": False}
            continue
        ov = np.array(ov)
        value_eq[k] = {
            "run": True, "n_pairs": len(ov),
            "mean_jaccard": round(float(ov.mean()), 4),
            "pct_equivalent": round(float((ov >= JACCARD_EQ).mean()), 4),
            "median": round(float(np.median(ov)), 4),
            "p10": round(float(np.percentile(ov, 10)), 4),
            "value_insensitive": bool(ov.mean() >= JACCARD_EQ),
        }
        log(f"  value-eq[{k}]: mean={value_eq[k]['mean_jaccard']:.3f} "
            f"p10={value_eq[k]['p10']:.3f} "
            f"insensitive={value_eq[k]['value_insensitive']}")

    # ---- pairwise: swap BOTH k1 and k2 values -> still equivalent? ----
    pair_eq: dict[str, dict] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            ov = []
            n_done = 0
            for _ in range(N_PAIRS * 4):
                if n_done >= N_PAIRS // 2:
                    break
                A = rng.choice(products)
                B = rng.choice(products)
                if A is B or k1 not in A or k2 not in A:
                    continue
                if k1 not in B or k2 not in B:
                    continue
                if (str(A[k1]).lower() == str(B[k1]).lower()
                        and str(A[k2]).lower() == str(B[k2]).lower()):
                    continue
                qA = dict(A)
                qB = dict(A)
                qB[k1] = B[k1]
                qB[k2] = B[k2]
                resA, resB = retrieve([attrs_to_text(qA), attrs_to_text(qB)])
                ov.append(jaccard(resA, resB))
                n_done += 1
            if not ov:
                continue
            ov = np.array(ov)
            pair_eq[f"{k1}|{k2}"] = {
                "n_pairs": len(ov),
                "mean_jaccard": round(float(ov.mean()), 4),
                "equivalent": bool(ov.mean() >= JACCARD_EQ),
            }

    result = {
        "version": "query_gen_retr_eq_v2",
        "seed": SEED, "n_products": len(products),
        "n_attrs": N_ATTR, "top_k": TOP_K, "n_pairs": N_PAIRS,
        "jaccard_eq_threshold": JACCARD_EQ,
        "keys": keys,
        "value_equivalence": value_eq,
        "pairwise_swap_equivalence": pair_eq,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    log("done: " + str(OUT))
    insens = [k for k, v in value_eq.items()
              if v.get("run") and v.get("value_insensitive")]
    sens = [k for k, v in value_eq.items()
            if v.get("run") and not v.get("value_insensitive")]
    log(f"VALUE-INSENSITIVE (平权): {insens}")
    log(f"VALUE-SENSITIVE (敏感): {sens}")


if __name__ == "__main__":
    main()
