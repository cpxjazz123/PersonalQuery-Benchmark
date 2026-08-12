#!/usr/bin/env python3
"""E4 — 3-variant pipeline ablation across 3 domains × 5 retrievers.

Variants:
- A (standard): correct query only (no style filter, no error injection)
- B (noisy injection): post-07_inject_noisy queries (error injection added)
- C (style-filtered, abstract): user-level style filter applied (use most abstract
      syntax_depth_queries[i] for each pair)

Uses E1 multi_index caches (10K asins per domain).
Reports per-(variant, retriever, domain) H@10 + ΔRange.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

E1_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
QUERY_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/04_query")
NOISE_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/07_inject_noisy")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E4_ablations")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_index(category: str) -> Dict:
    with open(E1_DIR / f"multi_index_{category}.pkl", "rb") as f:
        return pickle.load(f)


def load_accepted_query(category: str) -> Dict[Tuple[str, str], Dict]:
    p = QUERY_DIR / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    out = {}
    for r in data:
        key = (r.get("user_id"), r.get("asin"))
        for sq in r.get("syntax_depth_queries", []):
            if sq.get("query") and sq.get("accepted_candidate_index") is not None:
                out[key] = sq
                break
    return out


def load_abstract_query(category: str) -> Dict[Tuple[str, str], Dict]:
    """Variant C: most abstract (longest, fewest concrete words) — pick syntax_depth_queries[9]."""
    p = QUERY_DIR / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    out = {}
    for r in data:
        key = (r.get("user_id"), r.get("asin"))
        sqs = r.get("syntax_depth_queries", [])
        # pick the last (most abstract / longest)
        if sqs:
            out[key] = sqs[-1]
    return out


def load_noisy_pairs(category: str) -> Dict[Tuple[str, str], List[Dict]]:
    p = NOISE_DIR / category / "noisy_query.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    out = defaultdict(list)
    for d in data:
        out[(d.get("uid"), d.get("asin"))].append(d)
    return out


def retrieve_topk(emb, doc_embs_t, doc_ids, top_k=10):
    import torch
    from sentence_transformers import util
    q = torch.from_numpy(emb).float().to(doc_embs_t.device).unsqueeze(0)
    scores = util.cos_sim(q, doc_embs_t)[0]
    _, idx = torch.topk(scores, k=min(top_k, len(doc_ids)))
    return [doc_ids[i] for i in idx.tolist()]


def process_category(category: str) -> Dict:
    log(f"\n=== {category} ===")
    idx = load_index(category)
    asins = idx["asins"]
    doc_embs = idx["embeddings"]
    accepted = load_accepted_query(category)
    abstract = load_abstract_query(category)
    noisy = load_noisy_pairs(category)

    # Filter pairs that exist in the index
    asin_set = set(asins)
    pairs = [(k, v) for k, v in accepted.items() if k[1] in asin_set]
    log(f"  pairs_in_index={len(pairs)}")
    abstract_pairs = [(k, v) for k, v in abstract.items() if k[1] in asin_set]
    noisy_pairs = [(k, vs) for k, vs in noisy.items() if k[1] in asin_set]
    log(f"  noisy_pairs_in_index={len(noisy_pairs)} (avg {sum(len(v) for _, v in noisy_pairs)/max(1, len(noisy_pairs)):.2f} noisy/pair)")

    import torch
    sys.path.insert(0, "/fs04/ar57/wenyu/PersoanlQuery/06_retrieval")
    from utils.retrievers import BGERetriever, E5Retriever, DenseRetriever, STARRetriever, ANCERetriever
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    cls_map = {"bge": BGERetriever, "e5": E5Retriever, "minilm": DenseRetriever,
               "star": STARRetriever, "ance": ANCEReti萃rer if False else ANCERetriever}
    model_map = {
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "star": "BAAI/bge-base-en-v1.5",
        "ance": "castorini/ance-msmarco-passage",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = {}
    docs_list = [{"asin": a, "title": (idx["texts"][i].split(".")[0] if idx["texts"][i] else "")[:300],
                  "brand": "", "feature": "", "description": idx["texts"][i][:500]}
                 for i, a in enumerate(asins)]
    asin_to_meta = {a: docs_list[i] for i, a in enumerate(asins)}

    for name in RETRIEVERS:
        encoders[name] = cls_map[name](model_name=model_map[name])
        encoders[name].device = device

    # Encode queries per variant
    a_texts = [v["query"] for _, v in pairs]
    c_texts = [v.get("query") or "" for _, v in abstract_pairs]
    # For B (noisy): one entry per (uid, asin, noisy_pair_idx)
    b_texts = []
    b_lookup = []
    for k, vs in noisy_pairs:
        for nv in vs:
            t = nv.get("noisy_query") or ""
            b_texts.append(t)
            b_lookup.append((k[0], k[1]))

    a_embs = {n: encoders[n].encode_queries(a_texts, batch_size=64) for n in RETRIEVERS}
    c_embs = {n: encoders[n].encode_queries(c_texts, batch_size=64) for n in RETRIEVERS}
    b_embs = {n: encoders[n].encode_queries(b_texts, batch_size=64) for n in RETRIEVERS}
    log(f"  encoded: A={len(a_texts)} C={len(c_texts)} B={len(b_texts)}")
    del encoders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build per-retriever doc_embs_t (use cache for bge/e5/minilm; fresh-fit for star/ance)
    doc_embs_full = dict(doc_embs)
    if "star" in RETRIEVERS:
        log(f"  [star] fitting...")
        star = STARRetriever(model_name="BAAI/bge-base-en-v1.5")
        star.fit(docs_list, asin_to_meta)
        with torch.no_grad():
            doc_embs_full["star"] = star.encode_queries([d["title"] + " " + d["description"] for d in docs_list], batch_size=64)
        del star
    if "ance" in RETRIEVERS:
        log(f"  [ance] fitting...")
        ance = ANCERetriever(model_name="castorini/ance-msmarco-passage")
        ance.fit(docs_list, asin_to_meta)
        with torch.no_grad():
            doc_embs_full["ance"] = ance.encode_queries([d["title"] + " " + d["description"] for d in docs_list], batch_size=64)
        del ance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    doc_embs_t = {n: torch.from_numpy(doc_embs_full[n]).float().to(device) for n in RETRIEVERS}

    # Compute H@10 per (variant, retriever)
    summary = {"A_standard": {}, "B_noisy_injection": {}, "C_abstract": {}}
    # Variant A: correct
    for name in RETRIEVERS:
        hits = []
        for i, (k, v) in enumerate(pairs):
            retrieved = retrieve_topk(a_embs[name][i], doc_embs_t[name], asins)
            hits.append(1.0 if k[1] in retrieved[:10] else 0.0)
        summary["A_standard"][name] = {"n": len(pairs), "H@10": float(np.mean(hits)) if hits else 0.0}
    # Variant C: abstract
    for name in RETRIEVERS:
        hits = []
        for i, (k, v) in enumerate(abstract_pairs):
            if not (v.get("query") or "").strip():
                continue
            retrieved = retrieve_topk(c_embs[name][i], doc_embs_t[name], asins)
            hits.append(1.0 if k[1] in retrieved[:10] else 0.0)
        summary["C_abstract"][name] = {"n": len(abstract_pairs), "H@10": float(np.mean(hits)) if hits else 0.0}
    # Variant B: noisy
    # Build per-pair H@10 mean
    for name in RETRIEVERS:
        per_pair_hits = defaultdict(list)
        for i, (uid, asin) in enumerate(b_lookup):
            retrieved = retrieve_topk(b_embs[name][i], doc_embs_t[name], asins)
            per_pair_hits[(uid, asin)].append(1.0 if asin in retrieved[:10] else 0.0)
        means = [np.mean(v) for v in per_pair_hits.values()]
        summary["B_noisy_injection"][name] = {"n": len(per_pair_hits), "H@10": float(np.mean(means)) if means else 0.0}
    log(f"  A: bge={summary['A_standard']['bge']['H@10']:.4f} C: bge={summary['C_abstract']['bge']['H@10']:.4f} B: bge={summary['B_noisy_injection']['bge']['H@10']:.4f}")

    del doc_embs_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "category": category,
        "n_pairs_correct": len(pairs),
        "n_pairs_abstract": len(abstract_pairs),
        "n_pairs_noisy": summary["B_noisy_injection"]["bge"]["n"],
        "summary": summary,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E4 — 3-variant pipeline ablation (full-quantity)\n",
            "**Variants**: A=standard correct, B=noisy injection, C=most-abstract syntax_depth\n",
            f"| Category | Variant | n | {' | '.join(f'{r.upper()} H@10' for r in RETRIEVERS)} | ΔRange |",
            f"|---|---|---:|{'|'.join(['---:'] * len(RETRIEVERS))}|---:|"]
    for s in sums:
        cat = s["category"]
        for var in ["A_standard", "B_noisy_injection", "C_abstract"]:
            d = s["summary"][var]
            n = d["bge"]["n"]
            vals = " | ".join(f"{d[r]['H@10']:.4f}" for r in RETRIEVERS)
            hs = [d[r]["H@10"] for r in RETRIEVERS]
            dr = max(hs) - min(hs)
            rows.append(f"| {cat} | {var} | {n} | {vals} | {dr:.4f} |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary_full.md"
    with open(p, "w") as f:
        f.write("\n".join(rows))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sums = []
    for cat in args.categories:
        try:
            s = process_category(cat)
            with open(OUT_DIR / f"{cat}_ablation.json", "w") as f:
                json.dump(s, f, indent=2, default=str)
            sums.append(s)
        except Exception as e:
            log(f"ERROR {cat}: {e}")
            import traceback
            traceback.print_exc()
    if sums:
        p = write_summary_md(sums)
        log(f"wrote {p}")
    log("=== done ===")


if __name__ == "__main__":
    main()