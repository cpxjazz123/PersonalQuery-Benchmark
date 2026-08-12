#!/usr/bin/env python3
"""E3 — Token Jaccard stratification across 3 retrievers × 3 domains.

For each (uid, asin) pair:
- Compute token-Jaccard between accepted query and doc text
- Stratify into low (<20%) / med (20-40%) / high (>40%)
- Per stratum: per-(retriever, domain) Hit@10 + ΔRange (across retrievers)

Uses E1 multi_index caches (BGE/E5/MiniLM) for full-quantity retrieval.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

E1_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
QUERY_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/04_query")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E3_string_overlap")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]
LOW, HIGH = 0.2, 0.4

TOKEN_RE = re.compile(r"[a-z0-9]+")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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


def retrieve_topk(emb: np.ndarray, doc_embs: np.ndarray, doc_ids: List[str], top_k: int = 10) -> List[str]:
    import torch
    from sentence_transformers import util
    q = torch.from_numpy(emb).float().unsqueeze(0)
    d = torch.from_numpy(doc_embs).float()
    scores = util.cos_sim(q, d)[0]
    _, idx = torch.topk(scores, k=min(top_k, len(doc_ids)))
    return [doc_ids[i] for i in idx.tolist()]


def encode_queries(retriever, texts: List[str]) -> np.ndarray:
    import torch
    with torch.no_grad():
        return retriever.encode_queries(texts, batch_size=64)


def compute_hit_at_1(retrieved: List[str], gold: str) -> float:
    return 1.0 if retrieved and retrieved[0] == gold else 0.0


def compute_hit_at_10(retrieved: List[str], gold: str) -> float:
    return 1.0 if gold in retrieved[:10] else 0.0


def stratify(j: float) -> str:
    if j < LOW:
        return "low"
    elif j < HIGH:
        return "med"
    return "high"


def process_category(category: str) -> Dict:
    log(f"\n=== {category} ===")
    idx = load_index(category)
    asins = idx["asins"]
    texts = idx["texts"]
    doc_embs = idx["embeddings"]
    asin_to_text = {a: t for a, t in zip(asins, texts)}

    accepted = load_accepted_query(category)
    log(f"  accepted_pairs={len(accepted)}, in_index={sum(1 for k in accepted if k[1] in asin_to_text)}")

    # Compute Jaccard per pair
    pair_data = []
    for (uid, asin), sq in accepted.items():
        if asin not in asin_to_text:
            continue
        q_text = sq["query"]
        d_text = asin_to_text[asin]
        q_tok = tokenize(q_text)
        d_tok = tokenize(d_text)
        j = jaccard(q_tok, d_tok)
        pair_data.append({
            "uid": uid, "asin": asin, "query": q_text,
            "doc_text": d_text, "jaccard": j, "stratum": stratify(j),
        })
    log(f"  paired_with_doc={len(pair_data)}")
    # jaccard distribution
    j_by_strat = defaultdict(int)
    for p in pair_data:
        j_by_strat[p["stratum"]] += 1
    log(f"  strata: low={j_by_strat['low']} med={j_by_strat['med']} high={j_by_strat['high']}")

    # Encode queries once per retriever
    import torch
    import sys as _sys
    _sys.path.insert(0, "/fs04/ar57/wenyu/PersoanlQuery/06_retrieval")
    from utils.retrievers import BGERetriever, E5Retriever, DenseRetriever, STARRetriever, ANCERetriever
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    cls_map = {
        "bge": BGERetriever, "e5": E5Retriever, "minilm": DenseRetriever,
        "star": STARRetriever, "ance": ANCERetriever,
    }
    model_map = {
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "star": "BAAI/bge-base-en-v1.5",
        "ance": "castorini/ance-msmarco-passage",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retrievers = {}
    q_encoders = {}
    q_embs = {}
    q_texts = [p["query"] for p in pair_data]
    # Build fresh STAR/ANCE indexes on the 10K subset
    docs_list = None
    if "star" in RETRIEVERS or "ance" in RETRIEVERS:
        docs_list = [{"asin": a, "title": (texts[i].split(".")[0] if texts[i] else "")[:300],
                      "brand": "", "feature": "", "description": texts[i][:500]}
                     for i, a in enumerate(asins)]
        asin_to_meta_local = {a: docs_list[i] for i, a in enumerate(asins)}
    for name in RETRIEVERS:
        if name in ("bge", "e5", "minilm"):
            # use doc_embs from cache
            pass
        elif name == "star":
            log(f"  [star] fitting on 10K docs...")
            obj = STARRetriever(model_name="BAAI/bge-base-en-v1.5")
            obj.fit(docs_list, asin_to_meta_local)
            retrievers[name] = obj
            with torch.no_grad():
                star_doc_embs = obj.encode_queries([d["title"] + " " + d["description"] for d in docs_list], batch_size=64)
            doc_embs[name] = star_doc_embs
            log(f"  [star] done")
        elif name == "ance":
            log(f"  [ance] fitting on 10K docs...")
            obj = ANCERetriever(model_name="castorini/ance-msmarco-passage")
            obj.fit(docs_list, asin_to_meta_local)
            retrievers[name] = obj
            with torch.no_grad():
                ance_doc_embs = obj.encode_queries([d["title"] + " " + d["description"] for d in docs_list], batch_size=64)
            doc_embs[name] = ance_doc_embs
            log(f"  [ance] done")
        qobj = cls_map[name](model_name=model_map[name])
        qobj.device = device
        q_encoders[name] = qobj
        q_embs[name] = encode_queries(qobj, q_texts)
        log(f"  [{name}] encoded {len(q_texts)} queries")
    # doc_embs as torch tensors on device
    doc_embs_t = {name: torch.from_numpy(doc_embs[name]).float().to(device) for name in RETRIEVERS}

    # Per-(stratum, retriever) H@10
    by_stratum: Dict[str, List[Dict]] = defaultdict(list)
    for p in pair_data:
        by_stratum[p["stratum"]].append(p)
    summary: Dict[str, Dict[str, Dict]] = {}
    for strat, items in by_stratum.items():
        summary[strat] = {}
        for name in RETRIEVERS:
            hits = []
            for i, p in enumerate(items):
                q_idx = pair_data.index(p)  # slow but ok
                emb = q_embs[name][q_idx]
                retrieved = retrieve_topk_with_t(emb, doc_embs_t[name], asins)
                hits.append(compute_hit_at_10(retrieved, p["asin"]))
            summary[strat][name] = {"n": len(items), "H@10": float(np.mean(hits)) if hits else 0.0}
        log(f"  {strat}: bge={summary[strat]['bge']['H@10']:.4f} e5={summary[strat]['e5']['H@10']:.4f} minilm={summary[strat]['minilm']['H@10']:.4f} n={summary[strat]['bge']['n']}")

    # Compute ΔRange per stratum (max-min across retrievers)
    for strat in summary:
        hs = [summary[strat][n]["H@10"] for n in RETRIEVERS]
        summary[strat]["delta_range"] = max(hs) - min(hs)

    # overlap stats
    j_vals = [p["jaccard"] for p in pair_data]

    del retrievers, q_encoders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "category": category,
        "n_pairs": len(pair_data),
        "strata_counts": dict(j_by_strat),
        "jaccard_mean": float(np.mean(j_vals)) if j_vals else 0.0,
        "jaccard_std": float(np.std(j_vals)) if j_vals else 0.0,
        "summary_by_stratum": summary,
    }


def retrieve_topk_with_t(emb: np.ndarray, doc_embs_t, doc_ids: List[str], top_k: int = 10) -> List[str]:
    import torch
    from sentence_transformers import util
    q = torch.from_numpy(emb).float().to(doc_embs_t.device).unsqueeze(0)
    scores = util.cos_sim(q, doc_embs_t)[0]
    _, idx = torch.topk(scores, k=min(top_k, len(doc_ids)))
    return [doc_ids[i] for i in idx.tolist()]


def write_summary_md(sums: List[Dict]) -> str:
    ret_headers = " | ".join(f"{r.upper()} H@10" for r in RETRIEVERS)
    rows = ["# E3 — Token Jaccard Stratification (full-quantity)\n",
            "**Method** — per-(uid, asin) Jaccard between accepted query and target doc text; stratify low <0.2 / med 0.2-0.4 / high ≥0.4; recompute H@10 per stratum.\n",
            f"| Category | Stratum | n | {ret_headers} | ΔRange |",
            f"|---|---|---|{'|'.join([''] * len(RETRIEVERS))}|---|"]
    # fix header separator
    rows[2] = f"| Category | Stratum | n | {ret_headers} | ΔRange |"
    rows[3] = f"|---|---:|---:|{'|'.join(['---:'] * len(RETRIEVERS))}|---:|"
    for s in sums:
        cat = s["category"]
        for strat in ["low", "med", "high"]:
            d = s["summary_by_stratum"].get(strat)
            if not d:
                continue
            vals = " | ".join(f"{d[r]['H@10']:.4f}" for r in RETRIEVERS)
            rows.append(
                f"| {cat} | {strat} | {d['bge']['n']} | {vals} | {d.get('delta_range', 0):.4f} |"
            )
        rows.append(
            f"| {cat} | ALL | {s['n_pairs']} | mean_jaccard={s['jaccard_mean']:.4f} (std={s['jaccard_std']:.4f}) |  |"
        )
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
            with open(OUT_DIR / f"{cat}_strata.json", "w") as f:
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