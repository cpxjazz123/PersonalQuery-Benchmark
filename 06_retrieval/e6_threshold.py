#!/usr/bin/env python3
"""E6 — Threshold sensitivity (quality score percentile sweep).

Quality score per (uid, asin, query): mean PCA dim distance to user centroid
from 12_complexity. Lower distance = more aligned with user style.
Threshold q ∈ {0.90, 0.95, 0.99} keeps the bottom-fraction (best aligned).

For each q:
- Reject rate
- n_kept per category
- Downstream: per-retriever H@10 on kept set (BGE/E5/MiniLM)
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
COMPLEX_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E6_threshold")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
QUANTILES = [0.90, 0.95, 0.99]
RETRIEVERS = ["bge", "e5", "minilm"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_index(category: str) -> Dict:
    with open(E1_DIR / f"multi_index_{category}.pkl", "rb") as f:
        return pickle.load(f)


def load_complexity(category: str) -> List[Dict]:
    p = COMPLEX_DIR / category / "strict5550_query_gmm_features.jsonl"
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def compute_user_score(records: List[Dict]) -> Dict[Tuple[str, str], float]:
    """Per (uid, asin): how well does this query align with the user's other queries?

    For each (uid), compute centroid of all PCA embeddings. Score per (uid, asin) =
    cosine similarity between this query's PCA and user centroid.
    """
    # group by user
    user_pcas: Dict[str, List[np.ndarray]] = defaultdict(list)
    record_lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for r in records:
        uid = r["user_id"]
        pca = np.asarray(r["pca_embedding"], dtype=np.float32)
        user_pcas[uid].append(pca)
        record_lookup[(uid, r["asin"])] = pca
    # user centroids
    user_centroids: Dict[str, np.ndarray] = {}
    for uid, pcas in user_pcas.items():
        c = np.mean(pcas, axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        user_centroids[uid] = c
    # score per record
    scores: Dict[Tuple[str, str], float] = {}
    for (uid, asin), pca in record_lookup.items():
        n = pca / (np.linalg.norm(pca) + 1e-9)
        scores[(uid, asin)] = float(np.dot(n, user_centroids[uid]))
    return scores


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
    asin_set = set(asins)

    records = load_complexity(category)
    log(f"  complexity records: {len(records)}")

    scores = compute_user_score(records)
    # Filter to pairs that exist in index
    in_index = [(k, s) for k, s in scores.items() if k[1] in asin_set]
    log(f"  in_index: {len(in_index)}")

    # Compute thresholds
    score_vals = sorted([s for _, s in in_index])
    thresholds = {q: float(np.quantile(score_vals, q)) for q in QUANTILES}

    summary = {}
    import torch
    sys.path.insert(0, "/fs04/ar57/wenyu/PersoanlQuery/06_retrieval")
    from utils.retrievers import BGERetriever, E5Retriever, DenseRetriever
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    cls_map = {"bge": BGERetriever, "e5": E5Retriever, "minilm": DenseRetriever}
    model_map = {
        "bge": "BAAI/bge-large-en-v1.5",
        "e5": "intfloat/e5-large-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = {n: cls_map[n](model_name=model_map[n]) for n in RETRIEVERS}
    for n in RETRIEVERS:
        encoders[n].device = device

    # Pre-encode all queries once
    pairs = [k for k, _ in in_index]
    q_texts = [next(r["query_text"] for r in records if r["user_id"] == k[0] and r["asin"] == k[1])
               for k in pairs]
    q_embs = {n: encoders[n].encode_queries(q_texts, batch_size=64) for n in RETRIEVERS}
    log(f"  encoded {len(q_texts)} queries")
    del encoders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    doc_embs_t = {n: torch.from_numpy(doc_embs[n]).float().to(device) for n in RETRIEVERS}

    for q in QUANTILES:
        thr = thresholds[q]
        kept = [(i, k) for i, (k, s) in enumerate(in_index) if s >= thr]
        n_kept = len(kept)
        reject_rate = 1 - n_kept / len(in_index) if in_index else 0
        per_ret = {}
        for n in RETRIEVERS:
            hits = []
            for i, k in kept:
                retrieved = retrieve_topk(q_embs[n][i], doc_embs_t[n], asins)
                hits.append(1.0 if k[1] in retrieved[:10] else 0.0)
            per_ret[n] = {"n": n_kept, "H@10": float(np.mean(hits)) if hits else 0.0}
        avg_h = float(np.mean([per_ret[n]["H@10"] for n in RETRIEVERS]))
        summary[f"q={q}"] = {
            "threshold": thr,
            "n_kept": n_kept,
            "n_total": len(in_index),
            "reject_rate": reject_rate,
            "per_retriever": per_ret,
            "avg_H@10": avg_h,
        }
        log(f"  q={q}: kept={n_kept} reject={reject_rate:.2%} avg_H@10={avg_h:.4f}")

    del doc_embs_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "category": category,
        "n_total_in_index": len(in_index),
        "thresholds": thresholds,
        "by_quantile": summary,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E6 — Threshold sensitivity (full-quantity)\n",
            "**Method** — quality score = cosine similarity between query PCA and user centroid; thresholds at quantiles {0.90, 0.95, 0.99}.\n",
            "| Category | q | n_kept | reject_rate | BGE H@10 | E5 H@10 | MiniLM H@10 | Avg H@10 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in sums:
        cat = s["category"]
        for q in s["by_quantile"]:
            d = s["by_quantile"][q]
            pr = d["per_retriever"]
            rows.append(
                f"| {cat} | {q} | {d['n_kept']} | {d['reject_rate']:.2%} | "
                f"{pr['bge']['H@10']:.4f} | {pr['e5']['H@10']:.4f} | {pr['minilm']['H@10']:.4f} | "
                f"{d['avg_H@10']:.4f} |"
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
            with open(OUT_DIR / f"{cat}_threshold.json", "w") as f:
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