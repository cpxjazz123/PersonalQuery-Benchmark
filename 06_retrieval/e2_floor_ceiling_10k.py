#!/usr/bin/env python3
"""E2 — Full-quantity floor/ceiling using 10K cached indexes (BGE/E5/MiniLM) +
fresh BM25 + STAR/ANCE/SPLADE on the same 10K asin subset.

This avoids re-fitting 71K docs per dense retriever (~6min each).
Uses E1_multimetric/multi_index_<cat>.pkl (10018/10025/10017 asins).
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

HF_HOME = "/home/wlia0047/.cache/huggingface/hub"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HOME
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.retrievers import BM25, STARRetriever, ANCERetriever, SPLADERetriever  # noqa: E402
from utils.retrievers import BGERetriever, E5Retriever, DenseRetriever  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E2_floor_ceiling")
E1_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
CORPUS_ROOT = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
N_SAMPLES = 500
SEED = 42
TOP_K = 10


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_multi_index(category: str) -> Dict:
    with open(E1_DIR / f"multi_index_{category}.pkl", "rb") as f:
        return pickle.load(f)


def make_queries_from_asins(asins: List[str], texts: List[str], n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    """Floor: random_other_doc_text -> target. Ceiling: doc text -> doc itself."""
    floor = []
    for i in range(n):
        target = rng.choice(asins)
        for _ in range(10):
            qa = rng.choice(asins)
            if qa != target:
                break
        floor.append({"query_id": f"floor_{i}", "target_asin": target, "query_idx": asins.index(qa)})
    ceiling = []
    cands_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    for i in range(n):
        idx = rng.choice(cands_idx)
        ceiling.append({"query_id": f"ceil_{i}", "target_asin": asins[idx], "query_idx": idx})
    return floor, ceiling


def retrieve_with_dense_emb(emb: np.ndarray, doc_embs: np.ndarray, doc_asins: List[str], top_k: int = TOP_K) -> List[str]:
    """cosine sim top-k."""
    q = torch.from_numpy(emb).float()
    if q.ndim == 1:
        q = q.unsqueeze(0)
    d = torch.from_numpy(doc_embs).float()
    from sentence_transformers import util
    scores = util.cos_sim(q.to(d.device), d)[0]
    _, idx = torch.topk(scores, k=min(top_k, len(doc_asins)))
    return [doc_asins[i] for i in idx.tolist()]


def encode_queries_batch(retriever, texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        return retriever.encode_queries(texts, batch_size=64)


class CachedDenseSearcher:
    """For BGE/E5/MiniLM — pre-encode all 200 floor + 200 ceiling queries with the cached doc index."""
    def __init__(self, retriever_name: str, doc_embs: np.ndarray, doc_asins: List[str]):
        self.name = retriever_name
        self.doc_embs = doc_embs
        self.doc_asins = doc_asins
        # load retriever for query encoding
        cls_map = {"bge": BGERetriever, "e5": E5Retriever, "minilm": DenseRetriever}
        model_map = {
            "bge": "BAAI/bge-large-en-v1.5",
            "e5": "intfloat/e5-large-v2",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        self.retr = cls_map[retriever_name](model_name=model_map[retriever_name])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retr.device = self.device
        self.doc_embs_t = torch.from_numpy(doc_embs).float().to(self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            return self.retr.encode_queries(texts, batch_size=64)

    def search(self, query_text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        # encode single query
        emb = self.encode([query_text])[0]
        q = torch.from_numpy(emb).float().to(self.device).unsqueeze(0)
        from sentence_transformers import util
        scores = util.cos_sim(q, self.doc_embs_t)[0]
        _, idx = torch.topk(scores, k=min(top_k, len(self.doc_asins)))
        return [(self.doc_asins[i], float(scores[i])) for i in idx.tolist()]


class FreshSearcher:
    """For BM25/STAR/ANCE/SPLADE — fits on the 10K subset, then searches."""
    def __init__(self, name: str, docs: List[Dict], asin_to_meta: Dict):
        self.name = name
        self.docs = docs
        self.asin_to_meta = asin_to_meta
        if name == "bm25":
            self.obj = BM25()
            self.obj.fit(docs, asin_to_meta)
        elif name == "star":
            self.obj = STARRetriever(model_name="BAAI/bge-base-en-v1.5")
            self.obj.fit(docs, asin_to_meta)
        elif name == "ance":
            self.obj = ANCERetriever(model_name="castorini/ance-msmarco-passage")
            self.obj.fit(docs, asin_to_meta)
        elif name == "splade":
            self.obj = SPLADERetriever()
            self.obj.fit(docs, asin_to_meta)
        else:
            raise ValueError(name)

    def search(self, query_text: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        return self.obj.search(query_text, top_k=top_k)


def hit_at_k(retrieved: List[Tuple[str, float]], gold: str, k: int = TOP_K) -> int:
    return 1 if gold in [a for a, _ in retrieved[:k]] else 0


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"hit_at_10": 0.0, "p_at_10": 0.0, "mr_at_10": 0.0, "n": 0}
    hit = sum(r["hit"] for r in results) / n
    mrr = 0.0
    for r in results:
        for rank, (a, _) in enumerate(r["retrieved"][:TOP_K], start=1):
            if a == r["relevant_asin"]:
                mrr += 1.0 / rank
                break
    return {"hit_at_10": hit, "p_at_10": hit, "mr_at_10": mrr / n, "n": n}


def run_for_category(category: str, n: int, seed: int, retrievers: List[str]) -> Dict:
    log(f"\n{'='*70}\nE2 — {category} (n={n})\n{'='*70}")
    idx = load_multi_index(category)
    asins = idx["asins"]
    texts = idx["texts"]
    doc_embs = idx["embeddings"]
    log(f"  index: {len(asins)} asins, retrievers in cache: {list(doc_embs.keys())}")

    rng = random.Random(seed)
    floor, ceiling = make_queries_from_asins(asins, texts, n, rng)

    # Build retriever searchers
    searchers: Dict[str, object] = {}
    for name in retrievers:
        try:
            if name in ("bge", "e5", "minilm"):
                searchers[name] = CachedDenseSearcher(name, doc_embs[name], asins)
                log(f"  [{name}] cached index loaded")
            elif name in ("bm25", "star", "ance", "splade"):
                # build docs list from texts
                docs = [{"asin": a, "title": (texts[i].split(".")[0] if texts[i] else "")[:300],
                         "brand": "", "feature": "", "description": texts[i][:500]}
                        for i, a in enumerate(asins)]
                asin_to_meta = {a: docs[i] for i, a in enumerate(asins)}
                searchers[name] = FreshSearcher(name, docs, asin_to_meta)
                log(f"  [{name}] fresh fit done on {len(docs)} docs")
        except Exception as e:
            log(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    results: Dict[str, Dict] = {}
    for name, searcher in searchers.items():
        log(f"  [{name}] floor")
        t0 = time.time()
        fl_res = []
        for q in floor:
            qt = texts[q["query_idx"]]
            if not qt.strip():
                continue
            retrieved = searcher.search(qt, top_k=TOP_K)
            fl_res.append({"query_id": q["query_id"], "relevant_asin": q["target_asin"],
                           "retrieved": retrieved, "hit": hit_at_k(retrieved, q["target_asin"])})
        log(f"  [{name}] floor done in {time.time()-t0:.1f}s")
        t0 = time.time()
        log(f"  [{name}] ceiling")
        cl_res = []
        for q in ceiling:
            qt = texts[q["query_idx"]]
            if not qt.strip():
                continue
            retrieved = searcher.search(qt, top_k=TOP_K)
            cl_res.append({"query_id": q["query_id"], "relevant_asin": q["target_asin"],
                           "retrieved": retrieved, "hit": hit_at_k(retrieved, q["target_asin"])})
        log(f"  [{name}] ceiling done in {time.time()-t0:.1f}s")
        results[name] = {"floor_metrics": compute_metrics(fl_res),
                         "ceiling_metrics": compute_metrics(cl_res)}
        del searcher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "category": category,
        "n_samples": n,
        "n_docs": len(asins),
        "results": results,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E2 — Floor / Ceiling (10K cache, full-quantity)",
            "",
            "**Method** — floor: random_other_doc text → target_asin. ceiling: asin own text → asin.",
            "**Index** — 10K Amazon meta distractors + gold asins per category.",
            "",
            "| Category | Retriever | Floor H@10 | Ceiling H@10 | Floor MRR | Ceiling MRR | n |",
            "|---|---|---|---|---|---|---|"]
    for s in sums:
        cat = s["category"]
        for name, m in s["results"].items():
            fl = m["floor_metrics"]
            cl = m["ceiling_metrics"]
            rows.append(
                f"| {cat} | {name} | {fl['hit_at_10']:.4f} | {cl['hit_at_10']:.4f} | "
                f"{fl['mr_at_10']:.4f} | {cl['mr_at_10']:.4f} | {s['n_samples']} |"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary_full_10k.md"
    with open(p, "w") as f:
        f.write("\n".join(rows))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--retrievers", nargs="+",
                    default=["bm25", "bge", "e5", "minilm", "star", "ance", "splade"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sums = []
    for cat in args.categories:
        try:
            s = run_for_category(cat, args.n, args.seed, args.retrievers)
            with open(OUT_DIR / f"{cat}_floor_ceiling_full_10k.json", "w") as f:
                json.dump(s, f, indent=2, default=str)
            log(f"  wrote {cat}_floor_ceiling_full_10k.json")
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