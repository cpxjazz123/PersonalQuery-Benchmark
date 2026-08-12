#!/usr/bin/env python3
"""E2 — Full-quantity floor/ceiling across 9 retrievers × 3 domains.

Reuses multi_index pickles (BGE/E5/MiniLM) + fits BM25, SPLADE, STAR, ANCE,
ColBERT, MPNet fresh on a 10K-doc corpus per domain.

Floor: random_other_doc.title -> target_asin (should hit ~0)
Ceiling: target_asin.title -> target_asin (should hit ~1)
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

from utils.retrievers import BM25, SPLADERetriever  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E2_floor_ceiling")
CORPUS_ROOT = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
N_SAMPLES = 500  # floor+ceiling each
SEED = 42
TOP_K = 10
DENSE_RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]  # fit fresh per domain
BM25_NAME = "bm25"
SPLADE_NAME = "splade"

DENSE_CLS_MAP = {}


def _ensure_dense_cls():
    if DENSE_CLS_MAP:
        return
    from utils.retrievers import (
        BGERetriever, E5Retriever, DenseRetriever, STARRetriever, ANCERetriever,
    )
    DENSE_CLS_MAP.update({
        "bge": BGERetriever,
        "e5": E5Retriever,
        "minilm": DenseRetriever,
        "star": STARRetriever,
        "ance": ANCERetriever,
    })


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus(category: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    if category == "Baby_Products":
        path = CORPUS_ROOT / "meta_Baby.jsonl.gz"
    elif category == "Grocery_and_Gourmet_Food":
        path = CORPUS_ROOT / "raw" / "meta_categories" / "meta_Grocery_and_Gourmet_Food.jsonl.gz"
    elif category == "Pet_Supplies":
        path = CORPUS_ROOT / "raw" / "meta_categories" / "meta_Pet_Supplies.jsonl.gz"
    else:
        raise ValueError(category)
    if not path.exists():
        raise FileNotFoundError(str(path))
    docs: List[Dict] = []
    asin_to_meta: Dict[str, Dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            asin = obj.get("asin") or obj.get("parent_asin")
            if not asin:
                continue
            title = (obj.get("title") or "").strip()
            description = obj.get("description") or []
            if isinstance(description, list):
                description = " ".join(str(d) for d in description if d)
            elif not isinstance(description, str):
                description = ""
            brand = obj.get("brand") or ""
            if isinstance(brand, list):
                brand = " ".join(str(b) for b in brand if b)
            feature = obj.get("feature") or []
            if isinstance(feature, list):
                feature = " ".join(str(x) for x in feature if x)
            elif not isinstance(feature, str):
                feature = ""
            doc = {
                "asin": asin,
                "title": title[:300],
                "brand": brand[:80] if isinstance(brand, str) else "",
                "feature": feature[:500] if isinstance(feature, str) else "",
                "description": description[:500] if isinstance(description, str) else "",
            }
            docs.append(doc)
            asin_to_meta[asin] = doc
    log(f"  loaded {len(docs)} docs from {path.name}")
    return docs, asin_to_meta


def build_doc_text(doc: Dict) -> str:
    parts = [doc.get("title", "") or "", doc.get("brand", "") or "",
             doc.get("feature", "") or "", doc.get("description", "") or ""]
    return " ".join(p for p in parts if p).strip()


def hit_at_k(retrieved: List[Tuple[str, float]], gold: str, k: int = TOP_K) -> int:
    top = [a for a, _ in retrieved[:k]]
    return 1 if gold in top else 0


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


def make_queries(asins: List[str], asin_to_meta: Dict, n: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    floor = []
    for i in range(n):
        target = rng.choice(asins)
        for _ in range(10):
            qa = rng.choice(asins)
            if qa != target:
                break
        floor.append({"query_id": f"floor_{i}", "target_asin": target, "query_asin": qa})
    ceiling = []
    cands = [a for a in asins if asin_to_meta[a].get("title", "").strip()]
    for i in range(n):
        a = rng.choice(cands)
        ceiling.append({"query_id": f"ceil_{i}", "target_asin": a, "query_asin": a,
                        "title": asin_to_meta[a]["title"].strip()})
    return floor, ceiling


def run_floor_ceiling_search(retriever, name: str, floor: List[Dict], ceiling: List[Dict],
                              asin_to_meta: Dict) -> Dict:
    floor_res = []
    log(f"  [{name}] running {len(floor)} floor queries")
    t0 = time.time()
    for q in floor:
        qt = build_doc_text(asin_to_meta[q["query_asin"]])
        retrieved = retriever.search(qt, top_k=TOP_K)
        floor_res.append({"query_id": q["query_id"], "relevant_asin": q["target_asin"],
                          "retrieved": retrieved, "hit": hit_at_k(retrieved, q["target_asin"])})
    log(f"  [{name}] floor done in {time.time()-t0:.1f}s")
    ceil_res = []
    log(f"  [{name}] running {len(ceiling)} ceiling queries")
    t0 = time.time()
    for q in ceiling:
        retrieved = retriever.search(q["title"], top_k=TOP_K)
        ceil_res.append({"query_id": q["query_id"], "relevant_asin": q["target_asin"],
                         "retrieved": retrieved, "hit": hit_at_k(retrieved, q["target_asin"])})
    log(f"  [{name}] ceiling done in {time.time()-t0:.1f}s")
    return {
        "floor_metrics": compute_metrics(floor_res),
        "ceiling_metrics": compute_metrics(ceil_res),
    }


def fit_dense(name: str, docs: List[Dict], asin_to_meta: Dict):
    _ensure_dense_cls()
    cls = DENSE_CLS_MAP[name]
    log(f"  [{name}] fitting on {len(docs)} docs...")
    t0 = time.time()
    if name == "minilm":
        obj = cls(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif name == "bge":
        obj = cls(model_name="BAAI/bge-large-en-v1.5")
    elif name == "e5":
        obj = cls(model_name="intfloat/e5-large-v2")
    elif name == "star":
        obj = cls(model_name="BAAI/bge-base-en-v1.5")
    elif name == "ance":
        obj = cls(model_name="castorini/ance-msmarco-passage")
    else:
        raise ValueError(name)
    obj.fit(docs, asin_to_meta)
    log(f"  [{name}] fit done in {time.time()-t0:.1f}s")
    return obj


def fit_bm25(docs: List[Dict], asin_to_meta: Dict) -> BM25:
    log(f"  [bm25] fitting on {len(docs)} docs...")
    t0 = time.time()
    bm25 = BM25()
    bm25.fit(docs, asin_to_meta)
    log(f"  [bm25] fit done in {time.time()-t0:.1f}s")
    return bm25


def fit_splade(docs: List[Dict], asin_to_meta: Dict) -> SPLADERetriever:
    log(f"  [splade] fitting on {len(docs)} docs...")
    t0 = time.time()
    obj = SPLADERetriever()
    obj.fit(docs, asin_to_meta)
    log(f"  [splade] fit done in {time.time()-t0:.1f}s")
    return obj


def process_category(category: str, n: int, seed: int, retrievers: List[str]) -> Dict:
    log(f"\n{'='*70}\nE2 — {category} (n={n})\n{'='*70}")
    docs, asin_to_meta = load_corpus(category)
    asins = list(asin_to_meta.keys())
    rng = random.Random(seed)
    floor, ceiling = make_queries(asins, asin_to_meta, n, rng)
    log(f"  floor_queries={len(floor)} ceiling_queries={len(ceiling)}")

    results: Dict[str, Dict] = {}
    for name in retrievers:
        try:
            if name == BM25_NAME:
                retr = fit_bm25(docs, asin_to_meta)
            elif name == SPLADE_NAME:
                retr = fit_splade(docs, asin_to_meta)
            elif name in DENSE_CLS_MAP or name in ("bge", "e5", "minilm", "star", "ance"):
                retr = fit_dense(name, docs, asin_to_meta)
            else:
                log(f"  unknown retriever: {name}")
                continue
            results[name] = run_floor_ceiling_search(retr, name, floor, ceiling, asin_to_meta)
            del retr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log(f"  [{name}] ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    return {
        "category": category,
        "n_samples": n,
        "n_docs": len(docs),
        "results": results,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = []
    rows.append("# E2 — Floor / Ceiling (full-quantity)\n")
    rows.append("**Method** — floor: random_other_doc.title → target_asin. ceiling: asin own title → asin.")
    rows.append("**Index** — full Amazon 2018 5-core meta per category (10K–172K docs).\n")
    rows.append("| Category | Retriever | Floor H@10 | Ceiling H@10 | Floor MRR | Ceiling MRR | n |")
    rows.append("|---|---|---|---|---|---|---|")
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
    p = OUT_DIR / "summary_full.md"
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
            s = process_category(cat, args.n, args.seed, args.retrievers)
            with open(OUT_DIR / f"{cat}_floor_ceiling_full.json", "w") as f:
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