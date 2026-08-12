#!/usr/bin/env python3
"""Build BGE index for E1: gold asins + distractors from Amazon meta corpus.

For each domain:
1. Load meta_<cat>.jsonl.gz → {asin: meta}
2. Gold asins = unique asins from 04_query
3. Distractors = sample 10000 random non-gold asins
4. Encode all with BGE-large-en-v1.5
5. Cache to /home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric/bge_index_<cat>.pkl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
DEFAULT_RESULT = REPO_ROOT / "result" / "personal_query"
DEFAULT_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
CORPUS_ROOT = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
N_DISTRACTORS = 10000
SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_text(meta: Dict) -> str:
    """拼接 title + description + brand + categories."""
    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or [])
    if isinstance(desc, list):
        desc = " ".join([str(d) for d in desc if d])
    brand = (meta.get("brand") or "").strip()
    cat = (meta.get("category") or [])
    if isinstance(cat, list):
        cat = " ".join([str(c) for c in cat if c])
    return f"{title}. {desc}. Brand: {brand}. Categories: {cat}".strip()[:2000]


def load_meta_corpus(category: str) -> Dict[str, Dict]:
    fname_map = {
        "Baby_Products": "meta_Baby.jsonl.gz",
        "Grocery_and_Gourmet_Food": "raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz",
        "Pet_Supplies": "raw/meta_categories/meta_Pet_Supplies.jsonl.gz",
    }
    p = CORPUS_ROOT / fname_map[category]
    if not p.exists():
        log(f"  corpus not found: {p}")
        return {}
    out = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin")
            if asin:
                out[asin] = rec
    log(f"  loaded {len(out)} meta records from {p}")
    return out


def get_gold_asins(category: str) -> set:
    p = DEFAULT_RESULT / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    out = {r["asin"] for r in data if r.get("asin")}
    log(f"  {category}: {len(out)} gold asins from 04_query")
    return out


def load_bge(device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "06_retrieval" / "utils"))
    os.environ["HF_HOME"] = "/home/wlia0047/.cache/huggingface/hub"
    import retrievers as r
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    r._require_cuda_device = lambda: device
    from retrievers import BGERetriever
    obj = BGERetriever()
    obj.device = device
    return obj, device


def encode_texts(obj, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return obj.encode_queries(texts, batch_size=batch_size)


def build_for_category(category: str) -> None:
    log(f"=== {category} ===")
    cache_path = DEFAULT_OUT / f"bge_index_{category}.pkl"
    if cache_path.exists():
        log(f"  cache exists at {cache_path}, skipping")
        return
    corpus = load_meta_corpus(category)
    gold = get_gold_asins(category)
    non_gold = [a for a in corpus if a not in gold]
    rng = random.Random(SEED)
    rng.shuffle(non_gold)
    distractors = non_gold[:N_DISTRACTORS]
    chosen = list(gold) + distractors
    log(f"  chosen: {len(chosen)} asins ({len(gold)} gold + {len(distractors)} distractors)")
    # filter to asins that exist in corpus
    chosen_in_corpus = [a for a in chosen if a in corpus]
    skipped = len(chosen) - len(chosen_in_corpus)
    log(f"  in_corpus: {len(chosen_in_corpus)} (skipped {skipped})")
    texts = [build_text(corpus[a]) for a in chosen_in_corpus]
    chosen = chosen_in_corpus
    obj, device = load_bge(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"  encoding on {device}...")
    with torch.no_grad():
        embs = encode_texts(obj, texts, batch_size=64)
    log(f"  embs shape: {embs.shape}")
    data = {
        "asins": chosen,
        "texts": texts,
        "embeddings": embs,
        "device": str(device),
        "category": category,
    }
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    log(f"  wrote {cache_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()
    for cat in args.categories:
        build_for_category(cat)
    log("=== done ===")


if __name__ == "__main__":
    main()