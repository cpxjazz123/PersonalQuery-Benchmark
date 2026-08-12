#!/usr/bin/env python3
"""Build 3-retriever index (BGE + E5 + MiniLM) for E1.

Same as e1_build_bge_index.py but encodes with all 3 retrievers.
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
DEFAULT_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
CORPUS_ROOT = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
N_DISTRACTORS = 10000
SEED = 42
RETRIEVERS = ["bge", "e5", "minilm"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_text(meta: Dict) -> str:
    title = (meta.get("title") or "").strip()
    desc = (meta.get("description") or [])
    if isinstance(desc, list):
        desc = " ".join([str(d) for d in desc if d])
    brand = (meta.get("brand") or "").strip()
    cat = (meta.get("category") or [])
    if isinstance(cat, list):
        cat = " ".join([str(c) for c in cat if c])
    text = f"{title}. {desc}. Brand: {brand}. Categories: {cat}".strip()[:2000]
    return text


def load_meta_corpus(category: str) -> Dict[str, Dict]:
    fname_map = {
        "Baby_Products": "meta_Baby.jsonl.gz",
        "Grocery_and_Gourmet_Food": "raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz",
        "Pet_Supplies": "raw/meta_categories/meta_Pet_Supplies.jsonl.gz",
    }
    p = CORPUS_ROOT / fname_map[category]
    if not p.exists():
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
    return out


def get_gold_asins(category: str) -> set:
    p = REPO_ROOT / "result" / "personal_query" / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    return {r["asin"] for r in data if r.get("asin")}


def load_retriever(name: str, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "06_retrieval" / "utils"))
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    import retrievers as r
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    r._require_cuda_device = lambda: device
    cls_map = {"bge": r.BGERetriever, "e5": r.E5Retriever, "minilm": r.DenseRetriever}
    cls = cls_map[name]
    obj = cls()
    obj.device = device
    return obj, device


def encode_docs(obj, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return obj.encode_queries(texts, batch_size=batch_size)


def build_for_category(category: str) -> None:
    cache_path = DEFAULT_OUT / f"multi_index_{category}.pkl"
    if cache_path.exists():
        log(f"  {category}: cache exists, skipping")
        return
    log(f"=== {category} ===")
    corpus = load_meta_corpus(category)
    gold = get_gold_asins(category)
    non_gold = [a for a in corpus if a not in gold]
    rng = random.Random(SEED)
    rng.shuffle(non_gold)
    chosen = list(gold) + non_gold[:N_DISTRACTORS]
    chosen_in_corpus = [a for a in chosen if a in corpus]
    log(f"  in_corpus: {len(chosen_in_corpus)} asins")
    texts = [build_text(corpus[a]) for a in chosen_in_corpus]
    chosen = chosen_in_corpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_by_retriever: Dict[str, np.ndarray] = {}
    for name in RETRIEVERS:
        log(f"  encoding with {name} on {device}...")
        obj, _ = load_retriever(name, device)
        with torch.no_grad():
            embs = encode_docs(obj, texts, batch_size=64)
        embeddings_by_retriever[name] = embs
        log(f"    {name}: embs shape {embs.shape}")
        del obj
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    data = {
        "asins": chosen,
        "texts": texts,
        "embeddings": embeddings_by_retriever,
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