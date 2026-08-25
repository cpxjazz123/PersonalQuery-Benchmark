"""Stage 8.5: Retrieval (selected / random / farthest) — bm25s + GPU MiniLM.

Uses bm25s (C-accelerated BM25) + sentence-transformers on GPU.
ASIN-level metadata corpus (217K ASINs, not 6M reviews).

Output: stage8_5_retrieval_per_query.json + stage8_5_retrieval_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_retrieval.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SELECTION_IN = SCRATCH / "stage8_5_selection.json"
META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"
PER_QUERY_OUT = SCRATCH / "stage8_5_retrieval_per_query.json"
SUMMARY_OUT = SCRATCH / "stage8_5_retrieval_summary.json"

# Force GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_meta_corpus():
    """Load ASIN-level metadata (title + description + features). 217K ASINs."""
    asin_to_doc = {}
    log(f"  loading metadata from {META_FILE}")
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            elif isinstance(desc, str):
                pass
            else:
                desc = ""
            desc = desc.strip()
            if desc:
                parts.append(desc)
            feats = r.get("features", [])
            if isinstance(feats, list):
                feats = " ".join(feats)
            if feats:
                parts.append(feats[:500])
            doc = " | ".join(parts).strip()
            if doc:
                asin_to_doc[asin] = doc[:1000]
    log(f"  loaded {len(asin_to_doc)} ASIN docs")
    return asin_to_doc


def bm25s_search(corpus_texts: List[str], queries: List[str]):
    """Use bm25s for fast BM25 scoring.

    Returns: (docs_indices, scores) per query, where docs_indices is the
    original corpus index ordering.
    """
    import bm25s
    log(f"  tokenizing {len(corpus_texts)} corpus texts...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    log(f"  corpus tokenized in {time.time() - t0:.1f}s")

    log(f"  building bm25s index...")
    t0 = time.time()
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  index built in {time.time() - t0:.1f}s")

    log(f"  tokenizing {len(queries)} queries...")
    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    log(f"  queries tokenized in {time.time() - t0:.1f}s")

    log(f"  retrieving (k=all to get full scores)...")
    t0 = time.time()
    # k=corpus size to get all scores (we need target ASIN rank)
    results = retriever.retrieve(query_tokens, k=len(corpus_texts), show_progress=False)
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return results  # bm25s.Results with .documents (indices) and .scores


def minilm_encode_gpu(corpus_texts: List[str], queries_text: List[str]):
    """Encode corpus + queries via MiniLM on GPU."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  MiniLM device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    log(f"  encoding {len(corpus_texts)} corpus docs via MiniLM (GPU, batch=512)...")
    t0 = time.time()
    doc_embeds = model.encode(
        corpus_texts,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  docs encoded in {time.time() - t0:.1f}s, shape: {doc_embeds.shape}")

    log(f"  encoding {len(queries_text)} queries via MiniLM (GPU)...")
    t0 = time.time()
    q_embeds = model.encode(
        queries_text,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  queries encoded in {time.time() - t0:.1f}s")
    # Move to GPU for fast matmul
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    return doc_embeds_gpu, q_embeds_gpu


def main():
    log("=== Stage 8.5: Retrieval (selected / random / farthest) — bm25s + GPU MiniLM ===")

    # === 1. Load selection ===
    log("\n=== 1. Loading selection ===")
    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    log(f"  {len(entries)} entries")

    query_records = []
    for i, e in enumerate(entries):
        for variant in ("selected", "random", "farthest"):
            q = e[variant]
            if q is None:
                continue
            query_records.append({
                "entry_idx": i,
                "asin": e["asin"],
                "user_id": e["user_id"],
                "variant": variant,
                "selection_method": e["selection_method"],
                "user_source": e["user_source"],
                "query": q["query"],
                "n_tok": q["n_tok"],
                "attrs_covered": q["attrs_covered"],
                "strict": q["strict"],
            })
    log(f"  total queries to retrieve: {len(query_records)}")

    # === 2. Build meta corpus ===
    log("\n=== 2. Building ASIN metadata corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    n_missing = int((target_indices < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing} queries have missing target ASINs in corpus")

    # === 3. BM25 via bm25s ===
    log("\n=== 3. BM25 retrieval (bm25s) ===")
    corpus_texts = [asin_to_doc[a] for a in asins]
    results = bm25s_search(corpus_texts, queries)

    # Compute rank: for each query, find target ASIN's rank in score-sorted list
    # bm25s Results.documents[i] = array of corpus indices sorted by score (desc)
    log(f"  computing target ranks...")
    t0 = time.time()
    bm25_results = []
    bm25_doc_ids = results.documents  # shape (n_queries, n_corpus)
    bm25_scores = results.scores
    for i in range(len(queries)):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        # Position of tgt_idx in sorted docs
        sorted_docs = bm25_doc_ids[i]
        positions = np.where(sorted_docs == tgt_idx)[0]
        if len(positions) == 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
        else:
            rank = int(positions[0]) + 1
            bm25_results.append({
                "rank": rank,
                "RR": 1.0 / rank,
                "hit10": 1 if rank <= 10 else 0,
            })
    log(f"  ranks computed in {time.time() - t0:.1f}s")

    # === 4. MiniLM GPU ===
    log("\n=== 4. MiniLM retrieval (GPU) ===")
    import torch
    doc_embeds_gpu, q_embeds_gpu = minilm_encode_gpu(corpus_texts, queries)

    log(f"  computing cosine similarities on GPU...")
    t0 = time.time()
    sims = q_embeds_gpu @ doc_embeds_gpu.T   # (n_queries, n_corpus)
    log(f"  matmul done in {time.time() - t0:.1f}s")
    log(f"  sorting similarities...")
    t0 = time.time()
    # For each query: rank = number of docs with strictly higher sim
    # Top-k retrieval: use torch.topk for rank of target
    minilm_results = []
    for i in range(sims.shape[0]):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            minilm_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        # rank = (sim > sim[tgt_idx]).sum() + 1
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_results.append({
            "rank": rank,
            "RR": 1.0 / rank,
            "hit10": 1 if rank <= 10 else 0,
        })
    log(f"  ranks computed in {time.time() - t0:.1f}s")

    # Free GPU memory
    del doc_embeds_gpu, q_embeds_gpu, sims
    torch.cuda.empty_cache()

    # === 5. Build per-query records ===
    log("\n=== 5. Saving ===")
    for r, bm25_r, minilm_r in zip(query_records, bm25_results, minilm_results):
        r["bm25_rank"] = bm25_r["rank"]
        r["bm25_RR"] = bm25_r["RR"]
        r["bm25_hit10"] = bm25_r["hit10"]
        r["minilm_rank"] = minilm_r["rank"]
        r["minilm_RR"] = minilm_r["RR"]
        r["minilm_hit10"] = minilm_r["hit10"]

    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5 retrieval per query (bm25s + GPU MiniLM) on ASIN meta corpus",
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # === 6. Summary by variant ===
    by_variant = collections.defaultdict(lambda: {
        "bm25_RR": [], "minilm_RR": [], "bm25_hit10": [], "minilm_hit10": [],
        "bm25_rank": [], "minilm_rank": [],
    })
    for r in query_records:
        v = r["variant"]
        by_variant[v]["bm25_RR"].append(r["bm25_RR"])
        by_variant[v]["minilm_RR"].append(r["minilm_RR"])
        by_variant[v]["bm25_hit10"].append(r["bm25_hit10"])
        by_variant[v]["minilm_hit10"].append(r["minilm_hit10"])
        if r["bm25_rank"]:
            by_variant[v]["bm25_rank"].append(r["bm25_rank"])
        if r["minilm_rank"]:
            by_variant[v]["minilm_rank"].append(r["minilm_rank"])

    summary = {}
    for v in ("selected", "random", "farthest"):
        d = by_variant[v]
        summary[v] = {
            "n": len(d["bm25_RR"]),
            "bm25_MRR": float(np.mean(d["bm25_RR"])),
            "bm25_Hit@10": float(np.mean(d["bm25_hit10"])),
            "bm25_mean_rank": float(np.mean(d["bm25_rank"])) if d["bm25_rank"] else None,
            "minilm_MRR": float(np.mean(d["minilm_RR"])),
            "minilm_Hit@10": float(np.mean(d["minilm_hit10"])),
            "minilm_mean_rank": float(np.mean(d["minilm_rank"])) if d["minilm_rank"] else None,
        }

    log(f"\n=== Variant summary ===")
    for v, s in summary.items():
        bm25_rank_str = f"{s['bm25_mean_rank']:.1f}" if s["bm25_mean_rank"] else "n/a"
        minilm_rank_str = f"{s['minilm_mean_rank']:.1f}" if s["minilm_mean_rank"] else "n/a"
        log(f"  {v} (n={s['n']}):")
        log(f"    BM25:   MRR={s['bm25_MRR']:.4f}, Hit@10={s['bm25_Hit@10']*100:.1f}%, mean_rank={bm25_rank_str}")
        log(f"    MiniLM: MRR={s['minilm_MRR']:.4f}, Hit@10={s['minilm_Hit@10']*100:.1f}%, mean_rank={minilm_rank_str}")

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()