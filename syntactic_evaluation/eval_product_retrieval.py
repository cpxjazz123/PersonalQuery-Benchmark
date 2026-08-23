#!/usr/bin/env python3
"""商品检索评测：Query = best_query，Target = 全量商品 ASIN。

评测框架:
  Query: best_cands_10k.json 的 best_query
  Target: 用户实际购买的 ASIN (parent_asin)
  候选池: meta_Baby_Products_2023.jsonl.gz 中所有 parent_asin 的 title+description 拼接文档
  指标: Hit@10 / MRR / Mean Rank

检索: cosine similarity in all-MiniLM-L6-v2 space (384d, ~1000 docs/s CPU)
"""
from __future__ import annotations
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

BEST_CANDS = SCRATCH / "best_cands_10k.json"
META_FILE = REPO / "data/meta_Baby_Products_2023.jsonl.gz"
RECORDS_FILE = REPO / "result/query_records_with_query_inject_strict_10k.json"

SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_BATCH = 128


def build_product_docs():
    """Build asin → doc text mapping from metadata."""
    print("[build] Loading product metadata...")
    asin_to_doc = {}
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "")
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            d = desc.strip() if isinstance(desc, str) else ""
            if d:
                parts.append(d)
            feat = r.get("features", [])
            if isinstance(feat, list) and feat:
                parts.append(" | ".join(f.strip() for f in feat if f.strip()))
            asin_to_doc[asin] = " ".join(parts) if parts else r.get("title", asin)
    print(f"  {len(asin_to_doc)} products with parent_asin")
    return asin_to_doc


def load_queries():
    """Load eval records."""
    print("[load] Loading queries...")
    best_cands = json.load(open(BEST_CANDS, "r", encoding="utf-8"))

    eval_records = []
    for bc in best_cands:
        eval_records.append({
            "user_id": bc["user_id"],
            "asin": bc["asin"],
            "query": bc["best_query"],
        })

    print(f"  {len(eval_records)} eval records")
    return eval_records


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return v / norm


def main() -> int:
    t0 = time.time()

    # 1) Load data
    asin_to_doc = build_product_docs()
    eval_records = load_queries()
    all_asins = list(asin_to_doc.keys())
    all_docs = [asin_to_doc[a] for a in all_asins]
    n_products = len(all_asins)
    print(f"  {n_products} total ASINs in candidate pool")

    # 2) Load sentence-transformer
    print(f"\n[main] Loading {SBERT_MODEL}...")
    model = SentenceTransformer(SBERT_MODEL)
    print("[main] Model loaded")

    # 3) Encode all product docs
    print(f"\n[encode] Encoding {n_products} product docs...")
    product_emb = model.encode(
        all_docs,
        batch_size=SBERT_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # [N, 384]
    print(f"  product_emb shape: {product_emb.shape}")
    asin_to_idx = {a: i for i, a in enumerate(all_asins)}

    # 4) Encode eval queries
    print(f"\n[encode] Encoding {len(eval_records)} eval queries...")
    unique_queries = list(set(r["query"] for r in eval_records))
    print(f"  {len(unique_queries)} unique queries")

    query_emb_map = {}
    for i in range(0, len(unique_queries), SBERT_BATCH):
        chunk = unique_queries[i:i + SBERT_BATCH]
        vecs = model.encode(chunk, batch_size=SBERT_BATCH, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
        for q, vec in zip(chunk, vecs):
            query_emb_map[q] = vec.astype(np.float32)
        done = min(i + SBERT_BATCH, len(unique_queries))
        if done % 2000 == 0 or done == len(unique_queries):
            print(f"    [{done}/{len(unique_queries)}]")

    # 5) Retrieval evaluation (batch GEMM)
    print(f"\n[eval] Running retrieval...")
    query_to_targets = defaultdict(list)
    for rec in eval_records:
        q = rec["query"]
        target_asin = rec["asin"]
        if q not in query_emb_map or target_asin not in asin_to_idx:
            continue
        query_to_targets[q].append(asin_to_idx[target_asin])

    batch_qs = list(query_to_targets.keys())
    all_ranks = []
    hit10 = 0
    mrr_sum = 0.0

    for batch_start in range(0, len(batch_qs), 200):
        batch_qs_chunk = batch_qs[batch_start:batch_start + 200]
        q_matrix = np.stack([query_emb_map[q] for q in batch_qs_chunk], axis=0)  # [B, 384]
        scores = np.dot(q_matrix, product_emb.T)  # [B, N]

        for q in batch_qs_chunk:
            q_idx_in_batch = batch_qs_chunk.index(q)
            q_scores = scores[q_idx_in_batch]
            sorted_indices = np.argsort(-q_scores)
            for tgt_idx in query_to_targets[q]:
                rank = int(np.where(sorted_indices == tgt_idx)[0][0]) + 1
                all_ranks.append(rank)
                if rank <= 10:
                    hit10 += 1
                mrr_sum += 1.0 / rank

        done = batch_start + 200
        if done % 2000 == 0 or done >= len(batch_qs):
            print(f"    [{min(done, len(batch_qs))}/{len(batch_qs)}]")

    # 6) Results
    n = len(all_ranks)
    hit10_rate = hit10 / n if n else 0
    mrr = mrr_sum / n if n else 0
    mean_rank = np.mean(all_ranks) if all_ranks else float("inf")
    median_rank = np.median(all_ranks) if all_ranks else float("inf")
    p90_rank = np.percentile(all_ranks, 90) if all_ranks else float("inf")

    print("\n" + "=" * 50)
    print(f"  Product Retrieval Results (n={n})")
    print("=" * 50)
    print(f"  Hit@10              : {hit10_rate:.4f} ({hit10}/{n})")
    print(f"  MRR                 : {mrr:.4f}")
    print(f"  Mean Rank           : {mean_rank:.1f}")
    print(f"  Median Rank         : {median_rank:.1f}")
    print(f"  P90 Rank            : {p90_rank:.1f}")
    print("=" * 50)

    out = SCRATCH / "product_retrieval_results.json"
    with open(out, "w") as f:
        json.dump({
            "n": n,
            "hit10": hit10_rate,
            "mrr": mrr,
            "mean_rank": mean_rank,
            "median_rank": median_rank,
            "p90_rank": p90_rank,
            "ranks": all_ranks,
        }, f, indent=2)
    print(f"\n[saved] → {out}")
    print(f"[total time] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
