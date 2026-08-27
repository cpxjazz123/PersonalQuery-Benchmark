"""14B N=10 FULL POOL — Stage 5 retrieval + flip rate.

跟 7B full retrieval 一样逻辑,只是读 n_ablation_14b_n10_full/pool_N10.json
(2000 ASINs × K=20 strict pool)。
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

from syntax_subspace_utils import (
    ASIN_TO_DOC_CACHE, MINILM_CORPUS_EMBEDS_CACHE, log,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full")
SEED = 2024
K_SELECT = 10
N = 10


def main():
    rng = random.Random(SEED)
    log("=== 14B N=10 FULL POOL Stage 5 ===")
    corpus = json.load(open(ASIN_TO_DOC_CACHE))
    asin_list = sorted(corpus.keys())
    asin_to_pos = {a: i for i, a in enumerate(asin_list)}

    doc_embeds = np.load(MINILM_CORPUS_EMBEDS_CACHE)
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    doc_gpu = torch.from_numpy(doc_embeds).to(device)

    import bm25s
    log("  building BM25 index...")
    corpus_texts = [corpus[a] for a in asin_list]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens, show_progress=False)

    pool = json.load(open(OUT_DIR / f"pool_N{N}.json"))
    pools = pool["pools"]

    all_queries = []
    for asin, qs in pools.items():
        strict_qs = [q for q in qs if q["strict"]]
        if len(strict_qs) < 10:
            continue
        chosen = rng.sample(strict_qs, min(K_SELECT, len(strict_qs)))
        for q in chosen:
            all_queries.append((asin, q["k"], q["query"]))
    log(f"  ASINs: {len(set(a for a,_,_ in all_queries))}, queries: {len(all_queries)}")

    q_texts = [q[2] for q in all_queries]
    q_tokens = bm25s.tokenize(q_texts, stopwords="en", show_progress=False)
    bm_results, _ = bm25_retriever.retrieve(q_tokens, k=200, show_progress=False)

    q_embeds = model.encode(q_texts, convert_to_tensor=True, show_progress_bar=False,
                            batch_size=256)

    log(f"  MiniLM: full sims...")
    bs = 256
    mlm_ranks = []
    for i in range(0, len(q_texts), bs):
        chunk = q_embeds[i:i+bs] @ doc_gpu.T
        chunk_np = chunk.cpu().numpy()
        for r, row in enumerate(chunk_np):
            tp = asin_to_pos.get(all_queries[i+r][0])
            if tp is None:
                mlm_ranks.append(None); continue
            target_score = row[tp]
            rank = int((row > target_score).sum()) + 1
            mlm_ranks.append(rank)

    per_q_bm_rank = []
    for idx, (asin, k, text) in enumerate(all_queries):
        tp = asin_to_pos.get(asin)
        if tp is None:
            per_q_bm_rank.append(None); continue
        pos = np.where(bm_results[idx] == tp)[0]
        per_q_bm_rank.append(int(pos[0] + 1) if len(pos) else None)

    asin_groups = collections.defaultdict(list)
    for idx, (asin, k, text) in enumerate(all_queries):
        asin_groups[asin].append({
            "bm_rank": per_q_bm_rank[idx],
            "mlm_rank": mlm_ranks[idx],
            "n_tok": len(text.split()),
        })

    per_asin_flip = []
    for asin, qs in asin_groups.items():
        n_pairs = len(qs) * (len(qs) - 1) // 2
        if n_pairs == 0: continue
        bm_dis = 0; ml_dis = 0
        for i in range(len(qs)):
            for j in range(i+1, len(qs)):
                bm_hi = 1 if (qs[i]["bm_rank"] is not None and qs[i]["bm_rank"] <= 10) else 0
                bm_hj = 1 if (qs[j]["bm_rank"] is not None and qs[j]["bm_rank"] <= 10) else 0
                if bm_hi != bm_hj: bm_dis += 1
                ml_hi = 1 if (qs[i]["mlm_rank"] is not None and qs[i]["mlm_rank"] <= 10) else 0
                ml_hj = 1 if (qs[j]["mlm_rank"] is not None and qs[j]["mlm_rank"] <= 10) else 0
                if ml_hi != ml_hj: ml_dis += 1
        per_asin_flip.append({
            "asin": asin, "n_q": len(qs), "n_pairs": n_pairs,
            "bm_flip": bm_dis / n_pairs, "mlm_flip": ml_dis / n_pairs,
        })

    n_asins_eff = len(per_asin_flip)
    bm_flip_mean = float(np.mean([p["bm_flip"] for p in per_asin_flip]))
    mlm_flip_mean = float(np.mean([p["mlm_flip"] for p in per_asin_flip]))
    bm_hit10 = float(np.mean([1 if (r is not None and r <= 10) else 0 for r in per_q_bm_rank]))
    mlm_hit10 = float(np.mean([1 if (r is not None and r <= 10) else 0 for r in mlm_ranks]))

    log(f"  N={N}: n_asins={n_asins_eff}, n_queries={len(all_queries)}")
    log(f"  BM25:   hit@10={bm_hit10:.4f}, flip_mean={bm_flip_mean:.4f}")
    log(f"  MiniLM: hit@10={mlm_hit10:.4f}, flip_mean={mlm_flip_mean:.4f}")

    out = {
        "N": N, "model": "Qwen2.5-14B-Instruct",
        "config": {"K_select": K_SELECT, "SEED": SEED,
                   "n_asins_target": len(pools), "K_variants": 20,
                   "source": "product_attributes.json full pool"},
        "n_asins": n_asins_eff, "n_queries": len(all_queries),
        "bm25_hit10": bm_hit10, "minilm_hit10": mlm_hit10,
        "bm25_flip_mean": bm_flip_mean, "minilm_flip_mean": mlm_flip_mean,
    }
    json.dump(out, open(OUT_DIR / f"flip_N{N}.json", "w"), indent=2)
    log(f"  wrote → {OUT_DIR}/flip_N{N}.json")


if __name__ == "__main__":
    main()