"""Stage 5 cross-encoder rerank on v6m strict alignment selection.

两阶段 retrieval:
  Stage A: BM25 top-K retrieve (K=100 candidates per query)
  Stage B: cross-encoder rerank on top-K → final ranking

Cross-encoder models:
  1. BGE-reranker-base (BAAI/bge-reranker-base, 568M)
  2. cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M, fast)

输入:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json (v6m canonical, 3781 strict)
输出:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_rerank_per_query.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_rerank_summary.json

对比 base (BM25 full) vs rerank (BM25 top-100 + CE):
  - Hit@1 / Hit@5 / Hit@10
  - RR mean / median
  - Per-ASIN Hit@K flip + RR Std (sim09)
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASIN_TO_DOC_CACHE, META_FILE, log,
)

SEL_IN = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json")
PER_QUERY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_rerank_per_query.json")
SUMMARY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_rerank_summary.json")

# Stage A: BM25 top-K candidates for rerank
BM25_K_RERANK = 100

# Cross-encoder configs
CROSS_ENCODERS = [
    {"name": "bge_reranker_base", "hf_id": "BAAI/bge-reranker-base", "max_length": 256},
    {"name": "msmarco_minilm", "hf_id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "max_length": 256},
]


def build_meta_corpus():
    if ASIN_TO_DOC_CACHE.exists():
        t0 = time.time()
        asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
        log(f"  ✓ asin_to_doc cache hit ({len(asin_to_doc)} ASINs, {time.time() - t0:.2f}s)")
        return asin_to_doc
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
            elif not isinstance(desc, str):
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
                asin_to_doc[asin] = doc
    return asin_to_doc


def bm25_topk(queries: list[str], corpus_texts: list[str], k: int) -> list[list[tuple[int, float]]]:
    """BM25 retrieve top-K (idx, score) per query."""
    import bm25s
    log(f"\n=== BM25 top-{k} retrieve ===")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    log(f"  corpus tokenized in {time.time() - t0:.1f}s")

    log("  building bm25s index...")
    t0 = time.time()
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  index built in {time.time() - t0:.1f}s")

    log(f"  tokenizing {len(queries)} queries...")
    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    log(f"  queries tokenized in {time.time() - t0:.1f}s")

    log(f"  retrieving top-{k}...")
    t0 = time.time()
    BM25_BATCH = 1000
    out = [None] * len(queries)

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(queries), BM25_BATCH):
        e = min(s + BM25_BATCH, len(queries))
        sub_tokens = _slice_tok(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=k, show_progress=False)
        for i in range(e - s):
            gi = s + i
            docs = sub_res.documents[i]
            scores = sub_res.scores[i]
            out[gi] = [(int(docs[j]), float(scores[j])) for j in range(len(docs))]
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return out


def rerank_with_crossencoder(
    ce_name: str,
    hf_id: str,
    max_length: int,
    queries: list[str],
    corpus_texts: list[str],
    bm25_topk_per_q: list[list[tuple[int, float]]],
    target_indices: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Run cross-encoder rerank on BM25 top-K candidates per query."""
    from sentence_transformers import CrossEncoder

    log(f"\n=== Cross-encoder rerank: {ce_name} ({hf_id}) ===")
    device = "cuda"
    model = CrossEncoder(hf_id, max_length=max_length, device=device)

    # Build pairs (query, doc) for all top-K per query
    log(f"  building (query, doc) pairs for {len(queries)} queries × top-{BM25_K_RERANK}...")
    all_pairs = []
    pair_offsets = [0] * (len(queries) + 1)  # pair_offsets[i]..pair_offsets[i+1] is range for query i
    for qi, q in enumerate(queries):
        for cand_idx, _ in bm25_topk_per_q[qi]:
            all_pairs.append((q, corpus_texts[cand_idx]))
        pair_offsets[qi + 1] = pair_offsets[qi] + len(bm25_topk_per_q[qi])

    log(f"  total pairs: {len(all_pairs)}; predicting in batches...")
    t0 = time.time()
    CE_BATCH = 128  # cross-encoder is heavy
    all_scores = []
    for s in range(0, len(all_pairs), CE_BATCH):
        e = min(s + CE_BATCH, len(all_pairs))
        scores = model.predict(all_pairs[s:e], show_progress_bar=False, convert_to_numpy=True)
        all_scores.extend(scores.tolist())
    log(f"  CE predicted in {time.time() - t0:.1f}s")

    # Re-rank per query by CE score, compute final rank
    log(f"  re-ranking per query...")
    results = []
    for qi, q in enumerate(queries):
        tgt_idx = target_indices[qi]
        if tgt_idx < 0:
            results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        cands = bm25_topk_per_q[qi]
        start, end = pair_offsets[qi], pair_offsets[qi + 1]
        ce_scores = all_scores[start:end]
        # Sort by ce_score desc
        order = sorted(range(len(cands)), key=lambda j: -ce_scores[j])
        sorted_docs = [cands[j][0] for j in order]
        # Find target rank
        positions = np.where(np.array(sorted_docs) == tgt_idx)[0]
        if len(positions) == 0:
            # Target not in top-K, fall back to BM25 original position
            bm25_positions = np.where(np.array([c[0] for c in cands]) == tgt_idx)[0]
            if len(bm25_positions) == 0:
                rank = BM25_K_RERANK + 1  # not in top-100 at all
            else:
                rank = int(bm25_positions[0]) + 1
        else:
            rank = int(positions[0]) + 1
        results.append({
            "rank": rank,
            "RR": 1.0 / rank,
            "hit10": 1 if rank <= 10 else 0,
            "hit5": 1 if rank <= 5 else 0,
            "hit1": 1 if rank <= 1 else 0,
        })

    # Build embedding-like structure for sim09 — but cross-encoder doesn't expose embeddings,
    # so use MiniLM embeddings for sim09 slicing (consistent with multi-retrieval script)
    log(f"  encoding {len(queries)} queries for sim09 (using MiniLM)...")
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    q_embeds = minilm.encode(
        queries, batch_size=512, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    del model, minilm
    torch.cuda.empty_cache()
    return results, q_embeds


def compute_volatility_sim09(
    per_query: list[dict], rank_key: str, rr_key: str,
    q_embeds: np.ndarray, sim_threshold: float = 0.9,
) -> dict:
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in per_query:
        by_asin[r["asin"]].append(r)
    qid_to_embed = {id(r): q_embeds[i] for i, r in enumerate(per_query)}

    f1_list, f5_list, f10_list = [], [], []
    rr_std_list = []
    n_asins_used = 0
    for asin, qs in by_asin.items():
        if len(qs) < 2:
            continue
        ranks = [q.get(rank_key) for q in qs]
        if any(r is None for r in ranks):
            continue
        hit1 = [1 if r == 1 else 0 for r in ranks]
        hit5 = [1 if r <= 5 else 0 for r in ranks]
        hit10 = [1 if r <= 10 else 0 for r in ranks]
        rrs = [float(q.get(rr_key) or 0.0) for q in qs]
        rr_std = float(np.std(rrs, ddof=0)) if len(rrs) >= 2 else None
        embeds = np.stack([qid_to_embed[id(q)] for q in qs], axis=0)
        sim_mat = embeds @ embeds.T

        def flip_rate(hit_labels):
            n_q = len(hit_labels)
            used, disagree = 0, 0
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    if sim_mat[i, j] >= sim_threshold:
                        used += 1
                        if hit_labels[i] != hit_labels[j]:
                            disagree += 1
            if used == 0:
                return None
            return disagree / used

        for hit_labels, lst in [(hit1, f1_list), (hit5, f5_list), (hit10, f10_list)]:
            v = flip_rate(hit_labels)
            if v is not None:
                lst.append(v)
        if rr_std is not None:
            rr_std_list.append(rr_std)
        n_asins_used += 1

    return {
        "n_asins": n_asins_used,
        "Hit@1_FlipRate_mean": float(np.mean(f1_list)) if f1_list else None,
        "Hit@5_FlipRate_mean": float(np.mean(f5_list)) if f5_list else None,
        "Hit@10_FlipRate_mean": float(np.mean(f10_list)) if f10_list else None,
        "RR_Std_mean": float(np.mean(rr_std_list)) if rr_std_list else None,
        "RR_Std_median": float(np.median(rr_std_list)) if rr_std_list else None,
        "RR_Std_std": float(np.std(rr_std_list, ddof=0)) if rr_std_list else None,
    }


def main():
    log_start = time.time()
    log("=== Stage 5 cross-encoder rerank on v6m strict alignment ===")

    # ---- 1. Load v6m selection ----
    log("\n=== 1. Loading v6m selection ===")
    selection = json.load(open(SEL_IN))
    entries = selection["entries"]
    query_records = []
    for i, e in enumerate(entries):
        q = e["selected"]
        if q is None:
            continue
        query_records.append({
            "entry_idx": i,
            "asin": e["asin"],
            "user_id": e["user_id"],
            "selection_method": e["selection_method"],
            "selected_distance": e["selected_distance"],
            "selected_margin": e["selected_margin"],
            "user_source": e["user_source"],
            "query": q["query"],
            "n_tok": q["n_tok"],
            "attrs_covered": q["attrs_covered"],
            "strict": q["strict"],
        })
    log(f"  strict queries: {len(query_records)}")

    # ---- 2. Build corpus ----
    log("\n=== 2. Building ASIN corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])

    # ---- 3. BM25 top-K ----
    log("\n=== 3. BM25 top-K retrieve ===")
    corpus_texts = [asin_to_doc[a] for a in asins]
    bm25_topk_per_q = bm25_topk(queries, corpus_texts, k=BM25_K_RERANK)

    # Coverage: how often is target in top-K?
    n_in_topk = sum(
        1 for qi in range(len(queries))
        if target_indices[qi] >= 0 and target_indices[qi] in {c[0] for c in bm25_topk_per_q[qi]}
    )
    log(f"  target in top-{BM25_K_RERANK}: {n_in_topk}/{len(queries)} = {100 * n_in_topk / len(queries):.1f}%")

    # ---- 4. Cross-encoder rerank ----
    log("\n=== 4. Cross-encoder rerank ===")
    ce_results = {}
    ce_q_embeds = {}
    for ce in CROSS_ENCODERS:
        results, q_embeds = rerank_with_crossencoder(
            ce["name"], ce["hf_id"], ce["max_length"],
            queries, corpus_texts, bm25_topk_per_q, target_indices,
        )
        ce_results[ce["name"]] = results
        ce_q_embeds[ce["name"]] = q_embeds

    # ---- 5. Also compute BM25 top-100 same metric (for comparison) ----
    # Without rerank, BM25 top-100 results are exactly bm25_topk_per_q in BM25 score order
    log("\n=== 5. BM25 (top-K, no rerank) baseline ===")
    bm25_topk_results = []
    for qi in range(len(queries)):
        tgt_idx = target_indices[qi]
        if tgt_idx < 0:
            bm25_topk_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sorted_docs = [c[0] for c in bm25_topk_per_q[qi]]  # already sorted by bm25 desc
        positions = np.where(np.array(sorted_docs) == tgt_idx)[0]
        if len(positions) == 0:
            rank = BM25_K_RERANK + 1
        else:
            rank = int(positions[0]) + 1
        bm25_topk_results.append({
            "rank": rank,
            "RR": 1.0 / rank,
            "hit10": 1 if rank <= 10 else 0,
            "hit5": 1 if rank <= 5 else 0,
            "hit1": 1 if rank <= 1 else 0,
        })

    # ---- 6. Save per-query ----
    log("\n=== 6. Saving per-query results ===")
    for r in query_records:
        # BM25 (top-K only)
        idx = query_records.index(r)
        bres = bm25_topk_results[idx]
        r["bm25_topk_rank"] = bres["rank"]
        r["bm25_topk_RR"] = bres["RR"]
        r["bm25_topk_hit1"] = bres["hit1"]
        r["bm25_topk_hit5"] = bres["hit5"]
        r["bm25_topk_hit10"] = bres["hit10"]
        for ce in CROSS_ENCODERS:
            n = ce["name"]
            cres = ce_results[n][idx]
            r[f"{n}_rank"] = cres["rank"]
            r[f"{n}_RR"] = cres["RR"]
            r[f"{n}_hit1"] = cres["hit1"]
            r[f"{n}_hit5"] = cres["hit5"]
            r[f"{n}_hit10"] = cres["hit10"]

    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 cross-encoder rerank on v6m strict alignment",
                "stage_a": f"BM25 top-{BM25_K_RERANK}",
                "stage_b": [ce["hf_id"] for ce in CROSS_ENCODERS],
                "selection_file": str(SEL_IN),
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # ---- 7. Headline + volatility ----
    log("\n=== 7. Headline (per-query mean) ===")
    headline = {}
    # BM25 top-K (baseline)
    headline["bm25_topk"] = {
        "RR_mean": float(np.mean([r["bm25_topk_RR"] for r in query_records])),
        "RR_median": float(np.median([r["bm25_topk_RR"] for r in query_records])),
        "hit1": float(np.mean([r["bm25_topk_hit1"] for r in query_records])),
        "hit5": float(np.mean([r["bm25_topk_hit5"] for r in query_records])),
        "hit10": float(np.mean([r["bm25_topk_hit10"] for r in query_records])),
    }
    log(f"  BM25(top-{BM25_K_RERANK})  RR={headline['bm25_topk']['RR_mean']:.4f}  hit@1={headline['bm25_topk']['hit1']:.4f}  hit@5={headline['bm25_topk']['hit5']:.4f}  hit@10={headline['bm25_topk']['hit10']:.4f}")

    for ce in CROSS_ENCODERS:
        n = ce["name"]
        headline[n] = {
            "RR_mean": float(np.mean([r[f"{n}_RR"] for r in query_records])),
            "RR_median": float(np.median([r[f"{n}_RR"] for r in query_records])),
            "hit1": float(np.mean([r[f"{n}_hit1"] for r in query_records])),
            "hit5": float(np.mean([r[f"{n}_hit5"] for r in query_records])),
            "hit10": float(np.mean([r[f"{n}_hit10"] for r in query_records])),
        }
        h = headline[n]
        log(f"  {n:<20}  RR={h['RR_mean']:.4f}  hit@1={h['hit1']:.4f}  hit@5={h['hit5']:.4f}  hit@10={h['hit10']:.4f}")

    # Volatility (sim09)
    log("\n=== 8. Per-retriever volatility (sim09) ===")
    volatility = {}
    # Use MiniLM embeddings for sim09 slicing (consistent with multi-retrieval)
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    log("  encoding queries with MiniLM for sim09 pairs...")
    q_embeds_for_sim09 = minilm.encode(
        queries, batch_size=512, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    del minilm
    torch.cuda.empty_cache()

    # BM25 top-K volatility
    vol = compute_volatility_sim09(query_records, "bm25_topk_rank", "bm25_topk_RR", q_embeds_for_sim09)
    volatility["bm25_topk"] = vol
    log(f"  bm25_topk  Hit@1_flip={vol['Hit@1_FlipRate_mean']}  Hit@10_flip={vol['Hit@10_FlipRate_mean']}  RR_Std={vol['RR_Std_mean']}")

    for ce in CROSS_ENCODERS:
        n = ce["name"]
        vol = compute_volatility_sim09(query_records, f"{n}_rank", f"{n}_RR", q_embeds_for_sim09)
        volatility[n] = vol
        log(f"  {n:<20}  Hit@1_flip={vol['Hit@1_FlipRate_mean']}  Hit@10_flip={vol['Hit@10_FlipRate_mean']}  RR_Std={vol['RR_Std_mean']}")

    summary_data = {
        "config": {
            "description": "Stage 5 cross-encoder rerank on v6m strict alignment",
            "stage_a": f"BM25 top-{BM25_K_RERANK}",
            "stage_b": [ce["hf_id"] for ce in CROSS_ENCODERS],
            "n_target_in_topk": n_in_topk,
            "pct_target_in_topk": 100 * n_in_topk / len(queries),
        },
        "headline_per_query_mean": headline,
        "volatility_sim09": volatility,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    # Final print
    log("\n=== Final Headline (cross-encoder rerank on v6m strict alignment) ===")
    header = f"{'retriever':<20} {'RR':>7} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'Hit@1_flip':>11} {'Hit@10_flip':>12} {'RR_Std':>8}"
    log(header)
    log("-" * len(header))
    for label, h, v in [
        ("bm25_topk", headline["bm25_topk"], volatility["bm25_topk"]),
    ] + [(ce["name"], headline[ce["name"]], volatility[ce["name"]]) for ce in CROSS_ENCODERS]:
        h1 = f"{v['Hit@1_FlipRate_mean'] * 100:>10.2f}%" if v.get('Hit@1_FlipRate_mean') is not None else f"{'n/a':>11}"
        h10 = f"{v['Hit@10_FlipRate_mean'] * 100:>11.2f}%" if v.get('Hit@10_FlipRate_mean') is not None else f"{'n/a':>12}"
        rrs = f"{v['RR_Std_mean']:.4f}" if v.get('RR_Std_mean') is not None else f"{'n/a':>8}"
        log(f"{label:<20} {h['RR_mean']:>7.4f} {h['hit1']:>6.4f} {h['hit5']:>6.4f} {h['hit10']:>7.4f} {h1} {h10} {rrs}")
    log(f"\n=== Stage 5 cross-encoder rerank complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()
