"""Syntax Subspace — Stage 5 retrieval on v6m strict alignment selection.

对比 v6k K=200 L2-margin-max 与 v6m M>0 AND d_self ≤ R_95 strict alignment 在
BM25 / MiniLM 上的 retrieval outcome。

输入:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_v6m.json
  (Stage 4 v6m selection, 3781 strict + 3508 no_strict_candidate)

输出:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_v6m_per_query.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_v6m_summary.json

流程:
  1. 读 v6m selection (n_entries=7289), filter out no_strict_candidate (selected=None)
     → 3781 strict queries
  2. 跑 BM25 retrieval (bm25s)
  3. 跑 MiniLM retrieval (GPU)
  4. 输出 retrieval summary + per-query breakdown
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASIN_TO_DOC_CACHE, META_FILE, MINILM_CORPUS_EMBEDS_CACHE, log,
)

SEL_IN = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_v6m.json")
PER_QUERY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_v6m_per_query.json")
SUMMARY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_v6m_summary.json")


def build_meta_corpus():
    """Load Amazon metadata JSONL.gz → {asin: doc_text} (with ASIN_TO_DOC_CACHE reuse)."""
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
    log(f"  corpus size: {len(asin_to_doc)} ASINs")
    return asin_to_doc


def _get_or_build_corpus_embeds(model, corpus_texts: list[str]) -> np.ndarray:
    """Cache MiniLM corpus embeddings to MINILM_CORPUS_EMBEDS_CACHE."""
    cache = Path(MINILM_CORPUS_EMBEDS_CACHE)
    if cache.exists():
        arr = np.load(cache)
        log(f"  ✓ MiniLM corpus embeds cache hit ({arr.shape})")
        return arr
    log("  encoding corpus with MiniLM...")
    t0 = time.time()
    embeds = model.encode(
        corpus_texts,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  encoded in {time.time() - t0:.1f}s, shape={embeds.shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeds)
    return embeds


def main():
    log_start = time.time()
    log("=== Stage 5 retrieval on v6m strict alignment selection ===")

    # ---- 1. Load v6m selection ----
    log("\n=== 1. Loading v6m selection ===")
    selection = json.load(open(SEL_IN))
    entries = selection["entries"]
    log(f"  {len(entries)} total entries")

    by_method = collections.Counter(e["selection_method"] for e in entries)
    log(f"  by method: {dict(by_method)}")

    query_records = []
    n_skipped_no_strict = 0
    for i, e in enumerate(entries):
        q = e["selected"]
        if q is None:
            n_skipped_no_strict += 1
            continue
        query_records.append({
            "entry_idx": i,
            "asin": e["asin"],
            "user_id": e["user_id"],
            "selection_method": e["selection_method"],
            "selected_distance": e["selected_distance"],
            "selected_margin": e["selected_margin"],
            "n_candidates_in_strict": e["n_candidates"],
            "user_source": e["user_source"],
            "query": q["query"],
            "n_tok": q["n_tok"],
            "attrs_covered": q["attrs_covered"],
            "strict": q["strict"],
        })
    log(f"  strict queries to retrieve: {len(query_records)} (skipped {n_skipped_no_strict} no_strict_candidate)")

    # ---- 2. Build ASIN corpus ----
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

    # ---- 3. BM25 retrieval ----
    log("\n=== 3. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    log(f"  tokenizing {len(corpus_texts)} corpus texts...")
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

    log("  retrieving (k=20000, batched)...")
    t0 = time.time()
    BM25_BATCH = 2000
    BM25_K = 20000
    bm25_results = [None] * len(queries)

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(queries), BM25_BATCH):
        e = min(s + BM25_BATCH, len(queries))
        sub_tokens = _slice_tok(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=BM25_K, show_progress=False)
        for i in range(e - s):
            gi = s + i
            tgt_idx = target_indices[gi]
            if tgt_idx < 0:
                bm25_results[gi] = {"rank": None, "RR": 0.0, "hit10": 0}
                continue
            sorted_docs = sub_res.documents[i]
            positions = np.where(sorted_docs == tgt_idx)[0]
            if len(positions) == 0:
                bm25_results[gi] = {"rank": BM25_K + 1, "RR": 1.0 / (BM25_K + 1), "hit10": 0}
            else:
                rank = int(positions[0]) + 1
                bm25_results[gi] = {
                    "rank": rank,
                    "RR": 1.0 / rank,
                    "hit10": 1 if rank <= 10 else 0,
                }
    log(f"  retrieved in {time.time() - t0:.1f}s")

    # ---- 4. MiniLM retrieval ----
    log("\n=== 4. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  MiniLM device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    log(f"  encoding {len(corpus_texts)} corpus docs via MiniLM (GPU)...")
    doc_embeds = _get_or_build_corpus_embeds(model, corpus_texts)
    log(f"  doc_embeds shape: {doc_embeds.shape}")

    log(f"  encoding {len(queries)} queries via MiniLM (GPU)...")
    t0 = time.time()
    q_embeds = model.encode(
        queries,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  queries encoded in {time.time() - t0:.1f}s")
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()

    log("  computing cosine similarities on GPU (batched)...")
    t0 = time.time()
    minilm_results = []
    MINILM_BATCH = 8000
    tgt_idx_all = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    for s in range(0, q_embeds_gpu.shape[0], MINILM_BATCH):
        e = min(s + MINILM_BATCH, q_embeds_gpu.shape[0])
        sims_chunk = q_embeds_gpu[s:e] @ doc_embeds_gpu.T
        tgt_chunk = tgt_idx_all[s:e]
        for j in range(sims_chunk.shape[0]):
            tgt_idx = int(tgt_chunk[j].item())
            if tgt_idx < 0:
                minilm_results.append({"rank": None, "RR": 0.0, "hit10": 0})
                continue
            sc = sims_chunk[j]
            rank = int((sc > sc[tgt_idx]).sum().item()) + 1
            minilm_results.append({
                "rank": rank,
                "RR": 1.0 / rank,
                "hit10": 1 if rank <= 10 else 0,
            })
        del sims_chunk
    log(f"  matmul + ranks done in {time.time() - t0:.1f}s")

    del doc_embeds_gpu, q_embeds_gpu, tgt_idx_all
    torch.cuda.empty_cache()

    # ---- 5. Per-query save ----
    log("\n=== 5. Saving per-query results ===")
    for r, bm25_r, minilm_r in zip(query_records, bm25_results, minilm_results):
        r["bm25_rank"] = bm25_r["rank"]
        r["bm25_RR"] = bm25_r["RR"]
        r["bm25_hit10"] = bm25_r["hit10"]
        r["minilm_rank"] = minilm_r["rank"]
        r["minilm_RR"] = minilm_r["RR"]
        r["minilm_hit10"] = minilm_r["hit10"]

    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 retrieval on v6m strict alignment selection",
                "selection_file": str(SEL_IN),
                "selection_method_filter": "strict_personalized_l2_margin_max only",
                "corpus_size": len(asins),
                "bm25_method": "bm25s lucene k1=1.5 b=0.75",
                "minilm_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # ---- 6. Per-ASIN + aggregate summary ----
    log("\n=== 6. Per-ASIN retrieval summary ===")
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in query_records:
        by_asin[r["asin"]].append(r)

    per_asin_summary = []
    for asin, qs in by_asin.items():
        bm25_rrs = [q["bm25_RR"] for q in qs]
        minilm_rrs = [q["minilm_RR"] for q in qs]
        per_asin_summary.append({
            "asin": asin,
            "n_queries": len(qs),
            "mean_d_self": float(np.mean([q["selected_distance"] for q in qs])),
            "mean_M": float(np.mean([q["selected_margin"] for q in qs])),
            "bm25_RR_mean": float(np.mean(bm25_rrs)),
            "bm25_RR_median": float(np.median(bm25_rrs)),
            "bm25_hit10_frac": float(np.mean([q["bm25_hit10"] for q in qs])),
            "bm25_rank_median": float(np.median([q["bm25_rank"] for q in qs])),
            "minilm_RR_mean": float(np.mean(minilm_rrs)),
            "minilm_RR_median": float(np.median(minilm_rrs)),
            "minilm_hit10_frac": float(np.mean([q["minilm_hit10"] for q in qs])),
            "minilm_rank_median": float(np.median([q["minilm_rank"] for q in qs])),
        })

    n_asins = len(per_asin_summary)
    log(f"  ASINs with ≥1 strict query: {n_asins}")

    def _agg(key):
        vals = [a[key] for a in per_asin_summary if a[key] is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "n_total_queries": len(query_records),
        "n_total_pairs": len(entries),
        "n_strict_personalized": len(query_records),
        "n_no_strict_candidate": n_skipped_no_strict,
        "n_asins": n_asins,
        "bm25": {
            "RR_mean_per_query": float(np.mean([q["bm25_RR"] for q in query_records])),
            "RR_median_per_query": float(np.median([q["bm25_RR"] for q in query_records])),
            "RR_mean_per_asin_mean": _agg("bm25_RR_mean"),
            "RR_mean_per_asin_median": float(np.median([a["bm25_RR_mean"] for a in per_asin_summary])),
            "hit10_per_query": float(np.mean([q["bm25_hit10"] for q in query_records])),
            "hit10_per_asin_mean": _agg("bm25_hit10_frac"),
            "rank_median_per_query": float(np.median([q["bm25_rank"] for q in query_records])),
            "rank_median_per_asin_median": float(np.median([a["bm25_rank_median"] for a in per_asin_summary])),
        },
        "minilm": {
            "RR_mean_per_query": float(np.mean([q["minilm_RR"] for q in query_records])),
            "RR_median_per_query": float(np.median([q["minilm_RR"] for q in query_records])),
            "RR_mean_per_asin_mean": _agg("minilm_RR_mean"),
            "RR_mean_per_asin_median": float(np.median([a["minilm_RR_mean"] for a in per_asin_summary])),
            "hit10_per_query": float(np.mean([q["minilm_hit10"] for q in query_records])),
            "hit10_per_asin_mean": _agg("minilm_hit10_frac"),
            "rank_median_per_query": float(np.median([q["minilm_rank"] for q in query_records])),
            "rank_median_per_asin_median": float(np.median([a["minilm_rank_median"] for a in per_asin_summary])),
        },
    }

    summary_data = {
        "config": {
            "description": "Stage 5 retrieval on v6m strict alignment selection (3781 strict queries)",
            "selection_file": str(SEL_IN),
            "gate": "M(q,u) > 0 AND d_self(q,u) ≤ R_95=11.308",
            "selection_logic": "argmax M under gate (strict personalized)",
            "corpus_size": len(asins),
        },
        "aggregate": aggregate,
        "per_asin": per_asin_summary,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    # ---- 7. Headline print ----
    log("\n=== Headline (v6m strict personalized) ===")
    log(f"  total strict queries: {len(query_records)} / pairs {len(entries)} ({100 * len(query_records) / len(entries):.1f}%)")
    log(f"  ASINs with ≥1 strict: {n_asins}")
    log(f"  BM25: RR={aggregate['bm25']['RR_mean_per_query']:.4f} (per-Q mean), "
        f"hit@10={aggregate['bm25']['hit10_per_query']:.4f}, rank_med={aggregate['bm25']['rank_median_per_query']:.1f}")
    log(f"  MiniLM: RR={aggregate['minilm']['RR_mean_per_query']:.4f} (per-Q mean), "
        f"hit@10={aggregate['minilm']['hit10_per_query']:.4f}, rank_med={aggregate['minilm']['rank_median_per_query']:.1f}")
    log(f"  BM25 (per-ASIN mean): RR={aggregate['bm25']['RR_mean_per_asin_mean']:.4f}, "
        f"hit@10={aggregate['bm25']['hit10_per_asin_mean']:.4f}")
    log(f"  MiniLM (per-ASIN mean): RR={aggregate['minilm']['RR_mean_per_asin_mean']:.4f}, "
        f"hit@10={aggregate['minilm']['hit10_per_asin_mean']:.4f}")
    log(f"\n=== Stage 5 v6m retrieval complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()
