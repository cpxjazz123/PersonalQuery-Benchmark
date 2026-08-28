"""Stage 5 multi-retriever on v6m strict alignment selection.

按用户 2026-08-28 指令,在 v6m 3781 strict queries 上跑 5 个 retriever:
  1. BM25 (bm25s, lucene k1=1.5 b=0.75) — 已 baseline
  2. MiniLM-L6-v2 (384d) — 已 baseline
  3. MPNet-base-v2 (768d)
  4. BGE-base-en-v1.5 (768d, BAAI)
  5. GTE-base (768d, Alibaba)

所有 dense retriever 走 sentence-transformers + GPU batched encoding。
输出 per-query (5 retrievers 合并) + per-retriever 头对头比较 +
per-ASIN volatility (Hit@1/5/10 flip + RR Std sim09) across retrievers。

输入: stage8_5_selection.json (v6m canonical, 3781 strict)
输出:
  scratch2/.../stage8_5_multiretrieval_per_query.json
  scratch2/.../stage8_5_multiretrieval_summary.json
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
PER_QUERY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_multiretrieval_per_query.json")
SUMMARY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_multiretrieval_summary.json")
EMBED_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/multiretrieval_embeds")

# 5 retrievers
RETRIEVERS = [
    {"name": "bm25", "kind": "sparse"},
    {"name": "minilm", "kind": "dense", "hf_id": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
    {"name": "mpnet", "kind": "dense", "hf_id": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
    {"name": "bge_base_v15", "kind": "dense", "hf_id": "BAAI/bge-base-en-v1.5", "dim": 768},
    {"name": "gte_base", "kind": "dense", "hf_id": "thenlper/gte-base", "dim": 768},
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


def encode_with_cache(model, texts: list[str], cache_path: Path, batch_size: int = 256) -> np.ndarray:
    """Encode texts; cache to .npy for reuse across queries."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        arr = np.load(cache_path)
        log(f"  ✓ cache hit ({arr.shape}) ← {cache_path.name}")
        return arr
    log(f"  encoding {len(texts)} texts (batch={batch_size})...")
    t0 = time.time()
    embeds = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  encoded in {time.time() - t0:.1f}s, shape={embeds.shape}")
    np.save(cache_path, embeds)
    return embeds


def bm25_retrieve(queries: list[str], corpus_texts: list[str], target_indices: np.ndarray) -> list[dict]:
    """BM25 retrieval (bm25s lucene k1=1.5 b=0.75)."""
    import bm25s
    log("\n=== BM25 retrieval ===")
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
    results = [None] * len(queries)

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
                results[gi] = {"rank": None, "RR": 0.0, "hit10": 0}
                continue
            sorted_docs = sub_res.documents[i]
            positions = np.where(sorted_docs == tgt_idx)[0]
            if len(positions) == 0:
                results[gi] = {"rank": BM25_K + 1, "RR": 1.0 / (BM25_K + 1), "hit10": 0}
            else:
                rank = int(positions[0]) + 1
                results[gi] = {"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0}
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return results


def dense_retrieve(
    retr_name: str,
    hf_id: str,
    queries: list[str],
    corpus_embeds_gpu: torch.Tensor,
    target_indices: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Dense retrieval: encode queries (cached), matmul with cached corpus embeds."""
    from sentence_transformers import SentenceTransformer

    log(f"\n=== {retr_name} ({hf_id}) ===")
    device = "cuda"
    model = SentenceTransformer(hf_id, device=device)

    cache_dir = EMBED_CACHE_DIR / retr_name
    corpus_cache = cache_dir / "corpus_embeds.npy"
    query_cache = cache_dir / "query_embeds.npy"

    # corpus already cached?
    if not corpus_cache.exists():
        log(f"  encoding corpus...")
        # Load corpus texts again for encoding
        asin_to_doc = build_meta_corpus()
        asins = sorted(asin_to_doc.keys())
        corpus_texts = [asin_to_doc[a] for a in asins]
        corpus_embeds = encode_with_cache(model, corpus_texts, corpus_cache, batch_size=256)
    else:
        corpus_embeds = np.load(corpus_cache)

    q_embeds = encode_with_cache(model, queries, query_cache, batch_size=512)

    # matmul + ranks
    log(f"  matmul + ranks on GPU...")
    t0 = time.time()
    corpus_gpu = torch.from_numpy(corpus_embeds).cuda()
    q_gpu = torch.from_numpy(q_embeds).cuda()
    results = []
    MINILM_BATCH = 4000  # smaller for 768d (4x more memory)
    tgt_all = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    for s in range(0, q_gpu.shape[0], MINILM_BATCH):
        e = min(s + MINILM_BATCH, q_gpu.shape[0])
        sims_chunk = q_gpu[s:e] @ corpus_gpu.T
        tgt_chunk = tgt_all[s:e]
        for j in range(sims_chunk.shape[0]):
            tgt_idx = int(tgt_chunk[j].item())
            if tgt_idx < 0:
                results.append({"rank": None, "RR": 0.0, "hit10": 0})
                continue
            sc = sims_chunk[j]
            rank = int((sc > sc[tgt_idx]).sum().item()) + 1
            results.append({
                "rank": rank,
                "RR": 1.0 / rank,
                "hit10": 1 if rank <= 10 else 0,
            })
        del sims_chunk
    log(f"  matmul done in {time.time() - t0:.1f}s")
    del corpus_gpu, q_gpu, tgt_all
    torch.cuda.empty_cache()
    # also return q_embeds for sim09 slicing
    return results, q_embeds


def compute_volatility_by_retriever(per_query: list[dict], retr_name: str, q_embeds: np.ndarray | None = None,
                                    sim_threshold: float = 0.9) -> dict:
    """Hit@K Flip Rate + RR Std per-ASIN (sim09 slice)."""
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in per_query:
        by_asin[r["asin"]].append(r)

    rank_key = f"{retr_name}_rank"
    rr_key = f"{retr_name}_RR"

    if q_embeds is None:
        log(f"  ({retr_name}) no q_embeds, skipping volatility (BM25)")
        return {"n_asins": 0}

    n = sum(len(qs) for qs in by_asin.values())
    log(f"  ({retr_name}) encoding {n} queries for sim09 pairs...")

    # build index by asin query order
    selected_qs = [q for qs in by_asin.values() for q in qs]
    qid_to_embed = {id(q): q_embeds[i] for i, q in enumerate(selected_qs)}

    f1_list, f5_list, f10_list, f20_list = [], [], [], []
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
        hit20 = [1 if r <= 20 else 0 for r in ranks]
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

        for hit_labels, lst in [(hit1, f1_list), (hit5, f5_list), (hit10, f10_list), (hit20, f20_list)]:
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
        "Hit@20_FlipRate_mean": float(np.mean(f20_list)) if f20_list else None,
        "RR_Std_mean": float(np.mean(rr_std_list)) if rr_std_list else None,
        "RR_Std_median": float(np.median(rr_std_list)) if rr_std_list else None,
        "RR_Std_std": float(np.std(rr_std_list, ddof=0)) if rr_std_list else None,
    }


def main():
    log_start = time.time()
    log("=== Stage 5 multi-retriever on v6m strict alignment ===")
    log(f"  retrievers: {[r['name'] for r in RETRIEVERS]}")

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
    log(f"  strict queries to retrieve: {len(query_records)}")

    # ---- 2. Build corpus ----
    log("\n=== 2. Building ASIN corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    n_missing = int((target_indices < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing} queries have missing target ASINs in corpus")

    # ---- 3. Retrieve with each retriever ----
    log("\n=== 3. Per-retriever retrieval ===")
    retr_results: dict[str, list[dict]] = {}
    retr_q_embeds: dict[str, np.ndarray] = {}

    for retr in RETRIEVERS:
        if retr["kind"] == "sparse":
            results = bm25_retrieve(queries, [asin_to_doc[a] for a in asins], target_indices)
            retr_results[retr["name"]] = results
            retr_q_embeds[retr["name"]] = None  # BM25 has no embeddings
        else:
            results, q_embeds = dense_retrieve(
                retr["name"], retr["hf_id"], queries, None, target_indices,
            )
            retr_results[retr["name"]] = results
            retr_q_embeds[retr["name"]] = q_embeds

    # ---- 4. Per-query save ----
    log("\n=== 4. Saving per-query results ===")
    for r in query_records:
        for retr in RETRIEVERS:
            n = retr["name"]
            res = retr_results[n][query_records.index(r)]
            r[f"{n}_rank"] = res["rank"]
            r[f"{n}_RR"] = res["RR"]
            r[f"{n}_hit10"] = res["hit10"]

    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 multi-retriever on v6m strict alignment",
                "retrievers": [r["name"] for r in RETRIEVERS],
                "selection_file": str(SEL_IN),
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # ---- 5. Per-ASIN aggregate + volatility ----
    log("\n=== 5. Per-ASIN aggregate + volatility ===")
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in query_records:
        by_asin[r["asin"]].append(r)

    per_asin_summary = []
    for asin, qs in by_asin.items():
        entry = {"asin": asin, "n_queries": len(qs)}
        for retr in RETRIEVERS:
            n = retr["name"]
            rrs = [q[f"{n}_RR"] for q in qs]
            ranks = [q[f"{n}_rank"] for q in qs]
            hit10s = [q[f"{n}_hit10"] for q in qs]
            entry[f"{n}_RR_mean"] = float(np.mean(rrs))
            entry[f"{n}_RR_median"] = float(np.median(rrs))
            entry[f"{n}_hit10"] = float(np.mean(hit10s))
            entry[f"{n}_rank_median"] = float(np.median(ranks))
        per_asin_summary.append(entry)

    # ---- 6. Headline + volatility ----
    log("\n=== 6. Headline (per-query mean) ===")
    headline = {}
    for retr in RETRIEVERS:
        n = retr["name"]
        rrs = [q[f"{n}_RR"] for q in query_records]
        hit10s = [q[f"{n}_hit10"] for q in query_records]
        ranks = [q[f"{n}_rank"] for q in query_records]
        headline[n] = {
            "RR_mean": float(np.mean(rrs)),
            "RR_median": float(np.median(rrs)),
            "hit10": float(np.mean(hit10s)),
            "rank_median": float(np.median(ranks)),
        }
        log(f"  {n:<12} RR={headline[n]['RR_mean']:.4f}  hit@10={headline[n]['hit10']:.4f}  rank_med={headline[n]['rank_median']:.0f}")

    # Volatility per retriever
    log("\n=== 7. Per-retriever volatility (sim09) ===")
    volatility = {}
    for retr in RETRIEVERS:
        if retr["name"] == "bm25":
            # BM25 has no embed; use BM25 self-similarity via embedding separately if needed
            # For now skip BM25 volatility (BM25 sim not meaningful)
            volatility[retr["name"]] = {"n_asins": 0, "note": "BM25 sim09 not applicable (lexical retrieval, no semantic embedding)"}
            continue
        v = compute_volatility_by_retriever(query_records, retr["name"], retr_q_embeds[retr["name"]])
        volatility[retr["name"]] = v
        log(f"  {retr['name']:<12} Hit@1_flip={v.get('Hit@1_FlipRate_mean')}  Hit@10_flip={v.get('Hit@10_FlipRate_mean')}  RR_Std={v.get('RR_Std_mean')}")

    summary_data = {
        "config": {
            "description": "Stage 5 multi-retriever on v6m strict alignment",
            "retrievers": [r["name"] for r in RETRIEVERS],
            "selection_file": str(SEL_IN),
        },
        "headline_per_query_mean": headline,
        "volatility_sim09": volatility,
        "per_asin": per_asin_summary,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    # Final headline print
    log("\n=== Final Headline (multi-retriever on v6m strict alignment) ===")
    header = f"{'retriever':<14} {'RR_mean':>8} {'hit@10':>8} {'rank_med':>9} {'Hit@1_flip':>11} {'Hit@10_flip':>12} {'RR_Std':>8}"
    log(header)
    log("-" * len(header))
    for retr in RETRIEVERS:
        n = retr["name"]
        h = headline[n]
        v = volatility[n]
        h1 = f"{v['Hit@1_FlipRate_mean'] * 100:>10.2f}%" if v.get("Hit@1_FlipRate_mean") is not None else f"{'n/a':>11}"
        h10 = f"{v['Hit@10_FlipRate_mean'] * 100:>11.2f}%" if v.get("Hit@10_FlipRate_mean") is not None else f"{'n/a':>12}"
        rrs = f"{v['RR_Std_mean']:.4f}" if v.get("RR_Std_mean") is not None else f"{'n/a':>8}"
        log(f"{n:<14} {h['RR_mean']:>8.4f} {h['hit10']:>8.4f} {h['rank_median']:>9.0f} {h1} {h10} {rrs}")
    log(f"\n=== Stage 5 multi-retriever complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()
