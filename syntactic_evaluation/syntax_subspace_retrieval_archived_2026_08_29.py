"""Syntax Subspace — Stage 5 (volatility-only retrieval).

归 syntactic_evaluation/: 用 bm25s + GPU MiniLM 对 Stage 4 选出的
selected queries 做 retrieval,然后计算 per-ASIN stability flip
(Hit@1/@5/@10 Flip Rate, RR Std 描述性指标)。

用户指令 2026-08-27: 只做波动率指标的评估,不做 selected vs random
vs farthest 三组对比。Stage 4 selection 仍然保留三组策略(stage8_5_selection.json),
但本 stage 只 retrieve selected queries,然后算 per-ASIN stability flip。

用户指令 2026-08-27 (later): 加回 RR Std 评估(RR = 1/rank 的 std per ASIN,
聚合 mean/median/std across ASINs)。不做 permutation sig test — RR Std
只是描述性指标,显著性留给下游分析决定。

用法:
  python syntactic_evaluation/syntax_subspace_retrieval.py --stage retrieval

I/O 路径:
  输入: stage8_5_selection.json
        data/meta_Baby_Products_2023.jsonl.gz (ASIN metadata corpus)
  输出: result/syntactic_evaluation/volatility.json  (Stability flip + RR Std)
        scratch2/stage8_5_retrieval_per_query.json   (per-query intermediate)
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASIN_TO_DOC_CACHE, FEAT_CACHE, META_FILE, MINILM_CORPUS_EMBEDS_CACHE, POOL_IN,
    RETRIEVAL_PER_QUERY_OUT, RETRIEVAL_SUMMARY_OUT,
    SELECTION_IN, VOLATILITY_SUMMARY_OUT,
    K_SET, LEN_BAND_FALLBACK, MIN_LEN, PCA_DIM, PRIMARY_LEN_DELTA, log, feat_key,
)


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
                asin_to_doc[asin] = doc[:1000]
    log(f"  loaded {len(asin_to_doc)} ASIN docs")
    ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(ASIN_TO_DOC_CACHE, "w", encoding="utf-8") as f:
        json.dump(asin_to_doc, f, ensure_ascii=False)
    log(f"  cached → {ASIN_TO_DOC_CACHE} ({ASIN_TO_DOC_CACHE.stat().st_size/1e6:.1f} MB)")
    return asin_to_doc


def _get_or_build_corpus_embeds(model, corpus_texts: list[str]) -> np.ndarray:
    """Load cached MiniLM corpus embeddings (Part A→B reuse) or encode + save.

    Cache key: `MINILM_CORPUS_EMBEDS_CACHE` (scratch2/wenyu/gaussian_vades/minilm_corpus_embeds.npy).
    Validation: cached shape[0] must equal len(corpus_texts).
    """
    n = len(corpus_texts)
    if MINILM_CORPUS_EMBEDS_CACHE.exists():
        cached = np.load(MINILM_CORPUS_EMBEDS_CACHE)
        if cached.shape[0] == n:
            log(f"  ✓ cache hit ({cached.shape}, {MINILM_CORPUS_EMBEDS_CACHE.stat().st_size/1e6:.1f} MB)")
            return cached
        log(f"  ⚠ cache stale (cached={cached.shape[0]}, current={n}), re-encoding...")
    else:
        log(f"  no cache, encoding {n} corpus docs (batch=2048)...")
    t0 = time.time()
    doc_embeds = model.encode(
        corpus_texts,
        batch_size=2048,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  docs encoded in {time.time() - t0:.1f}s, shape: {doc_embeds.shape}")
    np.save(MINILM_CORPUS_EMBEDS_CACHE, doc_embeds)
    log(f"  cached → {MINILM_CORPUS_EMBEDS_CACHE} ({MINILM_CORPUS_EMBEDS_CACHE.stat().st_size/1e6:.1f} MB)")
    return doc_embeds


# ===========================================================================
# STAGE 5 — RETRIEVAL
# ===========================================================================

def stage_retrieval():
    log("=== STAGE 5 — RETRIEVAL ===")

    log("\n=== 1. Loading selection (selected variant only) ===")
    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    log(f"  {len(entries)} entries")

    # 用户指令 2026-08-27: 只做 selected 的 volatility 评估,
    # 省掉 random/farthest, queries 从 56760 → 18920 (3x 加速)
    query_records = []
    for i, e in enumerate(entries):
        q = e["selected"]
        if q is None:
            continue
        query_records.append({
            "entry_idx": i,
            "asin": e["asin"],
            "user_id": e["user_id"],
            "variant": "selected",
            "selection_method": e["selection_method"],
            "user_source": e["user_source"],
            "query": q["query"],
            "n_tok": q["n_tok"],
            "attrs_covered": q["attrs_covered"],
            "strict": q["strict"],
        })
    log(f"  total queries to retrieve: {len(query_records)}")

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

    log("\n=== 3. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
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

    log(f"  retrieving (k=20000, batched to avoid OOM)...")
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

    log(f"  computing cosine similarities on GPU (batched to avoid OOM)...")
    t0 = time.time()
    minilm_results = []
    # 用户指令 2026-08-27: 分批 matmul, 避免 56k × 217k = 46 GiB OOM
    # B=8000 -> 8000 × 217722 × 4B = 6.97 GiB per batch (safe within 39 GiB)
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

    log("\n=== 5. Saving ===")
    for r, bm25_r, minilm_r in zip(query_records, bm25_results, minilm_results):
        r["bm25_rank"] = bm25_r["rank"]
        r["bm25_RR"] = bm25_r["RR"]
        r["bm25_hit10"] = bm25_r["hit10"]
        r["minilm_rank"] = minilm_r["rank"]
        r["minilm_RR"] = minilm_r["RR"]
        r["minilm_hit10"] = minilm_r["hit10"]

    with open(RETRIEVAL_PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5 retrieval per query (bm25s + GPU MiniLM) on ASIN meta corpus",
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_PER_QUERY_OUT}")

    # === Stability flip + RR Std (per-ASIN Hit@K flip + RR Std 描述性) ===
    stability_flip = _compute_stability_flip_metrics(RETRIEVAL_PER_QUERY_OUT)
    with open(VOLATILITY_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stability flip + RR Std on selected_only_sim09 (MiniLM cosine ≥ 0.9) "
                                "per-ASIN Hit@1/@5/@10/@20 flip rate + RR Std mean/median/std across ASINs"),
                "slices": ["selected_only_sim09"],
            },
            "stability_flip": stability_flip,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {VOLATILITY_SUMMARY_OUT}")


def _compute_stability_flip_metrics(retrieval_per_query_path) -> dict:
    """Per-ASIN Hit@K Flip Rate + RR Std across selected queries (sim09 slice only).

    用户指令 2026-08-28: 只保留 'selected_only_sim09' — 只算 MiniLM cosine ≥ 0.9 的
    query pair,排除 "内容/关键词不同" 的 pair (e.g., "I need X in Y" vs "I'm hoping
    to find X in Y"), isolate "内容相同但风格骨架不同" 的纯 style volatility。
    之前 3 个 slice (all / lm3 / sim09) 中 all 和 lm3 删掉 — 用户决定只看 sim09。

    Flip rate definition: of all unique query pairs within the slice, fraction
    of pairs whose top-K hit/miss labels disagree (one hit, one miss)。

    RR Std: per ASIN, std of 1/rank across queries. Aggregated as mean/median/std
    over ASINs. 描述性指标,不做显著性测试。
    """
    log("\n=== Computing Hit@K Flip Rate + RR Std (per ASIN, per retriever, sim09) ===")
    data = json.load(open(retrieval_per_query_path))
    queries = data["queries"]
    log(f"  loaded {len(queries)} queries from {retrieval_per_query_path.name}")

    # group by asin, keep only selected variant
    by_asin: dict[str, list] = collections.defaultdict(list)
    for q in queries:
        if q.get("variant") == "selected":
            by_asin[q["asin"]].append(q)

    # 为 sim09 slice 重新 encode selected queries (7250 × 384),~2s GPU
    log("  encoding selected queries with MiniLM for semantic-sim pairs...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    selected_qs = [q for qs in by_asin.values() for q in qs]
    q_embeds_arr = model.encode(
        [q["query"] for q in selected_qs],
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # index back into by_asin
    embed_map = {id(q): q_embeds_arr[i] for i, q in enumerate(selected_qs)}
    log(f"  q_embeds shape: {q_embeds_arr.shape} (cached in memory)")

    def flip_rate_sim(hit_labels: list[int], sims: list[float],
                      threshold: float = 0.9) -> tuple[float | None, int, int]:
        """Semantic-similarity matched flip rate: 仅 sim >= threshold 的 query pairs。

        用户指令 2026-08-28: 排除 "内容不同" 的 pair, 只算 "内容相同但风格骨架不同"
        的 flip。threshold=0.9 是 MiniLM cosine 上的常用 paraphrasing cut-off。

        Returns (flip_rate, n_pairs_used, n_total_pairs)。
        """
        n = len(hit_labels)
        if n < 2:
            return None, 0, 0
        total_pairs = n * (n - 1) // 2
        used_pairs = 0
        disagree = 0
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i] is not None and sims[j] is not None and sims[i][j] >= threshold:
                    used_pairs += 1
                    if hit_labels[i] != hit_labels[j]:
                        disagree += 1
        if used_pairs == 0:
            return None, 0, total_pairs
        return float(disagree / used_pairs), used_pairs, total_pairs

    sub: dict = {}
    for retriever in ("bm25", "minilm"):
        rank_key = f"{retriever}_rank"
        rr_key = f"{retriever}_RR"
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
            # RR Std: std of 1/rank across queries (Stage 10H convention)
            rrs = [float(q.get(rr_key) or 0.0) for q in qs]
            rr_std = float(np.std(rrs, ddof=0)) if len(rrs) >= 2 else None
            embeds = np.stack([embed_map[id(q)] for q in qs], axis=0)
            # (n, n) cosine sim matrix; rows/cols aligned with qs order
            sim_mat = embeds @ embeds.T
            f1, _, _ = flip_rate_sim(hit1, [sim_mat[i] for i in range(len(qs))])
            f5, _, _ = flip_rate_sim(hit5, [sim_mat[i] for i in range(len(qs))])
            f10, _, _ = flip_rate_sim(hit10, [sim_mat[i] for i in range(len(qs))])
            f20, _, _ = flip_rate_sim(hit20, [sim_mat[i] for i in range(len(qs))])
            if f1 is not None:
                f1_list.append(f1)
            if f5 is not None:
                f5_list.append(f5)
            if f10 is not None:
                f10_list.append(f10)
            if f20 is not None:
                f20_list.append(f20)
            if rr_std is not None:
                rr_std_list.append(rr_std)
            n_asins_used += 1
        sub[retriever] = {
            "n_asins": n_asins_used,
            "Hit@1_FlipRate_mean": float(np.mean(f1_list)) if f1_list else None,
            "Hit@1_FlipRate_median": float(np.median(f1_list)) if f1_list else None,
            "Hit@5_FlipRate_mean": float(np.mean(f5_list)) if f5_list else None,
            "Hit@5_FlipRate_median": float(np.median(f5_list)) if f5_list else None,
            "Hit@10_FlipRate_mean": float(np.mean(f10_list)) if f10_list else None,
            "Hit@10_FlipRate_median": float(np.median(f10_list)) if f10_list else None,
            "Hit@20_FlipRate_mean": float(np.mean(f20_list)) if f20_list else None,
            "Hit@20_FlipRate_median": float(np.median(f20_list)) if f20_list else None,
            "RR_Std_mean": float(np.mean(rr_std_list)) if rr_std_list else None,
            "RR_Std_median": float(np.median(rr_std_list)) if rr_std_list else None,
            "RR_Std_std": float(np.std(rr_std_list, ddof=0)) if rr_std_list else None,
        }

    summary: dict = {"selected_only_sim09": sub}

    for retr in ("bm25", "minilm"):
        s = sub[retr]

        def _f(v):
            return f"{v:.4f}" if v is not None else "n/a"

        log(f"  selected_only_sim09 × {retr.upper()}: "
            f"Hit@1={_f(s['Hit@1_FlipRate_mean'])}  "
            f"Hit@5={_f(s['Hit@5_FlipRate_mean'])}  "
            f"Hit@10={_f(s['Hit@10_FlipRate_mean'])}  "
            f"Hit@20={_f(s['Hit@20_FlipRate_mean'])}  "
            f"RR_Std={_f(s['RR_Std_mean'])}")

    # 用户指令 2026-08-28: 显式 Hit@1/5/10 flip + RR Std summary block (sim09 only)
    log("\n=== Headline: Hit@1/5/10 Flip Rate + RR Std (sim09) ===")
    header = f"{'retriever':<10} {'Hit@1_flip':>12} {'Hit@5_flip':>12} {'Hit@10_flip':>12} {'RR_Std':>10}"
    log(header)
    log("-" * len(header))
    for retr in ("bm25", "minilm"):
        s = sub[retr]

        def _p(v):
            return f"{v * 100:>10.3f}%" if v is not None else f"{'n/a':>10}"

        log(f"{retr.upper():<10} "
            f"{_p(s['Hit@1_FlipRate_mean'])} "
            f"{_p(s['Hit@5_FlipRate_mean'])} "
            f"{_p(s['Hit@10_FlipRate_mean'])} "
            f"{_p(s['RR_Std_mean'])}")
    return summary


# ===========================================================================
# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "retrieval": stage_retrieval,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — syntactic_evaluation (retrieval + volatility)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_retrieval.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in STAGE_FUNCTIONS:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()