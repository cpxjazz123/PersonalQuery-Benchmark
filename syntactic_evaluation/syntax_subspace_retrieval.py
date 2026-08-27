"""Syntax Subspace — Stage 5 (retrieval + volatility).

归 syntactic_evaluation/: 用 bm25s + GPU MiniLM 评估选出来的查询能否
把对应 ASIN 拉回 rank-1,并用 V_low / V_user / V_high 标定查询波动率。

Stage 5:
  Part A: Stage 4 选出的 {selected, random, farthest} 三组查询全量跑 BM25 + MiniLM
  Part B: 为每个 ASIN 构造 length-matched low / user / high 3 组查询,验证 user
          queries 落在 retriever trust region(V_user < V_low)

用法:
  python syntactic_evaluation/syntax_subspace_retrieval.py --stage retrieval

I/O 路径:
  输入: stage8_5_selection.json
        data/meta_Baby_Products_2023.jsonl.gz (ASIN metadata corpus)
        stage8_5_pool.json / stage7b_query_features.jsonl.gz
  输出: stage8_5_retrieval_per_query.json + stage8_5_retrieval_summary.json (Part A)
        stage8_5v_volatility.json                                         (Part B)

共享工具 (log, feat_key, paths, hyperparams, _syntax_subspace_prepare) 来自:
  common/syntax_subspace_utils.py
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

    log(f"  computing cosine similarities on GPU...")
    t0 = time.time()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    log(f"  matmul done in {time.time() - t0:.1f}s")
    log(f"  sorting similarities...")
    t0 = time.time()
    minilm_results = []
    for i in range(sims.shape[0]):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            minilm_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_results.append({
            "rank": rank,
            "RR": 1.0 / rank,
            "hit10": 1 if rank <= 10 else 0,
        })
    log(f"  ranks computed in {time.time() - t0:.1f}s")

    del doc_embeds_gpu, q_embeds_gpu, sims
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

    log("\n=== 6. Summary by variant ===")
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

    with open(RETRIEVAL_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_SUMMARY_OUT}")

    # === Part B: V_low / V_user / V_high volatility calibration ===
    _volatility_eval()


# ===========================================================================
# PART B — VOLATILITY EVAL (invoked at the end of stage_retrieval)
# ===========================================================================

def _volatility_eval():
    log("=== STAGE 5 PART B — VOLATILITY (V_low / V_user / V_high) ===")

    log("\n=== 1. Loading inputs ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready, EV={pca.explained_variance_ratio_.sum():.4f}")

    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  loaded {len(feat_map)} features")

    log("\n=== 2. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs.append({"z": z, "q": q["query"], "n_tok": q["n_tok"], "strict": q["strict"], "k": q["k"]})
        pool_z[asin] = zs
    log(f"  pool_z: {len(pool_z)} ASINs, missed {miss} features")

    log("\n=== 3. Building V_low / V_user / V_high query sets ===")
    asin_query_sets = {}
    n_insufficient_low = 0
    n_insufficient_high = 0
    n_total = 0
    seen_asins = set()
    for entry in entries:
        asin = entry["asin"]
        if asin in seen_asins:
            continue  # already processed this ASIN via earlier (asin,user) entry
        seen_asins.add(asin)
        if asin not in pool_z:
            continue
        cands = [c for c in pool_z[asin] if c["strict"] and c["n_tok"] >= MIN_LEN]
        if len(cands) < K_SET * 2:
            continue
        zs = np.array([c["z"] for c in cands])
        n = len(cands)
        dist = np.zeros((n, n))
        for i in range(n):
            diff = zs - zs[i]
            dist[i] = np.linalg.norm(diff, axis=1)
        mean_dist = (dist.sum(axis=1) - np.diag(dist)) / (n - 1)
        sorted_idx = np.argsort(mean_dist)

        def length_match(anchor_idx, max_diff):
            ok = []
            a_tok = cands[anchor_idx]["n_tok"]
            for j in range(n):
                if abs(cands[j]["n_tok"] - a_tok) <= max_diff:
                    ok.append(j)
            return ok

        anchor_lo = sorted_idx[0]
        ok_idx_lo = []
        for max_diff in [PRIMARY_LEN_DELTA, LEN_BAND_FALLBACK, 10]:
            ok_idx_lo = length_match(anchor_lo, max_diff)
            if len(ok_idx_lo) >= K_SET:
                break
        if len(ok_idx_lo) < K_SET:
            n_insufficient_low += 1
            continue
        ok_dist_lo = dist[anchor_lo][ok_idx_lo]
        sorted_ok_lo = np.array(ok_idx_lo)[np.argsort(ok_dist_lo)][:K_SET]
        low_qs = [cands[i]["q"] for i in sorted_ok_lo]

        anchor_hi = sorted_idx[-1]
        ok_idx_hi = []
        for max_diff in [PRIMARY_LEN_DELTA, LEN_BAND_FALLBACK, 10]:
            ok_idx_hi = length_match(anchor_hi, max_diff)
            if len(ok_idx_hi) >= K_SET:
                break
        if len(ok_idx_hi) < K_SET:
            n_insufficient_high += 1
            continue
        ok_dist_hi = dist[anchor_hi][ok_idx_hi]
        sorted_ok_hi = np.array(ok_idx_hi)[np.argsort(-ok_dist_hi)][:K_SET]
        high_qs = [cands[i]["q"] for i in sorted_ok_hi]

        user_queries = {}
        for e_idx, e in enumerate(entries):
            if e["asin"] != asin:
                continue
            sel = e["selected"]
            if sel:
                user_queries[e["user_id"]] = sel["query"]
        if not user_queries:
            continue
        asin_query_sets[asin] = {
            "low": low_qs,
            "high": high_qs,
            "user_queries": user_queries,
        }
        n_total += 1

    log(f"  ASINs with all 3 sets: {n_total}")
    log(f"    insufficient low: {n_insufficient_low}")
    log(f"    insufficient high: {n_insufficient_high}")

    log("\n=== 4. Building ASIN metadata corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    log("\n=== 5. Building flat query list ===")
    flat_queries = []
    for asin, sets in asin_query_sets.items():
        for set_type in ("low", "high"):
            for q in sets[set_type]:
                flat_queries.append({
                    "asin": asin,
                    "set_type": set_type,
                    "user_id": None,
                    "query": q,
                })
        for uid, q in sets["user_queries"].items():
            flat_queries.append({
                "asin": asin,
                "set_type": "user",
                "user_id": uid,
                "query": q,
            })
    log(f"  total queries: {len(flat_queries)}")

    log("\n=== 6. BM25 retrieval (bm25s, batched) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    queries = [r["query"] for r in flat_queries]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    t0 = time.time()
    BM25_BATCH_PARTB = 1000
    BM25_K_PARTB = 20000
    bm25_ranks = [None] * len(flat_queries)

    def _slice_tok_p(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(flat_queries), BM25_BATCH_PARTB):
        e = min(s + BM25_BATCH_PARTB, len(flat_queries))
        sub_tokens = _slice_tok_p(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=BM25_K_PARTB, show_progress=False)
        for i in range(e - s):
            gi = s + i
            r = flat_queries[gi]
            tgt_idx = asin_to_idx.get(r["asin"], -1)
            if tgt_idx < 0:
                bm25_ranks[gi] = {"rank": None, "RR": 0.0, "hit10": 0}
                continue
            sorted_docs = sub_res.documents[i]
            positions = np.where(sorted_docs == tgt_idx)[0]
            if len(positions) == 0:
                bm25_ranks[gi] = {"rank": BM25_K_PARTB + 1, "RR": 1.0 / (BM25_K_PARTB + 1), "hit10": 0}
            else:
                rank = int(positions[0]) + 1
                bm25_ranks[gi] = {"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0}
    log(f"  bm25s retrieve done in {time.time() - t0:.1f}s")

    log("\n=== 7. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    doc_embeds = _get_or_build_corpus_embeds(model, corpus_texts)
    log(f"  encoding {len(queries)} queries...")
    t0 = time.time()
    q_embeds = model.encode(queries, batch_size=2048, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
    log(f"  queries encoded in {time.time() - t0:.1f}s")
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    log(f"  matmul done")

    minilm_ranks = []
    for i, r in enumerate(flat_queries):
        tgt_idx = asin_to_idx.get(r["asin"], -1)
        if tgt_idx < 0:
            minilm_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_ranks.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    del doc_embeds_gpu, q_embeds_gpu, sims
    torch.cuda.empty_cache()

    log("\n=== 8. Aggregating volatility metrics ===")
    per_query = []
    for r, bm25_r, minilm_r in zip(flat_queries, bm25_ranks, minilm_ranks):
        per_query.append({**r, "bm25_rank": bm25_r["rank"], "bm25_RR": bm25_r["RR"],
                          "bm25_hit10": bm25_r["hit10"],
                          "minilm_rank": minilm_r["rank"], "minilm_RR": minilm_r["RR"],
                          "minilm_hit10": minilm_r["hit10"]})

    grouped = collections.defaultdict(lambda: {"bm25_RR": [], "minilm_RR": []})
    for r in per_query:
        key = (r["asin"], r["set_type"])
        grouped[key]["bm25_RR"].append(r["bm25_RR"])
        grouped[key]["minilm_RR"].append(r["minilm_RR"])

    def variance_stats(arr):
        a = np.array(arr)
        if len(a) < 2:
            return {"std": None, "iqr": None, "gap": None, "mean": float(a.mean()) if len(a) else None, "n": len(a)}
        return {
            "std": float(a.std()),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "gap": float(a.max() - a.min()),
            "mean": float(a.mean()),
            "n": len(a),
        }

    asin_stats = {}
    for asin, sets in asin_query_sets.items():
        asin_stats[asin] = {}
        for set_type in ("low", "user", "high"):
            data = grouped[(asin, set_type)]
            asin_stats[asin][set_type] = {
                "bm25": variance_stats(data["bm25_RR"]),
                "minilm": variance_stats(data["minilm_RR"]),
            }

    def collect_var(metric_key, retriever, set_type):
        vals = []
        for asin, stats in asin_stats.items():
            v = stats[set_type][retriever][metric_key]
            if v is not None:
                vals.append(v)
        return np.array(vals) if vals else np.array([])

    summary_table = {}
    for retriever in ("bm25", "minilm"):
        summary_table[retriever] = {}
        for metric in ("std", "iqr", "gap", "mean"):
            row = {}
            for set_type in ("low", "user", "high"):
                v = collect_var(metric, retriever, set_type)
                row[set_type] = {
                    "mean": float(v.mean()) if len(v) else None,
                    "std": float(v.std()) if len(v) else None,
                    "n_asins": len(v),
                }
            if metric == "std":
                user_v = collect_var("std", retriever, "user")
                low_v = collect_var("std", retriever, "low")
                high_v = collect_var("std", retriever, "high")
                nvs = []
                for u, l, h in zip(user_v, low_v, high_v):
                    if h - l > 1e-6:
                        nvs.append((u - l) / (h - l))
                row["NV_mean"] = float(np.mean(nvs)) if nvs else None
                row["NV_std"] = float(np.std(nvs)) if nvs else None
            summary_table[retriever][metric] = row

    log(f"\n=== Volatility Summary Table ===")
    log(f"{'Retriever':<10} {'Metric':<8} {'V_low':<12} {'V_user':<12} {'V_high':<12} {'NV':<10}")
    for retriever in ("bm25", "minilm"):
        for metric in ("std", "iqr", "gap"):
            row = summary_table[retriever][metric]
            v_low = row["low"]["mean"]
            v_user = row["user"]["mean"]
            v_high = row["high"]["mean"]
            nv_str = f"{summary_table[retriever]['std'].get('NV_mean', 0):.3f}" if metric == "std" else "-"
            log(f"  {retriever:<10} {metric:<8} {v_low:<12.4f} {v_user:<12.4f} {v_high:<12.4f} {nv_str:<10}")

    log("\n=== 9. Saving summary ===")
    stability_flip = _compute_stability_flip_metrics(RETRIEVAL_PER_QUERY_OUT)
    with open(VOLATILITY_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5.V volatility calibration summary",
                "K_SET": K_SET,
                "MAX_LEN_DIFF": 3,
                "PRIMARY_LEN_DELTA": PRIMARY_LEN_DELTA,
            },
            "summary_table": summary_table,
            "stability_flip": stability_flip,
            "n_asins_with_sets": n_total,
            "n_insufficient_low": n_insufficient_low,
            "n_insufficient_high": n_insufficient_high,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {VOLATILITY_SUMMARY_OUT}")


def _compute_stability_flip_metrics(retrieval_per_query_path) -> dict:
    """Per-ASIN Hit@1 Flip Rate, Hit@5 Flip Rate, RR Std across selected queries.

    Single slice per retriever (BM25 / MiniLM):
      - 'selected_only': 10 user-selected queries per ASIN (realistic persona flip)

    Flip rate definition: of all unique query pairs within the slice, fraction
    of pairs whose top-K hit/miss labels disagree (one hit, one miss).
    """
    log("\n=== Computing Hit@K Flip Rate + RR Std (per ASIN, per retriever, selected_only) ===")
    data = json.load(open(retrieval_per_query_path))
    queries = data["queries"]
    log(f"  loaded {len(queries)} queries from {retrieval_per_query_path.name}")

    # group by asin, keep only selected variant
    by_asin: dict[str, list] = collections.defaultdict(list)
    for q in queries:
        if q.get("variant") == "selected":
            by_asin[q["asin"]].append(q)

    def flip_rate(hit_labels: list[int]) -> float | None:
        """Pair-wise flip rate: |pairs with disagreeing hit/miss| / total pairs."""
        n = len(hit_labels)
        if n < 2:
            return None
        total_pairs = n * (n - 1) // 2
        n_hits = sum(hit_labels)
        n_miss = n - n_hits
        agree_pairs = n_hits * (n_hits - 1) // 2 + n_miss * (n_miss - 1) // 2
        return float((total_pairs - agree_pairs) / total_pairs)

    summary: dict = {"selected_only": {}}
    for retriever in ("bm25", "minilm"):
        rank_key = f"{retriever}_rank"
        rr_key = f"{retriever}_RR"
        flip1_list, flip5_list, flip10_list, rrstd_list = [], [], [], []
        n_asins_used = 0
        for asin, qs in by_asin.items():
            if len(qs) < 2:
                continue
            ranks = [q.get(rank_key) for q in qs]
            rrs = [q.get(rr_key, 0.0) for q in qs]
            if any(r is None for r in ranks):
                continue
            hit1 = [1 if r == 1 else 0 for r in ranks]
            hit5 = [1 if r <= 5 else 0 for r in ranks]
            hit10 = [1 if r <= 10 else 0 for r in ranks]
            f1 = flip_rate(hit1)
            f5 = flip_rate(hit5)
            f10 = flip_rate(hit10)
            rs = float(np.std(rrs))
            if f1 is not None:
                flip1_list.append(f1)
            if f5 is not None:
                flip5_list.append(f5)
            if f10 is not None:
                flip10_list.append(f10)
            rrstd_list.append(rs)
            n_asins_used += 1
        summary["selected_only"][retriever] = {
            "n_asins": n_asins_used,
            "Hit@1_FlipRate_mean": float(np.mean(flip1_list)) if flip1_list else None,
            "Hit@1_FlipRate_median": float(np.median(flip1_list)) if flip1_list else None,
            "Hit@5_FlipRate_mean": float(np.mean(flip5_list)) if flip5_list else None,
            "Hit@5_FlipRate_median": float(np.median(flip5_list)) if flip5_list else None,
            "Hit@10_FlipRate_mean": float(np.mean(flip10_list)) if flip10_list else None,
            "Hit@10_FlipRate_median": float(np.median(flip10_list)) if flip10_list else None,
            "RR_Std_mean": float(np.mean(rrstd_list)) if rrstd_list else None,
            "RR_Std_median": float(np.median(rrstd_list)) if rrstd_list else None,
            "RR_Std_p90": float(np.percentile(rrstd_list, 90)) if rrstd_list else None,
        }
    log(f"  selected_only × BM25: Hit@1={summary['selected_only']['bm25']['Hit@1_FlipRate_mean']:.3f}  "
        f"Hit@5={summary['selected_only']['bm25']['Hit@5_FlipRate_mean']:.3f}  "
        f"Hit@10={summary['selected_only']['bm25']['Hit@10_FlipRate_mean']:.3f}  "
        f"RR_Std={summary['selected_only']['bm25']['RR_Std_mean']:.4f}")
    log(f"  selected_only × MiniLM: Hit@1={summary['selected_only']['minilm']['Hit@1_FlipRate_mean']:.3f}  "
        f"Hit@5={summary['selected_only']['minilm']['Hit@5_FlipRate_mean']:.3f}  "
        f"Hit@10={summary['selected_only']['minilm']['Hit@10_FlipRate_mean']:.3f}  "
        f"RR_Std={summary['selected_only']['minilm']['RR_Std_mean']:.4f}")
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