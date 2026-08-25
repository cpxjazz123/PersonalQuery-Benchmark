"""Stage 8.5.V: Empirical Volatility Calibration.

For each ASIN, calibrate user-style volatility against minimal-syntax and
maximal-syntax bounds, all from the same shared candidate pool.

For each (asin, attrs):
- V_low:  K=8 queries with smallest pairwise PCA48 distance (most similar syntax)
- V_user: 1 user-selected query (from stage8_5_selection.json)
- V_high: K=8 queries with largest pairwise PCA48 distance (most spread syntax)
- Length-matching: |Δlength| ≤ MAX_LEN_DIFF tokens

Run BM25 + MiniLM on each set; compute Std/IQR/best-worst gap of MRR.
NV = (V_user - V_low) / (V_high - V_low)

Output: stage8_5v_volatility.json + summary table

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5v_volatility.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5v_volatility.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
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
POOL_IN = SCRATCH / "stage8_5_pool.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
PER_QUERY_OUT = SCRATCH / "stage8_5v_retrieval.json"
SUMMARY_OUT = SCRATCH / "stage8_5v_volatility.json"

# === Constants ===
K_SET = 8                  # queries per (V_low / V_high / V_user) set
MAX_LEN_DIFF = 3           # |Δlength| within V_low / V_high pairs
MIN_LEN = 5                # min tokens per query (filter very short)
PRIMARY_LEN_DELTA = 2      # primary length band: ±2 tokens around anchor
LEN_BAND_FALLBACK = 5      # if not enough in primary, expand to ±5
USE_BOTH_USERS = True      # sample 2 users per ASIN for V_user (for paired test)
N_ASIN_SAMPLE = 100        # all 100 ASINs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 8.5.V: Empirical Volatility Calibration ===")

    # === 1. Load inputs ===
    log("\n=== 1. Loading inputs ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=48, random_state=42)
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

    # === 2. Project pool queries to PCA48 ===
    log("\n=== 2. Projecting pool queries ===")
    pool_z = {}  # asin → list of (z, query, n_tok, strict)
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

    # === 3. For each ASIN, build three query sets ===
    log("\n=== 3. Building V_low / V_user / V_high query sets ===")
    asin_query_sets = {}  # asin → {user_id: {"low": [qs], "user": q, "high": [qs]}}
    n_insufficient_low = 0
    n_insufficient_high = 0
    n_total = 0
    for entry in entries:
        asin = entry["asin"]
        if asin not in pool_z:
            continue
        # Get strict candidates only
        cands = [c for c in pool_z[asin] if c["strict"] and c["n_tok"] >= MIN_LEN]
        if len(cands) < K_SET * 2:
            continue
        # Project all to numpy for fast distance computation
        zs = np.array([c["z"] for c in cands])  # (n_cands, 48)
        # Pairwise distance
        n = len(cands)
        dist = np.zeros((n, n))
        for i in range(n):
            diff = zs - zs[i]
            dist[i] = np.linalg.norm(diff, axis=1)
        # For each candidate, mean distance to all other candidates
        mean_dist = (dist.sum(axis=1) - np.diag(dist)) / (n - 1)
        sorted_idx = np.argsort(mean_dist)

        # Length match helper
        def length_match(anchor_idx, max_diff):
            ok = []
            a_tok = cands[anchor_idx]["n_tok"]
            for j in range(n):
                if abs(cands[j]["n_tok"] - a_tok) <= max_diff:
                    ok.append(j)
            return ok

        # === V_low: K queries closest to smallest-mean-distance candidate, length-matched ===
        anchor_lo = sorted_idx[0]
        ok_idx_lo = []
        for max_diff in [PRIMARY_LEN_DELTA, LEN_BAND_FALLBACK, 10]:
            ok_idx_lo = length_match(anchor_lo, max_diff)
            if len(ok_idx_lo) >= K_SET:
                break
        if len(ok_idx_lo) < K_SET:
            n_insufficient_low += 1
            continue
        # Closest K to anchor_lo within length-matched
        ok_dist_lo = dist[anchor_lo][ok_idx_lo]
        sorted_ok_lo = np.array(ok_idx_lo)[np.argsort(ok_dist_lo)][:K_SET]
        low_qs = [cands[i]["q"] for i in sorted_ok_lo]

        # === V_high: K queries farthest from largest-mean-distance candidate, length-matched ===
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

        # V_user: user-selected queries (from selection.json)
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

    # === 4. Build retrieval corpus ===
    log("\n=== 4. Building ASIN metadata corpus ===")
    META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"
    asin_to_doc = {}
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
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    # === 5. Build flat query list (for batched retrieval) ===
    log("\n=== 5. Building flat query list ===")
    # Each (asin, set_type, user_id?) is one observation set
    flat_queries = []  # one entry per query to retrieve
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

    # === 6. BM25 via bm25s ===
    log("\n=== 6. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    queries = [r["query"] for r in flat_queries]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    t0 = time.time()
    results = retriever.retrieve(query_tokens, k=len(asins), show_progress=False)
    log(f"  bm25s retrieve done in {time.time() - t0:.1f}s")

    # Compute rank per query
    bm25_ranks = []
    for i, r in enumerate(flat_queries):
        tgt_idx = asin_to_idx.get(r["asin"], -1)
        if tgt_idx < 0:
            bm25_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sorted_docs = results.documents[i]
        positions = np.where(sorted_docs == tgt_idx)[0]
        if len(positions) == 0:
            bm25_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
        else:
            rank = int(positions[0]) + 1
            bm25_ranks.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    # === 7. MiniLM GPU ===
    log("\n=== 7. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    log(f"  encoding {len(corpus_texts)} corpus docs...")
    t0 = time.time()
    doc_embeds = model.encode(corpus_texts, batch_size=512, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
    log(f"  docs encoded in {time.time() - t0:.1f}s")
    log(f"  encoding {len(queries)} queries...")
    t0 = time.time()
    q_embeds = model.encode(queries, batch_size=512, show_progress_bar=False,
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

    # === 8. Aggregate per set ===
    log("\n=== 8. Aggregating volatility metrics ===")
    per_query = []
    for r, bm25_r, minilm_r in zip(flat_queries, bm25_ranks, minilm_ranks):
        per_query.append({**r, "bm25_rank": bm25_r["rank"], "bm25_RR": bm25_r["RR"],
                          "bm25_hit10": bm25_r["hit10"],
                          "minilm_rank": minilm_r["rank"], "minilm_RR": minilm_r["RR"],
                          "minilm_hit10": minilm_r["hit10"]})
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({"config": {"description": "Stage 8.5.V volatility calibration"},
                   "queries": per_query}, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # Per (asin, set_type) → list of RR per retriever
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

    # Per-ASIN stats: low, user, high → bm25, minilm
    asin_stats = {}
    for asin, sets in asin_query_sets.items():
        asin_stats[asin] = {}
        for set_type in ("low", "user", "high"):
            data = grouped[(asin, set_type)]
            asin_stats[asin][set_type] = {
                "bm25": variance_stats(data["bm25_RR"]),
                "minilm": variance_stats(data["minilm_RR"]),
            }

    # Aggregate across ASINs: mean of V_low, V_user, V_high
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
            # NV = (V_user - V_low) / (V_high - V_low) per ASIN, then mean
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

    # === 9. Save summary ===
    log("\n=== 9. Saving summary ===")
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5.V volatility calibration summary",
                "K_SET": K_SET,
                "MAX_LEN_DIFF": MAX_LEN_DIFF,
                "PRIMARY_LEN_DELTA": PRIMARY_LEN_DELTA,
            },
            "summary_table": summary_table,
            "n_asins_with_sets": n_total,
            "n_insufficient_low": n_insufficient_low,
            "n_insufficient_high": n_insufficient_high,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()