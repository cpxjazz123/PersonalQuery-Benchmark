"""Stage 10I — ASIN-Level Volatility Explanation.

Stage 10H-B/C observed that personalized syntactic expression induces **bounded**
retrieval volatility, concentrated in a small subset of products (Gini 0.78-0.95).
Stage 10I asks: **why** some ASINs are stable while others flip heavily?

We treat each ASIN's volatility (Hit@1 flip, Hit@5 flip, RR std) as the
dependent variable and test 4 categories of predictors:
  (1) within-ASIN query variation (PCA48 dist, length std, unique count, etc.)
  (2) target-competitor score margin (per-retriever)
  (3) item ambiguity / neighborhood density
  (4) retriever-specific sensitivity drivers

**Core hypothesis**: If the strongest predictors are small score margin and high
neighborhood density (NOT query variation), volatility is a corpus artifact
rather than a property of the personalization method.

Phase plan:
  Phase 1: BM25 raw scores (re-query existing bm25s index, capture scores)
  Phase 2: MiniLM corpus encoding + score capture (217K docs, ~25 min GPU)
  Phase 3: Query variation features per ASIN (PCA48, length, Jaccard)
  Phase 4: Spearman + OLS regression on assembled ASIN-level table

Inputs:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_per_query.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_volatility.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7_corpus_docs.jsonl.gz
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_pool.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz

Outputs:
  hj82_scratch2/.../stage10i_bm25_scores.jsonl.gz
  hj82_scratch2/.../stage10i_corpus_embeds.npy
  hj82_scratch2/.../stage10i_minilm_scores.jsonl.gz
  hj82_scratch2/.../stage10i_features.json
  hj82_scratch2/.../stage10i_analysis.json

Run:
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
      gaussian/syntax_subspace_stage10i_explain.py \\
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10i_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
PER_QUERY_IN = SCRATCH / "stage10h_b_per_query.json"
VOLATILITY_IN = SCRATCH / "stage10h_b_volatility.json"
POOL_IN = SCRATCH / "stage8_5_pool.json"
CORPUS_IN = SCRATCH / "stage7_corpus_docs.jsonl.gz"
FEAT_CACHE_IN = SCRATCH / "stage7b_query_features.jsonl.gz"

OUTPUT_BM25_SCORES = SCRATCH / "stage10i_bm25_scores.jsonl.gz"
OUTPUT_CORPUS_EMBEDS = SCRATCH / "stage10i_corpus_embeds.npy"
OUTPUT_MINILM_SCORES = SCRATCH / "stage10i_minilm_scores.jsonl.gz"
OUTPUT_FEATURES = SCRATCH / "stage10i_features.json"
OUTPUT_ANALYSIS = SCRATCH / "stage10i_analysis.json"

PCA_DIM = 48
PCA_SEED = 2024
SEED = 2024

# Number of nearest competitors to consider per query for score margin
N_COMPETITORS = 5


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def project_query_hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 10I — ASIN-Level Volatility Explanation ===")

    # === 0. Load scaler + PCA48 (same as Stage 10H-B) ===
    log("\n=== 0. Loading scaler + PCA48 ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    from sklearn.decomposition import PCA

    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    def project_query(text: str):
        k = project_query_hash(text)
        feats = feat_map.get(k)
        if not feats:
            return None
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        if not numeric or len(numeric) < len(feature_names) * 0.5:
            return None
        vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
        return pca.transform(scaler.transform(vec[None, :]))[0]

    # === 1. Load feature cache ===
    log("\n=== 1. Loading feature cache ===")
    feat_map = {}
    with gzip.open(FEAT_CACHE_IN, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feature cache: {len(feat_map)} entries")

    # === 2. Load per-query results ===
    log("\n=== 2. Loading Stage 10H-B per-query results ===")
    per_query = json.load(open(PER_QUERY_IN))
    log(f"  per-query entries: {len(per_query)}")
    queries = [r["query"] for r in per_query]
    target_asins = [r["asin"] for r in per_query]

    # === 3. Load corpus ===
    log("\n=== 3. Loading corpus ===")
    asin_to_doc = {}
    with gzip.open(CORPUS_IN, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin")
            doc = rec.get("doc", "") or rec.get("text", "")
            if asin and doc:
                asin_to_doc[asin] = doc[:1000]
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    target_indices = np.array([asin_to_idx.get(a, -1) for a in target_asins])
    n_missing = int((target_indices < 0).sum())
    log(f"  queries: {len(queries)}, missing target: {n_missing}")

    corpus_texts = [asin_to_doc[a] for a in asins]

    # === 4. Phase 1: BM25 raw scores ===
    log("\n=== Phase 1: BM25 raw scores ===")
    import bm25s
    t0 = time.time()
    log("  tokenizing queries (cached tokens not available)...")
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    log(f"  BM25 tokenize: {time.time() - t0:.1f}s")

    log("  building BM25 index (rebuild from scratch, ~2 min)...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  BM25 index built: {time.time() - t0:.1f}s")

    log("  retrieving all queries with full ranked lists + scores...")
    t0 = time.time()
    bm25_results_raw = retriever.retrieve(query_tokens, k=N_COMPETITORS + 1, show_progress=False)
    log(f"  BM25 retrieve: {time.time() - t0:.1f}s")

    # Extract per-query: target score, best competitor score, top-N asins
    log("  extracting target/competitor scores...")
    bm25_score_records = []
    bm25_top5_competitors = {}  # asin -> list of top-5 competitor asins
    for i, r in enumerate(per_query):
        tgt_idx = target_indices[i]
        scores_arr = bm25_results_raw.scores[i]
        idxs_arr = bm25_results_raw.documents[i]
        if tgt_idx < 0:
            bm25_score_records.append({
                "asin": r["asin"], "user_id": r["user_id"],
                "bm25_score_target": None, "bm25_score_margin": None,
                "bm25_rank_target": None, "bm25_top5_competitors": [],
            })
            continue
        # Find target in top-N retrieved
        tgt_in_top = (idxs_arr == tgt_idx)
        if tgt_in_top.any():
            tgt_score = float(scores_arr[tgt_in_top][0])
            tgt_rank_in_top = int(np.where(tgt_in_top)[0][0]) + 1
            competitors = [
                {"asin": asins[int(idxs_arr[j])], "score": float(scores_arr[j])}
                for j in range(len(idxs_arr)) if int(idxs_arr[j]) != tgt_idx
            ]
        else:
            # Target not in top-N (N_COMPETITORS+1), full corpus score needed
            # For score margin, use a fallback: rank from stage10h_b + score from top-N first
            tgt_score = None
            tgt_rank_in_top = None
            competitors = [
                {"asin": asins[int(idxs_arr[j])], "score": float(scores_arr[j])}
                for j in range(len(idxs_arr))
            ]
        best_competitor_score = competitors[0]["score"] if competitors else None
        score_margin = (tgt_score - best_competitor_score) if tgt_score is not None and best_competitor_score is not None else None
        bm25_score_records.append({
            "asin": r["asin"], "user_id": r["user_id"],
            "bm25_score_target": tgt_score,
            "bm25_score_best_competitor": best_competitor_score,
            "bm25_score_margin": score_margin,
            "bm25_rank_target": tgt_rank_in_top,
            "bm25_top5_competitors": [c["asin"] for c in competitors[:5]],
        })

    with gzip.open(OUTPUT_BM25_SCORES, "wt", encoding="utf-8") as f:
        for rec in bm25_score_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  wrote → {OUTPUT_BM25_SCORES} ({len(bm25_score_records)} records)")

    del retriever, corpus_tokens, query_tokens, bm25_results_raw
    import gc; gc.collect()

    # === 5. Phase 2: MiniLM corpus encoding + scores ===
    log("\n=== Phase 2: MiniLM corpus encoding + scores ===")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    t0 = time.time()
    log(f"  encoding {len(corpus_texts)} corpus docs...")
    doc_embeds = model.encode(
        corpus_texts, batch_size=512, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    log(f"  corpus encoded: {time.time() - t0:.1f}s, shape={doc_embeds.shape}")
    np.save(OUTPUT_CORPUS_EMBEDS, doc_embeds)
    log(f"  saved → {OUTPUT_CORPUS_EMBEDS}")

    t0 = time.time()
    q_embeds = model.encode(
        queries, batch_size=512, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    log(f"  queries encoded: {time.time() - t0:.1f}s")

    t0 = time.time()
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    sims = q_embeds_gpu @ doc_embeds_gpu.T  # [Q, D]
    log(f"  similarity computed: {time.time() - t0:.1f}s, shape={tuple(sims.shape)}")

    minilm_score_records = []
    for i, r in enumerate(per_query):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            minilm_score_records.append({
                "asin": r["asin"], "user_id": r["user_id"],
                "minilm_score_target": None, "minilm_score_margin": None,
                "minilm_rank_target": None, "minilm_top5_competitors": [],
            })
            continue
        sc = sims[i]
        tgt_score = float(sc[tgt_idx].item())
        # Mask target, find top-5 competitor scores
        sc_masked = sc.clone()
        sc_masked[tgt_idx] = -1e9
        top5_vals, top5_idxs = torch.topk(sc_masked, k=5)
        competitors = [
            {"asin": asins[int(top5_idxs[j])], "score": float(top5_vals[j].item())}
            for j in range(5)
        ]
        best_competitor_score = competitors[0]["score"]
        score_margin = tgt_score - best_competitor_score
        # Rank: how many docs have higher score
        rank_target = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_score_records.append({
            "asin": r["asin"], "user_id": r["user_id"],
            "minilm_score_target": tgt_score,
            "minilm_score_best_competitor": best_competitor_score,
            "minilm_score_margin": score_margin,
            "minilm_rank_target": rank_target,
            "minilm_top5_competitors": [c["asin"] for c in competitors],
        })

    with gzip.open(OUTPUT_MINILM_SCORES, "wt", encoding="utf-8") as f:
        for rec in minilm_score_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  wrote → {OUTPUT_MINILM_SCORES} ({len(minilm_score_records)} records)")

    del doc_embeds_gpu, q_embeds_gpu, sims, doc_embeds, q_embeds, model
    torch.cuda.empty_cache()
    gc.collect()

    # === 6. Phase 3: Query variation features per ASIN ===
    log("\n=== Phase 3: Query variation features per ASIN ===")
    asin_to_queries = defaultdict(list)
    asin_to_query_z = defaultdict(list)
    for r in per_query:
        asin_to_queries[r["asin"]].append(r["query"])
        z = project_query(r["query"])
        if z is not None:
            asin_to_query_z[r["asin"]].append(z)

    query_variation = {}
    for asin, qs in asin_to_queries.items():
        n_q = len(qs)
        unique_count = len(set(qs))
        lengths = np.array([len(q.split()) for q in qs])
        length_std = float(lengths.std(ddof=1)) if n_q > 1 else 0.0
        length_mean = float(lengths.mean())

        # Token Jaccard
        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa and not sb:
                return 0.0
            return len(sa & sb) / max(1, len(sa | sb))
        pairs = list(combinations(range(n_q), 2))
        jaccard_vals = [jaccard(qs[i], qs[j]) for i, j in pairs]
        token_jaccard_mean = float(np.mean(jaccard_vals)) if jaccard_vals else 0.0

        # PCA48 syntactic distance (projected queries)
        zs = asin_to_query_z.get(asin, [])
        if len(zs) >= 2:
            z_pairs = list(combinations(range(len(zs)), 2))
            syntax_dists = [float(np.linalg.norm(zs[i] - zs[j])) for i, j in z_pairs]
            syntax_dist_mean = float(np.mean(syntax_dists))
            syntax_dist_max = float(np.max(syntax_dists))
        else:
            syntax_dist_mean = 0.0
            syntax_dist_max = 0.0

        # Structural family diversity: simple heuristic
        # (number of distinct opening POS patterns)
        def first_token_pos(q):
            t = q.strip().split()
            return t[0].lower() if t else ""
        first_tokens = set(first_token_pos(q) for q in qs)
        family_diversity = len(first_tokens)

        query_variation[asin] = {
            "n_queries": n_q,
            "unique_count": unique_count,
            "length_mean": length_mean,
            "length_std": length_std,
            "token_jaccard_mean": token_jaccard_mean,
            "syntax_dist_mean": syntax_dist_mean,
            "syntax_dist_max": syntax_dist_max,
            "family_diversity": family_diversity,
        }
    log(f"  query variation features for {len(query_variation)} ASINs")

    # === 7. Aggregate BM25 / MiniLM score margin per ASIN ===
    log("\n=== 7. Aggregating per-ASIN score margins ===")
    bm25_by_asin = defaultdict(list)
    minilm_by_asin = defaultdict(list)
    bm25_top5_overlap_per_asin = defaultdict(list)  # for neighborhood density
    for rec in bm25_score_records:
        if rec["bm25_score_margin"] is not None:
            bm25_by_asin[rec["asin"]].append(rec["bm25_score_margin"])
        if rec["bm25_top5_competitors"]:
            bm25_top5_overlap_per_asin[rec["asin"]].append(rec["bm25_top5_competitors"])
    for rec in minilm_score_records:
        if rec["minilm_score_margin"] is not None:
            minilm_by_asin[rec["asin"]].append(rec["minilm_score_margin"])

    def agg(vals):
        arr = np.array(vals, dtype=np.float64)
        if len(arr) == 0:
            return {"mean": None, "min": None, "std": None, "median": None}
        return {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
        }

    score_features = {}
    for asin in query_variation.keys():
        bm25_margin = bm25_by_asin.get(asin, [])
        minilm_margin = minilm_by_asin.get(asin, [])
        # Neighborhood density: how often does the same top-1 competitor appear across queries?
        top1_lists = bm25_top5_overlap_per_asin.get(asin, [])
        if top1_lists:
            top1_first = [lst[0] if lst else None for lst in top1_lists]
            top1_first = [x for x in top1_first if x is not None]
            top1_most_common_frac = (
                max([top1_first.count(x) for x in set(top1_first)]) / len(top1_first)
                if top1_first else 0.0
            )
            # Pairwise top-5 overlap (Jaccard) for diversity
            top5_jaccards = []
            for i, j in combinations(range(len(top1_lists)), 2):
                s1, s2 = set(top1_lists[i]), set(top1_lists[j])
                if s1 or s2:
                    top5_jaccards.append(len(s1 & s2) / max(1, len(s1 | s2)))
            top5_overlap_mean = float(np.mean(top5_jaccards)) if top5_jaccards else 0.0
        else:
            top1_most_common_frac = 0.0
            top5_overlap_mean = 0.0
        score_features[asin] = {
            "bm25_margin": agg(bm25_margin),
            "minilm_margin": agg(minilm_margin),
            "bm25_top1_common_frac": top1_most_common_frac,
            "bm25_top5_overlap_mean": top5_overlap_mean,
        }
    log(f"  score features for {len(score_features)} ASINs")

    # === 8. Merge with volatility metrics + assemble final features table ===
    log("\n=== 8. Assembling final ASIN-level features ===")
    vol_data = json.load(open(VOLATILITY_IN))
    per_asin_vol = vol_data["per_asin"]

    final_features = {}
    for asin in query_variation.keys():
        vol = per_asin_vol.get(asin, {})
        qv = query_variation.get(asin, {})
        sf = score_features.get(asin, {})
        final_features[asin] = {
            # Volatility (Y)
            "bm25_hit1_flip": vol.get("bm25_hit1_flip_rate"),
            "bm25_hit5_flip": vol.get("bm25_hit5_flip_rate"),
            "bm25_rr_std": vol.get("bm25_rr_std"),
            "minilm_hit1_flip": vol.get("minilm_hit1_flip_rate"),
            "minilm_hit5_flip": vol.get("minilm_hit5_flip_rate"),
            "minilm_rr_std": vol.get("minilm_rr_std"),
            "n_users": vol.get("n_users"),
            # Query variation (predictor class 1)
            **qv,
            # Score margin (predictor class 2)
            "bm25_margin_mean": sf["bm25_margin"]["mean"],
            "bm25_margin_min": sf["bm25_margin"]["min"],
            "bm25_margin_std": sf["bm25_margin"]["std"],
            "bm25_margin_median": sf["bm25_margin"]["median"],
            "minilm_margin_mean": sf["minilm_margin"]["mean"],
            "minilm_margin_min": sf["minilm_margin"]["min"],
            "minilm_margin_std": sf["minilm_margin"]["std"],
            "minilm_margin_median": sf["minilm_margin"]["median"],
            # Neighborhood density (predictor class 3)
            "bm25_top1_common_frac": sf["bm25_top1_common_frac"],
            "bm25_top5_overlap_mean": sf["bm25_top5_overlap_mean"],
        }
    log(f"  final features: {len(final_features)} ASINs")

    with open(OUTPUT_FEATURES, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10I: ASIN-level explanation features",
                "n_asins": len(final_features),
                "predictor_classes": [
                    "query_variation",
                    "score_margin_bm25",
                    "score_margin_minilm",
                    "neighborhood_density",
                ],
            },
            "features": final_features,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_FEATURES}")

    # === 9. Phase 4: Statistics + regression ===
    log("\n=== Phase 4: Spearman correlations + OLS regression ===")
    from scipy.stats import spearmanr

    # Convert to numpy arrays
    rows = []
    asin_list = sorted(final_features.keys())
    for asin in asin_list:
        rows.append(final_features[asin])
    cols = list(rows[0].keys())

    def col(col_name):
        return np.array([
            r[col_name] for r in rows
            if r[col_name] is not None
        ], dtype=np.float64)

    Y_COLS = ["bm25_rr_std", "bm25_hit1_flip", "bm25_hit5_flip",
              "minilm_rr_std", "minilm_hit1_flip", "minilm_hit5_flip"]
    X_COLS = [
        # Class 1: query variation
        "syntax_dist_mean", "syntax_dist_max", "length_std", "length_mean",
        "unique_count", "family_diversity", "token_jaccard_mean",
        # Class 2: score margin (BM25 + MiniLM)
        "bm25_margin_mean", "bm25_margin_min", "bm25_margin_median",
        "minilm_margin_mean", "minilm_margin_min", "minilm_margin_median",
        # Class 3: neighborhood density
        "bm25_top1_common_frac", "bm25_top5_overlap_mean",
    ]

    # Spearman correlations
    spearman_results = {}
    for y_col in Y_COLS:
        spearman_results[y_col] = {}
        for x_col in X_COLS:
            # Match by ASIN (drop pairs where either is None)
            pairs = [(r[x_col], r[y_col]) for r in rows
                     if r[x_col] is not None and r[y_col] is not None]
            if len(pairs) < 5:
                spearman_results[y_col][x_col] = {"rho": None, "p": None, "n": len(pairs)}
                continue
            x_vals = np.array([p[0] for p in pairs])
            y_vals = np.array([p[1] for p in pairs])
            rho, p = spearmanr(x_vals, y_vals)
            spearman_results[y_col][x_col] = {
                "rho": float(rho), "p": float(p), "n": len(pairs),
            }
    log("  Spearman computed.")

    # Print Spearman table for review
    log("\n  Spearman ρ (volatility ~ predictor):")
    log(f"    {'predictor':<32} " + " ".join(f"{y[:8]:>9}" for y in Y_COLS))
    for x_col in X_COLS:
        cells = []
        for y_col in Y_COLS:
            r = spearman_results[y_col][x_col]
            if r["rho"] is None:
                cells.append(f"{'N/A':>9}")
            else:
                sig = "*" if r["p"] < 0.05 else " "
                cells.append(f"{r['rho']:+.3f}{sig}")
        log(f"    {x_col:<32} " + " ".join(f"{c:>9}" for c in cells))

    # OLS regression: volatility ~ score margin + neighborhood density + query variation
    from sklearn.linear_model import LinearRegression

    def fit_ols(y_col, x_cols):
        pairs = [(r[y_col], [r[c] for c in x_cols]) for r in rows
                 if r[y_col] is not None and all(r[c] is not None for c in x_cols)]
        if len(pairs) < 5:
            return None
        y = np.array([p[0] for p in pairs])
        X = np.array([p[1] for p in pairs])
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / max(1e-9, ss_tot)
        coefs = {x_cols[i]: float(model.coef_[i]) for i in range(len(x_cols))}
        intercept = float(model.intercept_)
        return {
            "n": len(pairs),
            "r2": r2,
            "intercept": intercept,
            "coefs": coefs,
            "predictors": x_cols,
        }

    bm25_reg = fit_ols("bm25_rr_std", [
        "bm25_margin_mean", "bm25_top1_common_frac",
        "syntax_dist_mean", "unique_count", "length_std"
    ])
    minilm_reg = fit_ols("minilm_rr_std", [
        "minilm_margin_mean", "bm25_top1_common_frac",
        "syntax_dist_mean", "unique_count", "length_std"
    ])
    bm25_hit1_reg = fit_ols("bm25_hit1_flip", [
        "bm25_margin_mean", "bm25_top1_common_frac",
        "syntax_dist_mean", "unique_count", "length_std"
    ])
    minilm_hit1_reg = fit_ols("minilm_hit1_flip", [
        "minilm_margin_mean", "bm25_top1_common_frac",
        "syntax_dist_mean", "unique_count", "length_std"
    ])

    if bm25_reg:
        log(f"\n  BM25 rr_std ~ margin + density + query variation:")
        log(f"    R² = {bm25_reg['r2']:.3f}, n = {bm25_reg['n']}")
        for c, v in bm25_reg["coefs"].items():
            log(f"    {c}: {v:+.4f}")
    if minilm_reg:
        log(f"\n  MiniLM rr_std ~ margin + density + query variation:")
        log(f"    R² = {minilm_reg['r2']:.3f}, n = {minilm_reg['n']}")
        for c, v in minilm_reg["coefs"].items():
            log(f"    {c}: {v:+.4f}")

    # === 10. Save analysis ===
    log("\n=== 10. Saving analysis ===")
    analysis = {
        "config": {
            "description": "Stage 10I: ASIN-level explanation analysis",
            "n_asins": len(rows),
            "y_cols": Y_COLS,
            "x_cols": X_COLS,
            "N_COMPETITORS": N_COMPETITORS,
            "SEED": SEED,
        },
        "spearman": spearman_results,
        "regression": {
            "bm25_rr_std": bm25_reg,
            "minilm_rr_std": minilm_reg,
            "bm25_hit1_flip": bm25_hit1_reg,
            "minilm_hit1_flip": minilm_hit1_reg,
        },
    }
    with open(OUTPUT_ANALYSIS, "w") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_ANALYSIS}")

    log("\n=== Stage 10I complete ===")


if __name__ == "__main__":
    main()