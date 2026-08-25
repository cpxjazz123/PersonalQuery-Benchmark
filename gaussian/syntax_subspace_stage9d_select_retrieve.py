"""Stage 9D — Mahalanobis rerank on Stage 9C clean pool + retrieval comparison.

Goal: verify Stage 9C's clean + compositionally-rich pool improves retrieval robustness
vs Stage 8.5 baseline.

Pipeline:
1. Load Stage 9C clean pool (786 queries across 10 ASINs)
2. Load Stage 8.5 user Gaussians (per-user PCA48 Mahalanobis)
3. Extend feature cache with 9C query features (cached in stage7b_query_features.jsonl.gz)
4. Project 9C queries to PCA48
5. For each (asin, user) in 10 ASINs:
   - selected: query with min Mahalanobis distance to user Gaussian
   - random: deterministic random pick per user
   - farthest: query with max distance (negative control)
6. Retrieve each via bm25s + MiniLM GPU
7. Compare:
   - 9C selected vs 9C random vs 9C farthest (within 9C pool)
   - 9C selected vs Stage 8.5 selected (cross-stage, if 8.5 had same users)

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_selection.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_selection_stats.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_retrieval_per_query.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_retrieval_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9d_select_retrieve.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats import wilcoxon


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
POOL_9C_IN = SCRATCH / "stage9c_pool_pilot.json"
GAUSSIANS_IN = SCRATCH / "stage8_5_user_gaussians.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"

SELECTION_OUT = SCRATCH / "stage9d_selection.json"
SELECTION_STATS_OUT = SCRATCH / "stage9d_selection_stats.json"
RETRIEVAL_PER_Q_OUT = SCRATCH / "stage9d_retrieval_per_query.json"
RETRIEVAL_SUMMARY_OUT = SCRATCH / "stage9d_retrieval_summary.json"

META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"

# Force GPU for retrieval
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 2024


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    """Diagonal Mahalanobis squared: d² = Σ (z - mu)² / sigma."""
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def load_or_extract_features(queries: List[str]) -> dict:
    """Load features from cache; extract missing via spaCy pipe."""
    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache loaded: {len(feat_map)} entries")

    missing = [q for q in queries if feat_key(q) not in feat_map]
    if not missing:
        log(f"  no missing features")
        return feat_map

    log(f"  extracting features for {len(missing)} missing queries via spaCy...")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from main import per_sentence_features_v2
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    t0 = time.time()
    docs = list(nlp.pipe(missing, batch_size=128, n_process=4))
    log(f"  parsed in {time.time() - t0:.1f}s")

    t0 = time.time()
    new_records = []
    for q, doc in zip(missing, docs):
        feats = per_sentence_features_v2(doc)
        new_records.append({"k": feat_key(q), "v": feats})
    log(f"  features extracted in {time.time() - t0:.1f}s")

    # Append to cache
    with gzip.open(FEAT_CACHE, "at", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    feat_map.update({r["k"]: r["v"] for r in new_records})
    log(f"  cache updated: {len(feat_map)} entries")
    return feat_map


def build_meta_corpus():
    """Load ASIN-level metadata (217K ASINs)."""
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
    return asin_to_doc


def bm25s_search(corpus_texts: List[str], queries: List[str]):
    """Use bm25s for fast BM25 scoring. Returns Results with full sorted docs."""
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
    log(f"  retrieving (k=all)...")
    t0 = time.time()
    results = retriever.retrieve(query_tokens, k=len(corpus_texts), show_progress=False)
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return results


def minilm_encode_gpu(corpus_texts: List[str], queries_text: List[str]):
    """Encode corpus + queries via MiniLM on GPU."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  MiniLM device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    log(f"  encoding {len(corpus_texts)} corpus docs (GPU, batch=512)...")
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
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    return doc_embeds_gpu, q_embeds_gpu


def main():
    log("=== Stage 9D — Mahalanobis rerank on 9C clean pool + retrieval ===")

    # === 1. Load PCA48 setup ===
    log("\n=== 1. Loading PCA48 ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    # === 2. Load 9C pool + Stage 8.5 user Gaussians ===
    log("\n=== 2. Loading inputs ===")
    pool_9C_data = json.load(open(POOL_9C_IN))
    pools_9C = pool_9C_data["pools"]
    log(f"  9C pool: {sum(len(v) for v in pools_9C.values())} queries across {len(pools_9C)} ASINs")

    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    global_var = np.array(gauss_data["global_var"])
    log(f"  user Gaussians: {len(users_gauss)}, global_var shape: {global_var.shape}")

    # Load 10-ASIN subset from Stage 8.5
    asin_data_full = json.load(open(ASINS_IN))["asins"]
    asin_data = [a for a in asin_data_full if a["asin"] in pools_9C]
    log(f"  ASINs in 9C ∩ 8.5: {len(asin_data)}")
    log(f"  total (asin, user) pairs: {sum(len(a['users_sampled']) for a in asin_data)}")

    # === 3. Extract features for 9C queries ===
    log("\n=== 3. Feature extraction for 9C queries ===")
    all_9C_queries = []
    for asin, qs in pools_9C.items():
        for q in qs:
            all_9C_queries.append(q["query"])
    feat_map = load_or_extract_features(all_9C_queries)
    log(f"  total features: {len(feat_map)}")

    # === 4. Project 9C queries to PCA48 ===
    log("\n=== 4. Projecting 9C queries to PCA48 ===")
    pool_z = {}
    miss_feat = 0
    for asin, qs in pools_9C.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss_feat += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss_feat}")

    # ASIN centroid fallback
    asin_centroid = {}
    for asin, zqs in pool_z.items():
        zs = np.stack([z for z, _ in zqs], axis=0)
        asin_centroid[asin] = zs.mean(axis=0)
    log(f"  ASIN centroids: {len(asin_centroid)}")

    # === 5. Mahalanobis selection per (asin, user) ===
    log("\n=== 5. Selection per (asin, user) ===")
    rng = random.Random(SEED)

    selection_entries = []
    n_mahal = 0
    n_asin_fallback = 0
    n_random_fallback = 0

    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        zqs = pool_z.get(asin, [])
        n_pool_strict = sum(1 for z, q in zqs if q.get("strict", True))
        log(f"  {asin}: pool_z={len(zqs)}, strict={n_pool_strict}")

        c_asin = asin_centroid.get(asin)
        if c_asin is None:
            log(f"  WARNING: no centroid for {asin}, skipping")
            continue

        for uid in entry["users_sampled"]:
            if uid in users_gauss:
                mu = np.array(users_gauss[uid]["mu"])
                sigma = np.array(users_gauss[uid]["sigma_diag"])
                source = users_gauss[uid]["source"]
                n_reviews = users_gauss[uid]["n_reviews"]
            else:
                mu = c_asin
                sigma = np.maximum(global_var, 1e-3)
                source = "asin_centroid_fallback"
                n_reviews = 0

            strict_zqs = [(z, q) for z, q in zqs if q.get("strict", True)]
            if not strict_zqs:
                strict_zqs = zqs
            if not strict_zqs:
                selection_entries.append({
                    "asin": asin,
                    "user_id": uid,
                    "attrs_used": attrs,
                    "selection_method": "no_pool",
                    "selected": None,
                    "random": None,
                    "farthest": None,
                    "selected_distance": None,
                    "random_distance": None,
                    "farthest_distance": None,
                    "n_candidates": 0,
                    "user_source": source,
                    "n_reviews": n_reviews,
                    "pool_size": len(zqs),
                })
                continue

            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            best_idx = int(np.argmin(distances))
            worst_idx = int(np.argmax(distances))

            selected_q = strict_zqs[best_idx][1]
            farthest_q = strict_zqs[worst_idx][1]

            rng_u = random.Random(hash(uid) & 0xffffffff)
            random_idx = rng_u.randint(0, len(strict_zqs) - 1)
            random_q = strict_zqs[random_idx][1]

            method = "mahal_min" if source != "asin_centroid_fallback" else "asin_centroid_fallback"
            if source in ("per_user", "global_var_fallback"):
                n_mahal += 1
            else:
                n_asin_fallback += 1

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "random": random_q,
                "farthest": farthest_q,
                "selected_distance": float(distances[best_idx]),
                "random_distance": float(distances[random_idx]),
                "farthest_distance": float(distances[worst_idx]),
                "n_candidates": len(strict_zqs),
                "user_source": source,
                "n_reviews": n_reviews,
                "pool_size": len(zqs),
            })

    log(f"  total entries: {len(selection_entries)}")
    log(f"    mahal_min: {n_mahal}, asin_centroid_fallback: {n_asin_fallback}")

    # === 6. Save selection ===
    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9D: Mahalanobis selection from Stage 9C clean pool",
                "pool_source": "stage9c_pool_pilot.json",
                "n_9c_queries": sum(len(v) for v in pools_9C.values()),
                "n_asins_9c": len(pools_9C),
                "SEED": SEED,
            },
            "n_entries": len(selection_entries),
            "n_mahal": n_mahal,
            "n_asin_fallback": n_asin_fallback,
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SELECTION_OUT}")

    # === 7. Validation stats: selected vs random vs farthest ===
    log("\n=== 7. Validation: selected vs random vs farthest self-distance ===")
    mahal_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected = np.array([e["selected_distance"] for e in mahal_entries])
    random_d = np.array([e["random_distance"] for e in mahal_entries])
    farthest = np.array([e["farthest_distance"] for e in mahal_entries])

    n_pairs = len(mahal_entries)
    log(f"  N pairs: {n_pairs}")
    log(f"  selected mean: {selected.mean():.3f}, median: {np.median(selected):.3f}")
    log(f"  random   mean: {random_d.mean():.3f}, median: {np.median(random_d):.3f}")
    log(f"  farthest mean: {farthest.mean():.3f}, median: {np.median(farthest):.3f}")

    # Wilcoxon paired: selected vs random
    diff_sr = random_d - selected
    diff_sf = farthest - selected
    w_sr = wilcoxon(diff_sr, alternative="greater")
    w_sf = wilcoxon(diff_sf, alternative="greater")
    log(f"  Wilcoxon random > selected: stat={w_sr.statistic:.1f}, p={w_sr.pvalue:.4e}")
    log(f"  Wilcoxon farthest > selected: stat={w_sf.statistic:.1f}, p={w_sf.pvalue:.4e}")

    pct_selected_better_random = float((selected < random_d).mean())
    pct_selected_better_farthest = float((selected < farthest).mean())
    log(f"  % selected < random: {pct_selected_better_random*100:.1f}%")
    log(f"  % selected < farthest: {pct_selected_better_farthest*100:.1f}%")

    selection_stats = {
        "n_pairs": n_pairs,
        "selected_mean_distance": float(selected.mean()),
        "selected_median_distance": float(np.median(selected)),
        "random_mean_distance": float(random_d.mean()),
        "random_median_distance": float(np.median(random_d)),
        "farthest_mean_distance": float(farthest.mean()),
        "farthest_median_distance": float(np.median(farthest)),
        "wilcoxon_random_gt_selected_p": float(w_sr.pvalue),
        "wilcoxon_farthest_gt_selected_p": float(w_sf.pvalue),
        "pct_selected_lt_random": pct_selected_better_random,
        "pct_selected_lt_farthest": pct_selected_better_farthest,
    }
    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(selection_stats, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SELECTION_STATS_OUT}")

    # === 8. Retrieval: bm25s + MiniLM GPU ===
    log("\n=== 8. Retrieval (selected / random / farthest) ===")

    # Build query records (3 per entry)
    query_records = []
    for i, e in enumerate(selection_entries):
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
                "n_tok": q.get("n_tok"),
                "attrs_covered": q.get("attrs_covered"),
                "family": q.get("family"),
            })
    log(f"  total queries: {len(query_records)}")

    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    n_missing = int((target_indices < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing} queries have missing target ASINs")

    # BM25
    log("\n=== 8a. BM25 (bm25s) ===")
    corpus_texts = [asin_to_doc[a] for a in asins]
    results = bm25s_search(corpus_texts, queries)
    log(f"  computing target ranks...")
    t0 = time.time()
    bm25_results = []
    bm25_doc_ids = results.documents
    for i in range(len(queries)):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
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

    # MiniLM GPU
    log("\n=== 8b. MiniLM (GPU) ===")
    import torch
    doc_embeds_gpu, q_embeds_gpu = minilm_encode_gpu(corpus_texts, queries)
    log(f"  computing cosine similarities...")
    t0 = time.time()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    log(f"  matmul done in {time.time() - t0:.1f}s")
    log(f"  sorting...")
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

    # === 9. Per-query + summary ===
    log("\n=== 9. Saving retrieval results ===")
    for r, bm25_r, minilm_r in zip(query_records, bm25_results, minilm_results):
        r["bm25_rank"] = bm25_r["rank"]
        r["bm25_RR"] = bm25_r["RR"]
        r["bm25_hit10"] = bm25_r["hit10"]
        r["minilm_rank"] = minilm_r["rank"]
        r["minilm_RR"] = minilm_r["RR"]
        r["minilm_hit10"] = minilm_r["hit10"]

    with open(RETRIEVAL_PER_Q_OUT, "w", encoding="utf-8") as f:
        json.dump(query_records, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_PER_Q_OUT}")

    # Summary
    summary = collections.defaultdict(lambda: {
        "n": 0,
        "bm25_rank_mean": [],
        "bm25_RR_mean": [],
        "bm25_hit10_count": 0,
        "minilm_rank_mean": [],
        "minilm_RR_mean": [],
        "minilm_hit10_count": 0,
    })
    for r in query_records:
        v = r["variant"]
        s = summary[v]
        s["n"] += 1
        if r["bm25_rank"] is not None:
            s["bm25_rank_mean"].append(r["bm25_rank"])
        s["bm25_RR_mean"].append(r["bm25_RR"])
        s["bm25_hit10_count"] += r["bm25_hit10"]
        if r["minilm_rank"] is not None:
            s["minilm_rank_mean"].append(r["minilm_rank"])
        s["minilm_RR_mean"].append(r["minilm_RR"])
        s["minilm_hit10_count"] += r["minilm_hit10"]

    final_summary = {}
    for v, s in summary.items():
        bm25_ranks = np.array(s["bm25_rank_mean"]) if s["bm25_rank_mean"] else np.array([])
        minilm_ranks = np.array(s["minilm_rank_mean"]) if s["minilm_rank_mean"] else np.array([])
        final_summary[v] = {
            "n": s["n"],
            "bm25_rank_mean": float(bm25_ranks.mean()) if len(bm25_ranks) else None,
            "bm25_rank_median": float(np.median(bm25_ranks)) if len(bm25_ranks) else None,
            "bm25_RR_mean": float(np.mean(s["bm25_RR_mean"])),
            "bm25_hit10_rate": s["bm25_hit10_count"] / s["n"],
            "minilm_rank_mean": float(minilm_ranks.mean()) if len(minilm_ranks) else None,
            "minilm_rank_median": float(np.median(minilm_ranks)) if len(minilm_ranks) else None,
            "minilm_RR_mean": float(np.mean(s["minilm_RR_mean"])),
            "minilm_hit10_rate": s["minilm_hit10_count"] / s["n"],
        }
    with open(RETRIEVAL_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_SUMMARY_OUT}")

    log("\n=== Final summary ===")
    for v, s in final_summary.items():
        log(f"  {v}: n={s['n']}, "
            f"bm25_rank={s['bm25_rank_mean']}, hit10={s['bm25_hit10_rate']*100:.1f}%, "
            f"minilm_rank={s['minilm_rank_mean']}, hit10={s['minilm_hit10_rate']*100:.1f}%, "
            f"minilm_MRR={s['minilm_RR_mean']*100:.3f}%")

    log("\n=== Stage 9D complete ===")


if __name__ == "__main__":
    main()