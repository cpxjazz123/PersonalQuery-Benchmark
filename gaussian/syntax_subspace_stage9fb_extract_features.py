"""Stage 9F-B feature extraction — extract 182d spaCy features for newly generated
9F pool queries and APPEND to stage7b_query_features.jsonl.gz (no overwrite).

Output: extended cache → stage7b_query_features.jsonl.gz
Also write: stage9fb_features.jsonl.gz (only 9F queries for quick reference)

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9fb_extract_features.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_feat_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_9F_IN = SCRATCH / "stage9fb_pool.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
FEAT_9F_OUT = SCRATCH / "stage9fb_features.jsonl.gz"
COVERAGE_OUT = SCRATCH / "stage9fb_coverage_stats.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 9F-B — Feature extraction for new pool ===")

    # Load 9F pool
    pool_9f = json.load(open(POOL_9F_IN))
    new_pool = pool_9f["new_pool"]
    all_queries = []
    for asin, qs in new_pool.items():
        for q in qs:
            all_queries.append({
                "asin": asin,
                "profile": q["profile"],
                "query": q["query"],
            })
    log(f"  total new queries: {len(all_queries)}")

    # Load existing cache
    existing_keys = set()
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            existing_keys.add(rec["k"])
    log(f"  existing cache keys: {len(existing_keys)}")

    # Load 182d feature extractor
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")
    from main import per_sentence_features_v2  # spaCy 182d extractor
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Load PCA48 pipeline
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  feature names: {len(fnames)}")

    # Extract features
    log("=== Extracting features (spaCy batch) ===")
    t0 = time.time()
    feats_new = {}  # k -> {name: value}
    miss_attr = 0
    cached = 0
    # First pass: identify which queries need extraction
    to_extract = []
    for q in all_queries:
        k = feat_key(q["query"])
        if k in existing_keys:
            cached += 1
            continue
        to_extract.append(q)
    log(f"  to extract: {len(to_extract)} (cached: {cached})")

    # Batch extract with spaCy pipe
    if to_extract:
        texts = [q["query"] for q in to_extract]
        docs = list(nlp.pipe(texts, batch_size=128))
        for q, doc in zip(to_extract, docs):
            try:
                feats = per_sentence_features_v2(doc)
                # Convert to dict name -> value
                feats_dict = {fn: float(feats.get(fn, 0.0)) for fn in fnames}
                k = feat_key(q["query"])
                feats_new[k] = feats_dict
            except Exception as e:
                miss_attr += 1
                continue
    log(f"  features extracted: {len(feats_new)} (miss: {miss_attr}, t={time.time() - t0:.1f}s)")

    # === APPEND new features to cache (no overwrite) ===
    log("=== Appending to cache ===")
    n_appended = 0
    with gzip.open(FEAT_CACHE, "at", encoding="utf-8") as f:
        for k, v in feats_new.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")
            n_appended += 1
    log(f"  appended: {n_appended}")

    # Save 9F-specific features for quick access
    with gzip.open(FEAT_9F_OUT, "wt", encoding="utf-8") as f:
        for q in all_queries:
            k = feat_key(q["query"])
            v = feats_new.get(k)
            if v:
                f.write(json.dumps({
                    "asin": q["asin"], "profile": q["profile"],
                    "query": q["query"], "k": k, "v": v,
                }) + "\n")
    log(f"  wrote → {FEAT_9F_OUT}")

    # === Re-run coverage analysis ===
    log("=== Re-running PCA48 coverage analysis ===")
    from sklearn.decomposition import PCA
    scaler = P["scaler"]
    train_idx = P["train_idx"]
    pca = PCA(n_components=48, random_state=2024)
    pca.fit(P["X_scaled"][train_idx])

    # Reload cache (with appended features)
    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  cache size now: {len(feat_map)}")

    def extract_pca48(queries):
        zs, miss = [], 0
        for q in queries:
            k = feat_key(q)
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs.append(z)
        return np.stack(zs) if zs else np.zeros((0, 48)), miss

    pool_9c = json.load(open(SCRATCH / "stage9c_pool_pilot.json"))["pools"]
    cov_stats = {}
    for asin in new_pool.keys():
        queries_9c = [q["query"] for q in pool_9c.get(asin, [])]
        queries_9f = [q["query"] for q in new_pool.get(asin, [])]
        queries_combined = queries_9c + queries_9f

        z9c, miss_9c = extract_pca48(queries_9c)
        z9f, miss_9f = extract_pca48(queries_9f)
        zcomb, miss_comb = extract_pca48(queries_combined)
        if len(z9c) == 0 or len(zcomb) == 0:
            continue

        std_9c = float(np.mean(z9c.std(axis=0)))
        std_comb = float(np.mean(zcomb.std(axis=0)))
        from scipy.spatial.distance import pdist
        d_9c = float(pdist(z9c).mean()) if len(z9c) > 1 else 0
        d_comb = float(pdist(zcomb).mean()) if len(zcomb) > 1 else 0

        # PCA48 axis-wise std growth
        std_per_dim = (zcomb.std(axis=0) / z9c.std(axis=0) - 1) * 100

        # Span (max - min per dim)
        span_9c = float(np.mean(z9c.max(axis=0) - z9c.min(axis=0)))
        span_comb = float(np.mean(zcomb.max(axis=0) - zcomb.min(axis=0)))

        cov_stats[asin] = {
            "n_9c": len(z9c), "n_9f": len(z9f), "n_combined": len(zcomb),
            "miss_features": {"9c": miss_9c, "9f": miss_9f, "combined": miss_comb},
            "mean_std_9c": std_9c,
            "mean_std_combined": std_comb,
            "std_growth_pct": (std_comb / std_9c - 1) * 100 if std_9c > 0 else 0,
            "mean_pairwise_dist_9c": d_9c,
            "mean_pairwise_dist_combined": d_comb,
            "dist_growth_pct": (d_comb / d_9c - 1) * 100 if d_9c > 0 else 0,
            "mean_span_9c": span_9c,
            "mean_span_combined": span_comb,
            "span_growth_pct": (span_comb / span_9c - 1) * 100 if span_9c > 0 else 0,
            "std_per_dim_growth_pct": std_per_dim.tolist(),
        }
        log(f"  {asin}: 9C n={len(z9c)}, 9F n={len(z9f)}, "
            f"std_growth={cov_stats[asin]['std_growth_pct']:.1f}%, "
            f"dist_growth={cov_stats[asin]['dist_growth_pct']:.1f}%, "
            f"span_growth={cov_stats[asin]['span_growth_pct']:.1f}%")

    with open(COVERAGE_OUT, "w", encoding="utf-8") as f:
        json.dump(cov_stats, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {COVERAGE_OUT}")
    log("\n=== Stage 9F-B feature extraction complete ===")


if __name__ == "__main__":
    main()