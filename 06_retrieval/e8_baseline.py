#!/usr/bin/env python3
"""E8 — Direct clustering baseline: log-likelihood vs GMM user-prior.

For each category:
- Fit GMM on all PCA features (12_complexity output)
- Compute log-likelihood per sample
- Compare with simple KMeans baseline (sklearn)
- Report mean log p per category + std

The "user-prior" log p should be higher than naive KMeans if user-level
hierarchical prior helps; otherwise they're similar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

COMPLEX_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E8_clustering_baseline")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pca(category: str) -> np.ndarray:
    p = COMPLEX_DIR / category / "strict5550_query_gmm_features.jsonl"
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line)["pca_embedding"])
    return np.asarray(out, dtype=np.float32)


def fit_and_score(pca: np.ndarray, k: int, seed: int = 42) -> Dict:
    """Fit GMM and KMeans on PCA, return log-likelihood per sample + within-cluster distance."""
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    gmm = GaussianMixture(n_components=k, random_state=seed, n_init=3)
    gmm.fit(pca)
    gmm_logp = gmm.score_samples(pca)  # log p per sample
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km_labels = km.fit_predict(pca)
    # Within-cluster sum of distances
    km_within = []
    for c in range(k):
        mask = km_labels == c
        if mask.sum() == 0:
            continue
        center = km.cluster_centers_[c]
        d = np.linalg.norm(pca[mask] - center, axis=1)
        km_within.append(float(np.mean(d)))
    return {
        "gmm_logp_mean": float(np.mean(gmm_logp)),
        "gmm_logp_std": float(np.std(gmm_logp)),
        "kmeans_within_mean": float(np.mean(km_within)) if km_within else 0.0,
        "kmeans_inertia": float(km.inertia_),
    }


def process_category(category: str) -> Dict:
    log(f"\n=== {category} ===")
    pca = load_pca(category)
    log(f"  samples={len(pca)} dim={pca.shape[1]}")
    results = {}
    for k in [3, 5, 7, 9]:
        try:
            r = fit_and_score(pca, k)
            results[f"k={k}"] = r
            log(f"  K={k}: gmm_logp={r['gmm_logp_mean']:.4f}±{r['gmm_logp_std']:.4f}  "
                f"kmeans_within={r['kmeans_within_mean']:.4f}")
        except Exception as e:
            log(f"  K={k} ERROR: {e}")
    return {
        "category": category,
        "n_samples": int(len(pca)),
        "results_by_k": results,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E8 — Direct clustering baseline (GMM vs KMeans log-likelihood)\n",
            "**Method** — fit GMM and KMeans on PCA-10 features from 12_complexity; compare log-likelihood per sample.\n",
            "| Category | n_samples | K | GMM log p (mean±std) | KMeans within-cluster dist | KMeans inertia |",
            "|---|---:|---:|---:|---:|---:|"]
    for s in sums:
        for k_label, r in s["results_by_k"].items():
            rows.append(
                f"| {s['category']} | {s['n_samples']} | {k_label} | "
                f"{r['gmm_logp_mean']:.4f}±{r['gmm_logp_std']:.4f} | "
                f"{r['kmeans_within_mean']:.4f} | {r['kmeans_inertia']:.2f} |"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary_full.md"
    with open(p, "w") as f:
        f.write("\n".join(rows))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sums = []
    for cat in args.categories:
        try:
            s = process_category(cat)
            with open(OUT_DIR / f"{cat}_baseline.json", "w") as f:
                json.dump(s, f, indent=2, default=str)
            sums.append(s)
        except Exception as e:
            log(f"ERROR {cat}: {e}")
            import traceback
            traceback.print_exc()
    if sums:
        p = write_summary_md(sums)
        log(f"wrote {p}")
    log("=== done ===")


if __name__ == "__main__":
    main()