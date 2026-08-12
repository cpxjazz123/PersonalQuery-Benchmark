#!/usr/bin/env python3
"""E7 — Cluster semantic validation: silhouette, inter/intra distance, K-stability.

For each category:
- Compute silhouette score per cluster
- Inter-cluster centroid distance vs intra-cluster distance
- For K∈{K-1, K, K+1}, refit GMM and report ΔRange via E1 retrieval cache
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

E1_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
COMPLEX_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E7_cluster_validation")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_complexity(category: str) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Returns records, pca matrix, cluster labels."""
    p = COMPLEX_DIR / category / "strict5550_query_gmm_features.jsonl"
    records = []
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))
    pca = np.asarray([r["pca_embedding"] for r in records], dtype=np.float32)
    labels = np.asarray([r["cluster_index"] for r in records])
    return records, pca, labels


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample silhouette score (vectorized is hard; use sklearn if available)."""
    try:
        from sklearn.metrics import silhouette_samples as sk_sil
        return sk_sil(X, labels)
    except ImportError:
        # fallback: simple formula
        n = len(X)
        out = np.zeros(n)
        unique_labels = np.unique(labels)
        for i in range(n):
            same = labels == labels[i]
            other = labels != labels[i]
            if same.sum() <= 1:
                out[i] = 0.0
                continue
            a = np.mean(np.linalg.norm(X[same] - X[i], axis=1))
            b_mean = float("inf")
            for l in unique_labels:
                if l == labels[i]:
                    continue
                mask = labels == l
                if mask.sum() == 0:
                    continue
                d = np.mean(np.linalg.norm(X[mask] - X[i], axis=1))
                if d < b_mean:
                    b_mean = d
            out[i] = (b_mean - a) / max(a, b_mean)
        return out


def compute_cluster_metrics(pca: np.ndarray, labels: np.ndarray) -> Dict:
    sil = silhouette_samples(pca, labels)
    unique = np.unique(labels)
    inter_dists = []
    intra_dists = []
    centroids = {}
    for u in unique:
        mask = labels == u
        if mask.sum() == 0:
            continue
        c = pca[mask].mean(axis=0)
        centroids[int(u)] = c
        # intra: mean distance from points to centroid
        intra_dists.append(float(np.mean(np.linalg.norm(pca[mask] - c, axis=1))))
    # inter: pairwise centroid distance
    keys = sorted(centroids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter_dists.append(float(np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])))
    return {
        "n_clusters": len(unique),
        "n_samples": int(len(pca)),
        "silhouette_mean": float(np.mean(sil)),
        "silhouette_per_cluster": {int(u): float(np.mean(sil[labels == u])) for u in unique},
        "intra_cluster_mean_distance": float(np.mean(intra_dists)) if intra_dists else 0.0,
        "inter_cluster_mean_distance": float(np.mean(inter_dists)) if inter_dists else 0.0,
        "inter_intra_ratio": float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-9)) if intra_dists else 0.0,
        "cluster_sizes": {int(u): int((labels == u).sum()) for u in unique},
    }


def refit_gmm(pca: np.ndarray, k: int, seed: int = 42):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=k, random_state=seed, n_init=3)
    gmm.fit(pca)
    return gmm.predict(pca)


def process_category(category: str) -> Dict:
    log(f"\n=== {category} ===")
    records, pca, labels = load_complexity(category)
    log(f"  records={len(records)} clusters={len(np.unique(labels))}")
    metrics = compute_cluster_metrics(pca, labels)
    log(f"  silhouette={metrics['silhouette_mean']:.4f} inter/intra={metrics['inter_intra_ratio']:.4f}")

    # K-stability: refit GMM for K-1, K, K+1, compute ΔRange of cluster sizes
    k_orig = metrics["n_clusters"]
    log(f"  K-stability check: K∈{{{k_orig - 1}, {k_orig}, {k_orig + 1}}}")
    size_drifts = []
    for k in [k_orig - 1, k_orig, k_orig + 1]:
        if k < 2:
            continue
        try:
            new_labels = refit_gmm(pca, k)
            new_sizes = np.bincount(new_labels)
            size_drifts.append({"k": k, "sizes": [int(x) for x in new_sizes]})
            log(f"    K={k}: sizes={[int(x) for x in new_sizes]}")
        except Exception as e:
            log(f"    K={k} ERROR: {e}")

    return {
        "category": category,
        "metrics": metrics,
        "k_stability": size_drifts,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E7 — Cluster validation (silhouette, inter/intra, K-stability)\n",
            "**Method** — PCA-10 features + GMM cluster labels from 12_complexity; silhouette via sklearn.\n",
            "| Category | n_clusters | n_samples | silhouette | intra_dist | inter_dist | inter/intra |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for s in sums:
        m = s["metrics"]
        rows.append(
            f"| {s['category']} | {m['n_clusters']} | {m['n_samples']} | "
            f"{m['silhouette_mean']:.4f} | {m['intra_cluster_mean_distance']:.4f} | "
            f"{m['inter_cluster_mean_distance']:.4f} | {m['inter_intra_ratio']:.4f} |"
        )
    rows += ["", "## K-stability (refit GMM with K±1)", ""]
    for s in sums:
        rows.append(f"### {s['category']}")
        rows.append("| K | cluster sizes |")
        rows.append("|---:|---|")
        for d in s["k_stability"]:
            rows.append(f"| {d['k']} | {d['sizes']} |")
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
            with open(OUT_DIR / f"{cat}_cluster.json", "w") as f:
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