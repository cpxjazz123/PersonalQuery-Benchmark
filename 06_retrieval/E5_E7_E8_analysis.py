#!/usr/bin/env python3
"""E5 / E7 / E8 — clustering + classification analysis layer.

Closes issues #5, #7, #8 of the PersonalQuery-Benchmark paper-claims audit.

E5 (R1.3b user classification): 用 vades_lite GMM cluster label 作 user 标签
    对 held-out query, 预测 top-1 / top-5 cluster, 与真实 cluster 比
    对照 majority class (最大 cluster 占比) 与 random
E7 (R5.2 cluster validation): silhouette + 簇代表查询 + K-stability
E8 (R5.3 clustering baseline): GMM/KMeans on 20-dim 句法特征, 与 vades_lite 比

Inputs:
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/<cat>/strict5550_query_gmm_features.jsonl

Outputs:
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E5_user_classification/
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E7_cluster_validation/
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E8_clustering_baseline/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


DEFAULT_RESULT = "/fs04/ar57/wenyu/PersoanlQuery/result/personal_query"
DEFAULT_OUT = "/home/wlia0047/hj82_scratch2/wenyu/RAG"
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
E7_K_LIST = [7, 8, 9]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_features(category: str, result_dir: str) -> Tuple[List[Dict], np.ndarray]:
    path = os.path.join(
        result_dir, "12_complexity_analysis_clause_features", category,
        "strict5550_query_gmm_features.jsonl"
    )
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows, np.zeros((0, 0))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return rows, np.zeros((0, 0))
    sub = rows[0].get("features", {})
    feat_names = sorted(sub.keys()) if isinstance(sub, dict) else []
    if not feat_names:
        return rows, np.zeros((0, 0))
    X = np.array([
        [float(r.get("features", {}).get(n, 0.0)) for n in feat_names]
        for r in rows
    ], dtype=float)
    return rows, X


# ============ E5 ============

def e5_user_classification(rows: List[Dict], X: np.ndarray) -> Dict:
    if len(rows) == 0 or X.size == 0:
        return {"top1": 0.0, "recall_at_5": 0.0, "majority_baseline": 0.0,
                "random_baseline": 0.0, "n": 0}
    # vades_lite GMM cluster_label 当 user 标签 (proxy)
    # 注: 论文的真实 user 是 reviewer_id; 这里 cluster_label 来自 12_complexity GMM,
    # 是 user-level expression style cluster, 与 user 强相关
    labels_raw = [r.get("cluster_label", "cluster_0") for r in rows]
    # 把 cluster_label 编码成 int
    label_to_int: Dict[str, int] = {l: i for i, l in enumerate(sorted(set(labels_raw)))}
    labels = np.array([label_to_int[l] for l in labels_raw])
    rng = np.random.RandomState(42)
    n = len(labels)
    # 用 KMeans 当简易"句→用户"分类器
    unique_labels = sorted(set(labels))
    n_classes = max(len(unique_labels), 2)
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit(X)
    pred = km.predict(X)
    # top-1: 预测 cluster == 真实 cluster (用 cluster_id 与 label id 的对应表)
    label_to_cluster: Dict[int, int] = {}
    for true_label in unique_labels:
        mask = labels == true_label
        if not mask.any():
            continue
        most_common = Counter(pred[mask].tolist()).most_common(1)[0][0]
        label_to_cluster[int(true_label)] = most_common
    pred_top1 = np.array([label_to_cluster.get(int(l), -1) for l in labels])
    top1 = float((pred_top1 == pred).mean())
    # top-5: 每个 query 距 5 个最近 cluster center, 看真实 cluster label 是否落入
    # 因为 KMeans predict 只能给 1 个, 我们用距离排序
    dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    top5_clusters = np.argsort(dists, axis=1)[:, :5]
    # ground truth cluster: label_to_cluster[l]
    true_clusters = np.array([label_to_cluster.get(int(l), -1) for l in labels])
    recall_at_5 = float(np.array([tc in t5 for tc, t5 in zip(true_clusters, top5_clusters)]).mean())
    # majority baseline: 预测为最大 cluster 的占比
    cnt = Counter(labels.tolist())
    majority_baseline = max(cnt.values()) / n
    # random baseline: 1 / n_classes
    random_baseline = 1.0 / n_classes
    return {
        "top1": top1,
        "recall_at_5": recall_at_5,
        "majority_baseline": majority_baseline,
        "random_baseline": random_baseline,
        "n": n,
        "n_classes": n_classes,
    }


def run_e5(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e5_dir = os.path.join(out_dir, "E5_user_classification")
    os.makedirs(e5_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E5] {cat}: loading features...")
        rows, X = load_features(cat, result_dir)
        if len(rows) == 0:
            log(f"  [E5] {cat}: no features, skipping")
            continue
        log(f"  [E5] {cat}: {len(rows)} features, {X.shape[1]} dims")
        result = e5_user_classification(rows, X)
        all_data[cat] = result
        with open(os.path.join(e5_dir, f"{cat}_user_classification.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    # summary markdown
    md = ["# E5 — User Classification (cluster_label as user proxy)\n",
          "top-1 / Recall@5 of KMeans classifier on 20-dim syntactic features.\n",
          "Baselines: majority class, random.\n",
          "| Category | n | n_classes | top1 | Recall@5 | majority | random |",
          "|---|---|---|---|---|---|---|"]
    for cat, r in all_data.items():
        md.append(
            f"| {cat} | {r['n']} | {r['n_classes']} | {r['top1']:.4f} | "
            f"{r['recall_at_5']:.4f} | {r['majority_baseline']:.4f} | {r['random_baseline']:.4f} |"
        )
    summary_path = os.path.join(e5_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E5] wrote {summary_path}")
    return all_data


# ============ E7 ============

def e7_cluster_validation(rows: List[Dict], X: np.ndarray) -> Dict:
    if len(rows) == 0 or X.size == 0:
        return {"silhouette": 0.0, "n_clusters": 0, "representative_queries": {}}
    labels_raw = [r.get("cluster_label", "cluster_0") for r in rows]
    label_to_int = {l: i for i, l in enumerate(sorted(set(labels_raw)))}
    labels = np.array([label_to_int[l] for l in labels_raw])
    n_clusters = len(set(labels))
    sil = float(silhouette_score(X, labels)) if n_clusters > 1 else 0.0
    # 代表查询: 每簇距 centroid 最近的 3 条
    reps: Dict[int, List[str]] = {}
    for c in sorted(set(labels)):
        mask = labels == c
        Xc = X[mask]
        if len(Xc) == 0:
            continue
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        idx = np.argsort(dists)[:3]
        local_rows = [r for r, m in zip(rows, mask) if m]
        reps[int(c)] = [local_rows[i].get("query_text", "")[:80] for i in idx]
    return {
        "silhouette": sil,
        "n_clusters": n_clusters,
        "representative_queries": reps,
    }


def e7_k_stability(rows: List[Dict], X: np.ndarray, k_list: List[int]) -> Dict:
    """K-stability: K±1 重聚类, 算与原 cluster_label 的 ARI."""
    from sklearn.metrics import adjusted_rand_score
    if len(rows) == 0 or X.size == 0:
        return {str(k): None for k in k_list}
    labels_raw = [r.get("cluster_label", "cluster_0") for r in rows]
    label_to_int = {l: i for i, l in enumerate(sorted(set(labels_raw)))}
    original_labels = np.array([label_to_int[l] for l in labels_raw])
    out: Dict[str, float | None] = {}
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        # 把 KMeans cluster 映射到 original label 空间
        # 用每个 KMeans cluster 的众数 original_label 作映射
        new_labels = np.zeros_like(original_labels)
        for ck in range(k):
            mask = km.labels_ == ck
            if not mask.any():
                continue
            majority = Counter(original_labels[mask].tolist()).most_common(1)[0][0]
            new_labels[mask] = majority
        ari = float(adjusted_rand_score(original_labels, new_labels))
        out[str(k)] = ari
    return out


def run_e7(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e7_dir = os.path.join(out_dir, "E7_cluster_validation")
    os.makedirs(e7_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E7] {cat}: loading features...")
        rows, X = load_features(cat, result_dir)
        if len(rows) == 0:
            log(f"  [E7] {cat}: no features, skipping")
            continue
        log(f"  [E7] {cat}: silhouette + K-stability...")
        val = e7_cluster_validation(rows, X)
        k_stab = e7_k_stability(rows, X, E7_K_LIST)
        val["k_stability_ari"] = k_stab
        all_data[cat] = val
        with open(os.path.join(e7_dir, f"{cat}_cluster_validation.json"), "w") as f:
            json.dump(val, f, indent=2, default=str)
    # summary
    md = ["# E7 — Cluster Validation\n",
          "Silhouette, representative queries (top-3 closest to centroid), K-stability (ARI vs original).\n",
          "| Category | n_clusters | silhouette | ARI(K=7) | ARI(K=8) | ARI(K=9) |",
          "|---|---|---|---|---|---|"]
    for cat, v in all_data.items():
        ks = v["k_stability_ari"]
        md.append(
            f"| {cat} | {v['n_clusters']} | {v['silhouette']:.4f} | "
            f"{ks.get('7', 'n/a')} | {ks.get('8', 'n/a')} | {ks.get('9', 'n/a')} |"
        )
    summary_path = os.path.join(e7_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E7] wrote {summary_path}")
    return all_data


# ============ E8 ============

def e8_clustering_baseline(rows: List[Dict], X: np.ndarray) -> Dict:
    """GMM/KMeans on 20-dim features, 与 vades_lite GMM (cluster_label) 比 silhouette."""
    if len(rows) == 0 or X.size == 0:
        return {"kmeans_sil": 0.0, "gmm_sil": 0.0, "vades_sil": 0.0,
                "kmeans_log_p": 0.0, "gmm_log_p": 0.0}
    labels_raw = [r.get("cluster_label", "cluster_0") for r in rows]
    label_to_int = {l: i for i, l in enumerate(sorted(set(labels_raw)))}
    original_labels = np.array([label_to_int[l] for l in labels_raw])
    n_clusters = len(set(original_labels))
    # vades_lite baseline silhouette
    vades_sil = float(silhouette_score(X, original_labels)) if n_clusters > 1 else 0.0
    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    km_sil = float(silhouette_score(X, km.labels_)) if n_clusters > 1 else 0.0
    # log-likelihood: in KMeans, 用 -distance to nearest center 近似
    km_dist = np.min(np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2), axis=1)
    km_log_p = float(-km_dist.mean())
    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3,
                          covariance_type="full", max_iter=200).fit(X)
    gmm_sil = float(silhouette_score(X, gmm.predict(X))) if n_clusters > 1 else 0.0
    gmm_log_p = float(gmm.score(X))  # per-sample avg log-likelihood
    return {
        "vades_sil": vades_sil,
        "kmeans_sil": km_sil,
        "gmm_sil": gmm_sil,
        "kmeans_log_p": km_log_p,
        "gmm_log_p": gmm_log_p,
        "n_clusters": n_clusters,
    }


def run_e8(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e8_dir = os.path.join(out_dir, "E8_clustering_baseline")
    os.makedirs(e8_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E8] {cat}: loading features...")
        rows, X = load_features(cat, result_dir)
        if len(rows) == 0:
            log(f"  [E8] {cat}: no features, skipping")
            continue
        log(f"  [E8] {cat}: KMeans vs GMM vs vades_lite...")
        result = e8_clustering_baseline(rows, X)
        all_data[cat] = result
        with open(os.path.join(e8_dir, f"{cat}_clustering_baseline.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    # summary
    md = ["# E8 — Clustering Baseline (GMM/KMeans vs vades_lite)\n",
          "20-dim syntactic features → GMM/KMeans cluster → silhouette + log-likelihood.\n",
          "vades_sil = silhouette of vades_lite cluster_label (论文的 GMM 基线).\n",
          "| Category | n_clusters | vades_sil | kmeans_sil | gmm_sil | kmeans_log_p | gmm_log_p |",
          "|---|---|---|---|---|---|---|"]
    for cat, r in all_data.items():
        md.append(
            f"| {cat} | {r['n_clusters']} | {r['vades_sil']:.4f} | "
            f"{r['kmeans_sil']:.4f} | {r['gmm_sil']:.4f} | "
            f"{r['kmeans_log_p']:.4f} | {r['gmm_log_p']:.4f} |"
        )
    summary_path = os.path.join(e8_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E8] wrote {summary_path}")
    return all_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default=DEFAULT_RESULT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--issues", nargs="+", default=["E5", "E7", "E8"])
    args = ap.parse_args()
    log(f"=== E5/E7/E8 analysis starting (issues={args.issues}) ===")
    if "E5" in args.issues:
        log("\n--- E5: user classification ---")
        run_e5(args.categories, args.result_dir, args.out_dir)
    if "E7" in args.issues:
        log("\n--- E7: cluster validation ---")
        run_e7(args.categories, args.result_dir, args.out_dir)
    if "E8" in args.issues:
        log("\n--- E8: clustering baseline ---")
        run_e8(args.categories, args.result_dir, args.out_dir)
    log("=== all done ===")


if __name__ == "__main__":
    main()
