#!/usr/bin/env python3
"""Cluster user style vectors (user_mu) from gaussian_vades.py train output.

Reads cluster_smoke_user_profiles.jsonl, runs KMeans / GMM for k=2..6,
computes silhouette score, picks best k, then reports per-cluster size +
representative user_ids (closest to centroid).

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/cluster_report.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/cluster_assignments.jsonl
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/cluster_k_sweep.png
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/cluster_pca_scatter.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROFILE_FILE = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products/cluster_smoke_user_profiles.jsonl")
SENTENCE_FILE = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products/cluster_smoke_sentences.jsonl")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster")
OUT_DIR.mkdir(parents=True, exist_ok=True)
K_RANGE = [2, 3, 4, 5, 6]
RANDOM_STATE = 42


def main() -> None:
    log = lambda msg: print(f"[cluster] {msg}", flush=True)
    log(f"读取 user profiles: {PROFILE_FILE}")
    user_ids: list[str] = []
    vectors: list[list[float]] = []
    with PROFILE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_ids.append(row["user_id"])
            vectors.append(row["user_mu"])
    X = np.asarray(vectors, dtype=np.float64)
    log(f"users={X.shape[0]}, dim={X.shape[1]}")

    # 也加载每用户的代表性句子 (cluster 报告里展示)
    log(f"读取 sentences: {SENTENCE_FILE}")
    user_repr_sentence: dict[str, str] = {}
    with SENTENCE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = row["user_id"]
            if uid not in user_repr_sentence and not row.get("is_holdout", False):
                user_repr_sentence[uid] = row.get("sentence_text", "")
    log(f"代表性句子覆盖用户: {len(user_repr_sentence)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k sweep
    log(f"开始 KMeans k={K_RANGE} sweep")
    kmeans_results = {}
    gmm_results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        kmeans_results[k] = {"silhouette": sil, "inertia": float(km.inertia_), "labels": labels.tolist()}
        log(f"  KMeans k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.2f}")

    log(f"开始 GMM k={K_RANGE} sweep")
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3, covariance_type="full", max_iter=200)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        bic = float(gmm.bic(X_scaled))
        sil = float(silhouette_score(X_scaled, labels))
        gmm_results[k] = {"silhouette": sil, "bic": bic, "labels": labels.tolist()}
        log(f"  GMM k={k}: silhouette={sil:.4f}, BIC={bic:.1f}")

    best_k_km = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
    best_k_gmm = max(gmm_results, key=lambda k: gmm_results[k]["silhouette"])
    log(f"最优 KMeans k={best_k_km}, GMM k={best_k_gmm}")

    # 取 best k 的 KMeans 结果做最终 cluster 报告
    final_k = best_k_km
    final_labels = np.asarray(kmeans_results[final_k]["labels"])
    km_final = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)
    km_final.fit(X_scaled)
    centroids_scaled = km_final.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # 每簇代表性 user (离 centroid 最近)
    log(f"对每个 cluster 找代表性 user (最近 centroid)")
    cluster_assignments = []
    cluster_summary = []
    for cid in range(final_k):
        mask = final_labels == cid
        n_users = int(mask.sum())
        cluster_user_ids = [user_ids[i] for i in range(len(user_ids)) if mask[i]]
        # 计算到 centroid 距离
        distances = np.linalg.norm(X_scaled[mask] - centroids_scaled[cid], axis=1)
        order = np.argsort(distances)
        closest_indices = np.where(mask)[0][order[:3]].tolist()
        closest_user_ids = [user_ids[i] for i in closest_indices]
        closest_sentences = [user_repr_sentence.get(uid, "<no sentence>")[:200] for uid in closest_user_ids]
        cluster_assignments.append({
            "cluster_id": cid,
            "n_users": n_users,
            "closest_user_ids": closest_user_ids,
            "closest_sentences": closest_sentences,
            "centroid_original": centroids_original[cid].tolist(),
        })
        for idx in np.where(mask)[0]:
            cluster_assignments_per_user = {
                "user_id": user_ids[idx],
                "cluster_id": cid,
            }
            cluster_assignments.append(cluster_assignments_per_user)
        cluster_summary.append({
            "cluster_id": cid,
            "n_users": n_users,
            "representative_user_ids": closest_user_ids,
            "representative_sentences": closest_sentences,
            "centroid": centroids_original[cid].tolist(),
            "centroid_norm": float(np.linalg.norm(centroids_original[cid])),
            "intra_cluster_distance_mean": float(distances.mean()),
            "intra_cluster_distance_std": float(distances.std()),
        })
        log(f"  Cluster {cid}: {n_users} 用户, representative={[uid[:8] + '...' for uid in closest_user_ids]}")

    report = {
        "n_users": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "k_range": K_RANGE,
        "best_k_kmeans": int(best_k_km),
        "best_k_gmm": int(best_k_gmm),
        "kmeans_results": {k: {"silhouette": r["silhouette"], "inertia": r["inertia"]} for k, r in kmeans_results.items()},
        "gmm_results": {k: {"silhouette": r["silhouette"], "bic": r["bic"]} for k, r in gmm_results.items()},
        "final_k": int(final_k),
        "cluster_summary": cluster_summary,
    }

    report_path = OUT_DIR / "cluster_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 {report_path}")

    assign_path = OUT_DIR / "cluster_assignments.jsonl"
    with assign_path.open("w", encoding="utf-8") as f:
        for row in cluster_assignments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"已写入 {assign_path}")

    # plot k sweep
    sweep_path = OUT_DIR / "cluster_k_sweep.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = list(kmeans_results.keys())
    axes[0].plot(ks, [kmeans_results[k]["silhouette"] for k in ks], "o-", label="KMeans silhouette")
    axes[0].plot(ks, [gmm_results[k]["silhouette"] for k in ks], "s--", label="GMM silhouette")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette score")
    axes[0].set_title("Silhouette vs k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ks, [gmm_results[k]["bic"] for k in ks], "s-", color="orange", label="GMM BIC")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("BIC")
    axes[1].set_title("GMM BIC vs k")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sweep_path, dpi=120)
    plt.close(fig)
    log(f"已写入 {sweep_path}")

    # plot PCA scatter
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    scatter_path = OUT_DIR / "cluster_pca_scatter.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, final_k))
    for cid in range(final_k):
        mask = final_labels == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[cid], label=f"Cluster {cid} (n={int(mask.sum())})", s=20, alpha=0.7)
    ax.scatter(pca.transform(centroids_scaled)[:, 0], pca.transform(centroids_scaled)[:, 1],
               color="black", marker="X", s=200, label="centroid")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"User style vectors PCA (KMeans k={final_k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    log(f"已写入 {scatter_path}")

    log("完成")


if __name__ == "__main__":
    main()
