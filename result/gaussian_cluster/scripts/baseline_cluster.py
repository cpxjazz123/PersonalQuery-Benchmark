#!/usr/bin/env python3
"""Baseline: 直接对 user-level 平均句法特征做 KMeans / GMM, 与 VADES user_mu 对比.

Reads cluster_smoke_sentences.jsonl (per-sentence clause features), aggregates
mean features per user (300 users × 20-dim), runs KMeans/GMM k=2..6 sweep,
then compares to gaussian_vades user_mu clustering via Adjusted Rand Index.

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/baseline_cluster_report.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster/baseline_pca_scatter.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SENTENCE_FILE = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products/cluster_smoke_sentences.jsonl")
VADES_PROFILE_FILE = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products/cluster_smoke_user_profiles.jsonl")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_cluster")
K_RANGE = [2, 3, 4, 5, 6]
RANDOM_STATE = 42


def main() -> None:
    log = lambda msg: print(f"[baseline] {msg}", flush=True)
    log(f"读取 sentences: {SENTENCE_FILE}")
    user_features: dict[str, list[dict]] = defaultdict(list)
    feature_names: list[str] | None = None
    with SENTENCE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            feats = row.get("features")
            if not feats:
                continue
            if feature_names is None:
                feature_names = list(feats.keys())
            user_features[row["user_id"]].append(feats)
    log(f"用户数={len(user_features)}, 特征维度={len(feature_names) if feature_names else 0}")

    user_ids = sorted(user_features.keys())
    X = np.asarray(
        [[float(user_features[uid][0][name]) for name in feature_names] for uid in user_ids],
        dtype=np.float64,
    )
    log(f"baseline X shape: {X.shape}")
    log(f"feature_names: {feature_names}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log(f"开始 KMeans k={K_RANGE} sweep (baseline)")
    kmeans_results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        kmeans_results[k] = {"silhouette": sil, "inertia": float(km.inertia_), "labels": labels.tolist()}
        log(f"  KMeans k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.2f}")

    log(f"开始 GMM k={K_RANGE} sweep (baseline, covariance=full)")
    gmm_results_full = {}
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3, covariance_type="full", max_iter=200)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        bic = float(gmm.bic(X_scaled))
        sil = float(silhouette_score(X_scaled, labels))
        gmm_results_full[k] = {"silhouette": sil, "bic": bic, "labels": labels.tolist()}
        log(f"  GMM full k={k}: silhouette={sil:.4f}, BIC={bic:.1f}")

    log(f"开始 GMM k={K_RANGE} sweep (baseline, covariance=tied)")
    gmm_results_tied = {}
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3, covariance_type="tied", max_iter=200)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        bic = float(gmm.bic(X_scaled))
        sil = float(silhouette_score(X_scaled, labels))
        gmm_results_tied[k] = {"silhouette": sil, "bic": bic, "labels": labels.tolist()}
        log(f"  GMM tied k={k}: silhouette={sil:.4f}, BIC={bic:.1f}")

    # 取 baseline 最优 k
    best_k_km = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
    best_k_gmm_full = max(gmm_results_full, key=lambda k: gmm_results_full[k]["silhouette"])
    best_k_gmm_tied = max(gmm_results_tied, key=lambda k: gmm_results_tied[k]["silhouette"])
    log(f"baseline 最优: KMeans k={best_k_km}, GMM-full k={best_k_gmm_full}, GMM-tied k={best_k_gmm_tied}")

    # baseline + VADES 配对比较
    log(f"读取 VADES user_profiles: {VADES_PROFILE_FILE}")
    vades_user_ids: list[str] = []
    vades_vectors: list[list[float]] = []
    with VADES_PROFILE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            vades_user_ids.append(row["user_id"])
            vades_vectors.append(row["user_mu"])
    log(f"VADES users={len(vades_user_ids)}, dim={len(vades_vectors[0])}")
    X_vades = np.asarray(vades_vectors, dtype=np.float64)
    scaler_vades = StandardScaler()
    X_vades_scaled = scaler_vades.fit_transform(X_vades)

    # 对齐 user_ids (应该一致，但保险)
    assert sorted(vades_user_ids) == sorted(user_ids), "VADES 与 baseline 用户顺序不一致"
    vades_idx_map = {uid: i for i, uid in enumerate(vades_user_ids)}
    vades_in_baseline_order = np.asarray([X_vades[vades_idx_map[uid]] for uid in user_ids], dtype=np.float64)
    X_vades_aligned = scaler_vades.fit_transform(vades_in_baseline_order)

    vades_kmeans_results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_vades_aligned)
        sil = float(silhouette_score(X_vades_aligned, labels))
        vades_kmeans_results[k] = {"silhouette": sil, "inertia": float(km.inertia_), "labels": labels.tolist()}
        log(f"  VADES-KMeans k={k}: silhouette={sil:.4f}")

    vades_gmm_results = {}
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3, covariance_type="full", max_iter=200)
        gmm.fit(X_vades_aligned)
        labels = gmm.predict(X_vades_aligned)
        bic = float(gmm.bic(X_vades_aligned))
        sil = float(silhouette_score(X_vades_aligned, labels))
        vades_gmm_results[k] = {"silhouette": sil, "bic": bic, "labels": labels.tolist()}
        log(f"  VADES-GMM k={k}: silhouette={sil:.4f}, BIC={bic:.1f}")

    # Adjusted Rand Index 对比 (各 k 下 baseline vs VADES 分配)
    log("=" * 70)
    log("Adjusted Rand Index (baseline vs VADES):")
    ari_table = {}
    for k in K_RANGE:
        ari_km = adjusted_rand_score(kmeans_results[k]["labels"], vades_kmeans_results[k]["labels"])
        ari_gmm = adjusted_rand_score(gmm_results_full[k]["labels"], vades_gmm_results[k]["labels"])
        ari_table[k] = {"kmeans": float(ari_km), "gmm": float(ari_gmm)}
        log(f"  k={k}: ARI_KMeans={ari_km:.4f}, ARI_GMM={ari_gmm:.4f}")

    report = {
        "n_users": int(X.shape[0]),
        "dim_baseline": int(X.shape[1]),
        "dim_vades": int(X_vades.shape[1]),
        "feature_names": feature_names,
        "k_range": K_RANGE,
        "baseline": {
            "kmeans": {k: {"silhouette": r["silhouette"], "inertia": r["inertia"]} for k, r in kmeans_results.items()},
            "gmm_full": {k: {"silhouette": r["silhouette"], "bic": r["bic"]} for k, r in gmm_results_full.items()},
            "gmm_tied": {k: {"silhouette": r["silhouette"], "bic": r["bic"]} for k, r in gmm_results_tied.items()},
            "best_k_kmeans": int(best_k_km),
            "best_k_gmm_full": int(best_k_gmm_full),
            "best_k_gmm_tied": int(best_k_gmm_tied),
        },
        "vades": {
            "kmeans": {k: {"silhouette": r["silhouette"], "inertia": r["inertia"]} for k, r in vades_kmeans_results.items()},
            "gmm_full": {k: {"silhouette": r["silhouette"], "bic": r["bic"]} for k, r in vades_gmm_results.items()},
        },
        "ari_baseline_vs_vades": ari_table,
    }

    report_path = OUT_DIR / "baseline_cluster_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 {report_path}")

    # 2D scatter baseline vs VADES side by side
    pca_baseline = PCA(n_components=2, random_state=RANDOM_STATE)
    X_base_pca = pca_baseline.fit_transform(X_scaled)
    pca_vades = PCA(n_components=2, random_state=RANDOM_STATE)
    X_vades_pca = pca_vades.fit_transform(X_vades_aligned)

    km_base_k2 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
    km_vades_k2 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit_predict(X_vades_aligned)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#4C72B0", "#DD8452"]
    for cid in [0, 1]:
        m1 = km_base_k2 == cid
        m2 = km_vades_k2 == cid
        axes[0].scatter(X_base_pca[m1, 0], X_base_pca[m1, 1], color=colors[cid], s=20, alpha=0.7, label=f"Cluster {cid} (n={int(m1.sum())})")
        axes[1].scatter(X_vades_pca[m2, 0], X_vades_pca[m2, 1], color=colors[cid], s=20, alpha=0.7, label=f"Cluster {cid} (n={int(m2.sum())})")
    axes[0].set_title(f"Baseline: 直接句法特征均值 → KMeans k=2\nsilhouette={kmeans_results[2]['silhouette']:.4f}")
    axes[0].set_xlabel(f"PC1 ({pca_baseline.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca_baseline.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(f"VADES: 训出 user_mu → KMeans k=2\nsilhouette={vades_kmeans_results[2]['silhouette']:.4f}")
    axes[1].set_xlabel(f"PC1 ({pca_vades.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca_vades.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_pca_scatter.png", dpi=120)
    plt.close(fig)
    log(f"已写入 {OUT_DIR / 'baseline_pca_scatter.png'}")

    log("完成")


if __name__ == "__main__":
    main()
