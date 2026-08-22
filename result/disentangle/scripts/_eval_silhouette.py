#!/usr/bin/env python3
"""Disentangle 评估: 比较 baseline vs disentangle user_mu 的聚类清晰度.

== 指标 ==
- silhouette score on user_mu (KMeans K=8 cluster)
- silhouette score on style_centers (KMeans K=8, 等价于 K=8 自身)
- intra/inter cluster L2 distance ratio

对比:
- baseline: diagonal_gmm 训练的 vades_encoder.pt / vades_user_table.pt
- disentangle: 新训练的 vades_encoder.pt / vades_user_table.pt (OUTPUT_TAG=vades_disentangled_...)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian_vades import (
    UserDistributionTableDisentangled,
    UserDistributionTableGMM,
    SentenceEncoder,
    pre_cluster_users_for_disentangle,
)

RESULT_DIR = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products")
K = 8
SEED = 42


def load_user_mu_baseline(n_users: int, latent_dim: int) -> np.ndarray:
    """Baseline: 从 vades_user_table.pt 加载 (UserDistributionTableGMM)."""
    ckpt = torch.load(RESULT_DIR / "vades_user_table.pt", map_location="cpu")
    # 基线是用 GMM(K=2) train 的, ckpt 含 mix_logits + log_scale + mu
    if "user_mu" in ckpt:
        return ckpt["user_mu"].numpy()
    raise ValueError(f"baseline ckpt keys: {list(ckpt.keys())}")


def load_user_mu_disentangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disentangle: 从 user_table.user_mu 属性 (style + offset) 计算."""
    ckpt = torch.load(RESULT_DIR / "vades_user_table.pt", map_location="cpu")
    if "style_centers" not in ckpt:
        raise ValueError(f"disentangle ckpt keys: {list(ckpt.keys())}")
    sc = ckpt["style_centers"]  # [n_clusters, latent_dim]
    off = ckpt["user_offsets"]  # [num_users, latent_dim]
    cluster_ids = ckpt["user_cluster_ids"]  # [num_users]
    user_mu = (sc[cluster_ids] + off).numpy()
    return user_mu, sc.numpy(), cluster_ids.numpy()


def silhouette_metrics(X: np.ndarray, k: int, name: str) -> dict:
    if X.shape[0] < k + 1:
        k = max(2, X.shape[0] - 1)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    sil = float(silhouette_score(X, labels))
    # intra/inter distance ratio
    dists_within = []
    dists_between = []
    centers = km.cluster_centers_
    for i, x in enumerate(X):
        c = labels[i]
        dists_within.append(float(np.linalg.norm(x - centers[c])))
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            dists_between.append(float(np.linalg.norm(centers[c1] - centers[c2])))
    intra_mean = float(np.mean(dists_within))
    inter_mean = float(np.mean(dists_between)) if dists_between else 0.0
    ratio = intra_mean / max(1e-9, inter_mean)
    return {
        "name": name,
        "n": int(X.shape[0]),
        "k": int(k),
        "silhouette": sil,
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "intra_inter_ratio": ratio,
        "cluster_sizes": [int((labels == c).sum()) for c in range(k)],
    }


def main() -> None:
    log = lambda m: print(f"[silhouette] {m}", flush=True)
    log("=== 对比 baseline vs disentangle user_mu 聚类清晰度 ===")

    # 1) Baseline: UserDistributionTableGMM 训的 user_mu
    try:
        baseline_ckpt = torch.load(RESULT_DIR / "vades_user_table.pt", map_location="cpu")
        if "user_mu" in baseline_ckpt:
            baseline_mu = baseline_ckpt["user_mu"].numpy()
        else:
            log(f"baseline ckpt 无 user_mu, keys={list(baseline_ckpt.keys())}")
            baseline_mu = None
    except Exception as e:
        log(f"baseline 加载失败: {e!r}")
        baseline_mu = None

    # 2) Disentangle: 重训后的 user_mu (从 user_cluster_ids + style_centers + user_offsets 推)
    try:
        disent_mu, disent_sc, disent_cluster = load_user_mu_disentangle()
    except Exception as e:
        log(f"disentangle 加载失败: {e!r}")
        disent_mu = None

    # 3) Style centers (disentangle 的 cluster anchors): 直接对 sc 跑 KMeans (K=n_clusters) 应 silhouette=1
    summary = {}
    if baseline_mu is not None:
        b = silhouette_metrics(baseline_mu, K, "baseline_user_mu(diagonal_gmm)")
        log(f"baseline: silhouette={b['silhouette']:.4f}  intra/inter={b['intra_inter_ratio']:.3f}")
        summary["baseline"] = b
    if disent_mu is not None:
        d = silhouette_metrics(disent_mu, K, "disentangle_user_mu")
        log(f"disentangle: silhouette={d['silhouette']:.4f}  intra/inter={d['intra_inter_ratio']:.3f}")
        summary["disentangle"] = d
        # 也对 style_centers 评估
        sc_metrics = silhouette_metrics(disent_sc, disent_sc.shape[0], "disentangle_style_centers(self_k)")
        log(f"style_centers K={disent_sc.shape[0]}: silhouette={sc_metrics['silhouette']:.4f}")
        summary["disentangle_style_centers"] = sc_metrics

    # 4) 验证: KMeans on style_centers 重新分配, 与 user_cluster_ids 的 ARI
    if disent_mu is not None and disent_sc.shape[0] > 1:
        from sklearn.metrics import adjusted_rand_score
        km_sc = KMeans(n_clusters=disent_sc.shape[0], random_state=SEED, n_init=10)
        km_sc.fit(disent_sc)
        sc_labels = km_sc.labels_
        # 重新 assign 用户 (按 user_mu → 最近 style_center)
        from scipy.spatial.distance import cdist
        user_d_to_sc = cdist(disent_mu, disent_sc)
        user_reassigned = np.argmin(user_d_to_sc, axis=1)
        ari = float(adjusted_rand_score(disent_cluster, user_reassigned))
        log(f"user_cluster_ids vs reassigned-from-user_mu ARI: {ari:.4f}")
        summary["ari_user_cluster_vs_reassigned"] = ari

    out_path = Path("/home/wlia0047/hj82_scratch2/wenyu/disentangle/silhouette_compare.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 {out_path}")


if __name__ == "__main__":
    main()