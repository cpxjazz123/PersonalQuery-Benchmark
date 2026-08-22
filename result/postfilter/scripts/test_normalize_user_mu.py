#!/usr/bin/env python3
"""快速测试: 把 user_mu 归一化到 encoder 输出尺度, 看是否能恢复 signal.

user_mu 实际尺度: std=2.645
encoder 输出尺度: std~2.48 (3000u, 实际测得)
目标: 让 user_mu 和 encoder 输出在同一尺度 (std≈1) 然后做 Mahalanobis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian.gaussian_vades import (
    SentenceEncoder, UserDistributionTableDisentangled,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_disentangled_v2_3000u"
ENCODER_CKPT = VADES_DIR / f"vades_encoder_{OUTPUT_TAG}.pt"
USER_TABLE_CKPT = VADES_DIR / f"vades_user_table_{OUTPUT_TAG}.pt"
USER_PROFILE_FILE = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_offsets"].shape[0])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=LATENT_DIM,
        user_cluster_ids=placeholder_cluster, style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    return encoder, user_table, ut_sd, n_clusters, num_users


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_n @ b_n.T


def main():
    log = lambda m: print(f"[norm] {m}", flush=True)
    log("=" * 70)
    log("归一化 user_mu 到 encoder 尺度")
    log("=" * 70)

    encoder, user_table, ut_sd, n_clusters, num_users = load_models()
    user_offsets = ut_sd["user_offsets"].cpu().numpy()
    style_centers = ut_sd["style_centers"].cpu().numpy()
    ucid = ut_sd["user_cluster_ids"].cpu().numpy()
    user_mu = style_centers[ucid] + user_offsets

    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    holdout = [r for r in rows if r.get("is_holdout")]

    n_eval = min(2000, len(holdout))
    holdout = holdout[:n_eval]
    mu_s = np.array([r["mu"] for r in holdout], dtype=np.float64)

    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i
    ho_uid_idx = np.array([user_id_to_idx[r["user_id"]] for r in holdout])

    log(f"  user_mu std: {user_mu.std():.4f}")
    log(f"  mu_s std: {mu_s.std():.4f}")

    # 方法 1: 简单缩放 user_mu 到 encoder scale
    scale_factor = mu_s.std() / user_mu.std()
    user_mu_scaled = user_mu * scale_factor
    log(f"\n  === 方法 1: scale user_mu by {scale_factor:.4f} ===")
    D = ((mu_s[:, None, :] - user_mu_scaled[None, :, :]) ** 2).sum(axis=2)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    log(f"  L2 dist: margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")
    log(f"  top-1 acc: {(np.argmin(D, axis=1) == ho_uid_idx).mean()*100:.2f}%")

    # 方法 2: 把 encoder 输出归一化到 user_mu scale
    scale_inv = user_mu.std() / mu_s.std()
    mu_s_scaled = mu_s * scale_inv
    log(f"\n  === 方法 2: scale mu_s by {scale_inv:.4f} (反方向) ===")
    D = ((mu_s_scaled[:, None, :] - user_mu[None, :, :]) ** 2).sum(axis=2)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    log(f"  L2 dist: margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")
    log(f"  top-1 acc: {(np.argmin(D, axis=1) == ho_uid_idx).mean()*100:.2f}%")

    # 方法 3: 都归一化到 std=1 (Z-score)
    user_mu_z = (user_mu - user_mu.mean(axis=0)) / (user_mu.std(axis=0) + 1e-9)
    mu_s_z = (mu_s - mu_s.mean(axis=0)) / (mu_s.std(axis=0) + 1e-9)
    log(f"\n  === 方法 3: Z-score both ===")
    D = ((mu_s_z[:, None, :] - user_mu_z[None, :, :]) ** 2).sum(axis=2)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    log(f"  L2 dist: margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")
    log(f"  top-1 acc: {(np.argmin(D, axis=1) == ho_uid_idx).mean()*100:.2f}%")

    # 方法 4: cosine (already tried but include for completeness)
    log(f"\n  === 方法 4: cosine distance ===")
    D = cosine_dist(mu_s, user_mu)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    log(f"  cosine: margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")
    log(f"  top-1 acc: {(np.argmin(D, axis=1) == ho_uid_idx).mean()*100:.2f}%")


if __name__ == "__main__":
    main()