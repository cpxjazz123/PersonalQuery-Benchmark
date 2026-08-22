#!/usr/bin/env python3
"""用 3000u VADES, 尝试 3 种 user 表示 + 距离方案 区分用户.

方法 A: Cosine on user_mu (style_centers + user_offsets)
方法 B: Cosine on user_offsets only (subtract cluster center)
方法 C: Mahalanobis on user_offsets only (subtract cluster center)

对每种方法, 在 500 holdout 上计算:
- 自己的 cosine/Mahalanobis
- 最近的其他人 cosine/Mahalanobis
- margin = D_cross - D_self
- top-1 正确率
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
    latent_dim = int(ut_sd["style_centers"].shape[1])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, latent_dim)
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=latent_dim,
        user_cluster_ids=placeholder_cluster, style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()
    return encoder, user_table, ut_sd, n_clusters, num_users


def cosine_dist(a, b):
    """a [N, D], b [M, D] → cosine distance [N, M] = 1 - cos_sim."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_n @ b_n.T


def mahalanobis_diag_batch(a, b, logvar):
    """a [N, D], b [M, D], logvar [M, D] → D [N, M]."""
    diff = a[:, None, :] - b[None, :, :]
    var = np.exp(logvar).clip(min=1e-6)
    return (diff ** 2 / var[None, :, :]).sum(axis=2)


def main():
    log = lambda m: print(f"[diag3] {m}", flush=True)
    log("=" * 70)
    log("3 种距离方案 in latent space")
    log("=" * 70)

    encoder, user_table, ut_sd, n_clusters, num_users = load_models()
    user_offsets = ut_sd["user_offsets"].cpu().numpy()
    style_centers = ut_sd["style_centers"].cpu().numpy()
    # user_logvar is fixed at -2.0 (per model definition), not stored in state_dict
    user_logvar = np.full((num_users, LATENT_DIM), -2.0, dtype=np.float32)
    # placeholder cluster: all users use cluster n_clusters-1
    # real user_cluster_ids from training, but since we use placeholder, simulate
    # Actually let's look at real user_cluster_ids used during training.
    # Saved in user_table state_dict? Check.
    if "user_cluster_ids" in ut_sd:
        ucid = ut_sd["user_cluster_ids"].cpu().numpy()
        log(f"  user_cluster_ids loaded from state_dict: shape={ucid.shape}, unique={np.unique(ucid).shape[0]}")
    else:
        ucid = np.full(num_users, n_clusters - 1, dtype=int)
        log(f"  user_cluster_ids not in state_dict, placeholder={n_clusters-1}")
    user_mu = style_centers[ucid] + user_offsets

    log(f"  user_mu shape: {user_mu.shape}, std: {user_mu.std():.4f}")
    log(f"  user_offsets std: {user_offsets.std():.4f}")

    log(f"加载 sentences ...")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    holdout = [r for r in rows if r.get("is_holdout")]
    log(f"  holdout: {len(holdout)}")

    # 取前 1000 个 holdout
    n_eval = min(1000, len(holdout))
    holdout = holdout[:n_eval]
    mu_s = np.array([r["mu"] for r in holdout], dtype=np.float64)
    log(f"  eval sample: {n_eval}")

    # user_id → idx
    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i
    idx_to_uid = {v: k for k, v in user_id_to_idx.items()}
    ho_uid_idx = np.array([user_id_to_idx[r["user_id"]] for r in holdout])

    # === A: Cosine on user_mu ===
    log(f"\n=== A: Cosine on user_mu ===")
    D_A = cosine_dist(mu_s, user_mu)
    self_d = np.array([D_A[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D_A.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin_A = cross_d - self_d
    top1 = np.argmin(D_A, axis=1)
    top1_acc = (top1 == ho_uid_idx).mean()
    log(f"  D_self: mean={self_d.mean():.4f}, std={self_d.std():.4f}")
    log(f"  D_cross_min: mean={cross_d.mean():.4f}, std={cross_d.std():.4f}")
    log(f"  margin: mean={margin_A.mean():.4f}, std={margin_A.std():.4f}")
    log(f"  margin > 0: {(margin_A > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1_acc*100:.2f}%")

    # === B: Cosine on user_offsets only ===
    log(f"\n=== B: Cosine on user_offsets only ===")
    # Subtract cluster center first to get "user deviation"
    # user_deviation[u] = user_offsets[u] (already cluster-independent)
    D_B = cosine_dist(mu_s, user_offsets)
    self_d = np.array([D_B[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D_B.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin_B = cross_d - self_d
    top1 = np.argmin(D_B, axis=1)
    top1_acc = (top1 == ho_uid_idx).mean()
    log(f"  D_self: mean={self_d.mean():.4f}, std={self_d.std():.4f}")
    log(f"  D_cross_min: mean={cross_d.mean():.4f}, std={cross_d.std():.4f}")
    log(f"  margin: mean={margin_B.mean():.4f}, std={margin_B.std():.4f}")
    log(f"  margin > 0: {(margin_B > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1_acc*100:.2f}%")

    # === C: Mahalanobis on user_offsets only ===
    log(f"\n=== C: Mahalanobis on user_offsets only ===")
    # user_logvar is per-user-per-dim, shared (fixed logvar=-2)
    # 用 user_offsets as user_mu for Mahalanobis
    D_C = mahalanobis_diag_batch(mu_s, user_offsets, user_logvar)
    self_d = np.array([D_C[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D_C.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin_C = cross_d - self_d
    top1 = np.argmin(D_C, axis=1)
    top1_acc = (top1 == ho_uid_idx).mean()
    log(f"  D_self: mean={self_d.mean():.4f}, std={self_d.std():.4f}")
    log(f"  D_cross_min: mean={cross_d.mean():.4f}, std={cross_d.std():.4f}")
    log(f"  margin: mean={margin_C.mean():.4f}, std={margin_C.std():.4f}")
    log(f"  margin > 0: {(margin_C > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1_acc*100:.2f}%")

    # === D: Mahalanobis on user_mu (original) ===
    log(f"\n=== D: Mahalanobis on user_mu (baseline) ===")
    D_D = mahalanobis_diag_batch(mu_s, user_mu, user_logvar)
    self_d = np.array([D_D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D_D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin_D = cross_d - self_d
    top1 = np.argmin(D_D, axis=1)
    top1_acc = (top1 == ho_uid_idx).mean()
    log(f"  D_self: mean={self_d.mean():.4f}, std={self_d.std():.4f}")
    log(f"  D_cross_min: mean={cross_d.mean():.4f}, std={cross_d.std():.4f}")
    log(f"  margin: mean={margin_D.mean():.4f}, std={margin_D.std():.4f}")
    log(f"  margin > 0: {(margin_D > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1_acc*100:.2f}%")

    out = {
        "n_eval": n_eval,
        "A_cos_user_mu": {
            "margin_mean": float(margin_A.mean()),
            "margin_pos_rate": float((margin_A > 0).mean()),
            "top1_acc": float(top1_acc),
        },
        "B_cos_user_offsets": {
            "margin_mean": float(margin_B.mean()),
            "margin_pos_rate": float((margin_B > 0).mean()),
            "top1_acc": float((np.argmin(D_B, axis=1) == ho_uid_idx).mean()),
        },
        "C_mahal_user_offsets": {
            "margin_mean": float(margin_C.mean()),
            "margin_pos_rate": float((margin_C > 0).mean()),
            "top1_acc": float((np.argmin(D_C, axis=1) == ho_uid_idx).mean()),
        },
        "D_mahal_user_mu_baseline": {
            "margin_mean": float(margin_D.mean()),
            "margin_pos_rate": float((margin_D > 0).mean()),
            "top1_acc": float((np.argmin(D_D, axis=1) == ho_uid_idx).mean()),
        },
    }
    out_path = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter/diagnose_three_distances.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()