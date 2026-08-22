#!/usr/bin/env python3
"""快速测试: 用训练好的 diagonal VADES 3000u 模型, 评估 encoder 是否学到 user 信息."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian.gaussian_vades import (
    SentenceEncoder, UserDistributionTable,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_diagonal_3000u"
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
    num_users = int(ut_sd["user_mu.weight"].shape[0])
    latent_dim = int(ut_sd["user_mu.weight"].shape[1])
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, latent_dim)
    user_table = UserDistributionTable(num_users, latent_dim)
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()
    return encoder, user_table, num_users


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_n @ b_n.T


def main():
    log = lambda m: print(f"[diagdiag] {m}", flush=True)
    log("=" * 70)
    log("Diagonal VADES 3000u 评估")
    log("=" * 70)

    encoder, user_table, num_users = load_models()

    user_mu = user_table.user_mu.weight.detach().cpu().numpy()
    user_logvar = user_table.user_logvar.weight.detach().cpu().numpy()
    log(f"  user_mu std: {user_mu.std():.4f}, user_logvar mean: {user_logvar.mean():.4f}")
    log(f"  user_mu mean_abs: {np.abs(user_mu).mean():.4f}")

    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    holdout = [r for r in rows if r.get("is_holdout")]
    n_eval = min(2000, len(holdout))
    holdout = holdout[:n_eval]
    mu_s = np.array([r["mu"] for r in holdout], dtype=np.float64)
    log(f"  mu_s std: {mu_s.std():.4f}, mean_abs: {np.abs(mu_s).mean():.4f}")

    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i
    ho_uid_idx = np.array([user_id_to_idx[r["user_id"]] for r in holdout])

    # === Mahalanobis (with diag) ===
    log(f"\n=== Mahalanobis (encoder latent + user_table) ===")
    var = np.exp(user_logvar).clip(min=1e-6)
    diff = mu_s[:, None, :] - user_mu[None, :, :]
    D = (diff ** 2 / var[None, :, :]).sum(axis=2)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    top1 = (np.argmin(D, axis=1) == ho_uid_idx).mean()
    log(f"  D_self: mean={self_d.mean():.4f}, D_cross_min: mean={cross_d.mean():.4f}")
    log(f"  margin mean={margin.mean():.4f}, margin > 0: {(margin > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1*100:.2f}%")

    # === Cosine on user_mu ===
    log(f"\n=== Cosine on user_mu ===")
    D = cosine_dist(mu_s, user_mu)
    self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
    d_rest = D.copy()
    for i in range(n_eval):
        d_rest[i, ho_uid_idx[i]] = np.inf
    cross_d = d_rest.min(axis=1)
    margin = cross_d - self_d
    top1 = (np.argmin(D, axis=1) == ho_uid_idx).mean()
    log(f"  margin mean={margin.mean():.4f}, margin > 0: {(margin > 0).mean()*100:.2f}%")
    log(f"  top-1 acc: {top1*100:.2f}%")

    # === Cosine sentence latent vs user_mu (correlation test) ===
    log(f"\n=== Direction check ===")
    # Per sentence: max cos sim, min L2 dist, top-1 user
    log(f"  user_offsets-cos-off-diag:")
    uo_n = user_mu / (np.linalg.norm(user_mu, axis=1, keepdims=True) + 1e-9)
    off_diag = uo_n @ uo_n.T
    mask = ~np.eye(num_users, dtype=bool)
    log(f"    mean={off_diag[mask].mean():.4f}, std={off_diag[mask].std():.4f}")


if __name__ == "__main__":
    main()