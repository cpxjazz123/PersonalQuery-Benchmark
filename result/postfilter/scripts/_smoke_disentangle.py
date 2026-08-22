#!/usr/bin/env python3
"""Smoke test: UserDistributionTableDisentangled forward + backward + loss."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian_vades import (
    UserDistributionTableDisentangled,
    SentenceEncoder,
    _compute_losses_for_batch,
    standard_normal_kl,
    diagonal_gaussian_kl,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[smoke] device: {DEVICE}")

NUM_USERS = 16
LATENT_DIM = 20
INPUT_DIM = 20
HIDDEN_DIM = 64
N_CLUSTERS = 4

# 1) 建 user cluster ids + style anchors
np.random.seed(0)
cluster_ids_np = np.random.randint(0, N_CLUSTERS, size=NUM_USERS).astype(np.int64)
anchors_np = np.random.randn(N_CLUSTERS, LATENT_DIM).astype(np.float32)
print(f"[smoke] cluster distribution: {np.bincount(cluster_ids_np, minlength=N_CLUSTERS)}")

user_table = UserDistributionTableDisentangled(
    NUM_USERS, LATENT_DIM,
    user_cluster_ids=torch.from_numpy(cluster_ids_np),
    style_anchors=torch.from_numpy(anchors_np),
).to(DEVICE)
print(f"[smoke] user_table created: n_clusters={user_table.n_clusters}, "
      f"style_centers.shape={user_table.style_centers.shape}, "
      f"user_offsets.shape={user_table.user_offsets.shape}")

# 2) Forward
all_user_idx = torch.arange(NUM_USERS, device=DEVICE)
user_mu, user_logvar = user_table(all_user_idx)
print(f"[smoke] forward: user_mu.shape={user_mu.shape}, user_logvar.shape={user_logvar.shape}")

# 验证 mu = style_center[cluster] + user_offset
sc = user_table.style_centers[user_table.user_cluster_ids]
off = user_table.user_offsets
expected_mu = sc + off
err = (user_mu - expected_mu).abs().max().item()
print(f"[smoke] mu = style + offset residual: {err:.2e}")
assert err < 1e-5, "disentangle forward 不一致"

# 3) encoder forward
encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
feats = torch.randn(64, INPUT_DIM, device=DEVICE)
mu, logvar, recon = encoder(feats)
print(f"[smoke] encoder: mu.shape={mu.shape}, logvar.shape={logvar.shape}, recon.shape={recon.shape}")

# 4) 模拟 _compute_losses_for_batch 的 disentangle 分支
batch_idx = torch.arange(64, device=DEVICE)
user_idx = torch.randint(0, NUM_USERS, (64,), device=DEVICE)
loss_dict = _compute_losses_for_batch(
    batch_idx=batch_idx,
    feature_tensor=feats,
    user_index_tensor=user_idx,
    num_users=NUM_USERS,
    encoder=encoder,
    user_table=user_table,
    feature_matrix_raw=feats.detach().cpu().numpy(),
    encoder_dist="gaussian",
    covariance_mode="diagonal_disentangled",
    user_match_weight=1.0,
    style_recon_weight=0.8,
    sent_kl_weight=0.05,
    user_prior_kl_weight=0.02,
    latent_align_weight=1.0,
)
print(f"[smoke] loss_dict keys: {list(loss_dict.keys())}")
print(f"  loss               = {loss_dict['loss'].item():.4f}")
print(f"  user_match_loss    = {loss_dict['user_match_loss'].item():.4f}")
print(f"  recon_loss         = {loss_dict['recon_loss'].item():.4f}")
print(f"  sent_kl            = {loss_dict['sent_kl'].item():.4f}")
print(f"  user_prior_kl      = {loss_dict['user_prior_kl'].item():.4f}")
print(f"  latent_align       = {loss_dict['latent_align'].item():.4f}")
print(f"  style_distinct_loss= {loss_dict['style_distinct_loss'].item():.4f}")
print(f"  style_anchor_loss  = {loss_dict['style_anchor_loss'].item():.4f}")

# 5) Backward
loss_dict["loss"].backward()
has_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
has_grad_ut = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in user_table.parameters())
print(f"[smoke] encoder params with grad: {has_grad}/{len(list(encoder.parameters()))}")
print(f"[smoke] user_table params with grad: {has_grad_ut}/{len(list(user_table.parameters()))}")

print("\n[ALL PASS] diagonal_disentangled forward+backward 正常")