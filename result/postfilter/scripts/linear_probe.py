#!/usr/bin/env python3
"""训练线性分类器从 encoder latent → user_id, 测试 encoder 是否保留可学的 user 信号.

如果 encoder 真没 user info, 线性分类器 acc ≈ random (1/2918).
如果有可学信号, acc > random.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]


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
    return encoder, num_users


def main():
    log = lambda m: print(f"[probe] {m}", flush=True)
    log("=" * 70)
    log("训练线性分类器: encoder latent → user_id")
    log("=" * 70)

    encoder, num_users = load_models()
    log(f"  num_users: {num_users}, random baseline: {1/num_users*100:.4f}%")

    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))

    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i

    # 标准化
    train_rows = [r for r in rows if not r.get("is_holdout")]
    ho_rows = [r for r in rows if r.get("is_holdout")]
    train_feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in train_rows])
    scaler = StandardScaler().fit(train_feat)

    train_X = scaler.transform(train_feat).astype(np.float32)
    train_y = np.array([user_id_to_idx[r["user_id"]] for r in train_rows], dtype=np.int64)
    ho_X = scaler.transform(np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in ho_rows])).astype(np.float32)
    ho_y = np.array([user_id_to_idx[r["user_id"]] for r in ho_rows], dtype=np.int64)
    log(f"  train: {len(train_X)}, holdout: {len(ho_X)}")

    # Encoder forward
    log("  Encoder forward ...")
    with torch.no_grad():
        mu_train, _, _ = encoder(torch.as_tensor(train_X, device=DEVICE))
        mu_ho, _, _ = encoder(torch.as_tensor(ho_X, device=DEVICE))
        mu_train = mu_train.cpu().numpy()
        mu_ho = mu_ho.cpu().numpy()
    log(f"  mu_train: {mu_train.shape}, std={mu_train.std():.4f}")
    log(f"  mu_ho: {mu_ho.shape}, std={mu_ho.std():.4f}")

    # 训练线性分类器 (从 latent 20 → user logits)
    log("\n  === Probe 1: Linear classifier from encoder latent (20-d) ===")
    probe = nn.Linear(LATENT_DIM, num_users).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    train_X_t = torch.as_tensor(mu_train, device=DEVICE)
    train_y_t = torch.as_tensor(train_y, device=DEVICE)
    ho_X_t = torch.as_tensor(mu_ho, device=DEVICE)
    ho_y_t = torch.as_tensor(ho_y, device=DEVICE)

    for epoch in range(1, 21):
        probe.train()
        # mini-batch
        perm = torch.randperm(len(train_X_t))
        loss_sum = 0
        for i in range(0, len(perm), 1024):
            idx = perm[i:i+1024]
            logits = probe(train_X_t[idx])
            loss = F.cross_entropy(logits, train_y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        train_loss = loss_sum / len(train_X_t)
        if epoch % 5 == 0 or epoch == 1:
            probe.eval()
            with torch.no_grad():
                train_logits = probe(train_X_t)
                train_acc = (train_logits.argmax(1) == train_y_t).float().mean().item()
                ho_logits = probe(ho_X_t)
                ho_acc = (ho_logits.argmax(1) == ho_y_t).float().mean().item()
            log(f"    epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, ho_acc={ho_acc*100:.4f}%")

    # Probe 2: 同样从 raw 20-d (标准化后)
    log(f"\n  === Probe 2: Linear classifier from raw 20-d features ===")
    probe2 = nn.Linear(INPUT_DIM, num_users).to(DEVICE)
    optimizer2 = torch.optim.Adam(probe2.parameters(), lr=1e-3, weight_decay=1e-4)

    train_X_raw_t = torch.as_tensor(train_X, device=DEVICE)
    ho_X_raw_t = torch.as_tensor(ho_X, device=DEVICE)

    for epoch in range(1, 21):
        probe2.train()
        perm = torch.randperm(len(train_X_raw_t))
        loss_sum = 0
        for i in range(0, len(perm), 1024):
            idx = perm[i:i+1024]
            logits = probe2(train_X_raw_t[idx])
            loss = F.cross_entropy(logits, train_y_t[idx])
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            loss_sum += loss.item() * len(idx)
        train_loss = loss_sum / len(train_X_raw_t)
        if epoch % 5 == 0 or epoch == 1:
            probe2.eval()
            with torch.no_grad():
                train_logits = probe2(train_X_raw_t)
                train_acc = (train_logits.argmax(1) == train_y_t).float().mean().item()
                ho_logits = probe2(ho_X_raw_t)
                ho_acc = (ho_logits.argmax(1) == ho_y_t).float().mean().item()
            log(f"    epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, ho_acc={ho_acc*100:.4f}%")


if __name__ == "__main__":
    main()