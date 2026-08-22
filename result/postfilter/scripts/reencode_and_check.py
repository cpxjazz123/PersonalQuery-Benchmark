#!/usr/bin/env python3
"""用训练好的 encoder 重新编码 raw 20-d 句法特征, 检查 encoder 是否编码 user 信息.

输入: raw 20-d features (从 sentences.jsonl 的 features 字段)
输出: 新 mu + 对比 saved mu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
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
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_n @ b_n.T


def main():
    log = lambda m: print(f"[reenc] {m}", flush=True)
    log("=" * 70)
    log("用训练好的 encoder 重新编码 raw 20-d 句法特征")
    log("=" * 70)

    encoder, user_table, ut_sd, n_clusters, num_users = load_models()
    user_offsets = ut_sd["user_offsets"].cpu().numpy()
    style_centers = ut_sd["style_centers"].cpu().numpy()
    ucid = ut_sd["user_cluster_ids"].cpu().numpy()
    user_mu = style_centers[ucid] + user_offsets

    log(f" 加载 sentences ...")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    holdout = [r for r in rows if r.get("is_holdout")]

    n_eval = min(1000, len(holdout))
    holdout = holdout[:n_eval]

    # 用 saved raw 20-d features
    raw_20d = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in holdout], dtype=np.float64)

    # Standardize (same as training)
    all_rows_train = [r for r in rows if not r.get("is_holdout")]
    all_train_feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in all_rows_train])
    scaler = StandardScaler().fit(all_train_feat)
    raw_20d_scaled = scaler.transform(raw_20d)
    log(f"  raw_20d_scaled: mean_abs={np.abs(raw_20d_scaled).mean():.4f}, std={raw_20d_scaled.std():.4f}")

    # Re-encode
    log(" Encoder forward (eval mode, no grad) ...")
    with torch.no_grad():
        mu_re, logvar_re, _ = encoder(torch.as_tensor(raw_20d_scaled, dtype=torch.float32, device=DEVICE))
        mu_re = mu_re.cpu().numpy()
    log(f"  re-encoded mu: mean_abs={np.abs(mu_re).mean():.4f}, std={mu_re.std():.4f}")

    # Saved mu (from training)
    mu_saved = np.array([r["mu"] for r in holdout], dtype=np.float64)
    log(f"  saved mu: mean_abs={np.abs(mu_saved).mean():.4f}, std={mu_saved.std():.4f}")

    # Cosine sim between re-encoded and saved (should be ~1)
    cos_re_saved = np.diag(cosine_dist(mu_re, mu_saved))
    log(f"  cosine(re, saved): mean={cos_re_saved.mean():.6f}, std={cos_re_saved.std():.6f}")

    # user_id mapping
    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i
    ho_uid_idx = np.array([user_id_to_idx[r["user_id"]] for r in holdout])

    # Use re-encoded mu for top-1 accuracy
    log("=== 用 re-encoded mu 评估 top-1 ===")
    for label, user_rep, name in [
        ("user_mu", user_mu, "Cosine on user_mu"),
        ("user_offsets", user_offsets, "Cosine on user_offsets only"),
    ]:
        D = cosine_dist(mu_re, user_rep)
        top1 = np.argmin(D, axis=1)
        top1_acc = (top1 == ho_uid_idx).mean()
        self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
        d_rest = D.copy()
        for i in range(n_eval):
            d_rest[i, ho_uid_idx[i]] = np.inf
        cross_d = d_rest.min(axis=1)
        margin = cross_d - self_d
        log(f"  {name}: top-1={top1_acc*100:.2f}%, margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")

    # Use raw_20d_scaled directly (no encoder) for top-1
    log("=== 用 raw_20d_scaled 直接评估 (无 encoder) ===")
    for label, user_rep, name in [
        ("user_mu", user_mu, "Cosine raw → user_mu"),
    ]:
        D = cosine_dist(raw_20d_scaled, user_rep)
        top1 = np.argmin(D, axis=1)
        top1_acc = (top1 == ho_uid_idx).mean()
        self_d = np.array([D[i, ho_uid_idx[i]] for i in range(n_eval)])
        d_rest = D.copy()
        for i in range(n_eval):
            d_rest[i, ho_uid_idx[i]] = np.inf
        cross_d = d_rest.min(axis=1)
        margin = cross_d - self_d
        log(f"  {name}: top-1={top1_acc*100:.2f}%, margin > 0: {(margin > 0).mean()*100:.2f}%, margin mean={margin.mean():.4f}")


if __name__ == "__main__":
    main()