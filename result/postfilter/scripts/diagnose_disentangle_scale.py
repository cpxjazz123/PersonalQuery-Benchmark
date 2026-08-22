#!/usr/bin/env python3
"""诊断 disentangle user_mu 来源.

加载 3000u VADES 的 encoder + user_table, 检查:
- user_offsets 相对 style_centers 的量级
- encoder 输出 vs user_mu 的尺度比
- user_offsets 之间的 cosine 相似度 (是否足以区分用户)
- sentence latent vs user_mu 的 cosine 分布
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


def main():
    log = lambda m: print(f"[diag] {m}", flush=True)
    log("=" * 70)
    log("诊断 disentangle 架构的 scale mismatch 来源")
    log("=" * 70)

    encoder, user_table, ut_sd, n_clusters, num_users = load_models()

    user_offsets = ut_sd["user_offsets"].cpu().numpy()
    style_centers = ut_sd["style_centers"].cpu().numpy()
    log(f"  user_offsets shape: {user_offsets.shape}")
    log(f"  style_centers shape: {style_centers.shape}")

    # Scale of each component
    log(f"  style_centers std: {style_centers.std():.4f}")
    log(f"  user_offsets std: {user_offsets.std():.4f}")
    log(f"  ratio user_offsets/style_centers std: {user_offsets.std()/style_centers.std():.4f}")
    log(f"  user_offsets mean_abs: {np.abs(user_offsets).mean():.4f}")
    log(f"  style_centers mean_abs: {np.abs(style_centers).mean():.4f}")

    # user_mu = style_centers[cluster_u] + user_offsets[u]
    # placeholder_cluster[0] = n_clusters-1
    # 模拟: user_mu[i] = style_centers[cluster_i] + user_offsets[i]
    log(f"  (placeholder cluster: {n_clusters-1})")
    cluster_for_user = n_clusters - 1  # placeholder
    user_mu_sim = style_centers[cluster_for_user][None, :] + user_offsets
    log(f"  simulated user_mu std: {user_mu_sim.std():.4f}")
    log(f"  simulated user_mu mean_abs: {np.abs(user_mu_sim).mean():.4f}")

    # Per-cluster style_centers stats
    log(f"\n  per-cluster style_centers std:")
    for c in range(n_clusters):
        log(f"    cluster {c}: std={style_centers[c].std():.4f}, mean_abs={np.abs(style_centers[c]).mean():.4f}")

    # User offsets L2 norms
    user_offsets_norm = np.linalg.norm(user_offsets, axis=1)
    log(f"\n  user_offsets L2 norm: mean={user_offsets_norm.mean():.4f}, "
        f"std={user_offsets_norm.std():.4f}, "
        f"min={user_offsets_norm.min():.4f}, max={user_offsets_norm.max():.4f}")

    # Check if user_offsets alone can identify users (cosine)
    log(f"\n  === user_offsets 之间的 cosine 相似度 ===")
    # 标准化
    uo_n = user_offsets / (np.linalg.norm(user_offsets, axis=1, keepdims=True) + 1e-9)
    cos_sim_uo = uo_n @ uo_n.T
    # 取 off-diagonal
    mask = ~np.eye(num_users, dtype=bool)
    off_diag = cos_sim_uo[mask]
    log(f"  off-diagonal cos sim: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    log(f"  min: {off_diag.min():.4f}, max: {off_diag.max():.4f}")
    # Self vs cross-user similarity (random pair)
    diag = cos_sim_uo.diagonal()
    log(f"  self cos sim (always 1.0): mean={diag.mean():.4f}")
    log(f"  vs random pair cos sim: {off_diag.mean():.4f}")

    # 加载 sentence mu
    log(f"\n  === encoder 输出 vs user_mu ===")
    log(f"  加载 holdout ...")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    holdout = [r for r in rows if r.get("is_holdout")]
    log(f"  holdout: {len(holdout)}")

    mu_s = np.array([r["mu"] for r in holdout[:500]], dtype=np.float64)
    log(f"  sentence latent (mu_s): mean_abs={np.abs(mu_s).mean():.4f}, std={mu_s.std():.4f}")

    # All user_mu (use simulated for first 500 users as a check)
    user_mu_all = user_mu_sim
    mu_n = mu_s / (np.linalg.norm(mu_s, axis=1, keepdims=True) + 1e-9)
    um_n = user_mu_all / (np.linalg.norm(user_mu_all, axis=1, keepdims=True) + 1e-9)
    # Sentence cosine to each user
    cos_s_u = mu_n @ um_n.T  # [500, 2918]
    log(f"  sentence-user cosine sim: "
        f"min={cos_s_u.min():.4f}, max={cos_s_u.max():.4f}, "
        f"std={cos_s_u.std():.4f}")
    # Per-sentence: max cosine (closest user) and argmax
    argmax = np.argmax(cos_s_u, axis=1)
    log(f"  per-sentence closest user cosine: mean={cos_s_u.max(axis=1).mean():.4f}, "
        f"std={cos_s_u.max(axis=1).std():.4f}")

    # Load user_id mapping to check if closest = true user
    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i
    idx_to_uid = {v: k for k, v in user_id_to_idx.items()}

    ho_uid_idx = np.array([user_id_to_idx[r["user_id"]] for r in holdout[:500]])
    is_own = (argmax == ho_uid_idx)
    log(f"  closest user == true user: {is_own.sum()}/{len(ho_uid_idx)} = {is_own.mean()*100:.2f}%")

    # user_offsets-only: project sentence onto user_offsets direction (cosine)
    log(f"\n  === user_offsets-only cosine ===")
    cos_s_uo = mu_n @ uo_n.T  # [500, 2918]
    argmax_uo = np.argmax(cos_s_uo, axis=1)
    is_own_uo = (argmax_uo == ho_uid_idx)
    log(f"  user_offsets-only closest == true: {is_own_uo.sum()}/{len(ho_uid_idx)} = {is_own_uo.mean()*100:.2f}%")

    # Per-user held-out recall: of 500 holdout samples, how many are closest to their true user
    # Style match margin: for each holdout, is true user in top-5 closest?
    top5_self = []
    for i, l in enumerate(ho_uid_idx):
        top5 = np.argpartition(-cos_s_u[i], 5)[:5]
        top5_self.append(int(l in top5))
    log(f"  top-5 closest contains true user: {sum(top5_self)}/{len(ho_uid_idx)} = {np.mean(top5_self)*100:.2f}%")

    top5_uo = []
    for i, l in enumerate(ho_uid_idx):
        top5 = np.argpartition(-cos_s_uo[i], 5)[:5]
        top5_uo.append(int(l in top5))
    log(f"  (user_offsets-only) top-5 contains true: {sum(top5_uo)}/{len(ho_uid_idx)} = {np.mean(top5_uo)*100:.2f}%")

    # 写入结果
    out = {
        "n_users": num_users,
        "n_clusters": n_clusters,
        "user_offsets_std": float(user_offsets.std()),
        "style_centers_std": float(style_centers.std()),
        "ratio": float(user_offsets.std() / style_centers.std()),
        "user_offsets_l2_mean": float(user_offsets_norm.mean()),
        "user_offsets_l2_std": float(user_offsets_norm.std()),
        "user_offsets_cosine_off_diag_mean": float(off_diag.mean()),
        "user_offsets_cosine_off_diag_std": float(off_diag.std()),
        "sentence_to_user_cosine_mean_max": float(cos_s_u.max(axis=1).mean()),
        "sentence_to_user_cosine_mean_max_std": float(cos_s_u.max(axis=1).std()),
        "closest_user_match_rate_user_mu": float(is_own.mean()),
        "closest_user_match_rate_offsets_only": float(is_own_uo.mean()),
        "top5_match_rate_user_mu": float(np.mean(top5_self)),
        "top5_match_rate_offsets_only": float(np.mean(top5_uo)),
    }
    out_path = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter/diagnose_disentangle_scale.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()