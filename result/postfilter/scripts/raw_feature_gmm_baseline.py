#!/usr/bin/env python3
"""空间一致性审计 + raw 20-d GMM baseline.

== Part 1: 空间一致性审计 ==
确认 user history / query / cluster center 是否同一空间。
读取 VADES 训练产物, 列出每个张量的 dim / mean / std / 来源。

== Part 2: raw 20-d GMM baseline ==
跳过 VADES encoder, 直接在 standardized raw 20-d 句法特征上:
- μ_u = 训练句子 20-d 均值
- Σ_u = 训练句子 20-d 协方差 (diagonal, identity 共测)
- 测试 holdout:
  - D_self(u, s) = (s - μ_u)^T Σ_u^-1 (s - μ_u)
  - margin = min_{v≠u} D(v, s) - D(u, s)
  - same-user top-1 accuracy
  - AUC (s classifier: closest user)
  - permutation p-value

输出:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/spatial_audit.json
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/raw_gmm_baseline.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian.gaussian_vades import (
    SentenceEncoder, UserDistributionTableDisentangled,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SENTENCE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_sentences.jsonl"
USER_PROFILE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_user_profiles.jsonl"
ENCODER_CKPT = VADES_DIR / "vades_encoder.pt"
USER_TABLE_CKPT = VADES_DIR / "vades_user_table.pt"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
SPATIAL_AUDIT = OUT_DIR / "spatial_audit.json"
RAW_GMM = OUT_DIR / "raw_gmm_baseline.json"

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


def part1_spatial_audit(log):
    log("=" * 70)
    log("Part 1: 空间一致性审计")
    log("=" * 70)

    # 加载 encoder + user_table
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_offsets"].shape[0])
    latent_dim = int(ut_sd["style_centers"].shape[1])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, latent_dim).to(DEVICE)
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=latent_dim,
        user_cluster_ids=placeholder_cluster, style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.eval()
    user_table.eval()

    # 加载训练句子的 raw 20-d 特征 + 用 scaler 标准化
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    train_rows = [r for r in rows if not r.get("is_holdout")]
    ho_rows = [r for r in rows if r.get("is_holdout")]
    train_feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in train_rows])
    scaler = StandardScaler().fit(train_feat)
    train_scaled = scaler.transform(train_feat)
    ho_scaled = scaler.transform(
        np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in ho_rows])
    )

    # 1a. raw 20-d 特征空间
    raw_mean = float(np.abs(train_scaled).mean())
    raw_std = float(train_scaled.std())
    audit = {
        "raw_20d": {
            "dim": 20,
            "scaler_mean_abs": float(np.abs(scaler.mean_).mean()),
            "scaler_scale_mean": float(np.abs(scaler.scale_).mean()),
            "after_scaling": {"mean_abs": raw_mean, "std": raw_std},
        },
    }

    # 1b. encoder 把 raw 20-d → latent 20-d
    log("Encoder forward on training sentences ...")
    with torch.no_grad():
        mu_latent, logvar_latent, _ = encoder(
            torch.as_tensor(train_scaled, dtype=torch.float32, device=DEVICE)
        )
        mu_latent_np = mu_latent.cpu().numpy()
        logvar_latent_np = logvar_latent.cpu().numpy()
    audit["encoder_output_latent_20d"] = {
        "dim": 20,
        "mu_mean_abs": float(np.abs(mu_latent_np).mean()),
        "mu_std": float(mu_latent_np.std()),
        "logvar_mean": float(logvar_latent_np.mean()),
        "logvar_std": float(logvar_latent_np.std()),
    }
    log(f"  encoder output mu: mean_abs={audit['encoder_output_latent_20d']['mu_mean_abs']:.3f}, "
        f"std={audit['encoder_output_latent_20d']['mu_std']:.3f}")

    # 1c. user_mu (latent 20-d) from user_table
    with torch.no_grad():
        all_idx = torch.arange(num_users)
        um, ul = user_table(all_idx)
        user_mu_np = um.cpu().numpy()
        user_logvar_np = ul.cpu().numpy()
    audit["user_mu_latent_20d"] = {
        "dim": 20,
        "mu_mean_abs": float(np.abs(user_mu_np).mean()),
        "mu_std": float(user_mu_np.std()),
        "logvar_mean": float(user_logvar_np.mean()),
        "logvar_std": float(user_logvar_np.std()),
    }
    log(f"  user_mu: mean_abs={audit['user_mu_latent_20d']['mu_mean_abs']:.3f}, "
        f"std={audit['user_mu_latent_20d']['mu_std']:.3f}")

    # 1d. style_centers (cluster centroid)
    sc_np = ut_sd["style_centers"].cpu().numpy()
    audit["style_centers_20d"] = {
        "n_clusters": n_clusters,
        "dim": 20,
        "mean_abs": float(np.abs(sc_np).mean()),
        "std": float(sc_np.std()),
    }
    log(f"  style_centers: {n_clusters} clusters, mean_abs={audit['style_centers_20d']['mean_abs']:.3f}, "
        f"std={audit['style_centers_20d']['std']:.3f}")

    # 1e. user_offsets
    uo_np = ut_sd["user_offsets"].cpu().numpy()
    audit["user_offsets_20d"] = {
        "n_users": num_users,
        "dim": 20,
        "mean_abs": float(np.abs(uo_np).mean()),
        "std": float(uo_np.std()),
    }
    log(f"  user_offsets: mean_abs={audit['user_offsets_20d']['mean_abs']:.3f}, "
        f"std={audit['user_offsets_20d']['std']:.3f}")

    # 1f. 关键诊断: scale mismatch
    log("")
    log("Scale mismatch diagnostic:")
    en_out_std = audit["encoder_output_latent_20d"]["mu_std"]
    user_mu_std = audit["user_mu_latent_20d"]["mu_std"]
    scale_ratio = user_mu_std / max(1e-9, en_out_std)
    audit["scale_mismatch"] = {
        "encoder_output_std": en_out_std,
        "user_mu_std": user_mu_std,
        "scale_ratio": float(scale_ratio),
        "verdict": (
            "严重 scale mismatch (user_mu 在空间外, encoder 把句子编码到另一个尺度)"
            if scale_ratio > 3.0 or scale_ratio < 0.33
            else "scale 接近"
        ),
    }
    log(f"  encoder output std: {en_out_std:.3f}")
    log(f"  user_mu std:        {user_mu_std:.3f}")
    log(f"  scale_ratio:        {scale_ratio:.3f}")
    log(f"  verdict:            {audit['scale_mismatch']['verdict']}")

    # 1g. user_mu 的 L2 距离 vs encoder output 的 L2 距离
    log("")
    log("L2 distance check (raw 20-d vs latent 20-d):")
    audit["raw_distance_scale"] = {
        "raw_M_pair_L2": float(np.linalg.norm(train_scaled[0] - train_scaled[1])),
        "raw_M_mean_pair_L2": float(np.mean([
            np.linalg.norm(train_scaled[i] - train_scaled[j])
            for i in range(min(100, len(train_scaled)))
            for j in range(i + 1, min(100, len(train_scaled)))
        ])),
    }
    log(f"  raw 20-d pair L2: {audit['raw_distance_scale']['raw_M_pair_L2']:.3f}")
    log(f"  raw 20-d mean pair L2 (100 samples): {audit['raw_distance_scale']['raw_M_mean_pair_L2']:.3f}")

    # user_mu pair L2
    user_mu_pair = float(np.linalg.norm(user_mu_np[0] - user_mu_np[1]))
    user_mu_mean_pair = float(np.mean([
        np.linalg.norm(user_mu_np[i] - user_mu_np[j])
        for i in range(min(100, num_users))
        for j in range(i + 1, min(100, num_users))
    ]))
    audit["user_mu_distance_scale"] = {
        "user_mu_pair_L2": user_mu_pair,
        "user_mu_mean_pair_L2": user_mu_mean_pair,
    }
    log(f"  user_mu pair L2: {user_mu_pair:.3f}")
    log(f"  user_mu mean pair L2 (100 samples): {user_mu_mean_pair:.3f}")

    SPATIAL_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    log(f"\n已写入 {SPATIAL_AUDIT}")
    return train_scaled, ho_scaled, train_rows, ho_rows, scaler


def part2_raw_gmm_baseline(train_scaled, ho_scaled, train_rows, ho_rows, scaler, log):
    log("=" * 70)
    log("Part 2: raw 20-d GMM baseline")
    log("=" * 70)

    # uid → idx
    uid_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            uid_to_idx[json.loads(line)["user_id"]] = i

    # 训练: 每用户 10 句子 → μ_u, Σ_u (diagonal)
    n_users = len(uid_to_idx)
    feat_dim = train_scaled.shape[1]
    mu_u = np.zeros((n_users, feat_dim))
    var_u = np.zeros((n_users, feat_dim))
    counts = np.zeros(n_users)
    for r, x in zip(train_rows, train_scaled):
        uidx = uid_to_idx[r["user_id"]]
        mu_u[uidx] += x
        counts[uidx] += 1
    mu_u = mu_u / counts[:, None]
    for r, x in zip(train_rows, train_scaled):
        uidx = uid_to_idx[r["user_id"]]
        var_u[uidx] += (x - mu_u[uidx]) ** 2
    var_u = var_u / counts[:, None]
    var_u = np.clip(var_u, 1e-6, None)
    log(f"  μ_u shape={mu_u.shape}, vary_u shape={var_u.shape}")
    log(f"  μ_u mean_abs={np.abs(mu_u).mean():.3f}, vary_u mean={var_u.mean():.3f}")

    # 测试 holdout
    log("Compute D_self and D_cross for holdout ...")
    n_ho = len(ho_scaled)
    # D[holdout, user]
    diff = ho_scaled[:, None, :] - mu_u[None, :, :]
    D = (diff ** 2 / var_u[None, :, :]).sum(axis=2)  # [N_ho, U]
    log(f"  D shape: {D.shape}, mean={D.mean():.3f}, std={D.std():.3f}")

    # same-user distance
    same_d = []
    for i, r in enumerate(ho_rows):
        uidx = uid_to_idx[r["user_id"]]
        same_d.append(D[i, uidx])
    same_d = np.array(same_d)

    # cross-user min distance
    cross_d = []
    for i, r in enumerate(ho_rows):
        uidx = uid_to_idx[r["user_id"]]
        d_rest = D[i].copy()
        d_rest[uidx] = np.inf
        cross_d.append(d_rest.min())
    cross_d = np.array(cross_d)

    margin = cross_d - same_d
    top1_correct = sum(
        1 for i, r in enumerate(ho_rows)
        if np.argmin(D[i]) == uid_to_idx[r["user_id"]]
    )
    top1_accuracy = top1_correct / n_ho

    log(f"  same-user D: mean={same_d.mean():.3f}, std={same_d.std():.3f}")
    log(f"  cross-user min D: mean={cross_d.mean():.3f}, std={cross_d.std():.3f}")
    log(f"  margin: mean={margin.mean():.3f}, std={margin.std():.3f}")
    log(f"  margin > 0: {(margin > 0).sum()}/{n_ho} = {(margin > 0).mean()*100:.1f}%")
    log(f"  top-1 user accuracy: {top1_accuracy:.4f} ({top1_correct}/{n_ho})")

    # AUC: classifier = which user is closest (one-vs-rest)
    # For each holdout, true_label = uidx, score = -D (lower distance = higher score)
    labels = np.array([uid_to_idx[r["user_id"]] for r in ho_rows])
    # AUC: for each holdout, is the true user among the closest N% users?
    # Use multi-class AUC if possible
    try:
        # One-vs-rest AUC averaged across users (only users with at least 1 holdout)
        from sklearn.metrics import roc_auc_score
        scores = -D  # higher = closer
        # binarize labels
        y_true_bin = np.zeros_like(D, dtype=int)
        for i, l in enumerate(labels):
            y_true_bin[i, l] = 1
        # macro average across users (with at least 1 positive)
        user_has_pos = y_true_bin.sum(axis=0) > 0
        aucs = []
        for u in range(n_users):
            if not user_has_pos[u]:
                continue
            try:
                a = roc_auc_score(y_true_bin[:, u], scores[:, u])
                aucs.append(a)
            except ValueError:
                pass
        macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    except Exception as e:
        log(f"  AUC 计算失败: {e!r}")
        macro_auc = float("nan")

    log(f"  macro AUC (one-vs-rest): {macro_auc:.4f}")

    # Permutation test: shuffle labels, recompute margin
    log("Permutation test (1000 random label shuffles) ...")
    rng = np.random.default_rng(42)
    n_perm = 1000
    obs_margin_mean = margin.mean()
    perm_means = np.zeros(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_same_d = np.array([D[i, perm_labels[i]] for i in range(n_ho)])
        perm_cross_d = np.array([D[i, np.concatenate([np.arange(uidx), np.arange(uidx + 1, n_users)]).min()] for i, uidx in enumerate(perm_labels)])
    # 简化: 直接每个 holdout 算了 cross
    perm_means = np.zeros(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_same_d = np.zeros(n_ho)
        perm_cross_d = np.zeros(n_ho)
        for i in range(n_ho):
            uidx = perm_labels[i]
            perm_same_d[i] = D[i, uidx]
            d_rest = D[i].copy()
            d_rest[uidx] = np.inf
            perm_cross_d[i] = d_rest.min()
        perm_means[p] = (perm_cross_d - perm_same_d).mean()
    p_value = (perm_means >= obs_margin_mean).mean()
    log(f"  observed margin mean: {obs_margin_mean:.3f}")
    log(f"  perm mean of margin: {perm_means.mean():.3f}, std: {perm_means.std():.3f}")
    log(f"  permutation p-value: {p_value:.4f}")

    # write
    out = {
        "n_users": n_users,
        "n_holdout": n_ho,
        "feat_dim": feat_dim,
        "mu_u_mean_abs": float(np.abs(mu_u).mean()),
        "var_u_mean": float(var_u.mean()),
        "same_user_d_mean": float(same_d.mean()),
        "same_user_d_std": float(same_d.std()),
        "cross_user_d_mean": float(cross_d.mean()),
        "cross_user_d_std": float(cross_d.std()),
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
        "margin_positive_rate": float((margin > 0).mean()),
        "top1_user_accuracy": float(top1_accuracy),
        "macro_auc_one_vs_rest": macro_auc,
        "permutation_p_value": float(p_value),
        "obs_margin_mean": float(obs_margin_mean),
        "perm_mean_mean": float(perm_means.mean()),
        "perm_std": float(perm_means.std()),
        "decision": (
            "raw 20-d GMM 可区分用户 → 风格信号足够, 重训 VADES"
            if (top1_accuracy > 0.5 and p_value < 0.05)
            else "raw 20-d 句法信号太弱, 单纯重训 VADES 难以解决, 需要更丰富的特征"
        ),
    }
    RAW_GMM.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n已写入 {RAW_GMM}")
    log(f"\n决策: {out['decision']}")
    return out


def main():
    log = lambda m: print(f"[audit] {m}", flush=True)
    train_scaled, ho_scaled, train_rows, ho_rows, scaler = part1_spatial_audit(log)
    out = part2_raw_gmm_baseline(train_scaled, ho_scaled, train_rows, ho_rows, scaler, log)


if __name__ == "__main__":
    main()
