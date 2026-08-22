#!/usr/bin/env python3
"""用用户留出文本校准 τ_u 和 margin m.

输入:
- VADES encoder + user_table (already trained)
- sentences.jsonl (含 is_holdout 标记, 10 train + 5 holdout per user)
- user_profiles.jsonl

输出:
- per_user_tau.json: {user_id: tau_u (Q95 of D_self distribution)}
- global_margin.json: {m, m_method, fpr_at_m, tpr_at_m, ...}
- calibration_summary.json: 总体统计

== 关键定义 ==
D_self(u, s) = (mu_s - user_mu_u)^T diag(Sigma_u)^-1 (mu_s - user_mu_u)
            = Σ_i (mu_s[i] - user_mu_u[i])^2 / exp(user_logvar_u[i])

τ_u = Q95(D_self(u, s_holdout))     # 用户本人 95% 留出文本落在这个范围内

margin(s, u) = min_{v≠u} D_self(v, s) - D_self(u, s)
              # 正数 = s 更像 u 不像别人

m = 选择使 Youden's J = TPR + (1 - FPR) 最大的 margin 阈值
  具体: 对所有 (u, s_holdout) 对, margin > 0 视为 true positive (style unique)
       选择最佳截断
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
SENTENCE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_sentences.jsonl"
USER_PROFILE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_user_profiles.jsonl"
ENCODER_CKPT = VADES_DIR / "vades_encoder.pt"
USER_TABLE_CKPT = VADES_DIR / "vades_user_table.pt"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
PER_USER_TAU = OUT_DIR / "per_user_tau.json"
GLOBAL_MARGIN = OUT_DIR / "global_margin.json"
CALIB_SUMMARY = OUT_DIR / "calibration_summary.json"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q95 for tau (per user)
TAU_QUANTILE = 0.95


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
    return encoder, user_table


def load_user_mu_logvar(user_table):
    """返回 user_mu [N, D], user_logvar [N, D] numpy 矩阵."""
    with torch.no_grad():
        all_idx = torch.arange(user_table.num_users, device=DEVICE)
        um, ul = user_table(all_idx)
        return um.cpu().numpy(), ul.cpu().numpy()


def mahalanobis_diag(mu_s, user_mu, user_logvar):
    """D_self(mu_s, user_u) = Σ_i (mu_s[i] - user_mu[i])^2 / exp(user_logvar[i])."""
    diff = mu_s - user_mu
    var = np.exp(user_logvar).clip(min=1e-6)
    return float(np.sum(diff ** 2 / var))


def main():
    log = lambda m: print(f"[calib] {m}", flush=True)
    log("=" * 70)
    log("用用户留出文本校准 τ_u 和 margin m")
    log("=" * 70)

    log("加载 encoder + user_table ...")
    encoder, user_table = load_models()
    user_mu, user_logvar = load_user_mu_logvar(user_table)
    n_users = user_mu.shape[0]
    log(f"  user_mu shape={user_mu.shape}, user_logvar shape={user_logvar.shape}")

    # user_id → idx
    user_id_to_idx: dict[str, int] = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            user_id_to_idx[row["user_id"]] = i
    log(f"  user_id_to_idx: {len(user_id_to_idx)}")

    # 加载 holdout 句子
    log("加载 holdout 句子 ...")
    holdout_sents: list[dict] = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            holdout_sents.append(row)
    holdout = [r for r in holdout_sents if r.get("is_holdout")]
    log(f"  total={len(holdout_sents)}, holdout={len(holdout)}")

    # 预计算所有 user 的 mu (300 × 20)
    # 对每个 holdout sentence, 拿其预计算的 mu (latent)
    n_holdout = len(holdout)

    log("Compute D_self for each (user, holdout) pair ...")
    # self_dist: dict[user_idx → list of D_self]
    self_dist: dict[int, list[float]] = {i: [] for i in range(n_users)}
    # 同时为每个 (u, s) 算 D_cross
    log("  precompute mu_matrix for cross-distance ...")
    # skip - 用直接对每个 sentence 算全部 user 的距离

    # 收集所有 (u_self, mu_s, dist_self) 数据
    pairs: list[dict] = []
    for sh in holdout:
        uid = sh["user_id"]
        u_idx = user_id_to_idx[uid]
        # 用 sentence 预计算的 mu (latent), 来自训练时的 encoder
        mu_s = np.asarray(sh["mu"], dtype=np.float64)
        d_self = mahalanobis_diag(mu_s, user_mu[u_idx], user_logvar[u_idx])
        self_dist[u_idx].append(d_self)
        pairs.append({"uidx": u_idx, "mu_s": mu_s, "d_self": d_self})

    log(f"  {len(pairs)} pairs computed")

    # Per-user tau_u = Q95(D_self)
    log("Compute per-user τ_u = Q95(D_self) ...")
    per_user_tau: dict[str, float] = {}
    per_user_n: dict[str, int] = {}
    per_user_mean_self: dict[str, float] = {}
    for u_idx, dists in self_dist.items():
        if not dists:
            continue
        uid = [k for k, v in user_id_to_idx.items() if v == u_idx][0]
        per_user_tau[uid] = float(np.quantile(dists, TAU_QUANTILE))
        per_user_n[uid] = len(dists)
        per_user_mean_self[uid] = float(np.mean(dists))
    log(f"  τ_u: mean={np.mean(list(per_user_tau.values())):.2f}, "
        f"min={min(per_user_tau.values()):.2f}, max={max(per_user_tau.values()):.2f}")

    # Compute min_{v≠u} D(v, s) for each pair → margin
    log("Compute margin (D_cross - D_self) for each pair ...")
    self_arr = np.array([p["d_self"] for p in pairs])  # [N_holdout]
    # cross_dist: [N_holdout, N_users]
    mu_s_mat = np.array([p["mu_s"] for p in pairs])  # [N_holdout, 20]
    # D(v, s) for all (s, v) pairs
    diff = mu_s_mat[:, None, :] - user_mu[None, :, :]  # [N_holdout, N_users, 20]
    var = np.exp(user_logvar).clip(min=1e-6)  # [N_users, 20]
    cross_dist = (diff ** 2 / var[None, :, :]).sum(axis=2)  # [N_holdout, N_users]
    # mask self: 对每个 sample, 把 self 距离设为 inf
    for i, p in enumerate(pairs):
        cross_dist[i, p["uidx"]] = np.inf
    d_cross_min = cross_dist.min(axis=1)  # [N_holdout]
    margin = d_cross_min - self_arr  # [N_holdout]
    log(f"  margin: mean={margin.mean():.2f}, std={margin.std():.2f}, "
        f"min={margin.min():.2f}, max={margin.max():.2f}")
    log(f"  margin > 0: {(margin > 0).sum()}/{len(margin)} = {(margin > 0).mean()*100:.1f}%")

    # ROC analysis: label = 1 if this user is the unique closest (positive: unique),
    # 选择最佳 margin 阈值
    log("ROC analysis on margin ...")
    # ground truth: 是否 s 唯一最像 u (即 margin > 0)
    y_true = (margin > 0).astype(np.int32)
    # scoring: margin 越大越像 unique
    # 在所有候选 margin 阈值上计算 TPR / FPR
    cand_thresholds = np.unique(np.concatenate([
        margin, np.linspace(margin.min() - 1, margin.max() + 1, 1000),
    ]))
    tpr_list, fpr_list = [], []
    for t in cand_thresholds:
        y_pred = (margin >= t).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    youden = tpr_arr + (1 - fpr_arr) - 1
    best_idx = int(np.argmax(youden))
    m_best = float(cand_thresholds[best_idx])
    tpr_best = float(tpr_arr[best_idx])
    fpr_best = float(fpr_arr[best_idx])
    log(f"  best margin m (Youden J) = {m_best:.3f}")
    log(f"    at m: TPR={tpr_best:.3f}, FPR={fpr_best:.3f}, J={youden[best_idx]:.3f}")

    # F1-best
    f1_list = []
    for t in cand_thresholds:
        y_pred = (margin >= t).astype(np.int32)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1_list.append(f1)
    f1_arr = np.array(f1_list)
    f1_best_idx = int(np.argmax(f1_arr))
    m_f1 = float(cand_thresholds[f1_best_idx])
    log(f"  best margin m (F1) = {m_f1:.3f}")
    log(f"    at m: TPR={tpr_arr[f1_best_idx]:.3f}, FPR={fpr_arr[f1_best_idx]:.3f}, F1={f1_arr[f1_best_idx]:.3f}")

    # 95% same-user recall
    target_tpr = 0.95
    idxs_above = np.where(tpr_arr >= target_tpr)[0]
    if len(idxs_above) > 0:
        m_95 = float(cand_thresholds[idxs_above[0]])
    else:
        m_95 = float(margin.min())
    log(f"  margin m (95% TPR) = {m_95:.3f}")

    # 写产物
    PER_USER_TAU.write_text(json.dumps({
        "tau_quantile": TAU_QUANTILE,
        "tau_per_user": per_user_tau,
        "n_holdout_per_user": per_user_n,
        "mean_self_per_user": per_user_mean_self,
    }, indent=2, ensure_ascii=False))
    log(f"已写入 {PER_USER_TAU}")

    GLOBAL_MARGIN.write_text(json.dumps({
        "n_pairs": len(pairs),
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
        "margin_min": float(margin.min()),
        "margin_max": float(margin.max()),
        "margin_positive_rate": float((margin > 0).mean()),
        "m_youden": m_best,
        "m_youden_tpr": tpr_best,
        "m_youden_fpr": fpr_best,
        "m_f1": m_f1,
        "m_f1_value": float(f1_arr[f1_best_idx]),
        "m_95_tpr": m_95,
    }, indent=2, ensure_ascii=False))
    log(f"已写入 {GLOBAL_MARGIN}")

    # summary
    summary = {
        "n_users": n_users,
        "n_holdout_pairs": len(pairs),
        "tau_quantile": TAU_QUANTILE,
        "tau_mean": float(np.mean(list(per_user_tau.values()))),
        "tau_median": float(np.median(list(per_user_tau.values()))),
        "tau_min": float(min(per_user_tau.values())),
        "tau_max": float(max(per_user_tau.values())),
        "d_self_mean": float(self_arr.mean()),
        "d_self_std": float(self_arr.std()),
        "d_cross_min_mean": float(d_cross_min.mean()),
        "margin_mean": float(margin.mean()),
        "margin_positive_rate": float((margin > 0).mean()),
        "m_youden": m_best,
        "m_f1": m_f1,
        "m_95_tpr": m_95,
    }
    CALIB_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"已写入 {CALIB_SUMMARY}")

    log("=" * 70)
    log("校准结果")
    log("=" * 70)
    log(f"  τ_u (Q95 D_self): mean={summary['tau_mean']:.2f}, "
        f"median={summary['tau_median']:.2f}, range=[{summary['tau_min']:.2f}, {summary['tau_max']:.2f}]")
    log(f"  D_self distribution: mean={summary['d_self_mean']:.2f}, std={summary['d_self_std']:.2f}")
    log(f"  D_cross_min distribution: mean={summary['d_cross_min_mean']:.2f}")
    log(f"  margin (D_cross_min - D_self): mean={summary['margin_mean']:.2f}, "
        f"%>0={summary['margin_positive_rate']*100:.1f}%")
    log(f"  m (Youden J) = {m_best:.3f}  (TPR={tpr_best:.3f}, FPR={fpr_best:.3f})")
    log(f"  m (F1 best) = {m_f1:.3f}  (F1={f1_arr[f1_best_idx]:.3f})")
    log(f"  m (95% TPR) = {m_95:.3f}")


if __name__ == "__main__":
    main()
