#!/usr/bin/env python3
"""评估 52d 扩展特征能否区分用户.

输入: expanded_features_52d.jsonl
输出: expanded_features_eval.json

评测:
- per-user μ_u (在 train sentences 上)
- shared Σ (在所有 train sentences 上, 避免 per-user variance 爆炸)
- 测试 holdout sentences:
  - D_self, D_cross_min, margin
  - top-1 user accuracy
  - macro AUC (one-vs-rest)
  - permutation p-value
- 与 20-d baseline 对比
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result/personal_query/12_complexity_analysis_clause_features/Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_user_profiles.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
EXPANDED_FILE = OUT_DIR / "expanded_features_52d.jsonl"
RAW_BASELINE = OUT_DIR / "raw_gmm_baseline.json"
EVAL_OUT = OUT_DIR / "expanded_features_eval.json"


def load_features() -> tuple[list[dict], list[dict], list[str]]:
    rows = []
    with EXPANDED_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    train = [r for r in rows if not r["is_holdout"]]
    ho = [r for r in rows if r["is_holdout"]]
    feature_keys = sorted(rows[0]["expanded_features"].keys())
    return train, ho, feature_keys


def evaluate(train: list[dict], ho: list[dict], feature_keys: list[str],
             mode: str = "shared_sigma_l2") -> dict:
    log = lambda m: print(f"[eval52] {m}", flush=True)
    log(f"  mode: {mode}")
    n_users = 300

    # 拼装矩阵
    train_X = np.array([[r["expanded_features"][k] for k in feature_keys] for r in train])
    ho_X = np.array([[r["expanded_features"][k] for k in feature_keys] for r in ho])
    log(f"  train_X: {train_X.shape}, ho_X: {ho_X.shape}")

    # 标准化
    scaler = StandardScaler().fit(train_X)
    train_Xs = scaler.transform(train_X)
    ho_Xs = scaler.transform(ho_X)

    # user_id → idx
    uid_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            uid_to_idx[json.loads(line)["user_id"]] = i
    assert len(uid_to_idx) == n_users

    # per-user mean (in scaled space)
    mu_u = np.zeros((n_users, train_Xs.shape[1]))
    counts = np.zeros(n_users)
    for r, x in zip(train, train_Xs):
        uidx = uid_to_idx[r["user_id"]]
        mu_u[uidx] += x
        counts[uidx] += 1
    mu_u /= counts[:, None]
    log(f"  μ_u: mean_abs={np.abs(mu_u).mean():.3f}")

    # D matrix
    if mode == "shared_sigma_l2":
        # Mahalanobis with shared Σ
        sigma = train_Xs.var(axis=0).clip(min=1e-6)
        D = ((ho_Xs[:, None, :] - mu_u[None, :, :]) ** 2 / sigma[None, None, :]).sum(axis=2)
        metric_name = "Mahalanobis (shared Σ)"
    elif mode == "l2":
        # L2 distance
        D = ((ho_Xs[:, None, :] - mu_u[None, :, :]) ** 2).sum(axis=2)
        metric_name = "L2"
    elif mode == "cosine":
        # Cosine distance (1 - cos sim)
        ho_n = ho_Xs / (np.linalg.norm(ho_Xs, axis=1, keepdims=True) + 1e-9)
        mu_n = mu_u / (np.linalg.norm(mu_u, axis=1, keepdims=True) + 1e-9)
        cos_sim = ho_n @ mu_n.T
        D = 1 - cos_sim
        metric_name = "Cosine distance"
    else:
        raise ValueError(f"unknown mode: {mode}")

    log(f"  D shape: {D.shape}, mean={D.mean():.3f}, std={D.std():.3f}")

    # same-user
    labels = np.array([uid_to_idx[r["user_id"]] for r in ho])
    same_d = np.array([D[i, labels[i]] for i in range(len(ho))])
    # cross-user min
    cross_d = np.zeros(len(ho))
    for i in range(len(ho)):
        d_rest = D[i].copy()
        d_rest[labels[i]] = np.inf
        cross_d[i] = d_rest.min()
    margin = cross_d - same_d

    # top-1
    top1_preds = np.argmin(D, axis=1)
    top1_acc = float((top1_preds == labels).mean())

    # macro AUC
    y_true_bin = np.zeros_like(D, dtype=int)
    for i, l in enumerate(labels):
        y_true_bin[i, l] = 1
    scores = -D
    aucs = []
    for u in range(n_users):
        if y_true_bin[:, u].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true_bin[:, u], scores[:, u]))
        except ValueError:
            pass
    macro_auc = float(np.mean(aucs))

    # permutation test
    log(f"  permutation test (1000 shuffles) ...")
    rng = np.random.default_rng(42)
    n_ho = len(ho)
    obs_margin_mean = margin.mean()
    perm_means = np.zeros(1000)
    for p in range(1000):
        perm_labels = rng.permutation(labels)
        perm_same_d = np.array([D[i, perm_labels[i]] for i in range(n_ho)])
        perm_cross_d = np.zeros(n_ho)
        for i in range(n_ho):
            d_rest = D[i].copy()
            d_rest[perm_labels[i]] = np.inf
            perm_cross_d[i] = d_rest.min()
        perm_means[p] = (perm_cross_d - perm_same_d).mean()
    p_value = float((perm_means >= obs_margin_mean).mean())

    out = {
        "mode": mode,
        "metric": metric_name,
        "n_users": n_users,
        "n_holdout": n_ho,
        "feat_dim": len(feature_keys),
        "same_user_d_mean": float(same_d.mean()),
        "same_user_d_std": float(same_d.std()),
        "cross_user_d_mean": float(cross_d.mean()),
        "cross_user_d_std": float(cross_d.std()),
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
        "margin_positive_rate": float((margin > 0).mean()),
        "top1_user_accuracy": top1_acc,
        "top1_user_correct": int(top1_acc * n_ho),
        "macro_auc_one_vs_rest": macro_auc,
        "permutation_p_value": p_value,
        "obs_margin_mean": float(obs_margin_mean),
        "perm_mean_mean": float(perm_means.mean()),
        "perm_std": float(perm_means.std()),
    }
    log(f"  same-user D: mean={out['same_user_d_mean']:.3f}, std={out['same_user_d_std']:.3f}")
    log(f"  cross-user min D: mean={out['cross_user_d_mean']:.3f}, std={out['cross_user_d_std']:.3f}")
    log(f"  margin: mean={out['margin_mean']:.3f}, std={out['margin_std']:.3f}")
    log(f"  margin > 0: {out['margin_positive_rate']*100:.1f}%")
    log(f"  top-1 acc: {out['top1_user_accuracy']*100:.2f}% ({out['top1_user_correct']}/{n_ho})")
    log(f"  macro AUC: {out['macro_auc_one_vs_rest']:.4f}")
    log(f"  permutation p: {out['permutation_p_value']:.4f}")
    return out


def main():
    log = lambda m: print(f"[eval52] {m}", flush=True)
    log("=" * 70)
    log("评估 52d 扩展特征能否区分用户")
    log("=" * 70)
    train, ho, feature_keys = load_features()
    log(f"  train: {len(train)}, holdout: {len(ho)}, feat_dim: {len(feature_keys)}")

    results = {}
    for mode in ["shared_sigma_l2", "l2", "cosine"]:
        log(f"\n--- Mode: {mode} ---")
        results[mode] = evaluate(train, ho, feature_keys, mode=mode)

    # 对比 20-d baseline
    raw_baseline = json.loads(RAW_BASELINE.read_text())
    log(f"\n{'='*70}")
    log(f"对比 20-d baseline vs 52-d (3 metrics)")
    log(f"{'='*70}")
    log(f"{'metric':<30} {'20d':>10} {'52d_shared':>12} {'52d_l2':>10} {'52d_cos':>10}")
    log(f"{'top-1 accuracy':<30} {raw_baseline['top1_user_accuracy']*100:>9.2f}% "
        f"{results['shared_sigma_l2']['top1_user_accuracy']*100:>11.2f}% "
        f"{results['l2']['top1_user_accuracy']*100:>9.2f}% "
        f"{results['cosine']['top1_user_accuracy']*100:>9.2f}%")
    log(f"{'macro AUC':<30} {raw_baseline['macro_auc_one_vs_rest']:>10.4f} "
        f"{results['shared_sigma_l2']['macro_auc_one_vs_rest']:>12.4f} "
        f"{results['l2']['macro_auc_one_vs_rest']:>10.4f} "
        f"{results['cosine']['macro_auc_one_vs_rest']:>10.4f}")
    log(f"{'margin > 0 (%)':<30} {raw_baseline['margin_positive_rate']*100:>9.2f}% "
        f"{results['shared_sigma_l2']['margin_positive_rate']*100:>11.2f}% "
        f"{results['l2']['margin_positive_rate']*100:>9.2f}% "
        f"{results['cosine']['margin_positive_rate']*100:>9.2f}%")
    log(f"{'permutation p':<30} {raw_baseline['permutation_p_value']:>10.4f} "
        f"{results['shared_sigma_l2']['permutation_p_value']:>12.4f} "
        f"{results['l2']['permutation_p_value']:>10.4f} "
        f"{results['cosine']['permutation_p_value']:>10.4f}")

    log(f"\n决策: ")
    best_mode = max(results, key=lambda m: results[m]["macro_auc_one_vs_rest"])
    best_auc = results[best_mode]["macro_auc_one_vs_rest"]
    best_top1 = results[best_mode]["top1_user_accuracy"]
    log(f"  best 52d mode: {best_mode} (AUC={best_auc:.4f}, top-1={best_top1*100:.2f}%)")
    if best_auc > 0.6 and best_top1 > 0.05:
        log(f"  ✅ 52d 特征显著优于 20d (AUC {raw_baseline['macro_auc_one_vs_rest']:.4f} → {best_auc:.4f})")
        log(f"     可以考虑作为 VADES 重训的输入特征")
    elif best_auc > 0.55:
        log(f"  ⚠️  52d 略优于 20d, 但仍很弱 (AUC {raw_baseline['macro_auc_one_vs_rest']:.4f} → {best_auc:.4f})")
    else:
        log(f"  ❌ 52d 与 20d 几乎无差别, 句法特征空间已饱和")

    out = {
        "results": results,
        "best_mode": best_mode,
        "best_auc": best_auc,
        "best_top1": best_top1,
        "raw_baseline_20d": {
            "top1_user_accuracy": raw_baseline["top1_user_accuracy"],
            "macro_auc_one_vs_rest": raw_baseline["macro_auc_one_vs_rest"],
            "margin_positive_rate": raw_baseline["margin_positive_rate"],
        },
    }
    EVAL_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n已写入 {EVAL_OUT}")


if __name__ == "__main__":
    main()
