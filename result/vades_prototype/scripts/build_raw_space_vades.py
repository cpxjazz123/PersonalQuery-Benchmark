#!/usr/bin/env python3
"""Raw-space VADES: 直接用 20d raw syntactic features 拟合用户高斯分布.

按用户最后指令 (v6 user-level 评估失败后的 fallback):
  - 不引入神经网络 encoder
  - 不引入 user_table learnable params
  - 直接计算 user_mu = mean(user's raw features), user_sigma = Ledoit-Wolf shrunk cov
  - 用全量 train 句产出 user_mu + user_log_scale 矩阵, 兼容下游 user_mu / user_logvar 属性
  - 把"vades prototype"框架彻底简化为"用户在 raw 20d 空间的 Gaussian 分布估计"

输入: v6 sentences.jsonl (2918 用户, 43770 句, 20d features, 14590 holdout)
输出:
  - {OUTPUT_TAG}_user_profiles.jsonl: 每用户 {user_id, user_mu (20d), user_logvar (20d)}
  - {OUTPUT_TAG}_sentences.jsonl: 每句 {user_id, features (dict), mu (20d), is_holdout}  (兼容格式)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_prototype_3000u_v6_raw"
SRC_SENTENCES = VADES_DIR / "vades_prototype_3000u_v6_sentences.jsonl"
OUT_USER_PROFILES = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
OUT_SENTENCES = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"

RAW_DIM = 20
LW_SHRINKAGE_FLOOR = 1e-4  # 防止 singular


def main():
    log = lambda m: print(f"[raw-space] {m}", flush=True)

    log("=" * 70)
    log(f"Raw-space VADES: {OUTPUT_TAG}")
    log("=" * 70)

    # === 1. Load sentences ===
    log(f"读取 {SRC_SENTENCES}")
    rows = []
    with SRC_SENTENCES.open() as f:
        for line in f:
            rows.append(json.loads(line))
    log(f"  n_sentences: {len(rows)}")

    # === 2. Build user_id → idx mapping (按出现顺序) ===
    user_id_to_idx: dict[str, int] = {}
    user_ids_order: list[str] = []
    for r in rows:
        u = r["user_id"]
        if u not in user_id_to_idx:
            user_id_to_idx[u] = len(user_ids_order)
            user_ids_order.append(u)
    num_users = len(user_ids_order)
    log(f"  num_users: {num_users}")

    # === 3. Build per-user feature arrays (train + holdout 一起用于估计 mu/sigma) ===
    feature_names = list(rows[0]["features"].keys())
    log(f"  feature_names ({len(feature_names)} dims): {feature_names[:5]}...")

    user_to_train_idx: dict[int, list[int]] = {}
    user_to_holdout_idx: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        u = user_id_to_idx[r["user_id"]]
        if r.get("is_holdout", False):
            user_to_holdout_idx.setdefault(u, []).append(i)
        else:
            user_to_train_idx.setdefault(u, []).append(i)

    log(f"  train sentences: {sum(len(v) for v in user_to_train_idx.values())}")
    log(f"  holdout sentences: {sum(len(v) for v in user_to_holdout_idx.values())}")

    # === 4. 计算每用户 raw_user_mu + Ledoit-Wolf shrunk user_sigma ===
    log("计算每用户 raw_user_mu + Ledoit-Wolf shrunk covariance...")
    user_mu = np.zeros((num_users, RAW_DIM), dtype=np.float32)
    user_logvar = np.zeros((num_users, RAW_DIM), dtype=np.float32)  # diagonal only (保持兼容下游)
    user_log_scale_diag = np.zeros((num_users, RAW_DIM), dtype=np.float32)

    feat_array = np.stack([
        np.array([float(r["features"][name]) for name in feature_names], dtype=np.float32)
        for r in rows
    ], axis=0)

    n_estimated = 0
    n_singular_fallback = 0
    for u in range(num_users):
        train_idx = user_to_train_idx.get(u, [])
        if not train_idx:
            # 没 train 句: 用 holdout 兜底
            train_idx = user_to_holdout_idx.get(u, [])
        if not train_idx:
            # 没任何句: 跳过 (但 num_users 不变, 占位 0)
            continue
        feats_u = feat_array[train_idx]  # [n_u, 20]
        # === raw user mu = mean(features) ===
        user_mu[u] = feats_u.mean(axis=0)
        # === Ledoit-Wolf shrunk covariance (full 20x20), 取 diagonal log-variance ===
        if feats_u.shape[0] >= 2:
            try:
                lw = LedoitWolf().fit(feats_u)
                cov = lw.covariance_
                # 加入 shrinkage floor 防止 singular
                cov = cov + np.eye(RAW_DIM, dtype=np.float32) * LW_SHRINKAGE_FLOOR
                diag = np.diag(cov).astype(np.float32)
                diag = np.clip(diag, 1e-6, None)  # 防 0
                user_logvar[u] = np.log(diag)
                user_log_scale_diag[u] = 0.5 * user_logvar[u]
                n_estimated += 1
            except Exception as e:
                # Fallback to per-dim variance
                var = feats_u.var(axis=0) + LW_SHRINKAGE_FLOOR
                user_logvar[u] = np.log(np.clip(var, 1e-6, None))
                user_log_scale_diag[u] = 0.5 * user_logvar[u]
                n_singular_fallback += 1
        else:
            # 单句用户: 用 0.5² 默认方差
            var = np.ones(RAW_DIM, dtype=np.float32) * 0.25
            user_logvar[u] = np.log(var)
            user_log_scale_diag[u] = 0.5 * user_logvar[u]
            n_singular_fallback += 1

    log(f"  estimated user distributions: {n_estimated}, fallback: {n_singular_fallback}")
    log(f"  user_mu std (across users): {user_mu.std(axis=0).mean():.4f}")
    log(f"  user_logvar mean: {user_logvar.mean():.4f}, std: {user_logvar.std():.4f}")

    # === 5. 输出 user_profiles.jsonl (兼容 v3/v4/v5/v6 格式) ===
    log(f"写入 {OUT_USER_PROFILES}")
    with OUT_USER_PROFILES.open("w") as f:
        for u, uid in enumerate(user_ids_order):
            row = {
                "user_id": uid,
                "user_mu": user_mu[u].tolist(),
                "user_logvar": user_logvar[u].tolist(),
                "user_log_scale_diag": user_log_scale_diag[u].tolist(),
                "source": "raw_space_vades",
            }
            f.write(json.dumps(row) + "\n")
    log(f"  done, n_users_written: {len(user_ids_order)}")

    # === 6. 输出 sentences.jsonl (兼容 v3/v4/v5/v6 格式, 把每句的 mu 设为 user_mu) ===
    log(f"写入 {OUT_SENTENCES}")
    with OUT_SENTENCES.open("w") as f:
        for i, r in enumerate(rows):
            u = user_id_to_idx[r["user_id"]]
            new_row = {
                "user_id": r["user_id"],
                "review_index": r.get("review_index", -1),
                "sentence_index": r.get("sentence_index", -1),
                "sentence_text": r.get("sentence_text", ""),
                "word_count": r.get("word_count", 0),
                "features": r["features"],  # 保留原始 dict
                "mu": user_mu[u].tolist(),  # 句级 mu = user_mu (raw 空间)
                "is_holdout": bool(r.get("is_holdout", False)),
            }
            f.write(json.dumps(new_row) + "\n")
    log(f"  done, n_sentences_written: {len(rows)}")

    log("=" * 70)
    log("完成 raw-space VADES 产出")
    log("=" * 70)


if __name__ == "__main__":
    main()