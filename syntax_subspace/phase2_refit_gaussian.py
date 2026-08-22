#!/usr/bin/env python3
"""Phase 2: per-user Gaussian refit on syntax-only PCA dims.

Phase 1 (phase1_explore.py) 识别了 KEEP_SYNTAX dims:
  Layer 16: [0, 2, 9, 11]
  Layer 20: [0, 12, 15, 16, 17]   ← 最优 (5-dim, exp_var=94.1%)
  Layer 24: [0, 10, 15]
  Layer 26: [0, 5, 11, 12]

本脚本:
  1. 加载 PCA components + Z (来自 pca_layer_*.npz)
  2. 加载 user_id per sentence (来自 sentences_for_rewrite.jsonl)
  3. 对每 layer:
     - 取 KEEP_SYNTAX dims 的 components (rows = keep_dims, cols = 3584)
       → U_keep[layer]: (n_keep, 3584)
     - 对每 user:
       * 取该 user 所有句子的 Z[:, keep_dims]  # (n_user_sent, n_keep)
       * fit Gaussian: z_u ~ N(mu_u, Sigma_u)
         Sigma_u = full covariance (n_keep × n_keep)
         + 单用户样本不足时加 diagonal jitter 1e-3 * I
  4. 输出:
     /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/
       - syntax_keep_dims.json (per layer keep dims, 与 U_keep 配套)
       - syntax_gaussian_layer_<L>.npz
         (user_ids, mu (n_users, n_keep), sigma (n_users, n_keep, n_keep),
          U_keep (n_keep, 3584), mean_residual_full (3584,))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
SCRATCH.mkdir(parents=True, exist_ok=True)

LAYERS = [16, 20, 24, 26]
JITTER = 1e-3  # 单用户 Σ 数值稳定


def load_keep_dims() -> dict[int, list[int]]:
    """从 correlation_layer_*.json 加载每 layer 的 KEEP_SYNTAX dim 列表。"""
    out = {}
    for L in LAYERS:
        with open(SCRATCH / f"correlation_layer_{L}.json") as f:
            data = json.load(f)
        keep = [d["dim"] for d in data["dims"] if d["verdict"] == "KEEP_SYNTAX"]
        out[L] = keep
    return out


def load_user_ids_per_sentence() -> list[str]:
    """sentences_for_rewrite.jsonl → sentence_text → user_id, 顺序与 npz['sentences'] 一致."""
    import json as _json
    rows = []
    with open(SCRATCH.parent / "gaussian_vades/sentences_for_rewrite.jsonl") as f:
        for line in f:
            rows.append(_json.loads(line))
    return [r["user_id"] for r in rows]


def fit_user_gaussian(Z_user: np.ndarray, jitter: float = JITTER
                      ) -> tuple[np.ndarray, np.ndarray]:
    """fit Gaussian 给定 (n_samples, n_keep) 矩阵。

    样本 < 2 时: mu = mean, sigma = jitter * I。
    样本 >= 2: mu = mean, sigma = np.cov(..., rowvar=False) + jitter * I。
    """
    n_keep = Z_user.shape[1] if Z_user.ndim == 2 else 1
    if Z_user.size == 0 or Z_user.shape[0] == 0:
        mu = np.zeros(n_keep, dtype=np.float64)
        sigma = np.eye(n_keep, dtype=np.float64) * jitter
        return mu, sigma
    mu = Z_user.mean(axis=0)
    if Z_user.shape[0] < 2:
        sigma = np.eye(n_keep, dtype=np.float64) * jitter
    else:
        sigma = np.cov(Z_user, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.eye(n_keep, dtype=np.float64) * float(sigma)
        sigma = sigma + np.eye(n_keep, dtype=np.float64) * jitter
    return mu, sigma


def main():
    keep_dims_per_layer = load_keep_dims()
    print("=== KEEP_SYNTAX dims per layer ===")
    for L, dims in keep_dims_per_layer.items():
        print(f"  Layer {L}: {dims} (n_keep={len(dims)})")
    print()

    user_ids = load_user_ids_per_sentence()
    print(f"[load] {len(user_ids)} sentences with user_ids; "
          f"unique users: {len(set(user_ids))}")
    print()

    save_summary = {}
    for L in LAYERS:
        keep_dims = keep_dims_per_layer[L]
        n_keep = len(keep_dims)
        if n_keep == 0:
            print(f"[Layer {L}] ⚠ n_keep=0, 跳过")
            continue

        # 加载 PCA model
        npz = np.load(SCRATCH / f"pca_layer_{L}.npz")
        components = npz["components"]  # (20, 3584)
        mean_residual = npz["mean"]  # (3584,)
        Z = npz["Z"]  # (n_samples, 20) -- all sentences

        # 取 KEEP_SYNTAX dims 的 component (rows = keep_dims, cols = 3584)
        U_keep = components[keep_dims]  # (n_keep, 3584)

        # per-user Gaussian
        unique_users = sorted(set(user_ids))
        n_users = len(unique_users)
        all_mu = np.zeros((n_users, n_keep), dtype=np.float64)
        all_sigma = np.zeros((n_users, n_keep, n_keep), dtype=np.float64)
        uid_to_idx = {uid: i for i, uid in enumerate(unique_users)}

        n_with_samples = 0
        n_skipped_empty = 0
        n_full_cov = 0
        for i, uid in enumerate(unique_users):
            mask = [j for j, u in enumerate(user_ids) if u == uid]
            Z_user = Z[mask][:, keep_dims]  # (n_user_sent, n_keep)
            if Z_user.shape[0] == 0:
                n_skipped_empty += 1
                all_mu[i] = 0.0
                all_sigma[i] = np.eye(n_keep) * JITTER
                continue
            mu, sigma = fit_user_gaussian(Z_user, jitter=JITTER)
            all_mu[i] = mu
            all_sigma[i] = sigma
            n_with_samples += 1
            if Z_user.shape[0] >= 2:
                n_full_cov += 1

        # 输出
        out_path = SCRATCH / f"syntax_gaussian_layer_{L}.npz"
        np.savez_compressed(
            out_path,
            user_ids=np.array(unique_users, dtype=object),
            mu=all_mu,
            sigma=all_sigma,
            U_keep=U_keep,
            mean_residual=mean_residual,
            keep_dims=np.array(keep_dims),
            n_keep=n_keep,
        )

        # 报告
        sigma_norms = [float(np.linalg.norm(all_sigma[i])) for i in range(n_users)]
        mu_norms = [float(np.linalg.norm(all_mu[i])) for i in range(n_users)]
        print(f"[Layer {L}]")
        print(f"  users: {n_users} (with_samples={n_with_samples}, "
              f"empty={n_skipped_empty}, full_cov={n_full_cov})")
        print(f"  mu  norm: min={min(mu_norms):.2f}, max={max(mu_norms):.2f}, "
              f"mean={sum(mu_norms)/len(mu_norms):.2f}")
        print(f"  sigma norm: min={min(sigma_norms):.2f}, max={max(sigma_norms):.2f}, "
              f"mean={sum(sigma_norms)/len(sigma_norms):.2f}")
        print(f"  U_keep shape: {U_keep.shape}")
        print(f"  → {out_path}")
        print()

        save_summary[str(L)] = {
            "keep_dims": keep_dims,
            "n_keep": n_keep,
            "n_users": n_users,
            "mu_norm_mean": float(sum(mu_norms) / len(mu_norms)),
            "sigma_norm_mean": float(sum(sigma_norms) / len(sigma_norms)),
        }

    # 总体 summary
    with open(SCRATCH / "syntax_gaussian_summary.json", "w") as f:
        json.dump(save_summary, f, indent=2)
    print(f"[save] summary → {SCRATCH / 'syntax_gaussian_summary.json'}")

    # 估算 Δh_syntax norm (per layer, mean + sample)
    print()
    print("=== 预估 Δh_syntax norm (mean residual only vs sampled) ===")
    for L in LAYERS:
        npz = np.load(SCRATCH / f"syntax_gaussian_layer_{L}.npz")
        U_keep = npz["U_keep"]
        mu = npz["mu"]
        sigma = npz["sigma"]
        n_users, n_keep = mu.shape
        # Δh from mu:  Δh = mu @ U_keep  → (n_users, n_keep) @ (n_keep, 3584) = (n_users, 3584)
        delta_mean = mu @ U_keep  # (n_users, 3584)
        norms_mean = np.linalg.norm(delta_mean, axis=1)
        # Δh from sampled z:  sample one z ~ N(mu, sigma), compute Δh
        rng = np.random.default_rng(42)
        sample_norms = []
        for i in range(min(50, n_users)):
            z = rng.multivariate_normal(mu[i], sigma[i])
            delta = z @ U_keep
            sample_norms.append(float(np.linalg.norm(delta)))
        print(f"  Layer {L}:")
        print(f"    Δh_mean norm:  min={norms_mean.min():.1f}, "
              f"max={norms_mean.max():.1f}, mean={norms_mean.mean():.1f}")
        print(f"    Δh_sample norm: min={min(sample_norms):.1f}, "
              f"max={max(sample_norms):.1f}, mean={sum(sample_norms)/len(sample_norms):.1f}")


if __name__ == "__main__":
    main()
