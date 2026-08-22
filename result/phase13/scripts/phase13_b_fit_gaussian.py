#!/usr/bin/env python3
"""Phase 13.B: Fit per-user per-layer Gaussian N(mu_u^ℓ, diag(sigma_u^ℓ)) on residuals.

For each user u and each layer ℓ, the residual samples are {r_i^ℓ}_{i=1..n_u}
where n_u = number of (user, sentence) pairs. Compute:
  mu_u^ℓ      = mean of r_i^ℓ
  sigma_u^ℓ   = std of r_i^ℓ  (diagonal covariance; 10 samples can't support full cov)

Output:
  - phase13_b_user_gaussians_qwen.npz
      user_ids    : [n_users]
      layers      : [28]
      mu          : [n_users, 28, 3584] fp32
      sigma_diag  : [n_users, 28, 3584] fp32  (variance, not std)
      std_diag    : [n_users, 28, 3584] fp32  (std, for fast sampling)
      n_per_user  : [n_users]
  - phase13_b_meta.json
  - phase13_b_validation.json (sanity: per-user own vs other cosine separation)

Sampling formula for Phase 13.C:
  s_u^(k)^ℓ = mu_u^ℓ + rho * std_u^ℓ ⊙ eps     # eps ~ N(0, I)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
IN_RESIDUALS = OUT_DIR / "phase13_a_per_sentence_residuals_qwen.npz"
OUT_GAUSSIANS = OUT_DIR / "phase13_b_user_gaussians_qwen.npz"
OUT_META = OUT_DIR / "phase13_b_meta.json"
OUT_VALIDATION = OUT_DIR / "phase13_b_validation.json"

# Hardcoded config
N_LAYERS = 28
HIDDEN_DIM = 3584
RANDOM_SEED = 42
LW_SHRINKAGE = 0.1     # Ledoit-Wolf-style shrinkage toward identity (variance regularization)
MIN_VAR = 1e-4         # floor on variance to avoid degenerate sigma


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 13.B: Fit per-user per-layer Gaussian on residuals")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)

    log(f"[1] Loading residuals from {IN_RESIDUALS} ...")
    npz = np.load(IN_RESIDUALS, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    sent_uids = list(npz["sent_uids"])
    residuals = npz["residuals"]  # [n_total, L, H] fp16
    n_total, n_layers_actual, hidden_actual = residuals.shape
    log(f"  n_total={n_total}, layers={n_layers_actual}, hidden={hidden_actual}")
    log(f"  n_users (phase10): {len(user_ids)}")

    # Group by user
    log("[2] Grouping residuals by user ...")
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    user_to_sidxs: dict[int, list[int]] = {i: [] for i in range(len(user_ids))}
    for s_i, uid in enumerate(sent_uids):
        if uid in uid_to_idx:
            user_to_sidxs[uid_to_idx[uid]].append(s_i)
    n_per_user = np.array([len(user_to_sidxs[i]) for i in range(len(user_ids))], dtype=np.int32)
    log(f"  per-user n: min={n_per_user.min()}, max={n_per_user.max()}, mean={n_per_user.mean():.1f}")
    valid_users = n_per_user >= 2
    log(f"  valid users (n>=2): {int(valid_users.sum())}/{len(user_ids)}")

    # Compute per-user per-layer mu and var
    log(f"[3] Computing per-user mu and sigma_diag with LW shrinkage={LW_SHRINKAGE} ...")
    mu = np.zeros((len(user_ids), n_layers_actual, hidden_actual), dtype=np.float32)
    sigma_diag = np.zeros((len(user_ids), n_layers_actual, hidden_actual), dtype=np.float32)
    std_diag = np.zeros((len(user_ids), n_layers_actual, hidden_actual), dtype=np.float32)

    # Pooled variance per layer (for shrinkage target)
    pooled_var = np.zeros((n_layers_actual, hidden_actual), dtype=np.float64)
    for li in range(n_layers_actual):
        pooled_var[li] = residuals[:, li, :].astype(np.float64).var(axis=0)

    n_done = 0
    for u_i in range(len(user_ids)):
        sidxs = user_to_sidxs[u_i]
        n = len(sidxs)
        if n < 1:
            continue
        local = residuals[sidxs].astype(np.float64)  # [n, L, H]
        m = local.mean(axis=0)  # [L, H]
        if n < 2:
            # Use pooled variance as fallback (no sample variance)
            v = pooled_var.copy()
        else:
            v = local.var(axis=0, ddof=0)  # [L, H] biased var
            # LW shrinkage: sigma = (1-α) * sample_var + α * pooled_var
            v = (1.0 - LW_SHRINKAGE) * v + LW_SHRINKAGE * pooled_var
        # Floor variance
        v = np.maximum(v, MIN_VAR)
        s = np.sqrt(v)
        mu[u_i] = m.astype(np.float32)
        sigma_diag[u_i] = v.astype(np.float32)
        std_diag[u_i] = s.astype(np.float32)
        n_done += 1
        if n_done % 200 == 0:
            log(f"  fit {n_done}/{len(user_ids)} users")

    log(f"  fitted {n_done} users, mean var per dim: {sigma_diag[valid_users].mean():.4f}")

    # === [4] Sanity: per-user style vector magnitude ===
    log("[4] Sanity: mu magnitude per layer ...")
    mu_norms = np.linalg.norm(mu[valid_users], axis=-1)  # [n_valid, L]
    for li in [0, 4, 8, 12, 14, 16, 20, 24, 27]:
        if li < n_layers_actual:
            log(f"  layer {li}: mean mu norm = {mu_norms[:, li].mean():.4f}  "
                f"(std={mu_norms[:, li].std():.4f})")

    # === [5] Sanity: per-user own-vs-other cosine separation (mean residual) ===
    log("[5] Sanity: own vs other cos separation (mean residual) ...")
    valid_u_idx = np.where(valid_users)[0]
    cos_metrics = {}
    rng = np.random.default_rng(RANDOM_SEED)
    for li in [12, 14, 16, 20, 24]:
        if li >= n_layers_actual:
            continue
        own_cos_list = []
        other_cos_list = []
        for u_i in valid_u_idx:
            u_vec = mu[u_i, li]
            u_norm = u_vec / (np.linalg.norm(u_vec) + 1e-9)
            # Sample 5 random other users
            others = rng.choice(valid_u_idx, size=min(5, len(valid_u_idx) - 1), replace=False)
            for o_i in others:
                o_vec = mu[o_i, li]
                o_norm = o_vec / (np.linalg.norm(o_vec) + 1e-9)
                other_cos_list.append(float(np.dot(u_norm, o_norm)))
            own_cos_list.append(1.0)  # cos with self = 1
        own_mean = float(np.mean(own_cos_list))
        other_mean = float(np.mean(other_cos_list))
        margin = own_mean - other_mean
        cos_metrics[f"layer{li}"] = {
            "own_mean_cos": own_mean,
            "other_mean_cos": other_mean,
            "margin": margin,
        }
        log(f"  layer {li}: own_cos={own_mean:.4f}  other_cos={other_mean:.4f}  margin={margin:.4f}")

    # === [6] Save ===
    log(f"[6] Saving Gaussian cache to {OUT_GAUSSIANS} ...")
    np.savez_compressed(
        OUT_GAUSSIANS,
        user_ids=np.asarray(user_ids, dtype=object),
        layers=np.arange(n_layers_actual, dtype=np.int32),
        mu=mu,
        sigma_diag=sigma_diag,
        std_diag=std_diag,
        n_per_user=n_per_user,
        valid=valid_users,
    )
    log(f"  saved: mu {mu.shape}, sigma_diag {sigma_diag.shape}")

    meta = {
        "phase": "13.B",
        "n_users": len(user_ids),
        "n_users_valid": int(valid_users.sum()),
        "n_layers": n_layers_actual,
        "hidden_dim": hidden_actual,
        "lw_shrinkage": LW_SHRINKAGE,
        "min_var_floor": MIN_VAR,
        "mean_var_per_dim": float(sigma_diag[valid_users].mean()),
        "mean_std_per_dim": float(std_diag[valid_users].mean()),
        "mu_norm_per_layer": {
            int(li): float(mu_norms[:, li].mean())
            for li in range(n_layers_actual)
        },
        "cos_metrics": cos_metrics,
        "sampling_formula": "s_u^(k)^ℓ = mu_u^ℓ + rho * std_u^ℓ ⊙ eps, eps ~ N(0, I_3584)",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta: {OUT_META}")

    validation = {
        "phase": "13.B",
        "n_users_valid": int(valid_users.sum()),
        "cos_metrics": cos_metrics,
        "verdict": "GO" if all(v["margin"] > 0.05 for v in cos_metrics.values()) else "PARTIAL",
    }
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    log(f"  validation: {OUT_VALIDATION} → {validation['verdict']}")

    log("=" * 70)
    log("PHASE 13.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()