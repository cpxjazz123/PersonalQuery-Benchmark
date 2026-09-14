#!/usr/bin/env python3
"""Diagnostic: does the 32d supervised syntax space actually separate users?

User 2026-09-14: 先判断 encoder representation 问题还是 Gaussian region 宽窄问题。

诊断:
  1. self-distance vs cross-distance distribution on held-out validation sentences
     - d_self(z) = D²(z_val, mu_u)  (true user's Gaussian)
     - d_cross(z) = D²(z_val, mu_v) for v != u (other users' Gaussians)
  2. Top-1 User Classification Accuracy:
     P(u = argmin_v d_v(z)) where z is user u's own validation sentence

如果 self << cross 但 95% regions 仍大量重叠 → encoder 有鉴别力,问题是 Gaussian 太宽
如果 self ≈ cross → encoder latent space 没有学出 personalized syntax
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
EMB_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz")
OUT_PATH = REPO_ROOT / "result/05_gaussian_audit/user_discriminability.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== user discriminability diagnostic ===")

    # Stage 04 Gaussian
    with open(STAGE04_PATH) as f:
        s4 = json.load(f)
    s4_users = s4["users"]
    s4_uids = list(s4_users.keys())
    n_users = len(s4_uids)
    log(f"  Stage 04 users: {n_users}")

    # Load embeddings
    emb = np.load(EMB_PATH, allow_pickle=False)
    z_val = emb["z_val"].astype(np.float32)       # (N_val, 32)
    val_uid_arr = emb["val_idx"]                  # sentence-level positions
    uid_list = list(emb["uid_list"])
    log(f"  val: {z_val.shape}")

    # Build position → uid_list index mapping (same as cross_user_overlap_audit)
    user_n_sents_path = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json")
    with open(user_n_sents_path) as f:
        user_n_sents = [int(x) for x in json.load(f)]
    cum_n_sents = np.array(user_n_sents, dtype=np.int64)
    np.cumsum(cum_n_sents, out=cum_n_sents)
    val_uidlist_idx = np.clip(
        np.searchsorted(cum_n_sents, val_uid_arr, side="right") - 1,
        0, len(uid_list) - 1
    )

    # Filter to Stage 04 users only
    uidlist_pos = {uid: i for i, uid in enumerate(uid_list)}
    s4_uidlist_indices = np.array(
        [uidlist_pos[u] for u in s4_uids if u in uidlist_pos],
        dtype=np.int64
    )
    s4_uids = [u for u in s4_uids if u in uidlist_pos]
    n_users = len(s4_uids)
    log(f"  Stage 04 users in uid_list: {n_users}")

    # Build val sentence → s4_uids index
    inv_pos = np.full(len(uid_list), -1, dtype=np.int32)
    inv_pos[s4_uidlist_indices] = np.arange(n_users, dtype=np.int32)
    val_s4_idx = inv_pos[val_uidlist_idx]   # -1 if not Stage 04

    # Filter to Stage 04 validation sentences
    s4_val_mask = val_s4_idx >= 0
    z_val_s4 = z_val[s4_val_mask]
    val_self_idx = val_s4_idx[s4_val_mask]  # s4_uids index of true user
    log(f"  Stage 04 val sents: {z_val_s4.shape[0]}")

    # Pack Gaussian params
    mu_arr = np.stack([np.asarray(s4_users[u]["mu"], dtype=np.float32) for u in s4_uids], axis=0)
    inv_arr = np.stack([np.asarray(s4_users[u]["sigma_inv"], dtype=np.float32) for u in s4_uids], axis=0)
    gate_T_arr = np.asarray([float(s4_users[u]["d2_q95"]) for u in s4_uids], dtype=np.float64)
    log(f"  mu: {mu_arr.shape}  inv: {inv_arr.shape}  gate_T: {gate_T_arr.shape}")

    # --- GPU: compute all pairwise D² for val sents ---
    log("  moving to GPU ...")
    device = torch.device("cuda:0")
    z_t = torch.from_numpy(z_val_s4).to(device)           # (N_val_s4, 32)
    mu_t = torch.from_numpy(mu_arr).to(device)            # (n_users, 32)
    inv_t = torch.from_numpy(inv_arr).to(device)          # (n_users, 32, 32)

    N_val = z_val_s4.shape[0]
    n_users_ = n_users

    # Sample a manageable number of validation sentences for the full pairwise matrix
    # Full: N_val × n_users × 32 — N_val=603K, n_users=10K → 24GB, too big
    # → sample 500 validation sentences per user (or 2000 total, whichever is smaller)
    N_SAMPLE = min(2000, N_val)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(N_val, size=N_SAMPLE, replace=False)
    sample_true_idx = val_self_idx[sample_indices]  # true user s4 index for each sample
    z_sample = z_t[sample_indices]                   # (N_SAMPLE, 32)
    log(f"  sampled {N_SAMPLE} validation sentences for full pairwise D²")

    # Compute D² for each sample against ALL users: (N_SAMPLE, n_users)
    log("  computing full D² matrix (sample × all users) on GPU ...")
    diff_all = z_sample[:, None, :] - mu_t[None, :, :]          # (N_SAMPLE, n_users, 32)
    left_all = torch.einsum("nud,ude->nue", diff_all, inv_t)    # (N_SAMPLE, n_users, 32)
    d2_all = (left_all * diff_all).sum(dim=2).cpu().numpy()     # (N_SAMPLE, n_users)
    log(f"    D² matrix: {d2_all.shape}  computed")

    # --- Metric 1: Self vs Cross distance distribution ---
    # For each sample, d_self = d2[sample_idx, true_user]
    d_self = d2_all[np.arange(N_SAMPLE), sample_true_idx]       # (N_SAMPLE,)
    # d_cross = mean of d2 to all OTHER users
    cross_dists = d2_all.copy()
    for i in range(N_SAMPLE):
        cross_dists[i, sample_true_idx[i]] = np.nan
    d_cross_mean = np.nanmean(cross_dists, axis=1)              # (N_SAMPLE,)
    d_cross_min = np.nanmin(cross_dists, axis=1)                # (N_SAMPLE,)

    log(f"\n  === Self vs Cross Distance ===")
    log(f"  d_self:    mean={d_self.mean():.4f}  median={np.median(d_self):.4f}  "
        f"p5={np.quantile(d_self,0.05):.4f}  p95={np.quantile(d_self,0.95):.4f}")
    log(f"  d_cross_m: mean={d_cross_mean.mean():.4f}  median={np.median(d_cross_mean):.4f}  "
        f"p5={np.quantile(d_cross_mean,0.05):.4f}  p95={np.quantile(d_cross_mean,0.95):.4f}")
    log(f"  d_cross_n: mean={d_cross_min.mean():.4f}  median={np.median(d_cross_min):.4f}  "
        f"p5={np.quantile(d_cross_min,0.05):.4f}  p95={np.quantile(d_cross_min,0.95):.4f}")
    separation = d_cross_mean.mean() / d_self.mean()
    log(f"  separation (d_cross_mean / d_self): {separation:.3f}")
    # d_self < d_cross_min: fraction
    frac_self_is_min = float((d_self <= d_cross_min).mean())
    log(f"  P(self < min_cross): {frac_self_is_min:.4f}")

    # --- Metric 2: Top-1 User Classification Accuracy ---
    # For each sample, argmin of d2_all[sample_idx, :]
    predicted_user = np.argmin(d2_all, axis=1)    # (N_SAMPLE,)
    top1_accuracy = float((predicted_user == sample_true_idx).mean())
    top5_accuracy = float(np.isin(sample_true_idx, np.argsort(d2_all, axis=1)[:, -5:][:, ::-1]).mean())
    log(f"\n  === Top-K User Classification ===")
    log(f"  Top-1 Accuracy: {top1_accuracy:.4f}  (random = {1/n_users:.5f})")
    log(f"  Top-5 Accuracy: {top5_accuracy:.4f}  (random = {5/n_users:.5f})")

    # --- Metric 3: d_self vs d2_q95 (are validation sentences inside 95% region?) ---
    d_self_q95_frac = float((d_self <= gate_T_arr[sample_true_idx]).mean())
    log(f"\n  === Self Distance vs 95% Gate ===")
    log(f"  fraction of val sents inside own 95% region: {d_self_q95_frac:.4f}")
    log(f"  (should be close to 0.95 if Gaussian fits well)")

    # --- Save results ---
    result = {
        "config": {
            "n_stage04_users": n_users,
            "n_val_sents_s4": int(N_val),
            "n_samples": N_SAMPLE,
            "description": (
                "Diagnostic: self vs cross distance on held-out validation sentences. "
                "If self << cross but regions still heavily overlap → encoder ok, Gaussian too wide. "
                "If self ≈ cross → encoder lacks user-discriminative signal."
            ),
        },
        "self_vs_cross": {
            "d_self_mean": float(d_self.mean()),
            "d_self_median": float(np.median(d_self)),
            "d_self_p5": float(np.quantile(d_self, 0.05)),
            "d_self_p95": float(np.quantile(d_self, 0.95)),
            "d_cross_mean_mean": float(d_cross_mean.mean()),
            "d_cross_mean_median": float(np.median(d_cross_mean)),
            "d_cross_min_mean": float(d_cross_min.mean()),
            "d_cross_min_median": float(np.median(d_cross_min)),
            "separation_ratio": float(separation),
            "frac_self_is_min": frac_self_is_min,
        },
        "user_classification": {
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "random_top1": 1 / n_users,
            "random_top5": 5 / n_users,
        },
        "gaussian_fidelity": {
            "frac_val_inside_95region": d_self_q95_frac,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"\n  wrote → {OUT_PATH}")

    # --- Interpretation ---
    log("\n  === INTERPRETATION ===")
    if separation > 2.0 and top1_accuracy > 0.5:
        log("  ✓ Encoder SEPARATES users well (self << cross).")
        log("  → Problem is Gaussian 95% region too wide, not encoder.")
        log("  → Consider: tighter quantile (Q_80), higher-dimensional space, or adaptive gate.")
    elif separation < 1.5 and top1_accuracy < 0.05:
        log("  ✗ Encoder DOES NOT separate users (self ≈ cross, top-1 ≈ random).")
        log("  → Problem is encoder representation, not Gaussian width.")
        log("  → Recommend: 32d→16d retrain, stronger contrastive, or user-discriminative training.")
    else:
        log(f"  ~ Mixed signal: separation={separation:.2f}, top1={top1_accuracy:.4f}")
        log("  → See user_discriminability.json for full diagnostics.")


if __name__ == "__main__":
    main()
