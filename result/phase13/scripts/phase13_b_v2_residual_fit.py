#!/usr/bin/env python3
"""Phase 13.B-v2 residual fit: subtract neutral hidden + refit Gaussian per N.

Loads cached hiddens + phase13_a neutral hidden, computes residuals (h - neutral_mean),
fits Gaussian with N ∈ {10, 20, 30, 50, 100}, reports σ/μ ratio in residual space
(comparable to Phase 15.D numbers).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
HIDDEN_NPZ = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

OUT_CSV = OUT_DIR / "phase13_b_v2_residual_ratio_per_user.csv"
OUT_SUMMARY = OUT_DIR / "phase13_b_v2_residual_summary.json"

N_LEVELS = [10, 20, 30, 50, 100]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 13.B-v2 residual fit: subtract neutral, refit per N")
    log("=" * 70)

    log("[1] Loading cached hiddens + neutral hidden ...")
    d = np.load(HIDDEN_NPZ, allow_pickle=True)
    user_ids = list(d["user_ids"])
    hiddens_per_user = list(d["hiddens"])
    counts = list(d["counts"])

    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    # neutral_hiddens.npz 字段: texts (str), vecs ([N, 28, 3584] float16)
    neutral_vecs = n_d["vecs"].astype(np.float32)  # [N, 28, 3584]
    log(f"  users: {len(user_ids)}, neutral vecs: {neutral_vecs.shape}")
    # 取 layer 14 的所有 neutral sentence 的均值 → [3584]
    neutral_vec = neutral_vecs[:, 14, :].mean(axis=0)
    log(f"  neutral (layer 14 mean): {neutral_vec.shape}")
    H = neutral_vec.shape[0]

    log(f"[2] Computing residuals & fitting per N (H={H}) ...")
    rows = []
    for N in N_LEVELS:
        log(f"  N={N}:")
        n_fitted = 0
        for i, uid in enumerate(user_ids):
            if counts[i] < N:
                continue
            h = hiddens_per_user[i][:N].astype(np.float32)  # [N, H]
            residual = h - neutral_vec[None, :]  # [N, H]
            mu = residual.mean(axis=0)
            sigma = residual.std(axis=0, ddof=0)
            mu_norm = float(np.linalg.norm(mu))
            sigma_norm = float(np.linalg.norm(sigma))
            ratio = sigma_norm / (mu_norm + 1e-12)
            rows.append({
                "user_id": uid,
                "N": N,
                "mu_norm": mu_norm,
                "sigma_norm": sigma_norm,
                "ratio_sigma_over_mu": ratio,
            })
            n_fitted += 1
        log(f"    {n_fitted} users")

    log(f"[3] Writing CSV: {OUT_CSV}")
    with OUT_CSV.open("w") as f:
        f.write("user_id,N,mu_norm,sigma_norm,ratio_sigma_over_mu\n")
        for r in rows:
            f.write(f"{r['user_id']},{r['N']},{r['mu_norm']:.4f},{r['sigma_norm']:.4f},{r['ratio_sigma_over_mu']:.4f}\n")

    log("[4] Aggregate per N ...")
    summary: dict = {
        "phase": "13.B-v2 residual",
        "n_users_total": len(user_ids),
        "N_levels": N_LEVELS,
        "per_N": {},
    }
    print(f"\n{'N':>4} {'n_users':>8} {'mean_mu':>10} {'mean_sigma':>11} {'mean_ratio':>11} {'med_ratio':>10} {'std_ratio':>10}")
    print("-" * 70)
    for N in N_LEVELS:
        ms = [r["mu_norm"] for r in rows if r["N"] == N]
        ss = [r["sigma_norm"] for r in rows if r["N"] == N]
        rs = [r["ratio_sigma_over_mu"] for r in rows if r["N"] == N]
        if not rs:
            continue
        stats = {
            "n_users": len(rs),
            "mean_mu_norm": float(np.mean(ms)),
            "mean_sigma_norm": float(np.mean(ss)),
            "mean_ratio": float(np.mean(rs)),
            "median_ratio": float(np.median(rs)),
            "std_ratio": float(np.std(rs)),
        }
        summary["per_N"][N] = stats
        print(f"{N:>4} {stats['n_users']:>8} {stats['mean_mu_norm']:>10.3f} {stats['mean_sigma_norm']:>11.3f} "
              f"{stats['mean_ratio']:>11.4f} {stats['median_ratio']:>10.4f} {stats['std_ratio']:>10.4f}")

    # plateau detection
    prev = None
    plateau_start = None
    for N in N_LEVELS:
        if N not in summary["per_N"]:
            continue
        r = summary["per_N"][N]["mean_ratio"]
        if prev is not None and prev > 0:
            change = abs(r - prev) / prev
            if change < 0.05 and plateau_start is None:
                plateau_start = N
        prev = r
    summary["plateau_start_N"] = plateau_start

    # compare to Phase 15.D numbers
    summary["comparison_to_phase_15d"] = {
        "phase_15d_N10_ratio_residual": 7.3,
        "phase_15d_mu_norm": 24,
        "phase_15d_sigma_norm": 175,
        "note": "Phase 15.D used N=10 sents/user (raw Amazon via vades_proto filter, 298 users). Current sweep uses 198 users with ≥50 reviews in raw data.",
    }

    print(f"\n→ ratio plateau starts at N={plateau_start}")
    if plateau_start:
        ps = summary["per_N"][plateau_start]
        print(f"  At N={plateau_start}: mean_ratio = {ps['mean_ratio']:.4f}, "
              f"||μ_residual|| = {ps['mean_mu_norm']:.2f}, ||σ_residual|| = {ps['mean_sigma_norm']:.2f}")
    print(f"\nPhase 15.D reference (N=10): ratio=7.3, ||μ||=24, ||σ||=175")
    print(f"  (current N=10 vs Phase 15.D — should be similar if protocols match)")

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  saved summary: {OUT_SUMMARY}")

    log("=" * 70)
    log("PHASE 13.B-v2 RESIDUAL FIT DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()