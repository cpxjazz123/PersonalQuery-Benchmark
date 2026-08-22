#!/usr/bin/env python3
"""Phase 13.B-v2 fit-only (load cached hiddens, skip Qwen load).

Reads phase13_b_v2_user_hiddens_n100.npz and fits Gaussian with N ∈ {10, 20, 30, 50, 100}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
HIDDEN_NPZ = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
OUT_RATIOS = OUT_DIR / "phase13_b_v2_ratio_per_user.csv"
OUT_SUMMARY = OUT_DIR / "phase13_b_v2_summary.json"
LOG_PATH = OUT_DIR / "phase13_b_v2_fit.log"

N_LEVELS = [10, 20, 30, 50, 100]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 13.B-v2 fit-only (skip Qwen, use cached hiddens)")
    log("=" * 70)

    log("[1] Loading cached hiddens ...")
    d = np.load(HIDDEN_NPZ, allow_pickle=True)
    user_ids = list(d["user_ids"])
    hiddens_per_user = list(d["hiddens"])
    counts = list(d["counts"])
    log(f"  users: {len(user_ids)}")
    log(f"  sents per user: min={min(counts)} median={sorted(counts)[len(counts)//2]} max={max(counts)}")

    # Filter users with ≥100 sentences for clean N=100 fit
    valid_100 = [i for i, c in enumerate(counts) if c >= 100]
    log(f"  users with ≥100 sents (full N=100 fit): {len(valid_100)}")

    log("[2] Fitting Gaussian with N levels ...")
    rows = []
    for N in N_LEVELS:
        log(f"  N={N}:")
        n_fitted = 0
        for i, uid in enumerate(user_ids):
            if counts[i] < N:
                continue  # skip users with too few sents
            h = hiddens_per_user[i][:N]  # [N, H]
            mu = h.mean(axis=0)
            sigma = h.std(axis=0, ddof=0)
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
        log(f"    {n_fitted} users (skip {len(user_ids) - n_fitted})")

    log(f"[3] Writing CSV: {OUT_RATIOS}")
    with OUT_RATIOS.open("w") as f:
        f.write("user_id,N,mu_norm,sigma_norm,ratio_sigma_overmu\n")
        for r in rows:
            f.write(f"{r['user_id']},{r['N']},{r['mu_norm']:.4f},{r['sigma_norm']:.4f},{r['ratio_sigma_over_mu']:.4f}\n")

    log("[4] Aggregate per N ...")
    summary: dict = {
        "phase": "13.B-v2",
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
            log(f"  N={N}: no users with ≥{N} sents")
            continue
        stats = {
            "n_users": len(rs),
            "mean_mu_norm": float(np.mean(ms)),
            "mean_sigma_norm": float(np.mean(ss)),
            "mean_ratio": float(np.mean(rs)),
            "median_ratio": float(np.median(rs)),
            "std_ratio": float(np.std(rs)),
            "p25_ratio": float(np.percentile(rs, 25)),
            "p75_ratio": float(np.percentile(rs, 75)),
        }
        summary["per_N"][N] = stats
        print(f"{N:>4} {stats['n_users']:>8} {stats['mean_mu_norm']:>10.3f} {stats['mean_sigma_norm']:>11.3f} "
              f"{stats['mean_ratio']:>11.4f} {stats['median_ratio']:>10.4f} {stats['std_ratio']:>10.4f}")

    # Find plateau (mean_ratio change <5% vs prev N)
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
    summary["interpretation"] = (
        f"||σ||/||μ|| ratio plateaus at N={plateau_start} "
        f"(ratio change <5% between consecutive N levels). "
        f"At this plateau, σ estimate is stable → Gaussian fit reliable. "
        f"At smaller N, σ is inflated by finite-sample variance."
    )
    print(f"\n→ ratio plateau starts at N={plateau_start}")
    if plateau_start:
        ps = summary["per_N"][plateau_start]
        print(f"  At N={plateau_start}: mean_ratio = {ps['mean_ratio']:.4f}, "
              f"||μ|| = {ps['mean_mu_norm']:.2f}, ||σ|| = {ps['mean_sigma_norm']:.2f}")

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  saved summary: {OUT_SUMMARY}")

    log("=" * 70)
    log("PHASE 13.B-v2 FIT DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()