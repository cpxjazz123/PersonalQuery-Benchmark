#!/usr/bin/env python3
"""Phase 10.9.5s: Stratified analysis of 5i2_1000 by user activity.

问题: n=60 → Δ_rank=-0.367, n=1000 → Δ_rank=-0.114 (效应缩小 3x).
假说 1: n=60 dev 是 cherry-picked 最活跃 60 用户, 效应高估.
假说 2: 真实效应就是 -0.1, n=60 是 noise.

验证方法: 按用户 stage1 review 数分 5 层 (Q1=最低 activity, Q5=最高 activity),
          计算每层的 Δ_rank / win rate. 如果 Q5 接近 n=60 的 -0.367 而 Q1 接近 0,
          → 假说 1 成立 (selection bias).
          如果各层都接近 -0.1, → 假说 2 成立 (n=60 是 noise).

输入: phase10_9_5i2_1000_per_user.jsonl (per-pair metrics)
      stage1_filtered_users_reviews_3000u.json (review count per user)
输出: stratified_metrics.json (各层 effect size)
"""
from __future__ import annotations

import json
import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

PER_USER_FILE = OUT_DIR / "phase10_9_5i2_1000_per_user.jsonl"
STAGE1_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews_3000u.json"
OUT_STRAT = OUT_DIR / "phase10_9_5i2_1000_stratified.json"

N_QUANTILES = 5  # Q1..Q5 by review count


def log(m):
    print(f"[phase10-9.5s-stratified] {m}", flush=True)


def load_review_counts() -> Dict[str, int]:
    """user_id → total review count."""
    with STAGE1_FILE.open() as f:
        data = json.load(f)
    out = {}
    for item in data:
        out[item["user_id"]] = len(item.get("reviews", []))
    return out


def load_user_mu_norm() -> Dict[str, float]:
    """user_id → ||user_mu|| (style vector magnitude)."""
    import numpy as np
    profile_file = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
    out = {}
    with profile_file.open() as f:
        for line in f:
            r = json.loads(line)
            mu = np.array(r["user_mu"], dtype=np.float32)
            out[r["user_id"]] = float(np.linalg.norm(mu))
    return out


def main():
    log("[1] Loading per-user metrics + style norms ...")
    per_user = []
    with PER_USER_FILE.open() as f:
        for line in f:
            per_user.append(json.loads(line))
    log(f"  per_user: {len(per_user)}")

    rc = load_review_counts()
    log(f"  review counts: {len(rc)} users")
    mu_norm = load_user_mu_norm()
    log(f"  user_mu_norm: {len(mu_norm)} users")

    # Build (user_id, delta_rank, win_rate, review_count)
    # Note: per_user is per-PAIR (one row per (user, asin) pair)
    # aggregate per user first
    user_agg = defaultdict(lambda: {"drs": [], "wins": [], "asins": []})
    for r in per_user:
        uid = r["user_id"]
        user_agg[uid]["drs"].append(r["delta_rank"])
        user_agg[uid]["wins"].append(r["win_rate"])
        user_agg[uid]["asins"].append(r["asin"])

    user_stats = {}
    for uid, agg in user_agg.items():
        user_stats[uid] = {
            "delta_rank_mean": float(np.mean(agg["drs"])),
            "win_rate": float(np.mean(agg["wins"])),
            "n_pairs": len(agg["asins"]),
            "review_count": rc.get(uid, 0),
            "mu_norm": mu_norm.get(uid, 0.0),
        }
    log(f"  user_stats (aggregated per user): {len(user_stats)}")

    # Assign to N_QUANTILES by mu_norm
    mu_list = sorted([v["mu_norm"] for v in user_stats.values()])
    if not mu_list:
        log("  ERR: no mu_norm")
        return
    quantiles = np.quantile(mu_list, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])
    log(f"  quantile thresholds (mu_norm): {quantiles.tolist()}")

    def assign_q(mu_val):
        for i, t in enumerate(quantiles):
            if mu_val <= t:
                return i
        return N_QUANTILES - 1

    by_q = defaultdict(list)
    for uid, st in user_stats.items():
        q = assign_q(st["mu_norm"])
        st["quantile"] = q
        by_q[q].append(st)

    log(f"\n=== Stratified results by user_mu_norm (Q1=lowest, Q{N_QUANTILES}=highest) ===")
    log(f"{'Q':>3} {'n_users':>8} {'mu_lo':>10} {'mu_hi':>10} "
        f"{'Δ_rank':>9} {'CI_lo':>9} {'CI_hi':>9} {'win%':>7} {'CI_lo':>7} {'CI_hi':>7}")

    stratified_results = {}
    for q in range(N_QUANTILES):
        users = by_q[q]
        if not users:
            continue
        drs = np.array([u["delta_rank_mean"] for u in users])
        wins = np.array([u["win_rate"] for u in users])
        mu_vals = sorted(u["mu_norm"] for u in users)
        # bootstrap CI
        n_boot = 5000
        rng = np.random.default_rng(42)
        boot_dr = []
        boot_win = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(users), len(users))
            boot_dr.append(drs[idx].mean())
            boot_win.append(wins[idx].mean())
        boot_dr = np.array(boot_dr)
        boot_win = np.array(boot_win)
        dr_lo, dr_hi = np.quantile(boot_dr, [0.025, 0.975])
        win_lo, win_hi = np.quantile(boot_win, [0.025, 0.975])

        dr_mean = float(drs.mean())
        win_mean = float(wins.mean())
        log(f"{q+1:>3} {len(users):>8} {mu_vals[0]:>10.3f} {mu_vals[-1]:>10.3f} "
            f"{dr_mean:>+9.3f} {dr_lo:>+9.3f} {dr_hi:>+9.3f} "
            f"{win_mean*100:>6.2f}% {win_lo*100:>+6.2f}% {win_hi*100:>+6.2f}%")

        stratified_results[f"Q{q+1}"] = {
            "n_users": len(users),
            "mu_norm_lo": mu_vals[0],
            "mu_norm_hi": mu_vals[-1],
            "delta_rank_mean": dr_mean,
            "delta_rank_ci_lo": float(dr_lo),
            "delta_rank_ci_hi": float(dr_hi),
            "win_rate_mean": win_mean,
            "win_rate_ci_lo": float(win_lo),
            "win_rate_ci_hi": float(win_hi),
        }

    OUT_STRAT.write_text(json.dumps(stratified_results, ensure_ascii=False, indent=2))
    log(f"\n  stratified → {OUT_STRAT}")

    # === Diagnostic: compare top quartile vs bottom ===
    log(f"\n=== 假说验证 ===")
    log(f"如果 n=60 dev 高估来自 selection bias, Q5 (top activity) 应接近 n=60 的 Δ_rank=-0.367")
    log(f"如果真实效应就是 -0.1, 所有 Q 都应在 -0.1 左右")
    log(f"实际: 见上表 Q1..Q5 的 Δ_rank")


if __name__ == "__main__":
    main()