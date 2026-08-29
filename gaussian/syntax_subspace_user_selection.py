#!/usr/bin/env python3
"""Cohort user selection: 3-arm experiment
(Random 10 / Quality-only 10 / Quality + Bhattacharyya max-min 10).

用户指令 2026-08-29: 验证 "Gaussian 质量高 + 彼此重叠少 的用户 cohort → 更好的个性化
Query 生成"。方法:
 1. 从 quality Gaussian 用户 (build_user Section 7 3-class gate 已过滤:
    Q1 σ_mean ≥ MIN_SIGMA_MEAN, Q2 r_floor ≤ MAX_R_FLOOR,
    Q3 inlier_frac ≥ MIN_INLIER_FRAC) 中筛每个 ASIN ≥MIN_N_QUALITY_USERS_PER_ASIN 个候选。
 2. 对每个 ASIN,生成 3 个 10-user 子集:
   arm_random: 随机抽 10 (seed=42)
   arm_quality: 前 10 (按 n_sentences 排序,作为 "评论数多" 的代理基线)
   arm_bhatta: max-min Bhattacharyya farthest-point 10
 3. 写 3 个 cohort JSON: stage8_5_asins_<arm>.json 给 Stage 4 分别跑。

Bhattacharyya distance (diagonal covariance, 48d PCA 空间):
 D_B(i,j) = (1/8) Σ_d (μ_i-μ_j)²/(σ_i²+σ_j²)/2
          + (1/2) [log|Σ_avg| - 0.5(log|Σ_i|+log|Σ_j|)]
 其中 Σ_avg[d,d] = (σ_i[d]²+σ_j[d]²)/2
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "common"))

from syntax_subspace_utils import (  # noqa: E402
    MAX_R_FLOOR, MIN_INLIER_FRAC, MIN_N_QUALITY_USERS_PER_ASIN,
    MIN_SIGMA_MEAN, log,
)

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
GAUSSIANS_IN = SCRATCH / "stage8_5_user_gaussians.json"

K_PER_ASIN = 10  # 每个 arm 选 10 个用户
RANDOM_SEED = 42
ARMS = ("random", "quality", "bhatta")


def log(msg: str) -> None:
    print(f"[user_selection] {msg}", flush=True)


def bhattacharyya_distance(mu_i, sd_i, mu_j, sd_j):
    """Diagonal-covariance Bhattacharyya distance between two 48d Gaussians.

    sd_i, sd_j: 1d arrays of length 48 (sigma_diag).
    mu_i, mu_j: 1d arrays of length 48.
    """
    import numpy as np
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    sd_i = np.asarray(sd_i, dtype=np.float64)
    sd_j = np.asarray(sd_j, dtype=np.float64)
    # Avoid log(0) by clamping
    sd_i = np.maximum(sd_i, 1e-12)
    sd_j = np.maximum(sd_j, 1e-12)
    var_avg = 0.5 * (sd_i ** 2 + sd_j ** 2)
    # First term: (1/8) * (mu_i - mu_j)^T * var_avg^{-1} * (mu_i - mu_j)
    delta = mu_i - mu_j
    term1 = 0.125 * np.sum(delta ** 2 / var_avg)
    # Second term: (1/2) * [log|var_avg| - 0.5*(log|var_i|+log|var_j|)]
    log_var_avg = np.sum(np.log(var_avg))
    log_var_i = 2.0 * np.sum(np.log(sd_i))  # log(|Σ_i|) = 2 Σ log(σ_i) (diagonal)
    log_var_j = 2.0 * np.sum(np.log(sd_j))
    term2 = 0.5 * (log_var_avg - 0.5 * (log_var_i + log_var_j))
    return term1 + term2


def max_min_select(user_ids, gaussian_data, k, seed=42):
    """Max-min farthest-point sampling in Bhattacharyya space.

    1. Start with the user with the highest n_sentences (most "central" estimate).
    2. Each next pick: argmax over remaining users of (min Bhattacharyya distance
       to the selected set).

    Returns: list of k user_ids.
    """
    import numpy as np
    if len(user_ids) <= k:
        return list(user_ids)
    # Pre-compute mus and sigmas
    mus = np.array([gaussian_data[u]["mu"] for u in user_ids])
    sigmas = np.array([gaussian_data[u]["sigma_diag"] for u in user_ids])
    n_sents = np.array([gaussian_data[u]["n_sentences"] for u in user_ids])

    selected = []
    selected_idx = set()
    # Pick seed: user with max n_sentences
    seed_idx = int(np.argmax(n_sents))
    selected.append(user_ids[seed_idx])
    selected_idx.add(seed_idx)

    # Distance matrix to selected set (lazy)
    min_d_to_selected = np.full(len(user_ids), np.inf)

    while len(selected) < k:
        last = selected[-1]
        last_idx = user_ids.index(last)
        # Update min distances using newly selected user
        for i in range(len(user_ids)):
            if i in selected_idx:
                continue
            d = bhattacharyya_distance(mus[last_idx], sigmas[last_idx],
                                        mus[i], sigmas[i])
            if d < min_d_to_selected[i]:
                min_d_to_selected[i] = d
        # Pick argmax of min-distance (farthest from all selected)
        candidates = [i for i in range(len(user_ids)) if i not in selected_idx]
        if not candidates:
            break
        next_idx = max(candidates, key=lambda i: min_d_to_selected[i])
        selected.append(user_ids[next_idx])
        selected_idx.add(next_idx)

    return selected


def main():
    log(f"=== User selection (3-arm experiment) ===")
    log(f"K_PER_ASIN={K_PER_ASIN}, MIN_N_QUALITY_USERS_PER_ASIN={MIN_N_QUALITY_USERS_PER_ASIN}")

    log(f"loading cohort {ASINS_IN}")
    cohort = json.load(open(ASINS_IN))
    asins = cohort.get("asins", [])
    log(f"  cohort ASINs: {len(asins)}")

    log(f"loading {GAUSSIANS_IN}")
    gdata = json.load(open(GAUSSIANS_IN))
    users_g = gdata["users"]
    log(f"  quality Gaussian users: {len(users_g)}")

    # For each ASIN: filter quality users, ensure ≥MIN_N_QUALITY_USERS_PER_ASIN
    log(f"\n=== Filtering quality users per ASIN ===")
    asins_eligible = []
    n_qualified_total = 0
    for entry in asins:
        asin = entry["asin"]
        users_all = entry.get("users_sampled", [])
        qualified = [u for u in users_all if u in users_g]
        n_qualified_total += len(qualified)
        if len(qualified) >= MIN_N_QUALITY_USERS_PER_ASIN:
            entry["qualified_users"] = qualified
            entry["n_qualified"] = len(qualified)
            asins_eligible.append(entry)

    log(f"  ASINs with ≥{MIN_N_QUALITY_USERS_PER_ASIN} qualified users: "
        f"{len(asins_eligible)} / {len(asins)}")
    if asins_eligible:
        n_qs = [e["n_qualified"] for e in asins_eligible]
        log(f"  qualified users per ASIN: min={min(n_qs)}, median={sorted(n_qs)[len(n_qs)//2]}, "
            f"max={max(n_qs)}, total={n_qualified_total}")

    # Generate 3 arm cohorts
    rng = random.Random(RANDOM_SEED)
    arm_data = {arm: [] for arm in ARMS}

    log(f"\n=== Generating 3 arm cohorts (K_PER_ASIN={K_PER_ASIN}) ===")
    # 用户指令 2026-08-29: 用户 1.29M cohort, ≥2 quality users 准入后 median=5 users/ASIN。
    # 3 arm 比较需要在每个 ASIN 选相同数量的用户, 所以用 min(K_PER_ASIN, n_qualified)
    # per ASIN。当 n_qualified < K_PER_ASIN 时 3 个 arm 都用全部 qualified users。
    k_actual_dist: list = []
    for entry in asins_eligible:
        asin = entry["asin"]
        qualified = entry["qualified_users"]
        k = min(K_PER_ASIN, len(qualified))
        k_actual_dist.append(k)

        # Arm 1: random k (deterministic via seed=42)
        arm_data["random"].append({
            "asin": asin,
            "n_qualified": entry["n_qualified"],
            "k_actual": k,
            "users_selected": rng.sample(qualified, k),
            "attrs_used": entry.get("attrs_used", {}),
        })

        # Arm 2: quality-only k (top-k by n_sentences)
        sorted_by_n = sorted(qualified,
                              key=lambda u: users_g[u]["n_sentences"],
                              reverse=True)
        arm_data["quality"].append({
            "asin": asin,
            "n_qualified": entry["n_qualified"],
            "k_actual": k,
            "users_selected": sorted_by_n[:k],
            "attrs_used": entry.get("attrs_used", {}),
        })

        # Arm 3: max-min Bhattacharyya k
        selected = max_min_select(qualified, users_g, k, seed=RANDOM_SEED)
        arm_data["bhatta"].append({
            "asin": asin,
            "n_qualified": entry["n_qualified"],
            "k_actual": k,
            "users_selected": selected,
            "attrs_used": entry.get("attrs_used", {}),
        })

    if k_actual_dist:
        log(f"  k_actual per ASIN: min={min(k_actual_dist)}, "
            f"median={sorted(k_actual_dist)[len(k_actual_dist)//2]}, "
            f"max={max(k_actual_dist)} (vs K_PER_ASIN={K_PER_ASIN})")

    log(f"  ASINs per arm: {len(arm_data['random'])}")

    # Write 3 cohort files
    for arm in ARMS:
        out_path = SCRATCH / f"stage8_5_asins_{arm}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "description": (f"3-arm experiment arm={arm}: K_PER_ASIN={K_PER_ASIN}, "
                                    f"MIN_N_QUALITY_USERS_PER_ASIN={MIN_N_QUALITY_USERS_PER_ASIN}"),
                    "arm": arm,
                    "K_PER_ASIN": K_PER_ASIN,
                    "MIN_SIGMA_MEAN": MIN_SIGMA_MEAN,
                    "MAX_R_FLOOR": MAX_R_FLOOR,
                    "MIN_INLIER_FRAC": MIN_INLIER_FRAC,
                    "RANDOM_SEED": RANDOM_SEED,
                },
                "n_asins": len(arm_data[arm]),
                "asins": arm_data[arm],
            }, f, ensure_ascii=False, indent=2)
        log(f"  wrote → {out_path} (n_asins={len(arm_data[arm])})")

    log(f"\n=== Summary ===")
    log(f"  Cohort ASINs: {len(asins)}")
    log(f"  Eligible (≥{MIN_N_QUALITY_USERS_PER_ASIN} quality users): {len(asins_eligible)}")
    log(f"  Per-arm ASINs: {len(arm_data['random'])}")
    log(f"  Per-arm (asin, user) pairs: {len(arm_data['random']) * K_PER_ASIN}")
    log(f"  Output: stage8_5_asins_{{random,quality,bhatta}}.json")

    log(f"=== DONE ===")


if __name__ == "__main__":
    main()