#!/usr/bin/env python3
"""Phase 10.15: Target rank-1 coverage evaluation.

真正的成功标准:
  - 选出的候选 margin > 0 (即比 nearest other user 更近 target)
  - target user 在所有用户中 rank=1 (即最近的就是 target)
  - 属性完整 (attr_pass)
  - 语义保留 (sem_sim >= 0.3)

新指标:
  - rank-1 coverage: target 在所有用户中排名第 1 的 pair 比例
  - rank-3 coverage: target 排名前 3
  - rank-10 coverage: target 排名前 10
  - mean rank percentile: target 平均排名百分位

实施步骤:
  1. 复用 phase10_14_bc 已缓存的 candidates + 318d features
  2. 重新计算 per-pair 全距离矩阵 (n_cands=96 × n_users=3000)
  3. 找到 max-margin 候选, 检查其 rank
  4. 报告 rank-1/3/10 coverage
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_CANDIDATES = OUT_DIR / "phase10_14_bc_candidates_96.jsonl"
OUT_CAND_FEATS_CACHE = OUT_DIR / "phase10_14_bc_candidates_318d.npy"
OUT_EVAL = OUT_DIR / "phase10_15_rank1_eval.json"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"


def bootstrap_ci(values_per_item: Dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    items = list(values_per_item.keys())
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0)
    obs_vals = np.array([values_per_item[k] for k in items])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals = obs_vals[sample_idx]
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high)


def main():
    log = lambda m: print(f"[phase10-15] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.15: Target Rank-1 Coverage Evaluation")
    log("=" * 70)

    # === Load candidates + features ===
    candidates = []
    with OUT_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")
    cand_feats_keep = np.load(OUT_CAND_FEATS_CACHE)
    log(f"  318d features: {cand_feats_keep.shape}")

    # === Load profiles + sentence features ===
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  VADES users: {n_users}")

    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    log(f"  effective dims: {n_dims_eff}")

    # === Per-user mean features ===
    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for si in range(len(sent_users)):
        ui = user_id_to_idx.get(sent_users[si])
        if ui is None:
            continue
        user_mu_318d[ui] += sent_feats_keep[si]
        counts[ui] += 1
    valid_user_mask = counts > 0
    user_mu_318d[valid_user_mask] /= counts[valid_user_mask, None]
    log(f"  valid users: {valid_user_mask.sum()}/{n_users}")

    # === Ledoit-Wolf covariance ===
    log("[1] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage: {shrinkage:.4f}")

    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)

    # === Group candidates by pair ===
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs: {len(cand_by_pair)}")

    # === Per-pair full distance matrix + rank computation ===
    log("[2] Per-pair full distance matrix + rank computation ...")
    pair_data = {}
    t0 = time.time()

    n_rank1 = 0
    n_rank3 = 0
    n_rank5 = 0
    n_rank10 = 0
    n_margin_pos = 0
    n_pass_all = 0  # rank1 + margin > 0 + attr_pass

    for pi, ((uid, asin), cand_indices) in enumerate(cand_by_pair.items()):
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue

        cands_feats_local = cand_feats_keep[cand_indices]
        n_cands = len(cand_indices)

        # Full distance matrix: n_cands × n_users
        cross = cands_feats_local @ inv_cov @ user_mu_318d.T
        quad_cand = np.einsum('ij,jk,ik->i', cands_feats_local, inv_cov, cands_feats_local)
        dist_local = quad_cand[:, None] + quad_mu[None, :] - 2 * cross

        d_target = dist_local[:, target_idx]  # (n_cands,)

        # For each cand: rank of target_idx (0-indexed)
        # target_rank[i] = number of users with smaller distance than target for cand i
        target_d = d_target[:, None]  # (n_cands, 1)
        target_rank = (dist_local < target_d).sum(axis=1)  # (n_cands,)
        # target_rank == 0 means target is the nearest user

        # Best cand by max-margin (using K=5 wrong user mean)
        rng = np.random.default_rng(SEED)
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx] = False
        wrong_idx = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)
        d_wrong = dist_local[:, wrong_idx]
        d_wrong_mean = d_wrong.mean(axis=1)
        margin = d_wrong_mean - d_target
        best_idx = int(np.argmax(margin))

        # Stats for best candidate
        best_rank = int(target_rank[best_idx])
        best_margin = float(margin[best_idx])
        best_d_target = float(d_target[best_idx])
        # Best cand's nearest_other distance
        masked_best = dist_local[best_idx].copy()
        masked_best[target_idx] = np.inf
        nearest_other_d = float(masked_best.min())
        nearest_other_idx = int(masked_best.argmin())

        # Track rank-1 stats for ANY cand (not just best)
        any_rank1 = int((target_rank == 0).any())
        n_rank1 += any_rank1
        n_rank3 += int((target_rank < 3).any())
        n_rank5 += int((target_rank < 5).any())
        n_rank10 += int((target_rank < 10).any())
        if best_margin > 0:
            n_margin_pos += 1
        if best_rank == 0 and best_margin > 0:
            n_pass_all += 1

        # Also pick best by max-margin AMONG rank-1 cands
        rank1_mask = (target_rank == 0)
        if rank1_mask.any():
            margin_among_rank1 = np.where(rank1_mask, margin, -np.inf)
            rank1_pick = int(np.argmax(margin_among_rank1))
            rank1_pick_margin = float(margin[rank1_pick])
            rank1_pick_d_target = float(d_target[rank1_pick])
        else:
            rank1_pick = None
            rank1_pick_margin = None
            rank1_pick_d_target = None

        pair_data[(uid, asin)] = {
            "n_cands": n_cands,
            "best_idx": best_idx,
            "best_rank": best_rank,
            "best_margin": best_margin,
            "best_d_target": best_d_target,
            "nearest_other_d": nearest_other_d,
            "nearest_other_idx": nearest_other_idx,
            "any_rank1": any_rank1,
            "any_rank3": int((target_rank < 3).any()),
            "rank1_pick_idx": rank1_pick,
            "rank1_pick_margin": rank1_pick_margin,
            "rank1_pick_d_target": rank1_pick_d_target,
        }

        if pi % 100 == 0:
            elapsed = time.time() - t0
            rate = (pi + 1) / max(elapsed, 0.001)
            eta = (len(cand_by_pair) - pi) / max(rate, 0.001)
            log(f"  [{pi}/{len(cand_by_pair)}] elapsed {elapsed:.1f}s, rate={rate:.1f}/s, ETA={eta:.0f}s")
        del dist_local

    log(f"  done in {time.time()-t0:.1f}s, {len(pair_data)} pairs")

    # === Aggregate stats ===
    n_total = len(pair_data)
    log(f"\n  Total pairs: {n_total}")
    log(f"\n  === Coverage Stats (any cand in pool achieves) ===")
    log(f"  Rank-1 coverage: {n_rank1}/{n_total} = {100*n_rank1/n_total:.1f}%")
    log(f"  Rank-3 coverage: {n_rank3}/{n_total} = {100*n_rank3/n_total:.1f}%")
    log(f"  Rank-5 coverage: {n_rank5}/{n_total} = {100*n_rank5/n_total:.1f}%")
    log(f"  Rank-10 coverage: {n_rank10}/{n_total} = {100*n_rank10/n_total:.1f}%")

    log(f"\n  === Best-by-max-margin cand stats ===")
    log(f"  Best margin > 0: {n_margin_pos}/{n_total} = {100*n_margin_pos/n_total:.1f}%")
    log(f"  Best rank=1 + margin > 0: {n_pass_all}/{n_total} = {100*n_pass_all/n_total:.1f}%")

    # Bootstrap CIs
    per_pair_best_rank = {k: v["best_rank"] for k, v in pair_data.items()}
    per_pair_best_margin = {k: v["best_margin"] for k, v in pair_data.items()}
    per_pair_any_rank1 = {k: v["any_rank1"] for k, v in pair_data.items()}
    per_pair_any_rank3 = {k: v["any_rank3"] for k, v in pair_data.items()}
    per_pair_nearest_other_d = {k: v["nearest_other_d"] for k, v in pair_data.items()}
    per_pair_nearest_other_idx = {k: v["nearest_other_idx"] for k, v in pair_data.items()}

    mean_rank, ci_rank = bootstrap_ci(per_pair_best_rank)
    mean_margin, ci_margin = bootstrap_ci(per_pair_best_margin)
    mean_any_rank1, ci_any_rank1 = bootstrap_ci(per_pair_any_rank1)
    mean_any_rank3, ci_any_rank3 = bootstrap_ci(per_pair_any_rank3)
    mean_nearest_other_d, ci_nearest_other_d = bootstrap_ci(per_pair_nearest_other_d)

    log(f"\n  Bootstrap 95% CI:")
    log(f"    mean best rank: {mean_rank:.2f} [{ci_rank[0]:.2f}, {ci_rank[1]:.2f}]")
    log(f"    mean best margin: {mean_margin:.1f} [{ci_margin[0]:.1f}, {ci_margin[1]:.1f}]")
    log(f"    any rank-1: {mean_any_rank1:.4f} [{ci_any_rank1[0]:.4f}, {ci_any_rank1[1]:.4f}]")
    log(f"    any rank-3: {mean_any_rank3:.4f} [{ci_any_rank3[0]:.4f}, {ci_any_rank3[1]:.4f}]")

    # === Rank distribution of best cand ===
    best_rank_counts = defaultdict(int)
    for v in pair_data.values():
        best_rank_counts[v["best_rank"]] += 1
    log(f"\n  Best cand rank distribution:")
    for r in sorted(best_rank_counts.keys())[:20]:
        if r == 0:
            log(f"    rank=0 (rank-1): {best_rank_counts[r]}/{n_total} = {100*best_rank_counts[r]/n_total:.1f}%")
        elif r < 10:
            log(f"    rank={r}: {best_rank_counts[r]}/{n_total} = {100*best_rank_counts[r]/n_total:.1f}%")
        elif r == 10:
            log(f"    rank={r}: {best_rank_counts[r]}")
    log(f"    rank>=20: {sum(c for r, c in best_rank_counts.items() if r >= 20)}")

    # === Save eval ===
    log("\n[3] Saving eval ...")
    eval_dict = {
        "n_pairs_total": n_total,
        "n_candidates_per_pair": 96,
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage",
        "shrinkage_intensity": float(shrinkage),
        "coverage_any_rank1": n_rank1 / n_total,
        "coverage_any_rank3": n_rank3 / n_total,
        "coverage_any_rank5": n_rank5 / n_total,
        "coverage_any_rank10": n_rank10 / n_total,
        "best_cand_margin_positive": n_margin_pos / n_total,
        "best_cand_rank1_and_margin_positive": n_pass_all / n_total,
        "mean_best_rank": mean_rank,
        "mean_best_rank_ci_95": list(ci_rank),
        "mean_best_margin": mean_margin,
        "mean_best_margin_ci_95": list(ci_margin),
        "mean_any_rank1": mean_any_rank1,
        "mean_any_rank1_ci_95": list(ci_any_rank1),
        "best_rank_distribution": {str(k): v for k, v in sorted(best_rank_counts.items())},
        "comparison": {
            "Phase10_14_A_P95_coverage": 0.938,
            "Phase10_14_C_96cand_P75_coverage": 0.965,
            "Phase10_13_A_318d_real_WR": 0.9786,
        },
        "note": "Phase 10.15: True target rank-1 coverage. Acceptance = target user is "
                "rank-1 (closest) for at least one cand in pool. This is the strictest "
                "criterion — query is genuinely closest to target user, not just in "
                "target's general neighborhood.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    log("=" * 70)
    log("PHASE 10.15 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()