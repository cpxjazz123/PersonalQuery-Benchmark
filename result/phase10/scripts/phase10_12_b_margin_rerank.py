#!/usr/bin/env python3
"""Phase 10.12.B: Margin Rerank (D_nearest_other - D_target) verification.

用户洞察: 只优化 D_target 是 limited, 应该优化 margin M(q, u) = D_nearest_other(q) - D_target(q).
让选出的候选不仅靠近 target, 还远离最近的其他 user.

Phase 10.10.7 已经做了 cross-user separation (FAIL in raw 318d space).
本脚本:
1. 用 21,024 句法多样性候选 (Phase 10.12.A)
2. 在 VADES 20d 特征空间 (Phase 10.10.8 baseline) 跑 margin rerank
3. 对比 3 种 rerank 策略:
   A. Min d_target (Phase 10.10 baseline)
   B. Max margin = D_nearest_other - D_target
   C. Weighted: -α·d_target + β·margin (α=β=1)
4. 评估指标:
   - Gap (d_target - d_wrong_5): 应 > 0
   - Win rate: 应 > 60% (Phase 10.10.7 baseline = 60.4%)
   - Target rank percentile: 应 < 50%

如果 win rate > 70%, 证明句法多样性 + margin rerank 突破 60% 上限.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_EVAL = OUT_DIR / "phase10_12_b_margin_rerank.json"
OUT_PER_PAIR = OUT_DIR / "phase10_12_b_per_pair.jsonl"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

TAG = "vades_prototype_3000u_v6_raw"
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"
CANDIDATES_FILE = OUT_DIR / "phase10_12_a_candidates_syntactic.jsonl"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"


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
    log = lambda m: print(f"[phase10-12.B] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.12.B: Margin Rerank Verification")
    log("=" * 70)

    # === Load user profiles + feat_mean/std ===
    log("[1] Loading VADES prototype user profiles ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    log(f"  users: {len(user_id_to_idx)}")

    feat_rows = []
    feature_names = None
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if feature_names is None:
                feature_names = list(r["features"].keys())
            feat_rows.append([float(r["features"][k]) for k in feature_names])
    feat_array = np.array(feat_rows, dtype=np.float32)
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9
    user_mu_norm = (user_mu - feat_mean) / feat_std
    n_users = user_mu_norm.shape[0]

    # === Load candidates ===
    log("[2] Loading candidates ...")
    candidates = []
    with CANDIDATES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")

    pair_lookup = {}
    with PAIRS_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            pair_lookup[(p["user_id"], p["asin"])] = p

    # Group candidates by pair
    cand_by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    # === Extract 20d features for candidates ===
    log(f"[3] Extracting 20d features for {len(candidates)} candidates ...")
    from extract_clause_features_single_query import extract_clause_features
    t0 = time.time()
    cand_feats_norm = np.zeros((len(candidates), len(feature_names)), dtype=np.float64)
    n_failed = 0
    for ci, c in enumerate(candidates):
        try:
            feat_dict = extract_clause_features(c["candidate_query"])
            feat_raw = np.array([float(feat_dict[n]) for n in feature_names], dtype=np.float32)
            cand_feats_norm[ci] = (feat_raw - feat_mean) / feat_std
        except Exception:
            n_failed += 1
    log(f"  done in {time.time()-t0:.1f}s, failed: {n_failed}")

    # === Compute maha distance matrix ===
    log("[4] Computing Mahalanobis distance matrix ...")
    inv_var = np.exp(-user_logvar)
    cand_norm_sq = (cand_feats_norm ** 2) @ inv_var.T
    user_norm_sq = (user_mu_norm ** 2 * inv_var).sum(axis=1)
    weighted_user = inv_var * user_mu_norm
    cross = cand_feats_norm @ weighted_user.T
    dist_matrix = cand_norm_sq + user_norm_sq[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}")

    # === Pre-sample wrong users per pair ===
    log(f"[4.5] Pre-sampling {N_WRONG_USERS_PER_PAIR} wrong users per pair ...")
    rng = np.random.default_rng(SEED)
    wrong_users_per_pair = {}
    for (uid, asin) in cand_by_pair.keys():
        target_idx_global = user_id_to_idx.get(uid)
        if target_idx_global is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx_global] = False
        wrong_idx_global = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)
        wrong_users_per_pair[(uid, asin)] = wrong_idx_global

    # === Per-pair: 3 rerank strategies + cross-user metrics ===
    log("[5] Computing per-pair metrics for 3 rerank strategies ...")
    results = {
        "A_min_d_target": {"gap": {}, "wr": {}, "rank": {}},
        "B_max_margin": {"gap": {}, "wr": {}, "rank": {}},
        "C_weighted": {"gap": {}, "wr": {}, "rank": {}},
    }

    t0 = time.time()
    for (uid, asin), cand_indices in cand_by_pair.items():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_users_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue
        dist_pair = dist_matrix[cand_indices]  # (n_cand, n_users)
        d_target = dist_pair[:, target_idx]
        d_wrong = dist_pair[:, wrong_idx]  # (n_cand, K=5)

        # Compute d_nearest_other per candidate
        masked = dist_pair.copy()
        masked[:, target_idx] = np.inf
        d_nearest_other = masked.min(axis=1)
        margin = d_nearest_other - d_target  # larger = better

        # 3 strategies to pick one candidate
        for strategy, score_fn in [
            ("A_min_d_target", lambda: np.argmin(d_target)),
            ("B_max_margin", lambda: np.argmax(margin)),
            ("C_weighted", lambda: np.argmin(-margin + d_target)),  # high margin, low d_target
        ]:
            pick_idx = score_fn()
            pick_d_target = float(d_target[pick_idx])
            pick_d_wrong = float(d_wrong[pick_idx].mean())
            results[strategy]["gap"][(uid, asin)] = pick_d_target - pick_d_wrong
            results[strategy]["wr"][(uid, asin)] = float(pick_d_target < pick_d_wrong)
            # Target rank of picked candidate
            ranks = np.argsort(np.argsort(dist_pair[pick_idx]))
            results[strategy]["rank"][(uid, asin)] = float(ranks[target_idx]) / n_users

    log(f"  per-pair done in {time.time()-t0:.1f}s")

    # === Aggregate ===
    log("[6] Aggregating ...")
    summary = {}
    for strategy in results:
        m_gap, ci_gap = bootstrap_ci(results[strategy]["gap"])
        m_wr, ci_wr = bootstrap_ci(results[strategy]["wr"])
        m_tr, ci_tr = bootstrap_ci(results[strategy]["rank"])
        summary[strategy] = {
            "n_pairs": len(results[strategy]["gap"]),
            "gap_mean": m_gap,
            "gap_ci_95": list(ci_gap),
            "win_rate_mean": m_wr,
            "win_rate_ci_95": list(ci_wr),
            "target_rank_pct_mean": m_tr,
            "target_rank_pct_ci_95": list(ci_tr),
            "gap_pass": bool(ci_gap[0] > 0),
            "wr_pass_60pct": bool(ci_wr[0] > 0.6),
            "rank_pass_50pct": bool(ci_tr[1] < 0.5),
            "all_pass": bool(ci_gap[0] > 0 and ci_wr[0] > 0.6 and ci_tr[1] < 0.5),
        }
        log(f"\n  Strategy {strategy}:")
        log(f"    Gap: mean={m_gap:.4f}, CI=[{ci_gap[0]:.4f},{ci_gap[1]:.4f}] {'PASS' if ci_gap[0]>0 else 'FAIL'}")
        log(f"    Win rate: mean={m_wr:.4f}, CI=[{ci_wr[0]:.4f},{ci_wr[1]:.4f}] {'PASS' if ci_wr[0]>0.6 else 'FAIL'}")
        log(f"    Target rank: mean={m_tr:.4f}, CI=[{ci_tr[0]:.4f},{ci_tr[1]:.4f}] {'PASS' if ci_tr[1]<0.5 else 'FAIL'}")
        log(f"    ALL PASS: {summary[strategy]['all_pass']}")

    # === Save ===
    log("\n[7] Saving ...")
    eval_dict = {
        "n_pairs": summary["A_min_d_target"]["n_pairs"],
        "n_candidates_per_pair": len(candidates) // summary["A_min_d_target"]["n_pairs"],
        "feature_space": "VADES 20d",
        "n_wrong_users_per_pair": N_WRONG_USERS_PER_PAIR,
        "summary_per_strategy": summary,
        "comparison": {
            "Phase10_10_7_baseline_318d_win_rate": 0.5635,
            "Phase10_10_8_vades_20d_win_rate": 0.6041,
        },
        "note": "Phase 10.12.B: Margin rerank verification. "
                "Strategy A: min d_target (Phase 10.10 baseline). "
                "Strategy B: max margin = d_nearest_other - d_target. "
                "Strategy C: weighted = argmin(-margin + d_target). "
                "Target: win rate > 60% (Phase 10.10.8 baseline).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair (only strategy B for size)
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in results["B_max_margin"]["gap"].keys():
            row = {
                "user_id": uid,
                "asin": asin,
                "A_gap": results["A_min_d_target"]["gap"][(uid, asin)],
                "A_wr": results["A_min_d_target"]["wr"][(uid, asin)],
                "A_rank": results["A_min_d_target"]["rank"][(uid, asin)],
                "B_gap": results["B_max_margin"]["gap"][(uid, asin)],
                "B_wr": results["B_max_margin"]["wr"][(uid, asin)],
                "B_rank": results["B_max_margin"]["rank"][(uid, asin)],
                "C_gap": results["C_weighted"]["gap"][(uid, asin)],
                "C_wr": results["C_weighted"]["wr"][(uid, asin)],
                "C_rank": results["C_weighted"]["rank"][(uid, asin)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.12.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()