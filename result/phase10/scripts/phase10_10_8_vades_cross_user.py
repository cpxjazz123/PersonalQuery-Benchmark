#!/usr/bin/env python3
"""Phase 10.10.8: Cross-user separation with VADES 20d feature space.

Phase 10.10.7 发现 raw 318d 句法特征空间 cross-user separation 全部 FAIL:
- Gap (d_target - d_wrong) < 0
- Win rate ~ 50-60%
- Target rank percentile ~ 50%

本脚本对比验证 VADES prototype 训练用的 20d 特征空间:
1. 加载 vades_prototype_3000u_v6_user_profiles.jsonl (每个用户 20d user_mu)
2. 加载训练集 _raw_sentences.jsonl (每条 20d features, 用于算 feat_mean/feat_std)
3. 对每个 candidate 用 extract_clause_features 提取 20d 特征, 标准化
4. 计算 candidate 到 target user_maha 与 到 K 个 wrong users_maha 的差
5. bootstrap CI

对比 Phase 10.10.7 的 318d 测试:
- 318d: Gap<0, win~50%, rank~50%
- 如果 20d 也 FAIL: 这是固有局限
- 如果 20d PASS: 318d 失败是因为维度太多反而引入噪声

注意: user_mu 在 20d 特征空间(不是 latent 空间), 所以这是 20d 特征空间的 cross-user test.
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
OUT_EVAL = OUT_DIR / "phase10_10_8_vades_cross_user.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_8_per_pair.jsonl"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

# VADES prototype outputs
TAG = "vades_prototype_3000u_v6_raw"
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"
CANDIDATES_FILE = OUT_DIR / "phase10_10_candidates_n10.jsonl"
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


def maha_one(query_norm: np.ndarray, user_mu_n: np.ndarray, user_logvar: np.ndarray) -> float:
    diff = query_norm - user_mu_n
    return float(((diff ** 2) * np.exp(-user_logvar)).sum())


def main():
    log = lambda m: print(f"[phase10-10.8] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.10.8: Cross-user separation with VADES 20d feature space")
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
    log(f"  users: {len(user_id_to_idx)}, user_mu: {user_mu.shape}, user_logvar: {user_logvar.shape}")

    # feat_mean/std from training sentences
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
    log(f"  feat_mean: {feat_mean.shape}, n_features={len(feature_names)}")
    log(f"  first 5 feature names: {feature_names[:5]}")

    # Normalize user_mu (this is what was used to compute z_target_norm)
    user_mu_norm = (user_mu - feat_mean) / feat_std
    log(f"  user_mu_norm: {user_mu_norm.shape}")

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
    log(f"  pairs: {len(pair_lookup)}")

    # === Group by pair ===
    cand_by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    n_users = len(user_id_to_idx)
    rng = np.random.default_rng(SEED)

    # === Pre-sample K=5 wrong users per pair ===
    log(f"[2.5] Pre-sampling {N_WRONG_USERS_PER_PAIR} wrong users per pair ...")
    wrong_users_per_pair = {}
    for (uid, asin) in cand_by_pair.keys():
        target_idx_global = user_id_to_idx.get(uid)
        if target_idx_global is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx_global] = False
        wrong_idx_global = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)
        wrong_users_per_pair[(uid, asin)] = wrong_idx_global

    # === Extract 20d features for all candidates ===
    log(f"[3] Extracting 20d features for {len(candidates)} candidates ...")
    from extract_clause_features_single_query import extract_clause_features
    t0 = time.time()
    cand_feats_norm = np.zeros((len(candidates), len(feature_names)), dtype=np.float64)
    cand_logvars = np.zeros(len(candidates), dtype=np.float64)
    n_failed = 0
    for ci, c in enumerate(candidates):
        try:
            feat_dict = extract_clause_features(c["candidate_query"])
            feat_raw = np.array([float(feat_dict[n]) for n in feature_names], dtype=np.float32)
            feat_n = (feat_raw - feat_mean) / feat_std
            cand_feats_norm[ci] = feat_n
        except Exception as e:
            n_failed += 1
            cand_feats_norm[ci] = 0.0
    log(f"  extracted in {time.time()-t0:.1f}s, failed: {n_failed}")
    log(f"  cand_feats_norm: {cand_feats_norm.shape}")

    # === Compute maha distance matrix (candidates × users) using user_logvar ===
    log("[4] Computing Mahalanobis distance matrix (diag) ...")
    inv_var = np.exp(-user_logvar)  # (n_users, 20)
    # maha_{ci, ui} = sum_d (feat_n[ci, d] - user_mu_norm[ui, d])^2 * inv_var[ui, d]
    # Decomposition: maha = (cand - user_mu)^2 @ inv_var
    #              = (cand^2) @ inv_var + (user_mu^2 * inv_var).sum - 2 cand @ (inv_var * user_mu).T
    cand_norm_sq = (cand_feats_norm ** 2) @ inv_var.T  # (n_cand, n_users)
    user_norm_sq = (user_mu_norm ** 2 * inv_var).sum(axis=1)  # (n_users,)
    weighted_user = inv_var * user_mu_norm  # (n_users, 20)
    cross = cand_feats_norm @ weighted_user.T  # (n_cand, n_users)
    dist_matrix = cand_norm_sq + user_norm_sq[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}")

    # === Per-pair cross-user metrics ===
    log("[5] Computing per-pair cross-user metrics ...")
    gap_per_pair = {}
    win_rate_per_pair = {}
    target_rank_per_pair = {}

    t0 = time.time()
    for (uid, asin), cand_indices in cand_by_pair.items():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_users_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue
        dist_pair = dist_matrix[cand_indices]  # (n_cand, n_users)
        d_target = dist_pair[:, target_idx]  # (n_cand,)
        d_wrong = dist_pair[:, wrong_idx]  # (n_cand, K=5)

        # (1) Gap
        per_cand_gap = d_target - d_wrong.mean(axis=1)
        gap_per_pair[(uid, asin)] = float(per_cand_gap.mean())
        # (2) Win rate
        win_rate_per_pair[(uid, asin)] = float((d_target < d_wrong.mean(axis=1)).mean())
        # (3) Target rank percentile
        ranks = np.argsort(np.argsort(dist_pair, axis=1), axis=1)
        target_ranks = ranks[:, target_idx]
        target_rank_per_pair[(uid, asin)] = float(target_ranks.mean()) / n_users
    log(f"  per-pair done in {time.time()-t0:.1f}s")

    # === Aggregate ===
    log("[6] Aggregating ...")
    m_gap, ci_gap = bootstrap_ci(gap_per_pair)
    m_wr, ci_wr = bootstrap_ci(win_rate_per_pair)
    m_tr, ci_tr = bootstrap_ci(target_rank_per_pair)

    log(f"  Gap (d_target - d_wrong): mean={m_gap:.4f}, CI=[{ci_gap[0]:.4f},{ci_gap[1]:.4f}]")
    log(f"  Win rate: mean={m_wr:.4f}, CI=[{ci_wr[0]:.4f},{ci_wr[1]:.4f}]")
    log(f"  Target rank percentile: mean={m_tr:.4f}, CI=[{ci_tr[0]:.4f},{ci_tr[1]:.4f}]")

    gap_pass = bool(ci_gap[0] > 0)
    wr_pass = bool(ci_wr[0] > 0.5)
    rank_pass = bool(ci_tr[1] < 0.5)
    log(f"  PASS conditions:")
    log(f"    gap_ci_above_zero: {gap_pass}")
    log(f"    win_rate_ci_above_50pct: {wr_pass}")
    log(f"    target_rank_ci_below_50pct: {rank_pass}")
    all_pass = gap_pass and wr_pass and rank_pass
    log(f"  ALL PASS: {all_pass}")

    # === Save ===
    log("\n[7] Saving results ...")
    eval_dict = {
        "n_pairs": len(gap_per_pair),
        "n_wrong_users_per_pair": N_WRONG_USERS_PER_PAIR,
        "feature_space": "VADES 20d clause_features (raw feature space, NOT latent)",
        "user_profiles_source": str(USER_PROFILE_FILE),
        "feature_names": feature_names,
        "distance_kind": "mahalanobis_diag_inv_var=user_logvar",
        "gap_target_vs_wrong_mean": m_gap,
        "gap_target_vs_wrong_ci_95": list(ci_gap),
        "gap_ci_above_zero": gap_pass,
        "win_rate_mean": m_wr,
        "win_rate_ci_95": list(ci_wr),
        "win_rate_ci_above_50pct": wr_pass,
        "target_rank_percentile_mean": m_tr,
        "target_rank_percentile_ci_95": list(ci_tr),
        "target_rank_ci_below_50pct": rank_pass,
        "PASS_all": all_pass,
        "comparison_to_phase10_10_7": {
            "phase10_10_7_318d_raw_l2_d5_gap": -0.0120,
            "phase10_10_7_318d_raw_l2_d5_wr": 0.5635,
            "phase10_10_7_318d_raw_l2_d5_rank": 0.4887,
        },
        "note": "Phase 10.10.8: VADES 20d feature space (NOT latent) cross-user separation. "
                "user_mu/user_logvar 在 20d clause_features 空间, 候选经同样化算 maha. "
                "如果 PASS: 20d 特征空间 (VADES proto 训练用) 能区分 target vs wrong. "
                "如果 FAIL: VADES 也无法区分用户, Phase 10.10 rerank 是 within-pool-only.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin), gap in gap_per_pair.items():
            row = {
                "user_id": uid,
                "asin": asin,
                "gap": gap,
                "win_rate": win_rate_per_pair[(uid, asin)],
                "target_rank_percentile": target_rank_per_pair[(uid, asin)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.10.8 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()