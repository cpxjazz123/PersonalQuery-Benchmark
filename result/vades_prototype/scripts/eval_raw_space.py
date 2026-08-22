#!/usr/bin/env python3
"""Raw-space VADES acceptance evaluation — 直接用 raw_user_mu (20d) 评估 7 条 criteria.

设计: 无神经网络, 无 user_table learnable params.
输入: build_raw_space_vades.py 产出的 {tag}_user_profiles.jsonl + {tag}_sentences.jsonl
评估目标 (per user's fallback 指令):
  - user-level AUC vs raw baseline ≥ 0.65
  - user-level self-cross gap CI < 0
  - user-level permutation p < 0.05
  - target median rank < num_users/2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"

OUT_JSON = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/eval_raw_space.json")


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D[i, j] = 1 - cos_sim(A[i], B[j]). shape [len(A), len(B)]"""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return 1.0 - A_n @ B_n.T


def main():
    log = lambda m: print(f"[raw-eval] {m}", flush=True)

    log("=" * 70)
    log(f"Raw-space VADES evaluation: {OUTPUT_TAG}")
    log("=" * 70)

    # === 1. Load user profiles ===
    log(f"读取 {USER_PROFILE_FILE}")
    user_profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            user_profiles.append(json.loads(line))
    num_users = len(user_profiles)
    user_ids = [p["user_id"] for p in user_profiles]
    user_mu = np.array([p["user_mu"] for p in user_profiles], dtype=np.float32)
    log(f"  num_users: {num_users}")
    log(f"  user_mu shape: {user_mu.shape}, std (mean across users): {user_mu.std(axis=0).mean():.4f}")

    # === 2. Load sentences ===
    log(f"读取 {SENTENCE_FILE}")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    log(f"  n_sentences: {len(rows)}")

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # === 3. 用户级 self-cross gap: 把每用户 train+holdout 句分成两半 ===
    # 每用户的 raw_features 都来自 sentences.jsonl
    feature_names = list(rows[0]["features"].keys())
    feat_array = np.stack([
        np.array([float(r["features"][name]) for name in feature_names], dtype=np.float32)
        for r in rows
    ], axis=0)

    user_to_h1: dict[int, list[int]] = {}
    user_to_h2: dict[int, list[int]] = {}
    user_indices_in_rows = []
    is_holdout_flags = []
    for i, r in enumerate(rows):
        u = user_id_to_idx[r["user_id"]]
        user_indices_in_rows.append(u)
        is_holdout_flags.append(bool(r.get("is_holdout", False)))

    user_indices_arr = np.array(user_indices_in_rows)
    is_holdout_arr = np.array(is_holdout_flags)

    # === 标准化 raw features (zero-mean unit-var per-dim) ===
    # 原始 20d raw features 维度间量级差异大 (acl_count 0-3 vs depth_variance 0-30),
    # 必须标准化才能让 cosine 距离有意义. 标准化参数在 train + holdout 全部 43770 句上统计.
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9
    feat_array_norm = (feat_array - feat_mean) / feat_std
    log(f"  feat_mean range: [{feat_mean.min():.2f}, {feat_mean.max():.2f}]")
    log(f"  feat_std range: [{feat_std.min():.2f}, {feat_std.max():.2f}]")

    # 把所有句 (train + holdout) 分两半做 user-level self-cross 评估
    for u in range(num_users):
        idx_all = np.where(user_indices_arr == u)[0]
        np.random.RandomState(42 + u).shuffle(idx_all)
        mid = len(idx_all) // 2
        user_to_h1[u] = idx_all[:mid]
        user_to_h2[u] = idx_all[mid:]

    z_u_h1 = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    z_u_h2 = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    for u in range(num_users):
        h1 = user_to_h1[u]
        h2 = user_to_h2[u]
        # 用标准化后的 features 聚合
        z_u_h1[u] = feat_array_norm[h1].mean(axis=0) if len(h1) else ((user_mu[u] - feat_mean) / feat_std)
        z_u_h2[u] = feat_array_norm[h2].mean(axis=0) if len(h2) else ((user_mu[u] - feat_mean) / feat_std)

    log(f"  z_u_h1 std: {z_u_h1.std(axis=0).mean():.4f}, z_u_h2 std: {z_u_h2.std(axis=0).mean():.4f}")

    # === 4. 用户级 D matrix: D[i,j] = cos(z_u_h1[i], z_u_h2[j]) ===
    D_user = cosine_distance_matrix(z_u_h1, z_u_h2)
    self_d_user = np.array([D_user[i, i] for i in range(num_users)])
    cross_d_user = np.array([np.delete(D_user[i], i).min() for i in range(num_users)])
    gap_user = self_d_user - cross_d_user  # 负值表示 self < cross (GOOD)
    log(f"  user-level self_d mean: {self_d_user.mean():.4f}")
    log(f"  user-level cross_d mean: {cross_d_user.mean():.4f}")
    log(f"  user-level gap mean: {gap_user.mean():.4f} (negative = self closer)")

    # === 5. Bootstrap 95% CI on user-level gap ===
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, num_users, size=num_users)
        boot_gaps.append(gap_user[idx].mean())
    boot_gaps = np.array(boot_gaps)
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
    log(f"  bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    # Acceptance #4: 用户期望 gap mean < 0 且 CI 全在 0 之下 (即 ci_high < 0)
    gap_pass = ci_high < 0
    log(f"  ** Acceptance #4 (user-level gap CI < 0): {'PASS' if gap_pass else 'FAIL'} **")

    # === 6. User-level pairwise AUC (对每个 user i, self_d vs min_cross_d) ===
    n_better = (np.delete(D_user, np.arange(num_users), axis=1) <
                np.diag(D_user)[:, None]).sum(axis=1)
    auc_user_pairwise = (n_better / (num_users - 1)).mean()
    log(f"  user-level pairwise AUC (self vs min cross): {auc_user_pairwise:.4f}")

    # === 7. Permutation test ===
    rng = np.random.default_rng(123)
    n_perm = 1000
    perm_null_gaps = []
    obs_gap = (cross_d_user - self_d_user).mean()  # 正值表示 self < cross (good)
    for _ in range(n_perm):
        perm = rng.permutation(num_users)
        perm_self_d = np.array([D_user[i, perm[i]] for i in range(num_users)])
        perm_cross_d = np.array([
            np.delete(D_user[i], perm[i]).min() for i in range(num_users)
        ])
        perm_null_gaps.append((perm_cross_d - perm_self_d).mean())
    perm_null_gaps = np.array(perm_null_gaps)
    p_value = (perm_null_gaps >= obs_gap).mean()
    log(f"  perm null mean: {perm_null_gaps.mean():.4f}, observed: {obs_gap:.4f}, p={p_value:.4f}")
    perm_pass = p_value < 0.05
    log(f"  ** Acceptance #5 (permutation p < 0.05): {'PASS' if perm_pass else 'FAIL'} **")

    # === 8. Target median rank: 每用户取 train 句均值为 query, vs 所有 user z_u_h2 距离 ===
    user_to_query_idx: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        u = user_id_to_idx[r["user_id"]]
        if not r.get("is_holdout", False):  # 用 train 句做 query
            user_to_query_idx.setdefault(u, []).append(i)

    z_query = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    for u in range(num_users):
        q_idx = user_to_query_idx.get(u, [])
        z_query[u] = feat_array_norm[q_idx].mean(axis=0) if len(q_idx) else ((user_mu[u] - feat_mean) / feat_std)

    D_query = cosine_distance_matrix(z_query, z_u_h2)
    ranks = np.zeros(num_users, dtype=np.int64)
    for i in range(num_users):
        order = np.argsort(D_query[i])
        rank = np.where(order == i)[0][0]
        ranks[i] = rank
    median_rank = float(np.median(ranks))
    log(f"  target median rank: {median_rank:.0f} (random ~{num_users/2:.0f})")
    rank_pass = median_rank < num_users / 2
    log(f"  ** Acceptance #6 (target rank < num_users/2): {'PASS' if rank_pass else 'FAIL'} **")

    # === 9. AUC vs raw baseline (raw_user_mu 直接作为 user-level prototype) ===
    # raw_user_mu 已计算 (user_mu 矩阵, 原始 20d)
    # 标准化后的 user_mu 才能跟标准化后的 query 比较
    user_mu_norm = (user_mu - feat_mean) / feat_std
    D_z_raw = cosine_distance_matrix(z_query, user_mu_norm)
    n_better = (np.delete(D_z_raw, np.arange(num_users), axis=1) <
                np.diag(D_z_raw)[:, None]).sum(axis=1)
    auc_z_vs_raw = (n_better / (num_users - 1)).mean()
    log(f"  user-level AUC vs raw_user_proto: {auc_z_vs_raw:.4f} (target >= 0.65)")
    auc_pass = auc_z_vs_raw >= 0.65
    log(f"  ** Acceptance #3 (AUC vs raw baseline 0.65): {'PASS' if auc_pass else 'FAIL'} **")

    # === 汇总 ===
    summary = {
        "tag": OUTPUT_TAG,
        "num_users": num_users,
        "n_sentences": len(rows),
        "user_level_self_d_mean": float(self_d_user.mean()),
        "user_level_cross_d_mean": float(cross_d_user.mean()),
        "user_level_gap_mean": float(gap_user.mean()),
        "user_level_gap_ci": [float(ci_low), float(ci_high)],
        "user_level_pairwise_auc": float(auc_user_pairwise),
        "user_level_auc_vs_raw": float(auc_z_vs_raw),
        "user_level_permutation_p": float(p_value),
        "user_level_target_median_rank": median_rank,
        "acceptance_3_user_auc_vs_raw": "PASS" if auc_pass else "FAIL",
        "acceptance_4_user_gap_ci_below_zero": "PASS" if gap_pass else "FAIL",
        "acceptance_5_permutation_p_below_005": "PASS" if perm_pass else "FAIL",
        "acceptance_6_target_rank_below_half": "PASS" if rank_pass else "FAIL",
    }

    log("\n" + "=" * 70)
    log("Raw-space VADES acceptance 评估结果汇总")
    log("=" * 70)
    for k, v in summary.items():
        log(f"  {k}: {v}")

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()