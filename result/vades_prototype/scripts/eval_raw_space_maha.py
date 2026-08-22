#!/usr/bin/env python3
"""Raw-space VADES acceptance eval (Mahalanobis 版) — 使用 user_logvar (Ledoit-Wolf shrunk diag cov) 作为协方差.

核心:
  - user_mu: mean(user's standardized 20d features) [U, 20]
  - user_logvar: log(diag(Ledoit-Wolf shrunk cov)) [U, 20]
  - 距离: Mahalanobis D(u, v; Σ_u) = Σ_k (z_u[k] - z_v[k])^2 / exp(user_logvar[u, k])
    即用 **目标用户** 的协方差作为距离度量 (Bayesian 对应: 后验距离)
  - self_d[i] = Mahalanobis(z_query[i] → user_mu[i] 用 user_logvar[i])
  - cross_d[i] = Mahalanobis(z_query[i] → user_mu[j] 用 user_logvar[i], j != i) 的 min 或 mean

目标 ≥ 0.65 AUC + self-cross gap CI < 0 + permutation p < 0.05 + rank < num_users/2
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

OUT_JSON = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/eval_raw_space_maha.json")


def maha_distance_target(query: np.ndarray, ref_mu: np.ndarray, ref_logvar: np.ndarray) -> np.ndarray:
    """D[i, j] = Σ_k (q[i, k] - ref[j, k])^2 / exp(ref_logvar[j, k])
    query: [N, D], ref_mu: [M, D], ref_logvar: [M, D]
    Returns: [N, M]
    """
    diff = query[:, None, :] - ref_mu[None, :, :]  # [N, M, D]
    sq = diff ** 2
    inv_var = np.exp(-ref_logvar)[None, :, :]  # [1, M, D]
    return (sq * inv_var).sum(axis=2)  # [N, M]


def main():
    log = lambda m: print(f"[raw-eval-maha] {m}", flush=True)

    log("=" * 70)
    log(f"Raw-space VADES (Mahalanobis) evaluation: {OUTPUT_TAG}")
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
    user_logvar = np.array([p["user_logvar"] for p in user_profiles], dtype=np.float32)
    log(f"  num_users: {num_users}")
    log(f"  user_mu shape: {user_mu.shape}")
    log(f"  user_logvar shape: {user_logvar.shape}, mean: {user_logvar.mean():.4f}")

    # === 2. Load sentences ===
    log(f"读取 {SENTENCE_FILE}")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    log(f"  n_sentences: {len(rows)}")

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    feature_names = list(rows[0]["features"].keys())
    feat_array = np.array(
        [[float(r["features"][name]) for name in feature_names] for r in rows],
        dtype=np.float32,
    )

    # === 3. 标准化 raw features (零均值单位方差) ===
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9
    feat_array_norm = (feat_array - feat_mean) / feat_std

    # 把 user_mu 也标准化 (因为 z_u_h1/h2 是标准化特征均值)
    user_mu_norm = (user_mu - feat_mean) / feat_std

    # === 4. 用户级 split-half: h1/h2 都是标准化特征聚合 ===
    user_indices_in_rows = np.array([user_id_to_idx[r["user_id"]] for r in rows])
    for u in range(num_users):
        idx_all = np.where(user_indices_in_rows == u)[0]
        np.random.RandomState(42 + u).shuffle(idx_all)
        if len(idx_all) < 2:
            # 单句用户, h1/h2 都是同一句
            h1 = idx_all
            h2 = idx_all
        else:
            mid = len(idx_all) // 2
            h1 = idx_all[:mid]
            h2 = idx_all[mid:]
        if u == 0:
            log(f"  user 0: {len(idx_all)} sent → h1={len(h1)}, h2={len(h2)}")

    z_u_h1 = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    z_u_h2 = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    for u in range(num_users):
        idx_all = np.where(user_indices_in_rows == u)[0]
        if len(idx_all) < 2:
            z_u_h1[u] = feat_array_norm[idx_all].mean(axis=0)
            z_u_h2[u] = z_u_h1[u]
        else:
            np.random.RandomState(42 + u).shuffle(idx_all)
            mid = len(idx_all) // 2
            z_u_h1[u] = feat_array_norm[idx_all[:mid]].mean(axis=0)
            z_u_h2[u] = feat_array_norm[idx_all[mid:]].mean(axis=0)

    log(f"  z_u_h1 std: {z_u_h1.std(axis=0).mean():.4f}")

    # === 5. Mahalanobis D matrix: D[i,j] = Mahalanobis(z_h1[i] → user_mu[j] 用 user_logvar[j]) ===
    log("计算 Mahalanobis D matrix...")
    D_user = maha_distance_target(z_u_h1, user_mu_norm, user_logvar)  # [U, U]

    self_d_user = np.array([D_user[i, i] for i in range(num_users)])
    # cross_d 用 MEAN (而不是 min, 避免 1 个近用户把 cross 拉成 0)
    cross_d_user_mean = np.array([np.delete(D_user[i], i).mean() for i in range(num_users)])
    cross_d_user_min = np.array([np.delete(D_user[i], i).min() for i in range(num_users)])
    gap_user_mean = self_d_user - cross_d_user_mean  # 负值表示 self < mean cross (GOOD)
    gap_user_min = self_d_user - cross_d_user_min  # 用于参考

    log(f"  user-level self_d mean: {self_d_user.mean():.4f}")
    log(f"  user-level cross_d mean (mean): {cross_d_user_mean.mean():.4f}")
    log(f"  user-level cross_d mean (min): {cross_d_user_min.mean():.4f}")
    log(f"  user-level gap mean (mean cross): {gap_user_mean.mean():.4f} (negative=self closer) **")
    log(f"  user-level gap mean (min cross): {gap_user_min.mean():.4f}")

    # === 6. Bootstrap CI on gap_mean ===
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, num_users, size=num_users)
        boot_gaps.append(gap_user_mean[idx].mean())
    boot_gaps = np.array(boot_gaps)
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
    log(f"  bootstrap 95% CI on gap (mean cross): [{ci_low:.4f}, {ci_high:.4f}]")
    gap_pass = ci_high < 0  # CI 全在 0 之下 = self < mean cross 强显著
    log(f"  ** Acceptance #4 (gap CI below 0): {'PASS' if gap_pass else 'FAIL'} **")

    # === 7. Pairwise AUC: P(self_d < cross_d) 对每用户 ===
    # 用 mask 排除对角线 (而不是 np.delete + axis=1 + np.arange, 那个会删除前 num_users 列)
    mask = np.ones((num_users, num_users), dtype=bool)
    np.fill_diagonal(mask, False)
    self_d_diag = np.diag(D_user)
    cross_d_all = D_user[mask].reshape(num_users, num_users - 1)
    n_better = (cross_d_all > self_d_diag[:, None]).sum(axis=1)  # 注意: maha 距离越大越不像,所以 count cross > self
    auc_pairwise = (n_better / (num_users - 1)).mean()
    log(f"  user-level pairwise AUC (maha, self vs all cross): {auc_pairwise:.4f}")

    # === 8. Permutation test ===
    rng = np.random.default_rng(123)
    n_perm = 1000
    perm_null_gaps = []
    obs_gap = gap_user_mean.mean()
    for _ in range(n_perm):
        perm = rng.permutation(num_users)
        # shuffle: D_user[i, perm[i]] 假装是 self
        perm_self_d = np.array([D_user[i, perm[i]] for i in range(num_users)])
        # cross: D[i, j != perm[i]]
        perm_cross = np.array([
            np.delete(D_user[i], perm[i]).mean() for i in range(num_users)
        ])
        perm_null_gaps.append((perm_self_d - perm_cross).mean())
    perm_null_gaps = np.array(perm_null_gaps)
    p_value = (perm_null_gaps <= obs_gap).mean()  # 观测 gap 比 perm null 更小 (更负) 才是 significant
    log(f"  perm null mean: {perm_null_gaps.mean():.4f}, observed: {obs_gap:.4f}, p={p_value:.4f}")
    perm_pass = p_value < 0.05
    log(f"  ** Acceptance #5 (permutation p < 0.05): {'PASS' if perm_pass else 'FAIL'} **")

    # === 9. Target median rank: 用 train 句聚合 z_query, vs 所有 user_mu 的 Mahalanobis 距离 ===
    user_to_query_idx = {}
    for i, r in enumerate(rows):
        u = user_id_to_idx[r["user_id"]]
        if not r.get("is_holdout", False):
            user_to_query_idx.setdefault(u, []).append(i)

    z_query = np.zeros((num_users, user_mu.shape[1]), dtype=np.float32)
    for u in range(num_users):
        q_idx = user_to_query_idx.get(u, [])
        z_query[u] = feat_array_norm[q_idx].mean(axis=0) if len(q_idx) else user_mu_norm[u]

    D_query = maha_distance_target(z_query, user_mu_norm, user_logvar)  # [U, U]
    ranks = np.zeros(num_users, dtype=np.int64)
    for i in range(num_users):
        order = np.argsort(D_query[i])  # ascending
        rank = np.where(order == i)[0][0]
        ranks[i] = rank
    median_rank = float(np.median(ranks))
    log(f"  target median rank (z_query train → user_mu): {median_rank:.0f} (random ~{num_users/2:.0f})")
    rank_pass = median_rank < num_users / 2
    log(f"  ** Acceptance #6 (target rank < num_users/2): {'PASS' if rank_pass else 'FAIL'} **")

    # === 9b. 严格 split-half 自检 (z_h1 vs z_h2, 不使用 user_mu, 排除数据泄漏) ===
    # 真实 VADES 部署场景: 训练阶段用 user_mu, 推理时需要把"新句子集合"聚合后匹配 user_mu
    # 但**新句子**跟训练句子**不在同一 half**。这里我们把每用户的所有句子分两半, 一半聚合
    # 当 query, 另一半聚合当 ref, 衡量 z_query[i] 能否匹配 z_h2[i] (而不是 user_mu[i])
    log("  [严格 split-half 自检] z_h1 vs z_h2 (排除 user_mu 数据泄漏)...")
    D_split = maha_distance_target(z_u_h1, z_u_h2, user_logvar)  # [U, U] — 用 user_logvar 作为协方差
    mask = np.ones((num_users, num_users), dtype=bool)
    np.fill_diagonal(mask, False)
    self_d_split_diag = np.diag(D_split)
    cross_d_split = D_split[mask].reshape(num_users, num_users - 1)
    n_better_split = (cross_d_split > self_d_split_diag[:, None]).sum(axis=1)
    auc_split = (n_better_split / (num_users - 1)).mean()
    self_d_split = self_d_split_diag
    cross_d_split_mean = cross_d_split.mean(axis=1)
    gap_split = self_d_split - cross_d_split_mean
    ranks_split = np.zeros(num_users, dtype=np.int64)
    for i in range(num_users):
        order = np.argsort(D_split[i])
        ranks_split[i] = np.where(order == i)[0][0]
    median_rank_split = float(np.median(ranks_split))
    log(f"    split-half self_d mean: {self_d_split.mean():.4f}")
    log(f"    split-half cross_d (mean) mean: {cross_d_split_mean.mean():.4f}")
    log(f"    split-half gap mean: {gap_split.mean():.4f}")
    log(f"    split-half AUC: {auc_split:.4f}")
    log(f"    split-half median rank: {median_rank_split:.0f}")

    # === 9c. split-half bootstrap CI on gap 和 AUC (用 users 作为 resampling 单位) ===
    log("  [split-half bootstrap CI] resampling users 1000 times...")
    rng_boot = np.random.default_rng(99)
    n_boot = 1000
    boot_gaps = np.zeros(n_boot)
    boot_aucs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng_boot.integers(0, num_users, size=num_users)
        # gap on bootstrap sample
        boot_gaps[b] = gap_split[idx].mean()
        # AUC: P(cross_d > self_d) on bootstrap sample
        self_b = self_d_split[idx]
        cross_b_all = D_split[idx][:, idx]
        mask_b = np.ones((num_users, num_users), dtype=bool)
        np.fill_diagonal(mask_b, False)
        cross_b = cross_b_all[mask_b].reshape(num_users, num_users - 1)
        boot_aucs[b] = (cross_b > self_b[:, None]).sum(axis=1).mean() / (num_users - 1)
    boot_gaps = np.array(boot_gaps)
    boot_aucs = np.array(boot_aucs)
    gap_ci_low, gap_ci_high = np.percentile(boot_gaps, [2.5, 97.5])
    auc_ci_low, auc_ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    log(f"    split-half gap bootstrap 95% CI: [{gap_ci_low:.4f}, {gap_ci_high:.4f}]")
    log(f"    split-half AUC bootstrap 95% CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")
    split_half_gap_ci_below_zero = gap_ci_high < 0
    split_half_auc_ci_above_065 = auc_ci_low > 0.65
    log(f"    ** split-half gap CI 全在 0 之下: {'PASS' if split_half_gap_ci_below_zero else 'FAIL'} **")
    log(f"    ** split-half AUC CI 完全 ≥ 0.65: {'PASS' if split_half_auc_ci_above_065 else 'FAIL'} **")

    # === 10. AUC vs raw baseline ===
    # D[i, j] = Mahalanobis(z_query[i] → user_mu[j] 用 user_logvar[j])
    # AUC: 对每用户 i, P(other_d > self_d) → P(D[i, j!=i] > D[i, i])
    self_d_query = np.diag(D_query)
    cross_d_query = D_query[mask].reshape(num_users, num_users - 1)
    n_better = (cross_d_query > self_d_query[:, None]).sum(axis=1)
    auc_vs_raw = (n_better / (num_users - 1)).mean()
    log(f"  user-level AUC vs raw baseline (maha, self vs all cross): {auc_vs_raw:.4f} (target >= 0.65)")
    auc_pass = auc_vs_raw >= 0.65
    log(f"  ** Acceptance #3 (AUC >= 0.65): {'PASS' if auc_pass else 'FAIL'} **")

    # === 汇总 ===
    summary = {
        "tag": OUTPUT_TAG,
        "distance_metric": "mahalanobis_with_user_logvar",
        "num_users": num_users,
        "n_sentences": len(rows),
        "user_level_self_d_mean": float(self_d_user.mean()),
        "user_level_cross_d_mean_via_mean": float(cross_d_user_mean.mean()),
        "user_level_cross_d_mean_via_min": float(cross_d_user_min.mean()),
        "user_level_gap_mean_via_mean_cross": float(gap_user_mean.mean()),
        "user_level_gap_ci_via_mean_cross": [float(ci_low), float(ci_high)],
        "user_level_pairwise_auc": float(auc_pairwise),
        "user_level_auc_vs_raw": float(auc_vs_raw),
        "user_level_permutation_p": float(p_value),
        "user_level_target_median_rank": median_rank,
        "split_half_strict_auc": float(auc_split),
        "split_half_strict_gap_mean": float(gap_split.mean()),
        "split_half_strict_median_rank": median_rank_split,
        "split_half_gap_bootstrap_95ci": [float(gap_ci_low), float(gap_ci_high)],
        "split_half_auc_bootstrap_95ci": [float(auc_ci_low), float(auc_ci_high)],
        "split_half_gap_ci_below_zero": "PASS" if split_half_gap_ci_below_zero else "FAIL",
        "split_half_auc_ci_above_065": "PASS" if split_half_auc_ci_above_065 else "FAIL",
        "acceptance_3_user_auc_vs_raw": "PASS" if auc_pass else "FAIL",
        "acceptance_4_user_gap_ci_below_zero": "PASS" if gap_pass else "FAIL",
        "acceptance_5_permutation_p_below_005": "PASS" if perm_pass else "FAIL",
        "acceptance_6_target_rank_below_half": "PASS" if rank_pass else "FAIL",
    }

    log("\n" + "=" * 70)
    log("Raw-space VADES (Mahalanobis) acceptance 评估结果汇总")
    log("=" * 70)
    for k, v in summary.items():
        log(f"  {k}: {v}")

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"已写入 {OUT_JSON}")


if __name__ == "__main__":
    main()