#!/usr/bin/env python3
"""VADES prototype v6 evaluation — 用户级聚合后的 7 条 acceptance criteria.

关键设计 (vs eval_prototype_acceptance.py):
  - 单句 encoder probe 仅作辅助诊断 (用户指令)
  - 用户级 prototype (z_u = mean(encoder mu over user's sentences)) 的 AUC 必须 ≥ 0.65
  - self-cross gap CI > 0 (用户级, 不是单句)
  - permutation p < 0.05 (用户级)
  - target rank 中位数 < num_users/2 (用户级)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian.gaussian_vades import (
    SentenceEncoder, UserDistributionTablePrototype, TeacherProjector,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_prototype_3000u_v6"
ENCODER_CKPT = VADES_DIR / f"vades_encoder_{OUTPUT_TAG}.pt"
USER_TABLE_CKPT = VADES_DIR / f"vades_user_table_{OUTPUT_TAG}.pt"
USER_PROFILE_FILE = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
TEACHER_CKPT = VADES_DIR / f"vades_teacher_{OUTPUT_TAG}.pt"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
RAW_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    """加载 v6 模型 (encoder + user_table, 可选 teacher_proj)."""
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_cluster_ids"].shape[0])
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    user_table = UserDistributionTablePrototype(
        num_users=num_users, latent_dim=LATENT_DIM,
        user_cluster_ids=ut_sd["user_cluster_ids"].long(),
        style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()

    teacher_proj = None
    if TEACHER_CKPT.exists():
        teacher_proj = TeacherProjector(LATENT_DIM, RAW_DIM).to(DEVICE)
        teacher_proj.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False))
        teacher_proj.eval()

    return encoder, user_table, teacher_proj, num_users


def aggregate_user_mu(encoder, user_to_indices: dict[int, list[int]], feature_tensor: torch.Tensor, num_users: int) -> np.ndarray:
    """对每个用户的所有句子做 encoder forward, 聚合 z_u = mean(mu).

    Returns:
        z_u_agg: [num_users, latent_dim] numpy
    """
    z_list = []
    with torch.no_grad():
        for u in range(num_users):
            idx = user_to_indices.get(u, [])
            if not idx:
                z_list.append(np.zeros(LATENT_DIM, dtype=np.float32))
                continue
            x = feature_tensor[idx]
            mu, _, _ = encoder(x)
            z_list.append(mu.mean(dim=0).cpu().numpy())
    return np.stack(z_list, axis=0)


def split_holdout_indices(holdout_mask: np.ndarray, user_indices: np.ndarray, num_users: int):
    """把每个用户的 holdout 句子分成两半 (用于用户级 self-vs-cross 评估)."""
    user_to_holdout: dict[int, list[int]] = {}
    user_to_train: dict[int, list[int]] = {}
    for i in range(len(holdout_mask)):
        u = int(user_indices[i])
        if holdout_mask[i]:
            user_to_holdout.setdefault(u, []).append(i)
        else:
            user_to_train.setdefault(u, []).append(i)
    return user_to_train, user_to_holdout


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """D[i, j] = 1 - cos_sim(A[i], B[j]). shape [len(A), len(B)]"""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return 1.0 - A_n @ B_n.T


def macro_auc_from_distance(D: np.ndarray, y: np.ndarray) -> float:
    """macro-AUC: 对每个 query, AUC(1 if self_d < min_cross_d else 0)."""
    n = D.shape[0]
    aucs = []
    for i in range(n):
        self_d = D[i, y[i]]
        cross_d = np.delete(D[i], y[i])
        n_better = (cross_d < self_d).sum()
        n_total = len(cross_d)
        aucs.append(n_better / n_total)
    return float(np.mean(aucs))


def macro_auc_pairwise(D: np.ndarray, y: np.ndarray) -> float:
    """Pairwise AUC: 对每个 user, 把自己的 z 与其他 user 的 z 比较. D[i,j] 是 user i 与 user j 的距离."""
    n = D.shape[0]
    aucs = []
    for i in range(n):
        self_d = D[i, i]
        cross_d = np.delete(D[i], i)
        n_better = (cross_d < self_d).sum()
        n_total = len(cross_d)
        aucs.append(n_better / n_total)
    return float(np.mean(aucs))


def user_level_eval(encoder, user_table, teacher_proj, num_users):
    """用户级聚合后的评估 (核心)."""
    log = lambda m: print(f"[eval-v6] {m}", flush=True)

    # Load sentences
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    log(f"  sentences loaded: {len(rows)}")

    # Build user_to_indices from sentences
    # 注意 sentence JSON 用 user_id (string), features 是 dict/array
    user_id_to_idx: dict[str, int] = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            user_id_to_idx[r["user_id"]] = i
    log(f"  num_users_in_profiles: {len(user_id_to_idx)}")

    user_to_indices: dict[int, list[int]] = {}
    user_ids_in_rows = []
    is_holdout_flags = []
    for i, r in enumerate(rows):
        u = user_id_to_idx[r["user_id"]]
        user_to_indices.setdefault(u, []).append(i)
        user_ids_in_rows.append(u)
        is_holdout_flags.append(bool(r.get("is_holdout", False)))
    user_ids_in_rows = np.array(user_ids_in_rows)
    is_holdout_flags = np.array(is_holdout_flags)

    log(f"  num_users_in_data: {len(user_to_indices)}")

    # === 1. 全用户聚合 z_u_agg (over all sentences) ===
    log("  聚合 z_u_agg (over all sentences)...")
    feature_names = list(rows[0]["features"].keys())
    log(f"  feature_names: {feature_names[:5]}... ({len(feature_names)} dims)")
    feat_array = np.stack([
        np.array([float(r["features"][name]) for name in feature_names], dtype=np.float32)
        for r in rows
    ], axis=0)
    z_u_all = aggregate_user_mu(encoder, user_to_indices, torch.as_tensor(feat_array, device=DEVICE), num_users)

    # === 2. 用户级 self vs cross: 把 holdout 句子再分两半 ===
    log("  划分 holdout 两半 (用户级 self-cross)...")
    user_to_h1: dict[int, list[int]] = {}
    user_to_h2: dict[int, list[int]] = {}
    for u, idx_list in user_to_indices.items():
        # 把所有 train+holdout 句子分成两半 (模拟 user-level)
        h1 = idx_list[:len(idx_list)//2]
        h2 = idx_list[len(idx_list)//2:]
        user_to_h1[u] = h1
        user_to_h2[u] = h2

    with torch.no_grad():
        feat_tensor = torch.as_tensor(feat_array, device=DEVICE)
        z_u_h1 = np.zeros((num_users, LATENT_DIM), dtype=np.float32)
        z_u_h2 = np.zeros((num_users, LATENT_DIM), dtype=np.float32)
        for u in range(num_users):
            for half, z_out in [(user_to_h1, z_u_h1), (user_to_h2, z_u_h2)]:
                idx = half.get(u, [])
                if idx:
                    mu, _, _ = encoder(feat_tensor[idx])
                    z_out[u] = mu.mean(dim=0).cpu().numpy()

    # === 3. 用户级 self-cross gap: D[i, j] = cos(z_u_h1[i], z_u_h2[j]) ===
    D_user = cosine_distance_matrix(z_u_h1, z_u_h2)
    self_d_user = np.array([D_user[i, i] for i in range(num_users)])
    cross_d_user = np.array([np.delete(D_user[i], i).min() for i in range(num_users)])
    gap_user = self_d_user - cross_d_user  # 负值表示 self < cross (GOOD)
    log(f"  user-level self_d mean: {self_d_user.mean():.4f}")
    log(f"  user-level cross_d mean: {cross_d_user.mean():.4f}")
    log(f"  user-level gap mean: {gap_user.mean():.4f} (negative = self closer)")

    # === 4. Bootstrap CI on user-level gap ===
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, num_users, size=num_users)
        boot_gaps.append(gap_user[idx].mean())
    boot_gaps = np.array(boot_gaps)
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
    log(f"  bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    ci_not_cross_zero = ci_high < 0  # 整个 CI < 0 表示 gap 全为负 (GOOD)
    log(f"  ** Acceptance #4 (user-level gap): {'PASS' if ci_not_cross_zero else 'FAIL'} **")

    # === 5. User-level AUC (pairwise) ===
    # 把 z_u_h1 和 z_u_h2 合并成 D[i, j] = cos distance (用 h1 vs h2 的 full matrix)
    # 这里用 pairwise macro AUC: 对每个 user i, self_d = D[i, i], cross = min(D[i, j!=i])
    n_better = (np.delete(D_user, np.arange(num_users), axis=1) <
                np.diag(D_user)[:, None]).sum(axis=1)
    n_total = num_users - 1
    auc_user_pairwise = (n_better / n_total).mean()
    log(f"  user-level pairwise AUC: {auc_user_pairwise:.4f}")

    # === 6. Permutation test on user-level gap ===
    rng = np.random.default_rng(123)
    n_perm = 1000
    perm_null_gaps = []
    for _ in range(n_perm):
        perm = rng.permutation(num_users)
        perm_self_d = self_d_user
        perm_cross_d = np.array([np.delete(D_user[i], i).min() if i != perm[i] else
                                  np.delete(D_user[i], perm[i]).min()
                                  for i in range(num_users)])
        perm_null_gaps.append((perm_cross_d - perm_self_d).mean())
    perm_null_gaps = np.array(perm_null_gaps)
    obs_gap = (cross_d_user - self_d_user).mean()
    p_value = (np.abs(perm_null_gaps - perm_null_gaps.mean()) >=
               np.abs(obs_gap - perm_null_gaps.mean())).mean()
    log(f"  perm null mean: {perm_null_gaps.mean():.4f}, observed: {obs_gap:.4f}, p={p_value:.4f}")
    log(f"  ** Acceptance #5 (permutation): {'PASS' if p_value < 0.05 else 'FAIL'} **")

    # === 7. Target rank (user-level): 把 query 句子的 z 与所有 user z_u_h2 比较 ===
    # 取 holdout 句子 (每个用户 5 句), 计算 query z_u_h1 vs 所有 user z_u_h2 的距离
    log("  计算 target rank (用户级)...")
    with torch.no_grad():
        # 取每个用户的 5 个 holdout 句子作为 query
        holdout_per_user: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            if r.get("is_holdout"):
                u = user_id_to_idx[r["user_id"]]
                holdout_per_user.setdefault(u, []).append(i)
        # 每个用户的 query = holdout 均值
        z_query = np.zeros((num_users, LATENT_DIM), dtype=np.float32)
        for u in range(num_users):
            idx = holdout_per_user.get(u, [])
            if idx:
                mu, _, _ = encoder(feat_tensor[idx])
                z_query[u] = mu.mean(dim=0).cpu().numpy()
    # D_query[i, j] = cos(z_query[i], z_u_h2[j])
    D_query = cosine_distance_matrix(z_query, z_u_h2)
    ranks = np.zeros(num_users, dtype=np.int64)
    for i in range(num_users):
        # rank of true user (j=i) in row i (lower distance = better rank)
        order = np.argsort(D_query[i])
        rank = np.where(order == i)[0][0]
        ranks[i] = rank
    median_rank = float(np.median(ranks))
    log(f"  target median rank: {median_rank} (random ~{num_users/2:.0f})")
    log(f"  ** Acceptance #6 (user-level rank): {'PASS' if median_rank < num_users/2 else 'FAIL'} **")

    # === 8. AUC against raw baseline (用户级 z_u vs raw_user_proto) ===
    log("  计算 user-level z_u vs raw_user_proto 的 AUC (raw baseline 0.65)...")
    # raw_user_proto 用 holdout 句子的 raw features 均值 (模拟 raw baseline)
    raw_per_user = np.zeros((num_users, RAW_DIM), dtype=np.float32)
    counts = np.zeros(num_users, dtype=np.int64)
    for r in rows:
        if r.get("is_holdout"):
            u = user_id_to_idx[r["user_id"]]
            raw_per_user[u] += np.array([float(r["features"][name]) for name in feature_names], dtype=np.float32)
            counts[u] += 1
    valid = counts > 0
    raw_per_user[valid] /= counts[valid, None]
    # D_z_raw[i, j] = cos(z_u_h1[i], raw_proto[j])
    D_z_raw = cosine_distance_matrix(z_u_h1, raw_per_user)
    n_better = (np.delete(D_z_raw, np.arange(num_users), axis=1) <
                np.diag(D_z_raw)[:, None]).sum(axis=1)
    auc_z_vs_raw = (n_better / (num_users - 1)).mean()
    log(f"  user-level z_u vs raw_user_proto AUC: {auc_z_vs_raw:.4f}")
    log(f"  ** Acceptance #3 (AUC vs raw baseline 0.65): {'PASS' if auc_z_vs_raw >= 0.65 else 'FAIL'} **")

    return {
        "user_level_self_d_mean": float(self_d_user.mean()),
        "user_level_cross_d_mean": float(cross_d_user.mean()),
        "user_level_gap_mean": float((self_d_user - cross_d_user).mean()),
        "user_level_gap_ci": [float(ci_low), float(ci_high)],
        "user_level_pairwise_auc": float(auc_user_pairwise),
        "user_level_auc_vs_raw": float(auc_z_vs_raw),
        "user_level_permutation_p": float(p_value),
        "user_level_target_median_rank": median_rank,
        "acceptance_3_user_auc_vs_raw": "PASS" if auc_z_vs_raw >= 0.65 else "FAIL",
        "acceptance_4_user_gap_ci_positive": "PASS" if ci_not_cross_zero else "FAIL",
        "acceptance_5_permutation": "PASS" if p_value < 0.05 else "FAIL",
        "acceptance_6_target_rank": "PASS" if median_rank < num_users/2 else "FAIL",
    }


def main():
    log = lambda m: print(f"[eval-v6] {m}", flush=True)
    log("=" * 70)
    log(f"VADES prototype v6 用户级评估: {OUTPUT_TAG}")
    log("=" * 70)
    encoder, user_table, teacher_proj, num_users = load_models()
    log(f"  num_users: {num_users}")
    log(f"  teacher_proj: {'loaded' if teacher_proj is not None else 'not loaded (v5/v6 direct mode)'}")

    result = user_level_eval(encoder, user_table, teacher_proj, num_users)

    log("\n" + "=" * 70)
    log("用户级评估结果汇总")
    log("=" * 70)
    for k, v in result.items():
        log(f"  {k}: {v}")

    out_file = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/eval_v6_user_level.json")
    with out_file.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"已写入 {out_file}")


if __name__ == "__main__":
    main()
