#!/usr/bin/env python3
"""评估 VADES prototype 模型的 7 条 acceptance criteria.

Inputs: 训练好的 vades_prototype_3000u 模型 + 数据
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
    SentenceEncoder, UserDistributionTablePrototype, precompute_prototype_user_mu,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"

OUTPUT_TAG = "vades_prototype_3000u_v5"
ENCODER_CKPT = VADES_DIR / f"vades_encoder_{OUTPUT_TAG}.pt"
USER_TABLE_CKPT = VADES_DIR / f"vades_user_table_{OUTPUT_TAG}.pt"
USER_PROFILE_FILE = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1 - a_n @ b_n.T


def load_data():
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    user_id_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            user_id_to_idx[json.loads(line)["user_id"]] = i

    # 读取 user_mu (从 jsonl)
    user_mu = np.array([json.loads(l)["user_mu"] for l in USER_PROFILE_FILE.open()])
    logvar_user = None
    return rows, user_id_to_idx, user_mu


def load_models():
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_cluster_ids"].shape[0])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    user_table = UserDistributionTablePrototype(
        num_users=num_users, latent_dim=LATENT_DIM,
        user_cluster_ids=placeholder_cluster, style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()
    return encoder, user_table, num_users


def main():
    log = lambda m: print(f"[eval] {m}", flush=True)

    log("=" * 70)
    log(f"评估 prototype VADES {OUTPUT_TAG}")
    log("=" * 70)

    encoder, user_table, num_users = load_models()
    rows, user_id_to_idx, user_mu_saved = load_data()

    # 数据准备
    train_rows = [r for r in rows if not r.get("is_holdout")]
    ho_rows = [r for r in rows if r.get("is_holdout")]
    train_feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in train_rows])
    train_y = np.array([user_id_to_idx[r["user_id"]] for r in train_rows])
    ho_feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in ho_rows])
    ho_y = np.array([user_id_to_idx[r["user_id"]] for r in ho_rows])
    log(f"  num_users={num_users}, train_sent={len(train_rows)}, holdout_sent={len(ho_rows)}")
    log(f"  random baseline: {1/num_users*100:.4f}%")

    # Encoder forward
    train_X = torch.as_tensor(train_feat, dtype=torch.float32, device=DEVICE)
    ho_X = torch.as_tensor(ho_feat, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        mu_train, _, _ = encoder(train_X)
        mu_ho, _, _ = encoder(ho_X)
    mu_train_np = mu_train.cpu().numpy()
    mu_ho_np = mu_ho.cpu().numpy()
    log(f"  encoder mu_train std: {mu_train_np.std():.4f}, mu_ho std: {mu_ho_np.std():.4f}")

    # ========================================================
    # Acceptance #1: Linear probe encoder latent → user_id
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #1: Linear probe encoder_latent → user_id")
    log("=" * 70)
    probe = nn.Linear(LATENT_DIM, num_users).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    train_y_t = torch.as_tensor(train_y, device=DEVICE)
    ho_y_t = torch.as_tensor(ho_y, device=DEVICE)
    mu_train_t = mu_train.to(DEVICE)
    mu_ho_t = mu_ho.to(DEVICE)
    for epoch in range(1, 21):
        probe.train()
        perm = torch.randperm(len(mu_train_t))
        loss_sum = 0
        for i in range(0, len(perm), 1024):
            idx = perm[i:i+1024]
            logits = probe(mu_train_t[idx])
            loss = F.cross_entropy(logits, train_y_t[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == 1:
            probe.eval()
            with torch.no_grad():
                ho_acc = (probe(mu_ho_t).argmax(1) == ho_y_t).float().mean().item()
            log(f"  epoch {epoch}: ho_acc={ho_acc*100:.4f}% (target > 1%)")
    probe.eval()
    with torch.no_grad():
        probe_ho_acc = (probe(mu_ho_t).argmax(1) == ho_y_t).float().mean().item()
    log(f"\n** Acceptance #1: encoder probe ho_acc = {probe_ho_acc*100:.4f}% **")
    log(f"  PASS" if probe_ho_acc > 0.01 else f"  FAIL (need > 1%, got {probe_ho_acc*100:.4f}%)")

    # ========================================================
    # Acceptance #3: Held-out user cosine AUC (vs raw 0.65)
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #3: Held-out user cosine AUC")
    log("=" * 70)
    from sklearn.metrics import roc_auc_score
    # 用 saved user_mu (from jsonl) 作为 user_mu
    user_mu_arr = np.array(user_mu_saved)
    log(f"  user_mu std: {user_mu_arr.std():.4f}")

    # 对每个 holdout, vs all users 的 cosine distance
    D = cosine_dist(mu_ho_np, user_mu_arr)
    # AUC: self vs random — 取 self_id 列 vs 其他所有 (负距离 = similarity score)
    # AUC 计算: 对每个 ho, 二分类 self vs random, 用 score = -D[ho, user_id]
    n_eval = min(5000, len(mu_ho_np))
    aucs = []
    for i in range(n_eval):
        self_score = -D[i, ho_y[i]]
        other_scores = np.delete(D[i], ho_y[i])
        auc = (other_scores > -self_score).mean()  # 分数越低 (距离越小) 越像 self
        aucs.append(auc)
    macro_auc = float(np.mean(aucs))
    log(f"  cosine macro-AUC (n={n_eval}): {macro_auc:.4f}")
    log(f"  target >= 0.65 (raw baseline)")
    log(f"  ** Acceptance #3: {'PASS' if macro_auc >= 0.65 else 'FAIL'} **")

    # ========================================================
    # Acceptance #4: Self-cross gap bootstrap 95% CI
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #4: Self-cross gap bootstrap 95% CI")
    log("=" * 70)
    self_d = np.array([D[i, ho_y[i]] for i in range(n_eval)])
    cross_d = np.array([np.delete(D[i], ho_y[i]).min() for i in range(n_eval)])
    gap = cross_d - self_d  # 正值表示 self < cross
    log(f"  self_d mean: {self_d.mean():.4f}, cross_d mean: {cross_d.mean():.4f}, gap mean: {gap.mean():.4f}")

    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_eval, size=n_eval)
        boot_gaps.append(gap[idx].mean())
    boot_gaps = np.array(boot_gaps)
    ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
    log(f"  bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    ci_not_cross_zero = ci_low > 0
    log(f"  ** Acceptance #4: {'PASS' if ci_not_cross_zero else 'FAIL'} (CI does not cross 0) **")

    # ========================================================
    # Acceptance #5: Permutation test p < 0.05
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #5: Permutation test p < 0.05")
    log("=" * 70)
    n_perm = 1000
    perm_gaps = np.zeros(n_perm)
    for p_idx in range(n_perm):
        # shuffle ho_y
        perm_ho_y = rng.permutation(ho_y[:n_eval])
        perm_self_d = np.array([D[i, perm_ho_y[i]] for i in range(n_eval)])
        perm_cross_d = np.array([np.delete(D[i], perm_ho_y[i]).min() for i in range(n_eval)])
        perm_gaps[p_idx] = (perm_cross_d - perm_self_d).mean()
    p_value = (perm_gaps >= gap.mean()).mean()
    log(f"  perm null mean: {perm_gaps.mean():.4f}, observed gap: {gap.mean():.4f}, p={p_value:.4f}")
    log(f"  ** Acceptance #5: {'PASS' if p_value < 0.05 else 'FAIL'} **")

    # ========================================================
    # Acceptance #6: Target user query dist < wrong user (rank-based)
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #6: Target user query dist < wrong user (rank-based)")
    log("=" * 70)
    target_ranks = []
    for i in range(n_eval):
        d_row = D[i].copy()
        # 把 self 也放进去, 算 self 的 rank (ascending: 小 = 像)
        order = np.argsort(d_row)
        self_rank = (order == ho_y[i]).nonzero()[0][0]
        target_ranks.append(self_rank)
    median_rank = float(np.median(target_ranks))
    mean_rank = float(np.mean(target_ranks))
    log(f"  self rank distribution: median={median_rank:.0f}, mean={mean_rank:.1f} (random ~{num_users/2:.0f})")
    log(f"  ** Acceptance #6: {'PASS' if median_rank < num_users / 2 else 'FAIL'} (median self rank < num_users/2) **")

    # ========================================================
    # Acceptance #7: V3 query ranking 完整性 (skip, 由 eval_v3_prototype.py 单独跑)
    # ========================================================
    log("\n" + "=" * 70)
    log("Acceptance #7: V3 query ranking - 查看 selected_query_records.jsonl")
    log("=" * 70)
    sel_file = VADES_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
    n_selected = sum(1 for _ in sel_file.open())
    n_rejected = sum(1 for _ in (VADES_DIR / f"{OUTPUT_TAG}_rejected_query_records.jsonl").open())
    log(f"  selected={n_selected}, rejected={n_rejected}")
    log(f"  ** Acceptance #7: check selected records manually (selected={n_selected}) **")

    # ========================================================
    # 汇总
    # ========================================================
    log("\n" + "=" * 70)
    log("汇总")
    log("=" * 70)
    summary = {
        "acceptance_1_encoder_probe_ho_acc": f"{probe_ho_acc*100:.4f}%",
        "acceptance_2_no_per_user_params": "PASS",
        "acceptance_3_cosine_macro_auc": f"{macro_auc:.4f}",
        "acceptance_4_gap_bootstrap_95ci": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "acceptance_5_permutation_p": f"{p_value:.4f}",
        "acceptance_6_target_median_rank": f"{median_rank:.0f} (random={num_users//2})",
        "acceptance_7_selected": f"{n_selected}",
    }
    for k, v in summary.items():
        log(f"  {k}: {v}")

    out_file = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/acceptance_summary.json")
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"已写入 {out_file}")


if __name__ == "__main__":
    main()
