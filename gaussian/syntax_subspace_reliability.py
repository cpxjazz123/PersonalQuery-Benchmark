"""Syntax Subspace — Gaussian Reliability Diagnostic.

评估每个 user Gaussian 的可靠性,基于 per-user log Bayes factor:
  log P(reviews | user Gaussian) - log P(reviews | global Gaussian)

- > 0: 用户 Gaussian 比 global pool 更好 (用户风格有信号)
- = 0: 用户 Gaussian 与 global 无差异
- < 0: 用户 Gaussian 比 global 更差 (过拟合 / 信号弱 / σ² 退化)

Tier 划分:
- HIGH: LBF > 50 AND n_samples >= 30 (有足够 sample + 强信号)
- MED:  LBF in [10, 50] AND n_samples >= 5
- LOW:  LBF < 10 OR n_samples < 5 (无信号)

用法:
  python gaussian/syntax_subspace_reliability.py

I/O 路径:
  输入: stage8_5_user_gaussians.json
        stage7b_query_features.jsonl.gz (spaCy 182d features cache)
  输出: result/gaussian/user_gaussians_reliability.json
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, REPO_ROOT, REVIEW_GZ,
    LAMBDA, PCA_DIM, VAR_EPS, feat_key, log,
)
sys.path.insert(0, str(REPO_ROOT / "common"))


RELIABILITY_OUT = REPO_ROOT / "result" / "gaussian" / "user_gaussians_reliability.json"


def main():
    log("=== STAGE 3.5 — USER GAUSSIAN RELIABILITY ===")

    # --- 1. Load PCA48 (复用 Stage 3 训练好的 scaler + pca) ---
    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # --- 2. Load user Gaussians + recompute global Gaussian ---
    log("\n=== 2. Loading user Gaussians ===")
    gauss_data = json.load(open(GAUSSIANS_OUT))
    users_gauss = gauss_data["users"]
    log(f"  users: {len(users_gauss)}")

    # --- 3. Load target users ---
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    user_to_asins_count = collections.Counter()
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
            user_to_asins_count[uid] += 1
    log(f"  target users: {len(target_users)}")

    # --- 4. Load feature cache ---
    log("\n=== 4. Loading feature cache ===")
    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  cache: {len(feat_map)} features")

    # --- 5. Scan reviews, project to z, group by user ---
    log("\n=== 5. Scanning reviews + projecting ===")
    user_z_list = collections.defaultdict(list)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users and r.get("text"):
            t = r["text"].replace("\n", " ").strip()
            if not t:
                continue
            feats = feat_map.get(feat_key(t))
            if not feats:
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            user_z_list[uid].append(z)
        n_records += 1
        if n_records % 1_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records, {len(user_z_list)} users")
    log(f"  done: {n_records/1e6:.1f}M records, {len(user_z_list)} users with z")

    # --- 6. Global Gaussian (all z across users) ---
    log("\n=== 6. Computing global Gaussian ===")
    all_z = []
    for uid, zs in user_z_list.items():
        all_z.extend(zs)
    all_z = np.stack(all_z, axis=0)
    global_mu = all_z.mean(axis=0)
    global_var = all_z.var(axis=0)
    global_var_safe = np.maximum(global_var, VAR_EPS)
    log(f"  global mu norm: {np.linalg.norm(global_mu):.4f}")
    log(f"  global var mean: {global_var.mean():.4f}")

    # --- 7. Per-user log Bayes factor ---
    log("\n=== 7. Per-user log Bayes factor ===")
    reliability = {}
    n_high = n_med = n_low = 0
    lbf_all = []

    for uid in target_users:
        zs = user_z_list.get(uid, [])
        if not zs:
            # 用户无 sample (应该 raise 上游过滤,但兜底记 LOW)
            reliability[uid] = {
                "n_samples": 0,
                "tier": "LOW",
                "log_bayes_factor": None,
                "user_ll_mean": None,
                "global_ll_mean": None,
                "mu_norm": None,
                "n_eff_dims": 0,
            }
            n_low += 1
            continue

        Z = np.stack(zs, axis=0)
        n_samples = len(zs)
        u = users_gauss.get(uid)
        if u is None:
            # 上游 Stage 3 应该 raise,这里兜底记录
            reliability[uid] = {
                "n_samples": n_samples,
                "tier": "LOW",
                "log_bayes_factor": None,
                "user_ll_mean": None,
                "global_ll_mean": None,
                "mu_norm": None,
                "n_eff_dims": 0,
            }
            n_low += 1
            continue

        mu = np.array(u["mu"])
        sigma_diag = np.array(u["sigma_diag"])
        sigma_safe = np.maximum(sigma_diag, VAR_EPS)

        # log-likelihood per review (Gaussian diag)
        # LL = -0.5 * (Σ (z-mu)²/sigma + log sigma) per review, summed
        user_ll_per_review = -0.5 * (
            ((Z - mu) ** 2 / sigma_safe).sum(axis=1) + np.log(sigma_safe).sum()
        )
        global_ll_per_review = -0.5 * (
            ((Z - global_mu) ** 2 / global_var_safe).sum(axis=1) + np.log(global_var_safe).sum()
        )

        user_ll = user_ll_per_review.sum()
        global_ll = global_ll_per_review.sum()
        lbf = user_ll - global_ll  # log Bayes factor
        lbf_all.append(lbf)

        n_eff_dims = int((sigma_diag > 2 * VAR_EPS).sum())
        mu_norm = float(np.linalg.norm(mu))

        # Tier
        if lbf > 50 and n_samples >= 30:
            tier = "HIGH"
            n_high += 1
        elif lbf >= 10 and n_samples >= 5:
            tier = "MED"
            n_med += 1
        else:
            tier = "LOW"
            n_low += 1

        reliability[uid] = {
            "n_samples": n_samples,
            "n_reviews": u.get("n_reviews", n_samples),
            "tier": tier,
            "log_bayes_factor": float(lbf),
            "user_ll_mean": float(user_ll_per_review.mean()),
            "global_ll_mean": float(global_ll_per_review.mean()),
            "mu_norm": mu_norm,
            "n_eff_dims": n_eff_dims,
            "sigma_diag_mean": float(sigma_diag.mean()),
            "sigma_diag_min": float(sigma_diag.min()),
        }

    lbf_arr = np.array([v for v in lbf_all if v is not None])
    log(f"\n=== Tier distribution ===")
    log(f"  HIGH: {n_high} ({100*n_high/len(target_users):.1f}%)")
    log(f"  MED : {n_med} ({100*n_med/len(target_users):.1f}%)")
    log(f"  LOW : {n_low} ({100*n_low/len(target_users):.1f}%)")
    log(f"\n=== Log Bayes factor ===")
    log(f"  min={lbf_arr.min():.2f} p25={np.percentile(lbf_arr,25):.2f} "
        f"median={np.median(lbf_arr):.2f} mean={lbf_arr.mean():.2f} "
        f"max={lbf_arr.max():.2f}")
    log(f"  (positive = user Gaussian better than global pool)")

    # --- 8. Save ---
    log("\n=== 8. Saving ===")
    RELIABILITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RELIABILITY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Per-user Gaussian reliability: log Bayes factor vs global pool",
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "PCA_DIM": PCA_DIM,
                "tier_thresholds": {
                    "HIGH": "LBF > 50 AND n_samples >= 30",
                    "MED":  "LBF in [10, 50] AND n_samples >= 5",
                    "LOW":  "LBF < 10 OR n_samples < 5",
                },
            },
            "summary": {
                "n_users": len(target_users),
                "n_high": n_high,
                "n_med": n_med,
                "n_low": n_low,
                "lbf_min": float(lbf_arr.min()),
                "lbf_p25": float(np.percentile(lbf_arr, 25)),
                "lbf_median": float(np.median(lbf_arr)),
                "lbf_mean": float(lbf_arr.mean()),
                "lbf_max": float(lbf_arr.max()),
            },
            "reliability": reliability,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {RELIABILITY_OUT}")


if __name__ == "__main__":
    main()