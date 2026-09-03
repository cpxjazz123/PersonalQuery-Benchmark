"""分析 Gaussian 拟合质量: Von Mises-Fisher 理论驱动的绝对阈值。

基于 Von Mises-Fisher 分布理论计算绝对质量阈值:
  - D=768 (Wegmann 维度)
  - 观测 mean cos_center → 估计 concentration κ
  - κ → std(cos) → 绝对阈值 cos_center > 0.624 (μ - 1σ)

输出: result/cohort_analysis/user_fitting_quality.json
       result/cohort_analysis/quality_filtered_users.json
"""

import json
import numpy as np
import pickle
import random
import sys

REPO_ROOT = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
EMBED_CACHE = REPO_ROOT + "/result/user_review_sentence_extract/uid_embed_cache.pkl"
GAUSSIAN_NPZ = REPO_ROOT + "/result/03_gaussian/empirical_user_gaussians.npz"
OUT_QUALITY  = REPO_ROOT + "/result/cohort_analysis/user_fitting_quality.json"
OUT_FILTERED = REPO_ROOT + "/result/cohort_analysis/quality_filtered_users.json"

# Von Mises-Fisher 理论常数
D = 768
COS_THRESHOLD_VMF = 0.624   # μ - 1σ (68% CI)，基于 VMF κ≈1093


def estimate_vmf_kappa(mean_cos, D=768):
    """从观测 mean cosine similarity 估计 VMF concentration κ。

    Large-D approximation: mean_cos ≈ 1 - (D-1)/(2*κ)
    解出 κ = (D-1) / (2*(1 - mean_cos))
    """
    if mean_cos >= 1.0:
        return np.inf
    if mean_cos <= 0.0:
        return 0.0
    kappa = 0.5 * (D - 1) / (1 - mean_cos)
    return kappa


def main():
    print("Loading Gaussian ...")
    g = np.load(GAUSSIAN_NPZ, allow_pickle=True)
    user_ids = g["user_ids"]
    mu_768 = g["mu_768d"]
    sigma_eucl = g["sigma_euclidean"]
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    print("Loading embed cache ...")
    with open(EMBED_CACHE, "rb") as f:
        cache = pickle.load(f)
    emb768 = cache["uid_to_embed_768"]
    del cache

    common_uids = [u for u in user_ids if u in emb768]
    print(f"  {len(common_uids)} / {len(user_ids)} Gaussian users in cache")

    # === 对所有 Gaussian 用户计算质量指标 ===
    print(f"  Computing quality metrics for all {len(common_uids)} users ...")
    cos_centers = np.full(len(common_uids), np.nan)
    inlier_ratios = np.full(len(common_uids), np.nan)
    d_self_reals = np.full(len(common_uids), np.nan)
    n_sents_arr = np.full(len(common_uids), np.nan)

    for i, uid in enumerate(common_uids):
        sents = np.array(emb768[uid], dtype=np.float32)
        idx = uid_to_idx[uid]
        mu = mu_768[idx].astype(np.float32)
        sigma = sigma_eucl[idx]

        sent_norms = np.linalg.norm(sents, axis=1, keepdims=True) + 1e-8
        mu_norm = np.linalg.norm(mu) + 1e-8
        cos_sim = np.sum((sents / sent_norms) * (mu / mu_norm), axis=1)
        cos_centers[i] = float(np.mean(cos_sim))

        d_self = np.linalg.norm(sents - mu, axis=1)
        inlier_ratios[i] = float(np.mean(d_self <= sigma))
        d_self_reals[i] = float(np.mean(d_self))
        n_sents_arr[i] = len(sents)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(common_uids)}")

    del emb768

    # === VMF 理论分析 ===
    mean_cos = float(np.nanmedian(cos_centers))
    kappa_est = estimate_vmf_kappa(mean_cos, D)
    std_cos = np.sqrt(D - 1) / kappa_est if kappa_est > 0 else np.nan
    threshold_1sigma = mean_cos - std_cos  # μ - 1σ

    print(f"\n=== Von Mises-Fisher Theory (D={D}) ===")
    print(f"  Observed median cos_center:  {mean_cos:.4f}")
    print(f"  Estimated concentration κ:     {kappa_est:.1f}")
    print(f"  std(cos) ≈ √(D)/κ:          {std_cos:.4f}")
    print(f"  Theoretical 1σ threshold:     {threshold_1sigma:.4f}")
    print(f"  (used as VMF-backed absolute threshold)")

    # === 质量分布 ===
    print(f"\n=== Quality Metrics (n={len(common_uids)}) ===")
    print(f"\ncos_center percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.nanpercentile(cos_centers, p):.4f}")

    print(f"\ninlier_ratio percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.nanpercentile(inlier_ratios, p):.4f}")

    print(f"\nd_self_real percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.nanpercentile(d_self_reals, p):.4f}")

    # === 应用阈值 ===
    mask_vmf = cos_centers >= COS_THRESHOLD_VMF
    mask_inlier = inlier_ratios >= 0.50

    print(f"\n=== Quality Filtering Results ===")
    print(f"  VMF threshold (cos_center >= {COS_THRESHOLD_VMF}): "
          f"{mask_vmf.sum():,} / {len(common_uids)} ({mask_vmf.mean()*100:.1f}%)")
    print(f"  inlier threshold (>= 0.50):                   "
          f"{mask_inlier.sum():,} / {len(common_uids)} ({mask_inlier.mean()*100:.1f}%)")
    print(f"  BOTH pass:                                    "
          f"{(mask_vmf & mask_inlier).sum():,} / {len(common_uids)} ({(mask_vmf & mask_inlier).mean()*100:.1f}%)")

    # === 保存 per-user 质量 ===
    quality_results = {
        uid: {
            "cos_center": float(cos_centers[i]),
            "inlier_ratio": float(inlier_ratios[i]),
            "d_self_real": float(d_self_reals[i]),
            "n_sentences": int(n_sents_arr[i]),
            "pass_vmf_threshold": bool(mask_vmf[i]),
            "pass_inlier_threshold": bool(mask_inlier[i]),
        }
        for i, uid in enumerate(common_uids)
    }

    print(f"\nWriting → {OUT_QUALITY}")
    with open(OUT_QUALITY, "w") as f:
        json.dump(quality_results, f, ensure_ascii=False)

    # === 保存通过质量过滤的用户列表 ===
    filtered_uids = [uid for i, uid in enumerate(common_uids) if mask_vmf[i] and mask_inlier[i]]
    filtered_data = {
        "threshold": "cos_center >= 0.624 AND inlier_ratio >= 0.50",
        "theory": "Von Mises-Fisher (D=768), κ≈1093, 1σ confidence interval",
        "n_filtered": len(filtered_uids),
        "n_total": len(common_uids),
        "user_ids": filtered_uids,
    }

    print(f"Writing → {OUT_FILTERED}")
    with open(OUT_FILTERED, "w") as f:
        json.dump(filtered_data, f, ensure_ascii=False)

    print(f"\nDone. {len(filtered_uids)} users passed quality filter.")


if __name__ == "__main__":
    main()
