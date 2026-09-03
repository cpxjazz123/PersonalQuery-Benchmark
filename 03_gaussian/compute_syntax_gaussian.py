"""Step 2: 计算 per-user syntactic Gaussian + exclusive d_self (语法空间).

读取 Step 1 缓存的句法特征 → 构建 per-user Gaussian → 计算 per-(ASIN, user) exclusive d_self
输出:
  result/syntax_style_encoder/user_syntax_gaussian.json
  result/syntax_style_encoder/syntax_exclusive_distance.json
"""

import json
import numpy as np
import os
import sys

REPO_ROOT = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
OUT_DIR = REPO_ROOT + "/result/syntax_style_encoder"
OUT_GAUSSIAN = OUT_DIR + "/user_syntax_gaussian.json"
OUT_EXCL = OUT_DIR + "/syntax_exclusive_distance.json"
ASIN_USERS_PATH = REPO_ROOT + "/asin_users/asin_to_users.json"
MIN_VALID_SENTS = 10  # 每个用户至少需要10个有效句子


def d_boundary_from_spheres(d_ij, sigma_i, sigma_j):
    if d_ij < 1e-8:
        return np.nan
    si = float(np.linalg.norm(sigma_i)) if hasattr(sigma_i, '__len__') else float(sigma_i)
    sj = float(np.linalg.norm(sigma_j)) if hasattr(sigma_j, '__len__') else float(sigma_j)
    return (d_ij**2 - sj**2 + si**2) / (2 * d_ij)


def compute_exclusive(mus, sigmas):
    n = len(mus)
    exclusive = np.full(n, np.nan)
    for i in range(n):
        min_bound = np.inf
        si = sigmas[i]
        for j in range(n):
            if i == j:
                continue
            d_ij = float(np.linalg.norm(mus[i] - mus[j]))
            bound = d_boundary_from_spheres(d_ij, si, sigmas[j])
            if bound < min_bound:
                min_bound = bound
        exclusive[i] = min_bound if np.isfinite(min_bound) else np.nan
    return exclusive


def main():
    print("Loading syntax features ...")
    uid_list = json.load(open(f"{OUT_DIR}/uid_list.json"))
    offsets = json.load(open(f"{OUT_DIR}/uid_offsets.json"))
    features = np.load(f"{OUT_DIR}/syntax_features.npy")  # (total, 112)
    print(f"  {len(uid_list)} users, {features.shape}")

    print("Loading ASIN→users ...")
    asin_to_users = json.load(open(ASIN_USERS_PATH))

    # 构建 per-user Gaussian
    print("Building per-user syntactic Gaussian ...")
    user_gaussian = {}
    for uid in uid_list:
        off = offsets[uid]
        start, n = off['start'], off['n']
        X = features[start:start+n]  # (n_sents, 112)
        # 过滤全零向量（无效句子）
        valid_mask = np.any(X != 0, axis=1)
        X_valid = X[valid_mask]
        if len(X_valid) < MIN_VALID_SENTS:
            continue
        mu = np.mean(X_valid, axis=0)
        sigma = np.std(X_valid, axis=0) + 1e-8
        user_gaussian[uid] = {
            'mu': mu.tolist(),
            'sigma': sigma.tolist(),
            'n_valid': int(len(X_valid)),
        }

    print(f"  {len(user_gaussian)} users with ≥{MIN_VALID_SENTS} valid sentences")

    # 保存 Gaussian
    print(f"Writing → {OUT_GAUSSIAN}")
    os.makedirs(os.path.dirname(OUT_GAUSSIAN), exist_ok=True)
    with open(OUT_GAUSSIAN, 'w') as f:
        json.dump(user_gaussian, f)
    print(f"  Saved {len(user_gaussian)} Gaussian profiles")

    # 计算 per-ASIN exclusive d_self
    print("Computing exclusive d_self in syntax space ...")
    gauss_uids = set(user_gaussian.keys())
    syntax_exclusive = {}

    for asin, asin_uid_list in asin_to_users.items():
        cohort = [u for u in asin_uid_list if u in gauss_uids]
        if len(cohort) < 2:
            continue
        mus = np.array([user_gaussian[u]['mu'] for u in cohort])
        sigmas = np.array([user_gaussian[u]['sigma'] for u in cohort])
        exclusive = compute_exclusive(mus, sigmas)
        syntax_exclusive[asin] = {u: float(exclusive[j]) for j, u in enumerate(cohort)}

    print(f"  {len(syntax_exclusive)} ASINs with ≥2 syntax users")

    # 统计
    all_excl = [d for asin_ds in syntax_exclusive.values() for d in asin_ds.values() if not np.isnan(d)]
    if all_excl:
        arr = np.array(all_excl)
        print(f"\nSyntax exclusive d_self distribution:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th: {np.percentile(arr, p):.4f}")
        print(f"  > 0:   {(arr > 0).mean()*100:.1f}%")
        print(f"  > 1:   {(arr > 1).mean()*100:.1f}%")
        print(f"  > 2:   {(arr > 2).mean()*100:.1f}%")
        print(f"  > 5:   {(arr > 5).mean()*100:.1f}%")

    print(f"Writing → {OUT_EXCL}")
    with open(OUT_EXCL, 'w') as f:
        json.dump(syntax_exclusive, f)
    print("Done.")


if __name__ == "__main__":
    main()
