"""对每个 ASIN 的所有 Gaussian 用户，计算每个用户的最大独占 d_self。

最大独占 d_self: 从用户 centroid 出发沿任意方向,
最后落在该用户 Gaussian 球内且不在其他用户球内的最远距离。
即 min_j d_boundary(i, j)，其中 d_boundary 由两球相交关系决定。

输出: per-(ASIN, user) max_exclusive_d_self
"""

import json
import numpy as np
from collections import defaultdict

REPO_ROOT = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
ASIN_USERS_PATH = REPO_ROOT + "/asin_users/asin_to_users.json"
GAUSSIAN_NPZ   = REPO_ROOT + "/result/03_gaussian/empirical_user_gaussians.npz"
OUT_PATH        = REPO_ROOT + "/result/cohort_analysis/asin_user_exclusive_distance.json"


def d_boundary_from_spheres(d_ij, sigma_i, sigma_j):
    """Compute distance from center i to the decision boundary with sphere j.

    Along the line from mu_i to mu_j, the decision boundary is where
    ||x - mu_i|| = d_bound, ||x - mu_j|| = sigma_j.
    Solving: d_bound = (d_ij^2 - sigma_j^2 + sigma_i^2) / (2 * d_ij).
    Returns NaN if d_ij < 1e-8 (identical centroids).
    """
    if d_ij < 1e-8:
        return np.nan
    return (d_ij**2 - sigma_j**2 + sigma_i**2) / (2 * d_ij)


def compute_exclusive_d_self(mus, sigmas):
    """Compute per-user max exclusive d_self for a cohort.

    Args:
        mus: (n, 768) centroid array
        sigmas: (n,) euclidean sigma per user

    Returns:
        exclusive_dsels: (n,) max exclusive d_self per user
        min_boundary_per_user: (n,) the min j-boundary that limits each user
    """
    n = len(mus)
    exclusive_dsels = np.full(n, np.nan)
    min_boundary_per_user = np.full(n, np.nan)

    for i in range(n):
        min_bound = np.inf
        for j in range(n):
            if i == j:
                continue
            d_ij = np.linalg.norm(mus[i] - mus[j])
            bound = d_boundary_from_spheres(d_ij, sigmas[i], sigmas[j])
            if bound < min_bound:
                min_bound = bound
        exclusive_dsels[i] = min_bound if np.isfinite(min_bound) else np.nan
        min_boundary_per_user[i] = min_bound

    return exclusive_dsels, min_boundary_per_user


def main():
    print("Loading Gaussian ...")
    g = np.load(GAUSSIAN_NPZ, allow_pickle=True)
    mu_768 = g["mu_768d"]
    user_ids_gauss = g["user_ids"]
    sigma_eucl = g["sigma_euclidean"]
    uid_to_idx = {u: i for i, u in enumerate(user_ids_gauss)}
    gauss_set = set(user_ids_gauss.tolist())

    print("Loading ASIN→users mapping ...")
    with open(ASIN_USERS_PATH) as f:
        asin_to_users = json.load(f)

    print(f"  {len(asin_to_users):,} ASINs, {len(user_ids_gauss):,} Gaussian users")

    # 结果: {asin: {user_id: exclusive_d_self}}
    results = {}

    asins_with_cohort = 0
    total_users = 0
    positive_exclusive = 0

    for asin, users in asin_to_users.items():
        cohort = [u for u in users if u in gauss_set]
        n = len(cohort)
        if n < 2:
            continue  # 需要至少2个用户才能计算独占距离

        uidxs = [uid_to_idx[u] for u in cohort]
        mus = mu_768[uidxs]
        sigmas = sigma_eucl[uidxs]

        exclusive_dsels, _ = compute_exclusive_d_self(mus, sigmas)

        asin_results = {}
        for u, d in zip(cohort, exclusive_dsels):
            if np.isfinite(d):
                asin_results[u] = float(d)
                if d > 0:
                    positive_exclusive += 1
            total_users += 1

        if asin_results:
            results[asin] = asin_results
            asins_with_cohort += 1

    print(f"\n=== Results ===")
    print(f"  ASINs with ≥2 cohort users: {asins_with_cohort:,}")
    print(f"  Total users processed:       {total_users:,}")
    print(f"  Users with positive exclusive d_self: {positive_exclusive:,}")

    # 统计分布
    all_dsels = [d for asin_ds in results.values() for d in asin_ds.values()]
    all_dsels = [d for d in all_dsels if d == d]  # filter nan
    if all_dsels:
        arr = np.array(all_dsels)
        print(f"\n  Exclusive d_self distribution:")
        print(f"    min:    {arr.min():.4f}")
        print(f"    25%:    {np.percentile(arr, 25):.4f}")
        print(f"    median: {np.percentile(arr, 50):.4f}")
        print(f"    75%:    {np.percentile(arr, 75):.4f}")
        print(f"    max:    {arr.max():.4f}")
        print(f"    > 5.0:  {(arr > 5.0).sum():,} ({(arr > 5.0).mean()*100:.1f}%)")
        print(f"    > 3.0:  {(arr > 3.0).sum():,} ({(arr > 3.0).mean()*100:.1f}%)")
        print(f"    > 0.0:  {(arr > 0.0).sum():,} ({(arr > 0.0).mean()*100:.1f}%)")
        print(f"    <= 0.0: {(arr <= 0.0).sum():,} ({(arr <= 0.0).mean()*100:.1f}%)")

    print(f"\n  Writing → {OUT_PATH}")
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print("  Done.")


if __name__ == "__main__":
    main()
