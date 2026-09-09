"""
Profile–Test covariance reproducibility test.

Question: Is the per-user 32×32 covariance Σ_u stable across two
independent halves of the user's history (SHA1 hash 50/50 split)?

For each user u:
  - fit Σ_u^profile on profile z (32×32, with α=0.1 shrinkage)
  - fit Σ_u^test    on test    z (32×32, same shrinkage)
Compare:
  - same-user same-half-cross:    ⟨Σ^profile_u, Σ^test_u⟩    (5000 pairs)
  - different-user cross-half:    ⟨Σ^profile_u, Σ^test_v⟩    (sampled)
  - different-user profile-only:   ⟨Σ^profile_u, Σ^profile_v⟩ (sampled)

Metrics (3):
  M1. Normalized Frobenius cosine:  tr(A Bᵀ) / (||A||·||B||)
  M2. Affine-invariant distance:    d²(A,B) = tr(ln(A^{-1/2} B A^{-1/2}))²
                                    (lower = more similar; Bhattacharyya-style)
  M3. Log-det divergence:          Bregman divergence on log-det
                                    (lower = more similar)

Output:
  /home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_cov_reproducibility.json
"""

import json
import os
from collections import Counter

import numpy as np
from scipy.linalg import inv, sqrtm, logm
from scipy.stats import mannwhitneyu

CACHE = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_cov_reproducibility.json"

SHRINK_ALPHA = 0.1  # same as Stage 04 ablation


def fit_cov(z: np.ndarray, alpha: float = SHRINK_ALPHA) -> np.ndarray:
    """Per-user Σ with shrinkage (same as Stage 04)."""
    n = z.shape[0]
    mu = z.mean(axis=0)
    centered = z - mu
    cov_uns = (centered.T @ centered) / max(n - 1, 1)
    # shrinkage: (1-α) cov_uns + α tr(cov_uns)/d I
    d = cov_uns.shape[0]
    trace = float(np.trace(cov_uns))
    cov = (1.0 - alpha) * cov_uns + alpha * (trace / d) * np.eye(d)
    return cov


def frob_cos(A, B):
    """Normalized Frobenius inner product, in [-1,1]."""
    na = np.linalg.norm(A, "fro")
    nb = np.linalg.norm(B, "fro")
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.trace(A @ B) / (na * nb))


def affine_invariant_distance(A, B):
    """d²(A,B) = tr(ln(A^{-1/2} B A^{-1/2}))²  (Riemannian distance on SPD).
    Lower = more similar."""
    A_inv_sqrt = inv(sqrtm(A))
    M = A_inv_sqrt @ B @ A_inv_sqrt
    # numerical stability: force hermitian
    M = (M + M.T) / 2.0
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    log_eigs = np.log(eigvals)
    return float(np.sum(log_eigs ** 2))


def logdet_divergence(A, B):
    """Bregman log-det divergence: tr(A B^{-1}) - log det(A B^{-1}) - d.
    Lower = more similar. Symmetric."""
    B_inv = inv(B)
    A_B_inv = A @ B_inv
    sign, logdet = np.linalg.slogdet(A_B_inv)
    if sign <= 0:
        return float("inf")
    return float(np.trace(A_B_inv) - logdet - A.shape[0])


def main():
    SMOKE = os.environ.get("SMOKE", "0") == "1"

    print("loading strict_embeddings.npz ...")
    npz = np.load(f"{CACHE}/strict_embeddings.npz")
    z_profile = npz["z_profile"]      # (407223, 32)
    z_test = npz["z_test"]            # (408800, 32)
    profile_idx = npz["profile_idx"]  # (407223,) flat indices into supervised_embeddings
    test_idx = npz["test_idx"]        # (408800,)
    uid_list = npz["uid_list"]        # (5000,) strings

    n_users = len(uid_list)
    assert n_users == 5000
    z_dim = z_profile.shape[1]

    # group indices by user
    print("grouping profile indices per user ...")
    # Pre-compute per-user row ranges in z_profile / z_test (sorted by profile_idx/test_idx order).
    user_n_sents = json.load(open(f"{CACHE}/user_n_sents.json"))
    offsets = np.concatenate([[0], np.cumsum(user_n_sents)])
    p_lo = np.zeros(n_users + 1, dtype=np.int64)
    p_hi = np.zeros(n_users + 1, dtype=np.int64)
    t_lo = np.zeros(n_users + 1, dtype=np.int64)
    t_hi = np.zeros(n_users + 1, dtype=np.int64)
    for ui in range(n_users):
        s, e = offsets[ui], offsets[ui + 1]
        p_lo[ui] = np.searchsorted(profile_idx, s)
        p_hi[ui] = np.searchsorted(profile_idx, e)
        t_lo[ui] = np.searchsorted(test_idx, s)
        t_hi[ui] = np.searchsorted(test_idx, e)
    p_lo[n_users] = len(profile_idx)
    p_hi[n_users] = len(profile_idx)
    t_lo[n_users] = len(test_idx)
    t_hi[n_users] = len(test_idx)

    # Skip the by_user list since we have row ranges now
    profile_by_user = None
    test_by_user = None

    n_prof_per_user = p_hi[:n_users] - p_lo[:n_users]
    n_test_per_user = t_hi[:n_users] - t_lo[:n_users]
    print(f"profile per user mean={n_prof_per_user.mean():.1f} test={n_test_per_user.mean():.1f}")
    if SMOKE:
        n_users = 50

    # ----- fit Σ^profile and Σ^test per user -----
    print(f"fitting Σ per user (n_users={n_users}) ...")
    cov_profile = []
    cov_test = []
    skipped = 0
    for ui in range(n_users):
        zp = z_profile[p_lo[ui]:p_hi[ui]]
        zt = z_test[t_lo[ui]:t_hi[ui]]
        if zp.shape[0] < 5 or zt.shape[0] < 5:
            cov_profile.append(None)
            cov_test.append(None)
            skipped += 1
            continue
        cov_profile.append(fit_cov(zp))
        cov_test.append(fit_cov(zt))
        if (ui + 1) % 500 == 0:
            print(f"  fitted {ui+1}/{n_users}", flush=True)
    print(f"skipped (n<5): {skipped}")

    # keep only users with both halves
    valid = [i for i in range(n_users) if cov_profile[i] is not None and cov_test[i] is not None]
    print(f"valid users: {len(valid)}")

    cov_p = np.stack([cov_profile[i] for i in valid], axis=0)  # (V, 32, 32)
    cov_t = np.stack([cov_test[i] for i in valid], axis=0)
    valid_uids = [uid_list[i] for i in valid]

    # ----- pairwise similarity matrix -----
    V = len(valid)
    print(f"computing pairwise metrics for {V} users ...")

    # M1: Frobenius cosine — same-user (profile vs test) for each user
    same_m1 = np.array([frob_cos(cov_p[i], cov_t[i]) for i in range(V)])
    # M2: affine invariant
    same_m2 = np.array([affine_invariant_distance(cov_p[i], cov_t[i]) for i in range(V)])
    # M3: log-det divergence
    same_m3 = np.array([logdet_divergence(cov_p[i], cov_t[i]) for i in range(V)])

    # Cross-user: sample 5000 random pairs (profile_p vs profile_q, with p≠q)
    rng = np.random.default_rng(42)
    n_pairs = min(5000, V * (V - 1) // 2)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.integers(0, V, size=2)
        if a != b and (min(a, b), max(a, b)) not in {tuple(p) for p in pairs}:
            pairs.append((min(a, b), max(a, b)))
    pairs = pairs[:n_pairs]

    cross_m1 = np.array([frob_cos(cov_p[a], cov_p[b]) for a, b in pairs])
    cross_m2 = np.array([affine_invariant_distance(cov_p[a], cov_p[b]) for a, b in pairs])
    cross_m3 = np.array([logdet_divergence(cov_p[a], cov_p[b]) for a, b in pairs])

    # Cross-half: profile_p vs test_q (different user)
    pairs2 = []
    while len(pairs2) < n_pairs:
        a, b = rng.integers(0, V, size=2)
        if a != b and (min(a, b), max(a, b)) not in {tuple(p) for p in pairs2}:
            pairs2.append((min(a, b), max(a, b)))
    pairs2 = pairs2[:n_pairs]
    cross_half_m1 = np.array([frob_cos(cov_p[a], cov_t[b]) for a, b in pairs2])
    cross_half_m2 = np.array([affine_invariant_distance(cov_p[a], cov_t[b]) for a, b in pairs2])
    cross_half_m3 = np.array([logdet_divergence(cov_p[a], cov_t[b]) for a, b in pairs2])

    # ----- stats -----
    def stats(arr):
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    out_metrics = {}
    for name, same, cross, crossh in [
        ("M1_frobenius_cosine", same_m1, cross_m1, cross_half_m1),
        ("M2_affine_invariant_d2", same_m2, cross_m2, cross_half_m2),
        ("M3_logdet_divergence", same_m3, cross_m3, cross_half_m3),
    ]:
        # Mann-Whitney U test: same > cross (for M1) or same < cross (for M2/M3)
        if "cosine" in name:
            u_stat, p_mw = mannwhitneyu(same, cross, alternative="greater")
        else:
            u_stat, p_mw = mannwhitneyu(same, cross, alternative="less")
        out_metrics[name] = {
            "same_user_profile_vs_test": stats(same),
            "cross_user_profile_vs_profile": stats(cross),
            "cross_user_profile_vs_test": stats(crossh),
            "mann_whitney_same_vs_cross": {
                "U": float(u_stat),
                "p_value": float(p_mw),
                "direction": "same>cross" if "cosine" in name else "same<cross",
            },
        }

    out = {
        "config": {
            "n_users_total": int(n_users),
            "n_users_valid": int(V),
            "n_pairs_sampled": int(n_pairs),
            "shrink_alpha": SHRINK_ALPHA,
            "split": "sha1_hash_50_50",
            "smoke": SMOKE,
        },
        "metrics": out_metrics,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote → {OUT_PATH}")
    print(f"\nM1 same-user median: {np.median(same_m1):.4f}")
    print(f"M1 cross-user median: {np.median(cross_m1):.4f}")
    print(f"M2 same-user median: {np.median(same_m2):.4f}")
    print(f"M2 cross-user median: {np.median(cross_m2):.4f}")


if __name__ == "__main__":
    main()