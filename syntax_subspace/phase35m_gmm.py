#!/usr/bin/env python3
"""Phase 35.M: Per-user 2-component GMM (shared global diagonal covariance).

User hypothesis: a single user may have multi-modal style (e.g., short keyword
+ long natural-language). Single Gaussian may average out the modes.

Design (per user feedback):
  - Per-user residuals in PCA32 subspace.
  - Fit KMeans K=2 → (mu_1, mu_2) cluster centers, (pi_1, pi_2) from cluster sizes.
  - Shared GLOBAL diagonal covariance (no per-component covariance to avoid
    overfit when each user has only ~10 residuals).
  - Score: log p(z|u) = log[ pi_1 * N(z; mu_1, Sigma_global)
                         + pi_2 * N(z; mu_2, Sigma_global) ].
  - Softmax over users in pool with temperature τ.

Compare to:
  - Single Gaussian softmax_g32_τ0.5 baseline (Phase 35.G/I-v2 SOTA = 50.7%).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35m"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
N_FIXED = 7  # V2 N=7 K=4 SOTA operating point
TAU = 0.5  # softmax temperature from Phase 35.G
PCA_D = 32  # PCA dim from Phase 35.C SOTA


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def count_attrs_covered(text, attrs):
    if not attrs: return 0
    return sum(1 for v in attrs.values() if v and str(v).strip().lower() in text.lower())


def log_mvn_diag(z: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    """Log of unnormalized diagonal Gaussian (without constant). z,mu: (..., d) → (...,)."""
    diffs = z - mu
    return -0.5 * (diffs * diffs * inv_sigma).sum(axis=-1)


def fit_user_gmm_2(user_z: np.ndarray, random_state: int = 0):
    """Fit 2-component GMM with shared diagonal covariance (constants drop out)."""
    n = user_z.shape[0]
    if n < 4:
        # Fallback to single Gaussian
        mu = user_z.mean(axis=0, keepdims=True)
        pi = np.array([1.0])
        return mu, pi
    km = KMeans(n_clusters=2, n_init=4, random_state=random_state).fit(user_z)
    labels = km.labels_
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    pi = counts / counts.sum()
    mu = km.cluster_centers_.astype(np.float64)
    return mu, pi


def user_log_likelihood_gmm2(user_z_cand: np.ndarray, mu: np.ndarray, pi: np.ndarray,
                             inv_sigma: np.ndarray) -> np.ndarray:
    """Score for each cand z (shape [n_cands, d]). Returns [n_cands] log p(z|u) (up to const).
    For ranking purposes, the normalizing constant is the same across users and
    cancels in softmax. Returns log[pi_1 N(z;mu_1) + pi_2 N(z;mu_2)].
    """
    log_n_1 = log_mvn_diag(user_z_cand, mu[0], inv_sigma) + np.log(max(pi[0], 1e-10))
    if len(pi) == 1:
        return log_n_1
    log_n_2 = log_mvn_diag(user_z_cand, mu[1], inv_sigma) + np.log(max(pi[1], 1e-10))
    m = np.maximum(log_n_1, log_n_2)
    return m + np.log(np.exp(log_n_1 - m) + np.exp(log_n_2 - m))


def main():
    t0 = time.time()
    # --- Load residuals (same as phase35i_v2_score.py) ---
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])
    print(f"[load] {len(user_resids)} users with residuals")

    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}

    user_z = {}
    for uid, vecs in user_resids.items():
        V = np.stack(vecs, axis=0).astype(np.float32)
        Z = (V - pca_means[PCA_D]) @ pca_components[PCA_D].T
        user_z[uid] = Z

    # Global diagonal covariance (shared)
    var_global = np.stack([user_z[u].var(axis=0) for u in user_z]).mean(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)
    print(f"[sigma] mean var={var_global.mean():.4f}, median={np.median(var_global):.4f}")

    # Per-user mu (for single-Gaussian baseline, mirrors phase35i_v2_score.py)
    user_pca_mu = {uid: user_z[uid].mean(axis=0) for uid in user_z}

    # Fit 2-GMM per user
    user_gmm = {}
    for uid, Z in user_z.items():
        mu, pi = fit_user_gmm_2(Z)
        user_gmm[uid] = {"mu": mu, "pi": pi, "n_samples": len(Z)}
    n_fallback = sum(1 for g in user_gmm.values() if len(g["pi"]) == 1)
    print(f"[gmm] {len(user_gmm)} users, {n_fallback} fallback (n<4)")

    # --- Load V2 N=7 cands ---
    cands = json.load(open(SCRATCH / f"phase35i_v2_cands_N{N_FIXED}.json"))
    cache2 = __import__("torch").load(SCRATCH / "phase35i_v2_cand_residuals.pt", weights_only=False)
    all_cand_residuals = cache2["residuals"]
    text_index = cache2["text_index"]
    cand_z_per_n = {}
    for k, (N, i) in enumerate(text_index):
        if N != N_FIXED: continue
        r = all_cand_residuals[k]
        cand_z_per_n[i] = (r - pca_means[PCA_D]) @ pca_components[PCA_D].T
    print(f"[load] {len(cands)} V2 N={N_FIXED} candidates")

    # --- Build per-asin pool ---
    asin_to_idx = {}
    for i, c in enumerate(cands):
        asin_to_idx.setdefault(c["asin"], []).append(i)
    print(f"[pool] {len(asin_to_idx)} asins")

    # --- Build per-pool score matrices (single, gmm2, hybrid) ---
    # Single Gaussian: logits = -D_g32 / τ, softmax → probs.
    # 2-GMM:        logits = log_lik_gmm / τ (shared constant drops), softmax → probs.
    # Hybrid:       logits = 0.5*single + 0.5*gmm2.

    def build_score_matrices():
        """Returns dict: per (asin) score matrices [size, n_users] for each method."""
        per_asin = {}
        for asin, idxs in asin_to_idx.items():
            pool = []
            for i in idxs:
                c = cands[i]
                cov = count_attrs_covered(c["query"], c["attrs_used"])
                L = len(c["query"].split())
                pool.append({"uid": c["user_id"], "cov": cov, "L": L,
                             "z": cand_z_per_n[i]})
            size = len(pool)
            if size < 2: continue
            uids_in_pool = list({p["uid"] for p in pool})
            if len(uids_in_pool) < 2: continue
            n_users = len(uids_in_pool)
            Z_pool = np.stack([p["z"] for p in pool], axis=0)  # [size, d]

            # === Single Gaussian D matrix [size, n_users] ===
            D = np.zeros((size, n_users))
            for j, uid in enumerate(uids_in_pool):
                mu = user_pca_mu[uid]
                diffs = Z_pool - mu[None, :]
                D[:, j] = np.sqrt(np.maximum((diffs * diffs * inv_sigma[None, :]).sum(axis=1), 1e-8))
            single_logits = -D / TAU  # [size, n_users]

            # === 2-GMM log-lik matrix [size, n_users] ===
            gmm_logits = np.zeros((size, n_users))
            for j, uid in enumerate(uids_in_pool):
                g = user_gmm[uid]
                gmm_logits[:, j] = user_log_likelihood_gmm2(Z_pool, g["mu"], g["pi"], inv_sigma)

            # === Hybrid: average logits before softmax ===
            hybrid_logits = 0.5 * single_logits + 0.5 * (gmm_logits / TAU)

            per_asin[asin] = {
                "pool": pool, "uids_in_pool": uids_in_pool,
                "single_logits": single_logits, "gmm_logits": gmm_logits,
                "hybrid_logits": hybrid_logits,
            }
        return per_asin

    print("\n[build] computing per-pool score matrices...")
    per_asin = build_score_matrices()
    print(f"[build] {len(per_asin)} asins computed")

    def eval_method(probs_fn, name):
        """probs_fn(score_matrix) -> [size, n_users] probs. Returns aggregated stats."""
        per_record_hits = []
        per_record_pool = []
        per_record_mean_rank = []
        for asin, d in per_asin.items():
            pool = d["pool"]
            size = len(pool)
            n_users = len(d["uids_in_pool"])
            uid_to_idx = {u: i for i, u in enumerate(d["uids_in_pool"])}
            score_matrix = probs_fn(d)
            for uid in d["uids_in_pool"]:
                target_idx = uid_to_idx[uid]
                scores = score_matrix[:, target_idx]
                order = np.argsort(-scores)
                rank1_uid = pool[int(order[0])]["uid"]
                hit = rank1_uid == uid
                own_indices = [i for i, p in enumerate(pool) if p["uid"] == uid]
                if not own_indices: continue
                own_ranks = [int(np.where(order == i)[0][0]) for i in own_indices]
                per_record_hits.append(1 if hit else 0)
                per_record_pool.append(n_users)
                per_record_mean_rank.append(float(np.mean(own_ranks)))

        n_records = len(per_record_hits)
        rank1 = np.mean(per_record_hits) * 100
        mean_pool = np.mean(per_record_pool)
        mean_rank = np.mean(per_record_mean_rank)
        random_acc = np.mean([1.0 / s for s in per_record_pool]) * 100
        lift = rank1 / random_acc if random_acc > 0 else 0
        print(f"  [{name}]")
        print(f"    n_records: {n_records}, pool avg: {mean_pool:.2f}")
        print(f"    Rank-1:    {rank1:.1f}%   (random = {random_acc:.2f}%, lift = {lift:.2f}x)")
        print(f"    mean_rank: {mean_rank:.2f}")
        return {"rank1_pct": rank1, "mean_rank": mean_rank, "n_records": n_records,
                "pool_avg": mean_pool, "random_acc": random_acc, "lift_vs_random": lift,
                "per_record_hits": per_record_hits, "per_record_pool": per_record_pool,
                "per_record_mean_rank": per_record_mean_rank}

    print(f"\n{'='*80}")
    print("Phase 35.M: 2-GMM per-user (shared covariance) vs single Gaussian")
    print(f"  PCA dim = {PCA_D}, τ = {TAU}, V2 N = {N_FIXED}")
    print(f"{'='*80}\n")

    res_baseline = eval_method(lambda d: softmax(d["single_logits"], axis=1),
                                "single-Gaussian softmax_g32_τ0.5 (Phase 35.G/I-v2 SOTA)")
    print()
    res_gmm2 = eval_method(lambda d: softmax(d["gmm_logits"] / TAU, axis=1),
                            "2-GMM shared-cov (NEW)")
    print()
    res_hybrid = eval_method(lambda d: softmax(d["hybrid_logits"], axis=1),
                              "hybrid 0.5*single + 0.5*GMM")

    print(f"\n{'='*80}")
    print(f"{'Method':<55} {'Rank-1':<10} {'mean_rank':<10} {'lift':<10}")
    print('-' * 85)
    for r, name in [(res_baseline, "single-Gaussian softmax_g32_τ0.5 (Phase 35.G/I-v2 SOTA)"),
                    (res_gmm2, "2-GMM shared-cov (NEW)"),
                    (res_hybrid, "hybrid 0.5*single + 0.5*GMM")]:
        print(f"{name:<55} {r['rank1_pct']:<10.1f} {r['mean_rank']:<10.2f} {r['lift_vs_random']:<10.2f}")

    # Pairwise hit diff
    n = len(res_baseline["per_record_hits"])
    n_gmm2_better = sum(1 for i in range(n) if res_gmm2["per_record_hits"][i] > res_baseline["per_record_hits"][i])
    n_gmm2_worse = sum(1 for i in range(n) if res_gmm2["per_record_hits"][i] < res_baseline["per_record_hits"][i])
    n_hybrid_better = sum(1 for i in range(n) if res_hybrid["per_record_hits"][i] > res_baseline["per_record_hits"][i])
    n_hybrid_worse = sum(1 for i in range(n) if res_hybrid["per_record_hits"][i] < res_baseline["per_record_hits"][i])
    print(f"\n  Pairwise hit diff (vs single-Gaussian baseline, n={n} records):")
    print(f"    2-GMM:  better={n_gmm2_better}, worse={n_gmm2_worse}, same={n-n_gmm2_better-n_gmm2_worse}")
    print(f"    hybrid: better={n_hybrid_better}, worse={n_hybrid_worse}, same={n-n_hybrid_better-n_hybrid_worse}")

    # Bootstrap CI for 2-GMM vs baseline (record-level)
    rng = np.random.default_rng(42)
    B = 1000
    diffs_gmm = []
    diffs_hybrid = []
    base_hits = np.array(res_baseline["per_record_hits"])
    gmm_hits = np.array(res_gmm2["per_record_hits"])
    hyb_hits = np.array(res_hybrid["per_record_hits"])
    for _ in range(B):
        idx = rng.integers(0, n, n)
        diffs_gmm.append(gmm_hits[idx].mean() - base_hits[idx].mean())
        diffs_hybrid.append(hyb_hits[idx].mean() - base_hits[idx].mean())
    diffs_gmm = np.array(diffs_gmm) * 100
    diffs_hybrid = np.array(diffs_hybrid) * 100
    print(f"\n  Bootstrap CI (1000 resamples, record-level, percentage points):")
    print(f"    2-GMM  Δ vs baseline: {diffs_gmm.mean():+.2f}pp [{np.percentile(diffs_gmm, 2.5):+.2f}, {np.percentile(diffs_gmm, 97.5):+.2f}]")
    print(f"    hybrid Δ vs baseline: {diffs_hybrid.mean():+.2f}pp [{np.percentile(diffs_hybrid, 2.5):+.2f}, {np.percentile(diffs_hybrid, 97.5):+.2f}]")

    # GMM stats
    pi_0_dist = [float(g["pi"][0]) for g in user_gmm.values() if len(g["pi"]) > 1]
    center_dists = [float(np.linalg.norm(g["mu"][0] - g["mu"][1])) for g in user_gmm.values() if len(g["pi"]) > 1]
    print(f"\n  GMM stats (n={len(pi_0_dist)} users with K=2):")
    print(f"    pi_0 (cluster 0 mix): mean={np.mean(pi_0_dist):.3f}, std={np.std(pi_0_dist):.3f}")
    print(f"    center_dist (PCA32): mean={np.mean(center_dists):.2f}, median={np.median(center_dists):.2f}")

    summary = {
        "config": {"N": N_FIXED, "tau": TAU, "pca_d": PCA_D, "n_users": len(user_gmm),
                   "n_fallback": n_fallback, "n_cands": len(cands)},
        "single_gaussian": {k: v for k, v in res_baseline.items() if not k.startswith("per_")},
        "gmm2_shared": {k: v for k, v in res_gmm2.items() if not k.startswith("per_")},
        "hybrid_50_50": {k: v for k, v in res_hybrid.items() if not k.startswith("per_")},
        "pairwise": {
            "gmm2_better_than_single": n_gmm2_better,
            "gmm2_worse_than_single": n_gmm2_worse,
            "hybrid_better_than_single": n_hybrid_better,
            "hybrid_worse_than_single": n_hybrid_worse,
            "n_records": n,
        },
        "bootstrap": {
            "gmm2_delta_pp_mean": float(diffs_gmm.mean()),
            "gmm2_delta_pp_ci_lo": float(np.percentile(diffs_gmm, 2.5)),
            "gmm2_delta_pp_ci_hi": float(np.percentile(diffs_gmm, 97.5)),
            "hybrid_delta_pp_mean": float(diffs_hybrid.mean()),
            "hybrid_delta_pp_ci_lo": float(np.percentile(diffs_hybrid, 2.5)),
            "hybrid_delta_pp_ci_hi": float(np.percentile(diffs_hybrid, 97.5)),
        },
        "gmm_stats": {
            "n_users_with_k2": len(pi_0_dist),
            "pi_0_mean": float(np.mean(pi_0_dist)) if pi_0_dist else None,
            "pi_0_std": float(np.std(pi_0_dist)) if pi_0_dist else None,
            "center_dist_mean": float(np.mean(center_dists)) if center_dists else None,
            "center_dist_median": float(np.median(center_dists)) if center_dists else None,
        },
    }
    json.dump(summary, open(OUT_DIR / "gmm_summary.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_DIR}/gmm_summary.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
