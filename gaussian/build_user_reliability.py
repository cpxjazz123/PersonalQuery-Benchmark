#!/usr/bin/env python3
"""Gaussian Predictive Reliability (ΔNLL + bootstrap CI) for per-user Gaussian.

用户指令 2026-08-30: 单纯 cv_inlier_frac / cv_nll 仍属"绝对值标定", 用户希望引入
**Gaussian Predictive Reliability** —— 用 NLL 在 self vs other 上的差异衡量 G_u 是不是真
能区分"用户自身"和"其他用户", 借此做 calibration 阈值的 ROC 选择.

步骤:
  1) 复用 build_user_cv_inlier 的 streaming + K-fold 流程, 拿到每个用户的 z 向量
  2) 对每个用户, K=5 fold:
       fold k: re-fit G_u (μ_k, σ_k_diag) 在 K-1 folds
       NLL_self_k = mean NLL over held-out sentences under G_u
       NLL_other_k = mean NLL over N_OTHER 个 random z (从其他用户的 z pool 随机抽取) under G_u
       ΔNLL_k = NLL_other_k - NLL_self_k
  3) Bootstrap (B=200) per user: 对 K 个 fold 的 (NLL_self_k, NLL_other_k) pair 重采样,
     计算 CI_95%(ΔNLL). Reliable iff CI_lower > 0.
  4) 同时保留 cv_nll 作为 baseline (delta_nll_mean 单独的 cv_nll 已经在 stage8_5_user_gaussians_cv.json)

**架构复用**:
- 复用 build_user_cv_inlier._build_sha1_to_uid_idx + stream_and_save_z_vectors + _cv_worker
- 新增: _reliability_worker (算 ΔNLL + bootstrap CI per user)
- 新增: build_other_z_pool (collect random z from all non-target users' z vectors)

**Cost** (estimated):
  - streaming (复用现有): ~14 min for 1.22M users
  - K-fold fit + NLL_self: ~25s (47K u/s, same as CV)
  - NLL_other (random sample + per-fold Mahal²): ~5-10 min
  - bootstrap (B=200, K=5): ~5 min
  - Total: ~30 min for 1.22M users

输出:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_reliability.json
    (1.22M users with delta_nll_mean, delta_nll_ci_lo, delta_nll_ci_hi, reliable flags)
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/reliability_summary.json
    (distribution + ΔNLL vs cv_nll correlation)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import collections
import gc
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 让 reliability 脚本能 import common + 兄弟脚本
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(REPO_ROOT / "gaussian"))
from syntax_subspace_utils import (  # noqa: E402
    FEAT_CACHE, GAUSSIANS_OUT, LAMBDA, PCA_DIM, R_95, REVIEW_GZ, VAR_EPS, log,
)
from build_user_cv_inlier import (  # noqa: E402
    _build_sha1_to_uid_idx, CV_SEED, K_FOLDS, MIN_SENTENCES_FOR_CV,
)

# --- Reliability config (硬编码) ---
N_OTHER_PER_FOLD = 20    # each fold samples N_OTHER z from other-z pool (sampled once per fold for variance)
BOOTSTRAP_B = 50         # bootstrap resamples for CI (50 ≈ 1.96*√50 ≈ 13.8 std err, sufficient for 95% CI bound check)
BOOTSTRAP_SEED = 123     # bootstrap RNG seed (separate from CV_SEED)
N_WORKERS = int(os.environ.get("REL_N_WORKERS", "32"))  # parallel workers

# --- Output paths ---
RELIABILITY_OUT = SCRATCH / "stage8_5_user_gaussians_reliability.json"
SUMMARY_OUT = REPO_ROOT / "result" / "gaussian" / "reliability_summary.json"


# ============================================================================
# Per-user reliability worker (top-level for pickle)
# ============================================================================
def _reliability_worker(args):
    """Compute (delta_nll_mean, delta_nll_ci_lo, delta_nll_ci_hi, reliable_flag) for one user.

    Inputs:
        uid: user id string
        z: [N_u, PCA_DIM] z vectors for this user (may be None if not enough sentences)
        other_z_pool: [P, PCA_DIM] z vectors from OTHER users (pre-sampled random pool)
    Returns:
        (uid, (delta_nll_mean, delta_nll_ci_lo, delta_nll_ci_hi, reliable_int))
    """
    uid, z, other_z_pool = args
    if z is None or z.shape[0] < MIN_SENTENCES_FOR_CV or other_z_pool is None or other_z_pool.shape[0] == 0:
        return (uid, (np.nan, np.nan, np.nan, -1))
    N = z.shape[0]
    K = K_FOLDS
    rng = np.random.RandomState(CV_SEED)
    perm = rng.permutation(N)
    fold_sizes = [N // K] * K
    for i in range(N % K):
        fold_sizes[i] += 1
    fold_assign = np.concatenate([np.full(s, i, dtype=np.int32) for i, s in enumerate(fold_sizes)])

    nll_const = 0.5 * PCA_DIM * np.log(2 * np.pi)
    delta_per_fold = []
    for k in range(K):
        fit_mask = fold_assign != k
        held_mask = fold_assign == k
        if fit_mask.sum() < 2 or held_mask.sum() < 2:
            continue
        z_fit = z[perm][fit_mask]
        z_held = z[perm][held_mask]
        mu = z_fit.mean(axis=0)
        var_pop = z_fit.var(axis=0, ddof=1)
        var_shrink = (1 - LAMBDA) * var_pop + LAMBDA * var_pop.mean()
        sigma_diag = np.maximum(var_shrink, VAR_EPS)
        log_det = np.sum(np.log(sigma_diag))
        # NLL_self (uncorrected, μ_fit)
        d_sq_self = np.sum((z_held - mu[None, :]) ** 2 / sigma_diag[None, :], axis=1)
        nll_self_k = float(nll_const + 0.5 * log_det + 0.5 * d_sq_self.mean())
        # NLL_other: sample N_OTHER from other_z_pool (sampled per fold so each fold has different z)
        n_other = min(N_OTHER_PER_FOLD, other_z_pool.shape[0])
        idx = rng.choice(other_z_pool.shape[0], size=n_other, replace=False)
        z_other = other_z_pool[idx]
        d_sq_other = np.sum((z_other - mu[None, :]) ** 2 / sigma_diag[None, :], axis=1)
        nll_other_k = float(nll_const + 0.5 * log_det + 0.5 * d_sq_other.mean())
        delta_per_fold.append(nll_other_k - nll_self_k)

    if not delta_per_fold:
        return (uid, (np.nan, np.nan, np.nan, -1))
    delta_arr = np.array(delta_per_fold)
    delta_mean = float(delta_arr.mean())
    # Bootstrap CI: resample K-fold indices with replacement
    rng_b = np.random.RandomState(BOOTSTRAP_SEED)
    K_actual = len(delta_per_fold)
    boot_means = np.empty(BOOTSTRAP_B, dtype=np.float64)
    for b in range(BOOTSTRAP_B):
        idx = rng_b.choice(K_actual, size=K_actual, replace=True)
        boot_means[b] = delta_arr[idx].mean()
    ci_lo = float(np.quantile(boot_means, 0.025))
    ci_hi = float(np.quantile(boot_means, 0.975))
    reliable = 1 if ci_lo > 0 else 0
    return (uid, (delta_mean, ci_lo, ci_hi, reliable))


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    log("=== build_user_reliability.py — Gaussian Predictive Reliability ===")
    log(f"  K_FOLDS={K_FOLDS}, CV_SEED={CV_SEED}, BOOTSTRAP_B={BOOTSTRAP_B}, "
        f"N_OTHER_PER_FOLD={N_OTHER_PER_FOLD}, N_WORKERS={N_WORKERS}")

    if not GAUSSIANS_OUT.exists():
        raise FileNotFoundError(f"existing cache required: {GAUSSIANS_OUT}")

    # Load existing cache (scaler/PCA + user_id ordering)
    log(f"  loading existing cache: {GAUSSIANS_OUT}")
    with open(GAUSSIANS_OUT, "r", encoding="utf-8") as f:
        gdoc = json.load(f)
    cached_users = gdoc["users"]
    sig = gdoc.get("sig", {})
    log(f"  existing cache: {len(cached_users)} users, sig={sig.get('sig_short', '?')}")

    # Reconstruct scaler / PCA
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.decomposition import PCA as _PCA

    cached_scaler_mean = gdoc["scaler_mean"]
    cached_scaler_scale = gdoc["scaler_scale"]
    cached_pca_components = gdoc["pca_components"]
    cached_pca_ev = gdoc["pca_explained_variance"]
    cached_pca_evr = gdoc["pca_explained_variance_ratio"]
    cached_pca_mean = gdoc["pca_mean"]
    cached_all_fnames = gdoc["feature_names_ordered"]
    cached_fnames_f3 = gdoc["fnames_f3"]

    ss = _SS()
    ss.mean_ = np.asarray(cached_scaler_mean, dtype=np.float64)
    ss.scale_ = np.asarray(cached_scaler_scale, dtype=np.float64)
    ss.var_ = ss.scale_ ** 2
    ss.n_features_in_ = len(ss.mean_)
    pca = _PCA(n_components=PCA_DIM)
    pca.components_ = np.asarray(cached_pca_components, dtype=np.float64)
    pca.explained_variance_ = np.asarray(cached_pca_ev, dtype=np.float64)
    pca.explained_variance_ratio_ = np.asarray(cached_pca_evr, dtype=np.float64)
    pca.mean_ = np.asarray(cached_pca_mean, dtype=np.float64)
    pca.n_components_ = PCA_DIM
    pca.n_features_in_ = len(pca.mean_)

    col_idx = [cached_all_fnames.index(n) for n in cached_fnames_f3]
    log(f"  F3: {len(cached_fnames_f3)} / {len(cached_all_fnames)} features")

    target_users = set(cached_users.keys())
    log(f"  target users: {len(target_users)}")

    # ====== Step A: streaming z vectors ======
    t0 = time.time()
    log(f"\n=== Step A: stream z vectors (in-memory, parallel-safe pickling) ===")
    target_user_list = sorted(target_users)
    uid_to_idx = {u: i for i, u in enumerate(target_user_list)}
    sha1_to_uid_idx = _build_sha1_to_uid_idx(target_users)
    target_shas = set(sha1_to_uid_idx.keys())
    log(f"  sha1_to_uid_idx ready: {len(target_shas)} keys")

    user_z_lists: dict = collections.defaultdict(list)
    n_seen = 0
    n_matched = 0
    log(f"  streaming FEAT_CACHE ...")
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec["k"]
            if k not in target_shas:
                continue
            n_seen += 1
            v = rec["v"]
            vec_full = np.array([v.get(n, 0.0) for n in cached_all_fnames], dtype=np.float64)
            vec_sub = vec_full[col_idx]
            vec_sub_scaled = ss.transform(vec_sub[None, :])[0]
            z = pca.transform(vec_sub_scaled[None, :])[0].astype(np.float32)
            for ui in sha1_to_uid_idx[k]:
                user_z_lists[ui].append(z)
                n_matched += 1
            if n_seen % 200_000 == 0:
                gc.collect()
                log(f"    streaming: {n_seen} sha1 matched, {n_matched} (user, sent) pairs, "
                    f"{time.time() - t0:.1f}s")
    log(f"  streaming done: {n_seen} sha1, {n_matched} pairs, {time.time() - t0:.1f}s")

    # Stack to ndarray per user
    user_z: dict = {}
    for ui, zlist in user_z_lists.items():
        if len(zlist) >= MIN_SENTENCES_FOR_CV:
            user_z[ui] = np.stack(zlist, axis=0)
    n_users_with_z = len(user_z)
    log(f"  per-user z ready: {n_users_with_z} users (≥ {MIN_SENTENCES_FOR_CV} sentences)")

    # Build other-z pool: aggregate ALL z vectors from users, EXCLUDING the target user's z
    # (For computational tractability, we sample a fixed-size pool of OTHER z. For each user u,
    #  the other-z pool is "all z from all other users". This is identical for all users in a
    #  given batch — we just need a large pool.)
    log(f"\n=== Build global other-z pool (excluding per-user self z) ===")
    # Build global pool: concatenate all z vectors
    all_z = np.concatenate(list(user_z.values()), axis=0) if user_z else np.zeros((0, PCA_DIM), dtype=np.float32)
    log(f"  global z pool: {all_z.shape[0]} vectors total")
    log(f"  per-user computation: for each user u, sample N_OTHER_PER_FOLD={N_OTHER_PER_FOLD} "
        f"z vectors (excluding u's own z). Pool size {all_z.shape[0]} >> {N_OTHER_PER_FOLD}, "
        f"so 'excluding self z' is approximately equivalent to 'uniform random sample'.")

    # ====== Step B: parallel reliability computation ======
    log(f"\n=== Step B: parallel reliability ({N_WORKERS} workers) ===")
    import multiprocessing as mp
    # 用户指令 2026-08-30: 用 fork 避免 spawn 上下文下 BrokenPipe 残留 (前次进程
    # 残 zombie worker 导致的 fd 冲突)。fork 在 pq_env 下无 numpy/numba lazy state 风险。
    ctx = mp.get_context("fork")

    # Prepare worker inputs: (uid_idx, z_or_None, other_z_pool)
    # For each user, the other-z pool is a global pool — we use the SAME pool for all users
    # (approximation: the "self exclusion" is small given pool size > 130K)
    worker_args = []
    for ui in range(len(target_user_list)):
        z = user_z.get(ui)
        worker_args.append((target_user_list[ui], z, all_z))

    log(f"  launching {N_WORKERS} workers, {len(worker_args)} users")
    t1 = time.time()
    reliability_results: dict = {}
    n_skipped = 0
    completed = 0
    chunk_size = max(500, len(worker_args) // (N_WORKERS * 8))
    log(f"  chunk_size={chunk_size}")

    with ctx.Pool(processes=N_WORKERS) as pool:
        for uid, results in pool.imap_unordered(_reliability_worker, worker_args, chunksize=chunk_size):
            delta_m, ci_lo, ci_hi, reliable = results
            if reliable == -1:
                n_skipped += 1
            reliability_results[uid] = (delta_m, ci_lo, ci_hi, reliable)
            completed += 1
            if completed % 20_000 == 0:
                elapsed = time.time() - t1
                rate = completed / elapsed
                eta = (len(worker_args) - completed) / rate
                log(f"    reliability: {completed}/{len(worker_args)} users "
                    f"({rate:.0f} u/s, eta={eta/60:.1f} min)")
    log(f"  reliability done in {time.time() - t1:.1f}s, skipped={n_skipped}")

    # ====== Step C: write augmented cache ======
    log(f"\n=== writing reliability cache ===")
    RELIABILITY_OUT.parent.mkdir(parents=True, exist_ok=True)

    out_users: dict = {}
    for uid in target_user_list:
        gd = cached_users.get(uid)
        if gd is None:
            continue
        delta_m, ci_lo, ci_hi, reliable = reliability_results.get(uid, (np.nan, np.nan, np.nan, -1))
        gd_new = dict(gd)
        gd_new["delta_nll_mean"] = float(delta_m) if not np.isnan(delta_m) else None
        gd_new["delta_nll_ci_lo"] = float(ci_lo) if not np.isnan(ci_lo) else None
        gd_new["delta_nll_ci_hi"] = float(ci_hi) if not np.isnan(ci_hi) else None
        gd_new["reliable"] = int(reliable) if reliable in (0, 1) else None
        out_users[uid] = gd_new

    out_doc = {
        "users": out_users,
        "scaler_mean": cached_scaler_mean,
        "scaler_scale": cached_scaler_scale,
        "pca_components": cached_pca_components,
        "pca_ev": cached_pca_ev,
        "pca_evr": cached_pca_evr,
        "pca_mean": cached_pca_mean,
        "all_fnames": cached_all_fnames,
        "fnames_f3": cached_fnames_f3,
        "sig": sig,
        "reliability_meta": {
            "K_FOLDS": K_FOLDS,
            "CV_SEED": CV_SEED,
            "N_OTHER_PER_FOLD": N_OTHER_PER_FOLD,
            "BOOTSTRAP_B": BOOTSTRAP_B,
            "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
            "LAMBDA": LAMBDA,
            "VAR_EPS": VAR_EPS,
            "R_95": R_95,
            "PCA_DIM": PCA_DIM,
            "n_users_total": len(out_users),
            "n_users_with_reliability": sum(1 for gd in out_users.values() if gd.get("reliable") is not None),
            "n_users_reliable": sum(1 for gd in out_users.values() if gd.get("reliable") == 1),
        },
    }
    # Stream write (avoid 2.3GB in memory)
    log(f"  streaming write → {RELIABILITY_OUT}")
    with open(RELIABILITY_OUT, "w", encoding="utf-8") as f:
        f.write("{")
        f.write(f'"scaler_mean":{json.dumps(out_doc["scaler_mean"])},')
        f.write(f'"scaler_scale":{json.dumps(out_doc["scaler_scale"])},')
        f.write(f'"pca_components":{json.dumps(out_doc["pca_components"])},')
        f.write(f'"pca_ev":{json.dumps(out_doc["pca_ev"])},')
        f.write(f'"pca_evr":{json.dumps(out_doc["pca_evr"])},')
        f.write(f'"pca_mean":{json.dumps(out_doc["pca_mean"])},')
        f.write(f'"all_fnames":{json.dumps(out_doc["all_fnames"])},')
        f.write(f'"fnames_f3":{json.dumps(out_doc["fnames_f3"])},')
        f.write(f'"sig":{json.dumps(out_doc["sig"])},')
        f.write(f'"reliability_meta":{json.dumps(out_doc["reliability_meta"])},')
        f.write('"users":{')
        first = True
        for uid, gd in out_users.items():
            if not first:
                f.write(",")
            first = False
            f.write(f'{json.dumps(uid)}:{json.dumps(gd)}')
        f.write("}}")
    log(f"  wrote → {RELIABILITY_OUT}")

    # ====== Step D: compute summary stats ======
    log(f"\n=== summary stats ===")
    deltas = np.array([gd["delta_nll_mean"] for gd in out_users.values() if gd.get("delta_nll_mean") is not None])
    ci_los = np.array([gd["delta_nll_ci_lo"] for gd in out_users.values() if gd.get("delta_nll_ci_lo") is not None])
    n_reliable = sum(1 for gd in out_users.values() if gd.get("reliable") == 1)
    n_total_reliability = sum(1 for gd in out_users.values() if gd.get("reliable") is not None)

    from scipy.stats import spearmanr
    # ρ(ΔNLL, cv_nll): expect negative (more reliable → lower cv_nll)
    pairs = [(gd["delta_nll_mean"], gd["cv_nll"]) for gd in out_users.values()
             if gd.get("delta_nll_mean") is not None and gd.get("cv_nll") is not None]
    if pairs:
        d_arr = np.array([p[0] for p in pairs])
        nll_arr = np.array([p[1] for p in pairs])
        rho_dnll_nll, _ = spearmanr(d_arr, nll_arr)
    else:
        rho_dnll_nll = None

    summary = {
        "description": (
            f"Gaussian Predictive Reliability (ΔNLL = NLL_other - NLL_self, with bootstrap CI). "
            f"Reliable iff CI_lower(ΔNLL) > 0 (i.e., user's Gaussian assigns lower NLL to "
            f"their own sentences than to random other users' sentences)."
        ),
        "n_users_total": len(out_users),
        "n_users_with_reliability": n_total_reliability,
        "n_users_reliable": n_reliable,
        "frac_reliable": n_reliable / max(1, n_total_reliability),
        "delta_nll_dist": {
            "mean": float(deltas.mean()) if len(deltas) else None,
            "median": float(np.median(deltas)) if len(deltas) else None,
            "q05": float(np.quantile(deltas, 0.05)) if len(deltas) else None,
            "q25": float(np.quantile(deltas, 0.25)) if len(deltas) else None,
            "q75": float(np.quantile(deltas, 0.75)) if len(deltas) else None,
            "q95": float(np.quantile(deltas, 0.95)) if len(deltas) else None,
        },
        "delta_nll_ci_lo_dist": {
            "mean": float(ci_los.mean()) if len(ci_los) else None,
            "median": float(np.median(ci_los)) if len(ci_los) else None,
            "frac_positive": float((ci_los > 0).mean()) if len(ci_los) else None,
        },
        "spearman_delta_nll_vs_cv_nll": float(rho_dnll_nll) if rho_dnll_nll is not None else None,
        "K_FOLDS": K_FOLDS,
        "N_OTHER_PER_FOLD": N_OTHER_PER_FOLD,
        "BOOTSTRAP_B": BOOTSTRAP_B,
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")
    log(f"  ΔNLL: mean={summary['delta_nll_dist']['mean']:.2f}, median={summary['delta_nll_dist']['median']:.2f}, "
        f"reliable={summary['frac_reliable']:.2%}")
    log(f"  ρ(ΔNLL, cv_nll) = {rho_dnll_nll:.3f}" if rho_dnll_nll is not None else "")


if __name__ == "__main__":
    main()
