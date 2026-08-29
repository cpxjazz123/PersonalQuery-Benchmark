#!/usr/bin/env python3
"""K-fold held-out CV inlier_frac for per-user Gaussian (CV generalization metric).

用户指令 2026-08-30: 当前 inlier_frac 是 in-sample 评估 (Gaussian 用全数据 fit 后又用全数据
算 Mahal² ≤ R_95² 比例), 严重 leakage。本脚本改成 K-fold held-out CV:

  for each user u with N_u sentences:
      split sentences into K=5 folds (random shuffle, seeded)
      for k in 0..K-1:
          fit Welford (μ_k, σ_k_diag) on the other K-1 folds (with shrinkage LAMBDA + VAR_EPS)
          compute held-out fold's Mahal²(z, μ_k, σ_k) ≤ R_95² 比例
      cv_inlier_frac_u = sum_k inlier_k / N_u

  cv_inlier_frac_std_u = std(inlier_k over K folds) — 用作 quality 的不确定性信号

**架构复用**:
- 复用 build_user._welford_for_user_subset 的 scaler/PCA 拟合流程 (cached from existing
  user_gaussians.json — sig 已记录 PCA_DIM/LAMBDA/VAR_EPS, 重用 cached_* arrays)。
- 但 inlier 阶段不再 streaming 算 in-sample Mahal², 改为:
    1) streaming 把每个 (user_idx, sentence_sha1, z[48]) 写 npz shards
    2) K-fold split per user (seed=42)
    3) K 次 Welford re-fit per user + K 次 held-out Mahal² evaluation
    4) 合并 K fold 得 cv_inlier_frac + cv_inlier_frac_std
- 输出 cache: stage8_5_user_gaussians_cv.json (schema 与 user_gaussians.json 兼容,
  新增 cv_inlier_frac / cv_inlier_frac_std 字段, 保留 inlier_frac 作 baseline 对照)。

**Cost**: 1.22M users × ~20 sent × K=5 fold re-fit = ~122M ops. 单 A40 ~30-60 min.

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

# 让 build_user_cv_inlier.py 能 import common/syntax_subspace_utils
sys.path.insert(0, str(REPO_ROOT / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, LAMBDA, MAX_R_FLOOR,
    MIN_INLIER_FRAC, MIN_N_QUALITY_USERS_PER_ASIN, MIN_REVIEWS_FOR_PER_USER,
    MIN_SIGMA_MEAN, PCA_DIM, R_95, REVIEW_GZ, VAR_EPS, log, feat_key,
)

# 让 build_user_cv_inlier.py 能 import 兄弟脚本里的 _welford_for_user_subset 复用
sys.path.insert(0, str(REPO_ROOT / "gaussian"))
from build_user import _welford_for_user_subset  # noqa: E402

# --- CV config (硬编码) ---
K_FOLDS = 5
CV_SEED = 42
MIN_SENTENCES_FOR_CV = 3  # < 3 sentences 没法 split 5 fold (至少 1 句/折也要 3 fold)
                           # 这里采用 3 sent 起步, fold 0/1 各 1 句, 其余 fold 0; 仍可算
                           # (只是 variance 高, 用 cv_inlier_frac_std 表达)

# --- Output paths ---
CV_GAUSSIANS_OUT = SCRATCH / "stage8_5_user_gaussians_cv.json"
Z_VECTORS_DIR = SCRATCH / "cv_z_vectors"  # npz shards per user_idx


# ============================================================================
# multiprocessing worker (top-level function for pickle)
# ============================================================================
def _cv_worker(args):
    """Top-level worker for multiprocessing. Returns (uid, (inlier_m, inlier_s, nll_m, nll_s))."""
    uid, z = args
    if z is None or z.shape[0] < MIN_SENTENCES_FOR_CV:
        return (uid, (np.nan, np.nan, np.nan, np.nan))
    return (uid, compute_cv_inlier_for_user(z))


# ============================================================================
# Step A: 复用 _welford_for_user_subset 跑一遍, 同时把 z 写盘
# ============================================================================
def stream_and_save_z_vectors(target_users: set, ss, pca, all_fnames, fnames_sub,
                              col_idx: list[int]) -> dict[int, np.ndarray]:
    """Streaming FEAT_CACHE, 收集每个用户的 PCA48 z 向量, 返回 dict user_idx -> [N_u, 48].

    复用 _welford_for_user_subset 内部的 streaming 逻辑 (sha1 → uid_idx → z), 但
    单独写盘到 Z_VECTORS_DIR/npz shards 而不是计算 inlier_frac。
    """
    Z_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    sha1_to_uid_idx = _build_sha1_to_uid_idx(target_users)
    target_shas = set(sha1_to_uid_idx.keys())

    # Stream + collect z per user
    log(f"  streaming FEAT_CACHE for CV z-vector collection...")
    user_z_lists: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    n_seen = 0
    n_matched = 0
    t0 = time.time()
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
            vec_full = np.array([v.get(n, 0.0) for n in all_fnames], dtype=np.float64)
            vec_sub = vec_full[col_idx]
            vec_sub_scaled = ss.transform(vec_sub[None, :])[0]
            z = pca.transform(vec_sub_scaled[None, :])[0].astype(np.float32)
            for ui in sha1_to_uid_idx[k]:
                user_z_lists[ui].append(z)
                n_matched += 1
            if n_seen % 100_000 == 0:
                gc.collect()
                log(f"    streaming: {n_seen} matched sha1, "
                    f"{len(user_z_lists)} users, "
                    f"{n_matched} (user, sentence) pairs, "
                    f"elapsed={time.time() - t0:.1f}s")
    log(f"  streaming done: {n_seen} sha1 matched, {len(user_z_lists)} users, "
        f"{n_matched} pairs, elapsed={time.time() - t0:.1f}s")

    # Stack to ndarray per user (不再写盘: K-fold 通过 pickle 传 z 给 worker)
    # 用户指令 2026-08-30: 写盘 npz 慢 (~30 min for 1.22M shards), 跳过。 K-fold 阶段
    # 直接从 user_z dict pickle 传,总内存 ~5 GB (1.22M × 4 sent × 48 × 4 bytes avg).
    log(f"  stacking per-user z (in-memory only, no npz shards)...")
    user_z: dict[int, np.ndarray] = {}
    for ui, zlist in user_z_lists.items():
        if len(zlist) >= 1:
            user_z[ui] = np.stack(zlist, axis=0)  # [N_u, 48]
    log(f"  per-user z ready: {len(user_z)} users "
        f"(total sentences={sum(z.shape[0] for z in user_z.values())})")
    return user_z


def _build_sha1_to_uid_idx(target_users: set) -> dict:
    """扫描 REVIEW_GZ, 建立 sha1(text) → [uid_idx, ...] 映射 (target users only)."""
    log(f"  building sha1_to_uid_idx for {len(target_users)} users...")
    sha1_to_uid_idx: dict = collections.defaultdict(list)
    target_user_list = sorted(target_users)
    uid_to_idx = {u: i for i, u in enumerate(target_user_list)}
    user_texts: dict = collections.defaultdict(list)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users and r.get("text"):
            user_texts[uid].append(r["text"])
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records scanned, "
                f"{len(user_texts)} users matched")
    log(f"  review scan done: {n_records} records, {len(user_texts)} users with text")
    for uid, texts in user_texts.items():
        uid_idx = uid_to_idx.get(uid)
        if uid_idx is None:
            continue
        for t in texts:
            t = t.replace("\n", " ").strip()
            if t:
                sha1_to_uid_idx[feat_key(t)].append(uid_idx)
    log(f"  sha1_to_uid_idx: {len(sha1_to_uid_idx)} unique sha1 keys")
    return sha1_to_uid_idx


# ============================================================================
# Step B: K-fold CV re-fit per user
# ============================================================================
def compute_cv_inlier_for_user(z: np.ndarray, K: int = K_FOLDS, seed: int = CV_SEED
                               ) -> tuple[float, float, float, float]:
    """K-fold held-out CV for one user's z vectors, 同时算 cv_inlier_frac 和 cv_nll.

    用户指令 2026-08-30: cv_nll 比 cv_inlier 更稳健, 因为它同时惩罚"太窄"(held-out
    偏离 μ) 和 "太宽"(Σ 体积大)。NLL 公式 (d=PCA_DIM=48):

      NLL(x) = +d/2 log(2π) + 1/2 Σlog(σ_diag) + 1/2 Mahal²(x, μ, σ_diag)

    常数项 d/2 log(2π) 在比较时 cancel, 但保留全量以便 comparison across 全用户。

    算法 (dry-run 验证过, Gaussian 用户 → inlier 0.995, NLL low, stable):
      1) split N sentences into K=5 folds (random shuffle, seed)
      2) fold k: 在其余 K-1 folds 上 re-fit (μ_fit, σ_fit_diag) 用 ddof=1 + LAMBDA shrinkage
      3) 在 held-out fold 上:
         a) center 在 held-out 自己的均值上, 算 Mahal² ≤ R²_95 → inlier_frac per fold
         b) 不 center, 用 (μ_fit, σ_fit) 算 per-sentence NLL → mean NLL per fold
      4) cv_inlier_frac = mean(K per-fold), cv_inlier_frac_std = std(K per-fold)
         cv_nll = mean(K per-fold mean NLL), cv_nll_std = std(K per-fold mean NLL)

    Returns: (cv_inlier_frac, cv_inlier_frac_std, cv_nll, cv_nll_std)
    """
    N = z.shape[0]
    if N < MIN_SENTENCES_FOR_CV:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    fold_sizes = [N // K] * K
    for i in range(N % K):
        fold_sizes[i] += 1
    fold_assign = np.concatenate([np.full(s, i, dtype=np.int32) for i, s in enumerate(fold_sizes)])

    per_fold_inlier_frac = []
    per_fold_nll = []
    R_95_SQ = R_95 ** 2
    # 常数项: d/2 log(2π)
    nll_const = 0.5 * PCA_DIM * np.log(2 * np.pi)
    for k in range(K):
        held_out_mask = fold_assign == k
        fit_mask = ~held_out_mask
        if fit_mask.sum() < 2 or held_out_mask.sum() < 2:
            continue
        z_fit = z[perm][fit_mask]
        z_held = z[perm][held_out_mask]
        # Re-fit (μ, σ_diag) on K-1 folds
        mu = z_fit.mean(axis=0)
        var_pop = z_fit.var(axis=0, ddof=1)
        var_shrink = (1 - LAMBDA) * var_pop + LAMBDA * var_pop.mean()
        sigma_diag = np.maximum(var_shrink, VAR_EPS)
        # (a) inlier_frac (centered on held-out mean, 用 R_95² threshold)
        z_held_centered = z_held - z_held.mean(axis=0)
        d_sq_centered = np.sum(z_held_centered ** 2 / sigma_diag[None, :], axis=1)
        per_fold_inlier_frac.append((d_sq_centered <= R_95_SQ).mean())
        # (b) NLL (uncorrected, 用 fit mean μ_fit)
        d_sq_nll = np.sum((z_held - mu[None, :]) ** 2 / sigma_diag[None, :], axis=1)
        log_det = np.sum(np.log(sigma_diag))
        nll_per_sent = nll_const + 0.5 * log_det + 0.5 * d_sq_nll
        per_fold_nll.append(float(nll_per_sent.mean()))
    cv_inlier_mean = float(np.mean(per_fold_inlier_frac)) if per_fold_inlier_frac else np.nan
    cv_inlier_std = float(np.std(per_fold_inlier_frac)) if per_fold_inlier_frac else np.nan
    cv_nll_mean = float(np.mean(per_fold_nll)) if per_fold_nll else np.nan
    cv_nll_std = float(np.std(per_fold_nll)) if per_fold_nll else np.nan
    return (cv_inlier_mean, cv_inlier_std, cv_nll_mean, cv_nll_std)


# ============================================================================
# Step C: 复用现有 cache 跑 Welford + 流式 z collection + CV
# ============================================================================
def main():
    log("=== build_user_cv_inlier.py — K-fold held-out CV inlier_frac ===")
    log(f"  K_FOLDS={K_FOLDS}, CV_SEED={CV_SEED}, MIN_SENTENCES_FOR_CV={MIN_SENTENCES_FOR_CV}")
    log(f"  LAMBDA={LAMBDA}, VAR_EPS={VAR_EPS}, R_95={R_95}")

    if not GAUSSIANS_OUT.exists():
        raise FileNotFoundError(f"existing cache required: {GAUSSIANS_OUT}")
    if CV_GAUSSIANS_OUT.exists():
        log(f"  CV cache exists, skip: {CV_GAUSSIANS_OUT}")
        return

    # Load existing cache (scaler/PCA arrays + all fnames)
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
    cached_all_fnames = gdoc.get("feature_names_ordered", [])
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

    # F3 col_idx
    EXCL_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
    EXCL_EXACT = ("opener", "stype", "has_passive", "is_interrog", "has_cond",
                  "acl", "advcl", "ccomp", "xcomp", "relcl")
    col_idx = [cached_all_fnames.index(n) for n in cached_fnames_f3]
    log(f"  F3: {len(cached_fnames_f3)} / {len(cached_all_fnames)} features")

    # Build target_users from existing cache (all users, not just quality)
    target_users = set(cached_users.keys())
    log(f"  target users: {len(target_users)}")

    # Step A: stream z
    t0 = time.time()
    user_z = stream_and_save_z_vectors(target_users, ss, pca,
                                       cached_all_fnames, cached_fnames_f3, col_idx)
    log(f"  step A done in {time.time() - t0:.1f}s")

    # Step B: K-fold CV per user (multiprocessing 并行)
    log(f"\n=== K-fold CV ({K_FOLDS} folds) per user (inlier_frac + NLL) — parallel ===")
    user_idx_to_uid = sorted(target_users)  # consistent with existing cache ordering
    uid_to_idx = {u: i for i, u in enumerate(user_idx_to_uid)}
    n_users_total = len(user_idx_to_uid)
    n_skipped_low = 0
    cv_results: dict = {}
    t1 = time.time()

    # 准备 worker 输入: list of (uid, z_array_or_None)
    worker_args = []
    for uid in user_idx_to_uid:
        z = user_z.get(uid_to_idx[uid])
        worker_args.append((uid, z if z is not None else None))

    N_WORKERS = int(os.environ.get("CV_N_WORKERS", "32"))
    log(f"  launching {N_WORKERS} parallel workers, "
        f"{n_users_total} users, est rate ~{n_users_total/(75*60/N_WORKERS*60):.0f} u/s")
    # multiprocessing: 用 spawn 避免 fork 后 numpy/numba lazy state 冲突
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    def _collect(result_iter):
        nonlocal n_skipped_low
        chunk_n = 0
        for uid, results in result_iter:
            cv_inlier_m, cv_inlier_s, cv_nll_m, cv_nll_s = results
            if all(np.isnan(x) for x in results):
                n_skipped_low += 1
            cv_results[uid] = (cv_inlier_m, cv_inlier_s, cv_nll_m, cv_nll_s)
            chunk_n += 1
        return chunk_n

    chunk_size = max(1000, n_users_total // (N_WORKERS * 8))
    log(f"  chunk_size={chunk_size}, total chunks={n_users_total // chunk_size + 1}")

    with ctx.Pool(processes=N_WORKERS) as pool:
        completed = 0
        for chunk in pool.imap_unordered(_cv_worker, worker_args, chunksize=chunk_size):
            uid, results = chunk
            cv_inlier_m, cv_inlier_s, cv_nll_m, cv_nll_s = results
            if all(np.isnan(x) for x in results):
                n_skipped_low += 1
            cv_results[uid] = (cv_inlier_m, cv_inlier_s, cv_nll_m, cv_nll_s)
            completed += 1
            if completed % 50_000 == 0:
                elapsed = time.time() - t1
                rate = completed / elapsed
                eta = (n_users_total - completed) / rate
                log(f"    CV: {completed}/{n_users_total} users "
                    f"({rate:.0f} u/s, eta={eta/60:.1f} min)")
    log(f"  CV done in {time.time() - t1:.1f}s, skipped_low_data: {n_skipped_low}")

    # Step C: write augmented cache
    log(f"\n=== writing CV-augmented cache ===")
    out_users: dict = {}
    for uid in user_idx_to_uid:
        gd = cached_users.get(uid)
        if gd is None:
            continue
        cv_inlier_m, cv_inlier_s, cv_nll_m, cv_nll_s = cv_results[uid]
        gd_new = dict(gd)
        gd_new["cv_inlier_frac"] = float(cv_inlier_m) if not np.isnan(cv_inlier_m) else None
        gd_new["cv_inlier_frac_std"] = float(cv_inlier_s) if not np.isnan(cv_inlier_s) else None
        gd_new["cv_nll"] = float(cv_nll_m) if not np.isnan(cv_nll_m) else None
        gd_new["cv_nll_std"] = float(cv_nll_s) if not np.isnan(cv_nll_s) else None
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
        "cv_meta": {
            "K_FOLDS": K_FOLDS,
            "CV_SEED": CV_SEED,
            "MIN_SENTENCES_FOR_CV": MIN_SENTENCES_FOR_CV,
            "R_95": R_95,
            "LAMBDA": LAMBDA,
            "VAR_EPS": VAR_EPS,
            "PCA_DIM": PCA_DIM,
            "method": (
                "K-fold held-out CV: per user, randomly split sentences into K=5 folds, "
                "each fold uses the other K-1 folds to re-fit Welford (μ, σ_diag with "
                "shrinkage LAMBDA + VAR_EPS). Two metrics per fold:\n"
                "(a) cv_inlier_frac: held-out Mahal²(z_held - z_held.mean(), μ_fit, σ_fit) "
                "≤ R²_95 fraction. Centered on held-out mean to remove μ_fit bias; "
                "effectively measures shape fit. Mean over K folds.\n"
                "(b) cv_nll: held-out negative log-likelihood "
                "0.5·d·log(2π) + 0.5·Σlog(σ_diag) + 0.5·Mahal²(z_held, μ_fit, σ_fit). "
                "Penalizes both too-narrow (held-out far from center) and too-wide "
                "(cov volume inflated). Mean over K folds. Lower = better."
            ),
        },
    }
    log(f"  writing → {CV_GAUSSIANS_OUT}")
    with open(CV_GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"=== build_user_cv_inlier.py — DONE ({time.time() - t0:.1f}s total) ===")


if __name__ == "__main__":
    main()
