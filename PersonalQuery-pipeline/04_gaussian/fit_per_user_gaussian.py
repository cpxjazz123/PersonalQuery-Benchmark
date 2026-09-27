#!/usr/bin/env python3
"""Stage 04 — 拟合 32d per-user Gaussian 与 ASIN cohort gate。

输入:
  - pcfg_cache/strict3_embeddings.npz（Stage 03 stage_strict3() 输出的 profile/val z）
  - pcfg_cache/user_n_sents.json（uid 顺序与句子总数）
  - result/02_user_review_sentence_extract/asin_to_users.json（parent_asin cohort）

输出:
  - result/04_gaussian/user_gaussian_stats.json (canonical, raw_full Σ_u)
  - result/04_gaussian/user_gaussian_stats_rank1.json (rank1+isotropic residual)

Schema 1 (full): users[uid] = {mu, sigma_inv, n, n_val, d2_q50/q75/q95/max}
Schema 2 (rank1): users[uid] = {mu, lambda1, v1, sigma_res, n, n_val,
                                d2_q50/q75/q95/max}

Stage 04 是唯一的 Gaussian 生产阶段。Stage 08 和 Stage 10 只读取 canonical
artifact，不在运行时重新拟合 Gaussian。
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path
from scipy.stats import chi2

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
# 用户指令 2026-09-23: 3 个 category 各自一份 (Baby / Musical / Video_Games),
# main() 改为串行跑 3 个 domain, 产物写到 result/04_gaussian/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]
OUT_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
OUT_PATH_RANK1 = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_rank1.json"
# 用户指令 2026-09-23: asin_to_users 改 pkl-only (Stage 02 已切换).
ASIN_USERS_PATH = (
    REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users_baby.pkl"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 硬编码配置（Rule 3）
SCHEMA_VERSION = 1
SCHEMA_VERSION_RANK1 = 2
CANONICAL_PRODUCT_KEY = "parent_asin"
SMOKE = False               # 2026-09-14: full cohort q20
N_SMOKE_USERS = 500        # 500 users 足够做 O_u 分布估计
N_SMOKE_ASINS = 5
MIN_PROFILE_SENTS = 40
MIN_VAL_SENTS = 10
MIN_EIGEN_RATIO = 1e-8
PSD_FLOOR = 1e-6          # 数值 PSD clip, 覆盖 n≈40-50 in 32d 的 fp64 roundoff 负特征值 (~1e-7)
GATE_QUANTILE = 0.05       # 2026-09-14: q=0.20→0.05 目标 O_u≤0.05
# 2026-09-15: theoretical gate (χ²(d, q)) — Stage 08 high-side (q=0.95) inclusion,
# Stage 10 low-side (q=0.05) rejection. Replaces empirical d2_qXX from val sentences.
THEORETICAL_QS = (0.05, 0.50, 0.75, 0.95)
THEORETICAL_HIGH_Q = 0.95
THEORETICAL_LOW_Q = 0.05
# Rank1+residual regularization (per L8.25): σ_res floor
RANK1_SR_FLOOR = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _validate_strict3_contract(npz, uid_list: list[str],
                              user_n_sents: list[int]) -> None:
    """强制 strict3 NPZ 与 cache manifest 完全一致（UID/partition/finite）。

    2026-09-15: strict3 改为 2-way 80/20, z_test/test_idx 是 z_val/val_idx 别名,
    partition 验证只看 profile+val, test 字段要求存在且形状对齐 val。
    """
    required_fields = ("z_profile", "z_val",
                       "profile_idx", "val_idx", "uid_list",
                       "cohort_fingerprint", "uid_layout_fingerprint",
                       "vocab_fingerprint")
    for field in required_fields:
        if field not in npz:
            raise ValueError(f"strict3 artifact missing field: {field}")
    artifact_uids = [str(uid) for uid in np.asarray(npz["uid_list"]).tolist()]
    if artifact_uids != uid_list:
        raise ValueError("strict3 artifact UID order/content differs from cache")
    n_total = int(sum(user_n_sents))
    arrays = []
    for key in ("profile_idx", "val_idx"):
        index = np.asarray(npz[key], dtype=np.int64)
        if index.ndim != 1:
            raise ValueError(f"strict3 {key} must be 1-D")
        if len(index) and (int(index.min()) < 0 or int(index.max()) >= n_total):
            raise ValueError(f"strict3 {key} index out of range [0,{n_total})")
        if len(np.unique(index)) != len(index):
            raise ValueError(f"strict3 {key} contains duplicate indices")
        arrays.append(index)
    merged = np.concatenate(arrays)
    if len(np.unique(merged)) != n_total or not np.array_equal(
            np.sort(merged), np.arange(n_total, dtype=np.int64)):
        raise ValueError("strict3 split indices do not partition all sentences")
    for key in ("z_profile", "z_val"):
        z = np.asarray(npz[key])
        if z.ndim != 2 or z.shape[1] not in (16, 32):
            raise ValueError(f"strict3 {key} shape invalid: {z.shape}")
        if not np.isfinite(z).all():
            raise ValueError(f"strict3 {key} contains NaN/Inf")
    # 2026-09-15: 接受 test=val 别名 (Stage 03 strict3 80/20 no test)
    if "z_test" in npz and "test_idx" in npz:
        z_test = np.asarray(npz["z_test"])
        test_idx = np.asarray(npz["test_idx"], dtype=np.int64)
        val_idx = np.asarray(npz["val_idx"], dtype=np.int64)
        if z_test.shape != np.asarray(npz["z_val"]).shape:
            raise ValueError("strict3 z_test must equal z_val shape (alias)")
        if not np.array_equal(test_idx, val_idx):
            raise ValueError("strict3 test_idx must equal val_idx (alias)")
        if not np.isfinite(z_test).all():
            raise ValueError("strict3 z_test contains NaN/Inf")


def _load_embedding_groups() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """加载 z 并按 uid 排序，返回每个 uid 的连续 profile/val 分组索引。

    输入: pcfg_cache/strict3_embeddings.npz
      (Stage 03 stage_strict3() 产出的 2-way 80/20 split embeddings, 无 test;
       z_test/test_idx 字段如存在则 = z_val/val_idx 别名)
    """
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    manifest_path = CACHE_DIR / "strict3_manifest.json"
    ready_path = CACHE_DIR / "cache_ready.json"
    n_sents_path = CACHE_DIR / "user_n_sents.json"
    for path in (npz_path, manifest_path, ready_path, n_sents_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing: {path} (run 03_spacy_encode.stage_strict3() first)")

    npz = np.load(npz_path, allow_pickle=False)
    with open(n_sents_path) as f:
        user_n_sents = [int(n) for n in json.load(f)]
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(ready_path) as f:
        ready = json.load(f)
    uid_list = [str(u) for u in npz["uid_list"]]

    _validate_strict3_contract(npz, uid_list, user_n_sents)
    source_keys = (
        ("cohort_fingerprint", "sentence_source_fingerprint"),
        ("uid_layout_fingerprint", "uid_layout_fingerprint"),
        ("vocab_fingerprint", "vocab_fingerprint"),
    )
    for npz_key, manifest_key in source_keys:
        npz_value = str(np.asarray(npz[npz_key]).item())
        if npz_value != manifest.get(npz_key):
            raise ValueError(
                f"strict3 artifact manifest mismatch at {npz_key}: "
                f"npz={npz_value}, manifest={manifest.get(npz_key)!r}")
        if manifest.get(npz_key) != ready.get(manifest_key):
            raise ValueError(
                f"strict3 cache manifest differs from cache_ready at {manifest_key}")
    if manifest.get("n_users") != len(uid_list) or manifest.get(
            "n_total_sents") != int(sum(user_n_sents)):
        raise ValueError("strict3 manifest n_users/n_total_sents mismatch")
    if ready.get("n_users") != len(uid_list) or ready.get(
            "n_total_sents") != int(sum(user_n_sents)):
        raise ValueError("cache_ready manifest n_users/n_total_sents mismatch")

    z_profile = np.asarray(npz["z_profile"], dtype=np.float64)
    z_val = np.asarray(npz["z_val"], dtype=np.float64)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(npz["val_idx"], dtype=np.int64)

    uid_idx_per_row = np.repeat(np.arange(len(uid_list), dtype=np.int64), user_n_sents)
    prof_uid = uid_idx_per_row[profile_idx]
    val_uid = uid_idx_per_row[val_idx]
    prof_order = np.argsort(prof_uid, kind="stable")
    val_order = np.argsort(val_uid, kind="stable")
    prof_counts = np.bincount(prof_uid, minlength=len(uid_list))
    val_counts = np.bincount(val_uid, minlength=len(uid_list))
    prof_offsets = np.concatenate(([0], np.cumsum(prof_counts, dtype=np.int64)))
    val_offsets = np.concatenate(([0], np.cumsum(val_counts, dtype=np.int64)))

    log(
        f"  loaded: {len(uid_list)} uids, z_profile={z_profile.shape}, "
        f"z_val={z_val.shape}, cohort={manifest['cohort_fingerprint'][:12]}"
    )
    return (
        z_profile[prof_order],
        z_val[val_order],
        uid_list,
        prof_offsets,
        val_offsets,
    )


_NUMBA_CACHE: dict = {}


def _fit_chunk(chunk: list) -> list[tuple[str, dict | None, dict | None]]:
    """ProcessPoolExecutor worker: fit a chunk of users, return list of
    (uid, full_stats, rank1_stats). Defined at module level for pickling.

    When invoked from ProcessPool, the worker's __main__ may not have the
    REPO_ROOT on sys.path; re-import fit_one_user from this module by file
    path to avoid ModuleNotFoundError.
    """
    try:
        fit_fn = fit_one_user  # in-process: fast path
    except NameError:  # pragma: no cover
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "_stage04_kernel",
            "/home/wlia0047/ar57/wenyu/PersoanlQuery/04_gaussian/"
            "fit_per_user_gaussian.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fit_fn = mod.fit_one_user
    out = []
    for uid, zp, zv in chunk:
        result = fit_fn(zp, zv)
        if result is None:
            out.append((uid, None, None))
            continue
        full_stats, rank1_stats = result
        out.append((uid, full_stats, rank1_stats))
    return out


def _build_numba_kernel():
    """Lazy compile a Numba-parallel inner kernel for raw full Σ fit.

    Returns a function f(z_prof, z_val, psd_floor, min_eigen_ratio,
    gate_quantile) -> (success:bool, mu[K], sigma_inv[K,K],
    d2_q50, d2_q75, d2_q95, d2_max, n, n_val, lam1).
    The kernel handles cov → eigh → PSD clip → cholesky → inv_sigma → d2_val.
    """
    if "kernel" in _NUMBA_CACHE:
        return _NUMBA_CACHE["kernel"]

    from numba import njit, prange

    @njit(cache=True, fastmath=False, parallel=False, boundscheck=False)
    def _fit_kernel(z_prof, z_val, psd_floor, min_eigen_ratio,
                    gate_quantile, rank1_sr_floor):
        n_prof = z_prof.shape[0]
        n_val = z_val.shape[0]
        K = z_prof.shape[1]
        # mu
        mu = np.zeros(K, dtype=np.float64)
        for j in range(K):
            s = 0.0
            for i in range(n_prof):
                s += z_prof[i, j]
            mu[j] = s / n_prof
        # cov = (centered.T @ centered) / (n_prof - 1)
        cov = np.zeros((K, K), dtype=np.float64)
        for i in range(n_prof):
            for j in range(K):
                cov[j, j] += (z_prof[i, j] - mu[j]) ** 2
        for i in range(n_prof):
            for j in range(K):
                d_j = z_prof[i, j] - mu[j]
                for l in range(j + 1, K):
                    d_l = z_prof[i, l] - mu[l]
                    cov[j, l] += d_j * d_l
        scale = 1.0 / (n_prof - 1)
        for j in range(K):
            for l in range(K):
                cov[j, l] *= scale
                if j != l:
                    cov[l, j] = cov[j, l]
        # eigh (cov is symmetric; built-in eigh supports symmetric)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # PSD clip: replace negatives with psd_floor, rebuild cov
        min_eig = eigvals[0]
        needs_rebuild = min_eig < psd_floor
        if needs_rebuild:
            for i in range(K):
                if eigvals[i] < psd_floor:
                    eigvals[i] = psd_floor
            # rebuild cov = V diag(clip_eigs) V^T
            new_cov = np.zeros((K, K), dtype=np.float64)
            for j in range(K):
                for l in range(K):
                    s = 0.0
                    for i in range(K):
                        s += eigvecs[j, i] * eigvals[i] * eigvecs[l, i]
                    new_cov[j, l] = s
            cov = new_cov
            eigvals, eigvecs = np.linalg.eigh(cov)
            min_eig = eigvals[0]
        if min_eig <= 0:
            return (False, mu, cov, 0.0, 0.0, 0.0, 0.0,
                    np.int64(0), np.int64(0), 0.0, np.zeros(K, dtype=np.float64), 0.0)
        # Cholesky (lower)
        chol = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(i + 1):
                s = cov[i, j]
                for k in range(j):
                    s -= chol[i, k] * chol[j, k]
                if i == j:
                    if s <= 0.0:
                        return (False, mu, cov, 0.0, 0.0, 0.0, 0.0,
                                np.int64(0), np.int64(0), 0.0,
                                np.zeros(K, dtype=np.float64), 0.0)
                    chol[i, j] = np.sqrt(s)
                else:
                    chol[i, j] = s / chol[j, j]
        # inv_sigma = chol^T^-1 · chol^-1; solve chol · X = I, then chol^T · Y = X
        inv_sigma = np.zeros((K, K), dtype=np.float64)
        for col in range(K):
            # solve chol · x = e_col (forward sub)
            x = np.zeros(K, dtype=np.float64)
            for i in range(K):
                rhs = 1.0 if i == col else 0.0
                for k in range(i):
                    rhs -= chol[i, k] * x[k]
                x[i] = rhs / chol[i, i]
            # solve chol^T · y = x (back sub)
            y = np.zeros(K, dtype=np.float64)
            for i in range(K - 1, -1, -1):
                rhs = x[i]
                for k in range(i + 1, K):
                    rhs -= chol[k, i] * y[k]
                y[i] = rhs / chol[i, i]
            for i in range(K):
                inv_sigma[i, col] = y[i]
        # symmetrize
        for i in range(K):
            for j in range(i + 1, K):
                avg = 0.5 * (inv_sigma[i, j] + inv_sigma[j, i])
                inv_sigma[i, j] = avg
                inv_sigma[j, i] = avg
        # d2_val: solve chol · w = (z_val - mu) row-wise
        d2 = np.zeros(n_val, dtype=np.float64)
        for r in range(n_val):
            w = np.zeros(K, dtype=np.float64)
            for i in range(K):
                rhs = z_val[r, i] - mu[i]
                for k in range(i):
                    rhs -= chol[i, k] * w[k]
                w[i] = rhs / chol[i, i]
            s = 0.0
            for i in range(K):
                s += w[i] * w[i]
            d2[r] = s
        # validate d2
        ok = True
        for r in range(n_val):
            if d2[r] < 0.0 or not np.isfinite(d2[r]):
                ok = False
                break
        if not ok:
            return (False, mu, cov, 0.0, 0.0, 0.0, 0.0,
                    np.int64(0), np.int64(0), 0.0,
                    np.zeros(K, dtype=np.float64), 0.0)
        # sort d2 (small n_val → insertion)
        d2_sorted = d2.copy()
        for i in range(1, n_val):
            key = d2_sorted[i]
            j = i - 1
            while j >= 0 and d2_sorted[j] > key:
                d2_sorted[j + 1] = d2_sorted[j]
                j -= 1
            d2_sorted[j + 1] = key
        q50 = d2_sorted[int(n_val * 0.50)]
        q75 = d2_sorted[int(n_val * 0.75)]
        q95 = d2_sorted[min(int(n_val * gate_quantile), n_val - 1)]
        d2max = d2_sorted[n_val - 1]
        # rank1 stats (needed for downstream as fallback; cheap)
        lam1 = eigvals[K - 1]
        v1 = eigvecs[:, K - 1].copy()
        # sigma_res = mean of pos eigs except lam1 + floor
        n_pos = 0
        s_pos = 0.0
        for i in range(K):
            if eigvals[i] > 0.0 and i < K - 1:
                s_pos += eigvals[i]
                n_pos += 1
        sr = (s_pos / n_pos if n_pos > 0 else 0.0) + rank1_sr_floor
        return (True, mu, inv_sigma, q50, q75, q95, d2max,
                n_prof, n_val, lam1, v1, sr)

    _NUMBA_CACHE["kernel"] = _fit_kernel
    return _fit_kernel


def fit_one_user(z_prof: np.ndarray, z_val: np.ndarray) -> tuple[dict | None, dict | None]:
    """拟合一个用户的 full Σ_u (Numba 内核加速, ~10-30× speedup).

    Returns (full_stats, rank1_stats). 任一失败则对应项为 None.
    """
    n_prof, n_val = len(z_prof), len(z_val)
    if n_prof < MIN_PROFILE_SENTS or n_val < MIN_VAL_SENTS:
        return None, None

    z_prof = np.ascontiguousarray(z_prof, dtype=np.float64)
    z_val = np.ascontiguousarray(z_val, dtype=np.float64)
    kernel = _build_numba_kernel()
    ok, mu, inv_sigma, q50, q75, q95, d2max, np_, nv, lam1, v1, sr = \
        kernel(z_prof, z_val, PSD_FLOOR, MIN_EIGEN_RATIO,
               GATE_QUANTILE, RANK1_SR_FLOOR)
    if not ok:
        return None, None

    full_stats = {
        "mu": mu.astype(np.float32).tolist(),
        "sigma_inv": inv_sigma.astype(np.float32).tolist(),
        "n": int(np_),
        "n_val": int(nv),
        "d2_q50": float(q50),
        "d2_q75": float(q75),
        "d2_q95": float(q95),
        "d2_max": float(d2max),
    }
    rank1_stats = {
        "mu": mu.astype(np.float32).tolist(),
        "lambda1": float(lam1),
        "v1": v1.astype(np.float32).tolist(),
        "sigma_res": float(sr),
        "n": int(np_),
        "n_val": int(nv),
        "d2_q50": float(q50),
        "d2_q75": float(q75),
        "d2_q95": float(q95),
        "d2_max": float(d2max),
    }
    return full_stats, rank1_stats


def _load_asin_users() -> dict[str, list[str]]:
    if not ASIN_USERS_PATH.exists():
        raise FileNotFoundError(
            f"missing: {ASIN_USERS_PATH} (run 02_user_review_sentence_extract first)"
        )
    # 用户指令 2026-09-23: 改用 pickle.load (Stage 02 已切换 pkl-only).
    with open(ASIN_USERS_PATH, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError("asin_to_users_baby.pkl must be an object")
    out: dict[str, list[str]] = {}
    for asin, users in raw.items():
        if not isinstance(asin, str) or not isinstance(users, list):
            raise ValueError("asin_to_users_baby.pkl must map string ASINs to user lists")
        normalized = sorted({str(uid) for uid in users})
        if normalized:
            out[asin] = normalized
    return out


def _build_cohort_gates(
    asin_to_users: dict[str, list[str]],
    users: dict[str, dict],
    smoke_asins: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """从独立 ASIN cohort mapping 建立 compact gate，不复制 Gaussian 矩阵。"""
    eligible: list[tuple[str, list[str]]] = []
    for asin, uids in asin_to_users.items():
        if smoke_asins is not None and asin not in smoke_asins:
            continue
        fitted = [uid for uid in uids if uid in users]
        if len(fitted) >= 2:
            eligible.append((asin, fitted))
    eligible.sort(key=lambda item: item[0])
    if SMOKE and smoke_asins is None:
        eligible = eligible[:N_SMOKE_ASINS]

    cohort_gates: dict[str, dict[str, dict]] = {}
    for asin, fitted in eligible:
        cohort_gates[asin] = {
            uid: {
                "gate_T": float(users[uid]["d2_q95"]),
                "n_profile": int(users[uid]["n"]),
                "n_val": int(users[uid]["n_val"]),
            }
            for uid in fitted
        }
    return cohort_gates



def _add_theoretical_gates(stats_path: Path, label: str, z_dim: int) -> None:
    """后处理: 给已有 user_gaussian_stats.json 加 d2_qXX_theoretical 字段 + cohort theo gate。

    等价于 rewrite_gaussian_with_theoretical_gate.py (已被合并)。
    """
    log(f"=== theoretical gate postprocess: {label} ===")
    if not stats_path.exists():
        log(f"  missing, skip")
        return
    bak = stats_path.with_suffix(stats_path.suffix + ".pre_theoretical_gate")
    if not bak.exists():
        shutil.copy2(stats_path, bak)
        log(f"  backup → {bak.name}")
    d = json.loads(stats_path.read_text())
    cfg = d.get("config", {})
    cfg["theoretical_gate_quantiles"] = list(THEORETICAL_QS)
    cfg["theoretical_gate_values"] = {
        f"q{int(q*100):02d}": float(chi2.ppf(q, df=z_dim))
        for q in THEORETICAL_QS
    }
    cfg["gate_source_note"] = (
        f"theoretical gate_T = chi2({z_dim}, q) for q in {THEORETICAL_QS}; "
        "Stage 08 high q=0.95 inclusion, Stage 10 low q=0.05 rejection; "
        "replaces empirical d2_qXX from val sentences"
    )
    high_val = float(chi2.ppf(THEORETICAL_HIGH_Q, df=z_dim))
    low_val = float(chi2.ppf(THEORETICAL_LOW_Q, df=z_dim))
    n_users = 0
    n_pairs = 0
    for uid, u in d.get("users", {}).items():
        if not isinstance(u, dict):
            continue
        for q in THEORETICAL_QS:
            u[f"d2_q{int(q*100):02d}_theoretical"] = float(chi2.ppf(q, df=z_dim))
        n_users += 1
    for asin, cohort in d.get("cohort_gates", {}).items():
        for uid, gate in cohort.items():
            if isinstance(gate, dict):
                gate["gate_T_high_theoretical"] = high_val
                gate["gate_T_low_theoretical"] = low_val
                n_pairs += 1
    d["config"] = cfg
    stats_path.write_text(json.dumps(d))
    log(f"  wrote {n_users} users × {len(THEORETICAL_QS)} quantiles + {n_pairs} cohort pairs")
    log(f"  cohort gate_T_high_theoretical = {high_val:.4f} (Stage 08)")
    log(f"  cohort gate_T_low_theoretical  = {low_val:.4f} (Stage 10)")


def main_task_body() -> None:
    t0 = time.time()
    log("=== Stage 04 — Per-User 32d Gaussian (full + rank1+residual) ===")
    log(
        f"  SMOKE={SMOKE}  MIN_PROFILE={MIN_PROFILE_SENTS} "
        f"MIN_VAL={MIN_VAL_SENTS}  Q={GATE_QUANTILE}  rank1_sr_floor={RANK1_SR_FLOOR}"
    )

    z_profile, z_val, uid_list, prof_offsets, val_offsets = _load_embedding_groups()
    eligible_indices = [
        i
        for i in range(len(uid_list))
        if prof_offsets[i + 1] - prof_offsets[i] >= MIN_PROFILE_SENTS
        and val_offsets[i + 1] - val_offsets[i] >= MIN_VAL_SENTS
    ]
    if SMOKE:
        eligible_indices = eligible_indices[:N_SMOKE_USERS]
    log(f"  eligible users: {len(eligible_indices)}" + (" (SMOKE)" if SMOKE else ""))

    users_full: dict[str, dict] = {}
    users_rank1: dict[str, dict] = {}
    n_full_singular = 0
    n_full_only = 0
    n_rank1_only = 0
    # Warm up Numba JIT (first compile ~30-60s; do it once before parallel loop)
    log("  warming up Numba kernel (first-call compile)...")
    warm_start = time.time()
    _build_numba_kernel()(
        np.ascontiguousarray(z_profile[:MIN_PROFILE_SENTS + 1], dtype=np.float64),
        np.ascontiguousarray(z_val[:MIN_VAL_SENTS + 1], dtype=np.float64),
        PSD_FLOOR, MIN_EIGEN_RATIO, GATE_QUANTILE, RANK1_SR_FLOOR)
    log(f"  Numba kernel ready ({time.time() - warm_start:.1f}s)")

    # Build per-user slice views (zero-copy) and dispatch via ProcessPoolExecutor
    # to bypass GIL and exploit multiple BLAS thread domains.
    tasks = []
    for uid_idx in eligible_indices:
        p_start, p_end = int(prof_offsets[uid_idx]), int(prof_offsets[uid_idx + 1])
        v_start, v_end = int(val_offsets[uid_idx]), int(val_offsets[uid_idx + 1])
        tasks.append((uid_list[uid_idx],
                      z_profile[p_start:p_end], z_val[v_start:v_end]))
    n_workers = min(8, max(1, (os.cpu_count() or 4)))
    log(f"  dispatching {len(tasks)} users to {n_workers} workers")
    completed = 0
    t_dispatch = time.time()
    if n_workers == 1 or len(tasks) < 16:
        # serial path (small workloads or no parallelism benefit)
        for uid, zp, zv in tasks:
            full_stats, rank1_stats = fit_one_user(zp, zv)
            if full_stats is None and rank1_stats is None:
                n_full_singular += 1
            elif full_stats is None:
                n_rank1_only += 1
                users_rank1[uid] = rank1_stats
            elif rank1_stats is None:
                continue
            else:
                users_full[uid] = full_stats
                users_rank1[uid] = rank1_stats
            completed += 1
            if completed % 500 == 0 or completed == len(tasks):
                log(f"    fitted {completed}/{len(tasks)} users "
                    f"({time.time() - t_dispatch:.1f}s)")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def _init_worker():
            import sys as _sys
            _sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

        # Submit in chunks to limit pickle overhead
        chunk_size = max(64, len(tasks) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker) as ex:
            futures = []
            for start in range(0, len(tasks), chunk_size):
                chunk = tasks[start:start + chunk_size]
                futures.append(ex.submit(_fit_chunk, chunk))
            for fut in as_completed(futures):
                results = fut.result()
                for uid, full_stats, rank1_stats in results:
                    if full_stats is None and rank1_stats is None:
                        n_full_singular += 1
                    elif full_stats is None:
                        n_rank1_only += 1
                        users_rank1[uid] = rank1_stats
                    elif rank1_stats is None:
                        continue
                    else:
                        users_full[uid] = full_stats
                        users_rank1[uid] = rank1_stats
                    completed += 1
                if completed % 1000 == 0 or completed == len(tasks):
                    log(f"    fitted {completed}/{len(tasks)} users "
                        f"({time.time() - t_dispatch:.1f}s)")

    log(
        f"  full Σ fitted={len(users_full)}  rank1+residual fitted={len(users_rank1)}  "
        f"rank1_only={n_rank1_only}  both_failed={n_full_singular}"
    )

    asin_to_users = _load_asin_users()
    # Read strict3 manifest once so the cohort fingerprint travels into
    # both user_gaussian_stats payloads (downstream cohort alignment).
    with open(CACHE_DIR / "strict3_manifest.json") as _mf:
        _strict3_manifest = json.load(_mf)
    smoke_asins = None
    if SMOKE:
        smoke_candidates = [
            asin for asin, uids in asin_to_users.items()
            if sum(uid in users_full for uid in uids) >= 2
        ]
        smoke_asins = sorted(smoke_candidates)[:N_SMOKE_ASINS]
        if not smoke_asins:
            raise RuntimeError("SMOKE found no ASIN with at least two fitted users")

    # === Write full Σ (canonical, unchanged schema) ===
    cohort_gates_full = _build_cohort_gates(asin_to_users, users_full, smoke_asins)
    n_pairs_full = sum(len(v) for v in cohort_gates_full.values())
    if not cohort_gates_full:
        raise RuntimeError("no ASIN has at least two full-Σ fitted users")
    payload_full = {
        "config": {
            "schema_version": SCHEMA_VERSION,
            "canonical_product_key": CANONICAL_PRODUCT_KEY,
            "smoke": SMOKE,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "covariance": "raw_full",
            "gate_quantile": GATE_QUANTILE,
            "gate_quantiles": [0.50, 0.75, 0.95],
            "syntax_dim": int(z_profile.shape[1]),
            "n_uids_total": len(uid_list),
            "n_uids_eligible": len(eligible_indices),
            "n_uids_fitted": len(users_full),
            "n_uids_singular": n_full_singular,
            "n_uids_nonpositive_covariance": n_full_singular,
            "n_asins_with_cohort": len(cohort_gates_full),
            "n_cohort_pairs": n_pairs_full,
            "n_asins_considered": len(asin_to_users),
            "pcfg_cache_source": str(CACHE_DIR / "strict3_embeddings.npz"),
            "asin_users_source": str(ASIN_USERS_PATH),
            "cohort_fingerprint": str(_strict3_manifest.get("cohort_fingerprint", "")),
            "uid_layout_fingerprint": str(_strict3_manifest.get("uid_layout_fingerprint", "")),
            "cache_schema_version": int(_strict3_manifest.get("cache_schema_version", 0)),
            "note": "32d supervised raw full-covariance Gaussian without ridge; "
            "cohort gate_T is validation d2_q95",
        },
        "users": users_full,
        "cohort_gates": cohort_gates_full,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload_full, f, ensure_ascii=False)
    log(
        f"  wrote → {OUT_PATH} ({len(users_full)} users, "
        f"{len(cohort_gates_full)} ASINs, {OUT_PATH.stat().st_size // 1024} KB)"
    )

    # === Write rank1+residual (new, schema v2) ===
    cohort_gates_rank1 = _build_cohort_gates(asin_to_users, users_rank1, smoke_asins)
    n_pairs_rank1 = sum(len(v) for v in cohort_gates_rank1.values())
    payload_rank1 = {
        "config": {
            "schema_version": SCHEMA_VERSION_RANK1,
            "canonical_product_key": CANONICAL_PRODUCT_KEY,
            "smoke": SMOKE,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "covariance": "rank1_plus_isotropic_residual",
            "rank1_formula": "Sigma = lambda1 * v1 v1^T + sigma_res^2 * (I - v1 v1^T)",
            "rank1_sr_floor": RANK1_SR_FLOOR,
            "d2_formula": "D^2 = (v1^T (z-mu))^2 / lambda1 + ||r||^2 / sigma_res^2, r = (z-mu) - a*v1",
            "gate_quantile": GATE_QUANTILE,
            "gate_quantiles": [0.50, 0.75, 0.95],
            "syntax_dim": int(z_profile.shape[1]),
            "n_uids_total": len(uid_list),
            "n_uids_eligible": len(eligible_indices),
            "n_uids_fitted": len(users_rank1),
            "n_uids_singular": 0,
            "n_uids_nonpositive_covariance": 0,
            "n_asins_with_cohort": len(cohort_gates_rank1),
            "n_cohort_pairs": n_pairs_rank1,
            "n_asins_considered": len(asin_to_users),
            "pcfg_cache_source": str(CACHE_DIR / "strict3_embeddings.npz"),
            "asin_users_source": str(ASIN_USERS_PATH),
            "cohort_fingerprint": str(_strict3_manifest.get("cohort_fingerprint", "")),
            "uid_layout_fingerprint": str(_strict3_manifest.get("uid_layout_fingerprint", "")),
            "cache_schema_version": int(_strict3_manifest.get("cache_schema_version", 0)),
            "n_full_users": len(users_full),
            "n_rank1_only_users": n_rank1_only,
            "note": "Rank1 + isotropic residual parameterization from L8.25: "
            "always numerically finite (no Cholesky); 33 floats per user "
            "(mu[32] + lambda1 + v1[32] + sigma_res); production candidates",
        },
        "users": users_rank1,
        "cohort_gates": cohort_gates_rank1,
    }
    with open(OUT_PATH_RANK1, "w") as f:
        json.dump(payload_rank1, f, ensure_ascii=False)
    log(
        f"  wrote → {OUT_PATH_RANK1} ({len(users_rank1)} users, "
        f"{len(cohort_gates_rank1)} ASINs, {OUT_PATH_RANK1.stat().st_size // 1024} KB)"
    )
    log(f"=== Stage 04 DONE in {time.time() - t0:.1f}s ===")

    # 2026-09-15: 自动 post-process 给两个产物加 theoretical gate 字段
    # (等价于旧 rewrite_gaussian_with_theoretical_gate.py, 已被合并)
    _add_theoretical_gates(OUT_PATH, "full Σ schema", z_dim=int(z_profile.shape[1]))
    _add_theoretical_gates(OUT_PATH_RANK1, "rank1+residual schema", z_dim=int(z_profile.shape[1]))


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """用户指令 2026-09-23: 串行运行 3 个 category.

    每个 category 重新绑定该脚本使用的路径常量为 category-specific 路径,
    然后调原 main_task_body() (保持原有逻辑不动). 产物写到
    result/<stage>/<baby|musical|video_games>/ 子目录.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, OUT_PATH_RANK1, CACHE_DIR  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "OUT_PATH_RANK1", "CACHE_DIR"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        # 用户指令 2026-09-23: Stage 03a 写到 pcfg_cache_<subdir>/, 04 必须按 subdir 重绑 CACHE_DIR.
        if "CACHE_DIR" in saved:
            CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu") / f"pcfg_cache_{subdir}"
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data") / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "OUT_PATH_RANK1" in saved:
            OUT_PATH_RANK1 = base_out / subdir / saved["OUT_PATH_RANK1"].name
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        OUT_PATH_RANK1.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH_RANK1" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()
