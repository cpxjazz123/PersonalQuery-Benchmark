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
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
OUT_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
OUT_PATH_RANK1 = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_rank1.json"
ASIN_USERS_PATH = (
    REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 硬编码配置（Rule 3）
SCHEMA_VERSION = 1
SCHEMA_VERSION_RANK1 = 2
CANONICAL_PRODUCT_KEY = "parent_asin"
SMOKE = False
N_SMOKE_USERS = 50
N_SMOKE_ASINS = 5
MIN_PROFILE_SENTS = 40
MIN_VAL_SENTS = 10
MIN_EIGEN_RATIO = 1e-8
GATE_QUANTILE = 0.95
# Rank1+residual regularization (per L8.25): σ_res floor
RANK1_SR_FLOOR = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_embedding_groups() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """加载 z 并按 uid 排序，返回每个 uid 的连续 profile/val 分组索引。

    输入: pcfg_cache/strict3_embeddings.npz
      (Stage 03 stage_strict3() 产出的 3-way 50/20/30 split embeddings)
    """
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    n_sents_path = CACHE_DIR / "user_n_sents.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing: {npz_path} (run 03_spacy_encode.stage_strict3() first)"
        )
    if not n_sents_path.exists():
        raise FileNotFoundError(f"missing: {n_sents_path}")

    npz = np.load(npz_path, allow_pickle=False)
    z_profile = np.asarray(npz["z_profile"], dtype=np.float64)
    z_val = np.asarray(npz["z_val"], dtype=np.float64)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(npz["val_idx"], dtype=np.int64)
    uid_list = [str(u) for u in npz["uid_list"]]

    with open(n_sents_path) as f:
        user_n_sents = [int(n) for n in json.load(f)]
    if len(user_n_sents) != len(uid_list):
        raise ValueError(
            f"user_n_sents ({len(user_n_sents)}) != uid_list ({len(uid_list)})"
        )
    if any(n < 0 for n in user_n_sents):
        raise ValueError("user_n_sents contains a negative count")
    n_total = int(sum(user_n_sents))
    max_index = max(int(profile_idx.max()), int(val_idx.max()))
    if n_total <= max_index:
        raise ValueError(f"row index out of range: n_total={n_total}, max={max_index}")
    if z_profile.shape[1] != 32 or z_val.shape[1] != 32:
        raise ValueError(
            f"expected 32d supervised z, got profile={z_profile.shape}, val={z_val.shape}"
        )

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
        f"z_val={z_val.shape}"
    )
    return (
        z_profile[prof_order],
        z_val[val_order],
        uid_list,
        prof_offsets,
        val_offsets,
    )


def fit_one_user(z_prof: np.ndarray, z_val: np.ndarray) -> tuple[dict | None, dict | None]:
    """拟合一个用户的 full Σ_u 与 rank1+residual 参数化。

    Returns (full_stats, rank1_stats). 任一失败则对应项为 None.
    """
    n_prof, n_val = len(z_prof), len(z_val)
    if n_prof < MIN_PROFILE_SENTS or n_val < MIN_VAL_SENTS:
        return None, None

    mu = z_prof.mean(axis=0)
    centered = z_prof - mu
    cov = (centered.T @ centered) / (n_prof - 1)
    cov = (cov + cov.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        return None, None
    eigen_ratio = float(eigenvalues[0] / eigenvalues[-1])
    if not np.isfinite(eigen_ratio) or eigen_ratio < MIN_EIGEN_RATIO:
        return None, None

    # ===== Rank1+isotropic residual (always finite, numerically stable) =====
    # Σ_u ≈ λ_1 v_1 v_1^T + σ_res^2 (I - v_1 v_1^T)
    # Mahalanobis D² = a²/λ_1 + ‖r‖²/σ_res²,  a = v_1^T (z-μ), r = (z-μ) - a v_1
    lam1 = float(eigenvalues[-1])
    v1 = eigenvectors[:, -1]
    pos_eigs = eigenvalues[eigenvalues > 0]
    if len(pos_eigs) > 1:
        sr = float(np.mean(pos_eigs[:-1])) + RANK1_SR_FLOOR
    else:
        sr = RANK1_SR_FLOOR
    diff_val = z_val - mu
    a_val = diff_val @ v1
    r_vec_val = diff_val - a_val[:, None] * v1
    d2_val_r1 = (a_val ** 2) / lam1 + (r_vec_val ** 2).sum(axis=1) / sr
    if not np.all(np.isfinite(d2_val_r1)) or np.any(d2_val_r1 < 0):
        return None, None
    rank1_stats = {
        "mu": mu.astype(np.float32).tolist(),
        "lambda1": lam1,
        "v1": v1.astype(np.float32).tolist(),
        "sigma_res": sr,
        "n": int(n_prof),
        "n_val": int(n_val),
        "d2_q50": float(np.quantile(d2_val_r1, 0.50)),
        "d2_q75": float(np.quantile(d2_val_r1, 0.75)),
        "d2_q95": float(np.quantile(d2_val_r1, GATE_QUANTILE)),
        "d2_max": float(d2_val_r1.max()),
    }

    # ===== Raw full Σ (canonical, may fail Cholesky on near-singular cov) =====
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None, rank1_stats

    identity = np.eye(cov.shape[0], dtype=np.float64)
    inv_sigma = np.linalg.solve(
        chol.T, np.linalg.solve(chol, identity)
    )
    inv_sigma = (inv_sigma + inv_sigma.T) * 0.5

    whitened = np.linalg.solve(chol, diff_val.T).T
    d2_val_full = np.sum(whitened * whitened, axis=1)
    if not np.all(np.isfinite(d2_val_full)) or np.any(d2_val_full < 0):
        return None, rank1_stats

    full_stats = {
        "mu": mu.astype(np.float32).tolist(),
        "sigma_inv": inv_sigma.astype(np.float32).tolist(),
        "n": int(n_prof),
        "n_val": int(n_val),
        "d2_q50": float(np.quantile(d2_val_full, 0.50)),
        "d2_q75": float(np.quantile(d2_val_full, 0.75)),
        "d2_q95": float(np.quantile(d2_val_full, GATE_QUANTILE)),
        "d2_max": float(d2_val_full.max()),
    }
    return full_stats, rank1_stats


def _load_asin_users() -> dict[str, list[str]]:
    if not ASIN_USERS_PATH.exists():
        raise FileNotFoundError(
            f"missing: {ASIN_USERS_PATH} (run 02_user_review_sentence_extract first)"
        )
    with open(ASIN_USERS_PATH) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("asin_to_users.json must be an object")
    out: dict[str, list[str]] = {}
    for asin, users in raw.items():
        if not isinstance(asin, str) or not isinstance(users, list):
            raise ValueError("asin_to_users.json must map string ASINs to user lists")
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


def main() -> None:
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
    for k, uid_idx in enumerate(eligible_indices):
        p_start, p_end = int(prof_offsets[uid_idx]), int(prof_offsets[uid_idx + 1])
        v_start, v_end = int(val_offsets[uid_idx]), int(val_offsets[uid_idx + 1])
        full_stats, rank1_stats = fit_one_user(
            z_profile[p_start:p_end], z_val[v_start:v_end]
        )
        uid = uid_list[uid_idx]
        if full_stats is None and rank1_stats is None:
            n_full_singular += 1
            continue
        if full_stats is None:
            n_full_only += 0  # rank1 succeeded but full failed
            n_rank1_only += 1
        if rank1_stats is None:
            # impossible if full_stats is not None (rank1 always finite on valid eigs)
            continue
        if full_stats is not None:
            users_full[uid] = full_stats
        users_rank1[uid] = rank1_stats
        if (k + 1) % 200 == 0:
            log(f"    fitted {k + 1}/{len(eligible_indices)} users")

    log(
        f"  full Σ fitted={len(users_full)}  rank1+residual fitted={len(users_rank1)}  "
        f"rank1_only={n_rank1_only}  both_failed={n_full_singular}"
    )

    asin_to_users = _load_asin_users()
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


if __name__ == "__main__":
    main()
