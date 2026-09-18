#!/usr/bin/env python3
"""Stage 04 cohort3mlp low-rank + diagonal variant.

Same data source / asset loading as fit_per_user_gaussian_cohort3mlp.py, but
replaces rank1+isotropic-residual fit with a generic low-rank + per-dim-diagonal
parameterization:
    Σ = Σ_i λ_i v_i v_i^T + diag(σ_d^2)
    D^2 = (V^T (z-μ))^2 / λ + ||r_diag||^2 / σ_d^2
         where r_diag = (I - V V^T) (z-μ) (residual after removing low-rank subspace)
         + diag correction for σ_d^2 per dim

K_RANK controls the low-rank dimension (1 = rank1 + per-dim residual diagonal;
2 = rank2 + diag; etc.). K_RANK is hardcoded at the top.

This is a variant artifact — does NOT touch the canonical supervised artifacts
or the rank1 cohort3mlp artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")

import torch
from fit_per_user_gaussian import (
    _build_cohort_gates,
    _add_theoretical_gates,
    _load_asin_users,
    _load_embedding_groups,
)
from fit_per_user_gaussian import (
    MIN_PROFILE_SENTS,
    MIN_VAL_SENTS,
    SCHEMA_VERSION,
    CANONICAL_PRODUCT_KEY,
    PSD_FLOOR,
    CACHE_DIR,
    ASIN_USERS_PATH,
    GATE_QUANTILE,
    SMOKE,
    N_SMOKE_USERS,
    N_SMOKE_ASINS,
)
from train_real_asin_cohort3 import StyleMLP

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
ENCODER_WEIGHTS_CANDIDATES = [
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp32_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_10ep.pt"),
]
TRAINED_UIDS_PATH = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_trained_uids.json")
SVD_COMPONENTS = CACHE_DIR / "svd_components.npz"
SENT_VECTORS = CACHE_DIR / "sent_vectors.npz"

K_DIM = 32        # cohort3 MLP output dim (current run)
K_RANK = 1         # low-rank dimension; 1 → rank1 + per-dim diag residual
DIAG_FLOOR = 1e-4  # floor on per-dim residual sigma^2 (avoids /0)

OUT_VARIANT = Path(
    f"/home/wlia0047/ar57/wenyu/PersoanlQuery/result/04_gaussian/"
    f"user_gaussian_stats_cohort3mlp32_lowrankdiag_rank{K_RANK}.json"
)
OUT_VARIANT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_one_user_lowrank_diag(P: np.ndarray, V: np.ndarray,
                              k_rank: int = K_RANK,
                              z_dim: int = K_DIM) -> dict | None:
    """Fit Σ = Σ_i λ_i v_i v_i^T + diag(σ_d²) per user.

    P (n_p, z_dim) profile sentences; V (n_v, z_dim) val sentences.
    Returns dict with d2 quantiles on val, mu, lambdas, V (z_dim, k_rank),
    sigma_diag_sq (z_dim,), n, n_val. None on failure (n_p < z_dim+k_rank+1).
    """
    n_p, _ = P.shape
    n_v = len(V)
    min_n = z_dim + k_rank + 1
    if n_p < min_n:
        return None

    P = np.ascontiguousarray(P, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)

    mu = P.mean(axis=0)
    P_c = P - mu
    cov = (P_c.T @ P_c) / (n_p - 1)

    # Eigen-decompose; take top-k_rank eigen-pairs by eigenvalue.
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; take the last k_rank.
    idx = np.argsort(eigvals)[::-1][:k_rank]
    lambdas = eigvals[idx]
    V_eig = eigvecs[:, idx]                       # (z_dim, k_rank)
    lambdas = np.clip(lambdas, PSD_FLOOR, None)   # floor on λ

    # Diagonal residual = trace of (cov - V_eig Λ V_eig^T) divided across all z_dim.
    rank_cov = (V_eig * lambdas) @ V_eig.T        # (z_dim, z_dim) low-rank reconstruction
    diag_cov = np.clip(np.diag(cov - rank_cov), DIAG_FLOOR, None)
    sigma_diag_sq = diag_cov                     # (z_dim,)

    # D^2 on val: project onto low-rank subspace + residual.
    V_c = V - mu
    low_proj = V_c @ V_eig                       # (n_v, k_rank)
    low_d2 = np.sum(low_proj ** 2 / lambdas, axis=1)  # (n_v,)

    # Residual component: r = V_c - low_proj @ V_eig.T, then per-dim |r|^2 / σ_d².
    residual = V_c - low_proj @ V_eig.T
    diag_d2 = np.sum(residual ** 2 / sigma_diag_sq, axis=1)  # (n_v,)
    d2 = low_d2 + diag_d2

    d2_sorted = np.sort(d2)
    q50 = float(d2_sorted[int(0.50 * (n_v - 1))])
    q75 = float(d2_sorted[int(0.75 * (n_v - 1))])
    q95 = float(d2_sorted[int(0.95 * (n_v - 1))])
    d2_max = float(d2_sorted[-1])

    return {
        "mu": mu.astype(np.float32).tolist(),
        "lambdas": lambdas.astype(np.float32).tolist(),
        "V": V_eig.astype(np.float32).tolist(),
        "sigma_diag_sq": sigma_diag_sq.astype(np.float32).tolist(),
        "n": int(n_p),
        "n_val": int(n_v),
        "d2_q50": q50,
        "d2_q75": q75,
        "d2_q95": q95,
        "d2_max": d2_max,
        "k_rank": int(k_rank),
    }


def main() -> None:
    t0 = time.time()
    log(f"=== Stage 04 cohort3mlp low-rank+diagonal variant (K_RANK={K_RANK}) ===")

    _z_prof_sup, _z_val_sup, uid_list, prof_offsets, val_offsets = _load_embedding_groups()
    npz = np.load(CACHE_DIR / "strict3_embeddings.npz", allow_pickle=False)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(npz["val_idx"], dtype=np.int64)
    log(f"  loaded strict3: {len(uid_list)} uids, profile={len(profile_idx)} val={len(val_idx)}")

    with np.load(SVD_COMPONENTS) as npz_svd:
        Vt = np.asarray(npz_svd["Vt"], dtype=np.float32)
    d = np.load(SENT_VECTORS, allow_pickle=True)
    X = sp.csr_matrix(
        (d["data"].astype(np.float32), d["indices"].astype(np.int32), d["indptr"].astype(np.int32)),
        shape=tuple(d["shape"]),
    )
    log(f"  loaded SVD Vt={Vt.shape}, sent_vectors X={X.shape}")
    z_all = (X @ Vt.T).astype(np.float32)
    del X

    enc = StyleMLP().to(DEVICE)
    weights_path = next((p for p in ENCODER_WEIGHTS_CANDIDATES if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"no encoder weights found in {ENCODER_WEIGHTS_CANDIDATES}")
    enc.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    enc.eval()
    log(f"  loaded encoder weights from {weights_path}")

    def encode_rows(rows: np.ndarray, bs: int = 4096) -> np.ndarray:
        out = np.empty((len(rows), K_DIM), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(rows), bs):
                idx = rows[s:s + bs]
                z_t = torch.from_numpy(z_all[idx]).to(DEVICE)
                out[s:s + bs] = enc(z_t).cpu().numpy()
        return out

    log(f"  encoding {len(profile_idx)} profile rows + {len(val_idx)} val rows via cohort3 MLP...")
    z_prof_mlp = encode_rows(profile_idx)
    z_val_mlp = encode_rows(val_idx)
    del z_all
    log(f"  encoded: z_prof_mlp={z_prof_mlp.shape}, z_val_mlp={z_val_mlp.shape}")

    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = [int(n) for n in json.load(f)]
    uid_idx_per_row = np.repeat(np.arange(len(uid_list), dtype=np.int64), user_n_sents)
    prof_uid = uid_idx_per_row[profile_idx]
    val_uid = uid_idx_per_row[val_idx]
    prof_order = np.argsort(prof_uid, kind="stable")
    val_order = np.argsort(val_uid, kind="stable")
    z_prof_mlp = z_prof_mlp[prof_order]
    z_val_mlp = z_val_mlp[val_order]

    if not TRAINED_UIDS_PATH.exists():
        raise FileNotFoundError(
            f"missing cohort3-trained uid whitelist: {TRAINED_UIDS_PATH}")
    with open(TRAINED_UIDS_PATH) as f:
        trained_uid_set = set(json.load(f))
    trained_uid_index_set = {
        i for i, u in enumerate(uid_list) if u in trained_uid_set
    }
    eligible_indices = [
        i
        for i in range(len(uid_list))
        if prof_offsets[i + 1] - prof_offsets[i] >= MIN_PROFILE_SENTS
        and val_offsets[i + 1] - val_offsets[i] >= MIN_VAL_SENTS
        and i in trained_uid_index_set
    ]
    log(f"  eligible users: {len(eligible_indices)}")

    users_lrd: dict[str, dict] = {}
    n_failed = 0
    t_fit = time.time()
    for k, uid_idx in enumerate(eligible_indices):
        p_start, p_end = int(prof_offsets[uid_idx]), int(prof_offsets[uid_idx + 1])
        v_start, v_end = int(val_offsets[uid_idx]), int(val_offsets[uid_idx + 1])
        stats = fit_one_user_lowrank_diag(
            z_prof_mlp[p_start:p_end], z_val_mlp[v_start:v_end]
        )
        if stats is None:
            n_failed += 1
            continue
        users_lrd[uid_list[uid_idx]] = stats
        if (k + 1) % 200 == 0 or k == len(eligible_indices) - 1:
            log(f"    fitted {k + 1}/{len(eligible_indices)} users "
                f"({time.time() - t_fit:.1f}s)")
    log(f"  lowrank+diag fitted={len(users_lrd)}  failed={n_failed}")

    asin_to_users = _load_asin_users()
    cohort_gates_lrd = _build_cohort_gates(asin_to_users, users_lrd, smoke_asins=None)
    n_pairs_lrd = sum(len(v) for v in cohort_gates_lrd.values())
    log(f"  cohort gates: {len(cohort_gates_lrd)} ASINs / {n_pairs_lrd} pairs")

    with open(CACHE_DIR / "strict3_manifest.json") as _mf:
        _strict3_manifest = json.load(_mf)

    payload = {
        "config": {
            "schema_version": SCHEMA_VERSION,
            "canonical_product_key": CANONICAL_PRODUCT_KEY,
            "smoke": False,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "covariance": f"lowrank_rank{K_RANK}_plus_diagonal_residual",
            "k_rank": K_RANK,
            "covariance_formula": (
                f"Sigma = sum_i λ_i v_i v_i^T (i=1..{K_RANK}) + diag(σ_d^2); "
                "λ from top-k eig of full Σ, σ_d^2 = diag(Σ - V Λ V^T) clipped."
            ),
            "d2_formula": (
                "D^2 = Σ_i (v_i^T (z-μ))^2 / λ_i + Σ_d r_d^2 / σ_d^2, "
                "r = (I - V V^T)(z-μ) residual after low-rank subspace projection."
            ),
            "diag_lambda_floor": DIAG_FLOOR,
            "psd_floor": PSD_FLOOR,
            "gate_quantile": GATE_QUANTILE,
            "gate_quantiles": [0.50, 0.75, 0.95],
            "syntax_dim": K_DIM,
            "encoder_source": "cohort3_mlp32_30ep",
            "encoder_weights": str(weights_path),
            "trained_uids_path": str(TRAINED_UIDS_PATH),
            "strict3_source": str(CACHE_DIR / "strict3_embeddings.npz"),
            "n_uids_total": len(uid_list),
            "n_uids_eligible": len(eligible_indices),
            "n_uids_fitted": len(users_lrd),
            "n_uids_failed": n_failed,
            "n_asins_with_cohort": len(cohort_gates_lrd),
            "n_cohort_pairs": n_pairs_lrd,
            "n_asins_considered": len(asin_to_users),
            "asin_users_source": str(ASIN_USERS_PATH),
            "cohort_fingerprint": str(_strict3_manifest.get("cohort_fingerprint", "")),
            "uid_layout_fingerprint": str(_strict3_manifest.get("uid_layout_fingerprint", "")),
            "cache_schema_version": int(_strict3_manifest.get("cache_schema_version", 0)),
            "note": (
                f"Variant: {K_RANK}-rank + per-dim-diagonal cohort3mlp 32d Gaussian; "
                "more general than rank1+isotropic-residual (k_rank=1 reduces to "
                "rank1 but with per-dim diag residual instead of isotropic residual); "
                "does NOT replace canonical supervised 32d artifact."
            ),
        },
        "users": users_lrd,
        "cohort_gates": cohort_gates_lrd,
    }
    with open(OUT_VARIANT, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    log(
        f"  wrote → {OUT_VARIANT} ({len(users_lrd)} users, "
        f"{len(cohort_gates_lrd)} ASINs, {OUT_VARIANT.stat().st_size // 1024} KB)"
    )

    _add_theoretical_gates(OUT_VARIANT, f"cohort3mlp lowrank_rank{K_RANK}+diag schema", z_dim=K_DIM)
    log(f"=== Stage 04 cohort3mlp lowrank+diag DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()