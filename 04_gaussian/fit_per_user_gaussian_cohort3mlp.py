#!/usr/bin/env python3
"""Stage 04 variant: per-user Gaussian on top of the cohort-3 MLP encoder.

Pipeline (no new code in fit_per_user_gaussian.py):
  1) load strict3 NPZ → keep uid_list, profile_idx, val_idx, z_source (32d supervised)
  2) load cohort3 MLP weights → forward 256d SVD-z of each row → 64d z_cohort3mlp
  3) call fit_one_user(z_prof, z_val) → full Σ + rank1+residual stats per user
  4) call _build_cohort_gates on cohort3mlp-z users
  5) write user_gaussian_stats_cohort3mlp{,_rank1}.json (parallel schema to canonical)

This is a variant artifact — does NOT touch the supervised canonical
user_gaussian_stats.json / user_gaussian_stats_rank1.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

# Re-use all the kernel / gates / numba JIT code from the canonical Stage 4 script
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")
from fit_per_user_gaussian import (
    fit_one_user,
    _build_numba_kernel,
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
    MIN_EIGEN_RATIO,
    GATE_QUANTILE,
    RANK1_SR_FLOOR,
    SCHEMA_VERSION_RANK1,
    CACHE_DIR,
    OUT_PATH,
    OUT_PATH_RANK1,
    ASIN_USERS_PATH,
)
from syntax_encoder import StyleMLP  # in 03_spacy_encode/

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
ENCODER_WEIGHTS_CANDIDATES = [
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort2_mlp16_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp16_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp16_10_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort2_mlp16_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp16_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp32_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_10ep.pt"),
]
COHORT3_TRAINED_UIDS = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_trained_uids.json")
SVD_COMPONENTS = CACHE_DIR / "svd_components.npz"
SENT_VECTORS = CACHE_DIR / "sent_vectors.npz"
OUT_VARIANT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/04_gaussian/user_gaussian_stats_cohort3mlp16_30.json")
OUT_VARIANT_RANK1 = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_rank1.json")
OUT_VARIANT.parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
SMOKE = False
N_SMOKE_USERS = 500
N_SMOKE_ASINS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    log("=== Stage 04 variant — per-user Gaussian on cohort-3 MLP 64d encoder ===")

    # Step 1: load strict3 NPZ (uid_list, profile_idx, val_idx) and SVD-z matrix
    z_prof_sup, z_val_sup, uid_list, prof_offsets, val_offsets = _load_embedding_groups()
    # We only need uid_list + offsets + profile_idx/val_idx, not the supervised z values
    npz = np.load(CACHE_DIR / "strict3_embeddings.npz", allow_pickle=False)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(npz["val_idx"], dtype=np.int64)
    log(f"  loaded strict3: {len(uid_list)} uids, profile={len(profile_idx)} val={len(val_idx)}")

    # Step 2: load SVD-z matrix (256d) and forward through cohort3 MLP → 64d
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
    state = torch.load(weights_path, map_location=DEVICE)
    enc.load_state_dict(state)
    enc.eval()
    log(f"  loaded encoder weights from {weights_path}")

    # Encode profile rows and val rows in chunks
    def encode_rows(rows: np.ndarray, bs: int = 4096) -> np.ndarray:
        out = np.empty((len(rows), 16), dtype=np.float32)
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

    # Reorder by user (mirror _load_embedding_groups logic for offsets)
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = [int(n) for n in json.load(f)]
    uid_idx_per_row = np.repeat(np.arange(len(uid_list), dtype=np.int64), user_n_sents)
    prof_uid = uid_idx_per_row[profile_idx]
    val_uid = uid_idx_per_row[val_idx]
    prof_order = np.argsort(prof_uid, kind="stable")
    val_order = np.argsort(val_uid, kind="stable")
    z_prof_mlp = z_prof_mlp[prof_order]
    z_val_mlp = z_val_mlp[val_order]

    eligible_indices = [
        i
        for i in range(len(uid_list))
        if prof_offsets[i + 1] - prof_offsets[i] >= MIN_PROFILE_SENTS
        and val_offsets[i + 1] - val_offsets[i] >= MIN_VAL_SENTS
    ]
    # Restrict to the user ids the cohort-3 MLP actually saw during training
    # (prevents encoding z for the rest of strict3's 72k users, whose embeddings
    # are merely forward-pass outputs without dedicated learning signal).
    if not COHORT3_TRAINED_UIDS.exists():
        raise FileNotFoundError(
            f"missing cohort3-trained uid whitelist: {COHORT3_TRAINED_UIDS} "
            "(run train_real_asin_cohort3.py once with cohort3 uids dumped)")
    with open(COHORT3_TRAINED_UIDS) as f:
        trained_uid_set = set(json.load(f))
    trained_uid_index_set = {
        i for i, u in enumerate(uid_list) if u in trained_uid_set
    }
    before = len(eligible_indices)
    eligible_indices = [i for i in eligible_indices if i in trained_uid_index_set]
    log(
        f"  eligible users: {len(eligible_indices)} "
        f"(from {before} strict3-eligible, restricted to cohort3-trained uids; "
        f"trained={len(trained_uid_index_set)})"
    )
    if SMOKE:
        eligible_indices = eligible_indices[:N_SMOKE_USERS]

    # Step 3: warm up Numba and dispatch fitting
    log("  warming up Numba kernel (first-call compile)...")
    warm_start = time.time()
    _build_numba_kernel()(
        np.ascontiguousarray(z_prof_mlp[:MIN_PROFILE_SENTS + 1], dtype=np.float64),
        np.ascontiguousarray(z_val_mlp[:MIN_VAL_SENTS + 1], dtype=np.float64),
        PSD_FLOOR, MIN_EIGEN_RATIO, GATE_QUANTILE, RANK1_SR_FLOOR,
    )
    log(f"  Numba kernel ready ({time.time() - warm_start:.1f}s)")

    users_full: dict[str, dict] = {}
    users_rank1: dict[str, dict] = {}
    n_full_singular = 0
    n_rank1_only = 0
    t_fit = time.time()
    for k, uid_idx in enumerate(eligible_indices):
        p_start, p_end = int(prof_offsets[uid_idx]), int(prof_offsets[uid_idx + 1])
        v_start, v_end = int(val_offsets[uid_idx]), int(val_offsets[uid_idx + 1])
        full_stats, rank1_stats = fit_one_user(z_prof_mlp[p_start:p_end], z_val_mlp[v_start:v_end])
        if full_stats is None and rank1_stats is None:
            n_full_singular += 1
        elif full_stats is None:
            n_rank1_only += 1
            users_rank1[uid_list[uid_idx]] = rank1_stats
        elif rank1_stats is None:
            continue
        else:
            users_full[uid_list[uid_idx]] = full_stats
            users_rank1[uid_list[uid_idx]] = rank1_stats
        if (k + 1) % 1000 == 0 or k == len(eligible_indices) - 1:
            log(f"    fitted {k + 1}/{len(eligible_indices)} users "
                f"({time.time() - t_fit:.1f}s)")
    log(
        f"  full Σ fitted={len(users_full)}  rank1+residual fitted={len(users_rank1)}  "
        f"rank1_only={n_rank1_only}  both_failed={n_full_singular}"
    )

    # Step 4: cohort gates
    asin_to_users = _load_asin_users()
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

    # full Σ payload
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
            "syntax_dim": 16,
            "encoder_source": "cohort2_mlp16_30_30ep",
            "encoder_weights": str(weights_path),
            "trained_uids_path": str(COHORT3_TRAINED_UIDS),
            "strict3_source": str(CACHE_DIR / "strict3_embeddings.npz"),
            "strict3_uid_list_size": len(uid_list),
            "n_uids_total": len(uid_list),
            "n_uids_eligible": len(eligible_indices),
            "n_uids_fitted": len(users_full),
            "n_uids_singular": n_full_singular,
            "n_uids_nonpositive_covariance": n_full_singular,
            "n_asins_with_cohort": len(cohort_gates_full),
            "n_cohort_pairs": n_pairs_full,
            "n_asins_considered": len(asin_to_users),
            "asin_users_source": str(ASIN_USERS_PATH),
            "cohort_fingerprint": str(_strict3_manifest.get("cohort_fingerprint", "")),
            "uid_layout_fingerprint": str(_strict3_manifest.get("uid_layout_fingerprint", "")),
            "cache_schema_version": int(_strict3_manifest.get("cache_schema_version", 0)),
            "note": "Variant: 32d cohort-3 MLP encoder (30-pair, 30ep, cohort=3) → per-user "
                    "raw full-covariance Gaussian; cohort gate_T = validation d2_q95; "
                    "user set restricted to the cohort3-trained uids only "
                    "(see config.trained_uids_path); "
                    "does NOT replace canonical supervised 32d artifact.",
        },
        "users": users_full,
        "cohort_gates": cohort_gates_full,
    }
    with open(OUT_VARIANT, "w") as f:
        json.dump(payload_full, f, ensure_ascii=False)
    log(
        f"  wrote → {OUT_VARIANT} ({len(users_full)} users, "
        f"{len(cohort_gates_full)} ASINs, {OUT_VARIANT.stat().st_size // 1024} KB)"
    )

    # rank1+residual payload
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
            "syntax_dim": 16,
            "encoder_source": "cohort2_mlp16_30_30ep",
            "encoder_weights": str(weights_path),
            "trained_uids_path": str(COHORT3_TRAINED_UIDS),
            "strict3_source": str(CACHE_DIR / "strict3_embeddings.npz"),
            "n_uids_total": len(uid_list),
            "n_uids_eligible": len(eligible_indices),
            "n_uids_fitted": len(users_rank1),
            "n_uids_singular": 0,
            "n_uids_nonpositive_covariance": 0,
            "n_asins_with_cohort": len(cohort_gates_rank1),
            "n_cohort_pairs": n_pairs_rank1,
            "n_asins_considered": len(asin_to_users),
            "n_full_users": len(users_full),
            "n_rank1_only_users": n_rank1_only,
            "note": "Variant: cohort3 MLP 64d rank1+residual Gaussian; same schema as canonical rank1.",
        },
        "users": users_rank1,
        "cohort_gates": cohort_gates_rank1,
    }
    with open(OUT_VARIANT_RANK1, "w") as f:
        json.dump(payload_rank1, f, ensure_ascii=False)
    log(
        f"  wrote → {OUT_VARIANT_RANK1} ({len(users_rank1)} users, "
        f"{len(cohort_gates_rank1)} ASINs, {OUT_VARIANT_RANK1.stat().st_size // 1024} KB)"
    )

    _add_theoretical_gates(OUT_VARIANT, "cohort3mlp full Σ schema", z_dim=16)
    _add_theoretical_gates(OUT_VARIANT_RANK1, "cohort3mlp rank1 schema", z_dim=16)
    log(f"=== Stage 04 cohort3mlp variant DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()