#!/usr/bin/env python3
"""Stage 05 audit variant — raw covariance validity on cohort-3 MLP 64d encoder.

Same protocol as raw_cov_validity.py (E2 val predictive + E4 bootstrap stability):
  - K = 64 (cohort-3 MLP output dim)
  - z source = strict3 profile_idx/val_idx × 256d SVD-z → 64d via cohort3 MLP
  - users restricted to the cohort3-trained uids (same whitelist as Stage 4 variant)

Re-uses evaluate_one / evaluate bootstrap / log utilities from raw_cov_validity.py
without duplicating the protocol. Writes a variant artifact next to the supervised
canonical raw_cov_validity.json — does not touch the canonical file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/05_gaussian_audit")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")

import torch
from raw_cov_validity import (
    evaluate_one,
    K_DIM as _K_SUP,
    MIN_VAL_SENTS,
    DIAG_LAMBDA,
    SUB_B,
    SUB_FRAC,
    E3_REL_THR,
    E4_PASS_RATE,
    E4_RHO_THRESHOLD,
    CACHE_DIR,
    EMBED_PATH,
    N_SENTS_PATH,
    OUT_DIR,
    _atomic_json_dump,
    _validate_strict3_artifact,
    _extract_cache_fingerprint,
    _sanitize_json,
    log,
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
K_DIM = 32  # cohort3 MLP output dim (this run)
SEED = 42
OUT_PATH = OUT_DIR / "raw_cov_validity_cohort3mlp32.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_cohort3mlp_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Mirror load_assets() but z = strict3 indices × SVD-z × cohort3 MLP → 64d."""
    d = np.load(EMBED_PATH, allow_pickle=True)
    n_sents = [int(n) for n in json.load(open(N_SENTS_PATH))]
    uid_list = [str(u) for u in d["uid_list"]]
    n_users_actual = len(uid_list)
    if len(n_sents) != n_users_actual:
        raise ValueError(
            f"user_n_sents ({len(n_sents)}) != uid_list ({n_users_actual})")
    _validate_strict3_artifact(d, n_sents)

    pid = np.asarray(d["profile_idx"], dtype=np.int64)
    vid = np.asarray(d["val_idx"], dtype=np.int64)
    row2uid = np.repeat(np.arange(n_users_actual), n_sents)
    prof_uid = row2uid[pid]
    val_uid = row2uid[vid]

    # Load SVD-z matrix (256d) and forward through cohort3 MLP → 64d
    with np.load(SVD_COMPONENTS) as npz_svd:
        Vt = np.asarray(npz_svd["Vt"], dtype=np.float32)
    sd = np.load(SENT_VECTORS, allow_pickle=True)
    X = sp.csr_matrix(
        (sd["data"].astype(np.float32), sd["indices"].astype(np.int32), sd["indptr"].astype(np.int32)),
        shape=tuple(sd["shape"]),
    )
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

    log(f"  encoding {len(pid)} profile rows + {len(vid)} val rows via cohort3 MLP...")
    z_p = encode_rows(pid)
    z_v = encode_rows(vid)
    del z_all

    op = np.argsort(prof_uid, kind="stable")
    ov = np.argsort(val_uid, kind="stable")
    z_p_sorted = z_p[op]
    z_v_sorted = z_v[ov]
    p_uid_sorted = prof_uid[op]
    v_uid_sorted = val_uid[ov]
    cp = np.bincount(p_uid_sorted, minlength=n_users_actual)
    cv = np.bincount(v_uid_sorted, minlength=n_users_actual)
    sp_off = np.concatenate([[0], np.cumsum(cp)])
    sv = np.concatenate([[0], np.cumsum(cv)])
    return z_p_sorted, z_v_sorted, sp_off, sv, uid_list, n_users_actual


def main() -> None:
    t0 = time.time()
    log("=== audit_raw_cov_validity_cohort3mlp (variant: 64d cohort-3 MLP encoder) ===")
    z_p_s, z_v_s, sp, sv, uid_list, n_users_actual = load_cohort3mlp_assets()
    log(f"  K={K_DIM}  N_USERS={n_users_actual}  MIN_VAL_SENTS={MIN_VAL_SENTS}  "
        f"SUB_B={SUB_B}  SUB_FRAC={SUB_FRAC}  E4_THR=pass>={E4_PASS_RATE}, rho>={E4_RHO_THRESHOLD}")
    log(f"  loaded z_profile={z_p_s.shape}  z_val={z_v_s.shape}")

    # Restrict to cohort3-trained uids (same whitelist as Stage 4 variant)
    if not TRAINED_UIDS_PATH.exists():
        raise FileNotFoundError(
            f"missing cohort3-trained uid whitelist: {TRAINED_UIDS_PATH}")
    with open(TRAINED_UIDS_PATH) as f:
        trained_uid_set = set(json.load(f))
    trained_uid_index_set = {
        i for i, u in enumerate(uid_list) if u in trained_uid_set
    }
    eligible = sorted(trained_uid_index_set)
    log(
        f"  eligible users: {len(eligible)} "
        f"(cohort3-trained uid whitelist; raw strict3={n_users_actual})"
    )

    per_user: dict = {}
    fail_counts = {"E2a_finite": 0, "E2b_beats_diag": 0, "E4_stability": 0}
    fail_union: set = set()
    eff_dim_hist: dict = {}
    eff_dim_per_user: list = []
    eff_rank_list: list = []
    d_pr_list: list = []
    valid_users = 0
    min_eigs: list = []
    conds: list = []
    margins: list = []
    e4_pass_rates: list = []
    e4_rhos: list = []
    rng = np.random.default_rng(SEED)

    for k, u in enumerate(eligible):
        P = z_p_s[sp[u]:sp[u + 1]]
        V = z_v_s[sv[u]:sv[u + 1]]
        r = evaluate_one(P, V, K_DIM, rng)
        per_user[uid_list[u]] = r
        eff_dim_hist[r["e3_eff_dim_rel"]] = eff_dim_hist.get(r["e3_eff_dim_rel"], 0) + 1
        eff_dim_per_user.append(int(r["e3_eff_dim_rel"]))
        eff_rank_list.append(r["e3_effective_rank"])
        d_pr_list.append(r["e3_participation_ratio"])
        if not r["valid_gaussian"]:
            fail_union.add(u)
            if not r["e2a_finite"]:
                fail_counts["E2a_finite"] += 1
            if not r["e2b_beats_diag"]:
                fail_counts["E2b_beats_diag"] += 1
            if r["e1_sample"] and r["e2_predict"] and not r["e4_pass"]:
                fail_counts["E4_stability"] += 1
        else:
            valid_users += 1
            min_eigs.append(r["min_eigval"])
            conds.append(r["cond_ratio"])
            margins.append(r["logp_margin"])
            e4_pass_rates.append(r["e4_pass_rate"])
            e4_rhos.append(r["e4_rho_med"])
        if (k + 1) % 100 == 0 or k == len(eligible) - 1:
            log(f"  {k + 1}/{len(eligible)}  valid={valid_users}  "
                f"fail_union={len(fail_union)}  t={time.time()-t0:.1f}s")

    n_total = len(eligible)
    n_fail_union = len(fail_union)
    n_valid = valid_users
    valid_rate = n_valid / n_total if n_total else 0.0

    summary = {
        "config": {
            "schema_version": 1,
            "encoder_source": "cohort3_mlp32_30ep",
            "trained_uids_path": str(TRAINED_UIDS_PATH),
            "K": K_DIM,
            "MIN_VAL_SENTS": MIN_VAL_SENTS,
            "DIAG_LAMBDA": DIAG_LAMBDA,
            "SUB_B": SUB_B,
            "SUB_FRAC": SUB_FRAC,
            "E3_REL_THR": E3_REL_THR,
            "E4_PASS_RATE": E4_PASS_RATE,
            "E4_RHO_THRESHOLD": E4_RHO_THRESHOLD,
            "SEED": SEED,
            "note": "Variant: raw covariance validity audit on the 32d cohort-3 MLP "
                    "encoder; users restricted to cohort3-trained uid whitelist; "
                    "same E2/E4 protocol as raw_cov_validity.py.",
        },
        "n_total": n_total,
        "n_valid": n_valid,
        "n_fail_union": n_fail_union,
        "valid_rate": valid_rate,
        "fail_counts": fail_counts,
        "eff_dim_hist": eff_dim_hist,
        "min_eigval_p50": float(np.percentile(min_eigs, 50)) if min_eigs else None,
        "min_eigval_p10": float(np.percentile(min_eigs, 10)) if min_eigs else None,
        "cond_ratio_p50": float(np.percentile(conds, 50)) if conds else None,
        "cond_ratio_p90": float(np.percentile(conds, 90)) if conds else None,
        "logp_margin_p50": float(np.percentile(margins, 50)) if margins else None,
        "logp_margin_p10": float(np.percentile(margins, 10)) if margins else None,
        "e4_pass_rate_p50": float(np.percentile(e4_pass_rates, 50)) if e4_pass_rates else None,
        "e4_rho_p50": float(np.percentile(e4_rhos, 50)) if e4_rhos else None,
        "per_user": per_user,
    }
    _atomic_json_dump(_sanitize_json(summary), OUT_PATH)
    log(f"  wrote → {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    log(f"  FINAL: valid_gaussian={n_valid}/{n_total} ({valid_rate*100:.2f}%)  "
        f"fail_union={n_fail_union}  t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()