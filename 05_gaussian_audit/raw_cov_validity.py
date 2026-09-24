#!/usr/bin/env python3
"""Stage 05 audit — A1+A2+A3 hard rule on cohort-3 MLP 16d encoder.

Σ = rank1 v v^T + diag(σ_d²), D² via Sylvester:
  D² = (v^T (z-μ))² / λ + Σ_d r_d² / σ_d²,  r = (I - v v^T)(z-μ)

  A1 Numerical Validity:  λ>0, σ_d²>0, log|Σ| finite, all val D² finite
  A2 Held-out Calibration: coverage95 ≥ 0.8  AND  median(val D²) ≤ chi²_{16,0.95}=26.30
  A3 Stability:           20 × 80% subsampling → valid_fit_rate ≥ 0.9,
                          median Spearman ρ ≥ 0.8, median gate-decision agreement ≥ 0.8

valid_gaussian = A1 ∧ A2 ∧ A3.

数据源: strict3 profile/val z × SVD-z × cohort3 MLP → 16d,
       用户限定在 cohort3-trained uids (与 Stage 04 cohort3mlp 一致)。

输出:
  result/05_gaussian_audit/raw_cov_validity.json  (A1/A2/A3 schema)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")

import torch
from scipy.stats import chi2 as _chi2
from scipy.stats import spearmanr  # noqa: E402  (avoid local import in hot loop)
from syntax_encoder import StyleMLP

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
EMBED_PATH = CACHE_DIR / "strict3_embeddings.npz"
MANIFEST_PATH = CACHE_DIR / "strict3_manifest.json"
READY_PATH = CACHE_DIR / "cache_ready.json"
N_SENTS_PATH = CACHE_DIR / "user_n_sents.json"
SVD_COMPONENTS = CACHE_DIR / "svd_components.npz"
SENT_VECTORS = CACHE_DIR / "sent_vectors.npz"

OUT_DIR = REPO_ROOT / "result/05_gaussian_audit"
OUT_PATH = OUT_DIR / "raw_cov_validity.json"
# 用户指令 2026-09-23: 3 个 category 各自一份 (Baby / Musical / Video_Games),
# main() 改为串行跑 3 个 domain, 产物写到 result/05_gaussian_audit/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]

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
TRAINED_UIDS_PATH = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_trained_uids.json")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

# ----- A1/A2/A3 thresholds -----
K_DIM = 16
MIN_VAL_SENTS = 1
K_RANK_LR = 1
PSD_FLOOR_LR = 1e-4
DIAG_FLOOR_LR = 1e-4
CHI2_Q95 = 0.95
COVERAGE95_MIN = 0.8
SUB_B = 20
SUB_FRAC = 0.8
SUBSAMPLE_FIT_RATE_MIN = 0.9
SUBSAMPLE_SPEARMAN_MIN = 0.8
SUBSAMPLE_GATE_AGREE_MIN = 0.8


# ---------------------------------------------------------------------------
# Atomic JSON / cache fingerprint / NaN sanitize / strict3 validate
# ---------------------------------------------------------------------------

def _atomic_json_dump(payload, path) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _extract_cache_fingerprint() -> str:
    if not READY_PATH.exists():
        return ""
    with open(READY_PATH) as f:
        return str(json.load(f).get("sentence_source_fingerprint", ""))


def _sanitize_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _validate_strict3_artifact(npz, n_sents: list[int]) -> None:
    """拒绝不属于当前 Stage 02 cohort 的 strict3 artifact。"""
    for path in (MANIFEST_PATH, READY_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"missing: {path} (run 03_spacy_encode.stage_strict3() first)")
    required = ("z_profile", "z_val",
                "profile_idx", "val_idx", "uid_list",
                "cohort_fingerprint", "uid_layout_fingerprint",
                "vocab_fingerprint")
    for key in required:
        if key not in npz:
            raise ValueError(f"strict3 artifact missing field: {key}")

    artifact_uids = [str(u) for u in np.asarray(npz["uid_list"]).tolist()]
    if len(artifact_uids) != len(n_sents):
        raise ValueError(
            f"strict3 uid_list ({len(artifact_uids)}) != user_n_sents "
            f"({len(n_sents)})")
    n_total = int(sum(n_sents))
    for key in ("profile_idx", "val_idx"):
        index = np.asarray(npz[key], dtype=np.int64)
        if index.ndim != 1:
            raise ValueError(f"strict3 {key} must be 1-D")
        if len(index) and (int(index.min()) < 0
                           or int(index.max()) >= n_total):
            raise ValueError(
                f"strict3 {key} index out of range [0,{n_total})")
        if len(np.unique(index)) != len(index):
            raise ValueError(f"strict3 {key} contains duplicate indices")
    merged = np.concatenate(
        [np.asarray(npz[k], dtype=np.int64) for k in
         ("profile_idx", "val_idx")])
    if len(np.unique(merged)) != n_total or not np.array_equal(
            np.sort(merged), np.arange(n_total, dtype=np.int64)):
        raise ValueError("strict3 split indices do not partition all sentences")
    for key in ("z_profile", "z_val"):
        z = np.asarray(npz[key])
        if z.ndim != 2 or z.shape[1] not in (16, 32):
            raise ValueError(f"strict3 {key} shape invalid: {z.shape}")
        if not np.isfinite(z).all():
            raise ValueError(f"strict3 {key} contains NaN/Inf")
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

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(READY_PATH) as f:
        ready = json.load(f)
    source_keys = (
        ("cohort_fingerprint", "sentence_source_fingerprint"),
        ("uid_layout_fingerprint", "uid_layout_fingerprint"),
        ("vocab_fingerprint", "vocab_fingerprint"),
    )
    for npz_key, ready_key in source_keys:
        npz_value = str(np.asarray(npz[npz_key]).item())
        manifest_value = str(manifest.get(npz_key))
        if npz_value != manifest_value:
            raise ValueError(
                f"strict3 artifact vs manifest mismatch at {npz_key}: "
                f"npz={npz_value}, manifest={manifest_value!r}")
        if manifest_value != str(ready.get(ready_key)):
            raise ValueError(
                f"strict3 manifest differs from cache_ready at {ready_key}")
    if int(manifest.get("n_users", -1)) != len(artifact_uids):
        raise ValueError("strict3 manifest n_users mismatch")
    if int(ready.get("n_users", -1)) != len(artifact_uids):
        raise ValueError("cache_ready n_users mismatch")
    if int(manifest.get("n_total_sents", -1)) != n_total:
        raise ValueError("strict3 manifest n_total_sents mismatch")
    if int(ready.get("n_total_sents", -1)) != n_total:
        raise ValueError("cache_ready n_total_sents mismatch")


# ---------------------------------------------------------------------------
# Stage 05 protocol: A1 ∧ A2 ∧ A3
# ---------------------------------------------------------------------------

def _chi2_thr_for_K(K: int) -> float:
    return float(_chi2.ppf(CHI2_Q95, K))


def fit_lowrank_diag(P: np.ndarray,
                     k_rank: int = K_RANK_LR,
                     z_dim: int | None = None) -> dict | None:
    """Low-rank + per-dim-diagonal Σ_u fit. Returns None when n_p < z_dim+k_rank+1."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    n_p, _ = P.shape
    if z_dim is None:
        z_dim = P.shape[1]
    min_n = z_dim + k_rank + 1
    if n_p < min_n:
        return None

    mu = P.mean(axis=0)
    P_c = P - mu
    cov = (P_c.T @ P_c) / (n_p - 1)
    eigvals_all, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals_all)[::-1][:k_rank]
    lambdas = eigvals_all[idx]
    V_eig = eigvecs[:, idx]
    lambdas = np.clip(lambdas, PSD_FLOOR_LR, None)

    rank_cov = (V_eig * lambdas) @ V_eig.T
    diag_cov = np.clip(np.diag(cov - rank_cov), DIAG_FLOOR_LR, None)
    sigma_diag_sq = diag_cov
    return {
        "mu": mu,
        "lambdas": lambdas,
        "V_eig": V_eig,
        "sigma_diag_sq": sigma_diag_sq,
        "eigvals_all": eigvals_all,
    }


def maha_d2_lowrank_diag(X: np.ndarray,
                         mu: np.ndarray,
                         V_eig: np.ndarray,
                         lambdas: np.ndarray,
                         sigma_diag_sq: np.ndarray) -> np.ndarray:
    """Vectorized Mahalanobis D² for Σ = λ v v^T + diag(σ_d²). +inf on bad input."""
    diff = np.asarray(X, dtype=np.float64) - mu[None, :]
    if not np.all(np.isfinite(diff)):
        return np.full(len(X), np.inf, dtype=np.float64)
    if np.any(lambdas <= 0) or np.any(sigma_diag_sq <= 0):
        return np.full(len(X), np.inf, dtype=np.float64)

    low_proj = diff @ V_eig
    low_d2 = np.sum(low_proj ** 2 / lambdas, axis=1)
    residual = diff - low_proj @ V_eig.T
    diag_d2 = np.sum(residual ** 2 / sigma_diag_sq, axis=1)
    return low_d2 + diag_d2


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    try:
        rho, _ = spearmanr(a, b)
        return float(rho) if np.isfinite(rho) else 0.0
    except Exception:
        return 0.0


def _a1_pass(fit: dict, d2_v: np.ndarray, n_v: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if fit is None:
        reasons.append("A1 fit failed (n_p too small)")
        return False, reasons
    lambdas = fit["lambdas"]
    sigma_diag_sq = fit["sigma_diag_sq"]
    if np.any(lambdas <= 0):
        reasons.append(f"A1 λ≤0 (λ_min={float(lambdas.min()):.3e})")
    if np.any(sigma_diag_sq <= 0):
        reasons.append(
            f"A1 σ_d²≤0 (σ_d²_min={float(sigma_diag_sq.min()):.3e})")
    logdet = (float(np.log(np.clip(lambdas, 1e-30, None)).sum()
                    + np.log(np.clip(sigma_diag_sq, 1e-30, None)).sum()))
    if not np.isfinite(logdet):
        reasons.append(f"A1 log|Σ| 非有限 (logdet={logdet:.3e})")
    if n_v < MIN_VAL_SENTS:
        reasons.append(f"A1 n_v<{MIN_VAL_SENTS} (n_v={n_v})")
    if len(d2_v) and not bool(np.all(np.isfinite(d2_v))):
        reasons.append("A1 val D² 包含非有限值")
    bad_lambda = bool(np.any(lambdas <= 0))
    bad_sigma = bool(np.any(sigma_diag_sq <= 0))
    bad_logdet = not np.isfinite(logdet)
    bad_d2 = len(d2_v) and not bool(np.all(np.isfinite(d2_v)))
    bad_nv = n_v < MIN_VAL_SENTS
    return not (bad_lambda or bad_sigma or bad_logdet or bad_d2 or bad_nv), reasons


def _a2_pass(d2_v: np.ndarray, chi2_thr: float) -> tuple[bool, float, float]:
    if len(d2_v) == 0:
        return False, 0.0, float("inf")
    coverage = float(np.mean(d2_v <= chi2_thr))
    median_d2 = float(np.median(d2_v))
    return (coverage >= COVERAGE95_MIN and median_d2 <= chi2_thr,
            coverage, median_d2)


def _a3_stability(P: np.ndarray, V: np.ndarray, K: int,
                  chi2_thr: float,
                  d2_full: np.ndarray,
                  rng: np.random.Generator) -> dict:
    n_p = len(P)
    if len(V) == 0 or n_p < K + 2:
        return {"valid_fit_rate": 0.0,
                "rho_med": 0.0,
                "gate_agreement_med": 0.0,
                "n_subs": 0}

    sub_size = max(int(SUB_FRAC * n_p), K + 2)
    sub_size = min(sub_size, n_p)

    n_pass_fit = 0
    rhos: list[float] = []
    agreements: list[float] = []
    gate_full = d2_full <= chi2_thr

    for _ in range(SUB_B):
        idx = rng.choice(n_p, size=sub_size, replace=False)
        P_sub = P[idx]
        fit_sub = fit_lowrank_diag(P_sub)
        if fit_sub is None:
            continue
        d2_sub = maha_d2_lowrank_diag(V, fit_sub["mu"], fit_sub["V_eig"],
                                       fit_sub["lambdas"], fit_sub["sigma_diag_sq"])
        if not bool(np.all(np.isfinite(d2_sub))):
            continue
        n_pass_fit += 1
        rho = spearman_rho(d2_full, d2_sub)
        rhos.append(rho)
        gate_sub = d2_sub <= chi2_thr
        agree = float(np.mean(gate_full == gate_sub))
        agreements.append(agree)

    valid_fit_rate = n_pass_fit / SUB_B if SUB_B else 0.0
    rho_med = float(np.median(rhos)) if rhos else 0.0
    agree_med = float(np.median(agreements)) if agreements else 0.0
    return {"valid_fit_rate": valid_fit_rate,
            "rho_med": rho_med,
            "gate_agreement_med": agree_med,
            "n_subs": n_pass_fit}


def evaluate_one(P: np.ndarray, V: np.ndarray, K: int,
                 rng: np.random.Generator,
                 chi2_thr: float | None = None) -> dict:
    """A1 ∧ A2 ∧ A3 三项 hard rule 评估."""
    if chi2_thr is None:
        chi2_thr = _chi2_thr_for_K(K)

    n_p = len(P)
    n_v = len(V)

    fit_full = fit_lowrank_diag(P)
    if fit_full is None:
        return {
            "n_p": n_p, "n_v": n_v,
            "a1_pass": False, "a2_pass": False, "a3_pass": False,
            "valid_gaussian": False,
            "a1_reasons": [f"n_p<{K + K_RANK_LR + 1} cannot fit lowrank Σ"],
            "a2_coverage95": 0.0,
            "a2_val_d2_p50": float("inf"),
            "a2_val_d2_p95": float("inf"),
            "a3_valid_fit_rate": 0.0,
            "a3_spearman_rho_med": 0.0,
            "a3_gate_agreement_med": 0.0,
            "a3_n_subs": 0,
            "lambdas": [], "sigma_diag_sq_min": 0.0,
        }

    d2_v_full = maha_d2_lowrank_diag(V, fit_full["mu"], fit_full["V_eig"],
                                     fit_full["lambdas"], fit_full["sigma_diag_sq"])

    a1_ok, a1_reasons = _a1_pass(fit_full, d2_v_full, n_v)
    a2_ok, coverage95, median_d2 = _a2_pass(d2_v_full, chi2_thr)
    val_d2_p95 = float(np.percentile(d2_v_full, 95)) if n_v else float("inf")
    a3 = _a3_stability(P, V, K, chi2_thr, d2_v_full, rng)
    a3_ok = (a3["valid_fit_rate"] >= SUBSAMPLE_FIT_RATE_MIN
             and a3["rho_med"] >= SUBSAMPLE_SPEARMAN_MIN
             and a3["gate_agreement_med"] >= SUBSAMPLE_GATE_AGREE_MIN)

    valid = a1_ok and a2_ok and a3_ok

    return {
        "n_p": n_p, "n_v": n_v,
        "a1_pass": a1_ok,
        "a2_pass": a2_ok,
        "a3_pass": a3_ok,
        "valid_gaussian": valid,
        "a1_reasons": a1_reasons,
        "a2_coverage95": coverage95,
        "a2_val_d2_p50": median_d2,
        "a2_val_d2_p95": val_d2_p95,
        "a3_valid_fit_rate": a3["valid_fit_rate"],
        "a3_spearman_rho_med": a3["rho_med"],
        "a3_gate_agreement_med": a3["gate_agreement_med"],
        "a3_n_subs": a3["n_subs"],
        "lambdas": fit_full["lambdas"].astype(float).tolist(),
        "sigma_diag_sq_min": float(fit_full["sigma_diag_sq"].min()),
    }


# ---------------------------------------------------------------------------
# Asset loader: strict3 indices × SVD-z × cohort3 MLP → 16d
# ---------------------------------------------------------------------------

def load_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
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

    with np.load(SVD_COMPONENTS) as npz_svd:
        Vt = np.asarray(npz_svd["Vt"], dtype=np.float32)
    sd = np.load(SENT_VECTORS, allow_pickle=True)
    X = sp.csr_matrix(
        (sd["data"].astype(np.float32), sd["indices"].astype(np.int32),
         sd["indptr"].astype(np.int32)),
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_task_body() -> None:
    t0 = time.time()
    log("=== Stage 05 audit: A1+A2+A3 on 16d lowrank+diag Σ_u (cohort3mlp encoder) ===")
    z_p_s, z_v_s, sp, sv, uid_list, n_users_actual = load_assets()
    chi2_thr = _chi2_thr_for_K(K_DIM)
    log(f"  K={K_DIM}  chi²_{{{K_DIM},{CHI2_Q95}}}={chi2_thr:.3f}  "
        f"MIN_VAL_SENTS={MIN_VAL_SENTS}  SUB_B={SUB_B}  SUB_FRAC={SUB_FRAC}  "
        f"COVERAGE95_MIN={COVERAGE95_MIN}  "
        f"A3_THR=fit>={SUBSAMPLE_FIT_RATE_MIN}, ρ>={SUBSAMPLE_SPEARMAN_MIN}, "
        f"agree>={SUBSAMPLE_GATE_AGREE_MIN}")
    log(f"  loaded z_profile={z_p_s.shape}  z_val={z_v_s.shape}")

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
    fail_counts = {"A1_num_invalid": 0,
                   "A2_calibration": 0,
                   "A3_stability": 0}
    fail_union: set = set()
    a2_coverages: list = []
    a2_d2_p50s: list = []
    a2_d2_p95s: list = []
    a3_fit_rates: list = []
    a3_rhos: list = []
    a3_agrees: list = []
    valid_users = 0
    rng = np.random.default_rng(SEED)

    for k, u in enumerate(eligible):
        P = z_p_s[sp[u]:sp[u + 1]]
        V = z_v_s[sv[u]:sv[u + 1]]
        r = evaluate_one(P, V, K_DIM, rng, chi2_thr=chi2_thr)
        per_user[uid_list[u]] = r
        a2_coverages.append(r["a2_coverage95"])
        a2_d2_p50s.append(r["a2_val_d2_p50"])
        a2_d2_p95s.append(r["a2_val_d2_p95"])
        a3_fit_rates.append(r["a3_valid_fit_rate"])
        a3_rhos.append(r["a3_spearman_rho_med"])
        a3_agrees.append(r["a3_gate_agreement_med"])
        if not r["valid_gaussian"]:
            fail_union.add(u)
            if not r["a1_pass"]:
                fail_counts["A1_num_invalid"] += 1
            if not r["a2_pass"]:
                fail_counts["A2_calibration"] += 1
            if not r["a3_pass"]:
                fail_counts["A3_stability"] += 1
        else:
            valid_users += 1
        if (k + 1) % 500 == 0 or k == len(eligible) - 1:
            log(f"  {k + 1}/{len(eligible)}  valid={valid_users}  "
                f"fail_union={len(fail_union)}  t={time.time()-t0:.1f}s")

    n_total = len(eligible)
    n_fail_union = len(fail_union)
    log(f"  FINAL:")
    log(f"    valid_gaussian        = {valid_users}/{n_total} "
        f"({valid_users/n_total*100:.2f}%)")
    log(f"    invalid (union)       = {n_fail_union}/{n_total} "
        f"({n_fail_union/n_total*100:.2f}%)")
    log(f"    fail by A1 numerical = {fail_counts['A1_num_invalid']}")
    log(f"    fail by A2 calibration = {fail_counts['A2_calibration']}")
    log(f"    fail by A3 stability  = {fail_counts['A3_stability']}")

    def _pct(arr, qs):
        arr = [x for x in arr if np.isfinite(x)]
        return np.percentile(arr, qs).tolist() if arr else None

    log(f"    A2 coverage95 (全体)   P25/50/75 = "
        f"{_pct(a2_coverages, [25, 50, 75])}")
    log(f"    A2 val_d2_p50 (全体)   P25/50/90 = "
        f"{_pct(a2_d2_p50s, [25, 50, 90])}")
    log(f"    A2 val_d2_p95 (全体)   P50/90/99 = "
        f"{_pct(a2_d2_p95s, [50, 90, 99])}")
    log(f"    A3 valid_fit_rate (全体) P10/50/90 = "
        f"{_pct(a3_fit_rates, [10, 50, 90])}")
    log(f"    A3 spearman_rho (全体) P10/50/90 = "
        f"{_pct(a3_rhos, [10, 50, 90])}")
    log(f"    A3 gate_agreement (全体) P10/50/90 = "
        f"{_pct(a3_agrees, [10, 50, 90])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "K_dim": K_DIM,
            "chi2_q95": CHI2_Q95,
            "chi2_threshold": chi2_thr,
            "covariance_formula": (
                f"Sigma = lambda * v v^T + diag(sigma_d^2), "
                f"lambda from top-{K_RANK_LR} eig of full Sigma, "
                "sigma_d^2 = diag(Sigma - v lambda v^T) clipped."),
            "d2_formula": (
                "D^2 = (v^T (z-mu))^2 / lambda + sum_d r_d^2 / sigma_d^2, "
                "r = (I - v v^T)(z-mu) residual."),
            "criteria": (
                "valid_gaussian = A1 AND A2 AND A3. "
                "A1: lambda>0, sigma_d^2>0, log|Σ| finite, all val D² finite; "
                f"A2: coverage95 (val D² <= chi²_{{{K_DIM},{CHI2_Q95}}}={chi2_thr:.3f}) "
                f">= {COVERAGE95_MIN} AND median(val D²) <= {chi2_thr:.3f}; "
                f"A3: {SUB_B} x {SUB_FRAC:.0%} subsampling → "
                f"valid_fit_rate >= {SUBSAMPLE_FIT_RATE_MIN}, "
                f"median Spearman ρ >= {SUBSAMPLE_SPEARMAN_MIN}, "
                f"median gate-decision agreement >= {SUBSAMPLE_GATE_AGREE_MIN}."),
            "min_val_sents": MIN_VAL_SENTS,
            "k_rank": K_RANK_LR,
            "psd_floor": PSD_FLOOR_LR,
            "diag_floor": DIAG_FLOOR_LR,
            "coverage95_min": COVERAGE95_MIN,
            "sub_B": SUB_B, "sub_frac": SUB_FRAC,
            "subsample_fit_rate_min": SUBSAMPLE_FIT_RATE_MIN,
            "subsample_spearman_min": SUBSAMPLE_SPEARMAN_MIN,
            "subsample_gate_agree_min": SUBSAMPLE_GATE_AGREE_MIN,
            "encoder_source": "cohort2_mlp16_30_30ep",
            "trained_uids_path": str(TRAINED_UIDS_PATH),
            "seed": SEED,
            "cohort_fingerprint": _extract_cache_fingerprint(),
        },
        "summary": {
            "n_total": n_total,
            "n_valid": valid_users,
            "n_invalid_union": n_fail_union,
            "valid_rate": valid_users / n_total,
            "fail_counts": fail_counts,
            "A2_coverage95_all_pct": _pct(a2_coverages, [25, 50, 75]),
            "A2_val_d2_p50_all_pct": _pct(a2_d2_p50s, [25, 50, 90]),
            "A2_val_d2_p95_all_pct": _pct(a2_d2_p95s, [50, 90, 99]),
            "A3_valid_fit_rate_all_pct": _pct(a3_fit_rates, [10, 50, 90]),
            "A3_spearman_rho_all_pct": _pct(a3_rhos, [10, 50, 90]),
            "A3_gate_agreement_all_pct": _pct(a3_agrees, [10, 50, 90]),
        },
        "per_user": per_user,
    }
    _atomic_json_dump(_sanitize_json(summary), OUT_PATH)
    log(f"  wrote → {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")
    log(f"  FINAL: valid_gaussian={valid_users}/{n_total} "
        f"({valid_users/n_total*100:.2f}%)  "
        f"fail_union={n_fail_union}  t={time.time()-t0:.1f}s")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """用户指令 2026-09-23: 串行运行 3 个 category.

    每个 category 重新绑定该脚本使用的路径常量为 category-specific 路径,
    然后调原 main_task_body() (保持原有逻辑不动). 产物写到
    result/<stage>/<baby|musical|video_games>/ 子目录.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, CACHE_DIR  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "CACHE_DIR"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        # 用户指令 2026-09-23: Stage 03a 写到 pcfg_cache_<subdir>/, 05 必须按 subdir 重绑 CACHE_DIR.
        if "CACHE_DIR" in saved:
            CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu") / f"pcfg_cache_{subdir}"
            # 用户指令 2026-09-23: EMBED_PATH 等 derived paths 在 import 时已绑定, 必须重新计算.
            globals()["EMBED_PATH"] = CACHE_DIR / "strict3_embeddings.npz"
            globals()["MANIFEST_PATH"] = CACHE_DIR / "strict3_manifest.json"
            globals()["READY_PATH"] = CACHE_DIR / "cache_ready.json"
            globals()["N_SENTS_PATH"] = CACHE_DIR / "user_n_sents.json"
            globals()["SVD_COMPONENTS"] = CACHE_DIR / "svd_components.npz"
            globals()["SENT_VECTORS"] = CACHE_DIR / "sent_vectors.npz"
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
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
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