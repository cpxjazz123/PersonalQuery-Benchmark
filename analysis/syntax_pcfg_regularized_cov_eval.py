"""L8.29 — Regularized Full-Covariance Gaussian 对比 (Raw vs Eigenvalue Floor vs Ledoit-Wolf).

L8.25 证明 rank1+residual NLL 优于 full Σ (100% finite, −25 nats/sample)。
L8.28 证明 full Σ attribution 优于 rank1 (softmax_maha_full 24.93x)。
结论:full Σ 是 canonical attribution 方法,但 5.5% 用户 near-singular (κ>1e6)。

L8.29 在不换模型前提下比较 3 种 full Σ 正则化策略:
1. Raw full Σ (current baseline, may fail Cholesky on near-singular)
2. Eigenvalue floor: λ_i' = max(λ_i, ε), ε=1e-3
3. Ledoit–Wolf: sklearn 自动估计 shrinkage α → (1−α)Σ + ατI

评估指标:
- Attribution lift (softmax-normalized maha, 同 L8.28 方法)
- Finite NLL rate (E2a)
- Bootstrap stability / E4 pass rate (B=29, frac=0.5)
- κ, λ_min
"""
from __future__ import annotations

import json
import time
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
OUT_DIR = REPO_ROOT / "result/analysis"
OUT_PATH = OUT_DIR / "syntax_pcfg_regularized_cov_eval.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 硬编码参数 (Rule 3) ===
SMOKE = False
N_SMOKE_USERS = 50
N_SMOKE_TEST = 5000
SEED = 42
Q_CHUNK = 4096
FLOOR_EPS = 1e-3
MIN_PROFILE_SENTS = 40
MIN_VAL_SENTS = 10
SUB_B = 29
SUB_FRAC = 0.5
E4_PASS_RATE = 0.90
E4_RHO_THRESHOLD = 0.9
DIAG_LAMBDA = 1e-3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_strict3():
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    npz = np.load(npz_path, allow_pickle=False)
    return {
        "z_profile": np.asarray(npz["z_profile"], dtype=np.float64),
        "z_val": np.asarray(npz["z_val"], dtype=np.float64),
        "z_test": np.asarray(npz["z_test"], dtype=np.float32),
        "val_idx": np.asarray(npz["val_idx"], dtype=np.int64),
        "test_idx": np.asarray(npz["test_idx"], dtype=np.int64),
        "profile_idx": np.asarray(npz["profile_idx"], dtype=np.int64),
        "uid_list": [str(u) for u in npz["uid_list"]],
    }


def eff_dim_relative(eigs: np.ndarray, rel_thr: float = 1e-3) -> int:
    if len(eigs) == 0:
        return 0
    lam_max = float(eigs[-1])
    if not np.isfinite(lam_max) or lam_max <= 0:
        return 0
    return int((eigs > lam_max * rel_thr).sum())


def full_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    K = cov.shape[0]
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2.0 * np.log(np.diag(L)).sum()
        diff = X - mu
        z = np.linalg.solve(L, diff.T).T
        quad = (z ** 2).sum(axis=1)
        return -0.5 * (quad + log_det + K * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        return np.full(len(X), -np.inf, dtype=np.float64)


def diag_logpdf(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    quad = (((X - mu) ** 2) / var).sum(axis=1)
    return -0.5 * (quad + X.shape[1] * np.log(2 * np.pi) + np.log(var).sum())


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def fit_3variants(z_prof: np.ndarray) -> tuple[dict, dict, dict]:
    """拟合 raw / floor / lw 三种 Σ_u. Returns (raw_stats, floor_stats, lw_stats)."""
    n_prof, K = z_prof.shape
    mu = z_prof.mean(axis=0)
    centered = z_prof - mu
    cov = (centered.T @ centered) / (n_prof - 1)
    cov = (cov + cov.T) * 0.5
    eigs, V = np.linalg.eigh(cov)

    # === Variant 1: Raw full Σ ===
    raw_stats = {"mu": mu.copy(), "cov": cov, "eigs": eigs.copy(), "V": V}
    if eigs[0] > 0:
        try:
            L = np.linalg.cholesky(cov)
            inv_S = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K)))
            raw_stats["inv_S"] = (inv_S + inv_S.T) * 0.5
            raw_stats["L"] = L
            raw_stats["finite"] = True
        except np.linalg.LinAlgError:
            raw_stats["finite"] = False
    else:
        raw_stats["finite"] = False

    # === Variant 2: Eigenvalue floor ===
    eigs_f = np.maximum(eigs, FLOOR_EPS)
    cov_floor = V @ np.diag(eigs_f) @ V.T
    cov_floor = (cov_floor + cov_floor.T) * 0.5
    try:
        L_f = np.linalg.cholesky(cov_floor)
        inv_S_f = np.linalg.solve(L_f.T, np.linalg.solve(L_f, np.eye(K)))
        floor_stats = {
            "mu": mu.copy(), "cov": cov_floor, "eigs": eigs_f,
            "inv_S": (inv_S_f + inv_S_f.T) * 0.5, "L": L_f, "finite": True,
        }
    except np.linalg.LinAlgError:
        floor_stats = {"mu": mu.copy(), "cov": cov_floor, "eigs": eigs_f, "finite": False}

    # === Variant 3: Ledoit-Wolf ===
    try:
        lw = LedoitWolf()
        lw.fit(z_prof)
        cov_lw = lw.covariance_
        cov_lw = (cov_lw + cov_lw.T) * 0.5
        L_lw = np.linalg.cholesky(cov_lw)
        inv_S_lw = np.linalg.solve(L_lw.T, np.linalg.solve(L_lw, np.eye(K)))
        eigs_lw = np.linalg.eigvalsh(cov_lw)
        lw_stats = {
            "mu": mu.copy(), "cov": cov_lw, "eigs": eigs_lw,
            "inv_S": (inv_S_lw + inv_S_lw.T) * 0.5, "L": L_lw,
            "shrinkage": float(lw.shrinkage_), "finite": True,
        }
    except Exception:
        lw_stats = {"mu": mu.copy(), "finite": False}

    return raw_stats, floor_stats, lw_stats


def compute_user_metrics(stats: dict, z_val: np.ndarray, rng: np.random.Generator) -> dict:
    """Compute NLL, beats_diag, e4 pass rate, κ, λ_min."""
    K = stats["cov"].shape[0] if "cov" in stats else stats["inv_S"].shape[0]
    mu = stats["mu"]
    cov = stats["cov"]
    eigs = stats["eigs"]

    lam_min = float(eigs[0])
    lam_max = float(eigs[-1])
    kappa = float(lam_max / lam_min) if lam_min > 0 else np.inf

    # NLL on val
    ll_full = full_logpdf(z_val, mu, cov)
    finite_nll = bool(np.all(np.isfinite(ll_full)))
    full_lp_mean = float(ll_full.mean()) if finite_nll else float("-inf")

    # Diagonal baseline
    var_d = z_val.var(axis=0, ddof=1) if len(z_val) > 1 else np.ones(K) * 0.1
    # Actually use profile var for diag
    # (we already have z_val, use simple diag mean var of profile)
    # Better: fit diag on profile separately
    beats_diag = False
    logp_margin = float("-inf")
    if finite_nll:
        prof_var = np.var(z_val, axis=0, ddof=1) + DIAG_LAMBDA  # placeholder
        # Use profile to compute diag baseline
        # We don't have profile here, use cov diagonal
        var_diag = np.diag(cov) + DIAG_LAMBDA
        ll_diag = diag_logpdf(z_val, mu, var_diag)
        diag_lp_mean = float(ll_diag.mean())
        beats_diag = full_lp_mean >= diag_lp_mean
        logp_margin = full_lp_mean - diag_lp_mean

    # E4 bootstrap
    e4_pass_rate = 0.0
    e4_rho_med = 0.0
    e4_pass = False
    n_pass_sub = 0
    rhos = []

    if finite_nll:
        # Reference D² ordering
        if "L" in stats:
            z_white = np.linalg.solve(stats["L"], (z_val - mu).T).T
        else:
            z_white = (z_val - mu) @ stats["inv_S"]
        d2_full = (z_white ** 2).sum(axis=1)
        valid_full = bool(np.all(np.isfinite(d2_full)))

        # Use profile-based bootstrap
        n_p = len(z_val)  # placeholder; we should pass profile
        # Skip bootstrap here; will be done separately with profile data
        e4_pass_rate = 1.0  # default; will recompute below
        e4_rho_med = 1.0
        e4_pass = True

    return {
        "kappa": kappa,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "eff_dim_rel": eff_dim_relative(eigs),
        "finite_nll": finite_nll,
        "nll_mean": full_lp_mean if finite_nll else None,
        "beats_diag": beats_diag,
        "logp_margin": logp_margin,
        "e4_pass_rate": e4_pass_rate,
        "e4_rho_med": e4_rho_med,
        "e4_pass": e4_pass,
    }


def compute_e4_bootstrap(z_prof: np.ndarray, z_val: np.ndarray, stats: dict,
                         rng: np.random.Generator) -> tuple[float, float, bool]:
    """E4 bootstrap: B=29 half-sampling, requires pass_rate>=0.9 AND rho_med>=0.9."""
    if "L" not in stats:
        return 0.0, 0.0, False
    mu = stats["mu"]
    cov = stats["cov"]
    n_p, K = z_prof.shape

    # Reference
    z_white = np.linalg.solve(stats["L"], (z_val - mu).T).T
    d2_ref = (z_white ** 2).sum(axis=1)
    valid_ref = bool(np.all(np.isfinite(d2_ref)))

    n_pass = 0
    rhos = []
    sub_size = max(int(SUB_FRAC * n_p), K + 1)
    sub_size = min(sub_size, n_p)

    for _ in range(SUB_B):
        idx = rng.choice(n_p, size=sub_size, replace=False)
        P_sub = z_prof[idx]
        if len(P_sub) < K + 1:
            continue
        mu_s = P_sub.mean(axis=0)
        cov_s = np.cov(P_sub.T)
        cov_s = (cov_s + cov_s.T) * 0.5
        try:
            L_s = np.linalg.cholesky(cov_s)
        except np.linalg.LinAlgError:
            continue
        ll_s = full_logpdf(z_val, mu_s, cov_s)
        if not np.all(np.isfinite(ll_s)):
            continue
        var_s = P_sub.var(axis=0, ddof=1) + DIAG_LAMBDA
        ll_d_s = diag_logpdf(z_val, mu_s, var_s)
        if ll_s.mean() < ll_d_s.mean():
            continue
        n_pass += 1
        z_white_s = np.linalg.solve(L_s, (z_val - mu_s).T).T
        d2_s = (z_white_s ** 2).sum(axis=1)
        if valid_ref and np.all(np.isfinite(d2_s)):
            rhos.append(spearman_rho(d2_ref, d2_s))

    pass_rate = n_pass / SUB_B
    rho_med = float(np.median(rhos)) if rhos else 0.0
    return pass_rate, rho_med, bool(pass_rate >= E4_PASS_RATE and rho_med >= E4_RHO_THRESHOLD and valid_ref)


def compute_maha_for_variant(z_q: np.ndarray, stats_list: list[dict],
                               device, chunk: int = 2048, user_chunk: int = 256) -> np.ndarray:
    """Compute full Maha for fitted users, return (n_q, n_users) float32."""
    n = len(z_q)
    n_users = len(stats_list)
    maha = np.zeros((n, n_users), dtype=np.float32)
    mu_dev = torch.from_numpy(np.stack([s["mu"] for s in stats_list]).astype(np.float32)).to(device)
    inv_S_dev = torch.from_numpy(np.stack([s["inv_S"] for s in stats_list]).astype(np.float32)).to(device)
    mu_inv_mu_dev = torch.einsum("ud,ude,ue->u", mu_dev.double(), inv_S_dev.double(), mu_dev.double()).float()

    for i in range(0, n, chunk):
        cz = torch.from_numpy(z_q[i:i + chunk]).to(device)
        for u_start in range(0, n_users, user_chunk):
            u_end = min(u_start + user_chunk, n_users)
            diff = cz.unsqueeze(1) - mu_dev[u_start:u_end].unsqueeze(0)
            mahal = torch.einsum("cud,ude,cue->cu", diff,
                                 inv_S_dev[u_start:u_end], diff)
            mahal = mahal + mu_inv_mu_dev[u_start:u_end].unsqueeze(0)
            maha[i:i + chunk, u_start:u_end] = (-mahal).cpu().numpy()

    del mu_dev, inv_S_dev, mu_inv_mu_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return maha


def main():
    t0 = time.time()
    log("=== L8.29: Regularized Full-Covariance Gaussian 对比 ===")
    data = load_strict3()
    z_profile_all = data["z_profile"]
    z_val_all = data["z_val"]
    z_test_all = data["z_test"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    profile_idx = data["profile_idx"]
    uid_list = data["uid_list"]
    n_users = len(uid_list)
    z_dim = z_profile_all.shape[1]

    log(f"loaded: n_users={n_users}, val={len(val_idx)} test={len(test_idx)} z_dim={z_dim}")

    # Build uid offsets
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    user_off = np.zeros(n_users + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents)

    # Identify eligible users (same as Stage 04: n_profile>=40, n_val>=10)
    prof_uid = np.clip(np.searchsorted(user_off, profile_idx, side="right") - 1, 0, n_users - 1)
    val_uid = np.clip(np.searchsorted(user_off, val_idx, side="right") - 1, 0, n_users - 1)
    prof_counts = np.bincount(prof_uid, minlength=n_users)
    val_counts = np.bincount(val_uid, minlength=n_users)
    eligible = np.where((prof_counts >= MIN_PROFILE_SENTS) & (val_counts >= MIN_VAL_SENTS))[0]

    if SMOKE:
        eligible = eligible[:N_SMOKE_USERS]
        # Limit test set too
        n_test_use = min(N_SMOKE_TEST, len(z_test_all))
        z_test_use = z_test_all[:n_test_use]
        test_idx_use = test_idx[:n_test_use]
    else:
        z_test_use = z_test_all
        test_idx_use = test_idx
        n_test_use = len(z_test_all)

    n_eligible = len(eligible)
    log(f"  eligible: {n_eligible}, test={n_test_use}" + (" (SMOKE)" if SMOKE else ""))

    # Fit 3 variants per eligible user
    log("  fitting 3 variants per user...")
    raw_stats_list = []
    floor_stats_list = []
    lw_stats_list = []
    fitted_eligible_idx = []  # track which eligible positions we actually fitted

    rng = np.random.default_rng(SEED)
    for k, uid_idx in enumerate(eligible):
        z_prof = z_profile_all[prof_uid == uid_idx]
        z_val = z_val_all[val_uid == uid_idx]
        if len(z_prof) < MIN_PROFILE_SENTS or len(z_val) < MIN_VAL_SENTS:
            continue
        r, f, l = fit_3variants(z_prof)
        raw_stats_list.append(r)
        floor_stats_list.append(f)
        lw_stats_list.append(l)
        fitted_eligible_idx.append(uid_idx)
        if (k + 1) % 50 == 0:
            log(f"    fitted {k + 1}/{n_eligible}")

    n_fitted = len(raw_stats_list)
    log(f"  fitted users: {n_fitted}")

    # Restrict to users with ALL 3 variants finite (so attribution comparison is apples-to-apples)
    all_finite_mask = np.array([
        r.get("finite", False) and f.get("finite", False) and l.get("finite", False)
        for r, f, l in zip(raw_stats_list, floor_stats_list, lw_stats_list)
    ])
    n_all_finite = int(all_finite_mask.sum())
    if n_all_finite < n_fitted:
        log(f"  restricting to {n_all_finite}/{n_fitted} users with ALL 3 variants finite")
        raw_stats_list = [s for s, m in zip(raw_stats_list, all_finite_mask) if m]
        floor_stats_list = [s for s, m in zip(floor_stats_list, all_finite_mask) if m]
        lw_stats_list = [s for s, m in zip(lw_stats_list, all_finite_mask) if m]
        fitted_eligible_idx = [u for u, m in zip(fitted_eligible_idx, all_finite_mask) if m]
        n_fitted = n_all_finite
    eligible_fitted = np.array(fitted_eligible_idx, dtype=np.int64)

    # Compute per-user metrics
    log("  computing per-user metrics (NLL, κ, λ_min, E4)...")
    metrics = {"raw": [], "floor": [], "lw": []}
    for k, (r, f, l) in enumerate(zip(raw_stats_list, floor_stats_list, lw_stats_list)):
        uid_idx = eligible[k]
        z_prof = z_profile_all[prof_uid == uid_idx]
        z_val = z_val_all[val_uid == uid_idx]
        rng_user = np.random.default_rng(SEED + k)

        for name, stats in [("raw", r), ("floor", f), ("lw", l)]:
            if not stats.get("finite", False):
                metrics[name].append({
                    "kappa": np.inf, "lam_min": 0.0, "lam_max": 0.0,
                    "eff_dim_rel": 0, "finite_nll": False, "nll_mean": None,
                    "beats_diag": False, "logp_margin": None,
                    "e4_pass_rate": 0.0, "e4_rho_med": 0.0, "e4_pass": False,
                })
                continue
            # NLL on val
            mu = stats["mu"]
            cov = stats["cov"]
            ll_full = full_logpdf(z_val, mu, cov)
            finite_nll = bool(np.all(np.isfinite(ll_full)))
            full_lp_mean = float(ll_full.mean()) if finite_nll else None
            var_diag = np.diag(cov) + DIAG_LAMBDA
            ll_diag = diag_logpdf(z_val, mu, var_diag)
            beats_diag = (full_lp_mean is not None) and (full_lp_mean >= float(ll_diag.mean()))
            logp_margin = (full_lp_mean - float(ll_diag.mean())) if finite_nll else None

            eigs = stats["eigs"]
            lam_min = float(eigs[0])
            lam_max = float(eigs[-1])
            kappa = float(lam_max / lam_min) if lam_min > 0 else np.inf

            # E4 bootstrap
            e4_pr, e4_rho, e4_pass = compute_e4_bootstrap(z_prof, z_val, stats, rng_user)

            metrics[name].append({
                "kappa": kappa, "lam_min": lam_min, "lam_max": lam_max,
                "eff_dim_rel": eff_dim_relative(eigs),
                "finite_nll": finite_nll,
                "nll_mean": full_lp_mean,
                "beats_diag": beats_diag,
                "logp_margin": logp_margin,
                "e4_pass_rate": e4_pr, "e4_rho_med": e4_rho, "e4_pass": e4_pass,
            })

    # Aggregate
    def agg(name):
        arr = metrics[name]
        kappa_arr = np.array([m["kappa"] for m in arr])
        lam_min_arr = np.array([m["lam_min"] for m in arr])
        finite_nll_pct = float(np.mean([m["finite_nll"] for m in arr])) * 100
        beats_diag_pct = float(np.mean([m["beats_diag"] for m in arr])) * 100
        e4_pass_pct = float(np.mean([m["e4_pass"] for m in arr])) * 100
        e4_pass_rate_med = float(np.median([m["e4_pass_rate"] for m in arr]))
        e4_rho_med = float(np.median([m["e4_rho_med"] for m in arr]))
        kappa_finite = kappa_arr[np.isfinite(kappa_arr)]
        if len(kappa_finite) > 0:
            kappa_med = float(np.median(kappa_finite))
            kappa_max = float(np.max(kappa_finite))
        else:
            kappa_med = kappa_max = None
        nll_margins = [m["logp_margin"] for m in arr if m["logp_margin"] is not None]
        nll_med = float(np.median(nll_margins)) if nll_margins else None
        alpha_med = None
        if name == "lw":
            alphas = [s["shrinkage"] for s in lw_stats_list if "shrinkage" in s]
            alpha_med = float(np.median(alphas)) if alphas else None
        return {
            "n_users": len(arr),
            "finite_nll_pct": finite_nll_pct,
            "beats_diag_pct": beats_diag_pct,
            "nll_logp_margin_med": nll_med,
            "kappa_med": kappa_med, "kappa_max": kappa_max,
            "lam_min_med": float(np.median(lam_min_arr[lam_min_arr > 0])) if np.any(lam_min_arr > 0) else 0.0,
            "lam_min_min": float(np.min(lam_min_arr)),
            "e4_pass_pct": e4_pass_pct,
            "e4_pass_rate_med": e4_pass_rate_med,
            "e4_rho_med": e4_rho_med,
            "alpha_med": alpha_med,
        }

    summary_per = {n: agg(n) for n in ["raw", "floor", "lw"]}

    # Attribution eval
    log("  computing attribution lifts...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"    device={device}")

    # Map test to uid
    q2u_test = np.clip(np.searchsorted(user_off, test_idx_use, side="right") - 1, 0, n_users - 1).astype(np.int32)

    # Restrict to fitted users for attribution
    fitted_set = set(int(u) for u in eligible_fitted)

    # Get cosine baseline
    log("    computing cosine baseline...")
    mu_norm = z_profile_all.mean(axis=0) if False else None  # placeholder

    # Actually compute cosine with mean of profile per user
    mu32 = np.stack([s["mu"] for s in raw_stats_list]).astype(np.float32)
    mu_norm32 = mu32 / (np.linalg.norm(mu32, axis=1, keepdims=True) + 1e-8)

    def cos_chunk(z_q):
        z_n = z_q / (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)
        return z_n @ mu_norm32.T  # (chunk, n_fitted)

    # Compute maha for each variant — store one at a time, free before next
    log("    computing maha (raw)...")
    t_m = time.time()
    maha_raw = compute_maha_for_variant(z_test_use, raw_stats_list, device)
    log(f"      ({time.time() - t_m:.0f}s)")

    # Softmax normalize — process one variant at a time to avoid GPU OOM
    log("    softmax normalize + attribution (one variant at a time)...")
    def to_gpu_softmax_chunked(m, chunk_rows=131072):
        """Row-chunked softmax: process m in row chunks to fit GPU memory."""
        n, k = m.shape
        out = np.empty_like(m)
        for i in range(0, n, chunk_rows):
            i_end = min(i + chunk_rows, n)
            t = torch.from_numpy(m[i:i_end]).to(device)
            s = torch.softmax(t, dim=1)
            out[i:i_end] = s.cpu().numpy()
            del t, s
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return out

    # Attribution argmax per query
    def run_attribution(method_name: str, maha_sm_arr: np.ndarray | None) -> dict:
        n_correct = 0
        n_q = len(z_test_use)
        for i in range(0, n_q, Q_CHUNK):
            i_end = min(i + Q_CHUNK, n_q)
            cos_c = cos_chunk(z_test_use[i:i_end])  # (chunk, n_fitted)
            q_truth = q2u_test[i:i_end]
            # Map q_truth to fitted_uidx position
            truth_pos = np.searchsorted(eligible_fitted, q_truth)
            truth_pos = np.clip(truth_pos, 0, n_fitted - 1)

            if method_name == "cosine":
                pred = cos_c.argmax(axis=1)
                truth_pred = truth_pos
            else:
                # Replace fitted columns
                cos_n = (cos_c + 1.0) / 2.0
                adapt = cos_n.copy()
                adapt[:, :] = maha_sm_arr[i:i_end]
                pred = adapt.argmax(axis=1)
                truth_pred = truth_pos

            n_correct += int((pred == truth_pred).sum())
        acc = n_correct / n_q
        return {"acc": float(acc), "lift": float(acc * n_users),
                "n_correct": int(n_correct), "n_total": int(n_q)}

    test_results = {}
    log("    test attribution:")
    # cosine first
    t_m = time.time()
    r = run_attribution("cosine", None)
    test_results["cosine"] = r
    log(f"      {'cosine':>22s}: {r['n_correct']:>5d}/{r['n_total']} "
        f"= {r['acc'] * 100:.2f}% lift {r['lift']:.2f}x ({time.time() - t_m:.0f}s)")

    # Process each variant end-to-end (maha → softmax → attribution → free)
    # raw maha already computed; floor & lw will be computed inside loop
    log("    computing maha (floor) + softmax + attribution...")
    t_m = time.time()
    maha_floor = compute_maha_for_variant(z_test_use, floor_stats_list, device)
    log(f"      ({time.time() - t_m:.0f}s)")
    t_m = time.time()
    maha_sm = to_gpu_softmax_chunked(maha_floor)
    log(f"    softmax softmax_maha_floor ({time.time() - t_m:.0f}s)")
    t_m = time.time()
    r = run_attribution("softmax_maha_floor", maha_sm)
    test_results["softmax_maha_floor"] = r
    log(f"      {'softmax_maha_floor':>22s}: {r['n_correct']:>5d}/{r['n_total']} "
        f"= {r['acc'] * 100:.2f}% lift {r['lift']:.2f}x ({time.time() - t_m:.0f}s)")
    del maha_sm, maha_floor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    log("    computing maha (raw) + softmax + attribution...")
    t_m = time.time()
    maha_sm = to_gpu_softmax_chunked(maha_raw)
    log(f"    softmax softmax_maha_raw ({time.time() - t_m:.0f}s)")
    t_m = time.time()
    r = run_attribution("softmax_maha_raw", maha_sm)
    test_results["softmax_maha_raw"] = r
    log(f"      {'softmax_maha_raw':>22s}: {r['n_correct']:>5d}/{r['n_total']} "
        f"= {r['acc'] * 100:.2f}% lift {r['lift']:.2f}x ({time.time() - t_m:.0f}s)")
    del maha_sm, maha_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    log("    computing maha (lw) + softmax + attribution...")
    t_m = time.time()
    maha_lw = compute_maha_for_variant(z_test_use, lw_stats_list, device)
    log(f"      ({time.time() - t_m:.0f}s)")
    t_m = time.time()
    maha_sm = to_gpu_softmax_chunked(maha_lw)
    log(f"    softmax softmax_maha_lw ({time.time() - t_m:.0f}s)")
    t_m = time.time()
    r = run_attribution("softmax_maha_lw", maha_sm)
    test_results["softmax_maha_lw"] = r
    log(f"      {'softmax_maha_lw':>22s}: {r['n_correct']:>5d}/{r['n_total']} "
        f"= {r['acc'] * 100:.2f}% lift {r['lift']:.2f}x ({time.time() - t_m:.0f}s)")
    del maha_sm, maha_lw
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Build summary
    chance = 1.0 / n_users
    summary = {
        "config": {
            "z_dim": z_dim, "n_users_total": n_users, "n_fitted": n_fitted,
            "n_eligible": n_eligible, "n_test": int(n_test_use),
            "split": "strict3_sha1_hash_50_20_30",
            "floor_eps": FLOOR_EPS,
            "lw_api": "sklearn.covariance.LedoitWolf",
            "e4_pass_rate_threshold": E4_PASS_RATE,
            "e4_rho_threshold": E4_RHO_THRESHOLD,
            "smoke": SMOKE,
        },
        "no_leakage": True,
        "chance": chance,
        "variants": summary_per,
        "test_results": test_results,
        "deltas": {
            "softmax_maha_floor_minus_raw": test_results["softmax_maha_floor"]["lift"]
                - test_results["softmax_maha_raw"]["lift"],
            "softmax_maha_lw_minus_raw": test_results["softmax_maha_lw"]["lift"]
                - test_results["softmax_maha_raw"]["lift"],
            "softmax_maha_floor_minus_lw": test_results["softmax_maha_floor"]["lift"]
                - test_results["softmax_maha_lw"]["lift"],
            "floor_finite_minus_raw_finite": summary_per["floor"]["finite_nll_pct"]
                - summary_per["raw"]["finite_nll_pct"],
            "lw_finite_minus_raw_finite": summary_per["lw"]["finite_nll_pct"]
                - summary_per["raw"]["finite_nll_pct"],
            "floor_e4_minus_raw_e4": summary_per["floor"]["e4_pass_pct"]
                - summary_per["raw"]["e4_pass_pct"],
            "lw_e4_minus_raw_e4": summary_per["lw"]["e4_pass_pct"]
                - summary_per["raw"]["e4_pass_pct"],
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_PATH}")
    log(f"=== Total: {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
