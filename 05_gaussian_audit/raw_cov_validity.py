#!/usr/bin/env python3
"""不加 λI, 看用户原始 Σ_u 能否建立真正有效的 32 维高斯.

有效高斯定义 (4 项 AND):
  E1: 样本量 n_p ≥ 2*K = 64 (否则 full cov 估计本身不稳)
  E2: 留出预测能力 — 同一用户 ≥5 条 val 句子上:
       E2a: full log-P 全部为有限值 (无 -inf / NaN)
       E2b: full log-P mean ≥ 对角高斯 log-P mean (full cov 必须胜过
             diagonal baseline, 否则 off-diagonal 信息无泛化能力)
  E3: 有效维度 = K = 32 (eigval > 1e-6 全部 32 维)
  E4: 重采样稳定性 — 50% bootstrap B=29 次重拟合:
       ≥90% 重采样仍同时通过 (PD + val log-P 有限 + full≥diag + eff_dim=32)
       且 val 句子距离排序与 full-fit 相关 ρ ≥ 0.9 (spearman)

E4 专门检查 "当前结果是不是偶然的", 筛掉 cond~4e4、对样本扰动敏感的
用户. 阈值预注册, 不允许后调.

附加报告:
  - 单独看每项失败的用户数
  - "不能建立 32 维高斯"的并集 (E1 ∪ E2 ∪ E3 ∪ E4)
  - 有效维度分布直方图
  - full-vs-diag log-P margin 分布
  - E4 重采样通过率 + val ρ 分布

输出: result/05_gaussian_audit/raw_cov_validity.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # noqa: E402  (avoid local import in hot loop)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")

EMBED_PATH = CACHE_DIR / "adaptive_embeddings.npz"
N_SENTS_PATH = CACHE_DIR / "user_n_sents.json"

OUT_DIR = REPO_ROOT / "result/05_gaussian_audit"
OUT_PATH = OUT_DIR / "raw_cov_validity.json"
ASIN_USERS_PATH = REPO_ROOT / "asin_users/asin_to_users.json"
ASIN_COVERAGE_PATH = OUT_DIR / "asin_coverage_valid_ge2.json"

K_DIM = 32
N_USERS_EXPECTED = 5000
SEED = 42
MIN_VAL_SENTS = 5          # E2a: 至少 5 条 val 句
DIAG_LAMBDA = 1e-3         # 对角基线 shrinkage (避免单维 var=0)
SUB_B = 29                 # E4: bootstrap 重采样次数
SUB_FRAC = 0.5             # E4: 每次采样比例 (50% half-sampling)
E4_PASS_RATE = 0.90        # E4: 至少 90% 重采样通过 (PD + finite + beats_diag + full_rank)
E4_RHO_THRESHOLD = 0.9     # E4: val 距离排序 spearman ρ ≥ 0.9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_assets():
    d = np.load(EMBED_PATH, allow_pickle=True)
    z_p = d["z_profile"]
    z_v = d["z_val"]
    pid = d["profile_idx"]
    vid = d["val_idx"]
    uid_list = list(d["uid_list"])
    n_sents = json.load(open(N_SENTS_PATH))
    assert len(n_sents) == N_USERS_EXPECTED == len(uid_list)
    row2uid = np.repeat(np.arange(N_USERS_EXPECTED), n_sents)
    prof_uid = row2uid[pid]
    val_uid = row2uid[vid]
    op = np.argsort(prof_uid, kind="stable")
    ov = np.argsort(val_uid, kind="stable")
    z_p_sorted = z_p[op]
    z_v_sorted = z_v[ov]
    p_uid_sorted = prof_uid[op]
    v_uid_sorted = val_uid[ov]
    cp = np.bincount(p_uid_sorted, minlength=N_USERS_EXPECTED)
    cv = np.bincount(v_uid_sorted, minlength=N_USERS_EXPECTED)
    sp = np.concatenate([[0], np.cumsum(cp)])
    sv = np.concatenate([[0], np.cumsum(cv)])
    return z_p_sorted, z_v_sorted, sp, sv, uid_list


# ---------------------------------------------------------------------------
# log-pdf 工具
# ---------------------------------------------------------------------------

_K_CACHE: dict = {}


def _k_log2pi(K: int) -> float:
    if K not in _K_CACHE:
        _K_CACHE[K] = K * float(np.log(2 * np.pi))
    return _K_CACHE[K]


def full_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multivariate Gaussian log-pdf per row. cov + 0·I (no ridge).

    矩阵奇异 → 返回 -inf (调用方负责判 E2a).
    """
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2.0 * np.log(np.diag(L)).sum()
        diff = X - mu
        # solve L y = diff  →  y = L^{-1} diff  →  ||y||² row-wise
        z = np.linalg.solve(L, diff.T).T  # (n, K)
        quad = (z ** 2).sum(axis=1)
        return -0.5 * (quad + log_det + _k_log2pi(cov.shape[0]))
    except np.linalg.LinAlgError:
        return np.full(len(X), -np.inf, dtype=np.float64)


def diag_logpdf(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Diagonal Gaussian log-pdf per row."""
    K = X.shape[1]
    quad = (((X - mu) ** 2) / var).sum(axis=1)
    return -0.5 * (quad + K * np.log(2 * np.pi) + np.log(var).sum())


# ---------------------------------------------------------------------------
# 单用户评估
# ---------------------------------------------------------------------------

def maha_d2_order(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Mahalanobis D² per row. cov 奇异返回 +inf (排名破坏)."""
    try:
        L = np.linalg.cholesky(cov)
        diff = X - mu
        z = np.linalg.solve(L, diff.T).T
        return (z ** 2).sum(axis=1)
    except np.linalg.LinAlgError:
        return np.full(len(X), np.inf, dtype=np.float64)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ between two 1D arrays."""
    try:
        rho, _ = spearmanr(a, b)
        return float(rho) if np.isfinite(rho) else 0.0
    except Exception:
        return 0.0


def bootstrap_stability(P: np.ndarray, V: np.ndarray, K: int,
                        rng: np.random.Generator) -> dict:
    """50% bootstrap B=29 次重拟合 + val 距离排序 ρ."""
    n_p = len(P)
    # full-fit reference D² 排序
    mu_full = P.mean(axis=0)
    cov_full = np.cov(P.T)
    d2_full = maha_d2_order(V, mu_full, cov_full)
    valid_full = bool(np.all(np.isfinite(d2_full)))

    # var diag baseline for full-fit
    var_full = P.var(axis=0, ddof=1) + DIAG_LAMBDA
    mu_d_full = P.mean(axis=0)
    ll_diag_full = diag_logpdf(V, mu_d_full, var_full)
    ll_full = full_logpdf(V, mu_full, cov_full)
    full_lp_full = float(ll_full.mean())
    diag_lp_full = float(ll_diag_full.mean())

    n_pass_sub = 0
    rhos = []
    for _ in range(SUB_B):
        idx = rng.choice(n_p,
                         size=min(n_p, max(int(SUB_FRAC * n_p), K + 1)),
                         replace=False)
        P_sub = P[idx]
        if len(P_sub) < K + 1:
            continue
        mu_s = P_sub.mean(axis=0)
        cov_s = np.cov(P_sub.T)
        # PD
        eigs_s = np.linalg.eigvalsh(cov_s)
        if eigs_s[0] <= 0:
            continue
        # full rank
        if int((eigs_s > 1e-6).sum()) < K:
            continue
        # finite val log-P
        ll_s = full_logpdf(V, mu_s, cov_s)
        if not np.all(np.isfinite(ll_s)):
            continue
        # beats diag
        var_s = P_sub.var(axis=0, ddof=1) + DIAG_LAMBDA
        ll_d_s = diag_logpdf(V, mu_s, var_s)
        if ll_s.mean() < ll_d_s.mean():
            continue
        n_pass_sub += 1
        # ρ 排序 (与 full D²)
        d2_s = maha_d2_order(V, mu_s, cov_s)
        if valid_full and np.all(np.isfinite(d2_s)):
            rho = spearman_rho(d2_full, d2_s)
            rhos.append(rho)

    pass_rate = n_pass_sub / SUB_B
    rho_med = float(np.median(rhos)) if rhos else 0.0
    return {
        "e4_pass_rate": pass_rate,
        "e4_rho_med": rho_med,
        "e4_n_rho": len(rhos),
        "e4_pass": bool(pass_rate >= E4_PASS_RATE and rho_med >= E4_RHO_THRESHOLD
                        and valid_full),
        "e4_full_lp_mean": full_lp_full,
        "e4_diag_lp_mean": diag_lp_full,
    }


def evaluate_one(P: np.ndarray, V: np.ndarray, K: int,
                 rng: np.random.Generator) -> dict:
    n_p = len(P)
    n_v = len(V)
    e1_sample = bool(n_p >= 2 * K)

    # n_p < K+1 直接跳过 cov 估计
    if n_p < K + 1:
        return {
            "n_p": n_p, "n_v": n_v,
            "e1_sample": e1_sample,
            "e2a_finite": False, "e2b_beats_diag": False, "e2_predict": False,
            "e3_full_rank": False, "e3_eff_dim": 0,
            "e4_pass": False, "e4_pass_rate": 0.0,
            "e4_rho_med": 0.0, "e4_n_pass": 0,
            "min_eigval": float("nan"),
            "max_eigval": float("nan"),
            "cond_ratio": float("inf"),
            "full_logp_mean": float("nan"),
            "diag_logp_mean": float("nan"),
            "logp_margin": float("nan"),
            "valid_gaussian": False,
            "fail_reasons": ["n_p<33 无法估计 full cov"],
        }

    mu = P.mean(axis=0)
    cov = np.cov(P.T)
    eigs = np.linalg.eigvalsh(cov)
    min_e = float(eigs[0])
    max_e = float(eigs[-1])
    n_eff = int((eigs > 1e-6).sum())
    e3_full_rank = bool(n_eff == K)

    # E2 留出预测能力
    e2a_finite = False
    e2b_beats_diag = False
    full_lp_mean = float("nan")
    diag_lp_mean = float("nan")
    logp_margin = float("nan")

    if n_v >= MIN_VAL_SENTS:
        # full Gaussian (无 ridge, 奇异返回 -inf)
        ll_full = full_logpdf(V, mu, cov)
        e2a_finite = bool(np.all(np.isfinite(ll_full)))
        if e2a_finite:
            full_lp_mean = float(ll_full.mean())
            # 对角基线 (per-dim var + DIAG_LAMBDA)
            var_d = P.var(axis=0, ddof=1) + DIAG_LAMBDA
            mu_d = P.mean(axis=0)
            ll_diag = diag_logpdf(V, mu_d, var_d)
            diag_lp_mean = float(ll_diag.mean())
            e2b_beats_diag = bool(full_lp_mean >= diag_lp_mean)
            logp_margin = full_lp_mean - diag_lp_mean
    e2_predict = e2a_finite and e2b_beats_diag

    # E4 重采样稳定性 — 仅当 E1E2E3 都通过时计算 (成本高, 提前 fail 直接跳过)
    e4_pass = False
    e4_pass_rate = 0.0
    e4_rho_med = 0.0
    e4_n_pass = 0
    if e1_sample and e2_predict and e3_full_rank:
        e4 = bootstrap_stability(P, V, K, rng)
        e4_pass = e4["e4_pass"]
        e4_pass_rate = e4["e4_pass_rate"]
        e4_rho_med = e4["e4_rho_med"]
        e4_n_pass = int(round(e4_pass_rate * SUB_B))

    fail_reasons = []
    if not e1_sample:
        fail_reasons.append(f"E1 n_p={n_p} < 2K=64")
    if not e2a_finite:
        fail_reasons.append(
            f"E2a val<{MIN_VAL_SENTS} or full log-P 非有限"
            + (f" (n_v={n_v})" if n_v < MIN_VAL_SENTS else ""))
    if not e2b_beats_diag:
        fail_reasons.append(
            f"E2b full log-P ({full_lp_mean:.3f}) < diag log-P ({diag_lp_mean:.3f})"
            if e2a_finite else "E2b 不可计算 (E2a fail)")
    if not e3_full_rank:
        fail_reasons.append(f"E3 eff_dim={n_eff} < {K}")
    if e1_sample and e2_predict and e3_full_rank and not e4_pass:
        fail_reasons.append(
            f"E4 pass_rate={e4_pass_rate:.2f} < {E4_PASS_RATE} or "
            f"rho_med={e4_rho_med:.3f} < {E4_RHO_THRESHOLD}")

    valid = e1_sample and e2_predict and e3_full_rank and e4_pass
    return {
        "n_p": n_p, "n_v": n_v,
        "e1_sample": e1_sample,
        "e2a_finite": e2a_finite,
        "e2b_beats_diag": e2b_beats_diag,
        "e2_predict": e2_predict,
        "e3_full_rank": e3_full_rank,
        "e3_eff_dim": n_eff,
        "e4_pass": e4_pass,
        "e4_pass_rate": e4_pass_rate,
        "e4_rho_med": e4_rho_med,
        "e4_n_pass": e4_n_pass,
        "min_eigval": min_e,
        "max_eigval": max_e,
        "cond_ratio": float(max_e / max(min_e, 1e-12)),
        "full_logp_mean": full_lp_mean,
        "diag_logp_mean": diag_lp_mean,
        "logp_margin": logp_margin,
        "valid_gaussian": valid,
        "fail_reasons": fail_reasons,
    }


def main():
    t0 = time.time()
    log("=== raw_cov_validity (no λI ridge, val predictive E2 + bootstrap E4) ===")
    log(f"  K={K_DIM}  N_USERS={N_USERS_EXPECTED}  "
        f"MIN_VAL_SENTS={MIN_VAL_SENTS}  SUB_B={SUB_B}  SUB_FRAC={SUB_FRAC}  "
        f"E4_THR=pass>={E4_PASS_RATE}, rho>={E4_RHO_THRESHOLD}")

    z_p_s, z_v_s, sp, sv, uid_list = load_assets()
    log(f"  loaded z_profile={z_p_s.shape}  z_val={z_v_s.shape}")

    per_user: dict = {}
    fail_counts = {"E1_sample": 0, "E2a_finite": 0, "E2b_beats_diag": 0,
                   "E3_full_rank": 0, "E4_stability": 0}
    fail_union: set = set()
    eff_dim_hist: dict = {}
    valid_users = 0
    min_eigs: list = []
    conds: list = []
    margins: list = []
    e4_pass_rates: list = []
    e4_rhos: list = []
    rng = np.random.default_rng(SEED)

    for u in range(N_USERS_EXPECTED):
        P = z_p_s[sp[u]:sp[u + 1]]
        V = z_v_s[sv[u]:sv[u + 1]]
        r = evaluate_one(P, V, K_DIM, rng)
        per_user[uid_list[u]] = r
        if not r["valid_gaussian"]:
            fail_union.add(u)
            if not r["e1_sample"]:
                fail_counts["E1_sample"] += 1
            if not r["e2a_finite"]:
                fail_counts["E2a_finite"] += 1
            if not r["e2b_beats_diag"]:
                fail_counts["E2b_beats_diag"] += 1
            if not r["e3_full_rank"]:
                fail_counts["E3_full_rank"] += 1
            if (r["e1_sample"] and r["e2_predict"] and r["e3_full_rank"]
                    and not r["e4_pass"]):
                fail_counts["E4_stability"] += 1
            eff_dim_hist[r["e3_eff_dim"]] = eff_dim_hist.get(r["e3_eff_dim"], 0) + 1
        else:
            valid_users += 1
            min_eigs.append(r["min_eigval"])
            conds.append(r["cond_ratio"])
            margins.append(r["logp_margin"])
            e4_pass_rates.append(r["e4_pass_rate"])
            e4_rhos.append(r["e4_rho_med"])
        if (u + 1) % 500 == 0:
            log(f"  {u + 1}/{N_USERS_EXPECTED}  valid={valid_users}  "
                f"fail_union={len(fail_union)}  t={time.time()-t0:.1f}s")

    n_total = N_USERS_EXPECTED
    n_fail_union = len(fail_union)
    log(f"  FINAL:")
    log(f"    valid_gaussian     = {valid_users}/{n_total} ({valid_users/n_total*100:.2f}%)")
    log(f"    invalid (union)    = {n_fail_union}/{n_total} ({n_fail_union/n_total*100:.2f}%)")
    log(f"    fail by E1 sample  = {fail_counts['E1_sample']}")
    log(f"    fail by E2a finite = {fail_counts['E2a_finite']}")
    log(f"    fail by E2b diag   = {fail_counts['E2b_beats_diag']}")
    log(f"    fail by E3 rank    = {fail_counts['E3_full_rank']}")
    log(f"    fail by E4 stability = {fail_counts['E4_stability']}")

    if min_eigs:
        log(f"    valid min_eigval P10/50/90 = "
            f"{np.percentile(min_eigs,[10,50,90]).tolist()}")
        log(f"    valid cond_ratio P50/90/99 = "
            f"{np.percentile(conds,[50,90,99]).tolist()}")
        log(f"    valid logp_margin P10/50/90 = "
            f"{np.percentile(margins,[10,50,90]).tolist()}")
        log(f"    valid E4 pass_rate P10/50/90 = "
            f"{np.percentile(e4_pass_rates,[10,50,90]).tolist()}")
        log(f"    valid E4 rho_med P10/50/90 = "
            f"{np.percentile(e4_rhos,[10,50,90]).tolist()}")
    log(f"    eff_dim hist (invalid users) = "
        f"{sorted(eff_dim_hist.items())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "K_dim": K_DIM, "n_users": N_USERS_EXPECTED,
            "criteria": ("E1: n_p>=64; "
                         f"E2a: val>={MIN_VAL_SENTS} 且 full log-P 有限; "
                         "E2b: full log-P mean >= diag log-P mean; "
                         "E3: eff_dim==32; "
                         f"E4: bootstrap B={SUB_B} frac={SUB_FRAC} "
                         f"pass_rate>={E4_PASS_RATE} AND rho_med>={E4_RHO_THRESHOLD}"),
            "ridge": "NONE (raw np.cov, no λI for full; diag baseline +1e-3)",
            "min_val_sents": MIN_VAL_SENTS,
            "diag_lambda": DIAG_LAMBDA,
            "sub_B": SUB_B, "sub_frac": SUB_FRAC,
            "e4_pass_rate_threshold": E4_PASS_RATE,
            "e4_rho_threshold": E4_RHO_THRESHOLD,
            "seed": SEED,
        },
        "summary": {
            "n_total": n_total,
            "n_valid": valid_users,
            "n_invalid_union": n_fail_union,
            "invalid_rate": n_fail_union / n_total,
            "fail_by_E1_sample": fail_counts["E1_sample"],
            "fail_by_E2a_finite": fail_counts["E2a_finite"],
            "fail_by_E2b_beats_diag": fail_counts["E2b_beats_diag"],
            "fail_by_E3_full_rank": fail_counts["E3_full_rank"],
            "fail_by_E4_stability": fail_counts["E4_stability"],
            "valid_min_eigval_pct": np.percentile(min_eigs,[10,50,90]).tolist() if min_eigs else None,
            "valid_cond_ratio_pct": np.percentile(conds,[50,90,99]).tolist() if conds else None,
            "valid_logp_margin_pct": np.percentile(margins,[10,50,90]).tolist() if margins else None,
            "valid_e4_pass_rate_pct": np.percentile(e4_pass_rates,[10,50,90]).tolist() if e4_pass_rates else None,
            "valid_e4_rho_pct": np.percentile(e4_rhos,[10,50,90]).tolist() if e4_rhos else None,
            "eff_dim_hist_invalid": {str(k): v for k, v in sorted(eff_dim_hist.items())},
        },
        "per_user": per_user,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"  saved -> {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB)")

    # ---- ASIN 覆盖分析: valid uid 覆盖的 ASIN (≥2 valid users) ----
    log(f"=== ASIN coverage (valid uids → ASIN ≥2 users) ===")
    valid_uids = set(uid for uid, r in per_user.items() if r["valid_gaussian"])
    with open(ASIN_USERS_PATH) as f:
        asin_to_users = json.load(f)
    asin_to_valid = {a: [u for u in uids if u in valid_uids]
                     for a, uids in asin_to_users.items()}
    asin_to_valid = {a: v for a, v in asin_to_valid.items() if len(v) >= 2}
    n_valid_users_used = len({u for v in asin_to_valid.values() for u in v})
    log(f"  valid uids total        = {len(valid_uids)}")
    log(f"  ASINs with ≥2 valid uids = {len(asin_to_valid)}")
    log(f"  valid uids involved      = {n_valid_users_used}")
    n_per = [len(v) for v in asin_to_valid.values()]
    if n_per:
        log(f"  users/ASIN P10/50/90/99  = "
            f"{np.percentile(n_per,[10,50,90,99]).astype(int).tolist()}")
        log(f"  users/ASIN max           = {max(n_per)}")
    coverage_out = {
        "config": {
            "criteria": "valid_gaussian == True (E1E2E3E4 all pass)",
            "min_users_per_asin": 2,
            "n_valid_uids_total": len(valid_uids),
            "n_valid_uids_used_in_coverage": n_valid_users_used,
            "n_asins_meeting_threshold": len(asin_to_valid),
        },
        "asin_to_valid_uids": asin_to_valid,
    }
    with open(ASIN_COVERAGE_PATH, "w") as f:
        json.dump(coverage_out, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {ASIN_COVERAGE_PATH} "
        f"({os.path.getsize(ASIN_COVERAGE_PATH)//1024} KB)")

    log(f"=== DONE in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
