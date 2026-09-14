#!/usr/bin/env python3
"""不加 λI, 看用户原始 Σ_u 能否建立真正有效的 32 维高斯.

有效高斯定义 (3 项 AND, E3 降为软约束报告):
  E2: 留出预测能力 — 同一用户 ≥5 条 val 句子上:
       E2a: full log-P 全部为有限值 (无 -inf / NaN)
       E2b: full log-P mean ≥ 对角高斯 log-P mean (full cov 必须胜过
             diagonal baseline, 否则 off-diagonal 信息无泛化能力)
  E3: 有效维度 (软约束, 仅报告) — 三种**尺度无关**相对定义:
       - eff_dim_rel = #{λ_i > λ_max · 1e-3}
       - effective_rank = exp(-Σ p_i log p_i),  p_i = λ_i/Σλ
       - participation_ratio = (Σλ)² / Σλ²
       原绝对阈值 eigval > 1e-6 在不同 z 尺度下结论不一致 (×100 → ×1e4 eigval),
       不能直接比较 cohort; 相对阈值在 z 整体放大或缩小时保持稳定。
       **不**再阻断 valid_gaussian, 仅作诊断。
  E4: 重采样稳定性 — 50% bootstrap B=29 次重拟合:
       ≥90% 重采样仍同时通过 (PD + val log-P 有限 + full≥diag)
       且 val 句子距离排序与 full-fit 相关 ρ ≥ 0.9 (spearman)

(E1 样本量 ≥2K=64 门槛已移除:full cov 在 n_p ≥ K+1=33 时即可估计,样本
越小越能反映 strict encoder 的真实 per-user 拟合质量.)

E4 专门检查 "当前结果是不是偶然的", 筛掉 cond~4e4、对样本扰动敏感的
用户. 阈值预注册, 不允许后调.

附加报告:
  - 单独看每项失败的用户数
  - E3 eff_dim 分布 (软约束, 不阻断)
  - "不能建立高斯"的并集 (E2 ∪ E4)
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

EMBED_PATH = CACHE_DIR / "strict3_embeddings.npz"
MANIFEST_PATH = CACHE_DIR / "strict3_manifest.json"
READY_PATH = CACHE_DIR / "cache_ready.json"
N_SENTS_PATH = CACHE_DIR / "user_n_sents.json"

OUT_DIR = REPO_ROOT / "result/05_gaussian_audit"
OUT_PATH = OUT_DIR / "raw_cov_validity.json"
ASIN_USERS_PATH = (
    REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"
)
ASIN_COVERAGE_PATH = OUT_DIR / "asin_coverage_valid_ge2.json"

K_DIM = 16
N_USERS_EXPECTED = None  # 不再硬编码: 由 len(uid_list) 动态确定
SEED = 42
MIN_VAL_SENTS = 5          # E2a: 至少 5 条 val 句
DIAG_LAMBDA = 1e-3         # 对角基线 shrinkage (避免单维 var=0)
SUB_B = 29                 # E4: bootstrap 重采样次数
SUB_FRAC = 0.5             # E4: 每次采样比例 (50% half-sampling)
E4_PASS_RATE = 0.90        # E4: 至少 90% 重采样通过 (PD + finite + beats_diag + full_rank)
E4_RHO_THRESHOLD = 0.9     # E4: val 距离排序 spearman ρ ≥ 0.9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_json_dump(payload, path) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _extract_cache_fingerprint() -> str:
    """从 cache_ready.json 读取 sentence_source_fingerprint 以在 audit 结果上记录。"""
    if not READY_PATH.exists():
        return ""
    with open(READY_PATH) as f:
        return str(json.load(f).get("sentence_source_fingerprint", ""))


def _sanitize_json(value):
    """替换 NaN/Inf 为 None,确保 JSON 严格合法。"""
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
    required = ("z_profile", "z_val", "z_test",
                "profile_idx", "val_idx", "test_idx", "uid_list",
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
    for key in ("profile_idx", "val_idx", "test_idx"):
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
         ("profile_idx", "val_idx", "test_idx")])
    if len(np.unique(merged)) != n_total or not np.array_equal(
            np.sort(merged), np.arange(n_total, dtype=np.int64)):
        raise ValueError("strict3 split indices do not partition all sentences")
    for key in ("z_profile", "z_val", "z_test"):
        z = np.asarray(npz[key])
        if z.ndim != 2 or z.shape[1] not in (16, 32):
            raise ValueError(f"strict3 {key} shape invalid: {z.shape}")
        if not np.isfinite(z).all():
            raise ValueError(f"strict3 {key} contains NaN/Inf")

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


def load_assets():
    for path in (EMBED_PATH, N_SENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"missing: {path} (run 03_spacy_encode first)")
    d = np.load(EMBED_PATH, allow_pickle=True)
    n_sents = json.load(open(N_SENTS_PATH))
    n_sents = [int(n) for n in n_sents]
    uid_list = [str(u) for u in d["uid_list"]]
    n_users_actual = len(uid_list)
    if len(n_sents) != n_users_actual:
        raise ValueError(
            f"user_n_sents ({len(n_sents)}) != uid_list ({n_users_actual})")
    if any(n < 0 for n in n_sents):
        raise ValueError("user_n_sents contains a negative count")
    _validate_strict3_artifact(d, n_sents)

    z_p = d["z_profile"]
    z_v = d["z_val"]
    pid = d["profile_idx"]
    vid = d["val_idx"]
    row2uid = np.repeat(np.arange(n_users_actual), n_sents)
    prof_uid = row2uid[pid]
    val_uid = row2uid[vid]
    op = np.argsort(prof_uid, kind="stable")
    ov = np.argsort(val_uid, kind="stable")
    z_p_sorted = z_p[op]
    z_v_sorted = z_v[ov]
    p_uid_sorted = prof_uid[op]
    v_uid_sorted = val_uid[ov]
    cp = np.bincount(p_uid_sorted, minlength=n_users_actual)
    cv = np.bincount(v_uid_sorted, minlength=n_users_actual)
    sp = np.concatenate([[0], np.cumsum(cp)])
    sv = np.concatenate([[0], np.cumsum(cv)])
    return z_p_sorted, z_v_sorted, sp, sv, uid_list, n_users_actual


# ---------------------------------------------------------------------------
# log-pdf 工具
# ---------------------------------------------------------------------------

_K_CACHE: dict = {}


def _k_log2pi(K: int) -> float:
    if K not in _K_CACHE:
        _K_CACHE[K] = K * float(np.log(2 * np.pi))
    return _K_CACHE[K]


# ---------------------------------------------------------------------------
# E3 有效维数 (尺度无关相对定义)
# ---------------------------------------------------------------------------

E3_REL_THR = 1e-3  # λ_i / λ_max > 1e-3 计为有效


def eff_dim_relative(eigs: np.ndarray, rel_thr: float = E3_REL_THR) -> int:
    """相对阈值有效维数: #{λ_i > λ_max · rel_thr}。尺度无关。"""
    if len(eigs) == 0:
        return 0
    lam_max = float(eigs[-1])
    if not np.isfinite(lam_max) or lam_max <= 0:
        return 0
    return int((eigs > lam_max * rel_thr).sum())


def effective_rank(eigs: np.ndarray) -> float:
    """Effective rank = exp(H(p))，H 为 Shannon entropy，p_i = λ_i/Σλ。
    衡量 variance 在多少个方向上"实质均匀"分布。"""
    pos = eigs[eigs > 0]
    if len(pos) == 0:
        return 0.0
    p = pos / pos.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def participation_ratio(eigs: np.ndarray) -> float:
    """Participation ratio d_PR = (Σλ)² / Σλ²。1 ≤ d_PR ≤ K，等同性指示。"""
    pos = eigs[eigs > 0]
    if len(pos) == 0:
        return 0.0
    s1 = float(pos.sum())
    s2 = float((pos ** 2).sum())
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


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
    e1_sample = True  # E1 已移除 (n_p ≥ 33 即可估计 full cov)

    # n_p < K+1 直接跳过 cov 估计
    if n_p < K + 1:
        return {
            "n_p": n_p, "n_v": n_v,
            "e1_sample": e1_sample,
            "e2a_finite": False, "e2b_beats_diag": False, "e2_predict": False,
            "e3_full_rank": False, "e3_eff_dim": 0,
            "e3_eff_dim_rel": 0, "e3_effective_rank": 0.0,
            "e3_participation_ratio": 0.0, "e3_rel_threshold": E3_REL_THR,
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
    n_eff_rel = eff_dim_relative(eigs, E3_REL_THR)
    eff_rank = effective_rank(eigs)
    d_pr = participation_ratio(eigs)
    e3_full_rank = bool(n_eff_rel == K)  # 保留字段名供兼容, 实际指相对阈值全秩

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

    # E4 重采样稳定性 — 仅当 E2 通过时计算 (成本高, 提前 fail 直接跳过)
    e4_pass = False
    e4_pass_rate = 0.0
    e4_rho_med = 0.0
    e4_n_pass = 0
    if e2_predict:
        e4 = bootstrap_stability(P, V, K, rng)
        e4_pass = e4["e4_pass"]
        e4_pass_rate = e4["e4_pass_rate"]
        e4_rho_med = e4["e4_rho_med"]
        e4_n_pass = int(round(e4_pass_rate * SUB_B))

    fail_reasons = []
    if not e2a_finite:
        fail_reasons.append(
            f"E2a val<{MIN_VAL_SENTS} or full log-P 非有限"
            + (f" (n_v={n_v})" if n_v < MIN_VAL_SENTS else ""))
    if not e2b_beats_diag:
        fail_reasons.append(
            f"E2b full log-P ({full_lp_mean:.3f}) < diag log-P ({diag_lp_mean:.3f})"
            if e2a_finite else "E2b 不可计算 (E2a fail)")
    if e2_predict and not e4_pass:
        fail_reasons.append(
            f"E4 pass_rate={e4_pass_rate:.2f} < {E4_PASS_RATE} or "
            f"rho_med={e4_rho_med:.3f} < {E4_RHO_THRESHOLD}")

    valid = e2_predict and e4_pass
    return {
        "n_p": n_p, "n_v": n_v,
        "e1_sample": e1_sample,
        "e2a_finite": e2a_finite,
        "e2b_beats_diag": e2b_beats_diag,
        "e2_predict": e2_predict,
        "e3_full_rank": e3_full_rank,
        "e3_eff_dim": n_eff_rel,
        "e3_eff_dim_rel": n_eff_rel,
        "e3_effective_rank": eff_rank,
        "e3_participation_ratio": d_pr,
        "e3_rel_threshold": E3_REL_THR,
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

    z_p_s, z_v_s, sp, sv, uid_list, n_users_actual = load_assets()
    log(f"  K={K_DIM}  N_USERS={n_users_actual}  "
        f"MIN_VAL_SENTS={MIN_VAL_SENTS}  SUB_B={SUB_B}  SUB_FRAC={SUB_FRAC}  "
        f"E4_THR=pass>={E4_PASS_RATE}, rho>={E4_RHO_THRESHOLD}")
    log(f"  loaded z_profile={z_p_s.shape}  z_val={z_v_s.shape}")

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

    for u in range(n_users_actual):
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
        if (u + 1) % 500 == 0:
            log(f"  {u + 1}/{n_users_actual}  valid={valid_users}  "
                f"fail_union={len(fail_union)}  t={time.time()-t0:.1f}s")

    n_total = n_users_actual
    n_fail_union = len(fail_union)
    log(f"  FINAL:")
    log(f"    valid_gaussian     = {valid_users}/{n_total} ({valid_users/n_total*100:.2f}%)")
    log(f"    invalid (union)    = {n_fail_union}/{n_total} ({n_fail_union/n_total*100:.2f}%)")
    log(f"    fail by E2a finite = {fail_counts['E2a_finite']}")
    log(f"    fail by E2b diag   = {fail_counts['E2b_beats_diag']}")
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
    eff_dim_pct = (np.percentile(eff_dim_per_user, [10, 50, 90]).tolist()
                   if eff_dim_per_user else None)
    log(f"    E3 eff_dim_rel (λ/λ_max>{E3_REL_THR:g}) P10/50/90 = "
        f"{eff_dim_pct}")
    log(f"    E3 eff_dim_rel hist (全体) = "
        f"{sorted(eff_dim_hist.items())[:20]}...")
    log(f"    E3 effective_rank (全体) P10/50/90 = "
        f"{np.percentile(eff_rank_list,[10,50,90]).tolist()}")
    log(f"    E3 participation_ratio (全体) P10/50/90 = "
        f"{np.percentile(d_pr_list,[10,50,90]).tolist()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "K_dim": K_DIM, "n_users": n_users_actual,
            "criteria": (f"E2a: val>={MIN_VAL_SENTS} 且 full log-P 有限; "
                         "E2b: full log-P mean >= diag log-P mean; "
                         f"E3: 尺度无关软约束 (eff_dim_rel: λ/λ_max>{E3_REL_THR:g}, "
                         "effective_rank, participation_ratio, 仅报告不阻断); "
                         f"E4: bootstrap B={SUB_B} frac={SUB_FRAC} "
                         f"pass_rate>={E4_PASS_RATE} AND rho_med>={E4_RHO_THRESHOLD}"),
            "ridge": "NONE (raw np.cov, no λI for full; diag baseline +1e-3)",
            "min_val_sents": MIN_VAL_SENTS,
            "diag_lambda": DIAG_LAMBDA,
            "sub_B": SUB_B, "sub_frac": SUB_FRAC,
            "e4_pass_rate_threshold": E4_PASS_RATE,
            "e4_rho_threshold": E4_RHO_THRESHOLD,
            "seed": SEED,
            "cache_schema_version": 2,
            "cohort_fingerprint": _extract_cache_fingerprint(),
        },
        "summary": {
            "n_total": n_total,
            "n_valid": valid_users,
            "n_invalid_union": n_fail_union,
            "invalid_rate": n_fail_union / n_total,
            "fail_by_E2a_finite": fail_counts["E2a_finite"],
            "fail_by_E2b_beats_diag": fail_counts["E2b_beats_diag"],
            "fail_by_E4_stability": fail_counts["E4_stability"],
            "valid_min_eigval_pct": np.percentile(min_eigs,[10,50,90]).tolist() if min_eigs else None,
            "valid_cond_ratio_pct": np.percentile(conds,[50,90,99]).tolist() if conds else None,
            "valid_logp_margin_pct": np.percentile(margins,[10,50,90]).tolist() if margins else None,
            "valid_e4_pass_rate_pct": np.percentile(e4_pass_rates,[10,50,90]).tolist() if e4_pass_rates else None,
            "valid_e4_rho_pct": np.percentile(e4_rhos,[10,50,90]).tolist() if e4_rhos else None,
            "e3_eff_dim_rel_hist_all": {str(k): v for k, v in sorted(eff_dim_hist.items())},
            "e3_eff_dim_rel_all_pct": eff_dim_pct,
            "e3_effective_rank_all_pct": np.percentile(eff_rank_list,[10,50,90]).tolist(),
            "e3_participation_ratio_all_pct": np.percentile(d_pr_list,[10,50,90]).tolist(),
            "e3_rel_threshold": E3_REL_THR,
        },
        "per_user": per_user,
    }
    out = _sanitize_json(out)
    _atomic_json_dump(out, OUT_PATH)
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
            "criteria": "valid_gaussian == True (E2 + E4 pass; E3 软约束不阻断; E1 已移除)",
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
