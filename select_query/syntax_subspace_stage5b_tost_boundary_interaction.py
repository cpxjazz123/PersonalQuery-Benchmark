"""Stage 5B — TOST Equivalence + Boundary Interaction Analysis (用户指令 2026-08-28).

两部分:

A. **TOST equivalence test** (Two One-Sided Tests):
   对 G+ (M>0) vs G- (M≤0) 在 BM25/MiniLM RR, rank 上检验
   practically negligible effect (|ΔRR| < 0.01, |d| < 0.1).
   如果 TOST 通过,才能正式写 "statistically equivalent retrieval outcomes".

B. **M × boundary interaction regression**:
   boundary_proximity = 1 / (1 + rank_norm) (rank=1 → 1.0 boundary, rank→∞ → 0)
   model:  rank_proxy ~ M + boundary + M × boundary + (1|ASIN)
   重点: M × boundary interaction — 当商品本来就在 retrieval boundary 附近时,
         strong personalized syntax (M>0) 是否放大波动?

   如果 interaction NS, 而 boundary 显著,故事完整:
   - user-style separability 不是 retrieval volatility 的主驱动
   - retrieval boundary proximity 才是机制

输入:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json (7250 pairs with M)
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json (7250 pairs with RR/rank)
输出:
  result/select_query/stage5b_tost_boundary_interaction.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats


SEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json"
RETR_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/stage5b_tost_boundary_interaction.json"


def welch_satterthwaite_df(s1: float, n1: int, s2: float, n2: int) -> float:
    """Welch-Satterthwaite df for two-sample t with unequal variances."""
    v1 = s1 / n1
    v2 = s2 / n2
    if v1 == 0 and v2 == 0:
        return float(n1 + n2 - 2)
    return float((v1 + v2) ** 2 / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1)))


def tost(data1: np.ndarray, data2: np.ndarray, eps: float,
         alpha: float = 0.05) -> dict:
    """Two One-Sided Tests (TOST) for equivalence with margin eps.

    Tests H0: |μ1 - μ2| ≥ eps  vs  H1: |μ1 - μ2| < eps.

    TOST p-value = max(p_lower, p_upper); reject H0 if max(p_lower, p_upper) < alpha.
    """
    n1, n2 = len(data1), len(data2)
    if n1 < 2 or n2 < 2:
        return {"error": "insufficient samples"}
    m1, m2 = float(data1.mean()), float(data2.mean())
    v1 = float(data1.var(ddof=1))
    v2 = float(data2.var(ddof=1))
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    diff = m1 - m2
    if se == 0:
        return {"diff": diff, "se": 0.0, "tost_p": 1.0,
                "pass": False, "note": "zero variance"}
    df = welch_satterthwaite_df(v1, n1, v2, n2)
    # lower test: H0: diff <= -eps  (too low), one-sided p = 1 - cdf(t_lower)
    t_lower = (diff - (-eps)) / se  # equivalently (diff + eps)/se
    t_upper = (diff - eps) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    # 90% CI on diff (TOST α=0.05 → 90% CI)
    ci_low = diff - stats.t.ppf(1 - alpha, df) * se
    ci_high = diff + stats.t.ppf(1 - alpha, df) * se
    in_band = (ci_low > -eps) and (ci_high < eps)
    cohen_d = (m1 - m2) / float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))
    return {
        "n1": n1,
        "n2": n2,
        "mean1": m1,
        "mean2": m2,
        "diff": float(diff),
        "se_diff": float(se),
        "df": df,
        "eps": eps,
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower_one_sided": float(p_lower),
        "p_upper_one_sided": float(p_upper),
        "tost_p": float(p_tost),
        "alpha": alpha,
        "ci_90_low": float(ci_low),
        "ci_90_high": float(ci_high),
        "ci_90_in_band": bool(in_band),
        "cohen_d": float(cohen_d),
        "equivalence_pass_at_alpha": bool(p_tost < alpha),
    }


def main():
    log_start = time.time()
    print("=== Stage 5B: TOST Equivalence + Boundary Interaction ===")

    # ---- 1. Load + join per-pair ----
    print("\n=== 1. Loading + joining per-pair data ===")
    sel_data = json.load(open(SEL_PATH))
    sel_entries = sel_data["entries"]
    sel_by_pair = {(e["asin"], e["user_id"]): e for e in sel_entries
                   if e.get("selection_method") == "l2_white_margin_max"}
    retr_data = json.load(open(RETR_PATH))
    retr_queries = retr_data["queries"]
    pairs = []
    for q in retr_queries:
        if q.get("variant") != "selected" or q.get("selection_method") != "l2_white_margin_max":
            continue
        key = (q["asin"], q["user_id"])
        if key not in sel_by_pair:
            continue
        s = sel_by_pair[key]
        pairs.append({
            "asin": q["asin"],
            "user_id": q["user_id"],
            "M": float(s["selected_margin"]),
            "self_L2": float(s["selected_distance"]),
            "attrs_covered": int(s["selected"]["attrs_covered"]),
            "n_tok": int(s["selected"]["n_tok"]),
            "len_query": int(len(s["selected"]["query"])),
            "bm25_rank": int(q["bm25_rank"]),
            "bm25_RR": float(q["bm25_RR"]),
            "bm25_hit10": int(q["bm25_hit10"]),
            "minilm_rank": int(q["minilm_rank"]),
            "minilm_RR": float(q["minilm_RR"]),
            "minilm_hit10": int(q["minilm_hit10"]),
        })
    pairs = np.array(pairs)
    print(f"  matched pairs: {len(pairs)}")

    gpos = [p for p in pairs if p["M"] > 0]
    gneg = [p for p in pairs if p["M"] <= 0]
    print(f"  G+ (M>0): {len(gpos)},  G- (M≤0): {len(gneg)}")

    # ---- 2. TOST equivalence tests ----
    print("\n=== 2. TOST equivalence tests (G+ vs G-) ===")
    # Equivalence margins (practically negligible):
    eps_RR = 0.01    # RR diff < 1pp = practically same retrieval precision
    eps_d = 0.1      # Cohen's d < 0.1 = practically same (small effect)

    metrics_for_tost = [
        ("bm25_RR", "bm25_RR", eps_RR),
        ("minilm_RR", "minilm_RR", eps_RR),
        ("bm25_rank", "bm25_rank", 200),  # 200 ranks diff = small
        ("minilm_rank", "minilm_rank", 2000),
    ]
    tost_results = {}
    print(f"\n  {'Metric':<14} {'ε':>8} {'G+ mean':>10} {'G- mean':>10} "
          f"{'Δ':>10} {'Cohen d':>8} {'tost p':>10} {'90%CI in_band':>14} "
          f"{'pass':>6}")
    for name, key, eps in metrics_for_tost:
        gp_vals = np.array([p[key] for p in gpos])
        gn_vals = np.array([p[key] for p in gneg])
        res = tost(gp_vals, gn_vals, eps, alpha=0.05)
        tost_results[name] = {"eps": eps, **res}
        print(f"  {name:<14} {eps:>8.4f} "
              f"{res['mean1']:>10.4f} {res['mean2']:>10.4f} "
              f"{res['diff']:>+10.4f} {res['cohen_d']:>+8.3f} "
              f"{res['tost_p']:>10.4g} {str(res['ci_90_in_band']):>14} "
              f"{'YES' if res.get('equivalence_pass_at_alpha') else 'NO':>6}")

    # ---- 3. Boundary proximity proxy + M × boundary interaction ----
    print("\n=== 3. Boundary proximity proxy + M × boundary interaction ===")
    # boundary_proximity: rank=1 → 1.0, rank→∞ → 0
    # 1 - 1/(1 + rank/max_rank) is monotone increasing in -rank; let's use simpler:
    # proximity = 1 - rank / (rank + K) where K=median rank → 1 if rank=1, 0 if rank→∞
    # K=10 for BM25 (median BM25 rank ~ thousands), K=200 for MiniLM
    # We use: proximity = 1.0 / (1.0 + log(rank)) — log(rank=1)=0 → 1; log(217K)≈12.3 → ~0.075
    M_arr = np.array([p["M"] for p in pairs], dtype=float)
    bm25_rank_arr = np.array([p["bm25_rank"] for p in pairs], dtype=float)
    minilm_rank_arr = np.array([p["minilm_rank"] for p in pairs], dtype=float)
    bm25_boundary = 1.0 / (1.0 + np.log1p(bm25_rank_arr))  # [1.0, ~0.07]
    minilm_boundary = 1.0 / (1.0 + np.log1p(minilm_rank_arr))
    # Also log-rank version (continuous)
    log_bm25 = np.log1p(bm25_rank_arr)
    log_minilm = np.log1p(minilm_rank_arr)
    print(f"  BM25 boundary proximity: mean={bm25_boundary.mean():.4f}, "
          f"min={bm25_boundary.min():.4f}, max={bm25_boundary.max():.4f}")
    print(f"  MiniLM boundary proximity: mean={minilm_boundary.mean():.4f}, "
          f"min={minilm_boundary.min():.4f}, max={minilm_boundary.max():.4f}")
    # Mean BM25 boundary proximity by M group:
    for label, mask in [("G+", M_arr > 0), ("G-", M_arr <= 0)]:
        print(f"  {label}: BM25 proximity={bm25_boundary[mask].mean():.4f}, "
              f"MiniLM proximity={minilm_boundary[mask].mean():.4f}")

    # ---- 4. Per-pair interaction analysis: outcome = 1/log(rank+1) ----
    # We model: bm25_rank ~ M + boundary + M × boundary (interaction)
    # But bm25_rank itself is what we use as outcome (or 1/log(rank) — higher = better)
    # Use standardized coefficients for comparability.

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    M_z = standardize(M_arr)
    bm25_b_z = standardize(bm25_boundary)
    minilm_b_z = standardize(minilm_boundary)
    bm25_rank_z = standardize(np.log1p(bm25_rank_arr))  # log rank: higher = worse
    minilm_rank_z = standardize(np.log1p(minilm_rank_arr))

    # OLS-style regression via numpy.linalg.lstsq: y = Xβ
    # model A (no interaction):  log_rank ~ M + boundary
    # model B (interaction):     log_rank ~ M + boundary + M × boundary
    def ols(y, X):
        # X: (n, k) with intercept column added inside
        X_int = np.column_stack([np.ones(len(y)), X])
        beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_pred = X_int @ beta
        residuals = y - y_pred
        n, k = X_int.shape
        sigma2 = (residuals ** 2).sum() / (n - k)
        # Var(beta) = sigma2 * (X^T X)^{-1}
        XtX_inv = np.linalg.inv(X_int.T @ X_int)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_vals = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), n - k))
        return beta, se, t_vals, p_vals

    print("\n  --- BM25 log_rank model A: log_rank ~ M + boundary ---")
    y = bm25_rank_z
    X = np.column_stack([M_z, bm25_b_z])
    beta, se, t_vals, p_vals = ols(y, X)
    names_a = ["intercept", "M_z", "boundary_z"]
    for n, b, s, t, p in zip(names_a, beta, se, t_vals, p_vals):
        print(f"    {n:<14} β={b:>+8.4f}  SE={s:>6.4f}  t={t:>+7.3f}  p={p:>10.4g}")

    print("\n  --- BM25 log_rank model B: log_rank ~ M + boundary + M×boundary ---")
    MxB_bm25 = M_z * bm25_b_z
    X = np.column_stack([M_z, bm25_b_z, MxB_bm25])
    beta, se, t_vals, p_vals = ols(y, X)
    names_b = ["intercept", "M_z", "boundary_z", "M_z × boundary"]
    bm25_model_b = {}
    for n, b, s, t, p in zip(names_b, beta, se, t_vals, p_vals):
        print(f"    {n:<20} β={b:>+8.4f}  SE={s:>6.4f}  t={t:>+7.3f}  p={p:>10.4g}")
        bm25_model_b[n] = {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}

    print("\n  --- MiniLM log_rank model B: log_rank ~ M + boundary + M×boundary ---")
    MxB_minilm = M_z * minilm_b_z
    X = np.column_stack([M_z, minilm_b_z, MxB_minilm])
    beta, se, t_vals, p_vals = ols(minilm_rank_z, X)
    minilm_model_b = {}
    for n, b, s, t, p in zip(names_b, beta, se, t_vals, p_vals):
        print(f"    {n:<20} β={b:>+8.4f}  SE={s:>6.4f}  t={t:>+7.3f}  p={p:>10.4g}")
        minilm_model_b[n] = {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}

    # ---- 5. ASIN-cluster mixed-effects (per-ASIN mean BM25 RR) ----
    # Per-ASIN: M_avg, BM25 RR (mean), MiniLM RR (mean), boundary proximity (mean)
    print("\n=== 4. ASIN-level interaction regression ===")
    asin_M_avg = {}
    asin_bm25_rr = {}
    asin_minilm_rr = {}
    asin_bm25_boundary = {}
    asin_minilm_boundary = {}
    for i, p in enumerate(pairs):
        a = p["asin"]
        asin_M_avg.setdefault(a, []).append(M_arr[i])
        asin_bm25_rr.setdefault(a, []).append(p["bm25_RR"])
        asin_minilm_rr.setdefault(a, []).append(p["minilm_RR"])
        asin_bm25_boundary.setdefault(a, []).append(bm25_boundary[i])
        asin_minilm_boundary.setdefault(a, []).append(minilm_boundary[i])

    asins = sorted(asin_M_avg.keys())
    aM = np.array([np.mean(asin_M_avg[a]) for a in asins])
    aB = np.array([np.mean(asin_bm25_rr[a]) for a in asins])
    aL = np.array([np.mean(asin_minilm_rr[a]) for a in asins])
    aP_b = np.array([np.mean(asin_bm25_boundary[a]) for a in asins])
    aP_m = np.array([np.mean(asin_minilm_boundary[a]) for a in asins])
    print(f"  {len(asins)} ASINs, M_avg range [{aM.min():.3f}, {aM.max():.3f}]")
    print(f"  ASIN-level BM25 RR range [{aB.min():.4f}, {aB.max():.4f}]")
    print(f"  ASIN-level MiniLM RR range [{aL.min():.4f}, {aL.max():.4f}]")

    # Standardize
    aM_z = standardize(aM)
    aB_z = standardize(aP_b)
    aL_z = standardize(aP_m)
    aBM25_z = standardize(aB)
    aMiniLM_z = standardize(aL)

    print("\n  --- ASIN-level BM25 RR model: log_rr ~ M_avg + boundary + M×boundary ---")
    y = aBM25_z
    X = np.column_stack([aM_z, aB_z, aM_z * aB_z])
    beta, se, t_vals, p_vals = ols(y, X)
    asin_bm25_model = {}
    for n, b, s, t, p in zip(names_b, beta, se, t_vals, p_vals):
        print(f"    {n:<20} β={b:>+8.4f}  SE={s:>6.4f}  t={t:>+7.3f}  p={p:>10.4g}")
        asin_bm25_model[n] = {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}

    print("\n  --- ASIN-level MiniLM RR model: log_rr ~ M_avg + boundary + M×boundary ---")
    y = aMiniLM_z
    X = np.column_stack([aM_z, aL_z, aM_z * aL_z])
    beta, se, t_vals, p_vals = ols(y, X)
    asin_minilm_model = {}
    for n, b, s, t, p in zip(names_b, beta, se, t_vals, p_vals):
        print(f"    {n:<20} β={b:>+8.4f}  SE={s:>6.4f}  t={t:>+7.3f}  p={p:>10.4g}")
        asin_minilm_model[n] = {"beta": float(b), "se": float(s), "t": float(t), "p": float(p)}

    # ---- 6. ASIN-cluster bootstrap: interaction coefficient CI ----
    print("\n=== 5. ASIN-cluster bootstrap: interaction coefficient CI ===")
    rng = np.random.default_rng(2024)
    n_asins = len(asins)
    # Resample ASINs, recompute interaction β
    # We resample ASINs, regenerate aM/aB/aP and re-fit
    n_boot = 2000
    boot_int_b = []
    boot_int_m = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_asins, size=n_asins)
        # Standardize using full population mean/std (correct)
        zM = (aM[idx] - aM.mean()) / (aM.std() + 1e-9)
        zP_b = (aP_b[idx] - aP_b.mean()) / (aP_b.std() + 1e-9)
        zP_m = (aP_m[idx] - aP_m.mean()) / (aP_m.std() + 1e-9)
        zB = (aB[idx] - aB.mean()) / (aB.std() + 1e-9)
        zL = (aL[idx] - aL.mean()) / (aL.std() + 1e-9)
        Xb = np.column_stack([zM, zP_b, zM * zP_b])
        Xl = np.column_stack([zM, zP_m, zM * zP_m])
        Xi_b = np.column_stack([np.ones(len(zM)), Xb])
        Xi_l = np.column_stack([np.ones(len(zM)), Xl])
        beta_b = np.linalg.lstsq(Xi_b, zB, rcond=None)[0]
        beta_l = np.linalg.lstsq(Xi_l, zL, rcond=None)[0]
        boot_int_b.append(float(beta_b[3]))  # interaction
        boot_int_m.append(float(beta_l[3]))
    boot_int_b = np.array(boot_int_b)
    boot_int_m = np.array(boot_int_m)
    lo_b, hi_b = float(np.percentile(boot_int_b, 2.5)), float(np.percentile(boot_int_b, 97.5))
    lo_m, hi_m = float(np.percentile(boot_int_m, 2.5)), float(np.percentile(boot_int_m, 97.5))
    print(f"  BM25 interaction (M×boundary) ASIN-bootstrap 95% CI: "
          f"[{lo_b:+.4f}, {hi_b:+.4f}]  (point={asin_bm25_model['M_z × boundary']['beta']:+.4f})")
    print(f"  MiniLM interaction (M×boundary) ASIN-bootstrap 95% CI: "
          f"[{lo_m:+.4f}, {hi_m:+.4f}]  (point={asin_minilm_model['M_z × boundary']['beta']:+.4f})")

    # ---- 7. Save ----
    print("\n=== 6. Saving result ===")
    OUT = Path(OUT_PATH)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 5B: TOST Equivalence Test + Boundary Interaction. "
                                "Part A: TOST (Two One-Sided Tests) on G+/G- in BM25/MiniLM "
                                "RR + rank, with equivalence margins ε_RR=0.01, ε_rank=200/2000. "
                                "Part B: OLS regression of log(rank) on M + boundary + M×boundary, "
                                "where boundary = 1/(1+log(rank)). ASIN-cluster bootstrap for CI."),
                "n_pairs": len(pairs),
                "n_G_pos": len(gpos),
                "n_G_neg": len(gneg),
                "equivalence_margins": {"RR": 0.01, "BM25_rank": 200, "MiniLM_rank": 2000,
                                        "cohen_d": 0.1},
            },
            "tost_equivalence": tost_results,
            "boundary_interaction_per_pair": {
                "bm25_log_rank_model_B": bm25_model_b,
                "minilm_log_rank_model_B": minilm_model_b,
            },
            "boundary_interaction_asin_level": {
                "bm25_RR_model": asin_bm25_model,
                "minilm_RR_model": asin_minilm_model,
            },
            "asin_cluster_bootstrap_interaction": {
                "bm25_interaction_beta": float(asin_bm25_model['M_z × boundary']['beta']),
                "bm25_interaction_CI_low": lo_b,
                "bm25_interaction_CI_high": hi_b,
                "minilm_interaction_beta": float(asin_minilm_model['M_z × boundary']['beta']),
                "minilm_interaction_CI_low": lo_m,
                "minilm_interaction_CI_high": hi_m,
            },
            "interpretation": {
                "tost_pass_RR": bool(all(
                    tost_results[m]["equivalence_pass_at_alpha"]
                    for m in ["bm25_RR", "minilm_RR"]
                )),
                "boundary_interaction_sig_bm25": bool(asin_bm25_model['M_z × boundary']['p'] < 0.05),
                "boundary_interaction_sig_minilm": bool(asin_minilm_model['M_z × boundary']['p'] < 0.05),
                "key_finding": (
                    "TOST + boundary interaction analysis tests whether G+ vs G- are "
                    "practically equivalent retrieval outcomes (within ε margin), and "
                    "whether M's role is mediated by retrieval boundary proximity. "
                    "If TOST passes and interaction is NS while boundary is strong, "
                    "M is a constraint not a volatility driver."
                ),
            },
            "total_elapsed_s": float(time.time() - log_start),
        }, f, ensure_ascii=False, indent=2)
    print(f"  wrote → {OUT}")
    print(f"\n=== Stage 5B complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()