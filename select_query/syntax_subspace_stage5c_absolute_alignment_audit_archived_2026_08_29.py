"""Stage 5C — Absolute Alignment Audit (用户指令 2026-08-28).

用户提出的关键观察:
  M = d_nearest_other - d_self 只回答 "相对问题"
  当 d_self=10, d_other=11 → M=1 > 0 (Rank@1)
  但如果真实用户句子 d_self 通常在 3-5, 那 d_self=10 已经太远

需要 audit "M>0 AND d_self ≤ R_X" 的双重条件,R_X 来自真实历史 L2 分布:
  R_50 = 5.181 (median)
  R_80 = 7.530
  R_95 = 11.308
  R_99 = 16.687

输出 4 个表:
A. Stage 5C.1: 全 7250 pairs 按 d_self quartiles 看 M>0 fraction
B. Stage 5C.2: M>0 AND d_self ≤ R_X 各阈值的 subset 大小 + retrieval outcome
C. Stage 5C.3: 严格定义 (M>0 AND d_self ≤ R_95) vs M>0 only 的 retrieval 对比
D. Stage 5C.4: per-ASIN strict-personalized ratio vs volatility

输入:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json
  R_X percentiles from real history (pre-computed, see compute_R_X in script)
输出:
  result/select_query/stage5c_absolute_alignment_audit.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "select_query"))

from common.syntax_subspace_utils import (  # noqa: E402
    _syntax_subspace_prepare, LAMBDA, PCA_DIM, PCA_SEED,
)
from select_query.syntax_subspace_repr_sweep import (  # noqa: E402
    FEATURE_SUBSETS, select_feature_names, fit_pca,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402


SEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json"
RETR_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/stage5c_absolute_alignment_audit.json"


def compute_real_L2_percentiles() -> dict:
    """Compute per-sentence whitened L2 to user-mean percentiles on real history."""
    print("  Loading _syntax_subspace_prepare...")
    P = _syntax_subspace_prepare()
    X = P["X"]; all_fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]; user_to_indices = P["user_to_indices"]

    cfg = FEATURE_SUBSETS["F3_CoreStruct"]
    fnames = select_feature_names(all_fnames, cfg["exclude_prefixes"], cfg["exclude_exact"])
    col_idx = [all_fnames.index(n) for n in fnames]

    print("  Fitting StandardScaler on F3_CoreStruct train fold...")
    ss = StandardScaler()
    ss.fit(X[train_idx][:, col_idx])
    X_sub_scaled = ss.transform(X[:, col_idx])

    print("  Fitting PCA48 + whitening...")
    pca, sqrt_lambda = fit_pca(X_sub_scaled[train_idx], PCA_DIM, PCA_SEED)
    Z_all = pca.transform(X_sub_scaled) / sqrt_lambda

    print("  Computing per-sentence whitened L2 to user-mean...")
    all_l2 = []
    for uid, idx in user_to_indices.items():
        if len(idx) < 2:
            continue
        Z_u = Z_all[idx]
        mu = Z_u.mean(axis=0)
        d = np.linalg.norm(Z_u - mu, axis=1)
        all_l2.extend(d.tolist())
    all_l2 = np.array(all_l2)
    print(f"  n_real_history_sents: {len(all_l2)}")

    pcts = {p: float(np.percentile(all_l2, p)) for p in [10, 25, 50, 75, 80, 90, 95, 97, 99]}
    return {"n": int(len(all_l2)), "percentiles": pcts, "max": float(all_l2.max())}


def main():
    log_start = time.time()
    print("=== Stage 5C: Absolute Alignment Audit ===")

    # ---- 1. Compute R_X from real history ----
    print("\n=== 1. Compute R_X from real history (whitened L2 to user-mean) ===")
    l2_stats = compute_real_L2_percentiles()
    pcts = l2_stats["percentiles"]
    R_50 = pcts[50]; R_80 = pcts[80]; R_95 = pcts[95]; R_99 = pcts[99]
    print(f"  R_50={R_50:.3f}, R_80={R_80:.3f}, R_95={R_95:.3f}, R_99={R_99:.3f}")

    # ---- 2. Load selection + retrieval per_query ----
    print("\n=== 2. Loading selection + retrieval_per_query ===")
    sel_data = json.load(open(SEL_PATH))
    sel_by_pair = {(e["asin"], e["user_id"]): e for e in sel_data["entries"]
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
            "bm25_rank": int(q["bm25_rank"]),
            "bm25_RR": float(q["bm25_RR"]),
            "bm25_hit10": int(q["bm25_hit10"]),
            "minilm_rank": int(q["minilm_rank"]),
            "minilm_RR": float(q["minilm_RR"]),
            "minilm_hit10": int(q["minilm_hit10"]),
        })
    pairs = np.array(pairs)
    print(f"  matched pairs: {len(pairs)}")

    M_arr = np.array([p["M"] for p in pairs])
    L_arr = np.array([p["self_L2"] for p in pairs])
    bm25_RR_arr = np.array([p["bm25_RR"] for p in pairs])
    minilm_RR_arr = np.array([p["minilm_RR"] for p in pairs])
    bm25_rank_arr = np.array([p["bm25_rank"] for p in pairs])
    minilm_rank_arr = np.array([p["minilm_rank"] for p in pairs])
    bm25_hit10_arr = np.array([p["bm25_hit10"] for p in pairs])
    minilm_hit10_arr = np.array([p["minilm_hit10"] for p in pairs])

    # ---- 3. d_self distribution vs R_X percentiles ----
    print("\n=== 3. d_self (selected query whitened L2 to user u mean) vs R_X ===")
    print(f"  d_self mean = {L_arr.mean():.3f}, median = {np.median(L_arr):.3f}")
    print(f"  d_self min = {L_arr.min():.3f}, max = {L_arr.max():.3f}")
    print(f"  fraction d_self ≤ R_50: {(L_arr <= R_50).mean()*100:.1f}%")
    print(f"  fraction d_self ≤ R_80: {(L_arr <= R_80).mean()*100:.1f}%")
    print(f"  fraction d_self ≤ R_95: {(L_arr <= R_95).mean()*100:.1f}%")
    print(f"  fraction d_self ≤ R_99: {(L_arr <= R_99).mean()*100:.1f}%")

    # ---- 4. Joint distribution of (M, d_self) ----
    print("\n=== 4. Joint (M, d_self) conditioning ===")
    print(f"  M>0 (relative Rank@1): {(M_arr > 0).mean()*100:.1f}%")
    print(f"  d_self ≤ R_95 (absolute alignment): {(L_arr <= R_95).mean()*100:.1f}%")
    print(f"  M>0 AND d_self ≤ R_95 (strict personalized): "
          f"{((M_arr > 0) & (L_arr <= R_95)).mean()*100:.1f}%")
    print(f"  M>0 AND d_self ≤ R_80 (very strict): "
          f"{((M_arr > 0) & (L_arr <= R_80)).mean()*100:.1f}%")

    # ---- 5. Per-threshold subset table ----
    print("\n=== 5. Subset table: M>0 + d_self ≤ R_X ===")
    thresholds = [("M>0 only", M_arr > 0, None),
                  ("M>0 + d_self ≤ R_50", (M_arr > 0) & (L_arr <= R_50), R_50),
                  ("M>0 + d_self ≤ R_80", (M_arr > 0) & (L_arr <= R_80), R_80),
                  ("M>0 + d_self ≤ R_95", (M_arr > 0) & (L_arr <= R_95), R_95),
                  ("M>0 + d_self ≤ R_99", (M_arr > 0) & (L_arr <= R_99), R_99),
                  ("M≤0 + d_self ≤ R_95", (M_arr <= 0) & (L_arr <= R_95), R_95)]
    print(f"\n  {'Subset':<28} {'N':>6} {'% of 7250':>10} "
          f"{'BM25 RR':>10} {'MiniLM RR':>10} "
          f"{'BM25 hit10%':>13} {'MiniLM hit10%':>14} "
          f"{'BM25 rank':>10} {'MiniLM rank':>12}")
    threshold_results = {}
    for label, mask, _ in thresholds:
        n = int(mask.sum())
        pct = n / len(pairs) * 100
        if n == 0:
            threshold_results[label] = {"n": 0, "pct": 0.0}
            continue
        bm25_rr = bm25_RR_arr[mask].mean()
        minilm_rr = minilm_RR_arr[mask].mean()
        bm25_hit = bm25_hit10_arr[mask].mean() * 100
        minilm_hit = minilm_hit10_arr[mask].mean() * 100
        bm25_rank = bm25_rank_arr[mask].mean()
        minilm_rank = minilm_rank_arr[mask].mean()
        threshold_results[label] = {
            "n": n,
            "pct_of_total": pct,
            "bm25_RR_mean": float(bm25_rr),
            "minilm_RR_mean": float(minilm_rr),
            "bm25_hit10_pct": float(bm25_hit),
            "minilm_hit10_pct": float(minilm_hit),
            "bm25_rank_mean": float(bm25_rank),
            "minilm_rank_mean": float(minilm_rank),
        }
        print(f"  {label:<28} {n:>6} {pct:>9.1f}% "
              f"{bm25_rr:>10.4f} {minilm_rr:>10.4f} "
              f"{bm25_hit:>12.1f}% {minilm_hit:>13.1f}% "
              f"{bm25_rank:>10.1f} {minilm_rank:>12.1f}")

    # ---- 6. Headline comparison: M>0 only vs M>0 + d_self ≤ R_95 ----
    print("\n=== 6. Headline comparison: M>0 only vs M>0 + d_self ≤ R_95 ===")
    mask_loose = M_arr > 0
    mask_strict = (M_arr > 0) & (L_arr <= R_95)
    print(f"  Loose (M>0):     N={mask_loose.sum()}")
    print(f"  Strict (M>0 + d≤R_95): N={mask_strict.sum()}")
    headline = {}
    for retriever, rr_arr, rank_arr, hit10_arr in [
        ("BM25", bm25_RR_arr, bm25_rank_arr, bm25_hit10_arr),
        ("MiniLM", minilm_RR_arr, minilm_rank_arr, minilm_hit10_arr),
    ]:
        loose_rr = rr_arr[mask_loose]
        strict_rr = rr_arr[mask_strict]
        loose_rank = rank_arr[mask_loose]
        strict_rank = rank_arr[mask_strict]
        loose_hit = hit10_arr[mask_loose]
        strict_hit = hit10_arr[mask_strict]
        # Mann-Whitney strict vs loose
        if strict_rr.std() > 0 and loose_rr.std() > 0:
            u, p_mw = stats.mannwhitneyu(strict_rr, loose_rr, alternative="two-sided")
        else:
            u, p_mw = 0, 1.0
        print(f"  {retriever}:")
        print(f"    RR mean:  Loose={loose_rr.mean():.4f}  Strict={strict_rr.mean():.4f}  "
              f"Δ={strict_rr.mean()-loose_rr.mean():+.4f}  p_mw={p_mw:.4g}")
        print(f"    rank median:  Loose={np.median(loose_rank):.1f}  Strict={np.median(strict_rank):.1f}")
        print(f"    hit@10 frac:  Loose={loose_hit.mean()*100:.1f}%  Strict={strict_hit.mean()*100:.1f}%")
        headline[retriever] = {
            "loose_RR_mean": float(loose_rr.mean()),
            "strict_RR_mean": float(strict_rr.mean()),
            "RR_delta": float(strict_rr.mean() - loose_rr.mean()),
            "mann_whitney_p": float(p_mw),
            "loose_rank_median": float(np.median(loose_rank)),
            "strict_rank_median": float(np.median(strict_rank)),
            "loose_hit10_frac": float(loose_hit.mean()),
            "strict_hit10_frac": float(strict_hit.mean()),
        }

    # ---- 7. Per-ASIN strict ratio ----
    print("\n=== 7. Per-ASIN strict-personalized ratio (M>0 AND d_self ≤ R_95) ===")
    asin_to_pairs = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        asin_to_pairs[p["asin"]].append(i)
    asin_strict_ratio = {}
    for asin, idxs in asin_to_pairs.items():
        n = len(idxs)
        n_strict = int(((M_arr[idxs] > 0) & (L_arr[idxs] <= R_95)).sum())
        asin_strict_ratio[asin] = {"n_pairs": n, "n_strict": n_strict,
                                    "ratio": n_strict / n}

    ratios = np.array([v["ratio"] for v in asin_strict_ratio.values()])
    print(f"  {len(ratios)} ASINs, mean strict ratio = {ratios.mean()*100:.1f}%")
    print(f"  ASIN strict ratio: median={np.median(ratios)*100:.1f}%, "
          f"P10={np.percentile(ratios, 10)*100:.1f}%, "
          f"P90={np.percentile(ratios, 90)*100:.1f}%, "
          f"min={ratios.min()*100:.1f}%, max={ratios.max()*100:.1f}%")

    # Per-ASIN strict ratio vs ASIN-level volatility proxy (RR Std)
    asin_RR_std = {}
    for asin, idxs in asin_to_pairs.items():
        asin_RR_std[asin] = float(np.std(bm25_RR_arr[idxs]))
    asin_minilm_RR_std = {}
    for asin, idxs in asin_to_pairs.items():
        asin_minilm_RR_std[asin] = float(np.std(minilm_RR_arr[idxs]))
    asin_hit1 = {}
    for asin, idxs in asin_to_pairs.items():
        asin_hit1[asin] = float((bm25_rank_arr[idxs] == 1).mean())
    asin_minilm_hit1 = {}
    for asin, idxs in asin_to_pairs.items():
        asin_minilm_hit1[asin] = float((minilm_rank_arr[idxs] == 1).mean())

    asins = sorted(asin_strict_ratio.keys())
    ratio_arr = np.array([asin_strict_ratio[a]["ratio"] for a in asins])
    rrstd_bm25 = np.array([asin_RR_std[a] for a in asins])
    rrstd_minilm = np.array([asin_minilm_RR_std[a] for a in asins])
    hit1_bm25 = np.array([asin_hit1[a] for a in asins])
    hit1_minilm = np.array([asin_minilm_hit1[a] for a in asins])

    rho_rrstd_bm25, p_rrstd_bm25 = stats.spearmanr(ratio_arr, rrstd_bm25)
    rho_rrstd_minilm, p_rrstd_minilm = stats.spearmanr(ratio_arr, rrstd_minilm)
    rho_hit1_bm25, p_hit1_bm25 = stats.spearmanr(ratio_arr, hit1_bm25)
    rho_hit1_minilm, p_hit1_minilm = stats.spearmanr(ratio_arr, hit1_minilm)
    print(f"\n  Spearman ρ(asin strict ratio, vol proxy):")
    print(f"    BM25 RR_Std:    {rho_rrstd_bm25:+.4f}  p={p_rrstd_bm25:.4g}")
    print(f"    MiniLM RR_Std:  {rho_rrstd_minilm:+.4f}  p={p_rrstd_minilm:.4g}")
    print(f"    BM25 Hit@1:     {rho_hit1_bm25:+.4f}  p={p_hit1_bm25:.4g}")
    print(f"    MiniLM Hit@1:   {rho_hit1_minilm:+.4f}  p={p_hit1_minilm:.4g}")

    # ---- 8. Save ----
    print("\n=== 8. Saving result ===")
    OUT = Path(OUT_PATH)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 5C Absolute Alignment Audit. Audit whether M>0 queries "
                                "are actually close to user-history style (d_self ≤ R_X) or just "
                                "relatively closer than other cohort users. R_X are whitened L2 "
                                "percentiles of real user history sentences to their user-mean."),
                "n_pairs": len(pairs),
            },
            "real_history_R_X_percentiles": l2_stats,
            "d_self_overall_stats": {
                "mean": float(L_arr.mean()),
                "median": float(np.median(L_arr)),
                "min": float(L_arr.min()),
                "max": float(L_arr.max()),
                "pct_leq_R50": float((L_arr <= R_50).mean()),
                "pct_leq_R80": float((L_arr <= R_80).mean()),
                "pct_leq_R95": float((L_arr <= R_95).mean()),
                "pct_leq_R99": float((L_arr <= R_99).mean()),
            },
            "joint_M_d_self_conditioning": {
                "M_gt_0_only_frac": float((M_arr > 0).mean()),
                "d_self_leq_R95_frac": float((L_arr <= R_95).mean()),
                "M_gt_0_AND_d_self_leq_R95_frac": float(((M_arr > 0) & (L_arr <= R_95)).mean()),
                "M_gt_0_AND_d_self_leq_R80_frac": float(((M_arr > 0) & (L_arr <= R_80)).mean()),
            },
            "subset_table_by_threshold": threshold_results,
            "headline_loose_vs_strict": headline,
            "per_asin_strict_ratio": {
                "n_asins": len(asin_strict_ratio),
                "mean_strict_ratio": float(ratios.mean()),
                "median_strict_ratio": float(np.median(ratios)),
                "P10_strict_ratio": float(np.percentile(ratios, 10)),
                "P90_strict_ratio": float(np.percentile(ratios, 90)),
                "min_strict_ratio": float(ratios.min()),
                "max_strict_ratio": float(ratios.max()),
            },
            "per_asin_spearman_strict_ratio_vs_volatility": {
                "bm25_RR_Std": {"rho": float(rho_rrstd_bm25), "p": float(p_rrstd_bm25)},
                "minilm_RR_Std": {"rho": float(rho_rrstd_minilm), "p": float(p_rrstd_minilm)},
                "bm25_Hit@1": {"rho": float(rho_hit1_bm25), "p": float(p_hit1_bm25)},
                "minilm_Hit@1": {"rho": float(rho_hit1_minilm), "p": float(p_hit1_minilm)},
            },
            "interpretation": {
                "key_finding": (
                    f"M>0 (relative Rank@1) = {(M_arr > 0).mean()*100:.1f}% of 7250 pairs. "
                    f"M>0 AND d_self ≤ R_95 (strict personalized) = "
                    f"{((M_arr > 0) & (L_arr <= R_95)).mean()*100:.1f}%. "
                    f"d_self mean={L_arr.mean():.2f} ≈ R_95={R_95:.2f}, showing most K=200 "
                    "selected queries sit at the R_95 boundary — relative Rank@1 alone "
                    "does NOT guarantee absolute alignment with user style."
                ),
                "selected_mean_self_L2": float(L_arr.mean()),
                "R_95_threshold": float(R_95),
                "selected_far_from_R_99": bool(L_arr.mean() < R_99),
            },
            "total_elapsed_s": float(time.time() - log_start),
        }, f, ensure_ascii=False, indent=2)
    print(f"  wrote → {OUT}")
    print(f"\n=== Stage 5C complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()