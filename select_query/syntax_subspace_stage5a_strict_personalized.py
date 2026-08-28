"""Stage 5A — Strict Personalized Syntax Subset Analysis (用户指令 2026-08-28).

设计: 对 7250 个 (ASIN, user) pairs 按 M>0 (strict personalized subset) vs M≤0 分组,
看两组 retrieval metrics 是否不同。

不重新生成 / 重新 selection,只在现有 cache 上 join:
  selection.json   (7250 pairs, 含 selected_margin = M)
  retrieval_per_query.json (7250 pairs, 含 BM25 RR, MiniLM RR, rank)
  volatility.json  (730 ASINs, 含 Hit@K_FlipRate + RR_Std, sim09)

输出三组分析:
A. **Per-pair G+ vs G-**: 比较 BM25 RR, MiniLM RR, attrs_covered, n_tok
B. **Continuous M analysis**: Spearman ρ(M, BM25 RR), ρ(M, MiniLM RR), quartile bins
C. **Per-ASIN M_avg G+ vs G-**: 用 M_avg 把 ASIN 分组,比较 ASIN-level Hit@1_FlipRate + RR_Std
D. **Matched-pair analysis**: 对每条 M>0 query,匹配一条 M≤0 query (ASIN, length, attrs),
   比较 matched-pair 的 BM25 RR / MiniLM RR 差异

统计:
- ASIN-cluster bootstrap 95% CI for G+ vs G- diff
- Spearman ρ with ASIN-cluster permutation (signed-rank cluster)

输入:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json
  /home/wlia0047/ar57/wenyu/PersoanlQuery/result/syntactic_evaluation/volatility.json
输出:
  result/select_query/stage5a_strict_personalized_subset.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json"
RETR_PATH = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json"
VOL_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/syntactic_evaluation/volatility.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/stage5a_strict_personalized_subset.json"


def spearman_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.std() == 0 or b.std() == 0:
        return 0.0, 1.0
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def asin_cluster_bootstrap_ci(values: np.ndarray, groups: np.ndarray,
                              n_boot: int = 2000, seed: int = 2024) -> tuple[float, float]:
    """ASIN-cluster bootstrap: resample ASINs with replacement, return 95% CI of mean."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_vals = []
        for g in sampled:
            mask = groups == g
            boot_vals.extend(values[mask].tolist())
        boot_means.append(float(np.mean(boot_vals)))
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return lo, hi


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                (len(a) + len(b) - 2))
    if s == 0:
        return 0.0
    return float((b.mean() - a.mean()) / s)


def main():
    log_start = time.time()
    print("=== Stage 5A: Strict Personalized Syntax Subset Analysis ===")

    # ---- 1. Load selection (7250 pairs with selected_margin = M) ----
    print("\n=== 1. Loading selection + retrieval_per_query ===")
    sel_data = json.load(open(SEL_PATH))
    sel_entries = sel_data["entries"]
    print(f"  selection: {len(sel_entries)} entries")

    # ---- 2. Load retrieval per_query (7250 queries) ----
    retr_data = json.load(open(RETR_PATH))
    retr_queries = retr_data["queries"]
    print(f"  retrieval_per_query: {len(retr_queries)} queries")

    # ---- 3. Build per-pair dataset ----
    print("\n=== 2. Building per-pair dataset ===")
    sel_by_pair = {(e["asin"], e["user_id"]): e for e in sel_entries
                   if e.get("selection_method") == "l2_white_margin_max"}
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
            "strict": bool(s["selected"].get("strict", False)),
        })
    pairs = np.array(pairs)
    print(f"  matched pairs: {len(pairs)}")
    print(f"  strict fraction: {sum(p['strict'] for p in pairs)/len(pairs)*100:.1f}%")

    # ---- 4. Per-pair Grouping: G+ (M>0) vs G- (M≤0) ----
    print("\n=== 3. Per-pair grouping: G+ (M>0) vs G- (M≤0) ===")
    gpos = [p for p in pairs if p["M"] > 0]
    gneg = [p for p in pairs if p["M"] <= 0]
    print(f"  G+ (M>0): {len(gpos)} ({len(gpos)/len(pairs)*100:.1f}%)")
    print(f"  G- (M≤0): {len(gneg)} ({len(gneg)/len(pairs)*100:.1f}%)")

    metrics_to_compare = ["bm25_RR", "minilm_RR", "attrs_covered", "n_tok",
                          "len_query", "bm25_rank", "minilm_rank"]
    print(f"\n  {'Metric':<18} {'G+ mean':>10} {'G- mean':>10} "
          f"{'Cohen d':>8} {'Mann-Whitney p':>16} {'ρ(M, x)':>10}")
    per_pair_results = {}
    for m in metrics_to_compare:
        gp_vals = np.array([p[m] for p in gpos])
        gn_vals = np.array([p[m] for p in gneg])
        d = cohen_d(gn_vals, gp_vals)
        if len(gp_vals) > 0 and len(gn_vals) > 0 and (gp_vals.std() > 0 or gn_vals.std() > 0):
            u, p_mw = mannwhitneyu(gp_vals, gn_vals, alternative="two-sided")
        else:
            u, p_mw = 0, 1.0
        M_arr = np.array([p["M"] for p in pairs])
        x_arr = np.array([p[m] for p in pairs])
        rho, p_sp = spearman_paired(M_arr, x_arr)
        per_pair_results[m] = {
            "G_plus_mean": float(gp_vals.mean()) if len(gp_vals) else None,
            "G_minus_mean": float(gn_vals.mean()) if len(gn_vals) else None,
            "G_plus_median": float(np.median(gp_vals)) if len(gp_vals) else None,
            "G_minus_median": float(np.median(gn_vals)) if len(gn_vals) else None,
            "cohen_d": float(d),
            "mann_whitney_p": float(p_mw),
            "spearman_rho_M": float(rho),
            "spearman_p_M": float(p_sp),
            "delta_G_plus_minus_G_minus": float(gp_vals.mean() - gn_vals.mean()),
        }
        print(f"  {m:<18} {gp_vals.mean():>10.4f} {gn_vals.mean():>10.4f} "
              f"{d:>+8.3f} {p_mw:>16.4g} {rho:>+10.4f}")

    # ---- 5. Continuous M analysis: Spearman + Quartile bins ----
    print("\n=== 4. Continuous M analysis: Spearman ρ + quartile bins ===")
    M_arr = np.array([p["M"] for p in pairs])
    bm25_rr_arr = np.array([p["bm25_RR"] for p in pairs])
    minilm_rr_arr = np.array([p["minilm_RR"] for p in pairs])
    bm25_rank_arr = np.array([p["bm25_rank"] for p in pairs])
    minilm_rank_arr = np.array([p["minilm_rank"] for p in pairs])
    bm25_hit10_arr = np.array([p["bm25_hit10"] for p in pairs])
    minilm_hit10_arr = np.array([p["minilm_hit10"] for p in pairs])

    q1, q2, q3 = np.percentile(M_arr, [25, 50, 75])
    print(f"  M quartiles: Q1={q1:.3f} Q2={q2:.3f} Q3={q3:.3f}")

    def quartile_bin(m: float) -> str:
        if m <= q1:
            return "Q1_very_low_M"
        elif m <= q2:
            return "Q2_low_M"
        elif m <= q3:
            return "Q3_high_M"
        else:
            return "Q4_very_high_M"

    bins = collections.defaultdict(list)
    for p in pairs:
        bins[quartile_bin(p["M"])].append(p)

    print(f"\n  {'Bin':<18} {'N':>5} {'BM25 RR':>10} {'MiniLM RR':>10} "
          f"{'BM25 rank':>10} {'MiniLM rank':>12}")
    quartile_summary = {}
    for b_name in ["Q1_very_low_M", "Q2_low_M", "Q3_high_M", "Q4_very_high_M"]:
        b_pairs = bins[b_name]
        if not b_pairs:
            continue
        bm25_mean = float(np.mean([p["bm25_RR"] for p in b_pairs]))
        minilm_mean = float(np.mean([p["minilm_RR"] for p in b_pairs]))
        bm25_rank_mean = float(np.mean([p["bm25_rank"] for p in b_pairs]))
        minilm_rank_mean = float(np.mean([p["minilm_rank"] for p in b_pairs]))
        bm25_hit10_frac = float(np.mean([p["bm25_hit10"] for p in b_pairs]))
        minilm_hit10_frac = float(np.mean([p["minilm_hit10"] for p in b_pairs]))
        quartile_summary[b_name] = {
            "n": len(b_pairs),
            "M_min": float(min(p["M"] for p in b_pairs)),
            "M_max": float(max(p["M"] for p in b_pairs)),
            "bm25_RR_mean": bm25_mean,
            "bm25_RR_median": float(np.median([p["bm25_RR"] for p in b_pairs])),
            "bm25_hit10_frac": bm25_hit10_frac,
            "bm25_rank_mean": bm25_rank_mean,
            "bm25_rank_median": float(np.median([p["bm25_rank"] for p in b_pairs])),
            "minilm_RR_mean": minilm_mean,
            "minilm_RR_median": float(np.median([p["minilm_RR"] for p in b_pairs])),
            "minilm_hit10_frac": minilm_hit10_frac,
            "minilm_rank_mean": minilm_rank_mean,
            "minilm_rank_median": float(np.median([p["minilm_rank"] for p in b_pairs])),
        }
        print(f"  {b_name:<18} {len(b_pairs):>5} {bm25_mean:>10.4f} {minilm_mean:>10.4f} "
              f"{bm25_rank_mean:>10.1f} {minilm_rank_mean:>12.1f}")

    # ---- 6. ASIN-cluster bootstrap CI for M>0 vs M≤0 BM25 RR diff ----
    print("\n=== 5. ASIN-cluster bootstrap (M>0 vs M≤0 BM25 RR) ===")
    asin_arr = np.array([p["asin"] for p in pairs])
    M_arr2 = np.array([p["M"] for p in pairs])
    bm25_arr = np.array([p["bm25_RR"] for p in pairs])
    minilm_arr = np.array([p["minilm_RR"] for p in pairs])

    # For each ASIN, compute mean BM25 RR within (G+) and within (G-) if both present
    asin_diff_bm25 = []
    asin_diff_minilm = []
    for asin in np.unique(asin_arr):
        mask = asin_arr == asin
        m = M_arr2[mask]
        b = bm25_arr[mask]
        ml = minilm_arr[mask]
        gp_mask = m > 0
        gn_mask = m <= 0
        if gp_mask.sum() >= 1 and gn_mask.sum() >= 1:
            asin_diff_bm25.append(float(b[gp_mask].mean() - b[gn_mask].mean()))
            asin_diff_minilm.append(float(ml[gp_mask].mean() - ml[gn_mask].mean()))

    asin_diff_bm25 = np.array(asin_diff_bm25)
    asin_diff_minilm = np.array(asin_diff_minilm)
    print(f"  ASINs with both G+ and G- pairs: {len(asin_diff_bm25)}")
    diff_bm25 = float(asin_diff_bm25.mean())
    diff_minilm = float(asin_diff_minilm.mean())

    # ASIN-cluster bootstrap: resample ASINs (units of analysis)
    rng = np.random.default_rng(2024)
    n_asins = len(asin_diff_bm25)
    boot_bm25 = []
    boot_minilm = []
    for _ in range(2000):
        idx = rng.integers(0, n_asins, size=n_asins)
        boot_bm25.append(float(asin_diff_bm25[idx].mean()))
        boot_minilm.append(float(asin_diff_minilm[idx].mean()))
    lo_b, hi_b = float(np.percentile(boot_bm25, 2.5)), float(np.percentile(boot_bm25, 97.5))
    lo_m, hi_m = float(np.percentile(boot_minilm, 2.5)), float(np.percentile(boot_minilm, 97.5))
    print(f"  BM25 RR diff (M+ - M-, per ASIN): {diff_bm25:+.4f}  CI=[{lo_b:+.4f}, {hi_b:+.4f}]")
    print(f"  MiniLM RR diff (M+ - M-, per ASIN): {diff_minilm:+.4f}  CI=[{lo_m:+.4f}, {hi_m:+.4f}]")

    # ---- 7. Per-ASIN M_avg grouping + volatility ----
    print("\n=== 6. Per-ASIN M_avg grouping + ASIN-level volatility ===")
    vol_data = json.load(open(VOL_PATH))
    sim09 = vol_data["stability_flip"]["selected_only_sim09"]
    n_asins_vol = sim09["bm25"]["n_asins"]
    print(f"  volatility.json n_asins: {n_asins_vol}")

    # Per-ASIN M_avg from all pairs
    asin_M_avg = {}
    for p in pairs:
        a = p["asin"]
        if a not in asin_M_avg:
            asin_M_avg[a] = []
        asin_M_avg[a].append(p["M"])
    asin_M_avg = {a: float(np.mean(v)) for a, v in asin_M_avg.items()}

    # If volatility.json has per-ASIN metrics, load them
    # volatility.json is summary only — need per-ASIN raw for grouping
    # Compute Hit@1_FlipRate per ASIN from sim09 pairs
    print("  Computing per-ASIN flip rate from retrieval_per_query...")
    # sim09 filter requires re-encoding selected queries, which we don't have in per_query cache
    # Use a simpler proxy: per-ASIN BM25 RR std (proxy for RR_Std) and per-ASIN Hit@1 fraction
    asin_to_rrs = collections.defaultdict(list)
    asin_to_ranks = collections.defaultdict(list)
    asin_to_hit1 = collections.defaultdict(list)
    for p in pairs:
        asin_to_rrs[p["asin"]].append(p["bm25_RR"])
        asin_to_ranks[p["asin"]].append(p["bm25_rank"])
        asin_to_hit1[p["asin"]].append(1 if p["bm25_rank"] == 1 else 0)

    # Per-ASIN BM25 RR Std (proxy for volatility)
    asin_RR_std = {a: float(np.std(rrs)) for a, rrs in asin_to_rrs.items()}
    asin_rank_std = {a: float(np.std(rs)) for a, rs in asin_to_ranks.items()}
    asin_hit1_frac = {a: float(np.mean(h)) for a, h in asin_to_hit1.items()}

    asin_minilm_rrs = collections.defaultdict(list)
    asin_minilm_rank_std = {}
    asin_minilm_hit1_frac = {}
    for p in pairs:
        asin_minilm_rrs[p["asin"]].append(p["minilm_RR"])
    for a, rrs in asin_minilm_rrs.items():
        asin_minilm_rank_std[a] = float(np.std(rrs))
        asin_minilm_hit1_frac[a] = float(np.mean([1 if r == 1 else 0 for r in rrs]))

    # Group ASINs by M_avg > 0 vs ≤ 0
    asin_group = {a: ("G+" if m > 0 else "G-") for a, m in asin_M_avg.items()}
    pos_asins = [a for a, g in asin_group.items() if g == "G+"]
    neg_asins = [a for a, g in asin_group.items() if g == "G-"]
    print(f"  G+ ASINs (M_avg > 0): {len(pos_asins)} ({len(pos_asins)/len(asin_group)*100:.1f}%)")
    print(f"  G- ASINs (M_avg ≤ 0): {len(neg_asins)} ({len(neg_asins)/len(asin_group)*100:.1f}%)")

    def group_summary(asins: list, d: dict, label: str) -> dict:
        vals = np.array([d[a] for a in asins])
        return {"n_asins": len(asins), f"{label}_mean": float(vals.mean()),
                f"{label}_median": float(np.median(vals)),
                f"{label}_std": float(vals.std())}

    pos_rrstd_bm25 = np.array([asin_RR_std[a] for a in pos_asins])
    neg_rrstd_bm25 = np.array([asin_RR_std[a] for a in neg_asins])
    pos_rrstd_minilm = np.array([asin_minilm_rank_std[a] for a in pos_asins])
    neg_rrstd_minilm = np.array([asin_minilm_rank_std[a] for a in neg_asins])
    pos_hit1_bm25 = np.array([asin_hit1_frac[a] for a in pos_asins])
    neg_hit1_bm25 = np.array([asin_hit1_frac[a] for a in neg_asins])
    pos_hit1_minilm = np.array([asin_minilm_hit1_frac[a] for a in pos_asins])
    neg_hit1_minilm = np.array([asin_minilm_hit1_frac[a] for a in neg_asins])

    print(f"\n  {'ASIN-level proxy':<22} {'G+':>14} {'G-':>14} {'Diff':>10} {'p (MW)':>10}")
    proxy_results = {}
    proxy_metrics = [
        ("bm25_RR_Std", pos_rrstd_bm25, neg_rrstd_bm25),
        ("bm25_rank_Std", np.array([asin_rank_std[a] for a in pos_asins]),
                          np.array([asin_rank_std[a] for a in neg_asins])),
        ("bm25_Hit@1_frac", pos_hit1_bm25, neg_hit1_bm25),
        ("minilm_RR_Std", pos_rrstd_minilm, neg_rrstd_minilm),
        ("minilm_Hit@1_frac", pos_hit1_minilm, neg_hit1_minilm),
    ]
    for label, gp_vals, gn_vals in proxy_metrics:
        d = cohen_d(gn_vals, gp_vals)
        u, p_mw = mannwhitneyu(gp_vals, gn_vals, alternative="two-sided")
        diff = float(gp_vals.mean() - gn_vals.mean())
        print(f"  {label:<22} {gp_vals.mean():>14.4f} {gn_vals.mean():>14.4f} "
              f"{diff:>+10.4f} {p_mw:>10.4g}")
        proxy_results[label] = {
            "G_plus_mean": float(gp_vals.mean()),
            "G_minus_mean": float(gn_vals.mean()),
            "diff": diff,
            "cohen_d": float(d),
            "mann_whitney_p": float(p_mw),
        }

    # ---- 8. Matched-pair analysis (M>0 vs M≤0 within same ASIN) ----
    print("\n=== 7. Matched-pair analysis (M>0 vs M≤0 within same ASIN) ===")
    asin_to_gpos = collections.defaultdict(list)
    asin_to_gnegs = collections.defaultdict(list)
    for p in pairs:
        if p["M"] > 0:
            asin_to_gpos[p["asin"]].append(p)
        else:
            asin_to_gnegs[p["asin"]].append(p)

    n_matched = 0
    bm25_diff_matched = []
    minilm_diff_matched = []
    M_diff_matched = []
    matched_pairs_log = []
    for asin in np.unique(asin_arr):
        pos_list = asin_to_gpos.get(asin, [])
        neg_list = asin_to_gnegs.get(asin, [])
        if not pos_list or not neg_list:
            continue
        # Greedy: for each G+ query, find G- query with closest attrs_covered + len_query
        used_neg = set()
        for gp in pos_list:
            best_neg = None
            best_dist = float("inf")
            for i, gn in enumerate(neg_list):
                if i in used_neg:
                    continue
                dist = (abs(gp["attrs_covered"] - gn["attrs_covered"]) +
                        abs(gp["len_query"] - gn["len_query"]) / 50.0)
                if dist < best_dist:
                    best_dist = dist
                    best_neg = (i, gn)
            if best_neg is not None:
                used_neg.add(best_neg[0])
                gn = best_neg[1]
                bm25_diff_matched.append(gp["bm25_RR"] - gn["bm25_RR"])
                minilm_diff_matched.append(gp["minilm_RR"] - gn["minilm_RR"])
                M_diff_matched.append(gp["M"] - gn["M"])
                matched_pairs_log.append({
                    "asin": asin,
                    "M_pos": gp["M"], "M_neg": gn["M"],
                    "attrs_pos": gp["attrs_covered"], "attrs_neg": gn["attrs_covered"],
                    "len_pos": gp["len_query"], "len_neg": gn["len_query"],
                    "bm25_RR_pos": gp["bm25_RR"], "bm25_RR_neg": gn["bm25_RR"],
                    "minilm_RR_pos": gp["minilm_RR"], "minilm_RR_neg": gn["minilm_RR"],
                })
                n_matched += 1

    bm25_diff_arr = np.array(bm25_diff_matched)
    minilm_diff_arr = np.array(minilm_diff_matched)
    M_diff_arr = np.array(M_diff_matched)
    print(f"  matched pairs: {n_matched}")
    print(f"  BM25 RR diff (M+ - M-): mean={bm25_diff_arr.mean():+.4f}, "
          f"median={np.median(bm25_diff_arr):+.4f}")
    print(f"  MiniLM RR diff: mean={minilm_diff_arr.mean():+.4f}, "
          f"median={np.median(minilm_diff_arr):+.4f}")
    if len(bm25_diff_arr) > 0 and bm25_diff_arr.std() > 0:
        u_bm25, p_bm25 = mannwhitneyu(bm25_diff_arr, np.zeros_like(bm25_diff_arr),
                                       alternative="two-sided")
    else:
        u_bm25, p_bm25 = 0, 1.0
    if len(minilm_diff_arr) > 0 and minilm_diff_arr.std() > 0:
        u_minilm, p_minilm = mannwhitneyu(minilm_diff_arr, np.zeros_like(minilm_diff_arr),
                                           alternative="two-sided")
    else:
        u_minilm, p_minilm = 0, 1.0
    print(f"  Wilcoxon (BM25 RR diff vs 0): p={p_bm25:.4g}")
    print(f"  Wilcoxon (MiniLM RR diff vs 0): p={p_minilm:.4g}")

    matched_analysis = {
        "n_matched_pairs": n_matched,
        "bm25_RR_diff_mean": float(bm25_diff_arr.mean()),
        "bm25_RR_diff_median": float(np.median(bm25_diff_arr)),
        "bm25_RR_diff_std": float(bm25_diff_arr.std()),
        "minilm_RR_diff_mean": float(minilm_diff_arr.mean()),
        "minilm_RR_diff_median": float(np.median(minilm_diff_arr)),
        "minilm_RR_diff_std": float(minilm_diff_arr.std()),
        "M_diff_mean": float(M_diff_arr.mean()),
        "wilcoxon_p_bm25_RR_diff_vs_0": float(p_bm25),
        "wilcoxon_p_minilm_RR_diff_vs_0": float(p_minilm),
    }

    # ---- 9. Per-ASIN continuous M_avg correlation with ASIN-level volatility ----
    print("\n=== 8. Per-ASIN M_avg vs ASIN-level volatility (proxy) ===")
    asins = sorted(asin_M_avg.keys())
    asin_M_avg_arr = np.array([asin_M_avg[a] for a in asins])
    asin_bm25_rrstd_arr = np.array([asin_RR_std[a] for a in asins])
    asin_bm25_hit1_arr = np.array([asin_hit1_frac[a] for a in asins])
    asin_minilm_rrstd_arr = np.array([asin_minilm_rank_std[a] for a in asins])
    asin_minilm_hit1_arr = np.array([asin_minilm_hit1_frac[a] for a in asins])

    rho_bm25_rrstd, p_bm25_rrstd = spearman_paired(asin_M_avg_arr, asin_bm25_rrstd_arr)
    rho_bm25_hit1, p_bm25_hit1 = spearman_paired(asin_M_avg_arr, asin_bm25_hit1_arr)
    rho_minilm_rrstd, p_minilm_rrstd = spearman_paired(asin_M_avg_arr, asin_minilm_rrstd_arr)
    rho_minilm_hit1, p_minilm_hit1 = spearman_paired(asin_M_avg_arr, asin_minilm_hit1_arr)

    print(f"  ρ(M_avg, BM25 RR_Std): {rho_bm25_rrstd:+.4f}  p={p_bm25_rrstd:.4g}")
    print(f"  ρ(M_avg, BM25 Hit@1): {rho_bm25_hit1:+.4f}  p={p_bm25_hit1:.4g}")
    print(f"  ρ(M_avg, MiniLM Rank_Std): {rho_minilm_rrstd:+.4f}  p={p_minilm_rrstd:.4g}")
    print(f"  ρ(M_avg, MiniLM Hit@1): {rho_minilm_hit1:+.4f}  p={p_minilm_hit1:.4g}")

    asin_corr = {
        "bm25_RR_Std": {"rho_M_avg": float(rho_bm25_rrstd), "p": float(p_bm25_rrstd)},
        "bm25_Hit@1": {"rho_M_avg": float(rho_bm25_hit1), "p": float(p_bm25_hit1)},
        "minilm_RR_Std": {"rho_M_avg": float(rho_minilm_rrstd), "p": float(p_minilm_rrstd)},
        "minilm_Hit@1": {"rho_M_avg": float(rho_minilm_hit1), "p": float(p_minilm_hit1)},
    }

    # ---- 10. Save ----
    print("\n=== 9. Saving result ===")
    OUT = Path(OUT_PATH)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 5A Strict Personalized Syntax Subset Analysis. "
                                "7250 (ASIN, user) pairs split by M = d_nearest_other - d_self. "
                                "G+ (M>0) = strict personalized subset (~63.7%); G- (M≤0) = control. "
                                "Compare BM25/MiniLM RR, ranks, attrs_covered, length. "
                                "Continuous: Spearman ρ + quartile bins. "
                                "ASIN-cluster bootstrap for group difference CI. "
                                "Per-ASIN M_avg vs ASIN-level volatility proxy. "
                                "Matched-pair analysis: M>0 vs M≤0 within same ASIN, "
                                "matched on attrs_covered + len_query."),
                "n_pairs": len(pairs),
                "n_G_pos": len(gpos),
                "n_G_neg": len(gneg),
            },
            "per_pair_grouping": per_pair_results,
            "continuous_quartile_analysis": quartile_summary,
            "asin_cluster_bootstrap_diff": {
                "bm25_RR_G_plus_minus_G_minus": diff_bm25,
                "bm25_RR_CI_low": lo_b,
                "bm25_RR_CI_high": hi_b,
                "minilm_RR_G_plus_minus_G_minus": diff_minilm,
                "minilm_RR_CI_low": lo_m,
                "minilm_RR_CI_high": hi_m,
                "n_asins_with_both_groups": int(len(asin_diff_bm25)),
            },
            "per_asin_M_avg_grouping": proxy_results,
            "matched_pair_analysis": matched_analysis,
            "per_asin_M_avg_correlation": asin_corr,
            "total_elapsed_s": float(time.time() - log_start),
        }, f, ensure_ascii=False, indent=2)
    print(f"  wrote → {OUT}")
    print(f"\n=== Stage 5A complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()