"""Stage 9D-Volatility — expression-induced volatility analysis on Stage 9D selected queries.

For each ASIN's 10 user-specific selected queries:
1. Hit@1 Flip Rate (pairwise, within ASIN): fraction of (i,j) pairs whose hit@1 status flips
2. Hit@5 Flip Rate: same for top-5
3. RR Std: std of Reciprocal Rank across the 10 users

Compute for both BM25 and MiniLM.

Then per ASIN paired bootstrap / Wilcoxon comparing BM25 vs MiniLM on each metric.

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_volatility.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_volatility_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9d_volatility.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9d_volatility.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
PER_Q_IN = SCRATCH / "stage9d_retrieval_per_query.json"
PER_ASIN_OUT = SCRATCH / "stage9d_volatility.json"
SUMMARY_OUT = SCRATCH / "stage9d_volatility_summary.json"

SEED = 2024
N_BOOT = 10000


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compute_flip_rate(hits: list[int]) -> float:
    """Pairwise flip rate: for all pairs (i,j), fraction where status_i != status_j."""
    pairs = list(combinations(range(len(hits)), 2))
    if not pairs:
        return 0.0
    flips = sum(1 for i, j in pairs if hits[i] != hits[j])
    return flips / len(pairs)


def compute_rr_std(rrs: list[float]) -> float:
    """Std of RR across the N user queries."""
    if len(rrs) < 2:
        return 0.0
    return float(np.std(rrs, ddof=1))


def bootstrap_paired_diff(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = SEED):
    """Bootstrap CI for mean(a - b). Returns (mean_diff, ci_low, ci_high, p_ge0)."""
    rng = np.random.RandomState(seed)
    a = np.array(a)
    b = np.array(b)
    diff = a - b
    mean_diff = float(diff.mean())
    n = len(diff)
    if n < 2:
        return mean_diff, mean_diff, mean_diff, 1.0
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diffs.append(diff[idx].mean())
    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    p_ge0 = float((boot_diffs >= 0).mean())
    return mean_diff, ci_low, ci_high, p_ge0


def main():
    log("=== Stage 9D-Volatility: expression-induced volatility on selected queries ===")

    # === 1. Load per-query retrieval ===
    log("\n=== 1. Loading per-query retrieval ===")
    records = json.load(open(PER_Q_IN))
    log(f"  total records: {len(records)}")

    # Filter to selected only (personalized per-user queries)
    sel_records = [r for r in records if r["variant"] == "selected"]
    log(f"  selected records: {len(sel_records)}")

    # === 2. Group by ASIN ===
    log("\n=== 2. Grouping by ASIN ===")
    asin_to_records = collections.defaultdict(list)
    for r in sel_records:
        asin_to_records[r["asin"]].append(r)
    asins = sorted(asin_to_records.keys())
    log(f"  ASINs: {len(asins)}")
    for a in asins:
        log(f"    {a}: {len(asin_to_records[a])} users")

    # === 3. Compute per-ASIN volatility metrics ===
    log("\n=== 3. Computing per-ASIN volatility ===")
    per_asin = {}
    for asin in asins:
        recs = asin_to_records[asin]
        n = len(recs)
        bm25_ranks = [r["bm25_rank"] for r in recs]
        bm25_rrs = [r["bm25_RR"] for r in recs]
        bm25_hit1 = [1 if (r["bm25_rank"] is not None and r["bm25_rank"] == 1) else 0 for r in recs]
        bm25_hit5 = [1 if (r["bm25_rank"] is not None and r["bm25_rank"] <= 5) else 0 for r in recs]

        minilm_ranks = [r["minilm_rank"] for r in recs]
        minilm_rrs = [r["minilm_RR"] for r in recs]
        minilm_hit1 = [1 if (r["minilm_rank"] is not None and r["minilm_rank"] == 1) else 0 for r in recs]
        minilm_hit5 = [1 if (r["minilm_rank"] is not None and r["minilm_rank"] <= 5) else 0 for r in recs]

        n_pairs = n * (n - 1) // 2

        per_asin[asin] = {
            "n_users": n,
            "n_pairs": n_pairs,
            # BM25
            "bm25_hit1_flip_rate": compute_flip_rate(bm25_hit1),
            "bm25_hit5_flip_rate": compute_flip_rate(bm25_hit5),
            "bm25_rr_std": compute_rr_std(bm25_rrs),
            "bm25_hit1_count": sum(bm25_hit1),
            "bm25_hit5_count": sum(bm25_hit5),
            "bm25_rr_mean": float(np.mean(bm25_rrs)),
            "bm25_rank_median": float(np.median([r for r in bm25_ranks if r is not None])),
            # MiniLM
            "minilm_hit1_flip_rate": compute_flip_rate(minilm_hit1),
            "minilm_hit5_flip_rate": compute_flip_rate(minilm_hit5),
            "minilm_rr_std": compute_rr_std(minilm_rrs),
            "minilm_hit1_count": sum(minilm_hit1),
            "minilm_hit5_count": sum(minilm_hit5),
            "minilm_rr_mean": float(np.mean(minilm_rrs)),
            "minilm_rank_median": float(np.median([r for r in minilm_ranks if r is not None])),
        }
        log(f"  {asin}: bm25_hit1_flip={per_asin[asin]['bm25_hit1_flip_rate']*100:.1f}%, "
            f"bm25_hit5_flip={per_asin[asin]['bm25_hit5_flip_rate']*100:.1f}%, "
            f"bm25_rr_std={per_asin[asin]['bm25_rr_std']*100:.3f}%, "
            f"minilm_hit1_flip={per_asin[asin]['minilm_hit1_flip_rate']*100:.1f}%, "
            f"minilm_hit5_flip={per_asin[asin]['minilm_hit5_flip_rate']*100:.1f}%, "
            f"minilm_rr_std={per_asin[asin]['minilm_rr_std']*100:.3f}%")

    with open(PER_ASIN_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9D-Volatility: expression-induced volatility on selected per-user queries",
                "n_asins": len(asins),
                "n_users_per_asin": [per_asin[a]["n_users"] for a in asins],
                "SEED": SEED,
                "N_BOOT": N_BOOT,
            },
            "per_asin": per_asin,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_ASIN_OUT}")

    # === 4. Aggregate + BM25 vs MiniLM comparison ===
    log("\n=== 4. BM25 vs MiniLM ASIN-level paired comparison ===")

    summary = {}
    for metric in ["hit1_flip_rate", "hit5_flip_rate", "rr_std"]:
        bm25_vals = np.array([per_asin[a][f"bm25_{metric}"] for a in asins])
        minilm_vals = np.array([per_asin[a][f"minilm_{metric}"] for a in asins])

        log(f"\n  Metric: {metric}")
        log(f"    BM25:   mean={bm25_vals.mean()*100:.3f}%, median={np.median(bm25_vals)*100:.3f}%")
        log(f"    MiniLM: mean={minilm_vals.mean()*100:.3f}%, median={np.median(minilm_vals)*100:.3f}%")

        # Wilcoxon paired (BM25 vs MiniLM)
        diff = bm25_vals - minilm_vals
        if np.all(diff == 0):
            w_p = 1.0
            w_stat = 0.0
        else:
            try:
                w = wilcoxon(bm25_vals, minilm_vals, alternative="two-sided")
                w_stat = float(w.statistic)
                w_p = float(w.pvalue)
            except ValueError as e:
                log(f"    Wilcoxon failed: {e}")
                w_stat = 0.0
                w_p = 1.0
        log(f"    Wilcoxon BM25 vs MiniLM: stat={w_stat:.1f}, p={w_p:.4f}")

        # Bootstrap CI on mean(B - M)
        m_diff, ci_low, ci_high, p_ge0 = bootstrap_paired_diff(bm25_vals.tolist(), minilm_vals.tolist())
        log(f"    Bootstrap mean(B-M)={m_diff*100:.3f}%, CI=[{ci_low*100:.3f}%, {ci_high*100:.3f}%], p(B>M)={p_ge0:.3f}")

        summary[metric] = {
            "bm25_mean": float(bm25_vals.mean()),
            "bm25_median": float(np.median(bm25_vals)),
            "bm25_std": float(bm25_vals.std(ddof=1)),
            "minilm_mean": float(minilm_vals.mean()),
            "minilm_median": float(np.median(minilm_vals)),
            "minilm_std": float(minilm_vals.std(ddof=1)),
            "mean_diff_bm25_minus_minilm": m_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "p_bm25_gt_minilm_bootstrap": p_ge0,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
        }

    # === 5. Per-ASIN pairwise winner breakdown ===
    log("\n=== 5. Per-ASIN pairwise winner breakdown ===")
    for metric in ["hit1_flip_rate", "hit5_flip_rate", "rr_std"]:
        bm25_better = sum(1 for a in asins if per_asin[a][f"bm25_{metric}"] < per_asin[a][f"minilm_{metric}"])
        equal = sum(1 for a in asins if per_asin[a][f"bm25_{metric}"] == per_asin[a][f"minilm_{metric}"])
        minilm_better = len(asins) - bm25_better - equal
        log(f"  {metric}: BM25 better in {bm25_better}/10, equal in {equal}/10, MiniLM better in {minilm_better}/10")
        summary[metric]["per_asin_bm25_better"] = bm25_better
        summary[metric]["per_asin_equal"] = equal
        summary[metric]["per_asin_minilm_better"] = minilm_better

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9D-Volatility summary: BM25 vs MiniLM comparison",
                "metrics": ["hit1_flip_rate", "hit5_flip_rate", "rr_std"],
                "n_asins": len(asins),
            },
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {SUMMARY_OUT}")

    log("\n=== Stage 9D-Volatility complete ===")


if __name__ == "__main__":
    main()