"""Stage 10H-C — Tight statistical tests for Stage 10H-B volatility.

Adds three analyses to the Stage 10H-B output:

1. **Paired permutation test** (label-swap within ASINs):
   - For each metric, swap BM25/MiniLM labels per ASIN with prob 0.5
   - Compute mean(B - M) under null distribution
   - p-value = fraction of permuted diffs >= observed diff (one-sided)
   - This complements Wilcoxon and bootstrap CI.

2. **Cohen's d effect size** for paired samples:
   - d = mean(BM25 - MiniLM) / std(BM25 - MiniLM)
   - Standard interpretation: small=0.2, medium=0.5, large=0.8

3. **Item-level heterogeneity (Gini coefficient)**:
   - Measures how concentrated flips are across ASINs.
   - High Gini → a few ASINs contribute most of the flips
   - Validates the "median=0%, mean>0% → concentrated" observation.

Inputs:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_volatility.json

Outputs:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_c_stats.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
VOLATILITY_IN = SCRATCH / "stage10h_b_volatility.json"
OUTPUT = SCRATCH / "stage10h_c_stats.json"

N_PERMUTATIONS = 10000
SEED = 2024


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    if len(diff) < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(np.mean(diff) / sd)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERMUTATIONS, seed: int = SEED) -> tuple:
    """Paired permutation test (label-swap within pairs).

    Returns (observed_diff, perm_mean, perm_std, p_greater, p_two_sided).

    - observed_diff: mean(a - b)
    - perm_mean/std: distribution of mean(a - b) under null (label swap)
    - p_greater: P(perm_diff >= observed_diff)
    - p_two_sided: 2 * min(p_greater, p_less)
    """
    rng = np.random.RandomState(seed)
    diff = a - b
    observed = float(np.mean(diff))
    n = len(diff)
    abs_obs = abs(observed)

    perm_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        # Swap labels with prob 0.5 per pair
        swap = rng.choice([-1, 1], size=n)
        perm_diffs[i] = float(np.mean(diff * swap))

    p_greater = float(np.mean(perm_diffs >= observed))
    p_less = float(np.mean(perm_diffs <= observed))
    p_two_sided = float(2 * min(p_greater, p_less))

    return observed, float(np.mean(perm_diffs)), float(np.std(perm_diffs)), p_greater, p_two_sided


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient. 0 = perfectly uniform, 1 = perfectly concentrated."""
    if len(values) == 0:
        return 0.0
    v = np.sort(np.abs(values))
    n = len(v)
    if np.sum(v) < 1e-9:
        return 0.0
    cum = np.cumsum(v)
    return float((2 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def main():
    log("=== Stage 10H-C — Tight statistical tests ===")

    # === 1. Load Stage 10H-B volatility ===
    log("\n=== 1. Loading Stage 10H-B volatility ===")
    v_data = json.load(open(VOLATILITY_IN))
    per_asin = v_data["per_asin"]
    asins = sorted(per_asin.keys())
    log(f"  ASINs: {len(asins)}")

    metrics = ["hit1_flip_rate", "hit5_flip_rate", "rr_std"]
    stats_results = {}

    # === 2. For each metric: paired permutation + Cohen's d ===
    log("\n=== 2. Paired tests per metric ===")
    for metric in metrics:
        log(f"\n  {metric}:")
        bm25 = np.array([per_asin[a][f"bm25_{metric}"] for a in asins])
        minilm = np.array([per_asin[a][f"minilm_{metric}"] for a in asins])
        diff = bm25 - minilm

        observed = float(np.mean(diff))
        cohens_d = cohens_d_paired(diff)
        obs_diff, perm_mean, perm_std, p_greater, p_two = paired_permutation_test(bm25, minilm)

        log(f"    BM25 mean:   {bm25.mean()*100:.3f}%")
        log(f"    MiniLM mean: {minilm.mean()*100:.3f}%")
        log(f"    Observed mean diff (BM25 - MiniLM): {observed*100:.3f}%")
        log(f"    Cohen's d (paired): {cohens_d:.3f}")
        log(f"    Paired permutation:")
        log(f"      perm mean = {perm_mean*100:.3f}%, std = {perm_std*100:.3f}%")
        log(f"      p(perm >= obs) = {p_greater:.4f}")
        log(f"      p_two_sided    = {p_two:.4f}")

        stats_results[metric] = {
            "bm25_mean": float(bm25.mean()),
            "minilm_mean": float(minilm.mean()),
            "observed_diff_bm25_minus_minilm": observed,
            "cohens_d_paired": cohens_d,
            "cohens_d_interpretation": (
                "large" if abs(cohens_d) > 0.8 else
                "medium" if abs(cohens_d) > 0.5 else
                "small" if abs(cohens_d) > 0.2 else
                "negligible"
            ),
            "perm_mean": perm_mean,
            "perm_std": perm_std,
            "p_permutation_greater": p_greater,
            "p_permutation_two_sided": p_two,
        }

    # === 3. Item-level heterogeneity (Gini) ===
    log("\n=== 3. Item-level heterogeneity (Gini coefficient) ===")
    heterogeneity = {}
    for metric in metrics:
        bm25 = np.array([per_asin[a][f"bm25_{metric}"] for a in asins])
        minilm = np.array([per_asin[a][f"minilm_{metric}"] for a in asins])

        # Concentration: top-K ASINs' contribution to total flips
        total_bm25 = bm25.sum()
        total_minilm = minilm.sum()
        gini_bm25 = gini_coefficient(bm25)
        gini_minilm = gini_coefficient(minilm)

        # Top-5 ASINs share
        if total_bm25 > 1e-9:
            top5_share_bm25 = float(np.sort(bm25)[-5:].sum() / total_bm25)
        else:
            top5_share_bm25 = 0.0
        if total_minilm > 1e-9:
            top5_share_minilm = float(np.sort(minilm)[-5:].sum() / total_minilm)
        else:
            top5_share_minilm = 0.0

        log(f"    {metric}:")
        log(f"      BM25:   Gini={gini_bm25:.3f}, top-5 share={top5_share_bm25*100:.1f}%")
        log(f"      MiniLM: Gini={gini_minilm:.3f}, top-5 share={top5_share_minilm*100:.1f}%")

        heterogeneity[metric] = {
            "gini_bm25": gini_bm25,
            "gini_minilm": gini_minilm,
            "top5_share_bm25": top5_share_bm25,
            "top5_share_minilm": top5_share_minilm,
        }

    # === 4. Item-level alignment (BM25 vs MiniLM per-ASIN) ===
    log("\n=== 4. Item-level alignment: which ASINs flip under each retriever? ===")
    alignment = {}
    for metric in metrics:
        bm25 = np.array([per_asin[a][f"bm25_{metric}"] for a in asins])
        minilm = np.array([per_asin[a][f"minilm_{metric}"] for a in asins])
        # Spearman correlation of ASIN-level values
        from scipy.stats import spearmanr
        if bm25.std() > 1e-9 and minilm.std() > 1e-9:
            rho, rho_p = spearmanr(bm25, minilm)
        else:
            rho, rho_p = 0.0, 1.0
        # Top-5 most volatile ASINs for each retriever
        top5_bm25_asins = [asins[i] for i in np.argsort(bm25)[-5:][::-1]]
        top5_minilm_asins = [asins[i] for i in np.argsort(minilm)[-5:][::-1]]
        overlap = len(set(top5_bm25_asins) & set(top5_minilm_asins))
        log(f"    {metric}: Spearman ρ={rho:.3f} (p={rho_p:.4f}), top-5 overlap={overlap}/5")
        alignment[metric] = {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "top5_bm25_asins": top5_bm25_asins,
            "top5_minilm_asins": top5_minilm_asins,
            "top5_overlap_count": overlap,
        }

    # === 5. Save ===
    log("\n=== 5. Saving ===")
    output = {
        "config": {
            "N_PERMUTATIONS": N_PERMUTATIONS,
            "SEED": SEED,
            "n_asins": len(asins),
        },
        "paired_stats": stats_results,
        "heterogeneity": heterogeneity,
        "alignment": alignment,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    log(f"  wrote → {OUTPUT}")

    log("\n=== Stage 10H-C complete ===")


if __name__ == "__main__":
    main()