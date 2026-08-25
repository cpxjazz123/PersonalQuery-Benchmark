# Stage 10H-B — Personalized Retrieval Volatility (mean_t50 selection)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage10h_b_volatility.py`
**Outputs**:
- `hj82_scratch2/.../stage10h_b_per_query.json`
- `hj82_scratch2/.../stage10h_b_volatility.json`
- `hj82_scratch2/.../stage10h_b_summary.json`
- `hj82_scratch2/.../stage10h_c_stats.json` (Stage 10H-C: paired stats)

## Research Question

After resolving the personalization collapse (Stage 10G: unique 1.97 → 6.46
per ASIN), how much does retrieval (Hit@1, Hit@5, RR) flip across users for
the same target product **when only personalized syntactic expression
varies**?

This is **NOT** a retrieval optimization study. The RQ is: does the
expression-induced variation in personalized queries destabilize retrieval?

## Method

1. **mean_t50 selection** per (asin, user): one personalized query per user
   from the Stage 8.5 strict pool (100 ASINs × 50 queries = 5000 candidates).
2. **Retrieval**: 1000 personalized queries against Baby_Products corpus
   (217722 ASIN docs) using BM25 (bm25s, k1=1.5, b=0.75) + MiniLM
   (GPU, all-MiniLM-L6-v2).
3. **Per-ASIN volatility** (same metric definitions as Stage 9D):
   - Hit@1 flip rate: fraction of (i,j) user pairs whose Hit@1 status differs
   - Hit@5 flip rate: same for top-5
   - RR std: std of Reciprocal Rank across the 10 users
4. **Paired tests**: bootstrap CI (N=10000), paired permutation test
   (label-swap, N=10000), Cohen's d for paired samples.

## Results

### Stage 10H-B core (100 ASINs, mean_t50)

| Metric         | mean   | median | BM25 mean | MiniLM mean | diff (B-M) |
|---------------|--------|--------|-----------|-------------|------------|
| Hit@1 flip    | 2.80%  | 0.00%  | 5.13%     | **2.80%**   | +2.33%     |
| Hit@5 flip    | 4.33%  | 0.00%  | 6.24%     | **4.33%**   | +1.91%     |
| RR std        | 3.33%  | 0.10%  | 5.89%     | **3.33%**   | +2.56%     |

### Stage 10H-C: Paired statistical tests

| Metric      | Cohen's d | Interpretation | Paired permutation p (two-sided) | Bootstrap 95% CI (B - M) |
|-------------|-----------|----------------|----------------------------------|--------------------------|
| Hit@1 flip  | 0.127     | negligible     | **0.202** (NOT sig)              | [−1.24%, +6.02%] (CI includes 0) |
| Hit@5 flip  | 0.117     | negligible     | **0.248** (NOT sig)              | [−1.20%, +5.24%] (CI includes 0) |
| RR std      | 0.200     | small          | **0.040** (sig at α=0.05)        | [+0.11%, +5.14%] (CI excludes 0) |

### Item-level heterogeneity (Gini coefficient)

| Metric      | BM25 Gini | MiniLM Gini | Top-5 ASINs share (BM25 / MiniLM) |
|-------------|-----------|-------------|-----------------------------------|
| Hit@1 flip  | 0.899     | 0.947       | 53.2% / **92.9%**                 |
| Hit@5 flip  | 0.875     | 0.916       | 43.4% / 61.0%                    |
| RR std      | 0.783     | 0.876       | 32.8% / 58.2%                    |

**Interpretation**: Gini values 0.78–0.95 indicate high concentration. For
MiniLM Hit@1 flip, the top-5 most-volatile ASINs contribute 92.9% of all
flips — a small subset of products drives almost all observed volatility.

### Item-level alignment (which ASINs flip under each retriever?)

| Metric      | Spearman ρ (BM25 vs MiniLM ASIN-level) | p       | top-5 overlap |
|-------------|----------------------------------------|---------|---------------|
| Hit@1 flip  | 0.148                                  | 0.14    | 0/5           |
| Hit@5 flip  | 0.279                                  | 0.005   | 2/5           |
| RR std      | 0.352                                  | 0.0003  | 0/5           |

**Interpretation**: ASINs that flip under BM25 are largely distinct from
those that flip under MiniLM (Spearman ρ ≤ 0.35, top-5 overlap 0-2/5). This
suggests lexical and dense retrievers are sensitive to **different**
expression-induced variations.

## Interpretation

1. **Most ASINs are stable; a few ASINs drive all observed volatility.**
   Median Hit@1 flip = 0% on MiniLM (84/100 ASINs have zero flip), but the
   mean is 2.80% — the upper tail is concentrated. Gini = 0.947 confirms
   this concentration. This is item-level heterogeneity, not uniform noise.

2. **Item-level alignment is weak.** Different ASINs flip under BM25 vs
   MiniLM (top-5 overlap 0-2/5, Spearman ρ ≤ 0.35). Lexical and dense
   retrievers respond to different expression variations.

3. **Only RR Std reaches formal significance.** The paired permutation test
   yields p = 0.040 for RR Std but p = 0.20 (Hit@1 flip) and p = 0.25
   (Hit@5 flip). Effect sizes follow: Cohen's d = 0.20 (small), 0.13
   (negligible), 0.12 (negligible). For Hit@1 and Hit@5 flip, the
   confidence intervals include 0, so the difference cannot be declared
   significant at α = 0.05.

4. **Stage 10H-B is the first valid retrieval-volatility evidence after
   resolving personalization collapse.** The earlier Stage 9D 0% flip was
   measured on a small (10 ASIN) pool and reflected personalization
   collapse (mean_t50's predecessor showed shared-centroid queries for
   many users). After Stage 10G, different users receive different queries
   (unique 6.46/10), so this is the appropriate setting for retrieval
   robustness analysis.

## Decision: **GO with rigorous framing**

> Stage 10H-B confirms that personalized syntactic expression induces
> measurable but **concentrated** retrieval volatility. Most products
> (84-95% of ASINs) remain stable under user-specific query variation, but
> a minority of products contributes nearly all observed flips (Gini ≥ 0.78
> across all metrics; MiniLM Hit@1 top-5 ASINs account for 92.9% of flips).
>
> BM25 shows larger point estimates on all three volatility indicators, and
> the difference in RR Std reaches statistical significance (paired
> permutation p = 0.040; Cohen's d = 0.20; 95% bootstrap CI [+0.11%,
> +5.14%], excluding 0). The differences in Hit@1 and Hit@5 flip rates are
> **not statistically significant** at α = 0.05 (paired permutation p = 0.20
> and 0.25 respectively; Cohen's d < 0.2; bootstrap CIs include 0).
>
> Item-level alignment between BM25 and MiniLM is weak (Spearman ρ ≤ 0.35),
> suggesting lexical and dense retrievers are sensitive to different
> expression-induced variations.

## Files

- Scripts: `gaussian/syntax_subspace_stage10h_b_volatility.py`,
  `gaussian/syntax_subspace_stage10h_c_stats.py`
- Outputs:
  - `stage10h_b_per_query.json` (1000 user queries × retrieval results)
  - `stage10h_b_volatility.json` (per-ASIN metrics)
  - `stage10h_b_summary.json` (BM25 vs MiniLM comparison)
  - `stage10h_c_stats.json` (paired stats + heterogeneity + alignment)

## Why the self-distance threshold is necessary

Pure contrastive (mean margin alone) and contrastive + threshold (mean_t50)
provide a clean ablation:

| Selection      | Unique | Mean sel_dist | Mean rnd | Sel/Rnd ratio |
|----------------|--------|--------------|-----------|---------------|
| Pure mean      | 4.98   | **272**      | 166       | **0.61** ✗   |
| mean_t50       | 6.46   | 78           | 166       | 2.12 ✓       |

> The self-distance threshold is necessary to prevent the contrastive
> objective from selecting queries that are distinctive from other users but
> no longer representative of the target user.

This is the paper-ready formulation of the ablation.

## Conclusion

> **Stage 10H-B confirms mean_t50 personalization produces user-different
> queries that retrieve the same target product in most cases** (median
> Hit@1 flip 0%, 84/100 ASINs stable on MiniLM). Item-level heterogeneity is
> high: a small subset of products drives nearly all observed volatility.
> BM25 shows larger point-estimate volatility than MiniLM on all three
> indicators, with the RR Std difference reaching statistical significance
> (paired permutation p = 0.040).