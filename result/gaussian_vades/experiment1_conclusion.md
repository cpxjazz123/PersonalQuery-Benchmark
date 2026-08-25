# Experiment 1 — Personalized Syntactic Expression Induces Bounded Retrieval Volatility

**Date**: 2026-08-25
**Stages**: 10G (selection) + 10H-A (selection stability) + 10H-B (retrieval volatility) + 10H-C (paired stats)
**Status**: GO with rigorous framing

## Research Question

Does personalized syntactic expression induce retrieval volatility for the
same target product? After resolving the personalization collapse (Stage 10G),
we evaluate whether user-specific syntactic queries destabilize retrieval of
the correct product.

## Method (one paragraph)

For each of 100 ASINs × 10 users (1000 (asin, user) pairs), Stage 10G
**mean_t50** selection picks one query from the Stage 8.5 strict pool
(100 ASINs × 50 candidates = 5000 candidates) by maximizing the contrastive
margin `S(q, u) = mean_{v≠u} D(q, v) - D(q, u)` subject to a self-distance
threshold (≤ 50th percentile of all candidates for that user). Each selected
query is retrieved against the Baby_Products corpus (217722 ASIN docs) by
BM25 (bm25s, k1=1.5, b=0.75) and MiniLM (GPU, all-MiniLM-L6-v2). For each
ASIN we compute Hit@1 flip, Hit@5 flip, and RR std across the 10 users.

## Key Findings

### 1. Selection: different AND user-aligned (Stage 10G)

mean_t50 produces **user-different queries** that **stay close to the target
user's syntactic style**:

| Selection      | Unique per ASIN | Mean sel_dist | Mean rnd | Sel/Rnd ratio |
|----------------|-----------------|---------------|----------|---------------|
| Absolute Maha  | 1.97            | 52            | 166      | 0.31 ✗        |
| Pure mean      | 4.98            | **272**       | 166      | **0.61** ✗    |
| mean_t50       | **6.46**        | 78            | 166      | **2.12** ✓    |

The **self-distance threshold is necessary**, not an engineering trick.
Without it, the contrastive objective selects queries that are distinctive
from other users but no longer representative of the target user (sel_dist
272 > rnd 166 = anti-personalization). The threshold restores user-likeness
(sel_dist 78 < rnd 166).

### 2. Selection stability: medium, not cohort-invariant (Stage 10H-A)

Across 20 negative-subset replicates at sizes {5, 10, 20, 50}:

| subset | agreement | unique | sel_dist |
|--------|-----------|--------|----------|
| 5      | 0.646     | 3.95   | 78.21    |
| 10     | 0.699     | 3.42   | 79.35    |
| 20     | 0.725     | 3.09   | 80.45    |
| 50     | 0.745     | 2.53   | 81.70    |

A perfectly cohort-invariant selection would yield agreement ≈ 100%. The
observed 0.65–0.75 reflects **partial cohort dependence**. We do NOT claim
"Pareto property" or "cohort-independent"; we report agreement honestly.

### 3. Retrieval volatility: bounded for most products, concentrated in a few (Stage 10H-B + 10H-C)

**Core result**: median Hit@1 flip = 0%, Hit@5 flip = 0%, RR std = 0.10%
on MiniLM. **84/100 ASINs have zero hit@1 flip on MiniLM**.

**BM25 vs MiniLM, paired permutation (N=10000)**:

| Metric      | BM25 mean | MiniLM mean | Cohen's d | Paired perm p (two-sided) | 95% bootstrap CI (B − M) | Sig (α=0.05)? |
|-------------|-----------|-------------|-----------|---------------------------|--------------------------|---------------|
| Hit@1 flip  | 5.13%     | **2.80%**   | 0.127     | 0.202                     | [−1.24%, +6.02%]         | **No**        |
| Hit@5 flip  | 6.24%     | **4.33%**   | 0.117     | 0.248                     | [−1.20%, +5.24%]         | **No**        |
| RR std      | 5.89%     | **3.33%**   | 0.200     | **0.040**                 | [+0.11%, +5.14%]         | **Yes**       |

BM25 shows larger point-estimate volatility than MiniLM on **all three**
indicators, but only the **RR Std difference reaches formal significance**
at α = 0.05. Hit@1 / Hit@5 flip differences are direction-consistent but
**not statistically significant** (CI includes 0). The previous "p=0.0051"
Wilcoxon claim for RR std was directionally consistent but used an
asymptotic approximation; the label-swap paired permutation (the correct
test for paired within-ASIN data) yields p = 0.040.

### 4. Item-level heterogeneity (Gini coefficient)

| Metric      | BM25 Gini | MiniLM Gini | Top-5 ASINs share (BM25 / MiniLM) |
|-------------|-----------|-------------|-----------------------------------|
| Hit@1 flip  | 0.899     | 0.947       | 53.2% / **92.9%**                 |
| Hit@5 flip  | 0.875     | 0.916       | 43.4% / 61.0%                    |
| RR std      | 0.783     | 0.876       | 32.8% / 58.2%                    |

Gini 0.78–0.95 indicates high concentration. For MiniLM Hit@1 flip, the
top-5 most-volatile ASINs contribute **92.9%** of all flips. Volatility is
**item-level**, not uniform noise.

### 5. Item-level alignment (which products flip under each retriever?)

| Metric      | Spearman ρ (BM25 vs MiniLM ASIN-level) | p       | top-5 overlap |
|-------------|----------------------------------------|---------|---------------|
| Hit@1 flip  | 0.148                                  | 0.14    | 0/5           |
| Hit@5 flip  | 0.279                                  | 0.005   | 2/5           |
| RR std      | 0.352                                  | 0.0003  | 0/5           |

Spearman ρ ≤ 0.35 with top-5 overlap 0–2/5 indicates that **BM25 and
MiniLM are sensitive to different products' expression-induced variations**.
There is no single "MiniLM more stable" rule — only a tendency.

## Conclusion (paper-ready)

> **Personalized syntactic expression induces bounded retrieval volatility.**
>
> After resolving the personalization collapse with contrastive Mahalanobis
> selection under a self-distance threshold (mean_t50; unique 6.46/ASIN,
> sel/rnd = 2.12), we assess retrieval robustness on 100 ASINs × 10 users
> against a 217K ASIN Baby_Products corpus. Most products (84/100 on MiniLM
> Hit@1) remain stable under user-specific query variation, but a minority
> of products contributes nearly all observed flips (Gini 0.78–0.95; MiniLM
> Hit@1 top-5 ASINs account for 92.9% of flips).
>
> BM25 shows larger point-estimate volatility than MiniLM on all three
> indicators (Hit@1 flip, Hit@5 flip, RR std), but only the RR Std
> difference reaches formal significance under a paired permutation test
> (label-swap within ASINs, N=10000): p = 0.040, Cohen's d = 0.20 (small),
> 95% bootstrap CI [+0.11%, +5.14%]. The Hit@1 and Hit@5 flip differences
> are direction-consistent but **not statistically significant** (p = 0.20
> and p = 0.25 respectively; Cohen's d < 0.2; bootstrap CIs include 0).
>
> Item-level alignment between BM25 and MiniLM is weak (Spearman ρ ≤ 0.35),
> suggesting lexical and dense retrievers are sensitive to different
> expression-induced variations. Volatility is **item-level**, not
> retriever-level.

## What this means for the paper

1. **The personalization collapse (Stage 10G) was the correct diagnosis.**
   Earlier "0% volatility" findings reflected shared queries for many users,
   not robustness. mean_t50 unblocks valid retrieval analysis.
2. **Selection mechanism can be defended quantitatively.**
   The mean_t50 selection is bounded-but-not-trivial in stability (0.65–0.75
   agreement), and the threshold is a necessary physical constraint.
3. **Volatility is real but bounded for most products.**
   median = 0% across three metrics, with concentrated item-level
   heterogeneity that should be reported honestly.
4. **BM25 vs MiniLM claim should be limited to RR Std** at formal
   significance. The point-estimate direction holds for all three
   metrics, but only one survives paired testing.

## Files

- Scripts:
  - `gaussian/syntax_subspace_stage10g_contrastive.py` (Stage 10G)
  - `gaussian/syntax_subspace_stage10h_a_stability.py` (Stage 10H-A)
  - `gaussian/syntax_subspace_stage10h_b_volatility.py` (Stage 10H-B)
  - `gaussian/syntax_subspace_stage10h_c_stats.py` (Stage 10H-C paired stats)
- Outputs:
  - `hj82_scratch2/wenyu/gaussian_vades/stage10g_contrastive_selection.json`
  - `hj82_scratch2/wenyu/gaussian_vades/stage10h_a_stability.json`
  - `hj82_scratch2/wenyu/gaussian_vades/stage10h_b_per_query.json`
  - `hj82_scratch2/wenyu/gaussian_vades/stage10h_b_volatility.json`
  - `hj82_scratch2/wenyu/gaussian_vades/stage10h_b_summary.json`
  - `hj82_scratch2/wenyu/gaussian_vades/stage10h_c_stats.json`

## Next Steps

With selection stability + retrieval volatility established, the next
research questions are:

1. **Selection scaling**: do larger user pools (currently 293 negative users)
   push agreement higher, or does ~0.75 represent a structural ceiling?
2. **Item-level heterogeneity origin**: are the high-volatility ASINs
   characterized by ambiguous titles, multi-product listings, or short
   description text that interacts badly with user-style drift?
3. **Volatility vs quality trade-off**: does the mean_t50 selection's
   50% sel_dist increase (78 vs 52) translate to measurable retrieval
   lift in held-out user-eval against the absolute Maha baseline?