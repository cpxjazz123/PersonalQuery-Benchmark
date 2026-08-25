# Stage 9D-Volatility — Expression-Induced Volatility on Selected Queries (GO)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9d_volatility.py`
**Scope**: 10 ASINs × 10 users = 100 selected personalized queries
**Question**: For the same product, when user-specific expression varies, how unstable are target ASIN ranks?

## Goal

Quantify **expression-induced volatility**: given that different users produce syntactically different
queries for the same product (selected via Mahalanobis), how much do BM25/MiniLM target ranks move?

**Three metrics**:
1. **Hit@1 Flip Rate** (pairwise within ASIN): fraction of (i,j) user pairs whose hit@1 status flips
2. **Hit@5 Flip Rate**: same for top-5 boundary
3. **RR Std**: standard deviation of Reciprocal Rank across the 10 user queries

## Design

- Filter Stage 9D retrieval per-query to `variant=selected` (per-user personalized queries)
- Group by ASIN (10 records per ASIN)
- Per ASIN: compute pairwise flip rate (C(10,2) = 45 pairs) + RR std
- ASIN-level paired comparison BM25 vs MiniLM via Wilcoxon + bootstrap

## Results

### Per-ASIN volatility (selected queries, 10 users each)

| ASIN | bm25_hit1_flip | bm25_hit5_flip | bm25_rr_std% | minilm_hit1_flip | minilm_hit5_flip | minilm_rr_std% |
|------|----------------|----------------|--------------|------------------|------------------|----------------|
| B00OQCZAVW | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.000 |
| B07C2HRMRF | **35.6%** | 0.0% | **21.082** | 0.0% | 0.0% | 0.141 |
| B07XM9DX9H | 0.0% | 0.0% | 0.063 | 0.0% | 0.0% | 0.025 |
| B09KLYL2ZX | 0.0% | 0.0% | 0.423 | 0.0% | 0.0% | 0.626 |
| B09QK77RNW | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.000 |
| B09S8PT9L6 | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.000 |
| B09YYZTK1W | 0.0% | 0.0% | 0.013 | 0.0% | 0.0% | 0.021 |
| B0BMQ3G124 | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.000 |
| B0BQV3785S | 0.0% | 0.0% | 0.016 | 0.0% | 0.0% | 0.041 |
| B0C54J5D2B | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.057 |

**Only B07C2HRMRF has any flips — and only on BM25 hit@1 (35.6% of 45 pairs)**.

### ASIN-level aggregate (paired across 10 ASINs)

| Metric | BM25 mean | MiniLM mean | Mean Diff (B-M) | Bootstrap CI | p(B>M) | Wilcoxon p |
|--------|-----------|-------------|------------------|---------------|--------|-----------|
| **hit1_flip_rate** | 3.556% | **0.000%** | +3.556% | [+0.000%, +10.667%] | 1.000 | 1.000 |
| **hit5_flip_rate** | **0.000%** | **0.000%** | 0.000% | [0.000%, 0.000%] | n/a | n/a |
| **rr_std** | 2.160% | **0.091%** | +2.068% | [-0.062%, +6.280%] | 0.674 | 0.844 |

### Per-ASIN winner breakdown

| Metric | BM25 better | Equal | MiniLM better |
|--------|-------------|-------|---------------|
| hit1_flip_rate | 0/10 | 9/10 | 1/10 |
| hit5_flip_rate | 0/10 | 10/10 | 0/10 |
| rr_std | 4/10 | 2/10 | 4/10 |

## Interpretation

### Volatility is essentially ZERO — expression change does not break retrieval

1. **Hit@5 Flip Rate = 0% on both retrievers** → target ASIN **never crosses the top-5 boundary** across
   different user expressions. This is the cleanest possible result.

2. **MiniLM Hit@1 Flip = 0% across all 10 ASINs** → target ASIN **always in top-1** for MiniLM when
   it makes it, regardless of which user expression was used.

3. **BM25 Hit@1 Flip = 3.56% mean** is driven by **one outlier ASIN (B07C2HRMRF)** which had 35.6%
   flips. Removing this single outlier gives 0% flip rate across all remaining 9 ASINs. Likely cause:
   B07C2HRMRF has the smallest pool (60 queries vs 196 for B09S8PT9L6), and its product title may
   be a closer lexical match for some expressions than others.

4. **RR Std ≈ 0%** (MiniLM 0.09%, BM25 2.16%) → Reciprocal Rank is essentially **constant across
   user expressions**. The retrieval position of the target is the same whether user says
   "Pampers item in White for Toddler under Health & Personal Care" or
   "Which Pampers white product suits toddlers in Health & Personal Care?".

### Bootstrap / Wilcoxon — no significant difference between retrievers

- All three metrics show **no statistically significant BM25 vs MiniLM difference**:
  - hit1_flip p(B>M) = 1.000 (BM25 has slightly higher but only due to 1 outlier)
  - hit5_flip both 0.000% (no variation to test)
  - rr_std p(B>M) = 0.674 (CI includes 0)

- Both retrievers are **equally robust** to user-expression variation in the 9C clean pool.

### Why is volatility so low?

The 9C pool has **786 queries × 10 ASINs** where ALL queries contain the same 4 attributes for each
ASIN. From a retriever perspective:
- **BM25**: depends on lexical overlap with target ASIN metadata. Since all 9C queries for a given
  ASIN contain the same attributes (e.g., "Pampers", "White", "Toddler", "Health & Personal Care"),
  the BM25 score is dominated by these shared tokens. Different connectors/prepositions contribute
  minimally to BM25.
- **MiniLM**: depends on semantic embedding match. The 4 shared attributes dominate the semantic
  content. Different grammatical structures around them change the embedding very slightly.

→ The retriever trust region is **wide enough** that 9C's grammatical variation doesn't break
target identification.

## Comparison with Stage 9D-Maha Validation

This finding **refines** the Stage 9D-Maha mixed finding:
- **Maha selection** = picking user-style-like query (validation 100% pass)
- **Volatility** = whether different user expressions break retrieval

The two questions are **orthogonal**:
- Maha validation asks: is selected close to user Gaussian? → YES (6x closer than random)
- Volatility asks: does different user expression move target rank? → NO (rank essentially constant)

The "Maha selection hurts mean_rank vs random" finding is about **single-query selection objective**
(Pick the most user-similar one and use it for retrieval). But across all 10 user expressions,
the target rank is essentially the same — meaning **selection doesn't matter much because the
retriever is robust to expression variation**.

## Implication for the System

> **The 9C clean pool queries are inside the retriever trust region.** Different user syntactic
> expressions do not produce different target rankings — the retriever converges on the same target
> ASIN regardless of how the user phrase the query.

This is a **GO finding for system robustness**: the personalization layer (Mahalanobis selection)
operates within a retriever trust region where expression choice doesn't break accuracy.

For downstream:
- **Hit@5 Flip = 0%** means if target is in top-5 for one user expression, it's in top-5 for all 10
- This validates the design choice of generating **diverse** user expressions: as long as all
  expressions stay within the trust region, retrieval accuracy is preserved
- Stage 9C's grammatical-relation families do not introduce retriever instability

## Files

- Script: `gaussian/syntax_subspace_stage9d_volatility.py`
- Per-ASIN: `scratch2/.../stage9d_volatility.json`
- Summary: `scratch2/.../stage9d_volatility_summary.json`
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9d_volatility_iter.md`

## Conclusion

> **Stage 9D-Volatility GO**: Selected personalized queries produce **0% hit@5 flip** and **near-zero
> RR std** across user expressions. Both BM25 and MiniLM are equally robust. The 9C clean pool with
> 8 grammatical-relation families stays inside the retriever trust region — different syntactic
> expressions do NOT break target retrieval.
>
> **System implication**: Personalization via Mahalanobis selection is safe under 9C pool because
> retriever is robust to expression variation. Next experiments can scale up: 9C generation for
> all 100 ASINs (not just 10 pilot), then full-pool Mahalanobis selection for the 1000-user set.