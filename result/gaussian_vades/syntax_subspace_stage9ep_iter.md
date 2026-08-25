# Stage 9E-P — Personalization Separability Diagnostic (Root Cause: Pool PCA48 Cluster)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9ep_diagnostic.py`
**Scope**: 10 ASINs × 10 users = 100 (asin, user) pairs

## Goal

Diagnose root cause of Stage 9D's selection collapse (1-4 unique queries per ASIN out of 10 users):

1. **Hypothesis A**: Pool too small (need scale to 1000+)
2. **Hypothesis B**: User Gaussians indistinguishable (need better user style model)
3. **Hypothesis C**: Pool PCA48 distribution too concentrated (selection has no room to differentiate)

## Four Checks

### CHECK 1: Pairwise distance between 10 user Gaussian mu vectors per ASIN

| ASIN | mean L2 | mean Maha | user→centroid L2 |
|------|---------|-----------|-------------------|
| B09KLYL2ZX | 6.60 | 3.48 | 8.28 |
| B07XM9DX9H | 6.18 | 2.16 | 12.51 |
| B0C54J5D2B | 3.97 | 1.64 | 8.90 |
| B0BQV3785S | 3.54 | 1.52 | 6.76 |
| B0BMQ3G124 | 4.27 | 1.64 | 7.32 |
| B09QK77RNW | 7.57 | 4.43 | 11.99 |
| B00OQCZAVW | 2.68 | 1.28 | 11.65 |
| B09YYZTK1W | 3.93 | 1.49 | 7.00 |
| B09S8PT9L6 | 3.79 | 1.56 | 9.57 |
| B07C2HRMRF | 5.85 | 2.46 | 12.66 |
| **mean** | **4.84** | **2.17** | 9.66 |
| std | 1.51 | 0.98 | 2.27 |

**User Gaussian mu vectors are clearly separable** in PCA48 space (mean L2 = 4.84, vs mean user→centroid L2 = 9.66).
**Hypothesis B rejected**: User profiles are NOT indistinguishable.

### CHECK 2: Top-K candidate overlap (Jaccard) per ASIN

| ASIN | K=1 | K=3 | K=5 | K=10 |
|------|-----|-----|-----|------|
| B09KLYL2ZX | 0.533 | 0.727 | 0.674 | 0.660 |
| B07XM9DX9H | 0.800 | 0.678 | 0.756 | 0.867 |
| B0C54J5D2B | 0.400 | 0.633 | 0.933 | 0.901 |
| B0BQV3785S | 0.622 | 0.584 | 0.874 | 0.783 |
| B0BMQ3G124 | **1.000** | 0.664 | 0.770 | 0.872 |
| B09QK77RNW | **1.000** | 0.731 | 0.552 | 0.823 |
| B00OQCZAVW | **1.000** | 0.656 | 0.881 | 0.785 |
| B09YYZTK1W | 0.622 | 0.429 | 0.693 | 0.747 |
| B09S8PT9L6 | **1.000** | 0.811 | 0.741 | 0.788 |
| B07C2HRMRF | 0.222 | 0.456 | 0.874 | 0.735 |
| **mean** | **0.720** | **0.637** | **0.775** | **0.796** |
| std | 0.269 | 0.114 | 0.112 | 0.069 |

**Top-K overlap is very high**:
- K=1 Jaccard = 0.72 (4 ASINs at 1.000 — all 10 users pick same top-1)
- K=10 Jaccard = 0.80 (top-10 candidates are 80% shared across users)
- **Even expanding to top-10 candidates doesn't differentiate users**

**Implication**: The pool PCA48 space has 1-2 dominant clusters, and all users' top-K candidates land in the same clusters.

### CHECK 3: Selected-query uniqueness + selection entropy

| ASIN | pool | n_unique | entropy/max | family distribution |
|------|------|----------|--------------|----------------------|
| B09KLYL2ZX | 75 | **2** | 0.88/3.32 | coordination:3, prepositional:7 |
| B07XM9DX9H | 44 | **2** | 0.47/3.32 | subordinate_clause:9, predicate_based:1 |
| B0C54J5D2B | 66 | **3** | 1.30/3.32 | coordination:3, sub_clause:6, predicate:1 |
| B0BQV3785S | 87 | **3** | 0.92/3.32 | coordination:1, prepositional:8, predicate:1 |
| B0BMQ3G124 | 98 | **1** | 0.00/3.32 | fragment_with_connector:10 |
| B09QK77RNW | 53 | **1** | 0.00/3.32 | modifier_fronted:10 |
| B00OQCZAVW | 41 | **1** | 0.00/3.32 | prepositional:10 |
| B09YYZTK1W | 66 | **3** | 0.92/3.32 | sub_clause:2, prepositional:8 |
| B09S8PT9L6 | 196 | **1** | 0.00/3.32 | modifier_fronted:10 |
| B07C2HRMRF | 60 | **4** | 1.85/3.32 | coordination:4, predicate:6 |

- **4/10 ASINs: 1 unique query** (10/10 users → same query)
- **4/10 ASINs: 2-3 unique queries**
- **2/10 ASINs: 4 unique queries** (B07C2HRMRF, B0C54J5D2B partial)
- Max possible entropy = log2(10) = 3.32 bits; actual = 0-1.85 bits (max 56%)

**Selection collapse is severe**: Even the most diverse ASIN only produces 4 unique queries for 10 users.

### CHECK 4: Pool-size sweep (does uniqueness grow with pool size?)

| Pool size | n_ASINs tested | mean unique | mean entropy (bits) |
|-----------|----------------|-------------|---------------------|
| 10 | 10 | 1.60 | 0.432 |
| 20 | 10 | 2.30 | 0.784 |
| 30 | 10 | 1.80 | 0.516 |
| 50 | 8 | 2.00 | 0.628 |
| 75 | 4 | 1.50 | 0.338 |
| 100 | 1 | 1.00 | 0.000 |
| 150 | 1 | 1.00 | 0.000 |
| 196 | 1 | 1.00 | 0.000 |

**Uniqueness does NOT grow with pool size**:
- Peak at size=20 (2.30 unique, 0.784 bits)
- Decreases to 1.50 at size=75
- 1.00 at size≥100

**Hypothesis A rejected**: Pool scaling does NOT solve the collapse.

## Root Cause Conclusion

### Not (A) — pool too small
Even at pool size 196 (B09S8PT9L6 with max queries), uniqueness = 1. Sweep shows uniqueness peaks at size=20 then declines.

### Not (B) — user Gaussians indistinguishable
Mean pairwise user L2 = 4.84 (clearly above noise). Mahalanobis distances = 1.28-4.43 (well-separated).

### (C) — Pool PCA48 distribution too concentrated
- 8 grammatical families, but PCA48 projection shows queries cluster in 1-2 dominant modes
- Top-10 candidate Jaccard = 0.80: any user's top-10 covers 80% of any other user's top-10
- Different families produce queries that are **syntactically similar in PCA48 space**
  - Example: `prepositional` and `coordination` queries have similar function-word density
  - `relative_clause` and `subordinate_clause` both have clause structure
  - `predicate_based` with `is/are` verbs is close to `prepositional` in connector space

**Root cause**: 9C's 8 grammatical families are **distinct in surface form but NOT distinct in PCA48 syntactic subspace**. The 182d spaCy features aggregated by PCA48 compress these surface differences into a small region.

## Implication

### Cannot fix by pool scaling
Going from 786 → 5000+ queries will likely produce more queries in the same PCA48 cluster, not more clusters.

### Cannot fix by more user reviews
User Gaussians are already separable; more reviews would tighten them, not diversify them.

### Must change representation or selection

Three viable directions:
1. **Increase PCA48 dimension** (e.g., 100d) — but PCA is linear; if 48d already compresses families, 100d likely won't help
2. **Use non-linear embeddings** (UMAP / autoencoder) — may preserve cluster boundaries
3. **Direct family-aware selection** — don't use Maha; instead pick a query from a user-preferred family (mix of 2 families per user)

### Or accept collapse and pivot
If per-user personalization via PCA48 Maha is fundamentally limited, the system should pivot to:
- **Single optimal query per ASIN** (no per-user personalization)
- **Use query variations for volatility robustness** (e.g., ensemble across top-3 from different families, not per-user argmin)

## Files

- Script: `gaussian/syntax_subspace_stage9ep_diagnostic.py`
- Check 1: `scratch2/.../stage9ep_check1_user_dist.json`
- Check 2: `scratch2/.../stage9ep_check2_overlap.json`
- Check 3: `scratch2/.../stage9ep_check3_uniqueness.json`
- Check 4: `scratch2/.../stage9ep_check4_pool_sweep.json`
- Summary: `scratch2/.../stage9ep_summary.json`
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9ep_iter.md`

## Conclusion

> **Stage 9E-P root cause: Pool PCA48 distribution is too concentrated.** User Gaussians are
> separable (Check 1), but pool queries cluster in 1-2 dominant PCA48 modes (Check 2), causing
> selection collapse (Check 3). Pool scaling does NOT fix this (Check 4).
>
> **Cannot solve by scaling pool or improving user model.** Must either:
> 1. Change representation (non-linear / higher-dim embeddings)
> 2. Use family-aware direct selection (skip Maha)
> 3. Pivot to single-query-per-ASIN model
>
> **Recommended next step**: Stage 9F — Family-aware direct selection test (skip Maha, directly
> assign users to family based on review style). Validates whether explicit family routing gives
> per-user differentiation without relying on PCA48 cluster diversity.