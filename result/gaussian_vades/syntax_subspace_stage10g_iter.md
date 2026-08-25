# Stage 10G — Contrastive Mahalanobis Selection (GO)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage10g_contrastive.py`
**Output**: `hj82_scratch2/.../stage10g_contrastive_selection.json`

## Goal

Resolve Stage 9G's personalization collapse (3.30 unique / 10 users baseline on
9C+9F pool) by **changing the SELECTION MECHANISM** instead of the feature
representation. Specifically, instead of `q* = argmin_q D(q, u)`, use a
contrastive score that penalizes queries that are similarly close to other users.

## Method

### Three selection variants

1. **Absolute Maha** (Stage 8.5 baseline):
   `q* = argmin_q D(q, u)`

2. **Nearest-other margin**:
   `M_nearest(q, u) = min_{v ≠ u} D(q, v) - D(q, u)`
   `q* = argmax_q M_nearest(q, u)`

3. **Mean-other margin**:
   `M_mean(q, u) = mean_{v ≠ u} D(q, v) - D(q, u)`
   `q* = argmax_q M_mean(q, u)`

Intuition: A query at the centroid of many users will have low margin even if its
absolute distance is small (deprioritized). A query uniquely close to user u
(vs other users) gets prioritized.

### Self-distance threshold constraint

To prevent "anti-personalization" (selecting a query that's far from everyone,
including u), apply threshold `D(q, u) ≤ percentile(d)`:
- For each user, threshold = percentile(d_self, pct) where pct ∈ {25, 50, 75}
- Only consider queries with d_self ≤ threshold

### Pool + user Gaussian setup

- **Pool**: Stage 8.5 strict pool — 100 asins × 50 queries = 5000 queries
- **User Gaussians**: Stage 9G base level — 294 users with mu/sigma_diag in PCA48
- **Scaler + PCA48**: `gaussian_vades._syntax_subspace_prepare()` (same as Stage 8.5/9G)
- **Vectorized Maha**: `D[q,u] = sum((z_q - mu_u)^2 / sigma_u)` via broadcasting

## Results

### Variant Comparison (100 asins, 1000 user-asin pairs)

| Variant         | Unique | top1 Jacc | sel dist | rnd dist | sel/rnd |
|-----------------|--------|-----------|----------|----------|---------|
| **absolute** (baseline) | 1.97 | 0.803 | 52.11 | 165.55 | 3.18 |
| nearest         | 4.23 | 0.577 | 74.16 | 165.55 | 2.23 |
| mean            | 4.98 | 0.502 | 271.53 | 165.55 | **0.61** ✗ |
| nearest_t25     | 3.91 | 0.609 | 54.35 | 165.55 | 3.05 |
| nearest_t50     | 4.03 | 0.597 | 55.90 | 165.55 | 2.96 |
| nearest_t75     | 4.10 | 0.590 | 58.95 | 165.55 | 2.81 |
| mean_t25        | 6.12 | 0.388 | 66.16 | 165.55 | 2.50 |
| **mean_t50** ★ | **6.46** | **0.354** | 78.24 | 165.55 | **2.12** |
| mean_t75        | 6.31 | 0.369 | 100.98 | 165.55 | 1.64 |

### Per-ASIN unique distribution

| Variant | min | median | max |
|---------|-----|--------|-----|
| absolute (Stage 8.5) | 1 | 2 | 4 |
| mean_t50 (Stage 10G) | 3 | 6 | 10 |

mean_t50 boosts uniqueness across the whole ASIN population — even the
worst-case ASIN goes from 1 unique (full collapse) to 3 unique.

## Decision: **GO**

User-defined GO criteria evaluation:

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Unique selected / 10 users | 3.3 → 5-6+ | **6.46** | ✓ |
| Top-1 Jaccard | < 0.46 | **0.354** | ✓ |
| Selected vs random distance ratio | preserved (< 1.0× is fail) | **2.12×** | ✓ |

**Stage 10G contrastive selection achieves a 3.3× improvement in uniqueness
(1.97 → 6.46) while preserving user-style alignment (sel_dist 78 vs rnd_dist
166, ratio 2.12).**

## Why contrastive works

The personalization collapse root cause is **shared centroid pull**:
- Many users' Mahalanobis Gaussians are close to the pool centroid
- `argmin D(q, u)` always picks the same low-distance query (the one closest
  to centroid)
- This is amplified when user Gaussians are similar to each other

Contrastive margin (`M = D_others - D_self`) explicitly rewards "user-specific"
queries that are far from OTHER users while still being close to the target
user. The threshold prevents anti-personalization (M can be maximized by
selecting a query that's far from everyone, including u).

### Why threshold is critical

Pure `mean` (no threshold) gives 4.98 unique BUT sel_dist = 271.53, **worse
than random distance 165.55** (ratio 0.61). The pure margin pushes selection
to "far from everyone, including u" — anti-personalization. The 50th
percentile threshold ensures D(q, u) ≤ median(d_self), keeping selected
queries user-relevant.

### Why absolute_t variants don't differ from absolute

`absolute_t25/t50/t75` produce identical results to `absolute`. Reason:
absolute Maha picks the minimum-D query, which is always below any reasonable
threshold (D_min << percentile). The threshold doesn't filter anything.

## Sweet spot: mean_t50

- **Unique**: 6.46 (median 6, max 10) — vs 1.97 (max 4) baseline
- **Top-1 Jaccard**: 0.354 — vs 0.803 baseline (-56% reduction)
- **sel/rnd ratio**: 2.12 — preserves user style
- **Robustness**: balanced sel_dist (78) between mean_t25 (66) and mean_t75 (101)

Alternative sweets:
- mean_t25 (6.12 unique, 0.388 Jaccard, ratio 2.50): more conservative, prefers user-likeness
- mean_t75 (6.31 unique, 0.369 Jaccard, ratio 1.64): more aggressive uniqueness

## Files

- Script: `gaussian/syntax_subspace_stage10g_contrastive.py`
- Output: `hj82_scratch2/.../stage10g_contrastive_selection.json`

## Conclusion

> **Stage 10G GO with mean_t50.** Contrastive Mahalanobis selection
> (mean-other margin + 50th percentile threshold) lifts unique selected
> queries from 1.97/10 to 6.46/10 (3.3×), reduces top-1 Jaccard from
> 0.803 to 0.354 (well below 0.46 target), and preserves user-style
> alignment (sel/rnd ratio 2.12×).
>
> **Personalization ceiling was the SELECTION MECHANISM, not feature
> representation.** Argmin collapse on shared centroid queries was the
> root cause; contrastive scoring breaks it by rewarding user-specific
> queries that are far from other users while still close to target user.

## Next step (potential Stage 10H)

1. **Validate on retrieval downstream**: does mean_t50 selection translate
   to better retrieval hit@10 / mean_rank? Re-run retrieval with mean_t50
   selected queries.
2. **Other contrastive forms**:
   - Per-pair margin (max over v≠u of D(q,v) - D(q,u), instead of mean)
   - Cosine margin (relative angular distance, not Maha)
3. **Threshold optimization** — find optimal balance via Pareto curve
4. **Combine with retrieval scores** — reweight contrastive margin by
   initial retrieval rank