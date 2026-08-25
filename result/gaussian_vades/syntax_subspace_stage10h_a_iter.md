# Stage 10H-A — Contrastive Selection Stability (medium stability, not cohort-invariant)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage10h_a_stability.py`
**Output**: `hj82_scratch2/.../stage10h_a_stability.json`

## Goal

Verify how much the mean_t50 contrastive margin's selected query depends on
**which other users are sampled as negatives**. The contrastive term:
```
S(q, u) = mean_{v ≠ u} D(q, v) - D(q, u)
```
is structurally sensitive to the negative cohort. This stage measures whether
the resulting selection is **stable enough** to support the personalization
hypothesis, or whether it is **cohort-driven**.

## Method

For each (asin, user), Stage 10H-A:
1. Compute D matrix [Q × U] for that ASIN's pool queries.
2. Sample negative users from the **GLOBAL pool** of 293 other users (across
   all ASINs). Per-ASIN sampling is infeasible because most ASINs have only
   ~10 users, while subset sizes 5/10/20/50 demand larger pools.
3. For each subset size ∈ {5, 10, 20, 50}, sample N_REPLICATES = 20 random subsets.
4. For each subset, run mean_t50 selection.
5. Aggregate per-user stability metrics across replicates.

**Stability Metrics** (per user × subset_size × N_REPLICATES subsamples):
- **agreement**: fraction of subsamples that produce the mode query
- **unique_count**: number of distinct selected queries across subsamples
- **full_match_rate**: fraction matching the FULL negative-set selection
- **matches_abs_rate**: fraction matching absolute Maha (Stage 8.5 baseline)
- **mean_self_dist**: average D(q_sel, u) across subsamples

## Results

| subset | agreement | unique | sel_dist | full_match | matches_abs |
|--------|-----------|--------|----------|------------|-------------|
| 5      | 0.646     | 3.95   | 78.21    | 0.559      | 0.207       |
| 10     | 0.699     | 3.42   | 79.35    | 0.604      | 0.191       |
| 20     | 0.725     | 3.09   | 80.45    | 0.642      | 0.174       |
| 50     | 0.745     | 2.53   | 81.70    | 0.689      | 0.156       |

**Absolute Maha baseline**: mean sel_dist = 52.11

## Decision: **Medium stability** (not "cohort-invariant")

The contrastive selection is **moderately stable**, but not cohort-invariant.
A perfectly cohort-invariant selection would produce agreement ≈ 100% across
all subset sizes; observed agreement is 65–75%, indicating partial cohort
dependence.

**Acceptable interpretation**:
> mean_t50 exhibits medium-degree selection stability with respect to
> negative-user sampling. Agreement rises from 0.646 to 0.745 as the negative
> subset size grows from 5 to 50, indicating that the contrastive term
> converges with more sampled negatives rather than being driven by any
> specific small cohort.

**What NOT to claim**:
- ❌ "Cohort-independent" (agreement is far below 1.0)
- ❌ "Pareto-optimal query region" (no Pareto frontier has been empirically
  constructed; the observation that unique_count ≪ N_REPLICATES only shows
  that different negative subsets concentrate on a small set of candidates)

### Key observations

1. **Stability improves with subset size**: agreement 0.646 (size=5) →
   0.745 (size=50). With more negatives, the mean margin estimate converges.

2. **Low unique_count** (~2.5-4 across 20 replicates): different negative
   subsets concentrate on a small set of candidates (2-4 distinct queries).
   This shows the selection is not random, but it does not by itself
   establish a Pareto frontier.

3. **Stable self-distance**: mean self distance stays 78-82 across subset
   sizes (vs 52 for absolute Maha). The ~50% increase is the contrastive
   margin's trade-off — sacrificing some closeness for uniqueness.

4. **matches_abs_rate 0.16-0.21**: mean_t50 is a distinct selection
   mechanism, not a near-equivalent of absolute Maha.

5. **full_match_rate 0.56-0.69**: 56-69% of subsamples match the FULL-set
   selection. Subset sampling has 31-44% deviation from full-set result.

## Files

- Script: `gaussian/syntax_subspace_stage10h_a_stability.py`
- Output: `hj82_scratch2/.../stage10h_a_stability.json`

## Conclusion

> mean_t50 contrastive selection shows medium stability under negative-user
> subset variation (agreement 0.65-0.75 across subset sizes 5-50). The
> selected query is **not cohort-invariant** (agreement ≠ 1.0), but it is
> not random either: 20 replicate selections concentrate on a small candidate
> set (unique_count 2.5-4) and preserve consistent self-distance. The
> selection has limited but non-trivial sensitivity to which users form the
> negative cohort.

## Next step

Proceed to Stage 10H-B (Retrieval Volatility) to verify whether the moderate
stability of selection translates to retrieval results.