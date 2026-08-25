# Stage 10K — Selection Improvement: A1 Reject-Repeat Solves Unique Ceiling

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage10k_selection.py`
**Outputs**:
- `hj82_scratch2/.../stage10k_selection.json`
- `hj82_scratch2/.../stage10k_summary.json`

## Research Question

> Can we push per-ASIN unique selected queries from 6.46 (Stage 10G mean_t50)
> higher while preserving personalization (sel_dist < rnd_dist)?

Stage 10G mean_t50 has a **3.54 / 10 unique gap** (mean 6.46, median 6, max 10).
Collisions happen because two users with similar Mahalanobis profiles pick
the same best-margin candidate. We tested two strategies:

- **A1 (reject-repeat)**: greedy post-processing — when a user's best-margin
  candidate collides with an already-picked query, take next-best unique.
- **C3 (cohort cluster)**: cluster users by Mahalanobis profile (KMeans K=2-5),
  force intra-cluster distinctness via A1.

## Method

Same setup as Stage 10G:
- User Gaussians: 294 base-level users, mu/sigma_diag in PCA48 space
- Pool: 100 ASINs × 50 candidates = 5000 queries (Stage 8.5 strict pool)
- (asin, user) pairs: 100 ASINs × 10 users from Stage 8.5 selection
- Per ASIN: D matrix [Q=50, U=10], mean-other margin = `mean_v D(q,v) - D(q,u)`
- Threshold: 50th percentile of self-distance (matches Stage 10G mean_t50)

### A1 (reject-repeat)
```python
for user in users_in_order:
    ranked = argsort(-margin(user))  # best first
    chosen = first not in already_picked
    selected.append(chosen)
    already_picked.add(chosen)
```

### C3 (cohort cluster) — v2 after fix
- KMeans(K=k) on user mu vectors (Mahalanobis profile)
- For each cluster: run A1 within cluster
- Cross-cluster: distinctness constraint propagates (already_picked global)

## Results

| Strategy | unique_mean | unique_median | sel_dist | rnd_dist | sel/rnd | Δ vs baseline |
|----------|------------:|--------------:|---------:|---------:|--------:|---------------|
| mean_t50 (baseline) | 6.46 | 6.0 | 78.2 | 165.6 | 0.47 | — |
| **A1_reject_repeat** | **10.00** | **10.0** | **78.6** | **165.6** | **0.47** | **+3.54** ⭐ |
| C3_kmeans_k2 | 10.00 | 10.0 | 78.8 | 165.6 | 0.48 | +3.54 (no extra) |
| C3_kmeans_k3 | 10.00 | 10.0 | 78.7 | 165.6 | 0.48 | +3.54 (no extra) |
| C3_kmeans_k4 | 10.00 | 10.0 | 78.7 | 165.6 | 0.48 | +3.54 (no extra) |
| C3_kmeans_k5 | 10.00 | 10.0 | 78.5 | 165.6 | 0.47 | +3.54 (no extra) |

**A1 alone is sufficient to reach the 10/10 ceiling.** C3 clustering adds
no value because A1 already enforces uniqueness greedily.

## Decision: **A1 REJECT-REPEAT = GO** ⭐⭐⭐

A single-line change in the selection post-processing:
```python
# Before (mean_t50):
idx = int(np.argmax(margin[valid_mask]))
# After (A1):
ranked = np.argsort(-margin[valid_mask])
for cand in ranked:
    if cand not in picked: chosen = cand; break
```

Effects:
- **unique: 6.46 → 10.00** (+55%, ceiling hit)
- **sel_dist: 78.2 → 78.6** (Δ +0.4, no degradation)
- **sel/rnd ratio: 0.47 → 0.47** (unchanged)
- **Complexity: O(U × Q log Q)** (negligible vs O(U² × Q) for full D matrix)

## Why C3 (KMeans cluster) Adds No Value

When intra-cluster A1 is run, the algorithm naturally enforces global
distinctness via the `used_global` set. KMeans partitioning changes
the order users are processed but not the greedy outcome: each user
takes their best unclaimed candidate. The clustering has no benefit
when the underlying scoring (margin) is already unique-maximizing.

**Lesson**: Once the selection objective produces a unique candidate
when collision is removed, no clustering heuristic is needed.

## Per-ASIN Distribution

A1 achieves 10/10 unique on EVERY ASIN — not just a few. Before A1,
the distribution was: 1 ASIN hit 10 unique, 3 hit 9, 20 hit 8, 24 hit 7,
30 hit 6, 14 hit 5, 7 hit 4, 1 hit 3. After A1, **all 100 ASINs reach
10 unique**, eliminating the lower tail entirely.

## What does this mean?

1. **The Stage 10G mean_t50 mechanism was correct** — the contrastive
   margin objective naturally produces user-distinctive queries. The
   only thing missing was collision handling.
2. **Selection diversity was not the bottleneck** — the pool has
   43.9 unique / 50 candidates mean, and the margin surface has enough
   structure to give 10 distinct optima when collision is removed.
3. **The 50th percentile threshold does not over-constrain** — even
   after A1 enforces uniqueness, the self-distance distribution stays
   close to the baseline (sel_dist 78.6 vs 78.2), confirming the
   threshold is safe under the relaxed assignment.

## Paper-ready framing

> **Selection diversity can be pushed to the 10/10 ceiling via a one-line
> post-processing step (reject-repeat), without sacrificing personalization.**
>
> Stage 10G mean_t50 produces 6.46 unique queries per ASIN across 10
> users (median 6, range 3-10). We find that the entire 3.54 unique gap
> is caused by collision: two users with similar Mahalanobis profiles
> independently pick the same best-margin candidate. Adding a
> reject-repeat step (when a user's best collides, take next-best unique)
> lifts unique to 10.00 (100% ceiling) while keeping `sel_dist` essentially
> unchanged (78.2 → 78.6) and the `sel/rnd` ratio at 0.47. This means
> the personalization collapse previously attributed to "limited pool
> diversity" was actually a **collision artifact** in the greedy assignment,
> not a fundamental limitation of the margin objective.

## Files

- Script: `gaussian/syntax_subspace_stage10k_selection.py`
- Outputs (scratch2):
  - `stage10k_selection.json` — per-ASIN × strategy selected_idx + sel_d_self
  - `stage10k_summary.json` — aggregated metrics

## Next Steps

1. **Apply A1 to Stage 10G/H/I/J re-analysis**: the entire pipeline
   (selection → retrieval → volatility) needs to be re-run with A1 to
   confirm retrieval behavior is unchanged or improves.
2. **Verify A1 at larger pool sizes**: with K=100-200 candidates, does
   A1 still achieve 10/10 unique?
3. **Stress test at larger user counts**: with 20-50 users per ASIN
   (not just 10), does the ceiling hold or degrade?

## What changed from initial run

Initial C3 implementation had a bug: I forced ONE candidate per cluster,
which collapsed 10 users to K candidates (unique=2 for K=2). Fixed by
running A1 within each cluster (intra-cluster distinctness) with a
global `used_global` set. After fix, C3 = A1 ceiling (10.00).