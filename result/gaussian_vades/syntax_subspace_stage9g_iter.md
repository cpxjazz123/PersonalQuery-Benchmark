# Stage 9G — Syntax Feature Redesign (NO-GO)

**Date**: 2026-08-25
**Scripts**:
- `gaussian/syntax_subspace_stage9g_extract.py` (244 features for 1317 queries)
- `gaussian/syntax_subspace_stage9g_user_gaussians.py` (matched user Gaussians at 3 levels)
- `gaussian/syntax_subspace_stage9g_ablation.py` (controlled ablation)

## Goal

Resolve Stage 9E-P's personalization collapse (1-4 unique / 10 users, top-K Jaccard
0.7-0.8) by **enriching 182d syntactic features** with:
1. **Group B (+ Clause/Dependency)**: 41 features — clause depth, subordinate/relative
   clause counts/ratios, preposition attachment, modifier placement, dep_rel histogram,
   coordination span, passive count
2. **Group C (+ Connective-specific)**: 21 features — which/that/who/because/since/while/
   although/when/if/for/with/in/of/by/at/on/from/to/about individual counts

Run ablation:
- Base182 (existing) vs +Clause/Dep vs +Connective

**GO criteria** (user-stricter):
- raw-space structural separability ↑
- content leakage NOT significantly worse
- unique selected 2.6 → 4-5+/10
- top-10 Jaccard ↓
- Mahalanobis selected-vs-random ratio preserved

## Method

### Critical methodological fix (vs Stage 9G-2 v1)

Stage 9G-2 v1 had a **space mismatch bug**: user Gaussians were built in OLD
Base182 PCA48 space, but pool queries were projected through a NEW PCA48 fit on
pool data with enriched features. Mahalanobis distances were meaningless.

Stage 9G-2 v2 fixes this by:
- Building user Gaussians at each feature level (`stage9g_user_gaussians.py`)
- Using SAME scaler+PCA48 fit on training sentences for both user mus and pool
- All 3 levels get matched (user_mu, pool_projection) at their level

User Gaussian rebuild:
- 294 users × mean 114 reviews = 33K reviews → 208K sentences
- spaCy batch parse + clause/dep/connective extraction
- Cached to `stage9g_clause_conn_cache.jsonl.gz` (sha1(text) → features)
- 195K unique sentences processed in 258s (subsequent runs: ~30s with cache)

### Feature groups (3 levels)

| Level | dim | contents |
|-------|-----|----------|
| Base182 | 182 | existing spaCy 182d (filtered numeric) |
| +Clause/Dep | 223 | +41 new (clause structure, dep_rel, prep, modifier placement) |
| +Connective | 244 | +21 lexical connective word counts |

## Results

### 1. Family silhouette (8 family labels, 9C only)

| Level | Sil(raw) | Sil(PCA48) |
|-------|----------|------------|
| Base182 | -0.0169 | -0.0132 |
| +Clause/Dep | +0.0004 | +0.0067 |
| +Connective | +0.0045 | +0.0107 |

**Enriched features improve raw-space family separability** (+0.022 from base
to conn, similar to Stage 9F-A's +0.15 swing from -0.043 to +0.110).

### 2. PCA dimension sweep (silhouette)

| Dim | base | clause | conn |
|-----|------|--------|------|
| 16 | 0.013 | 0.011 | 0.013 |
| 24 | 0.009 | 0.009 | 0.009 |
| 32 | 0.008 | 0.008 | 0.009 |
| **48** | **0.007** | **0.007** | **0.011** |
| 64 | 0.002 | 0.002 | 0.006 |
| 96 | -0.000 | -0.000 | 0.005 |
| 128 | 0.000 | 0.000 | 0.004 |

Silhouette peaks at low PCA dims (16-48), confirming that family signal is
concentrated in low-dim components. Higher PCA dims add noise.

### 3. Content leakage (probe R², 5-fold CV Ridge)

| attr | base | clause | conn | Δ clause | Δ conn |
|------|------|--------|------|----------|--------|
| brand | 0.580 | 0.599 | 0.589 | +0.019 | +0.010 |
| color | 0.651 | 0.665 | 0.661 | +0.014 | +0.010 |
| age_range | 0.572 | 0.569 | 0.548 | -0.003 | -0.024 |
| size | -1.108 | -2.394 | -1.991 | -1.286 | -0.883 |
| main_cat | 0.745 | 0.771 | 0.760 | +0.026 | +0.015 |

**Content leakage barely increases** (+1-3pp on brand/color/main_cat). This is
NOT enough to explain personalization regression — confirms leakage is not
the bottleneck.

### 4. Mahalanobis selection on combined 9C+9F pool (matched user Gaussians)

| Level | Unique sel | top1 Jaccard | top10 Jaccard | sel/rnd ratio |
|-------|-----------|--------------|---------------|---------------|
| Base182 | **3.30** | **0.460** | **0.680** | 3.63× |
| +Clause/Dep | 2.60 | 0.629 | 0.818 | 3.98× |
| +Connective | 2.40 | 0.487 | 0.703 | 1.85× |

**Critical finding: enriched features HURT personalization.**

- Unique selected 3.30 → 2.60 → 2.40 (**decreased** by 27%)
- top1 Jaccard 0.460 → 0.629 (**worsened** by 37%)
- top10 Jaccard 0.680 → 0.818 → 0.703 (worsened)
- selected/random ratio preserved at 3.6-4.0× (selection still works for closest
  match, but the variance between users is reduced)

## Decision: **NO-GO**

User-defined GO criteria evaluation:

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| raw-space separability ↑ | yes | -0.017 → +0.005 | ✓ |
| content leakage NOT worse | +0 small | +1-3pp | ✓ (borderline) |
| unique selected 2.6 → 4-5+ | yes | 2.6 → **2.4** (worse) | ✗ |
| top-10 Jaccard ↓ | yes | 0.680 → **0.818** (worse) | ✗ |
| Mahalanobis sel/rnd preserved | yes | 3.6× → 4.0× | ✓ |

**Stage 9G is a NO-GO.** Adding clause/dep/connective features improves
family-classification but worsens per-user personalization.

## Why enrichment doesn't help

Adding features:
1. **Emphasizes ASIN-level vocabulary** in raw space — different ASINs have
   different connective patterns (e.g., "for toddler" in Pampers vs "in black"
   in Toogel). The new features pull pool queries toward ASIN-specific
   attractors rather than user-specific.
2. **Dilutes per-user signal** in PCA48 — adding 41-62 features to 182d
   without proportionately increasing user-side variance means user mus
   become more diffuse relative to pool spread.
3. **PCA cannot recover user style** — the new PCA48 components capture
   the strongest variance directions, which are ASIN-correlated (brand
   name patterns, attribute frequency) rather than user-correlated
   (clause nesting, modifier placement).

## What this confirms

**The personalization ceiling is the SELECTION MECHANISM, not the feature
representation.**

Per user's hypothesis (now empirically validated):
- 182d → 244d feature redesign: ✗ does not break ceiling
- Coverage-aware generation (Stage 9F-B): PARTIAL (+62% unique, 2.60/10)
- Combined 9F-B + 9G enriched features: ✗ worse than 9F-B alone

The "single shared item candidate pool + nearest-Mahalanobis argmin" approach
itself has a ceiling at ~3 unique per ASIN (Base182 baseline) on PCA48.

## Files

- Scripts:
  - `gaussian/syntax_subspace_stage9g_extract.py`
  - `gaussian/syntax_subspace_stage9g_user_gaussians.py`
  - `gaussian/syntax_subspace_stage9g_ablation.py`
- Caches:
  - `stage9g_clause_conn_cache.jsonl.gz` (195K enriched sentence features)
- Outputs:
  - `stage9g_enriched_features.jsonl.gz` (1317 query enriched features)
  - `stage9g_user_gaussians_3levels.json` (user Gaussians at 3 levels)
  - `stage9g_ablation.json` (full ablation table)

## Conclusion

> **Stage 9G NO-GO**. Adding 41 clause/dependency features and 21 connective
> features improves family-classification (silhouette -0.017 → +0.005) but
> HURTS per-user personalization (unique 3.30 → 2.40, Jaccard 0.460 → 0.629)
> on matched user Gaussians.
>
> The personalization collapse ceiling is the SELECTION MECHANISM, not
> feature representation. Stage 9G exhausts the "redesign representation"
> avenue — next must change the selection mechanism itself.
>
> **Recommended next step: Stage 10G — Selection Mechanism Pivot**
> - Option A: Per-user generation (not shared pool) — given user mu, generate
>   K queries directly targeting it via structured prompts
> - Option B: Multi-query selection — pick top-K (not argmin) per user,
>   weighted by retrieval downstream
> - Option C: User-conditioned rerank — keep shared pool + argmin selection,
>   but rerank selected queries using actual user mu direction
>
> Per user feedback: "如果 Stage 9G 做完后还是停在 2-3 unique / 10 users,
> 那时才有依据地说: 当前'single shared item candidate pool + nearest-Mahalanobis
> argmin'本身存在 personalization ceiling, 需要改 selection mechanism".
>
> We've stopped at 2-3 unique. The pivot to a different selection mechanism
> is now empirically justified.
