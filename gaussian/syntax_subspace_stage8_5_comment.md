# Stage 8.5 Results — Mahalanobis User Selection

**Date**: 2026-08-25
**Status**: Pipeline complete through retrieval + analysis

## 1. Motivation

Stage 8 measured **general syntactic variation** — all strict variants per (asin, user) sent to retrieval independently, with NO user-specific selection. This tests "does query syntax affect retrieval?" but NOT "does user-matched syntax affect retrieval more than syntax alone?"

Stage 8.5 implements the formal personalization method:
1. Generate a SHARED candidate pool per ASIN (K=50, attrs-only prompt)
2. Build per-user Gaussian from Baby_Products review texts
3. For each (asin, user), select query with minimum Mahalanobis distance to user Gaussian
4. Retrieve selected/random/farthest queries → compare MRR/Hit@10

## 2. Pipeline (8 steps)

| Step | Script | Output |
|------|--------|--------|
| 1 | `syntax_subspace_stage8_5_user_gaussians.py` | `stage8_5_user_gaussians.json` |
| 2 | `syntax_subspace_stage8_5_regen.py` | `stage8_5_pool.json` (5000 queries) |
| 3 | `syntax_subspace_stage8_5_features.py` | extends `stage7b_query_features.jsonl.gz` |
| 4 | `syntax_subspace_stage8_5_select.py` | `stage8_5_selection.json` + `_stats.json` |
| 5 | `syntax_subspace_stage8_5_retrieval.py` | `stage8_5_retrieval_*.json` |
| 6 | `syntax_subspace_stage8_5_analyze.py` | `stage8_5_analyze.json` |

## 3. Pre-retrieval Validation (selection effectiveness)

966/1000 users have per-user Gaussian (695 high-confidence + 271 fallback), 34 use ASIN centroid.

| Comparison | Mean diff | Wilcoxon W | p-value |
|-----------|-----------|------------|---------|
| Selected vs Random | -376 | 0.0 | 5.4e-151 |
| Selected vs Farthest | -1800 | 0.0 | 5.5e-159 |
| Random vs Farthest | -1424 | 0.0 | 1.9e-152 |

**Per-source** (selected < random):
- per_user (n=695): mean diff -507, 91.4% sel<rnd
- global_var_fallback (n=271): mean diff -84, 90.4% sel<rnd
- asin_centroid_fallback (n=34): mean diff -24, 91.2% sel<rnd

**Conclusion**: Mahalanobis selection produces a query significantly closer to user style than random/farthest. Selection works as designed.

## 4. Retrieval (selected vs random vs farthest)

[To be filled after retrieval completes]

## 5. Interpretation

[To be filled]

---

## Files

- Scripts: `/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian/syntax_subspace_stage8_5_*.py`
- Results: `/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_*.json`