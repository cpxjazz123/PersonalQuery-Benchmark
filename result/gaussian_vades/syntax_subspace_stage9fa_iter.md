# Stage 9F-A — 182d vs PCA48 Family Separability (FEATURE BOTTLENECK)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9fa_separability.py`
**Scope**: 786 9C queries, 8 grammatical families

## Goal

Locate where 8 grammatical families lose diversity:
- **Option (a)**: 182d raw spaCy features are already indistinguishable across families → feature bottleneck
- **Option (b)**: 182d is separable but PCA48 collapses families → PCA bottleneck

## Results

### Silhouette scores (family-as-cluster, on different spaces)

| Space | silhouette | interpretation |
|-------|-----------|----------------|
| **182d raw** | **-0.0429** (NEGATIVE) | family labels predict distance structure WORSE than random |
| PCA10 | 0.0633 | low |
| PCA20 | 0.1096 | moderate |
| **PCA30 (peak)** | **0.1157** | highest separation |
| **PCA48 (current)** | **0.1104** | moderate |
| PCA64 | 0.1091 | moderate |
| PCA96 | 0.1080 | moderate |
| PCA128 | 0.1071 | moderate |
| PCA150 | 0.1058 | moderate |
| PCA182 (full scaled) | 0.1060 | moderate |

**Surprising finding**: 182d raw silhouette is **negative**, meaning the 8 family labels are
**anti-correlated** with intra/inter distance structure. PCA actually *improves* separation
from -0.043 → +0.110 (a +0.15 swing) by picking up the strongest variance direction.

### Intra-family vs inter-family mean L2

**182d raw**:
| Family | intra L2 | inter (pointwise) L2 |
|--------|----------|---------------------|
| coordination | 12.48 | 14.67 |
| fragment_with_connector | 10.32 | 14.49 |
| modifier_fronted | 7.17 | 13.44 |
| predicate_based | 11.93 | 13.18 |
| prepositional | 10.89 | 13.16 |
| question | 10.81 | 13.22 |
| relative_clause | 13.01 | 15.68 |
| subordinate_clause | 12.05 | 13.25 |
| **mean** | **11.08** | **13.89** |

Ratio (inter/intra) = 13.89/11.08 = **1.253**

**PCA48**:
| Family | intra L2 | inter (pointwise) L2 |
|--------|----------|---------------------|
| coordination | 11.08 | 16.75 |
| fragment_with_connector | 8.38 | 15.35 |
| modifier_fronted | 6.53 | 15.29 |
| predicate_based | 12.28 | 17.38 |
| prepositional | 9.64 | 15.84 |
| question | 9.20 | **28.79** ← outlier |
| relative_clause | 12.67 | 18.89 |
| subordinate_clause | 12.05 | 17.41 |
| **mean** | **10.23** | **17.21** |

Ratio (inter/intra) = 17.21/10.23 = **1.781** (preserved +42.1% from 1.253)

### Question family is the outlier

In PCA48, `question` family has inter-family L2 = 28.79 vs mean 17.21 — it sits in a distinct region
of PCA48 space (interrogative form with `?`). This is why some ASINs (B09S8PT9L6 etc.) saw
question-family queries as "farthest" — they're in a separate PCA48 cluster.

## Decision: **Option (a) — Feature Bottleneck**

```
Decision: Feature bottleneck (Option a): 182d raw features are already indistinguishable
silhouette 182d = -0.04285124422137296
silhouette PCA48 = 0.11039251952386231
separation ratio preserved: 142.1%
```

The 182d spaCy features **cannot distinguish 8 grammatical families** in raw space:
- Negative silhouette means queries of the same family are NOT closer to each other than queries
  of different families in 182d
- LLM-generated queries labeled with the same family have heterogeneous 182d feature profiles
- The 8 family labels are essentially **decorative annotations** in 182d feature space

**PCA actually helps (not hurts)**:
- PCA projects 182d → 48d while preserving the strongest variance direction
- This *exposes* whatever weak family signal exists in the data
- Going from 182d → 48d improves silhouette by 0.15
- Going from 48d → 182d (no PCA) gives silhouette 0.106 — virtually the same as 48d

**PCA dimension doesn't matter much**:
- Silhouette peaks at PCA30 (0.116), then plateaus around 0.105-0.110 from PCA48 to PCA182
- Increasing PCA dim beyond 48 does NOT improve separability
- **Higher PCA dim does NOT solve the collapse**

## Implications

### For Stage 9D personalization collapse

Stage 9D selection collapse (1-4 unique selected per ASIN) is caused by:
1. **Pool queries cluster in 1-2 PCA48 modes** (Check 2 of 9E-P, Jaccard K=10 = 0.80)
2. **NOT because PCA loses family info** — PCA actually gains family info
3. **NOT because pool is too small** — adding more same-distribution queries won't help
4. **BECAUSE** LLM-generated 8-family queries **don't actually occupy 8 distinct 182d regions**

The pool has 8 family labels but the 182d features only see ~1-2 modes. This is because:
- `prepositional` query "Pampers in white for toddler under Health" has 3 prepositions
- `relative_clause` query "Pampers that is white, which is designed for toddlers" has clause + 2 preps + 1 coord
- Both have similar function-word density and similar syntactic tree depth
- spaCy 182d features don't encode the **specific connective word used** (in vs which vs that)

### For the system

**Cannot fix by**:
- Increasing PCA dim (PCA30 → PCA182: same silhouette)
- Using non-linear embedding on same features (won't create cluster that doesn't exist)
- Scaling pool (same distribution, same cluster)

**Can fix by**:
- **Coverage-aware generation**: generate queries that explicitly target different 182d regions
  (e.g., queries that vary clause depth, sub-clause nesting, modifier placement)
- **Better 182d features**: add features for specific connective words, clause depth, sub-clause count
- **Per-family constraint enforcement**: when generating family-X queries, require feature profile
  matching family-X prototype

## Files

- Script: `gaussian/syntax_subspace_stage9fa_separability.py`
- 182d separability: `scratch2/.../stage9fa_182d_separability.json`
- PCA48 separability: `scratch2/.../stage9fa_pca48_separability.json`
- Summary: `scratch2/.../stage9fa_summary.json`
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9fa_iter.md`

## Conclusion

> **Stage 9F-A confirms Feature Bottleneck (Option a)**. The 8 grammatical families are NOT
> distinguishable in 182d raw feature space (silhouette = -0.043). PCA actually *helps* a bit
> (silhouette = 0.110 at PCA48, peaks at PCA30 = 0.116) by picking up the strongest signal.
> Going higher than PCA48 does not improve separation.
>
> **Cannot fix by**:
> - PCA dim increase (PCA48 → PCA182: same silhouette 0.10-0.11)
> - Non-linear embedding on same 182d features
> - Pool scaling (queries cluster in same 1-2 modes regardless of count)
>
> **Can fix by** (Stage 9F-B):
> - Coverage-aware candidate generation targeting different 182d regions
> - Add connective-word-specific / clause-depth features to 182d
> - Per-family generation constraint enforcement
>
> Recommended next: **Stage 9F-B Coverage-aware Generation** — generate queries that explicitly
> vary features the 182d can capture (sub-clause depth, modifier count, function-word density),
> then re-run Stage 9E-P to verify unique selected and top-K Jaccard both improve.