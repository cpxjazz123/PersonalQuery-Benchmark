# Stage 9F-B — Coverage-Aware Generation (PARTIAL-GO)

**Date**: 2026-08-25
**Scripts**:
- `gaussian/syntax_subspace_stage9fb_coverage_gen.py` (480 prompt → 543 accepted queries)
- `gaussian/syntax_subspace_stage9fb_extract_features.py` (spaCy 182d for new queries)
- `gaussian/syntax_subspace_stage9fb_revalidate.py` (Mahalanobis selection + 9E-P recheck)

## Goal

Resolve Stage 9E-P's personalization collapse (1-4 unique selected / 10 users,
top-10 Jaccard = 0.80) by **coverage-aware generation**: produce queries that
explicitly target different PCA48 regions, then re-run selection.

## Method

### 12 Feature-Extreme Profiles

Each profile is a structural instruction to LLM designed to push queries
toward different PCA48 directions:

| Profile | Target feature direction |
|---------|--------------------------|
| very_short_fragment | low tokens, no function words |
| very_long_compound | high clauses + prepositions + modifiers |
| heavy_prepositional | n_prep ≥ 4 |
| heavy_clause_nesting | n_clause ≥ 3 (relative + subordinate) |
| heavy_modifier_fronted | attributes placed before head noun |
| heavy_predicate_verb | n_verb ≥ 4 |
| heavy_coordination | "and", "as well as", "while also" |
| extreme_question | ends with `?` |
| heavy_negation | "not", "no", "except", "without" |
| heavy_comparative | "better", "best", "most", "rather than" |
| passive_voice | "is/are + past participle" |
| first_person_pronoun | "I", "my", "we" |

10 ASINs × 12 profiles × 8 variants = **960 prompts** (relaxed filter: kept all
queries that passed attr-coverage + no field-name leak).

### Filter

Two-stage filter:
1. **Soft strict filter**: must contain all 4 attrs, no field-name leak (brand:, color:, size:),
   no multiple `?`, no `;`, length 5-250 chars.
2. **Per-ASIN dedup** by lowercased query string.

Removed the regex-based profile_mismatch enforcement from Stage 9F-B v1
(LLM doesn't reliably hit exact word counts; the prompt's structural
instruction is what matters).

## Generation Results

| ASIN | 9C pool | 9F new pool | Combined | 9F growth % |
|------|---------|-------------|----------|-------------|
| B09KLYL2ZX | 75 | 48 | 123 | +64% |
| B07XM9DX9H | 44 | 21 | 65 | +48% |
| B0C54J5D2B | 66 | 46 | 112 | +70% |
| B0BQV3785S | 87 | 48 | 135 | +55% |
| B0BMQ3G124 | 98 | 74 | 172 | +76% |
| B09QK77RNW | 53 | 64 | 117 | +121% |
| B00OQCZAVW | 41 | 68 | 109 | +166% |
| B09YYZTK1W | 66 | 58 | 124 | +88% |
| B09S8PT9L6 | 196 | 80 | 276 | +41% |
| B07C2HRMRF | 60 | 38 | 98 | +63% |
| **Total** | **786** | **545** | **1331** | **+69%** |

## PCA48 Coverage Analysis

| ASIN | std_growth % | mean_dist_growth % | span_growth % |
|------|--------------|---------------------|---------------|
| B09KLYL2ZX | +2.9% | -5.1% | +28.7% |
| B07XM9DX9H | +2.1% | -6.6% | +17.4% |
| B0C54J5D2B | +7.2% | -3.9% | +38.2% |
| B0BQV3785S | -0.3% | -4.0% | +18.5% |
| B0BMQ3G124 | +2.6% | -2.9% | +28.3% |
| B09QK77RNW | -1.9% | -9.6% | +20.2% |
| B00OQCZAVW | +1.7% | -9.2% | +29.2% |
| B09YYZTK1W | +10.5% | +1.3% | +29.6% |
| B09S8PT9L6 | +3.6% | -1.1% | +31.2% |
| B07C2HRMRF | +4.6% | -4.2% | +29.3% |

**Key signal: span_growth +17% to +38%** — coverage queries DO reach new
PCA48 regions. The negative dist_growth reflects that 9F queries fill
**near-cluster gaps** rather than spanning the whole space (new queries
land in unoccupied regions between existing 9C clusters).

## Revalidation: Stage 9E-P CHECK 2 + CHECK 3

Combined 9C + 9F pool, re-run Mahalanobis selection, check uniqueness and
Jaccard.

### Selection method breakdown

- `mahal_min`: 100/100 (all per-user Gaussians available)
- `asin_fallback`: 0

### Selected source breakdown

- From 9C pool: 52 (52%)
- From 9F pool: **48 (48%)** ← coverage queries contribute significantly

### Distance validation (selected vs random)

| Method | Mean Mahalanobis distance |
|--------|---------------------------|
| **selected** | **26.48** |
| random | 260.73 |

**Selected / random ratio: 9.85× improvement** — Mahalanobis selection still works.

### Stage 9E-P CHECK 3 (selected uniqueness)

| Metric | 9C baseline | 9F-B combined | Δ |
|--------|-------------|---------------|---|
| unique (mean) | 1.60 | **2.60** | **+62%** ✓ |
| unique (median) | 1.0 | 2.5 | +1.5 |
| unique (min) | 1 | 1 | 0 |
| unique (max) | 4 | 5 | +1 |

### Stage 9E-P CHECK 2 (top-K candidate Jaccard)

| K | 9C baseline | 9F-B combined | Δ |
|---|-------------|---------------|---|
| 1 | 0.720 | **0.629** | **-9%** ✓ |
| 5 | - | 0.689 | - |
| 10 | 0.796 | **0.754** | **-5%** ✓ |

## Decision: **PARTIAL-GO**

### What's improved
1. **Unique selected 1.60 → 2.60** (+62%): coverage queries DO inject
   personalization variance, but only marginally
2. **Jaccard decreased at all K** (top1 -9%, top10 -5%): pool coverage
   expansion works as intended
3. **48% of selected queries now from 9F** — coverage pool is meaningful
4. **Selection quality preserved**: 9.85× selected vs random distance
5. **span_growth +17% to +38%** — actual PCA48 region coverage expansion

### What's NOT solved

User-defined GO target was:
- unique selected 1-4 → **6-8 / 10** ← **got 2.6, far short**
- top-K Jaccard K=10 0.80 → **0.3-0.4** ← **got 0.754, basically same**

The coverage queries fill **inter-cluster gaps** (span grew) but don't
**escape the 9C cluster modes** (mean pairwise distance decreased).
LLM-generated queries, even with explicit structural guidance, still
land near the same syntactic attractors.

### Why coverage-only isn't enough

**Stage 9F-A feature bottleneck is the real ceiling**:
- 182d raw silhouette = -0.043 (NEGATIVE): 8 family labels predict
  distance structure WORSE than random
- PCA48 silhouette = 0.110: weak but non-negative
- LLM with explicit prompts still produces queries that look like
  "lists with some function words" — the structural variation gets
  swallowed by the 182d feature extractor (which doesn't capture
  specific connective words, clause depth, sub-clause count)

Coverage-aware generation is **necessary but not sufficient**:
- ✓ Increases pool coverage (span +28%)
- ✗ Doesn't break out of LLM's "list + light function words" attractor
- ✗ Feature extractor can't see the structural difference between
  "Pampers in white for toddler under Health" (prepositional) and
  "Pampers that is white, which is for toddler" (relative clause)
  in 182d

## Files

- Scripts:
  - `gaussian/syntax_subspace_stage9fb_coverage_gen.py` (480 prompts)
  - `gaussian/syntax_subspace_stage9fb_extract_features.py` (spaCy 182d)
  - `gaussian/syntax_subspace_stage9fb_revalidate.py` (selection + diag)
- Pools:
  - `scratch2/.../stage9fb_pool.json` (545 new queries)
  - `scratch2/.../stage9fb_features.jsonl.gz` (new query features)
- Coverage stats: `scratch2/.../stage9fb_coverage_stats.json`
- Selection: `scratch2/.../stage9fb_selection_combined.json`
- Diagnostics: `scratch2/.../stage9fb_diagnostic.json`

## Conclusion

> **Stage 9F-B PARTIAL-GO**: coverage-aware generation improves personalization
> (+62% unique, -9% Jaccard) but doesn't break the 1-4 unique / 0.80 Jaccard
> ceiling. PCA48 Mahalanobis selection still works (9.85× selected vs random
> distance). 48% of selected queries now come from the 9F pool, confirming
> the coverage queries contribute.
>
> **The personalization collapse ceiling is the 182d feature bottleneck**
> (Stage 9F-A confirmed). Coverage queries can fill inter-cluster gaps but
> cannot escape LLM's "list + function words" syntactic attractor. To break
> the ceiling, need either:
>
> 1. **Better 182d features** that capture specific connective words
>    (which/that/in/because), clause depth, sub-clause count
> 2. **Joint pool + feature redesign**: use new features that distinguish
>    the 12 profiles structurally
> 3. **Accept the ceiling**: 2.60 unique / 10 with 0.629 top1 Jaccard is the
>    personalization headroom on 182d; downstream retrieval / rerank should
>    adapt accordingly
>
> **Recommended next step**: Stage 9G — Joint 182d feature redesign
> (add connective-word-specific features, clause-depth features) + repeat
> Stage 9E-P / 9F-A / 9F-B with enriched features. This is the only path
> that respects the "explicit syntactic feature → PCA → user Gaussian →
> Mahalanobis" core pipeline.
