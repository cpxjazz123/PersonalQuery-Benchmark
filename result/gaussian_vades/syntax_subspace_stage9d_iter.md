# Stage 9D — Mahalanobis Rerank on Stage 9C Clean Pool + Retrieval (MIXED)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9d_select_retrieve.py`
**Scope**: 10 ASINs × 10 users = 100 (asin, user) pairs × 3 variants = 300 retrievals

## Goal

Verify that Stage 9C's clean + compositionally-rich pool (786 queries, 8 grammatical-relation families)
yields better retrieval robustness when combined with Stage 8.5 Mahalanobis per-user selection.

## Design

**Reuse Stage 8.5 infrastructure**:
- User Gaussians: `stage8_5_user_gaussians.json` (PCA48 + diagonal Maha + λ=0.1 shrinkage)
- PCA48: `gaussian_vades._syntax_subspace_prepare()` (frozen scaler + 48 components)

**Stage 9C clean pool**: 786 queries across 10 ASINs, all `strict=True` after QA panel
(0% extra-sem / template / field / dup / attr-stack / no-connector).

**Per (asin, user)**:
- `selected`: query with min Mahalanobis distance to user Gaussian
- `random`: deterministic random pick seeded by user_id hash
- `farthest`: query with max distance (negative control)

**Retrieval**: bm25s k=all (217K ASIN corpus) + MiniLM GPU cosine sim.

## Results

### Layer 1: Selection Validation (PCA48 Mahalanobis space)

| Variant | mean distance | median | Wilcoxon p | % beat |
|---------|---------------|--------|------------|--------|
| **selected** | **30.5** | 20.9 | — | — |
| random | 185.5 | 85.0 | random > selected: **p = 4.17e-18** | 98% selected < random |
| farthest | 880.8 | 497.9 | farthest > selected: **p = 1.95e-18** | 100% selected < farthest |

**Mahalanobis selection works perfectly** in syntactic space:
- Selected is **6.1x closer** to user Gaussian than random pick
- Selected is **28.9x closer** than farthest
- 98% of selections beat random; 100% beat farthest

### Layer 2: Retrieval (mixed signal — important finding)

| Variant | bm25_rank | bm25_hit@10 | minilm_rank | minilm_hit@10 | minilm_MRR |
|---------|-----------|-------------|-------------|---------------|------------|
| **selected** | 9935.82 | 30.0% | 10404.94 | **20.0%** ✓ | 3.897% |
| random | 9730.94 | **37.0%** ✓ | **9165.49** ✓ | 14.0% | **8.499%** ✓ |
| farthest | **4596.76** ✓ | 35.0% | 13432.54 | 13.0% | 2.873% |

**Surprises**:

1. ✅ **MiniLM hit@10**: selected 20% beats random 14% (selection helps top-10 hits)
2. ❌ **MiniLM mean_rank**: selected 10404 WORSE than random 9165 (long-tail outliers)
3. ❌ **MiniLM MRR**: random 8.5% beats selected 3.9% (2.2x worse)
4. ❌ **BM25 hit@10**: random 37% best, selected 30% worst
5. ⚠️ **Farthest BM25 rank 4596 = BEST** (counter-intuitive — farthest query has strongest BM25 match)

## Interpretation

**The Mahalanobis selection is correct in PCA48 space but does NOT translate to better retrieval.**

### Why selection hurts mean_rank / MRR

- PCA48 distance measures "how user-style-like" a query is syntactically
- But retrieval is dominated by **lexical overlap** (BM25) and **semantic embedding similarity** (MiniLM)
- 9C pool queries with same 4 attrs have ~same lexical content; only grammatical structure varies
- The queries MOST user-style-like (high relative-clause density, many prep phrases) tend to:
  - Use more function words → less lexical density on attribute nouns
  - Embed attributes inside subordinate clauses → weaker BM25 term frequency
  - Use different word order (modifier-fronted) → weaker MiniLM cosine

### Why farthest has best BM25 rank

- 9C's "farthest" queries are typically:
  - modifier_fronted (15 queries, 1.60 modifier/query) — front attributes pre-modifying head noun
  - question (157 queries) — interrogative form uses attributes as noun phrases
- These naturally have **higher BM25 term frequency** on attribute words (no clause dispersion)
- BM25 rewards lexical density, not syntactic richness

### Why MiniLM hit@10 improves but mean_rank worsens

- Selection narrows the candidate → if the right query is "near", it gets picked
- But the right query is rarely "near" in PCA48; mostly the picked query is "near" but wrong target
- Mean_rank gets pulled down by long-tail failures where picked query is "near user" but "far from target"

## ⚠️ CRITICAL CAVEAT — Personalization Collapse (added 2026-08-25, post-volatility inspection)

**The "selected" queries above are NOT actually per-user personalized.** Direct inspection shows:

- **B09KLYL2ZX (Pampers White Toddler)**: 8/10 users → same `prepositional` query, 2/10 → same `coordination` query → **only 2 unique selected queries**
- **B09S8PT9L6 (Summer Infant 4-Sided Pad)**: 10/10 users → **same `modifier_fronted` query** → 1 unique selected
- **B0BMQ3G124 (Mama Bear White Infant)**: 10/10 users → **same `fragment_with_connector` query** → 1 unique selected

This means PCA48 Mahalanobis selection is **degenerate to global pool minimum** for most ASINs — all users
within an ASIN get the SAME query. The 30.5 mean Maha distance vs random 185.5 is the *pool minimum*,
not a per-user minimum.

**Implication for the Stage 9D-Volatility conclusion**: The 0% Hit@5 flip rate cannot be interpreted as
"Retriever is robust to user-expression variation" — it is because **all 10 users within an ASIN use
the same query**, so there is no expression variation to flip across.

**Root cause hypotheses**:
1. **Pool too small**: 41-196 queries per ASIN may not span enough PCA48 space to differentiate users
2. **User Gaussians too similar**: users buying the same product have similar syntactic profiles
3. **Pool PCA48 distribution has one dominant cluster**: argmin finds it for everyone

**What this changes**:
- Stage 9D Maha validation 100% pass is *expected* when selection is degenerate (everyone picks the same query)
- Stage 9D retrieval "selected vs random" comparison is really *one-query-per-ASIN vs random* comparison
- Stage 9D-Volatility 0% flip is a **degenerate artifact of selection collapse**, not evidence of retriever robustness

**Decision**: Cannot conclude anything about retriever robustness or per-user personalization from this data.
Must first run Stage 9E-P **Personalization Separability Diagnostic** to determine whether the collapse
is due to (a) insufficient pool coverage, (b) indistinguishable user Gaussians, or (c) pool distribution shape.

## Comparison with Stage 8.5 baseline

Need to compare against Stage 8.5 retrieval summary (computed earlier on different pool but same users).

## Conclusion

> **Stage 9D finding (revised after collapse caveat)**:
> 1. PCA48 Mahalanobis selection validates in syntactic space (p<1e-18) — but is **degenerate to global pool minimum**
>    for most ASINs (1-2 unique queries per ASIN, not 10).
> 2. Retrieval comparison selected vs random is *single-query-per-ASIN vs random*, NOT per-user personalization.
> 3. Stage 9D-Volatility 0% flip is **artifact of selection collapse** (all users share same query), not retriever robustness.
>
> **Cannot conclude** anything about per-user personalization or retriever robustness from this experiment.
>
> **Next** (mandatory): **Stage 9E-P — Personalization Separability Diagnostic** to determine root cause
> (insufficient pool coverage vs indistinguishable user Gaussians vs pool distribution shape).
> After diagnosis, decide whether to scale pool, redesign user style model, or use a different selection objective.

## Files

- Script: `gaussian/syntax_subspace_stage9d_select_retrieve.py`
- Selection: `scratch2/.../stage9d_selection.json`
- Selection stats: `scratch2/.../stage9d_selection_stats.json`
- Retrieval per-query: `scratch2/.../stage9d_retrieval_per_query.json`
- Retrieval summary: `scratch2/.../stage9d_retrieval_summary.json`
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9d_iter.md`