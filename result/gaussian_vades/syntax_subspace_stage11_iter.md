# Stage 11 — Content Lock + Structural Class + Selection: PROTOTYPE GO

**Date**: 2026-08-26
**Script**: `gaussian/syntax_subspace_stage11_content_locked.py`
**Outputs**:
- `hj82_scratch2/.../stage11_structural_class.json`
- `hj82_scratch2/.../stage11_content_locked_pool.json`
- `hj82_scratch2/.../stage11_comparison.json`

## Research Question

> Can "content locked + structural conditioned" generation diversify
> the syntactic variation in the pool, and does A1 reject-repeat still
> hit the 10/10 ceiling on this new pool while preserving personalization?

This is the prototype for the user's framework:
**固定 Content + 控制 Function/Structure → Syntactic Variation**

## Method

### Step 1 — Structural class detector (regex, no LLM)

Pure heuristic classifier, no spaCy dependency. 7 classes:

| Class | Trigger |
|-------|---------|
| QUESTION | ends with `?` |
| RELATIVE_CLAUSE | `that/which/who` + finite verb |
| COORDINATION | `and/but/or` joining NPs |
| PREPOSITIONAL | ends with `in/for/within/under <NP>` |
| MODIFIER_FRONTED | starts with `For/With/In/To/As <X>,` |
| PREDICATE | copula `is/are + a/an/the/designed` |
| SIMPLE | none of the above |

### Step 2 — Content-locked generation (560 queries)

For top-20 ASINs by pool size:
- Extract 4 attrs (brand / color / age / category) from existing pool
  queries via regex (with fallback to placeholder for missing attrs)
- One vLLM batched call per ASIN: **7 structural classes × K=4 = 28 prompts**
  → 28 outputs per ASIN → 560 candidates total
- System-level: explicit blacklist (durable, lightweight, comfortable,
  perfect, modern, stylish, premium, high-quality, top-quality, quality,
  ideal, great, excellent, best, amazing, "for baby", "for your baby")
- Variant suffix `(variant N)` added to avoid vLLM dedup

### Step 3 — Selection comparison

Per ASIN (with ≥2 users in Stage 8.5 selection):
1. Project 28 content-locked queries to 48d PCA48 via scaler + PCA
2. Compute D matrix [Q=28, U=10] with diagonal Mahalanobis
3. Run **mean_t50** (baseline) and **A1 reject-repeat** selection
4. Compare: unique / 10 users, class diversity, sel_dist / rnd_dist

## Results

### Pool-level structural diversity

| Pool | Mean classes/asin | Range |
|------|------------------:|-------|
| Original Stage 8.5 (5000 queries) | **2.39** | [1, 2, 3, 4] |
| Content-locked Stage 11 (560 queries) | **5.85** | [5, 6, 7] |

**+145% structural diversity** — content-locked generation actually
delivers the 7-class variation we wanted.

### Per-class hit rate (predicted_class vs requested class)

| Class | hits / 80 | rate |
|-------|----------:|-----:|
| SIMPLE | 80 / 80 | **100.0%** |
| RELATIVE_CLAUSE | 75 / 80 | 93.8% |
| MODIFIER_FRONTED | 53 / 80 | 66.2% |
| PREDICATE | 52 / 80 | 65.0% |
| QUESTION | 50 / 80 | 62.5% |
| COORDINATION | 34 / 80 | 42.5% |
| PREPOSITIONAL | 1 / 80 | **1.2%** ← structural prompt failure |

LLM is great at SIMPLE / RELATIVE_CLAUSE; struggles with PREPOSITIONAL
(1.2% — almost always picks "with" instead of "in <category>").

### Generation quality

| Metric | Value |
|--------|------:|
| non-empty outputs | 560 / 560 (100.0%) |
| content_ok (brand+color+age in text) | 379 / 560 (67.7%) |
| no-bias (clean of blacklist) | 361 / 560 (64.5%) |
| bias hits total | 199 / 560 (35.5%) |

**Top leakages**: `best` (101), `for baby` (80), `quality` (27),
`ideal` (11), `stylish` (11), `high-quality` (10)

The blacklist catches "for baby" but the LLM still says "best" once per
prompt on average. Need stronger anti-bias phrasing in prompts.

### Selection metrics (combined pool, content-locked only)

| Metric | Value |
|--------|------:|
| n ASINs with ≥2 users | 20 |
| **unique_mean_t50 mean** | 5.65 |
| **unique_a1 mean** | **10.00** ⭐ |
| unique_a1 ceiling | 100.0% |
| **class_diversity_a1** (selected queries span this many classes) | 4.60 |
| sel_dist_a1 mean | 1513.43 |
| rnd_dist mean | 3321.71 |
| sel/rnd ratio | 0.456 |

**A1 still hits 10/10 ceiling**, and selected queries span **4.6 of 7**
structural classes on average (vs original Stage 8.5 selection which
spans ~2 classes/asin). The content-locked pool IS producing
structurally-distinct selections per user.

sel/rnd = 0.456 (Stage 10K was 0.47) — **personalization preserved**.

## Decision: **PROTOTYPE GO** ⭐

The user's framework — **content lock + structural condition + post-parse
verification** — works at prototype scale. Specifically:

1. **Pool diversification**: structural classes/asin 2.39 → 5.85 (+145%)
2. **Selection ceiling holds**: A1 reject-repeat still produces 10/10 unique
3. **Class-aware selection**: selected queries span 4.6 classes on average
4. **Personalization preserved**: sel/rnd = 0.456 (vs 0.47 Stage 10K)

## What works / what doesn't

### ✅ Works

- 7-class prompt diversification for 5/7 classes (RELATIVE_CLAUSE
  94%, MODIFIER_FRONTED 66%, PREDICATE 65%, QUESTION 63%, COORDINATION 43%)
- Content-locked attribute extraction + fallback
- vLLM batched generation (28 prompts per ASIN in ~1.2s)
- A1 reject-repeat on combined pool → 10/10 unique

### ⚠️ Needs improvement

- **PREPOSITIONAL**: 1.2% — LLM does not obey "ends with `in <category>`"
  pattern. The few that did try, used natural PP like "with my baby" not
  the formal category slot.
- **Bias leaks**: 35.5% contain blacklist terms. Top offenders are
  `best` (101) and `for baby` (80) — too common in Baby_Products domain.
  Need to ban in the prompt with **stronger phrasing** ("NEVER use the
  word best") and post-generation regex filter.
- **Content coverage**: 67.7% — `colored` placeholder appears when real
  color isn't extractable; need to skip those or pick a real category
  color from the asin metadata.
- **Pool size**: only 28 candidates per ASIN is small. Stage 8.5 has 50.
  Need K=8 per class × 7 classes = 56 cands to match Stage 8.5 pool.

## Sample outputs

```
=== B09KLYL2ZX: ('Pampers', 'white', 'toddler', 'Health & Personal Care') ===
[     PREPOSITIONAL] "Find Pampers white toddler product options in Health & Personal Care"
[      COORDINATION] "Pampers White Toddler Diapers and Rash Relief Cream"
[   RELATIVE_CLAUSE] "What are the features of Pampers white toddler products in the Health & Personal Care category?"
[  MODIFIER_FRONTED] "For toddler needs in Health & Personal Care, explore Pampers white products."
[         PREDICATE] "A Pampers product is specifically designed for toddlers."
[          QUESTION] "What types of products does the Pampers brand offer for toddlers?"
[            SIMPLE] "Pampers white toddler diapers"
```

## Paper-ready framing

> We prototype a "content locked + structural conditioned" generation
> pipeline. For each product, we fix four core attributes (brand, color,
> age range, main category) and generate queries conditioned on seven
> structural classes (relative clause, coordination, modifier-fronted,
> predicate, question, prepositional, simple). The pipeline produces
> queries that span 5.85 distinct structural classes per ASIN (vs 2.39
> in the unconstrained baseline pool), and the per-user A1 reject-repeat
> selection still hits the 10/10 uniqueness ceiling while spanning 4.6
> classes on average. Selection sel/rnd = 0.456 preserves personalization.
>
> Two classes (PREPOSITIONAL 1.2%, COORDINATION 42.5%) underperform and
> require prompt engineering. Bias-leakage is 35.5% (mostly `best` and
> `for baby`), suggesting that prompt-side suppression is necessary but
> not sufficient — a post-generation regex filter is needed.

## Files

- Script: `gaussian/syntax_subspace_stage11_content_locked.py`
- Outputs (scratch2):
  - `stage11_structural_class.json` — per-query class labels for 5000 Stage 8.5 pool
  - `stage11_content_locked_pool.json` — 560 generated candidates (20 ASINs × 28)
  - `stage11_comparison.json` — selection metrics vs original pool

## Next Steps

1. **Improve PREPOSITIONAL prompt**: ask LLM to literally end with
   "in Health & Personal Care" / "in Pet Supplies" etc. (today it's 1.2%).
2. **Stronger anti-bias**: prompt "NEVER use 'best', 'perfect', 'ideal'"
   + post-gen regex filter (drop or rewrite leaked queries).
3. **Scale up**: K=8 per class (56 cands/ASIN) on all 100 ASINs.
4. **Pair with retrieval evaluation**: do these structurally-diverse
   selections actually retrieve better? Run MiniLM / BM25 hit@10
   on the content-locked pool.
5. **Spice attribute extraction**: use ASIN metadata (title) as a
   supplementary source for color/category when pool-extracted attrs
   fall back to placeholder.
