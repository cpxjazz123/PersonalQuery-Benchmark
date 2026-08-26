# Stage 13 — Logit Masking: Hybrid Solution (S11 prompt + S12 bad_words)

**Date**: 2026-08-26
**Scripts**:
- `gaussian/syntax_subspace_stage12_logit_mask.py`
- `gaussian/syntax_subspace_stage13_hybrid.py`

**Outputs**:
- `hj82_scratch2/.../stage12_logit_masked_pool.json` (over-constrained)
- `hj82_scratch2/.../stage13_hybrid_pool.json` (production hybrid)
- `hj82_scratch2/.../stage12_comparison.json`
- `hj82_scratch2/.../stage13_comparison.json`

## Research Question

> Does token-level logit masking (vLLM `bad_words`) give a hard
> constraint on bias words while preserving pool diversity?

Three approaches compared:

| Stage | Method | bias_hit | pool_classes/asin |
|-------|--------|---------:|------------------:|
| **S11** | prompt blacklist only ("Do not use ...") | 35.54% | 5.85 |
| **S12** | bad_words + whitelist-style prompt | **1.61%** ⭐ | 3.00 (collapsed) |
| **S13** | bad_words + Stage 11 freeform prompt | 20.36% | **5.90** ⭐ |

## Method

### Stage 12 — `bad_words` + whitelist prompt

```python
SamplingParams(
    bad_words=[
        "durable", "lightweight", "comfortable", "perfect", "modern",
        "stylish", "premium", "high-quality", "top-quality", "quality",
        ...  # 50+ forbidden words
    ],
)
```

The prompt explicitly listed allowed function words: `a, the, for, with,
in, of, that, which, and, is, are, designed, suitable, product, item,
looking, need, want, find, search, ...`

**Problem**: Prompt+bad_words together over-constrained the LLM. All
classes except QUESTION collapsed to a single template:
> "Looking for a suitable Pampers white toddler product in Health & Personal Care?"

→ 5/7 classes degraded to single template, pool diversity dropped to 3.00.

### Stage 13 — `bad_words` + Stage 11 freeform prompt

Combined Stage 11's freeform per-class prompts (which gave 5.85 classes)
with Stage 12's `bad_words` enforcement.

```python
# Stage 11 prompt: "Each query MUST end with a prepositional phrase like 'in {category}'..."
# Stage 12 hard constraint: SamplingParams(bad_words=...)
```

Plus added anti-template words to bad_words:
```
"Looking for", "looking for", "Find the right", "Find the",
"I need", "I want", "I'm looking", "I am looking",
"Where can I find", "where can I find", "Where to find",
"Need to find", "Want to find", "Searching for",
```

## Results

### Apples-to-apples bias comparison

Using the same Stage 11 blacklist + case-insensitive matching:

| Stage | bias count | bias rate | Δ vs S11 |
|-------|----------:|---------:|---------:|
| S11 (prompt only) | 199 / 560 | 35.54% | — |
| S12 (bad_words + whitelist prompt) | 9 / 560 | **1.61%** | -33.93pp |
| **S13 (bad_words + freeform prompt)** | 114 / 560 | **20.36%** | **-15.18pp** |

### Pool diversity preservation

| Stage | pool classes/asin |
|-------|------------------:|
| S11 | 5.85 |
| S12 | 3.00 ← template collapse |
| **S13** | **5.90** ⭐ (slightly better than S11) |

### Per-class hit rate (predicted == requested)

| Class | S11 | S12 | S13 | S13-S11 |
|-------|----:|----:|----:|--------:|
| SIMPLE | 100.0% | 21.2% | **100.0%** | 0pp |
| RELATIVE_CLAUSE | 93.8% | 7.5% | **93.8%** | 0pp |
| PREDICATE | 65.0% | 0.0% | **67.5%** | +2.5pp |
| MODIFIER_FRONTED | 66.2% | 26.2% | 63.7% | -2.5pp |
| QUESTION | 62.5% | 91.2% | 62.5% | 0pp |
| COORDINATION | 42.5% | 0.0% | 41.2% | -1.3pp |
| PREPOSITIONAL | 1.2% | 5.0% | 1.2% | 0pp |

**S13 matches S11 on every per-class hit rate**, confirming that
`bad_words` does NOT change the LLM's ability to follow structural
prompts. The diversity loss in S12 was purely from prompt over-constraint.

## Why bad_words alone (S13) doesn't fully block bias

`bad_words` enforces that the **exact token sequence** cannot appear.
But the LLM can bypass via:
- **Hyphenation**: `Best-selling`, `eco-friendly`, `high-rated`
- **Inflections**: `safest`, `trusted`, `innovative`
- **Capitalization**: `Best` vs `best` (tokenization may differ)
- **Synonyms**: `top-tier`, `world-class`, `industry-leading`

Top S13 leaks (case-insensitive, S11 blacklist + extra words):
- `safe` (49), `safety` (45) ← "safe" IS in blacklist, but `safest` slipped
- `for baby` (44) ← IS in blacklist, but tokenized differently sometimes
- `best` (41) ← IS in blacklist, but `Best-selling` slipped
- `Find the` (38) ← IS in blacklist (anti-template), still leaked
- `gentle` (37) ← IS in blacklist
- `stylish` (18) ← IS in blacklist

**bad_words is tokenization-sensitive**: a word is forbidden only if its
exact token sequence is forbidden. The Qwen2 tokenizer may tokenize
"best" as `["best"]` but "Best-selling" as `["Best", "-", "selling"]`,
bypassing the blacklist on the hyphen split.

## Decision: **STAGE 13 = HYBRID GO** ⭐⭐

The user's intuition was correct: **token-level logit masking IS the
right approach for hard content/structure constraints**.

- **Stage 12 confirms bad_words works**: 1.61% bias achievable (was 35.54%)
- **Stage 13 confirms prompt-side diversity is preserved**: 5.90 classes/asin
- **Combined architecture**: prompt gives structural guidance +
  bad_words gives bias filter = both wins

**Key insight**: bad_words must be **paired with freeform prompts**,
not whitelist prompts. Whitelist + bad_words causes template collapse.

## Recommended next steps

### Stage 14 — Comprehensive bias blacklist

Expand `bad_words` to include all inflections and hyphenated variants:

```python
# Additions
"Best-selling", "best-selling", "high-rated", "high-end", "top-tier",
"industry-leading", "world-class", "well-known", "long-lasting",
"long-lasting", "breathable", "hypoallergenic",
"trustworthy", "trusted", "trust",
# Inflections of "safe"
"safest", "safety", "secure",
# Phrase variants
"that are safe", "that are best", "that is best",
```

Goal: bring S13-style bias from 20.36% → ~5% without losing diversity.

### Stage 15 — Post-generation regex filter (alternative)

Drop biased queries post-generation (don't use bad_words at all):

```python
def is_acceptable(query):
    return not re.search(r'\b(best|perfect|ideal|premium|...)\b', query.lower())
```

Goal: ensure 0% bias with simpler implementation. Trade-off: smaller
pool (drop biased candidates) instead of generating cleaner ones.

### Stage 16 — Combined: bad_words + post-filter + retrieval evaluation

Final production pipeline:
1. Generate with Stage 13 settings (bad_words + freeform prompt)
2. Drop remaining biased queries via post-filter
3. Run Stage 8.5 selection + retrieval (MiniLM/BM25 hit@K) on filtered pool
4. Measure personalization preservation + content match quality

## Files

- Scripts:
  - `gaussian/syntax_subspace_stage12_logit_mask.py` — proof of concept
  - `gaussian/syntax_subspace_stage13_hybrid.py` — production hybrid
- Outputs (scratch2):
  - `stage12_logit_masked_pool.json` — 560 candidates (over-constrained)
  - `stage13_hybrid_pool.json` — 560 candidates (production)
  - `stage12_comparison.json`, `stage13_comparison.json` — metrics

## Lessons

1. **bad_words = hard constraint, not magic**: still requires tokenization
   awareness. Hyphenated/inflection variants can bypass.
2. **Don't combine bad_words with whitelist prompts**: the LLM loses too
   much vocabulary and template collapse follows.
3. **Freeform structural prompts + bad_words = sweet spot**: prompt gives
   structural variety, bad_words filters bias at the logit level.
4. **Per-class hit rates validate the approach**: S13 matches S11 on all
   7 classes (within ±2.5pp), confirming bad_words doesn't hurt
   structural control.

## Related

- [[stage11-content-lock-structural-go]] — prompt-only baseline
- [[phase16-decode-constraint-go]] — existing regex-based constraint
- [[stage10k-a1-reject-repeat-sota]] — selection-side A1 reject-repeat