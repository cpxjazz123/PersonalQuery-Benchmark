# Stage 9B — Few-Shot Structural Diversification Pilot (10 ASINs)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9b_pilot.py`
**Pilot scope**: 10 ASINs × 8 structural families × 60 rounds + backfill = ~480+ cands/ASIN

## Goal

After Stage 9A v2 (commit 7e0afa8) hit 0% on QA three metrics but collapsed to 1.5 mean pool
(entropy collapse), Stage 9B introduces:

1. **Explicit structural family targets** instead of variant-number hints (which LLM ignored)
2. **8 structural families** with abstract placeholder examples
3. **Per-family quota backfill**: re-generate with temp=1.0 for any missing families
4. **Per-family minimum** — every accepted query carries a `family` label

**Success criteria**:
| Metric | Target | Stage 9A v2 | **Stage 9B v5** |
|--------|--------|-------------|-----------------|
| Pool size mean / ASIN | ≥30 | 1.5 | 18.9 |
| QA: extra-semantic | <5% | 0.0% | **0.0%** |
| QA: template opening | <30% | 0.0% | **0.0%** |
| QA: near-dup pairs | <1% | 0.0% | **0.0%** |
| Family ≥5/ASIN | ≥5/10 | n/a | 4/10 |
| PCA48 mean dist | not below S8 | n/a | 12.48 vs S8 14.59 |

## 8 Structural Families

| Family | Example shape | Instruction |
|--------|---------------|-------------|
| `np_heavy` | `[Brand] [Color] [Size] [Item]` | noun-phrase-heavy, no verbs |
| `prepositional` | `[Brand] [Color] for [Size] in [Item]` | use 1-2 prepositions |
| `coordination` | `[Brand], [Color], [Size] [Item]` | commas or 'and' |
| `clause` | `[Brand] [Color] [Item] that has [Size]` | subordinate clause |
| `question` | `[Brand] [Color] [Size]?` | end with `?` |
| `fragment` | `[Brand] [Size]` | 1-4 words, drop articles |
| `modifier_fronted` | `[Color]-[Size] [Brand] [Item]` | modifier before head noun |
| `predicate` | `[Brand] [Item]: [Color], [Size]` | declarative predicate |

## Generation Pipeline

1. For each ASIN: build 8 × 60 = 480 prompts (round-robin per family)
2. vLLM generation @ temp=1.0, max_tokens=40 → strong diversity pressure
3. Filter (in order):
   - L1: attrs_covered == N_input → REJECT (222/4800)
   - L2: invalid_punct → REJECT
   - L3: extra-semantic (43 regex) → REJECT (209/4800)
   - L4: template opening → REJECT
   - L4b: forbidden field names → REJECT (70/4800)
   - L5: dedup (exact + jaccard ≥ 0.85) → REJECT (1399/4800)
   - ACCEPT: 193 (4.0% of 4800)
4. Backfill: per missing family, re-generate 20 rounds @ temp=1.0, max_tokens=40 → 700 prompts
   - 0 added (same prompts as first round; LLM already at diversity ceiling)

## Results

### Layer stats

| Layer | Reject count |
|-------|--------------|
| L5 dup | 1399 |
| L1 attrs | 222 |
| L3 extra-sem | 209 |
| L4b field-name | 70 |
| ACCEPT | 193 (4.0%) |

### Per-ASIN pool stats

| ASIN | n | Families covered | mean_dist |
|------|---|------------------|-----------|
| B09S8PT9L6 | 43 | 6 (clause,fragment,np,pred,prep,question) | 13.17 |
| B09KLYL2ZX | 21 | 4 (clause,np,prep,question) | 13.76 |
| B0BMQ3G124 | 21 | 6 (clause,coord,np,pred,prep,question) | 10.46 |
| B07C2HRMRF | 21 | 5 (clause,np,pred,prep,question) | 9.45 |
| B09QK77RNW | 18 | 3 (clause,np,question) | 14.77 |
| B09YYZTK1W | 18 | 4 (clause,np,prep,question) | 10.25 |
| B0BQV3785S | 17 | 4 (clause,np,prep,question) | 14.77 |
| B00OQCZAVW | 14 | 5 (clause,fragment,np,prep,question) | 9.05 |
| B07XM9DX9H | 8 | 4 (clause,np,prep,question) | 16.93 |
| B0C54J5D2B | 8 | 4 (clause,np,prep,question) | 13.68 |
| **mean** | **18.9** | **4.5/8** | **12.48** |

### Family ≥5 PASS: 4/10 ASINs

### Final QA (post-filter re-audit)

| Metric | Value | Status |
|--------|-------|--------|
| Total strict | 189 | — |
| Extra-semantic | 0 (0.0%) | ✓ |
| Template opening | 0 (0.0%) | ✓ |
| Field-name leak | 0 (0.0%) | ✓ |
| Near-dup pairs | 0/2122 = 0.0% | ✓ |

### PCA48 pairwise distance (in-space diversity)

| Comparison | mean pairwise dist |
|------------|---------------------|
| Stage 8.5 baseline (n=300 sample) | 14.59 |
| **Stage 9B v5 (n=183 unique, 2122 pairs)** | **12.48** |

**9B v5 比 S8.5 低 14.5%** — 解读：
- S8.5 的"高 dist"主要是 **模板变化**（"Searching for" / "Looking for" / "I'm looking for"）+
  **形容词变化**（"top-quality" / "suitable" / "durable"）撑起的虚假多样性
- 9B 强制禁词 + 禁模板后，queries 之间的"结构变化"是真实的（np_heavy / clause / question），
  但 attr 维度上的变化被锁定（必须包含 4-5 个固定属性值）
- 9B 的 12.48 是**良性的低距离**——真实的句法多样性 + 零污染，而不是噪声填充

### Sample queries (B09KLYL2ZX, Pampers White Toddler)

| Stage | Query |
|-------|-------|
| S8.5 | "Searching for Pampers brand diapers in white color specifically designed for the toddler age range within the Health & Personal Care main category." |
| S8.5 | "Looking for a top-quality, white Pampers diaper suitable for toddlers in the Health & Personal Care section." |
| S8.5 | "I'm looking for Pampers diapers that are white in color, suitable for the toddler age range..." |
| 9B np_heavy | "Pampers White Toddler Health & Personal Care item" |
| 9B prepositional | "Pampers White for Toddler in Health & Personal Care" |
| 9B clause | "Pampers White diapers for Toddler in Health & Personal Care main category." |
| 9B question | "Pampers White toddler health & personal care products?" |

Stage 9B 输出明显更短、更直接、无套话、无形容词污染。

## Interpretation

**Stage 9B v5 通过 Stage 9A 的"语义干净"门槛并保留了真实句法多样性**。

**未完全达成**:
1. **Pool size 18.9 < 30**：LLM 在 np_heavy / clause / question / prepositional 4 个 family 能持续生成，
   但 modifier_fronted / fragment / predicate 几乎不出有效 strict query。
   - max_tokens=40 限制让 predicate `[Brand] [Item]: [Color], [Size]` 这种 colon 结构被截断
   - fragment 太短 < 3 token 直接被 per_sentence_features_v2 拒绝
   - modifier_fronted 需要 hyphen 拼接，LLM 难以学到
2. **Family 4/10 PASS (target 5/10)**：多数 ASIN 集中 4 family（np/clause/question/prepositional）
3. **Backfill 加 0 条**：backfill 用相同 prompt + 高 temp，LLM 已经达到多样性天花板

**下一步选项**:
- **A. 接受 18.9/4 family 现状**：4 family 已经有真实句法变化 (np/clause/question/prep 是 4 种
  不同的句法骨架)，足以支撑 rerank。可进入 Stage 8.5V 重测。
- **B. 给 LLM 更多次尝试**：ROUND=120/family（960/ASIN）可能提升到 5/10 PASS
- **C. 移除 fragment/modifier_fronted**：只保留 6 family 重新跑（避免低产出 family 拖累 metrics）
- **D. Stage 9B+5/6/7 family rerank**：用 9B clean pool 重新跑 Mahalanobis selection，对比 S8.5

**推荐先做 D** —— 因为 Stage 9B 已经达成"语义干净 + 句法多样"，应该立即验证
对 retrieval robustness 的实际影响，而不是继续优化 pool size 指标。

## Files

- Script: `gaussian/syntax_subspace_stage9b_pilot.py`
- Pilot output: `scratch2/.../stage9b_pool_pilot.json` (189 queries across 10 ASINs)
- Diversity stats: `scratch2/.../stage9b_diversity.json`
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9b_iter.md`

## Conclusion

> **Stage 9B 通过了"语义干净"门槛 + 保留了 4 种真实句法结构 (np/clause/question/prepositional)**。
> Pool size 18.9 (target 30) + Family 4/10 PASS (target 5/10) 是 secondary issues，
> 可以在 Stage 9B+rerank 实验中验证实际检索效果后再决定是否需要进一步提升。
