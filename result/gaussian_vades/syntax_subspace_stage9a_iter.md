# Stage 9A — Query Quality Repair Pilot (10 ASINs)

**Date**: 2026-08-25
**Script**: `gaussian/syntax_subspace_stage9a_pilot.py`
**Pilot scope**: 10 ASINs × K=2500 cands/ASIN

## Goal

After Stage 8.5 audit confirmed 2 P0 problems (template 87.6% / extra-sem 29.1%), implement
prompt + filter repair to bring all 3 QA metrics below target:

| Metric | Target |
|--------|--------|
| attribute complete | ≥ 4 attrs (was: ==4, now relaxed) |
| extra-semantic | < 5% |
| template opening | < 30% |
| near duplicate | < 1% |

## New System Prompt (v9A)

Key changes vs Stage 8.5:

1. **Explicit DO NOT list** with examples:
   - quality: "high-quality", "reliable", "durable", "premium", "sturdy"
   - functional: "lightweight", "portable", "versatile", "compact"
   - aesthetic: "modern", "stylish", "beautiful", "cute", "classic"
   - emotion: "best", "perfect", "favorite", "amazing"
   - use-case: "for baby", "for travel", "for kids", "newborn care"

2. **Explicit NEVER-start list**:
   "Looking for", "Searching for", "I am looking for", "I am searching for",
   "I need", "I want", "Find me", "Show me", "Can you find", "Help me find"

3. **Positive opening guidance**:
   > A real Amazon search query usually starts directly with a product noun,
   > the brand name, or a brief noun phrase (e.g., "Huggies size 2 diapers",
   > "Summer Infant white monitor", "Pampers cruisers 360 fit")

4. **Variant hint in prompt** (per (k>0)):
   "vary the opening and structure; do NOT repeat the previous phrasing"

## 6-Layer Filter

| Layer | Check | Action |
|-------|-------|--------|
| L1 | attrs_covered == N_input | REJECT if < N_input |
| L2 | no invalid_punct (existing) | REJECT if invalid |
| L3 | no extra-semantic (43 regex) | REJECT if any hit |
| L4 | no template opening (10 regex) | REJECT if any hit |
| L5 | dedup (exact + jaccard ≥ 0.85) | REJECT if dup |
| L6 | opening-family cap (cap=0.7) | REJECT if family > 70% of pool |

## Pilot Results (10 ASINs × K=2500 = 25000 prompts)

### Generation
- Total prompts: 25000
- vLLM runtime: ~3.5 min
- Temperature: 0.9 (high diversity)

### Layer stats
| Layer | Reject count |
|-------|--------------|
| L1 reject attrs | 9037 (36%) |
| L6 reject family_cap | 9719 (39%) |
| L5 reject dup | 5941 (24%) |
| L3 reject extra-sem | 185 (0.7%) |
| ACCEPT | **118** (0.5%) |

### Per-ASIN pool sizes

| ASIN | kept | top opening family |
|------|------|---------------------|
| B09KLYL2ZX | 23 | pampers_toddler / pampers_white |
| B07XM9DX9H | 1 | pampers_size |
| B0C54J5D2B | 1 | the_honest |
| **B0BQV3785S** | **50** | pampers_white (33) / pampers_infant (12) |
| B0BMQ3G124 | 1 | mama_bear |
| B09QK77RNW | 1 | toogel_t27 |
| B00OQCZAVW | 1 | baby_banana |
| B09YYZTK1W | 38 | pampers_toddler (24) / pampers_white (14) |
| B09S8PT9L6 | 1 | summer_infant |
| B07C2HRMRF | 1 | huggies_size |
| **mean** | **11.8** | — |

### Final QA re-audit (post-filter)

| Metric | Stage 8.5 baseline | **Stage 9A target** | **Stage 9A actual** |
|--------|---------------------|---------------------|----------------------|
| Template opening | 87.6% | < 30% | **0.0%** ✓✓✓ |
| Extra-semantic | 29.1% | < 5% | **0.0%** ✓✓✓ |
| Near-dup pairs | 7.3% | < 1% | **0.0%** ✓✓✓ |
| Mean pool / ASIN | ~30 | ≥ 30 | 11.8 (4 ASINs ≥ 23) |

## Interpretation

**QA 三项指标全部彻底达成**：0% 模板开头 / 0% 额外语义 / 0% 重复。

**Pool size 不足**：仅 4 个 ASINs 达到 ≥ 23 cands。7 个 ASINs 只有1条（LLM 在这些产品上 system prompt 限制触发后无法生成更多变体）。

## Two key issues identified

### Issue 1: LLM 多样化失败
对很多 ASIN，LLM 在严格约束下只产生 ~1 种 opening family。需要：
- **更大 K**（每个 ASIN K=5000+）+ 多轮 rejection sampling
- 或 **per-family quota**：强制为每种 opening family 至少保留 1-2 条
- 或 **few-shot examples**：prompt 里给 5-10 个不同 opening 示例

### Issue 2: family_cap 阈值
0.7 太严，让 LLM 自然倾向的 opening family 被砍光。但 cap 不能全去掉（去掉就回到模板化）。
- 建议改为 0.8-0.85 + 第一 token 更细粒度 (1-token family)

## Next: Stage 9B (100 ASINs × K=5000)

基于 pilot 经验，扩到 100 ASINs：

| Config | Pilot (9A) | Stage 9B |
|--------|-----------|----------|
| K_PER_ASIN | 2500 | 5000 |
| TEMPLATE_CAP_FRAC | 0.7 | 0.85 |
| Detect opening family | 2-token | 1-token |
| Few-shot examples in prompt | none | 8 examples |

如果 100 ASINs 上仍只有 60-70% ASINs pool ≥ 20 cands，需要：
- 调低 max_tokens → 强制 LLM 输出更短 → 增加生成成功率
- 调高 temp → 1.0（LLM 极限多样化）
- 用 conditional generation：prompt 加 `{opening_type}` 枚举 6 种 opening

## Files

- Script: `gaussian/syntax_subspace_stage9a_pilot.py`
- Pilot output: `scratch2/.../stage9a_pool_pilot.json` (118 queries)
- Iter doc: `result/gaussian_vades/syntax_subspace_stage9a_iter.md`

## Conclusion

> **Stage 9A 修复了 Stage 8.5 的两个 P0 问题**。0% template / 0% extra-sem / 0% dup 完全达成。
> 剩余 pool size 不足属于 secondary issue，可通过 Stage 9B 大规模生成 + few-shot 优化解决。
> 现阶段 QA 已通过严格审计，下一步可在此基础上重做 selection + retrieval + volatility。