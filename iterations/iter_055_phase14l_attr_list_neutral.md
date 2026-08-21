# Phase 14.L: 属性堆叠 (attr_list) 作为真中性 — Style Offset NO-GO v3

**Date**: 2026-08-21
**Question**: 是否 LLM 改写 query 会破坏结构,导致 norm 8x 量级?换成直接 attribute list (逗号分隔,无动词) 作为真中性 query,是否能让 style offset 范式 work?
**Answer**: **NO-GO**。即使最干净的真中性 (attribute list), q_offset norm mean = **923** (vs user 71, ratio **13x**), 比 LLM neutral (576) **更大**。attr_list 不是真中性,而是真"无风格骨架" query,但生成时带 style injection,所以 query offset 范式本质性失败。

## 关键发现

### 1. 真中性 = Attribute List

| 范式 | Phase 14.K2 | **Phase 14.L** |
|---|---|---|
| 中性 query | LLM 改写 (重组句式) | **属性堆叠 (无动词)** |
| 例 | "Looking for a Summer Infant product in white, constructed from wood and metal..." | "BabyBond, 4.93 pounds, 16.2"D x 13.24"W x 13.61"H, Green, Silicone" |

ATTR_LIST_PROMPT 要求 LLM 输出**仅属性值的逗号分隔**,不输出句法/动词/连接词。

### 2. Norm Diagnostic 关键对比

| 量级 | Phase 14.J2 (LLM/退化) | Phase 14.K2 (LLM/干净) | **Phase 14.L (attr_list/干净)** |
|---|---|---|---|
| user_offset norm | 71 | 71 | **71** |
| **q_offset norm** | 685 | 576 | **923** |
| **q/user ratio** | 9.7x | 8.1x | **13.04x** |

**attr_list 中性反而让 q_offset norm 最大** (923 vs LLM 的 576) — 因为:
- attr_list 完全去除风格 (无 "Find me" 句式,无 "I want" 情感词)
- 但 query 仍有完整属性 + style injection (来自 user 评论的风格被注入到属性顺序/标点上)
- **user 评论 vs query** 的 norm 差异是**结构差异** (短 vs 长, 散文 vs 属性),不是风格差异

### 3. Phase 14.L 结果 (30 pair, A22 K=8 layer 26)

| Method | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **best_of_k_maha_a22** (Phase 14.F SOTA) | 0/30 | 3/30 (10%) | **20/30 (66.7%)** | **80.4** |
| **style_offset_attr_list maha_pooled** | **1/30 (3.3%)** | 2/30 (6.7%) | 14/30 (46.7%) | 103.8 |
| style_offset_attr_list bhattacharyya | 0/30 | 4/30 (13.3%) | **17/30 (56.7%)** | 94.3 |
| style_offset_attr_list w2 | 0/30 | 0/30 | 11/30 (36.7%) | 114.6 |
| style_offset_attr_list symmetric_kl | 0/30 | 1/30 (3.3%) | 12/30 (40%) | 107.4 |

**Style Offset + attr_list 中性:** top-100 36.7-56.7% (vs best_of_k_maha 66.7%) → 全部 NO-GO。
**唯一 work**: best-of-K Maha K=8 layer 26 (Phase 14.F SOTA, 无需中性 query,无需 distribution 估计)。

### 4. 为什么 attr_list 中性反而更糟

#### A. norm 13x 量级是结构差异,不是风格差异

| 维度 | user 评论 | attr_list | A22 query |
|---|---|---|---|
| 平均长度 | ~60 chars | ~150 chars | ~160 chars |
| 句法 | 散文 | 无句法 | 完整句法 |
| Norm | ~71 | ~50 (更短) | ~923 (带 style injection) |

**Query norm 是 user norm 的 13x 不是因为"风格差异"**,而是因为:
1. **长度 3x**: user 60 chars → query 160 chars → hidden norm 长度线性增长
2. **内容堆叠**: query 把 5 个属性 + 风格前缀堆在一起,语义密度远高于 user
3. **Style injection**: Qwen layer 26 把 style vector 注入后,query hidden 范数爆炸

#### B. 即使 q_norm = user_norm × 13, Maha 仍失效

```python
# 假设 user_mu_norm = 71, user_var_norm = 30 (per dim)
# q_mu_norm = 923, q_var_norm ~150

# Maha: sum((q-u)² / pooled_var) ~ (923)² / (30) ~ 28,400
# user 之间的差异 ~ 71² / 30 ~ 168
# q_offset 量级 >> user 之间的差异 → Maha 完全被 q 量级主导
```

**任何 user 只要 μ 方向接近 q_offset 的方向,就能 rank 高 → 失去区分度**。
**rank-1 唯一 1/30** 在 maha_pooled 下,但毫无意义 (vs best_of_k_maha 0/30,top-100 66.7% 才是真实区分)。

### 5. 关键论断澄清

| 论断 | 之前 | **Phase 14.L 之后** |
|---|---|---|
| "A14 + LLM neutral NO-GO" | ✓ 真实 | ✓ 真实 |
| "干净 generator + LLM neutral 会 work" | ✗ 推断错误 | ✗ 仍然 NO-GO (Phase 14.K2) |
| "attr_list 真中性会 work" | 待验证 | **✗ 仍然 NO-GO** (Phase 14.L) |
| "norm 8x 是 generator 退化的副作用" | ✗ 错 | **✗ 错** — norm 13x 即使 attr_list |
| "norm 8x 是 LLM 改写破坏结构" | ✗ 错 | **✗ 错** — attr_list 无改写, norm 反而更大 |
| **"norm 8-13x 是结构差异 (query vs comment)"** | ✗ 未意识到 | **✓ 真实** — 这是结构性差异, style offset 范式不可能 work |

**Style Offset 范式全范式 NO-GO**,**与中性 query 选择完全无关**:
- D_off (Phase 14.J/K): 8x norm
- LLM neutral (Phase 14.J2/K2): 8.1x norm
- attr_list (Phase 14.L): **13x norm**

Style offset 的根本问题是:**query 和 user comment 是不同长度的不同文本范式**,offset 范式无法对齐两个分布。

## 决策

### Style Offset 范式彻底失败 — 4 种中性方案全 NO-GO

| Phase | 中性 | generator | top-100 | norm ratio |
|---|---|---|---|---|
| 14.J | D_off | A14 K=30 | 50-63% | 8x |
| 14.J2 | LLM | A14 K=30 | 36.7% | 9.7x |
| 14.K | D_off | A22 K=8 (干净) | 40-50% | 8x |
| 14.K2 | LLM | A22 K=8 (干净) | 36.7% | 8.1x |
| **14.L** | **attr_list** | **A22 K=8 (干净)** | **36.7-56.7%** | **13x** |

**Style Offset 在 5 种中性方案下全 NO-GO**:
- 不管中性是 D_off / LLM / attr_list
- 不管 generator 是 A14 K=30 (退化) / A22 K=8 (干净)
- norm 量级差 8-13x 是**query vs comment 的结构差异**,不是风格差异

### 推荐

1. **保持 Phase 14.F K=8 best-of-K Maha SOTA** — top-100 86.7% (phase10 first-30 pair) / 66.7% (Phase 14.L A22 K=8)
2. **不要再尝试 style offset** — 5 种范式穷举 NO-GO
3. **不要再尝试 distribution-to-distribution** — 4 种范式穷举 NO-GO (Phase 14.E/H/I)

### 不要做什么

- **不要再尝试任何 style offset** — 已穷举 5 种中性方案,全 NO-GO
- **不要再尝试任何 dist2dist** — 已穷举 4 种范式,全 NO-GO
- **不要再尝试 LLM neutral** — norm 576 即使干净 generator
- **不要再尝试 attr_list** — norm 923 反而最大

## Pipeline 总结

1. **Generator**: A22_a0.5 K=8 (Phase 14.B, 0% 退化)
2. **真中性**: ATTR_LIST_PROMPT 生成 attribute list (无句法)
3. **Encode**: A22 + attr_list at layer 26
4. **q_offset = a22 - attr_list** (per-pair mean+var over K=8)
5. **user_offset = user_orig - user_paired_neutral** (Phase 14.F-paired cache)
6. **Rerank**: 4 metrics (maha_pooled, bhattacharyya, w2, symmetric_kl) + best-of-K Maha baseline
7. **Diagnostic**: norm comparison (q vs user)

**总时长**: ~3 min (含 Qwen 加载 + 240 attr_list 改写 + 240×2 encoding + rerank)

## 工程

- **240 attr_list 生成**: 8 batches × 30, ~22.3s (clean, no verb, no connector)
- **240×2 encoding**: ~30s
- **Rerank CPU**: ~5s
- **Total**: ~3 min

## 文件

| Path | Purpose |
|---|---|
| `phase14_l_attr_list_neutral.py` | style offset with CLEAN A22 + attr_list 中性 |
| `phase14_l_attr_list_neutral.jsonl` | 240 unique A22 → attr_list 改写 |
| `phase14_l_attr_list_hiddens.npy` | 240 × 3584d (attr_list hiddens) |
| `phase14_l_a22_residuals.npy` | 240 × 3584d (a22 - attr_list) |
| `phase14_l_attr_list_neutral_sents.jsonl` | paired (q_styled, q_neutral_attr_list) |
| `phase14_l_eval.json` | 5 methods × 4 metrics eval |
| `phase14_l_per_pair.jsonl` | per-pair ranks |
| `phase14_l_meta.json` | metadata + norm diagnostic |

## 决策总结

**Phase 14.L 关键澄清**:

1. **q_offset norm 13x 是结构性,不是风格性**:
   - user 评论 60 chars vs query 160 chars (长度 2.7x)
   - comment 散文 vs query 属性堆叠 (内容密度差)
   - 不管中性 query 选什么 (D_off / LLM / attr_list), 都不可能消除结构差异

2. **Style Offset 范式根本性失败**:
   - 5 种中性方案穷举全 NO-GO
   - norm 量级差 8-13x 是 query vs comment 的结构差异,不是风格差异
   - **结论**: offset 范式不可能 work

3. **SOTA 仍是 Phase 14.F K=8 best-of-K Maha**:
   - phase10 first-30 pair: top-100 86.7%
   - Phase 14.B 30 pair: top-100 83.3% (A14) / 66.7% (A22)
   - **不需要 offset / neutral / distribution estimation**

**Why attr_list 中性最差**:
1. **属性堆叠无句法**,但 query 仍有完整句法 + style injection
2. **两者 norm 差距最大**: attr_list ~50 (短,无 style) vs query ~923 (长,带 style)
3. **norm 13x 量级远超 LLM neutral 的 8x** → Maha 完全失效

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换
2. **不要再探索任何 style offset 范式** — 5 种范式穷举 NO-GO
3. **不要再探索任何 dist2dist 范式** — 4 种范式穷举 NO-GO
4. **如果要继续提升 rerank**,考虑:
   - 双 layer 融合 (22 + 26)
   - per-user σ² LW shrinkage
   - PCA 30d + best-of-K Maha

## 相关

- [[phase14j-style-offset-nogo]] — Phase 14.J D_off neutral parent
- [[phase14j2-llm-neutral-style-offset-nogo]] — Phase 14.J2 LLM neutral (退化 generator)
- [[phase14k-generator-vs-rerank-separation]] — Phase 14.K Generator vs Rerank 分离
- [[phase14k2-llm-neutral-clean]] — Phase 14.K2 LLM neutral clean
- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F SOTA
- [[phase14h-dist2dist-nogo]] — Phase 14.H dist2dist
- [[phase14i-pca-dist2dist]] — Phase 14.I PCA PARTIAL-GO