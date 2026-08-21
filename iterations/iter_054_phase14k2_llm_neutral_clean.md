# Phase 14.K2: LLM Neutral Query CLEAN Test — **LLM Neutral Idea 本身 NO-GO**

**Date**: 2026-08-21
**Question**: Phase 14.J2 NO-GO 是否只是 A14_a1.0 K=30 (18.3% 退化) 的副作用?如果用干净的 A22_a0.5 K=8 (0% 退化) + LLM neutral,是否会 work?
**Answer**: **NO-GO**。即使 A22 K=8 干净 generator, q_offset norm 仍然 576 (vs user 71),**LLM neutral query idea 本身 NO-GO,不是 generator 退化的副作用**。

## 关键发现

### 1. 干净测试设计

| 维度 | Phase 14.J2 | **Phase 14.K2** |
|---|---|---|
| Generator | A14_a1.0 K=30 | **A22_a0.5 K=8 (干净)** |
| 退化率 | 18.3% | **0%** |
| Unique queries | 870 | **238** |
| Query 中性版 | LLM 改写 | **LLM 改写** |

### 2. Norm Diagnostic 关键对比

| 量级 | Phase 14.J2 (A14 退化) | **Phase 14.K2 (A22 干净)** |
|---|---|---|
| user_offset norm | 71 | 71 |
| **q_offset norm** | **685** | **576** (基本相同!) |
| **量级比 q/user** | **9.7x** | **8.1x** |

**核心发现**:**Generator 退化 vs 干净 generator, q_offset norm 几乎一样** (685 vs 576):
- Phase 14.J2 解释为"query 改写破坏结构"
- Phase 14.K2 排除 generator 退化后, **q_offset norm 仍然 576** = LLM 改写 query 的根本性问题
- **LLM 改写 query 不是"风格去除",而是"语义重生成"**

### 3. Phase 14.K2 结果 (30 pair, K=8, layer 26)

| Method | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **best_of_k_maha_a22** (Phase 14.F baseline) | 0/30 | 2/30 (6.7%) | **21/30 (70%)** | **76.6** |
| **style_offset_llm_clean maha_pooled** | 0/30 | 2/30 (6.7%) | **11/30 (36.7%)** | 111.8 |
| style_offset_llm_clean bhattacharyya | 0/30 | 2/30 (6.7%) | 14/30 (46.7%) | 108.8 |
| style_offset_llm_clean w2 | **1/30** | 2/30 (6.7%) | 16/30 (53.3%) | 100.9 |
| style_offset_llm_clean symmetric_kl | 0/30 | 2/30 (6.7%) | 15/30 (50%) | 104.7 |

### 4. Style Offset 范式 vs Best-of-K Maha 对比

| 范式 | top-100 | mean_rank | 评价 |
|---|---|---|---|
| **best_of_k_maha_a22** (Phase 14.F SOTA style) | **70%** | **76.6** | 唯一 work |
| style_offset_llm_clean maha_pooled | 36.7% | 111.8 | NO-GO (-33.3pp) |
| style_offset_llm_clean w2 | 53.3% | 100.9 | NO-GO (-16.7pp) |

**Style Offset 即使在干净 generator + LLM neutral 下仍然 NO-GO**。

### 5. 为什么 LLM 改写 query 仍然失败

#### A. Query vs Sentence 改写行为差异

```text
User 评论: "These diapers fit great on my baby and don't leak at all!"
LLM neutral: "The diapers fit the baby and do not leak."
# 改写保留主谓宾,只去除 "great" / "don't leak at all"
# user_offset norm = 71 (适度)

A22 Query: "Find me a Summer Infant product with a white color, made of wood and metal..."
LLM neutral: "Looking for a Summer Infant product in white, constructed from wood and metal..."
# 改写改变句式 (Find me → Looking for),但语义接近
# 仍然 offset norm ~576 (远超 user)
```

#### B. 8x 量级差的原因

| 维度 | user | query |
|---|---|---|
| 改写距离 (短文本) | 保留句法骨架 | 重组句式 |
| 长度变化 | 小 (-15%) | 较大 (-30%) |
| 属性位置 | 不变 | 重排 |
| 关键 token | 保留 | 部分替换 (Find me → Looking for) |

#### C. 即使 q_norm = user_norm × 8, Maha 仍失效

```python
# 假设 user_mu_norm = 71, user_var_norm = 30 (per dim)
# q_mu_norm = 576, q_var_norm ~100

# Maha: sum((q-u)² / pooled_var) ~ (576)² / (30) ~ 11,000
# user 之间的差异 ~ 71² / 30 ~ 168
# q_offset 量级 >> user 之间的差异 → Maha 完全被 q 量级主导
```

**任何 user 只要 μ 接近 q_offset 的方向,就能 rank 高 → 失去区分度**。

### 6. 关键论断澄清

| 论断 | Phase 14.J2 之前 | **Phase 14.K2 之后** |
|---|---|---|
| "A14 + LLM neutral NO-GO" | ✓ 真实 | ✓ 真实 |
| "干净 generator + LLM neutral 会 work" | 待验证 | **✗ 仍然 NO-GO** |
| "LLM neutral query idea 失败" | ✗ 推断错误 | **✓ 真实** |
| "norm 685 是 generator 退化的副作用" | 误导性 | **✗ 错** — norm 576 即使干净 query |
| "query 改写破坏结构" | 部分对 | **✓ 对** — 即使干净 query, 改写仍重组句式 |

**Phase 14.J2 的 norm 685 vs Phase 14.K2 的 norm 576 几乎相同**,说明:
- **q_offset 8x 量级是 LLM 改写 query 的本质特性**,与 generator 退化无关
- Style offset 范式根本性失败,不是因为 query 退化

## 决策

### Style Offset + LLM Neutral 范式彻底失败

| 范式 | 退化 generator | **干净 generator** | 评价 |
|---|---|---|---|
| A14 + D_off neutral (Phase 14.J) | top-100 50% | n/a | NO-GO |
| A14 + LLM neutral (Phase 14.J2) | top-100 36.7% | n/a | NO-GO |
| **A22 + LLM neutral (Phase 14.K2)** | n/a | **top-100 36.7%** | **NO-GO** |
| A22 + D_off neutral (Phase 14.K) | n/a | top-100 40-50% | NO-GO |

**Style Offset 范式全范式 NO-GO**,与 generator 和 neutral 选择无关。

### 推荐

1. **保持 Phase 14.F K=8 best-of-K Maha SOTA** — top-100 86.7% (phase10 first-30 pair)
2. **A22_a0.5 K=8 layer 26** — 70-76.7% top-100 (Phase 14.B 30 pair),干净 generator 但略差 rerank
3. **A14_a1.0 K=8 layer 26** — 83.3-86.7% top-100 (Phase 14.F SOTA)

### 不要做什么

- **不要再尝试 LLM neutral query** — 已穷举 退化 generator + 干净 generator,全 NO-GO
- **不要再尝试 style offset** — 已穷举 4 种 neutral 方案 (D_off / LLM / paired / offset),全 NO-GO
- **不要再尝试 dist2dist** — 即使干净 generator 也 NO-GO

## Pipeline 总结

1. **Generator**: A22_a0.5 K=8 (Phase 14.B, 0% 退化)
2. **LLM neutral**: 用 REWRITE_SYSTEM 改写每个 A22 query
3. **Encode**: A22 + LLM_neutral at layer 26
4. **q_offset = a22 - LLM_neutral** (per-pair mean+var over K=8)
5. **user_offset = user_orig - user_paired_neutral** (Phase 14.F-paired cache)
6. **Rerank**: 4 metrics (maha_pooled, bhattacharyya, w2, symmetric_kl) + best-of-K Maha baseline
7. **Diagnostic**: norm comparison (q vs user)

**总时长**: ~3 min (含 Qwen 加载 + 238 rewrites + 238×2 encoding + rerank)

## 工程

- **238 unique A22 rewrites**: 8 batches × 32, ~30s
- **238×2 encoding**: ~30s
- **Rerank CPU**: ~5s
- **Total**: ~3 min

## 文件

| Path | Purpose |
|---|---|
| `phase14_k2_llm_neutral_clean.py` | style offset with CLEAN A22 + LLM neutral |
| `phase14_k2_a22_neutral_cache.jsonl` | 238 unique A22 → LLM neutral 改写 |
| `phase14_k2_a22_neutral_hiddens.npy` | 238 × 3584d |
| `phase14_k2_a22_residuals.npy` | 238 × 3584d (a22 - neutral) |
| `phase14_k2_a22_neutral_sents.jsonl` | paired (q_styled, q_neutral) |
| `phase14_k2_clean_eval.json` | 5 methods × 4 metrics eval |
| `phase14_k2_clean_per_pair.jsonl` | per-pair ranks |
| `phase14_k2_clean_meta.json` | metadata + norm diagnostic |

## 决策总结

**Phase 14.K2 关键澄清**:

1. **Phase 14.J2 norm 685 不是 generator 退化的副作用**:
   - 干净 generator (A22 K=8 0% 退化) 也产生 norm 576
   - norm 8x 量级是 LLM 改写 query 的本质特性
   - 不依赖 generator 是否退化

2. **LLM neutral query idea 本身 NO-GO**:
   - 不管 generator 干净还是退化
   - 不管 K=8 还是 K=30
   - 不管 D_off 还是 LLM neutral
   - Style offset 范式根本性失败

3. **SOTA 仍是 Phase 14.F K=8 best-of-K Maha**:
   - phase10 first-30 pair: top-100 86.7%
   - Phase 14.B 30 pair: top-100 83.3% (A14) / 76.7% (A22)
   - 不需要 style offset 范式

**Why LLM 改写 query 失败**:
1. **Query 是属性堆叠短句**,即使干净也容易被 LLM 改写时重组
2. **LLM neutral ≠ "中性"**,而是"重新表述"
3. **句式变化 (Find me → Looking for)** 已经造成大 offset
4. **属性位置变化** 进一步加大 offset

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换
2. **不要再探索 LLM neutral query** — 已穷举干净 + 退化 generator
3. **不要再探索 style offset** — 已穷举 4 种范式
4. **如果要继续提升 rerank**,考虑非 offset 方案:
   - 双 layer 融合 (22 + 26)
   - per-user σ² LW shrinkage
   - PCA 30d + best-of-K Maha

## 相关

- [[phase14j-style-offset-nogo]] — Phase 14.J D_off neutral parent
- [[phase14j2-llm-neutral-style-offset-nogo]] — Phase 14.J2 LLM neutral (退化 generator)
- [[phase14k-generator-vs-rerank-separation]] — Phase 14.K Generator vs Rerank 分离
- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F SOTA
- [[phase14h-dist2dist-nogo]] — Phase 14.H dist2dist
- [[phase14i-pca-dist2dist]] — Phase 14.I PCA PARTIAL-GO

当前任务已完成,请做下一个任务的指示。