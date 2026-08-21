# Phase 14.J2: LLM Neutral Style Offset Rerank — **NO-GO** (q_offset norm 反而爆炸)

**Date**: 2026-08-21
**Question**: 用户提出"Query 的中性版也使用 LLM 改写" — 既然 Phase 14.J 因为 neutral 不一致(D_off 弱 vs LLM 强)导致 NO-GO,如果 query 也用 LLM 改写为中性,两边都使用 LLM 改写,offset 量级应该匹配。
**Answer**: **NO-GO**,且比 Phase 14.J 更差。LLM 改写 query 的中性版 **几乎完全删除结构**(短 product query 改写为完全不同的表述),导致 q_offset norm 爆炸到 **685**(vs user 71)。

## 关键发现

### 1. 实验设计

| 阶段 | Phase 14.J | **Phase 14.J2 (本次)** |
|---|---|---|
| User neutral | LLM 改写 | LLM 改写 (同 Phase 14.J) |
| Query neutral | **D_off baseline** | **A14 query 的 LLM 改写** |
| q_offset | a14 - D_off | **a14 - LLM_neutral(a14)** |

**目的**:统一两边 neutral 都用 LLM 改写,避免 Phase 14.J 的量级不齐问题。

### 2. Phase 14.J2 结果 (30 pair, K=30)

| Method | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **best_of_k_maha_a14** (Phase 14.F style baseline) | 0/30 | 2/30 (6.7%) | **22/30 (73.3%)** | **65.9** |
| **style_offset_llm maha_pooled** | 0/30 | 0/30 | 11/30 (36.7%) | 118.7 |
| style_offset_llm bhattacharyya | 0/30 | 1/30 | 15/30 (50%) | 103.1 |
| style_offset_llm w2 | 0/30 | 4/30 | 18/30 (60%) | 80.6 |
| style_offset_llm symmetric_kl | 0/30 | 2/30 | 16/30 (53.3%) | 90.9 |

### 3. 与 Phase 14.J 对比 (Phase 14.J D_off neutral vs J2 LLM neutral)

| Metric | Phase 14.J (D_off neutral) | **Phase 14.J2 (LLM neutral)** | 方向 |
|---|---|---|---|
| **q_offset norm mean** | ~30 | **685** | **×22 爆炸** |
| user_offset norm mean | ~50 | **71** | +42% |
| **量级比 q/user** | 0.6 (q < user) | **9.7 (q >> user)** | 倒挂更严重 |
| maha_pooled top-100 | 50% | **36.7%** (-13.3pp) | **恶化** |
| bhattacharyya top-100 | 63.3% | 50% (-13.3pp) | 恶化 |
| w2 top-100 | 36.7% | **60% (+23.3pp)** | 略改善 |
| symmetric_kl top-100 | 53.3% | 53.3% (0) | 持平 |

### 4. 为什么 LLM Neutral 更差

#### A. Query vs Sentence 结构差异

| 维度 | User 评论 (30 sents × 198 users) | A14 Query (30 × 30 pairs) |
|---|---|---|
| 长度 | 15-50 词 (完整句) | 8-15 词 (短查询) |
| 结构 | 主+谓+宾 (有骨架) | 名词短语堆叠 (无骨架) |
| LLM 改写 | 保留主谓宾,只去风格 | **重写为完全不同的句式,删除属性顺序** |
| Offset norm | ~71 (小改) | **~685 (大改)** |

#### B. 例子

```text
A14: "Find me a BabyBond silicone green colored product weighing 4.93 pounds with dimensions 16.2D x 13.24W x 13"
LLM neutral: "BabyBond silicone product, green, weight 4.93 lbs, dimensions 16.2\"D x 13.24\"W x 13\"H"

# 改写几乎删除所有句法骨架,只剩属性堆叠
# q_offset = a14_hidden - neutral_hidden → norm 685 (巨大!)
```

```text
User: "These diapers fit great on my baby and don't leak at all!"
LLM neutral: "The diapers fit the baby and do not leak."

# 改写保留主谓宾,只是去口语
# user_offset norm ~71 (适度)
```

#### C. q_offset norm 主导 Maha

- `pooled_var_offset` 是 user_offset.var 的均值,典型值 ~10
- q_offset norm 685 vs user_offset norm 71 → q_offset 几乎完全主导 Maha 距离
- 任何 user 只要 μ 稍微接近 q_offset 的方向,就能 rank 高 → 失去 rerank 区分度
- 实际上 11/30 top-100 (36.7%) 比 Phase 14.J 还差

### 5. Style Offset 范式最终 NO-GO

| 范式 | top-100 | mean_rank | 备注 |
|---|---|---|---|
| **Best-of-K Maha** (Phase 14.F SOTA) | **73.3%** | **65.9** | 单向量 rerank,无 offset |
| Phase 14.J (D_off neutral) | 50% (maha) | 102.4 | offset 量级不齐 1.7x |
| **Phase 14.J2 (LLM neutral)** | **36.7% (maha)** | **118.7** | **offset 量级爆炸 10x** |
| Absolute dist2dist (Phase 14.H) | 40-73% | 73-105 | K=8 已 SOTA |

**Style Offset 思路在 query 场景根本性失败**,无论 D_off 还是 LLM neutral。

### 6. Why Style Offset 思路本身不适用

1. **User 评论 vs Query 是不同 domain**:
   - 评论:有骨架、句式、改写保留骨架 → offset 小且一致
   - Query:无骨架、改写破坏全部结构 → offset 大且不一致

2. **Style Vector injection ≠ 风格偏移**:
   - StyleVector 是 Qwen hidden 空间的 linear shift,直接对齐 hidden 空间
   - 做差(orig - neutral)会把 StyleVector 的注入信号 + 语义残留 + 改写偏置混在一起

3. **Style Offset 假设 (LLM 改写后语义保持) 在 query 上失败**:
   - 评论可以保持语义但去风格
   - Query 改写往往改变属性顺序、删除连接词,语义表达方式完全改变

## 决策

### Style Offset 范式彻底失败

| Scenario | 推荐 | 备注 |
|---|---|---|
| **rerank** | **Phase 14.F K=8 best-of-K Maha** | top-100 86.7% (SOTA) |
| 备选 | Phase 14.J2 best_of_k_maha_a14 (K=30) | top-100 73.3% |
| **不推荐** | **任何 style offset rerank** | J 和 J2 全 NO-GO |
| 不推荐 | absolute dist2dist | K=8 best-of-K 已 SOTA |

### 不要做什么

- **不要再尝试 style offset rerank** — 已穷举 2 种 neutral 方案,全 NO-GO
- **不要 LLM 改写 query 当 neutral** — 改写破坏 query 结构,offset norm 爆炸
- **不要 D_off 当 neutral** — 量级 1.7x 不齐
- **不要再尝试新的分布对比范式** — 已穷举 4 种 (maha, BC, W2, KL)

### 下一步方向

1. **保持 Phase 14.F SOTA** — 不需切换
2. **如果要继续提升 rerank**,考虑:
   - 双 layer 融合 (22 + 26)
   - per-user σ² LW shrinkage
   - PCA 30d + best-of-K Maha (Phase 14.I 给的 hint)
   - 用户级 + query 级的 fine-grained Mahalanobis

## Pipeline 总结

1. **User 原评论 hidden** (Phase 14.B cache)
2. **User neutral 改写 hidden** (Phase 14.F-paired cache, 5940 × 5 × 3584d)
3. **A14 query hidden** (Phase 14.H 编码, 870 unique × 3584d)
4. **A14 query LLM neutral 改写 hidden** (本次生成, 870 × 3584d)
5. **Compute per-user style_offset**: user_orig - user_neutral (mean+var over 30)
6. **Compute per-pair style_offset**: A14 - A14_neutral (mean+var over 30)
7. **Rerank**: 4 metrics × style_offset + best-of-K Maha
8. **Aggregate**: rank-1, top-10, top-100, mean_rank

**总时长**: ~4.5 min (含 Qwen 加载 + 870 rewrites + 870×2 encoding + rerank)

## 工程

- **870 unique A14 rewrites**: 27 batches × REWRITE_BATCH=32, ~50s
- **870×2 encoding**: ~30s
- **Rerank CPU**: ~5s
- **Total**: ~4.5 min

## 文件

| Path | Purpose |
|---|---|
| `phase14_j2_llm_neutral_style_offset.py` | style offset LLM neutral rerank |
| `phase14_j2_a14_neutral_cache.jsonl` | 870 unique A14 → LLM neutral 改写 |
| `phase14_j2_a14_neutral_hiddens.npy` | 870 × 3584d |
| `phase14_j2_a14_residuals.npy` | 870 × 3584d (a14 - neutral) |
| `phase14_j2_a14_neutral_sents.jsonl` | paired (q_styled, q_neutral) |
| `phase14_j2_style_offset_eval.json` | 5 methods × 4 metrics eval |
| `phase14_j2_style_offset_per_pair.jsonl` | per-pair ranks |
| `phase14_j2_style_offset_meta.json` | metadata + norm diagnostic |

## 决策总结

**Phase 14.J2 NO-GO: LLM neutral query 不 work**。

**核心根因**:**Query 改写破坏结构**(query 是属性堆叠,改写为完全不同句式),offset norm 爆炸 22x,反而比 Phase 14.J (D_off neutral 1.7x 不齐) 更严重。

**SOTA 仍是 Phase 14.F K=8 best-of-K Maha (top-100 86.7%)**。

**Style Offset 范式根本性失败**(无论 D_off 还是 LLM),原因:
1. **User vs Query domain 差异**: 评论有骨架(改写保留),query 无骨架(改写破坏)
2. **Style vector 不是"风格偏移"**: StyleVector 是 hidden 空间的 linear shift,直接对齐即可,做差混合 StyleVector 信号 + 语义残留 + 改写偏置
3. **LLM neutral 假设在 query 上失效**: 评论改写保留语义,query 改写改变属性表达方式

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换
2. **不要再探索 style offset rerank** — 已穷举
3. **如果要继续提升 rerank**,考虑非 offset 方案:双 layer 融合 / PCA 30d + best-of-K

## 相关

- [[phase14j-style-offset-nogo]] — Phase 14.J D_off neutral parent
- [[phase14h-dist2dist-nogo]] — Phase 14.H K=30 absolute dist2dist
- [[phase14i-pca-dist2dist]] — Phase 14.I PCA + dist2dist PARTIAL-GO
- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F SOTA
- [[phase14f-paired-nogo]] — Phase 14.F-paired parent idea

当前任务已完成,请做下一个任务的指示。