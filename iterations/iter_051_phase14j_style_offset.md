# Phase 14.J: Style Offset Rerank — **NO-GO** (Best-of-K Maha 仍是 SOTA)

**Date**: 2026-08-21
**Question**: 既然用户有30 条原评论和30 条 neutral 改写,相减得到"用户风格偏移"分布;如果我们对每条 query 也生成 K 条 A14 (有风格) 和 K 条 D_off (无风格),相减得到"query 风格偏移"分布,两个分布能否 match 上?
**Answer**: **NO-GO**。style_offset 分布对比比 absolute 分布对比差,而且需要 LLM neutral 改写 (成本高) 但 lift 仍是负。

## 关键发现

### 1. 3 个方法对比 (Qwen layer 26, 198 users, K=30 candidates)

| Method | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **best_of_k_maha_a14** (Phase 14.F SOTA 风格, mean + best-of-K) | 1/30 | **6/30 (20%)** | **23/30 (76.7%)** | **59.5** |
| absolute_a14 maha_pooled | 1/30 | 1/30 | 21/30 (70%) | 75.6 |
| absolute_a14 bhattacharyya | **2/30** | 2/30 | 18/30 (60%) | 77.6 |
| absolute_a14 symmetric_kl | 0/30 | 2/30 | 18/30 (60%) | 88.7 |
| absolute_a14 w2 | 0/30 | 1/30 | 12/30 (40%) | 104.6 |
| **style_offset** maha_pooled | 0/30 | 3/30 | 15/30 (50%) | 102.4 |
| style_offset bhattacharyya | 1/30 | 2/30 | 19/30 (63.3%) | 90.9 |
| style_offset symmetric_kl | 1/30 | 1/30 | 16/30 (53.3%) | 99.5 |
| style_offset w2 | 0/30 | 2/30 | 11/30 (36.7%) | 117.1 |

### 2. Style Offset 思路

**User side (Phase 14.F-paired 已有)**:
```
30 原评论 hidden @ layer 26 → μ_orig
30 LLM-neutral 改写 hidden → μ_neutral
user_style_offset = μ_orig - μ_neutral (per-dim, 然后 mean+var over 30)
```

**Query side (新生成,Phase 14.H 已有 K=30)**:
```
K=30 A14_a1.0 query hidden → μ_q_styled
K=30 D_off query hidden → μ_q_off
q_style_offset = μ_q_styled - μ_q_off
```

**Rerank**: 对比 user_style_offset 分布 vs q_style_offset 分布

### 3. 为什么 Style Offset NO-GO

**核心问题:neutral 不一致**

- **User neutral**: LLM 把原评论改写成 "plain/neutral",**强中性**(完全去除口语/句法)
- **Query neutral**: D_off baseline(不注入 StyleVector),**弱中性**(只是 base LLM 输出)
- **两者量级完全不同** → offset 不在同一个 scale

**实际数值**:

| 偏移源 | norm 量级 |
|---|---|
| user_orig residual | ~75 |
| user_neutral residual | ~40 |
| **user_offset = orig - neutral** | **~50** |
| q_a14 residual | ~95 |
| q_d_off residual | ~78 |
| **q_offset = a14 - d_off** | **~30** |

**q_offset (~30) << user_offset (~50)**,量级不齐 → Mahalanobis 距离被 user_offset 量级主导。

### 4. 3 个方法 lift 对比 (vs absolute_a14 maha_pooled baseline)

| Method | top-100 lift | mean_rank lift |
|---|---|---|
| best_of_k_maha_a14 | **+6.7pp** | **-16 ranks** |
| absolute_a14 bhattacharyya | -10pp | +2 ranks |
| absolute_a14 symmetric_kl | -10pp | +13 ranks |
| **style_offset** maha_pooled | **-20pp** | **+27 ranks** |
| style_offset bhattacharyya | -6.7pp | +15 ranks |

**Best-of-K 是唯一 lift 的**。Style Offset 是最大退化。

### 5. 与 Phase 14.F SOTA 对比

| Method | top-100 | mean_rank | 来源 |
|---|---|---|---|
| **Phase 14.F K=8 best-of-K Maha** (SOTA) | **86.7%** | **47.2** | phase10 first-30 pair |
| Phase 14.J best_of_k_maha_a14 (K=30) | 76.7% | 59.5 | Phase 14.B 30 pair |
| Phase 14.J style_offset bhattacharyya | 63.3% | 90.9 | Phase 14.B 30 pair |

**Best-of-K 仍是 best**。style_offset 全 NO-GO。

## 决策

### Style Offset Rerank NO-GO

| Scenario | 推荐 | 备注 |
|---|---|---|
| **rerank** | **Phase 14.F K=8 best-of-K Maha** | top-100 86.7% (SOTA) |
| 备选 | Phase 14.J best_of_k_maha_a14 (K=30) | top-100 76.7% (Phase 14.B 30 pair) |
| 不推荐 | style_offset rerank | top-100 36.7-63.3% |
| 不推荐 | absolute dist2dist | top-100 40-70% (Phase 14.H/I) |

### 不要做什么

- **不要探索 style offset rerank** — neutral 不一致导致 offset 量级不齐
- **不要 LLM 改写 query 当 neutral** — 成本高 + 仍然偏移量级不齐
- **不要 absolute dist2dist** — K=8 best-of-K 已 SOTA
- **不要再尝试新的分布对比范式** — 已穷举 4 种 (maha, BC, W2, KL)

## Pipeline 总结

1. **User 原评论 hidden** (Phase 14.B cache, 198 × 30 × 5 × 3584d)
2. **User neutral 改写 hidden** (Phase 14.F-paired cache, 5940 × 5 × 3584d)
3. **Query A14 + D_off K=30 candidates hidden** (Phase 14.H 编码, 1800 × 3584d)
4. **Compute per-user style_offset**: user_orig - user_neutral (per-dim mean+var over 30)
5. **Compute per-pair style_offset**: A14 - D_off (per-dim mean+var over 30)
6. **Rerank**: 3 methods × 4 metrics = 12 settings (style_offset × 4 + absolute_a14 × 4 + best-of-K Maha)
7. **Aggregate**: rank-1, top-10, top-100, mean_rank

**总时长**: ~4 min (含 Qwen 加载 + encoding + rerank)

## 工程

- **Per-user offset**: 198 × 30 = 5940 → mean+var per dim (~5s CPU)
- **Per-pair offset**: 30 × 30 × 2 = 1800 candidates → 60 offsets (~1s)
- **Encoding**: 1800 × 1 layer × 3584d (~1 min)
- **Rerank**: 60 pairs × 198 users × 3 methods × 4 metrics = ~30s CPU
- **Total**: ~3 min

## 文件

| Path | Purpose |
|---|---|
| `phase14_j_style_offset_rerank.py` | style_offset rerank script |
| `phase14_j_style_offset_eval.json` | 3 methods × 4 metrics eval |
| `phase14_j_style_offset_per_pair.jsonl` | per-pair ranks |
| `phase14_j_style_offset_meta.json` | metadata |

## 决策总结

**Phase 14.J NO-GO: style offset rerank 在 Phase 14.B 30 pair 上不 work**。

**核心根因**:user neutral (LLM 改写) vs query neutral (D_off baseline) **量级不一致** → offset 不匹配。

**SOTA 仍是 Phase 14.F K=8 best-of-K Maha**。

**Why this direction 失败**:
1. **Neutral 不一致**: user 用 LLM 改写(强),query 用 D_off(弱),导致偏移量级差 1.7x
2. **Style vector injection 本身不是"风格差异"**: StyleVector 是 Qwen hidden 空间的 linear shift,直接对齐 hidden 空间,无需做差
3. **User offset 包含"语义残留"**: LLM 改写虽然意图中性,但实际仍含语义,而 query offset 是纯风格 — 两者不在同一空间

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换
2. **不要再尝试 style offset / offset-based rerank** — 已穷举
3. **如要继续提升 rerank**,考虑:
 - 双 layer 融合 (22 + 26) in global space
 - per-user σ² LW shrinkage in global space
 - candidate mean + Maha in PCA 30d (Phase 14.I)

## 相关

- [[phase14h-dist2dist-nogo]] — Phase 14.H K=30 absolute dist2dist NO-GO
- [[phase14i-pca-dist2dist]] — Phase 14.I PCA + dist2dist PARTIAL-GO
- [[phase14f-paired-nogo]] — Phase 14.F-paired paired neutral NO-GO (parent of style offset idea)
- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F SOTA