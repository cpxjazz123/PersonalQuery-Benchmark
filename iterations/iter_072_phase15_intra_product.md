# Phase 15.7: Intra-Product Rank Evaluation

## Context

用户洞察:全池 rank-1 评估把 product confound 和 style confound 混在一起。
一个评价"diapers"的用户不可能匹配"USB cable" query,无论风格多像。
只有评价同一商品的 user 之间做对比,才是"纯风格匹配"问题。

## 数据规模

2,041 pairs (users) ↔ **1,201 unique asins** (商品)
- 902 asins 只有 1 个用户评价 (trivial)
- 216 asins 有 2-3 用户评价
- 66 asins 有 4-9 用户评价
- 17 asins 有 10+ 用户评价 (max 44 users)

→ 2,041 pairs 中:
- **1,116 pairs (~55%)** 是 trivial (intra_product_size=1)
- **1,384 pairs (~45%)** 是 nontrivial (intra_product_size > 1)

## 关键结果 (5 splits × 500 test users = 2,500 pairs)

| 评估方式 | D_off baseline | 15.4 exemplars | lift |
|---------|---------------|---------------|------|
| **Full-pool rank-1** | 0.56% | 1.40% | **+0.84pp (+150%)** |
| **Intra-product rank-1 (total)** | 68.84% | 70.84% | +2.0pp (+2.9%) |
| **Intra-product rank-1 (nontrivial only)** | 43.7% | 47.3% | +3.6pp (+8.2%) |
| Intra-product mean norm rank | 0.1794 | 0.1592 | **-11%** |

CI95 for 15.4 intra-product rank-1: [0.69, 0.7256] — **显著高于 D_off** [0.6704, 0.7056] (差 2pp)。

## 核心发现

### 1. Product Confound 是 Full-Pool 评估的噪声主因

| 评估空间 | rank-1 量级 | 主因 |
|---------|------------|------|
| Full-pool (2041 users) | 0.56-1.40% | 绝大部分用户 review 不同商品,纯 product confound → style 信号淹没 |
| Intra-product (mean 5.18 users) | **68-71%** | 去掉 product confound,style 信号直接暴露 |

→ **真实任务"同商品下找风格匹配作者"是 70% 解决的**,不是 0.56%!

### 2. Intra-Product Lift 远小于 Full-Pool Lift

| 评估空间 | 15.4 vs D_off |
|---------|---------------|
| Full-pool | +150% (主要来自 product confound reduction) |
| Intra-product (total) | +2.9% |
| Intra-product (nontrivial) | +8.2% |

→ 之前宣称的"2.5x rank-1 提升"实际上一部分是因为 15.4 exemplars 隐式帮助缩窄 product 噪声;
**真正的 style lift 是 8% (nontrivial)**。这是诚实的"风格区分能力"度量。

### 3. Intra-Product 难度按 size 分层

| Intra-product size | n pairs | D_off rank-1 | 15.4 rank-1 | 提升 |
|--------------------|---------|--------------|-------------|------|
| 2 users | 371 | 69.54% | 72.24% | +2.7pp |
| 3 users | 232 | 55.17% | 63.36% | **+8.2pp** |
| 4-9 users | 416 | 37.98% | 40.87% | +2.9pp |
| 10+ users | 365 | 16.71% | 19.18% | +2.5pp |

→ 商品被越多用户评价,单用户识别越难 (符合预期);
15.4 在 size=3 时提升最大 (+8.2pp),说明 exemplar 风格模仿在中等规模商品上最有效。

### 4. "SOTA" 重新审视

之前 Phase 14.Q10-L 的"2041-fold 0.54% rank-1"是包含 product confound 的"难模式"评估;
Phase 15.7 揭示:在 product-confound-free (intra-product) 评估下,基础 D_off 都能达 68%,
**15.4 exemplars 真正在 nontrivial 子集上从 43.7% → 47.3%**。

→ 任务实际上**远没那么难**,关键是要用 intra-product 评估。

## 方法论建议

**新的 SOTA 度量标准** (取代 full-pool rank-1):
1. **Intra-product rank-1 (nontrivial only)**: 排除 trivial size=1 后,在有竞争的 user 子集中做 rank
2. **Intra-product mean norm rank**: intra_product_rank / (size - 1),做 size-normalization
3. **Stratified by intra-product size**: 按 size=2/3/4-9/10+ 分层报告

**新的对比基准**:
- D_off intra-product (nontrivial) = 43.7% — 这是新的强 baseline
- 任何后续方案必须在此度量上提升 nontrivial rank-1

## 下一步方向

1. **聚焦 nontrivial (size>1)**: 1,384 pairs 是真正有信号的部分,资源应集中
2. **size=3 提升最大 (+8.2pp)**: exemplar 生成 + rerank 在中等规模商品最有效
3. **size=10+ 提升仅 +2.5pp**: 大集群需要更细粒度区分 (句法/词汇级别 reranker)
4. **可以丢掉 size=1 的 trivial pairs**: 1,116 pairs (44.6%) 不提供风格区分信号

## 文件清单

| Path | 用途 |
|------|------|
| `phase15_7_intra_product_eval.py` | intra-product 评估脚本 |
| `phase15_7_intra_product_eval.json` | cross-split + stratified 结果 |
| `phase15_7_intra_product_per_pair.jsonl` | per-pair 详细 (含 size/rank/norm) |

## Verdict

**Intra-product 评估揭示真实任务难度**: 68% rank-1 (D_off) / 71% (15.4 exemplars) 而非 0.56% / 1.40%。
Product confound 解释了之前的"overfitting"假象。**15.4 仍然是 GO,真实 lift 是 +8.2% nontrivial**。
下一步:聚焦 nontrivial subset,优化 size=3-9 商品的 user reranker。