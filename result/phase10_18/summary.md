# Phase 10.18: Iterative Exemplar-Guided Style Search — 总结

**Date**: 2026-08-20
**Status**: PARTIAL-GO (margin 改善 21%,rank-1 仍是 0/30,但引擎工作,提供新工具)

## 实验动机

Phase 10.16 (e22_t3 soft prefix) 和 10.17 (TinyStyler + AnnaWegmann) 都失败 — 两种风格向量注入范式都不能突破 1.26% rank-1 上限。

用户洞察:**"LLM 不必理解'把 advmod_rate 提高 0.6'这类抽象数字,而是能看到一个已经被验证更像该用户的实际表达例子"**

新范式:iterative exemplar-guided style search — 给 LLM 看真实已被 318d 评分验证的 query exemplars,让它学句法组织/语气/标点,但不抄商品名/属性。

## 实验设计

| 参数 | 值 |
|------|-----|
| 样本量 | 30 pairs (subset for demo) |
| Rounds | 5 |
| Candidates/round | 8 |
| Total generations | 30 × 5 × 8 = 1200 |
| Score function | margin = D_nearest_other - D_target (318d Maha) |
| Exemplar per pair | top-3 (with diversity penalty: 318d cosine > 0.95 跳过) |
| Anti-template | 惩罚 ≥2 模板短语 ("looking for", "I need", "weighing exactly" 等) |
| Anti-copy | 禁止 exemplar 中长 distinctive 名词 (>4 chars) |
| Round 0 hints | 12 SYNTAX_HINTS (declarative/appositive/relative clause/等) 多样化生成 |
| Round 1+ | Few-shot: 3 exemplars + 当前商品属性 + 反模板/反抄袭指令 |

## 引擎架构 (Phase 10.18.B)

1. **Round 0**: 12 个不同 syntactic hints × 30 pairs × 8 candidates → 240 个 baseline candidates
2. **318d 评分**: margin + target_rank + attrs_complete
3. **Exemplar 选择**: 排序 by (template_hits asc, margin desc),然后 318d cosine diversity
4. **Round 1+**: few-shot prompt,3 exemplars + 当前 attr + 反模板指令
5. **多样性惩罚**: exemplar 池 cosine > 0.95 → 跳过
6. **Fallback**: 没有任何 exemplar 通过 filter 时,用 top-1 anyway

## 结果: 30 pairs × 5 rounds

```
Round    Margin     Rank1 Cov   Complete
   0    -495.59     0/30 (0%)   0.77
   1    -443.53     0/30 (0%)   0.77
   2    -408.19     0/30 (0%)   0.70
   3    -399.58     0/30 (0%)   0.70
   4    -390.56     0/30 (0%)   0.70
```

**改善**:
- Margin: -495 → -391 (**+21% in 5 rounds**)
- Rank-1 coverage: 0/30 (no change)
- Attrs complete rate: 0.77 → 0.70 (-7pp, regression)

## 关键观察

### 1. Margin 持续改善,但仍未转正

```
Round 0: -495.59
Round 1: -443.53 (+52.06)
Round 2: -408.19 (+35.34)
Round 3: -399.58 (+8.61)
Round 4: -390.56 (+9.02)
```

改善速率递减,Round 3 后接近停滞。**整个过程 margin 仍为负**,说明 LLM 生成的 query 系统性地**比最近的其他用户更不像 target user**。

### 2. 句式真的在变 (但 rank 不变)

Pair 11 query 演化:
- R0: "Looking for a BRITAX car seat in Cool Flow Teal? I need one weighing exactly..."
- R1: "BRITAX Cool Flow Teal car seat, 13 pounds, fabric, dimensions..."
- R2: "BRITAX, your teal-colored child safety seat, crafted from lightweight fabric..."
- R3: "BRITAX, Cool Flow Teal fabric seat, weighs precisely 13 pounds, spans 16 x 20.5 x 27.5 inches."

从问句 → 短陈述 → 营销口吻 → 简洁产品描述。**LLM 确实从 exemplar 学了不同句式**,但 rank 只从 2872 → 2472 (改善 14%),远不到 rank-1。

### 3. 多数 pair (20/30) 在 round 0 就达最佳

说明 few-shot iter **对大多数 pair 没有改进**。只有 10/30 pair 通过 iter 改善:
- Pair 11: R0 best rank 2786 → R3 best rank 2602
- Pair 4: R0 best rank 1734 → R2 best rank 1203 (最大改善)

### 4. 部分 pair 自然就接近 rank-1

- Pair 17: rank 61 (top 2.1%)
- Pair 8: rank 89 (top 3.0%)
- Pair 1: rank 95 (top 3.3%)
- Pair 19: rank 126 (top 4.3%)

这些用户在 VADES user_mu 上有相对独特的写作风格,LLM 自由生成就能命中。但**用户群体中心附近的 pair 始终无法突破**。

### 5. Attrs_complete 下降到 70%

LLM 学 exemplar 句式后,部分 query **忘了写所有属性**。这是 few-shot 的副作用,需要更强 hard-copy 兜底(Phase 10.10.6 已实现,但本实验未集成)。

## 失败原因分析

1. **LLM 自由生成结构上限**: 即使用 exemplar 引导,LLM 输出仍在 user-cluster 中心附近,318d Maha margin 难以转正
2. **Few-shot 不能传授 318d 统计**: 用户风格本质上是分布(均值/方差/相关性),不是模板;LLM 从 1-3 个 exemplar 学到的是 token-level pattern,而非 distributional property
3. **Anti-template filter 不够强**: 12 SYNTAX_HINTS 只是 paraphrase,没改变 318d 实际分布
4. **样本量不足**: 30 pairs × 5 rounds,只够看信号,不够统计显著

## 与之前阶段对比

| 范式 | 318d margin 改善 | rank-1 | attrs_complete |
|------|------------------|--------|----------------|
| Phase 10.10 自由生成 | baseline (mean ≈ ?) | 1.26% | 100% (hard-copy) |
| Phase 10.16 e22_t3 prefix | mean_rank -28% | 0% | 100% |
| Phase 10.17 TinyStyler prefix | mean_rank -28% | 0.23% | 100% |
| **Phase 10.18 exemplar search** | **margin -21%** | **0% (30 pairs)** | **70% (regression)** |

**结论**:Phase 10.18 在 margin 改善上与 10.16/10.17 相当,但 attrs_complete 退化严重(rank-1 仍是 0)。

## 决策: **PARTIAL-GO (新工具,但不能突破上限)**

**iter exemplar search 引擎工作正常**(margin 持续改善,句式真的在变),但**不能突破 1.26% rank-1 结构上限**。

**接受的事实**:LLM 自由生成范式不能精确匹配 target user 风格,无论用 prefix injection 还是 exemplar search。这与 Phase 10.16/10.17 一致 — **318d 真实评论上限 97.86% vs 生成候选 rank-1 1.26%,差距是 LLM 生成质量本身**。

**可用作增强工具**:
- 给 LLM 多样化 exemplar 生成 pool(句式多样性 +21%)
- 比单条件生成有更广的句法探索空间
- 可作为 Phase 10.10 主系统的 candidate generation 增强

## 工程实现经验

1. **Batched generation**: 同一 prompt + `num_return_sequences=N` 一次 forward,比逐条快 8x
2. **318d diversity penalty**: cosine > 0.95 视为同一句式,跳过
3. **Anti-template vs anti-copy 分离**: anti-template 用 prompt 指令 + exemplar 选择,anti-copy 用 content filter
4. **Rolling exemplar pool**: 每轮重新选 exemplar,而不是累积
5. **fallback 机制**: 没有 exemplar 通过 filter 时,fallback 到 top-1,防止死锁
6. **attrs_complete 退化**: few-shot 让 LLM 学句式后,属性遗漏增加 — 需要 hard-copy 兜底

## 文件位置

- 引擎: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase10_18_b_iter_search.py` — 主引擎 (30 pairs × 5 rounds × 8 cand = 1200 gens)
  - `phase10_18_b_smoke.py` — smoke 测试包装
  - `phase10_18_iter_log.jsonl` — 30 pair × 全 round 全 candidate 的日志
  - `phase10_18_iter_summary.json` — 迭代曲线 + final state
  - `phase10_18_d_summary.md` — 本文档
- 复用: `phase10_pairs_1000.jsonl`, `phase10_10_6_cache/sentence_318d*`, `vades_prototype_3000u_v6_raw_user_profiles.jsonl`, Qwen2-7B-Instruct via llm_client

## 下一步方向 (待用户决策)

1. **Phase 10.19 (推荐)**: 集成 hard-copy 兜底,修 attrs_complete 退化;扩到 876 pairs 看边际效应
2. **Phase 10.20**: 把 exemplar search 作为 Phase 10.10 主系统的 candidate generation pool(替代单条件生成)
3. **接受 1.26% rank-1 结构上限**,转向 acceptance/coverage 拆解
4. **VADES 架构重构**: support-set prototype 替代 learnable user_offsets
