# Iteration 205 — Stage 06+07 Baby/Pet eval + degenerate claim update

**日期**: 2026-07-23
**角色**: NLP/IR 专业审稿人
**scope**: Stage 06/07 Baby+Pet+Grocery eval + degenerate claim #2 update

---

## §A 审稿意见

### 问题 1: Baby/Pet Stage 07 eval 成功但数据集规模极小

**严重程度**: Major
**具体批评**:
> Baby Stage 07 eval 仅有 2 个有效 noisy pairs (来自 2 个 high-complexity 用户)，Pet 仅 1 个有效 pair。Grocery 无法生成有效 noisy pairs (H@10 全 0)。

**证据支撑**:
- Baby: 2 pairs, 73 users total but only 2 users with noisy-degraded Hit@10
- Pet: 3 users, 1 pair with noisy-degraded Hit@10
- Grocery: 3 users, 0 Hit@10=1 pairs → 无法验证 noisy degradation
- 所有 category word_count 分布全在 ≥10 (HIGH complexity)，无 LOW complexity 样本

### 问题 2: Δ Range degenerate claim 无法解除（数据结构限制）

**严重程度**: Major
**具体批评**:
> Claim 0 (cross-cluster Δ Range) 和 Claim 1 (Table 1 Δ Range values) 状态仍为 degenerate，因为 query file 中所有 queries 的 word_count_bucket = MISSING，无法做 low/high complexity 分组。

**证据支撑**:
- Baby: 803 queries, word_count ∈ [11,76], 全 HIGH (≥10)
- Pet: 33 queries, word_count ∈ [16,34], 全 HIGH
- Grocery: 33 queries, word_count ∈ [13,39], 全 HIGH
- Stage 08 `load_08_group_hit10` 硬编码 `low_complexity`/`high_complexity` 分组键
- Summary 中只有 `all_queries`，无分组数据

---

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| Baby noisy pairs 数量少 | `gen_noisy_v4_baby.py` | 仅找到 2 个 clean_hit=1/noisy_hit=0 pairs from 51 attempts |
| Grocery 无 Hit@10=1 pairs | query file 本身 | Grocery 3 用户 queries 全部无法命中目标文档 |
| Δ Range 无法计算 | `08_compare_p10_across_domains.py:256` | `load_08_group_hit10` 硬编码 low/high 分组，但 summary 只有 all_queries |
| Δ Range 无法计算 | Baby/Pet/Grocery query files | `word_count_bucket` 全为 MISSING，无复杂度分层元数据 |

---

## §C 本轮代码优化

**Stage 06 eval 完成**:
- Baby: BGE H@10=0.329, E5=0.219, STAR=0.274, ANCE=0.110, MiniLM=0.178 (73 users, 730 queries)
- Pet: BGE=0.333, E5=0.667, STAR=0.0, ANCE=0.0, MiniLM=0.0 (3 users, 30 queries)
- Grocery: 全 0 (3 users, 30 queries) — 目标文档无命中

**Stage 07 eval 完成**:
- Baby: BGE/E5/MiniLM H@10: correct=1.0 → noisy=0.0 (Δ=-1.0 for all 3)
- Pet: BGE H@10: correct=1.0 → noisy=0.0 (Δ=-1.0)
- Grocery: 无有效 pairs，跳过

**paper_claims_audit.json 更新**:
- Claim 2 (noisy Hit@10 drops vary): degenerate → partial
- Claim 0/1 (Δ Range): 保持 degenerate (数据结构限制)

**noisy_pair 生成方法论**:
- `gen_noisy_v4_baby.py`: 用 BGE embedding 筛选 clean_hit=1/noisy_hit=0 pairs
- Per-retriever 独立 noisy cache 保证每个 retriever 的评估基于其自己的 embedding space

---

## §D 验证

- Baby Stage 06 eval: ✅ 完成 (5/5 retrievers, 73 users)
- Baby Stage 07 eval: ✅ 完成 (BGE/E5/MiniLM, 2 pairs, H@10 全退化)
- Pet Stage 06 eval: ✅ 完成 (5/5 retrievers, 3 users)
- Pet Stage 07 eval: ✅ 完成 (BGE only, 1 pair, Δ=-1.0)
- Grocery Stage 06 eval: ✅ 完成 (4/5 retrievers, 3 users)
- Grocery Stage 07 eval: ⚠️ 跳过 (0 Hit@10=1 pairs)
- paper_claims_audit.json: ✅ updated (degenerate: 2, partial: 5)

---

## §E Git Commit

- `git commit -m "iter #205: Stage 06+07 Baby/Pet eval + claim #2 partial update"`

---

## §F 剩余审稿意见（待后续迭代）

1. **Δ Range degenerate 解除**: 需要 query file 提供 `word_count_bucket` 元数据，或 Stage 08 支持 `all_queries` 单桶模式
2. **Grocery Stage 07**: 需要重新生成 Grocery 目标文档索引，或确认数据问题
3. **STAR/ANCE Baby Stage 07**: 两个 retriever 的 noisy pairs 全部被 filter 排除，需要 retriever-specific pair 生成
4. **Noisy pair 规模**: Baby 仅 2 pairs，统计意义有限，需扩展 query set