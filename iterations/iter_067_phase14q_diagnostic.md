# Phase 14.Q-Diagnostic: 12 Hard Pairs — Off-target + Broken Pollution

## 背景

Phase 14.Q-8 / Q-combine 把 LOPO 18/30 (60.0%) 锁在 rerank ceiling 上,
Phase 14.Q-7 已被确认是当前最优 ranking (Q-8 + Q-combine 都无新增 top-1)。
问题: **12 个 missed pairs 失败原因是什么?为什么加 candidates 不能突破?**

## 5 维度诊断

### dim1: 真实用户 residual 与最佳候选的距离

12 pairs 全部 min_maha_target ∈ [101.808, 198.070] (PCA-300 Maha)
- 命中 pairs (rank 0) 估计 min_maha_target < 50
- **结论**: 12 missed pairs 的用户 μ 远在任何 generator 输出可达范围之外

### dim2: 最佳候选实际接近哪个用户? (off-target match)

**12/12 missed pairs 的 best cand 实际匹配的是其他用户,不是 target!**

| Off-target 用户 | 吸引的 missed pairs |
|---|---|
| **AFO2SXNWPO** | 4: AEGJUYLQ33, AEQCZRWLHY, AFR35RN5SZ, AH5ONINJRN |
| **AHDNIUCT** | 3: AG3CK7F7NP, AGPIQK2GLZ, AGV7MJIYWV |
| AE5IMGWR | 1: AGLWDF5SJN |
| AHDKNBCG | 1: AH5DZ3X7JL |
| AEJUJLXY | 1: AHA2VLVJNU |
| AGZ2367H | 1: AEP4KJDGBH |
| AGIDJMSM | 1: AHWREDWG6T |

**off-target ratio** (best_user maha / target maha):
- 0.607 (AEQCZRWLHY) → 0.953 (AGPIQK2GLZ)
- **结论**: rerank 失败的核心不是"找不到接近 target 的 cand",而是
  "cand 实际更接近 AFO2SXNWPO / AHDNIUCT 这两个特定用户"

### dim3: 用户风格稳定性 (intra-user residual std)

- σ_norm ∈ [1254.96, 1754.77] (12 pairs 全 > 1250)
- std_mean ∈ [4.18, 5.85]
- **0 narrow_user, 12 wide_user**
- **结论**: 失败的 12 个用户都写"风格很杂"的 reviews (std_mean 4-6 vs 一般 1-2),
  μ 不一定代表他们;generator 的 sentence style 不在这个分布内

### dim4: 句法模式覆盖 (AEQCZRWLHY rank 197 案例)

**该用户真实 reviews (第一人称叙事)**:
- "I LOVE giving my toddlers this product to eat their snacks out of."
- "It is indeed spill-proof, super soft for their hands to move in and out of, and it's cute!"
- "The top does not come off, so I hand wash it and then use the bottle sterilizer to clean it."
- "With 13 month old twins, I absolutely needed this for under their highchairs."

风格特征:
- 第一人称 (I/my/we)
- 个人叙事 (my toddlers, 13 month old twins)
- 因果连接 (so/and/indeed/because)
- 情感标记 (LOVE/great/amazing)

**Q_DynamicExemplar4_s4 生成 (第三人称产品描述)**:
- "Looking for a mushie brand silicone item in Cambridge Blue color?"
- "Ideal for feeding or as a teething tool, it's easy to hold and use."
- "Perfect for feeding your child confidently. Easy to hold and clean."

**结论**: 生成器偏向"looking for X" 产品描述,而 narrative user 写
第一人称故事;**句法模式覆盖严重不足**

### dim5: StyleVector broken pollution

| cond | broken 比例 (non-ASCII > 10%) |
|---|---|
| **A26_a1.0** | **32.1% (77/240)** |
| **A26_a0.5** | **14.2% (34/240)** |
| A14_a1.0 | 2.5% |
| A8_a0.5 / a1.0 | 3.3% / 1.7% |
| A18 / A22 / D_off | 0-0.8% |

**结论**: A26 StyleVector 高 α 让 LLM 输出乱码 (中文/特殊字符),
这些 broken cands 在 PCA 空间占"无用户区",污染 rerank 池;
**A26_a1.0 必须过滤**,否则永远误判 target ranking

## Rank 197 极差 pair (AEQCZRWLHY / B08QV7K3) 详细

### 12 conditions 全部 rank 197

- D_off: 197
- A14_a0.5/1.0, A22_*, A26_*, A8_*: 197
- Q_DynamicExemplar4_s1..s8: 197

**所有 generator 风格都"远离"该用户**,无法达成 top-1。

### 实际最佳 cand 文本
> "Looking for a silicone product in Cambridge Blue color with a brand of mushie, weighing exactly 2.12 ounces..."

### 该用户真实 reviews
- "I LOVE giving my toddlers this product to eat their snacks out of."
- "It is indeed spill-proof, super soft for their hands to move in and out of, and it's cute!"

**生成器永远不能写出第一人称叙事,query 必须是第三人称** → 风格不可避免冲突

## 决策

### 12 missed pairs 失败模式分类

| 模式 | 数量 | 解决方向 |
|---|---|---|
| **StyleVector broken pollution** | A26_a1.0 (32%) + A26_a0.5 (14%) | **过滤**: 删除 non-ASCII > 10% 的 cands |
| **Off-target match (AFO2SXNWPO attracts)** | 4 pairs | 改 generator,跳出"generic product desc" |
| **Off-target match (AHDNIUCT attracts)** | 3 pairs | 改 generator |
| **第一人称 narrative users** | 全部 12 (尤其 AEQCZRWLHY) | **第一人称 prompt**: "I want...", "for my kids..." |

### Phase 14.Q9 方向

1. **过滤 broken cands**: 删除含 > 10% 非 ASCII 的 cands
2. **第一人称 narrative 生成**: Q_NarrativeFirstPerson
   - prompt: "Write a personal shopping note from this user's perspective: 'I want...', 'for my kids...', 'I love...'"
   - exemplar: 选用户 narrative-heavy sentences (含 "I/my/we")
3. **保留 Q_DynamicExemplar4** 4 subsets + 8 subsets 作为 base

### 预期收益

- 过滤 broken → 减少 rerank pool 污染,可能 +1-2 rank-1
- 第一人称生成 → narrative users 风格匹配 +2-3 rank-1
- 合计可能 LOPO 18/30 → 21/30 (70%)

## 关键复用要点

- **off-target match 是核心问题**: 不是"找不到接近 target 的 cand",而是
  "cand 实际更接近 AFO2SXNWPO"
- **A26_a1.0 严重污染**: 32% broken outputs,必须过滤
- **第一人称 narrative 是新杠杆**: 用户写 "I/my kids/indeed" 风格,但
  generator 默认第三人称
- **不要再加 cands**: rerank ceiling 18/30 确认;下一步必须改 generator

## 下一步

1. **Phase 14.Q9-1**: 过滤 broken cands → 重新 BoK-152 eval (LOPO)
2. **Phase 14.Q9-2**: 第一人称 narrative prompt + 新 cond 生成
3. **Phase 14.Q9-3**: 合并 BoK-filtered + Q_Narrative → LOPO 验证

## 输出文件

| Path | Purpose |
|------|---------|
| `phase14_q_diagnostic.py` | 5-dim per-pair analysis |
| `phase14_q_diagnostic.json` | full diagnostic records |
| `phase14_q_diagnostic_missed.jsonl` | 12 missed pairs detailed |
