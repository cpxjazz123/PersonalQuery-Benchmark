# Phase 14.Q9: First-person Narrative Generation + Broken Filtering (PARTIAL-GO)

## 背景

Phase 14.Q-Diagnostic 揭示 12 missed pairs 失败模式:
1. **Off-target match**: 12/12 的 best cand 实际更接近 AFO2SXNWPO (4) 或 AHDNIUCT (3) 等其他用户
2. **StyleVector 污染**: A26_a1.0 32% broken outputs, A26_a0.5 14% broken
3. **第一人称 narrative users**: 用户写 "I/my kids/indeed" 但 generator 默认第三人称

Phase 14.Q9 探索两个修正方向:
- **Q9-1**: 过滤 broken (non-ASCII > 10%) cands
- **Q9-2**: 第一人称 narrative generation (Q_NarrativeFirstPerson + 5 conds)

## Phase 14.Q9-1: 过滤 broken cands

| cond | broken ratio |
|---|---|
| A26_a1.0 | 32.1% (77/240) |
| A26_a0.5 | 14.2% (34/240) |
| A14_a1.0 | 2.5% |
| A8_a0.5 / a1.0 | 3.3% / 1.7% |
| D_off / A18 / A22 | 0% |

**LOPO BoK-152 FILTERED 结果: 17/30 (56.7%), mean_rank 12.4**

**vs baseline BoK-152: 18/30 (60.0%), mean_rank 12.0**

**结论: 过滤 broken = NO-GO (-1 rank-1)**。Broken cands 实际不污染 rerank
对 target user 的 ranking — 它们在 PCA 空间占"乱码区",target user 排名
靠前是因为 cand 整体距离 target 不远。删除它们反而减少了 pool 多样性。

## Phase 14.Q9-2: 第一人称 Narrative Generation

### 设计

**5 个 conds**:
- Q_NarrativeFirstPerson (default)
- Q_NarrativeFirstPerson_s1 (top-fp + longest)
- Q_NarrativeFirstPerson_s2 (top-fp + shortest)
- Q_NarrativeFirstPerson_s3 (random fp)
- Q_NarrativeFirstPerson_s4 (alt-fp-rank)

每个 cond 4 first-person exemplars × K=8 → 30 pairs × 5 conds × K=8 = **1200 queries**

### Exemplar selection

按 first-person pronouns (I/my/we/our) 数量排序,优先 fp-positive sentences:
- 0 fp-positive → fallback top-n by fp_score then length

**30/30 pairs** 都有 >=1 first-person exemplar

### Generation 验证

mean fp pronouns per cand = **6.33-6.82**, **0/240 zero-fp cands** →
prompt engineering 100% 成功

### LOPO 结果

| 配置 | cands | rank-1 | mean_rank |
|---|---|---|---|
| Q9_default_alone | 8 | 1/30 (3.3%) | 77.8 |
| Q9_5subsets_alone | 40 | 1/30 (3.3%) | 48.2 |
| Q8_8subsets_alone | 64 | 7/30 (23.3%) | 16.6 |
| cached_88 | 88 | 15/30 (50.0%) | 20.1 |
| BoK_88+Q9_default | 96 | 15/30 | 18.6 |
| BoK_152+Q9_default | 160 | 18/30 (60.0%) | 12.0 |
| **BoK_152+Q9_5subsets (= BoK_full_192)** | **192** | **18/30 (60.0%)** | **★ 11.5** |

### Per-pair diff (Q-8 vs Q-9)

| pair | Q-8 rank | Q-9 rank | diff |
|---|---|---|---|
| AH5DZ3X7JL / B0C49Q5D | 54 | 41 | +13 (Q-9 better) |

**Q-9 improved 1 pair (out of 30), regressed 0**

### 关键发现

1. **Q9 alone 表现很差** (1/30 vs Q8 alone 7/30): 第一人称 prompt 本身产生
   风格"过窄",反而集中到特定 users (off-target)
2. **Q9 + cached + Q-8 BoK-192 = 18/30 mean_rank 11.5** (-4% vs 12.0 baseline):
   略提升 ranking quality 但不突破 rank-1 ceiling
3. **AH5DZ3X7JL pair 改善** (54 → 41): 一位用户的第一人称 narrative 接近
   target,但还不够 top-1

### 决策

- **Q9 first-person narrative generator PARTIAL-GO**: alone 弱,但 +BoK-152 微调 mean_rank
- **rank-1 ceiling 18/30 仍未突破**
- 第一人称 prompt 让 LLM 生成"个人购物笔记"风格,但 query 类型本身
  与"产品描述/询问"有冲突,导致 alone 弱

## 关键复用要点

- **A26 StyleVector broken pollution 不影响 rerank**: 过滤 broken 反而 -1 rank-1
- **Q_NarrativeFirstPerson alone = 1/30**: 第一人称 prompt 风格太窄,off-target
- **Q_NarrativeFirstPerson 与 Q_DynamicExemplar4 orthogonal**:
  BoK-192 mean_rank 11.5 < BoK-152 12.0 (-4% ranking quality)
- **第一人称 prompt 不是 rank-1 突破杠杆**: ceiling 18/30 仍然存在

## 下一步方向

1. **Phase 14.Q10 - 多维度 prompt engineering**:
   - 同时第一人称 + 具体场景 ("for my toddler's lunch", "for my morning routine")
   - + target_length 控制 (短 vs 长)
   - + sentiment 控制 ("positive experience" vs "concerned about quality")
2. **Phase 14.R - rerank 模型升级**:
   - LLM judge rerank: 让 Qwen 直接判 style match score
   - 多视角 rerank: layer 14 + 22 + 26 投票
3. **Phase 14.S - 真正的 user-conditioned generation**:
   - 把 user residual 投影到 token embedding 空间 (类似 soft prompt)
   - 在 LLM 内部注入 user style information

## 输出文件

| Path | Purpose |
|------|---------|
| `query_gen/phase14_q9_filter.py` | 过滤 broken cands + LOPO |
| `query_gen/phase14_q9_prep.py` | 5 first-person narrative conds prep |
| `query_gen/phase14_q9_generate.py` | vLLM batched K=8 gen (1200) |
| `query_gen/phase14_q9_encode.py` | Qwen 5 layers encode |
| `query_gen/phase14_q9_lopo.py` | LOPO BoK-full-192 |
| `phase14_q9_filter_lopo_eval.json` | Q9-1 LOPO result (NO-GO 17/30) |
| `phase14_q9_lopo_eval.json` | Q9-2 LOPO BoK-192 (18/30 mean_rank 11.5) |
| `phase14_q9_cand_residuals_qwen.npy` | (1200, 5, 3584) |
| `phase14_q9_generated.jsonl` | 1200 generated queries |
