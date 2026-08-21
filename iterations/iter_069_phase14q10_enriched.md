# Phase 14.Q10: Enriched User-Conditioned Prompt Generation (GO — new SOTA mean_rank)

## 背景

Phase 14.Q9 提示第一人称 prompt alone 弱 1/30,但作为 BoK 多样性贡献者
mean_rank 12.0 → 11.5 (-4%)。Phase 14.Q10 进一步 enrich prompt:

- **第一人称 narrative** (Q9 沿用)
- **Length target**: 用户平均句长 (中位数)
- **Top-3 distinguishing features** (20d z-score > 0.5 → natural language description)
- **4 first-person exemplars** (median length, top-fp)

**4 conds**:
- Q_EnrichedUserProfile: top-fp + median-length
- Q_EnrichedUserProfile_s1: top-fp + longest
- Q_EnrichedUserProfile_s2: top-fp + shortest
- Q_EnrichedUserProfile_s3: random fp

总共 30 pairs × 4 conds × K=8 = 960 queries。

## 设计

### 20d syntactic feature → natural language

每个用户 20d 均值与 population 均值/标准差算 z-score,选 |z| > 0.5 的 top-3
features,转换为 (low_desc, high_desc) 形式:

```python
clause_rate → "use simpler sentences" / "use richer clause structures"
passive_rate → "prefer active voice" / "use passive voice constructions"
opener_pron → "avoid opening with pronouns" / "open sentences with pronouns"
...
```

### Prompt 结构

```
Product attributes: {attrs}

User's typical review style (first-person narrative):
Example 1: {user_exemplar_1}
Example 2: {user_exemplar_2}
Example 3: {user_exemplar_3}
Example 4: {user_exemplar_4}

This user's distinguishing style features:
1. {feature direction 1} (strongly characteristic)
2. {feature direction 2} (moderately characteristic)
3. {feature direction 3} (strongly characteristic)

Task: Write a personal shopping note (about N words, 1-2 sentences) from this user's perspective.
Use first-person voice ('I want...', 'for my kids...', 'I love...').
Match the user's distinguishing style features above. Include every product attribute.
Match the user's expression style from the examples but do not copy them.
```

## LOPO 结果

| 配置 | cands | rank-1 | top100 | mean_rank |
|---|---|---|---|---|
| Q10_default_alone | 8 | 1/30 (3.3%) | 23/30 (76.7%) | 62.0 |
| Q10_4subsets_alone | 32 | 4/30 (13.3%) | 24/30 (80.0%) | 48.7 |
| Q8_8subsets_alone | 64 | 7/30 (23.3%) | 29/30 (96.7%) | 16.6 |
| cached_88 | 88 | 15/30 (50.0%) | 29/30 (96.7%) | 20.1 |
| BoK_88+Q10_default | 96 | 15/30 (50.0%) | 29/30 (96.7%) | 18.3 |
| BoK_152+Q10_default | 160 | 18/30 (60.0%) | 29/30 (96.7%) | 12.0 |
| BoK_152+Q10_4subsets | 184 | 18/30 (60.0%) | 29/30 (96.7%) | **10.9** |
| BoK_88+Q10_4subsets | 120 | 15/30 (50.0%) | 29/30 (96.7%) | 16.9 |
| **BoK_full_184** | **184** | **18/30 (60.0%)** | **29/30 (96.7%)** | **★ 10.9** |

**vs baseline BoK-152 (Q-8)**: rank-1 18/30 (持平), mean_rank **10.9** (-8% vs 12.0) ★
**vs Phase 14.Q-9 BoK-192**: rank-1 18/30 (持平), mean_rank **10.9** (-5% vs 11.5)

## 关键发现

### 1. Q10 alone 弱 (4/30 vs Q-8 alone 7/30)

Enriched prompt alone 反而比 Q-8 (DynamicExemplar4) 弱:
- 4 个 exemplars + features + length 一起给 LLM 信息过载
- LLM 倾向于"过度模仿"用户句式,但 query 类型仍是"产品描述"
- 用户 top-3 features 可能让 LLM 偏到 off-target 风格

### 2. Q10 + cached + Q-8 BoK-184 = 18/30 mean_rank 10.9 (-8% vs Q-8 baseline)

pool 多样性是关键:Q10 alone 4/30,**mean_rank 48.7**,
但加入 BoK-184 后帮助 5 pairs:
- AH5DZ3X7JLNN/B0C49Q5D1Q: Q-8 rank 61 → Q-10 rank 22 (在 BoK-184 中降回 22)
- AEGJUYLQ33YR/B084GY9388: Q-8 rank 44 → Q-10 rank 28
- AHV5LMB2ZZ3Q/B07LD7R21T: Q-8 rank 4 → Q-10 rank 0 (rank-1 命中)
- AE6SPZALZWZR/B0BYNRCHT9: Q-8 rank 1 → Q-10 rank 0 (rank-1 命中)
- AHOEMNCGNZJM/B08X1Q8K3B: Q-8 rank 1 → Q-10 rank 0 (rank-1 命中)

### 3. Q10 = Q9 (第一人称) 延续

Q10 与 Q9 的 AH5DZ3X7JLNN pair 改善一致 (54→41 in Q9, 61→22 in Q10 enriched),
证明 **首位为第一人称风格用户** 的 query generation 真的能接近 target,只是
单 prompt alone 7/30 弱,需要 others 一起来。

### 4. rank-1 18/30 ceiling 仍未突破

BoK-184 全配置 18/30 mean_rank 10.9,与 Q-combine (BoK-232) 18/30 mean_rank 10.0
差 1.0,意味着 **rank-1 ceiling 18/30 已经稳态**。后续提升必须:
- 用户级 residual → 软 token/prefix (类似 soft-prompt)
- LLM judge rerank
- 多视角 rerank 投票

## 决策

- **Q10 enriched user-conditioned prompt GO**: alone 弱 (4/30),
  but BoK-184 mean_rank **10.9** (-8% vs Q-8 baseline 12.0),新 SOTA ranking quality
- **rank-1 18/30 持平**: 候选扩展 + 多样化 prompt 已达 rerank ceiling
- **下一步**: 转向 generation-side user conditioning (Phase 14.Q11+)
  - 软 prefix / soft prompt (user residual → token embedding)
  - 或 LLM judge rerank (直接生成 score)
  - 或 per-user exemplar 大幅扩 (K=32+ per cond)

## 关键复用要点

- **Q10 alone 不能用**: 4/30, prompts 信息过载 → LLM 偏 off-target
- **Q10 作为 BoK diversity 来源**: 帮助 5/30 pairs 改进 mean_rank
- **首位用户 (first-person) 一致改善**: Q9/Q10 在 AH5DZ3X7JLNN 一致,证明
  user-conditioned narrative prompt 在特定用户上行得通
- **20d feature → NL 转换 有效**: LLM 接受自然语言描述,不需要数字
- **Length target 副作用小**: 用户平均句长 12 词,与 attr 列表长度匹配

## 输出文件

| Path | Purpose |
|------|---------|
| `phase14_q10_prep.py` | 4 enriched conds prep (features + exemplars + length) |
| `phase14_q10_generate.py` | vLLM batched K=8 gen (960 queries) |
| `phase14_q10_encode.py` | Qwen 5 layers encode |
| `phase14_q10_lopo.py` | LOPO BoK-full-184 |
| `phase14_q10_lopo_eval.json` | LOPO result (18/30 mean_rank 10.9) |
| `phase14_q10_lopo_per_pair.jsonl` | per-pair ranks |
| `phase14_q10_cand_residuals_qwen.npy` | (960, 5, 3584) |
| `phase14_q10_generated.jsonl` | 960 generated queries |
| `phase14_q10_generation_prompts.jsonl` | 120 generation prompts |
| `phase14_q10_user_conditions.json` | per-pair features + exemplars + avg_len |
| `phase14_q10_prep_meta.json` | prep metadata |
