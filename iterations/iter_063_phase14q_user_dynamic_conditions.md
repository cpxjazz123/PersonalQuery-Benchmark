# Phase 14.Q: 用户动态条件 + Exemplar 风格引导生成

## 背景

Phase 14.P-5 SOTA (rank-1 15/30, mean_rank 20.1) 验证**最有希望的杠杆是候选数**。但本质是
**先生成通用 query → 再用 rerank 挑最像用户的**。用户提出更优范式:**先生成时就读用户
风格 → rerank 只做最后校正**。

## 用户提出的 5 步方案

1. 提取 user style target (residual r_u,i = h(orig) - h(neutral))
2. 把 style → text conditions (不用 raw 300/500 向量)
3. 用 user exemplars 2-4 句历史
4. 迭代生成 + 反馈
5. 训练 Adapter/LoRA

## 实验设计

**Q-1 准备**: 30 pairs × 20-d syntactic vector → z-score → top-5 distinct dims → 文字条件;
2 exemplars (从 user_sents_cache2 中选 5-30 words 句子, word count 接近 median)

**Q-2 生成**: vLLM batched K=8 per (pair × cond). 60 prompts × 8 = 480 queries in 5.8s (82.8 prompts/s)

**两种条件**:
- Q_Dynamic: 文字条件 only
- Q_DynamicExemplar: 文字条件 + 2 exemplars

**Q-3 编码**: 480 Q queries 通过 Qwen 5 layers mean-pool → 减去 global neutral → 残差 (480, 5, 3584)

**Q-4 评估**: 11 缓存 conds + 2 Q conds (共 13 conds × 8 = 3120 cands),用 Phase 14.P-5 SOTA rerank
**PCA-300+PCA-500 α=0.25**。**关键**——PCA 拟合只在 (user + 2640 缓存) 上,Q cands 投影到固定 basis
(否则 Q cands 把 basis 移到偏差方向,BoK-88 也会从 15/30 掉到 8/30)。

## 完整结果 (full-fit, PCA-300+PCA-500 α=0.25)

| 配置 | cand数 | rank-1 | top-100 | mean_rank |
|---|---|---|---|---|
| **BoK_120_88plus32c** (88+32) | **120** | **★ 16/30 (53.3%)** | 29/30 | **15.5** |
| **BoK_88_plus_q_8c** (88+8) | **96** | **★ 16/30 (53.3%)** | 29/30 | **15.5** |
| BoK_88_88c (Phase 14.P-5 baseline) | 88 | 15/30 (50.0%) | 29/30 | 20.1 |
| BoK_5_conds_40c | 40 | 7/30 (23.3%) | 29/30 | 23.6 |
| A14_a1.0_8c (StyleVector) | 8 | 3/30 (10.0%) | 25/30 | 47.3 |
| Q_DynamicExemplar_8c | 8 | 1/30 (3.3%) | 27/30 | 41.8 |
| D_off_8c (uniform) | 8 | 1/30 (3.3%) | 22/30 | 68.1 |
| Q_Dynamic_8c (text only) | 8 | **0/30 (0%)** | 22/30 | 78.4 |

## LOPO 验证 (BoK-96 = 88 + Q_DynamicExemplar)

```
rank-1: 15/30 (50.0%)
top-10: 18/30 (60.0%)
top-100: 29/30 (96.7%)
mean_rank: 19.1 (CI95 [8.3, 34.1])
```

**LOPO 比 Phase 14.P-5 (15/30, mean_rank 20.1) 略优 (mean_rank -5%)**。full-fit 的 +1 是噪声。

## 关键发现

### 1. Q_Dynamic 文字条件 alone 严重退化 (NO-GO)

`rank-1 0/30, mean_rank 78.4` — 比 uniform D_off 更差。**结论**: 20-d syntactic vector 抽
象出的文字条件 (modifier_density, clause_rate 等) 没有真正帮助 LLM 控制风格。LLM 不知
道"fewer clauses"具体指什么。

### 2. Q_DynamicExemplar 接近 StyleVector (PARTIAL-GO)

`rank-1 1/30, mean_rank 41.8` — 与 A14_a1.0 (3/30, 47.3) 相当。**2 个 exemplar 提供了
足够具体的目标风格信号** —— LLM 能模仿例句的措辞和句子结构。

### 3. Q cands 加入 rerank 池带来边际 +1 (full-fit) / 0 (LOPO)

- Full-fit: 88+(8) → 16/30 (vs 88 -> 15/30), 96 cands 池多 1 个 rank-1
- LOPO: 15/30 = Phase 14.P-5 SOTA, mean_rank 19.1 vs 20.1 (-5%)

**结论**: Q_DynamicExemplar 1 个 cond (8 cands) 边际贡献在 ±0 ~ +1,不是根本性的 SOTA 突破。
真正的杠杆仍是候选数(11 conds × 8 = 88)。

### 4. Q_Dynamic (text only) 在 96 池中拖后腿

将 Q_Dynamic (8 cands, mean_rank 78.4) 加入 88 池,实际只增加 96 cands。但 BoK_120
(88+32) = 16/30 = BoK_88_plus_q_8c (88+8) = 16/30 → **Q_Dynamic 16 cands 贡献了 0 个 rank-1
改善**。结论,Q_Dynamic 完全没用,只是白加候选数。

## 与 Phase 14.P-5 SOTA 比较

| 指标 | Phase 14.P-5 (BoK-88) | Phase 14.Q (BoK-96) | Δ |
|---|---|---|---|
| rank-1 (full-fit) | 15/30 (50.0%) | 16/30 (53.3%) | +1 (+6.7%) |
| rank-1 (LOPO) | 15/30 (50.0%) | 15/30 (50.0%) | 0 |
| mean_rank (full-fit) | 20.1 | 15.5 | -4.6 (-23%) |
| mean_rank (LOPO) | 20.1 | 19.1 | -1.0 (-5%) |
| top-100 | 29/30 | 29/30 | 0 |

**LOPO 验证下 rank-1 无提升, mean_rank 略输 5%**。Phase 14.Q 仅在 full-fit 下有 +1,
边际噪声。

## 决策

- **保持 Phase 14.P-5 BoK-88 (rank-1 15/30, mean_rank 20.1) 作为 SOTA** — LOPO 一致
- 88 池已满,加 Q cands (尤其 Q_Dynamic) 是边际噪声
- **下一步**: 增加 exemplar 数量 (3-4 个) + 改进选择 (residual space closest to μ, 非
  median length),看能否突破 15/30

## 输出文件

| Path | Purpose |
|------|---------|
| `phase14_q_prep.py` | 准备 per-user 条件 + exemplars |
| `phase14_q_user_conditions.json` | 30 pairs × 20-d vector + z-scores + conditions + exemplars |
| `phase14_q_generation_prompts.jsonl` | 60 prompts (2 conds × 30 pairs) |
| `phase14_q_generate.py` | vLLM batched K=8 generation |
| `phase14_q_generated.jsonl` | 480 Q queries |
| `phase14_q_encode.py` | Encode 480 queries through Qwen 5 layers |
| `phase14_q_cand_residuals_qwen.npy` | (480, 5, 3584) |
| `phase14_q_evaluation.py` | Q conds vs control eval **with fixed PCA basis** |
| `phase14_q_eval_fixed_pca.json` | 12 subsets summary |
| `phase14_q_lopo.py` | LOPO 30-fold for BoK-96 |
| `phase14_q_lopo_eval.json` | LOPO result |

## 性能

- Q-1 prep: 8s (876 pairs population + 30 pairs dynamics)
- Q-2 generate: 5.8s (480 queries, vLLM batched)
- Q-3 encode: 60s (480 queries × 5 layers)
- Q-4 eval (full-fit): 67s (12 subsets × 30 pairs)
- Q-5 LOPO: 4.3s (30 folds × 96 cands)
- **总耗时 ~3 min**

## 关键复用要点

- **vLLM batched**: `client._backend.model.generate(chat_prompts, sampling)` 一次传所有
  prompts (60 × 8 = 480),变体 suffix `(variant N/8)` 避免 vLLM dedup (CLAUDE.md Rule 4/5)
- **Qwen hidden**: `client.get_hidden_states(texts, layers=[8,14,18,22,26], batch_size=32)`
- **PCA basis fixed**: 拟合只能用 (user + cached cands), 新 cands 投影到 fixed basis
  才能公平对比
- **Conds mapping**: 20-d syntactic feature → 文字条件 (Z_THRESHOLD=0.5, top-5 by |z|)
- **Exemplar selection**: 5-30 words 句, word count closest to median (后续待改进)
