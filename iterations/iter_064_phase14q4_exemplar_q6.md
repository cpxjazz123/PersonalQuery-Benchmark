# Phase 14.Q-6: 4 Exemplars + Median Length Selection (PARTIAL-GO)

## 背景

Phase 14.Q-5 LOPO 验证 BoK-96 (88 + Q_DynamicExemplar2) = 15/30 (与 SOTA 持平)。下一步验证:
1. **加 exemplar 数**：从 2 → 4 看 Q_DynamicExemplar4 是否更好
2. **residual-space closest-to-μ 选择**：原 median word count 太粗，try 残差空间 closest to 用户 μ

## 实验设计

**Q-6-1 prep**: 30 pairs × 4 exemplars per user。**Note**: 最初尝试 residual-space (用 user_hiddens[30, 3584] 残差选 closest to μ),但 user_hiddens cache 只有 30 sents/user,与 user_sents 完整池大小不匹配。**Fallback 到 median-length 选择** (n=4),与 Phase 14.Q-1 逻辑相同但 exemplar 数翻倍。

**Q-6-2 generate**: vLLM batched 60 prompts × 8 = 480 queries in 5.8s (82.1 prompts/s)

**Q-6-3 encode**: transformers 480 queries × 5 layers in 4s

**Q-6-4 eval**: 11 缓存 + Q-2 (2-exemplar) + Q-6 (4-exemplar) = 13 conds × 8 = 3120+480 = 3600 cands
(同时持有 Q-2 与 Q-6 便于对比)
**PCA-300+PCA-500 α=0.25** fixed on (user + 2640 cached),Q cands 投影到 fixed basis

**Q-6-5 LOPO**: 30-fold validation of BoK-96 (88 + Q_DynamicExemplar4)

## Full-fit 结果

| 配置 | cand数 | rank-1 | top-100 | mean_rank |
|---|---|---|---|---|
| BoK_88 (Phase 14.P-5 SOTA) | 88 | 15/30 (50.0%) | 29/30 | 20.1 |
| BoK_96 (88 + Q2) | 96 | 16/30 (53.3%) | 29/30 | 15.5 |
| **BoK_96 (88 + Q4)** | **96** | **★ 17/30 (56.7%)** | 29/30 | **12.6** |
| BoK_104 (88 + Q2 + Q4) | 104 | 17/30 (56.7%) | 29/30 | 12.4 |
| BoK_120 (88 + Q_Dynamic + Q2 + Q4) | 120 | 17/30 (56.7%) | 29/30 | 12.4 |
| Q_DynamicExemplar4 (4 exemplar) | 8 | **4/30 (13.3%)** | 27/30 | 34.4 |
| Q_DynamicExemplar (2 exemplar) | 8 | 1/30 (3.3%) | 27/30 | 41.8 |
| Q_Dynamic4 (text only, 4-exemplar prompt) | 8 | 0/30 (0%) | 22/30 | 77.8 |

## LOPO 验证 (BoK-96 = 88 + Q_DynamicExemplar4)

```
rank-1: 15/30 (50.0%)
top-10: 18/30 (60.0%)
top-100: 29/30 (96.7%)
mean_rank: 19.1 (CI95 [8.3, 34.1])
```

**LOPO 完全等同 Phase 14.Q-5 (15/30, mean_rank 19.1)**。full-fit +2 rank-1 是 over-fit
到 dev 30 pairs,不是稳定提升。

## 关键发现

### 1. 单 cond 4-exemplar 显著优于 2-exemplar (PARTIAL-GO)

- Q_DynamicExemplar4: rank-1 **4/30 (13.3%)**, mean_rank **34.4**
- Q_DynamicExemplar: rank-1 1/30 (3.3%), mean_rank 41.8
- 提升 **+3 rank-1 (4x)**, mean_rank **-18%**

**结论**: 4 exemplars 给 LLM 更丰富的风格样本,即使没有 StyleVector injection,也能
生成更接近用户风格的 query。

### 2. 但 BoK-96 整体无 LOPO 提升 (NO SOTA)

- Full-fit: BoK-96 (88+Q4) = 17/30 (vs 88 alone 15/30, +2)
- LOPO: 15/30 (等同 Phase 14.P-5 SOTA), mean_rank 19.1 (等同 Phase 14.Q-5)

**结论**: 候选池已饱和 (88→96),rerank 上限被 rerank 本身的能力限制,而非候选数。
exemplar 数量提升是边际的。

### 3. Q_Dynamic (text only) 4-exemplar 仍 NO-GO

rank-1 0/30, mean_rank 77.8 — 与 Phase 14.Q-1 (0/30, 78.4) 完全一致。
**结论**: 文字条件 alone 永远失败,即使有 4 exemplars 的 prompt context, text-only
条件 (modifier_density 等) 仍不被 LLM 理解。

### 4. Q4 单 cond 比 2-exemplar 更好但都比完整 StyleVector 差

- Q_DynamicExemplar4 (4-exemplar): 4/30, mean_rank 34.4
- A14_a1.0 (StyleVector): 3/30, mean_rank 47.3
- **Q4 比 A14_a1.0 +1 rank-1, mean_rank -27%**

**结论**: 4 exemplars 的 generation + plain residual rerank 已略微超过 StyleVector
injection + rerank,这是 generation 优于 StyleVector 的第一个迹象。

## SOTA 状态

| 阶段 | 配置 | rank-1 (full-fit) | rank-1 (LOPO) | mean_rank (LOPO) |
|---|---|---|---|---|
| Phase 14.P-5 | BoK-88 | 15/30 | 15/30 | 20.1 |
| Phase 14.Q-5 | BoK-96 (88+Q2) | 16/30 | 15/30 | 19.1 |
| **Phase 14.Q-6** | **BoK-96 (88+Q4)** | **★ 17/30** | **15/30** | **19.1** |

**没有 LOPO 改进**。候选池 rerank 在 88-96 cands 已被 PCA pooled Maha 触及上限。

## 决策

- **保持 Phase 14.P-5 BoK-88 (rank-1 15/30, mean_rank 20.1) 作为 SOTA** — LOPO 一致
- Q4 单 cond 4/30 比 2-exemplar 1/30 强 4x,这是 generation 方向的真正杠杆
- **下一步**: 把 Q4 的 8 cands 加到 rerank 池不解决问题 (saturated),但 Q4 single cond
  单独看就有 4/30。下一步应探索 **Q4 + 之前的 query 候选** 的混合策略,或
  **Q4 × 多个 cond (如不同 exemplar subsets)** 扩充到 K=16/32

## 输出文件

| Path | Purpose |
|------|---------|
| `phase14_q6_prep.py` | 4 exemplars 选择 (median length) |
| `phase14_q6_user_conditions.json` | 30 pairs × 4 exemplars |
| `phase14_q6_generation_prompts.jsonl` | 60 prompts |
| `phase14_q6_generate.py` | vLLM batched K=8 gen |
| `phase14_q6_encode.py` | Qwen 5 layers encode |
| `phase14_q6_cand_residuals_qwen.npy` | (480, 5, 3584) |
| `phase14_q6_evaluation.py` | 2-PCA maha rerank with 4-exemplar |
| `phase14_q6_eval.json` | 9 subsets summary |
| `phase14_q6_per_pair.jsonl` | 30 pairs × 7 subsets |
| `phase14_q6_lopo.py` | LOPO 30-fold |
| `phase14_q6_lopo_eval.json` | LOPO result |

## 性能

- Q-6-1 prep: 22s (876 pairs + 30 pairs dynamics)
- Q-6-2 generate: 5.8s (vLLM batched)
- Q-6-3 encode: 24s (transformers)
- Q-6-4 eval: 14s (12 subsets × 30 pairs)
- Q-6-5 LOPO: 4s (30 folds)
- **总耗时 ~70s**

## 复用要点

- **OOM 教训**: vLLM + transformers 同时加载 7B 模型 → 35.96 GiB out of 39.49 GiB → OOM
  - 解决: 拆成 gen (vLLM only) + encode (transformers only) 两个独立脚本
- **PCA basis fixed**: 拟合只用 (user + 2640 cached), Q cands 投影到 fixed basis
- **exemplar selection**: median length 简单可靠;residual-space closest-to-μ 需要 user
  完整池 residual encoding (cache 只有 30 sents/user 不够)
- **cond 命名**: 复用 `Q_Dynamic` (text only) 和 `Q_DynamicExemplar` (2-exemplar);
  Q-6 引入 `Q_DynamicExemplar4` (4-exemplar) 作为新 cond
