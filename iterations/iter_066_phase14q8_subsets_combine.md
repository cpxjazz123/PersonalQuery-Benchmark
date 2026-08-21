# Phase 14.Q-8: 8 Q4 Subsets + All-Phase Combine — Saturation Confirmed

## 背景

Phase 14.Q-7 用 4 种不同 exemplar subsets 取得 LOPO 18/30 (60.0%) 新 SOTA。
问题: **更多 subsets (8 种) 能否继续突破?** **Q-1/Q-6 老 cands (Q_Dynamic /
Q_DynamicExemplar1/4) 与 Q-8 8-subsets 合并能否进一步提升?**

## 实验设计

**Phase 14.Q-8**: 8 种 exemplar subsets (s1..s8) per user
- s1 = median (Phase 14.Q-6/Q-7)
- s2 = median+5 (longer)
- s3 = median-3 (shorter)
- s4 = random (seed=user_id)
- **s5 = longest 4** (max word count)
- **s6 = shortest 4** (min word count)
- **s7 = alternating odd** (1st/3rd/5th/7th closest to median)
- **s8 = alternating even** (2nd/4th/6th/8th closest to median)

每个 subset 4 exemplars × K=8 → 30 pairs × 8 subsets × K=8 = **1920 queries**

**Phase 14.Q-combine**: 合并所有 Q phase candidates
- 88 cached (Phase 14.B) + 96 Q-8 8-subsets (含 Q-7 重叠)
  + 24 Q-1 (Q_Dynamic + Q_DynamicExemplar) + 24 Q-6 (Q_Dynamic + Q_DynamicExemplar4)
  = **232 cands per pair**

注意: Q-7 的 s1..s4 labels 与 Q-8 的 s1..s4 labels 重复,导致 8 subsets 合并后
实际包含 96 cands/pair (32 from Q-7 + 64 from Q-8)。Q-1 和 Q-6 的 Q_Dynamic
也重复(16 cands/pair)。

## Full-fit (PCA basis fixed on user + 2640 cached)

| 配置 | cand数/pair | rank-1 | top-100 | mean_rank |
|---|---|---|---|---|
| Q_Dynamic (Q-1+Q-6) | 16 | 0/30 | 22/30 | 77.8 |
| Q_DynamicExemplar (Q-1) | 8 | 1/30 | 27/30 | 41.8 |
| Q_DynamicExemplar4 (Q-6) | 8 | 4/30 | 27/30 | 34.4 |
| Q7 4-subsets (s1..s4, 含 Q-8 重叠) | 64 | 5/30 | 29/30 | 17.5 |
| **Q8 8-subsets** | **96** | **7/30 (23.3%)** | 29/30 | **15.8** |
| cached_88 (Phase 14.P-5) | 88 | 15/30 | 29/30 | 20.1 |
| cached+Q1_all | 112 | 16/30 | 29/30 | 15.5 |
| cached+Q6_all | 112 | 17/30 | 29/30 | 12.6 |
| cached+Q7_4subsets | 152 | 18/30 | 29/30 | 11.9 |
| cached+Q8_8subsets | 184 | 18/30 | 29/30 | 11.7 |
| cached+Q8+Q1 | 208 | 18/30 | 29/30 | 11.7 |
| **cached+all (Q8+Q1+Q6)** | **232** | **18/30 (60.0%)** | 29/30 | **★ 10.0** |

## LOPO (Phase 14.Q-7 与 Q-8 标准 LOPO)

| 配置 | rank-1 (LOPO) | mean_rank (LOPO) |
|---|---|---|
| BoK-88 (Phase 14.P-5) | 15/30 (50.0%) | 20.1 |
| BoK-120 (Q-7, 88+32) | **18/30 (60.0%)** | **11.9** |
| **BoK-152 (Q-8, 88+64)** | **18/30 (60.0%)** | **12.0** |

**Phase 14.Q-8 LOPO**: rank-1 = 18/30, mean_rank = 12.0 (CI95 [2.6, 26.3]) —
与 Phase 14.Q-7 完全持平!

## 关键发现

### 1. 8 subsets 已饱和 (rank-1 ceiling at 18/30)

**BoK-152 (88 + 64 Q4 8-subsets) LOPO 18/30 = BoK-120 (88 + 32 Q4 4-subsets)**
- 单 subset 最佳 s1: 2/30 → 全 8 subsets: 7/30 (+5)
- 但合并到 BoK-152 后,rerank 找不到额外的 top-1 hits
- mean_rank 11.9 → 12.0 也几乎无变化
- **结论**: subset diversity 在 4 subsets 已经饱和,再加 4 个 selection 策略
  不能让 rerank 突破 18/30 的硬 ceiling

### 2. Q-1 / Q-6 老 cands 与 Q-7/Q-8 合并提升 mean_rank 但不提升 rank-1

**BoK-232 (全 Q phases) rank-1 18/30, mean_rank 10.0 (-16% vs 11.9)**
- 88 cached + 96 Q-8 8-subsets + 24 Q-1 + 24 Q-6 = 232 cands
- mean_rank 从 11.9 → 10.0 (+5 个 ranking quality)
- 但 rank-1 卡在 18/30 (12 pairs 永远找不到对的候选)
- **结论**: rank-1 是硬 ceiling,只有 mean_rank 还能微调;Q-1 / Q-6 老 cands
  的 K=8 候选与 Q-7/Q-8 K=8 候选部分重合(都是同一 group 的 LLM 风格)

### 3. Rerank ceiling 确认在 18/30

过去 6 个 phase 一直卡在 LOPO 15/30 (Phase 14.P-5 SOTA),
Phase 14.Q-7 用 4 subsets 把 ceiling 推到 18/30,
Phase 14.Q-8 + Q-1 + Q-6 加到 232 cands 都没法突破 18/30。

可能的解释:
- **12 个 pairs 是"固有难例"**: 该用户的 exemplar 与 μ 偏离太大,任何 LLM 生成
  都达不到 rerank top-1
- **PCA Maha 自身能力限制**: 18/30 可能就是 layer 26 + PCA-300/500 这个
  rerank 模型的天花板
- **生成器天花板**: Qwen-7B 在某些 pair 上完全不能产生匹配用户风格的输出

## SOTA 进展

| 阶段 | 配置 | LOPO rank-1 | LOPO mean_rank |
|---|---|---|---|
| Phase 14.M | BoK-24 (4 conds × 8) | 4/30 | 91.4 |
| Phase 14.O | BoK-24 (3 conds × 8) | 5/30 | 25.3 |
| Phase 14.P-5 | BoK-88 (PCA-300+500 α=0.25) | 15/30 | 20.1 |
| Phase 14.Q-5/6 | BoK-96 (88 + Q4 1 subset) | 15/30 | 19.1 |
| **Phase 14.Q-7** | **BoK-120 (88 + 32 Q4 subsets)** | **★ 18/30** | **11.9** |
| **Phase 14.Q-8** | **BoK-152 (88 + 64 Q4 8-subsets)** | **★ 18/30** | **12.0** |
| Phase 14.Q-combine (full-fit) | BoK-232 (88 + 96 + 24 + 24) | 18/30 (full-fit) | 10.0 (full-fit) |

## 决策

- **新 SOTA 维持 Phase 14.Q-7: BoK-120 LOPO 18/30 (60.0%), mean_rank 11.9**
- **Phase 14.Q-8 8 subsets 与 Phase 14.Q-7 4 subsets 等效** (rerank ceiling 已被 4 subsets 触顶)
- **Phase 14.Q-combine 232 cands 在 full-fit 下 mean_rank 10.0** (vs 11.9) —
  仅 ranking quality 提升,无 rank-1 增益
- 进一步增加 candidate 数是边际效应,**rank-1 ceiling 在 18/30 是真实硬墙**

## 下一步方向

1. **更换 rerank 模型/空间**: 当前 PCA-300+500 α=0.25 在 layer 26 残差空间
   已被 18/30 顶住;可试:
   - **Qwen2-7B 不同 layer (24/27/28) + 不同 α**
   - **768d AnnaWegmann style 空间** (Phase 13.F rerank 视角)
   - **Qwen2-1.5B 更小模型残差** (虽然 Phase 15.E NO-GO)
   - **prompt-level rerank** (LLM judge style match)

2. **生成器升级**:
   - **更多 K (16/32)**: 增加候选多样性可能找到 hard cases 的合适表达
   - **更长更结构化 prompts**: 给 LLM 更具体的风格指令 (而不是 only 4 exemplars)
   - **Chain-of-thought generation**: 让 LLM 先解释用户风格,再写 query

3. **其他 SOTA 杠杆**:
   - **per-pair α calibration**: 不同 pair 用不同 α + 不同 pca_dim
   - **rerank weight learning**: 用 LOPO train fold 学习 (cached, Q1, Q4 subsets) 权重

## 输出文件

| Path | Purpose |
|------|---------|
| `query_gen/phase14_q8_prep.py` | 8 different exemplar subsets (median/+5/-3/random/longest/shortest/alt-odd/alt-even) |
| `query_gen/phase14_q8_generate.py` | vLLM batched K=8 gen (1920 queries) |
| `query_gen/phase14_q8_encode.py` | Qwen 5 layers encode (separate, avoid OOM) |
| `query_gen/phase14_q8_evaluation.py` | 2-PCA Maha rerank + 14 conds full-fit |
| `query_gen/phase14_q8_lopo.py` | LOPO 30-fold BoK-152 |
| `query_gen/phase14_q_combine.py` | Full-fit combine all Q phases (232 cands) |
| `phase14_q8_user_conditions.json` | 30 pairs × 8 subsets × 4 exemplars |
| `phase14_q8_generation_prompts.jsonl` | 240 prompts |
| `phase14_q8_generated.jsonl` | 1920 generated queries |
| `phase14_q8_cand_residuals_qwen.npy` | (1920, 5, 3584) |
| `phase14_q8_eval.json` | full-fit 14 cond comparison |
| `phase14_q8_per_pair.jsonl` | per-pair full-fit detail |
| `phase14_q8_lopo_eval.json` | LOPO BoK-152 result |
| `phase14_q8_lopo_per_pair.jsonl` | per-pair LOPO rank |
| `phase14_q_combine_eval.json` | full-fit combine all Q phases |
| `phase14_q_combine_per_pair.jsonl` | per-pair combine detail |

## 性能

- Q-8 prep: ~10s
- Q-8 gen: ~30s (vLLM batched 1920 queries, peak 18k tok/s)
- Q-8 encode: ~85s (transformers, batched)
- Q-8 evaluation: ~10s
- Q-8 LOPO: ~6s
- Q-combine: ~10s
- **总耗时 ~150s**

## 关键复用要点

- **subset diversity 4 subsets 已饱和** — 加到 8 不再突破 rank-1 ceiling
- **rerank ceiling 在 18/30**: 8 subsets + 232 cands 都无法突破
- **mean_rank 还有微调空间** (-16% 从 11.9 → 10.0) 但不能转化为 top-1 hits
- **下一步**: 换 rerank 模型/空间 或 升级生成器,而不是再加 candidates
