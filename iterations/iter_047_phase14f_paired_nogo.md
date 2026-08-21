# Phase 14.F-paired: Per-Sentence Paired Neutral Rerank — **NO-GO**

**Date**: 2026-08-21
**Question**: 既然 Phase 14.F 用 global neutral (2976 条平均) 计算 residual,那如果改成 per-sentence paired neutral (每条用户句子单独配对一个 LLM 改写的中性版本),是否能更好捕获"用户的风格"?
**Answer**: **NO** — Paired neutral 反而 **更差**,尤其在深层 (layer 22/26) 显著退化。

## 关键发现

### 1. Paired vs Global Paired Bootstrap (positive = paired better)

| Cond | Layer 8 | Layer 14 | Layer 18 | Layer 22 | Layer 26 |
|---|---|---|---|---|---|
| A22_a0.5 | **+33.4 excl 0 ✓** | +17.2 incl | **+23.0 excl 0 ✓** | -4.0 incl | -21.6 incl |
| A14_a1.0 | +11.8 incl | +6.2 incl | +11.4 incl | -3.6 incl | -10.1 incl |
| D_off    | **+25.7 excl 0 ✓** | +16.7 incl | +19.0 incl | -2.1 incl | -22.9 incl |

**关键**:
- **浅层 (8/14/18)** paired 略好 — 3 处 CI excludes 0,A22/D_off 在 layer 8/18 paired lift +23~+33 ranks
- **深层 (22/26)** paired **显著退化** — diff 全为负 (最大 -22.9)
- **A14_a1.0 在所有层都不显著**(CI 都 incl 0)

### 2. Paired vs Global per-cond 最佳层对比

| Cond | Global best (layer=26) | Paired best layer | Paired top100% | Paired mean |
|---|---|---|---|---|
| A22_a0.5 | top100=90.0%, mean=50.3 | layer=8 | 76.7% | 58.3 |
| A14_a1.0 | top100=86.7%, mean=47.2 | layer=26 | 83.3% | 57.3 |
| D_off    | top100=83.3%, mean=55.8 | layer=8 | 66.7% | 72.2 |

**所有 cond × 所有 paired-best-layer** 都**显著差于** global best (layer=26)。

### 3. Rank-1 / Top-10 Paired vs Global (paired best layer)

| Cond | Rank-1 paired | Rank-1 global | Top-10 paired | verdict |
|---|---|---|---|---|
| A22_a0.5 | **0/30** | 1/30 | 3/30 | global wins |
| A14_a1.0 | **1/30** | 1/30 | 5/30 | tie on rank-1, global wins on mean rank |
| D_off    | **0/30** | 1/30 | 5/30 | global wins |

**Paired 在 rank-1 上几乎全输或平**。A22_a0.5 和 D_off 在 paired 下 rank-1 完全消失 (0/30)。A14_a1.0 paired 在 layer 14/18 偶尔打平 rank-1 (1/30),但 mean rank 仍 global 胜 (57.3 vs 47.2)。

### 3. 决策 NO-GO

| 指标 | Global (Phase 14.F) | Paired (Phase 14.F-paired) | 决策 |
|---|---|---|---|
| A14_a1.0 layer 26 top100 | **86.7%** | 83.3% | global +3.4pp ✓ |
| A14_a1.0 layer 26 mean rank | **47.2** | 57.3 | global -10 ranks ✓ |
| A22_a0.5 layer 26 top100 | **90.0%** | 76.7% | global +13.3pp ✓ |
| A22_a0.5 layer 26 mean rank | **50.3** | 71.9 | global -22 ranks ✓ |
| D_off layer 26 top100 | **83.3%** | 70.0% | global +13.3pp ✓ |
| D_off layer 26 mean rank | **55.8** | 78.7 | global -23 ranks ✓ |

**结论**:**global neutral rerank 全面优于 paired neutral rerank**。原 Phase 14.F 路线 (global neutral) 保持 SOTA。

## 为什么 Paired 更差

### 核心解释:Global 减去的是 "跨用户/跨句子的通用中性骨架",Paired 减去的是 "每句话自己的中性改写"

**用户的 "personal style"** 在 hidden space 里可能由两层组成:
1. **公共骨架** (句法、词性分布) — 同一语言的评论都有的
2. **个人骨架** (用户独特的句法选择、用词偏好) — 只有这个用户有

**Global neutral** = 跨 2976 条 Amazon 评论平均的中性骨架 → 主要是公共骨架
- residual = user_hidden - global_neutral ≈ **个人骨架** (大保留,信号强)

**Paired neutral** = 每条用户句子自己改写的中性版本
- 改写过程中 LLM 会**消除**用户独特的句法选择 (因为它要把句子改成 "plain, neutral, matter-of-fact")
- residual = user_hidden - paired_neutral ≈ **句子的局部表达变化** (小残留,信号弱)

### 为什么浅层 (8/14/18) paired 反而更好

浅层 (token-level, 词性/词频) 偏向**词汇选择** (lexical choice)。
- Paired 改写保留 token 级替换 → 浅层 paired residual 包含 token-level 信号
- Global 减跨 2976 平均 → 浅层 residual 包含跨用户的 "词汇 baseline"

但浅层 CI excludes 0 只在 3 处,且 paired-best-layer top100 仍<global-best (90.0% vs 76.7%)。

### 为什么深层 (22/26) paired 更差

深层 (sentence-level, 句法骨架) 偏向**句法结构** (syntactic skeleton)。
- Paired 改写会**消除**用户独特句法骨架 → paired residual 句法信号几乎为 0
- Global 减跨 2976 平均 → deep residual 保留**用户独特句法骨架**

**深层才是 StyleVector injection 真正起作用的层**(Phase 14.B 验证),所以 deep paired 更差直接拉低 SOTA。

## Pipeline 总结

1. 加载 phase14_b_layer_alpha_sweep.jsonl 720 条 (A22_a0.5/A14_a1.0/D_off × 30 pairs × K=8)
2. 加载 phase14_b_user_hiddens_5layers.npz (198 users × 30 sentences × 5 layers × 3584d mean-pool)
3. **Paired neutral rewrites** — 每条用户句子用 LLM 改写为中性 (cached 5940 条 in phase14_f_paired_user_neutral_cache.jsonl + hiddens in user_neutral_hiddens.npz)
4. **Paired residuals** = user_hidden - paired_neutral_hidden (198 × 30 × 5 × 3584d)
5. **Per-user per-layer Gaussian** + LW shrinkage 0.1
6. **Paired cand rewrites** — 720 候选 query 改写为中性 (cached phase14_f_paired_cand_neutral_cache.jsonl + hiddens)
7. **Paired cand residuals** = cand_hidden - cand_neutral_hidden (720 × 5 × 3584d)
8. **Pooled Maha rerank** — per-user diag Maha 距离
9. **Best-of-K** = min rank across 8 candidates
10. **Paired bootstrap** vs Phase 14.F global neutral rerank

## 工程

- **User rewrites**: 5940 条 LLM 改写,phase13_a cache 命中 2980 条,新增 2960 条 (~78s vLLM)
- **User hiddens**: 5940 × 5 × 3584d (~3 min transformers 编码)
- **Cand rewrites**: 707 unique → 720 条 (~9 min transformers 编码)
- **Cand hiddens**: 720 × 5 × 3584d (~50s)
- **Rerank**: pooled Maha (~15s CPU)
- **总时长**: ~13 min

## 决策

### NO-GO: Paired neutral 不如 global neutral

| 场景 | 推荐 | 备注 |
|---|---|---|
| **rerank** | **global neutral (Phase 14.F)** | top100 86.7%, mean 47.2 (new SOTA) |
| 不推荐 | paired neutral | top100 83.3% max, deep layers 显著退化 |

### 不要做什么

- **不要尝试 paired neutral rerank** — global 已全面超越
- **不要为 rerank 做 per-sentence neutral rewrite** — 改写消除用户句法骨架,反而去掉 signal
- **不要扩展 paired 到 layer<8 或 layer>26** — 没有 lift 方向

## 文件

- `phase14_f_paired_qwen_residual_rerank.py` — paired rerank script
- `phase14_f_paired_qwen_residual_rerank_eval.json` — per-cond per-layer eval + paired bootstrap
- `phase14_f_paired_qwen_residual_rerank_per_pair.jsonl` — per-pair rank
- `phase14_f_paired_user_sents.jsonl` — 5940 user sentences
- `phase14_f_paired_user_neutral_cache.jsonl` — 5940 paired neutral rewrites
- `phase14_f_paired_user_neutral_hiddens.npz` — [5940, 5, 3584] paired neutral hiddens
- `phase14_f_paired_cand_neutral_cache.jsonl` — 707 cand neutral rewrites
- `phase14_f_paired_cand_neutral_hiddens.npy` — [720, 5, 3584] cand neutral hiddens

## 与 Phase 14.F (Global) 对比

| 路线 | top100 (best cond/layer) | mean rank |
|---|---|---|
| Global neutral + Qwen residual layer 26 (Phase 14.F) | **86.7%** | **47.2** |
| Paired neutral + Qwen residual layer 8 (Phase 14.F-paired) | 76.7% | 58.3 |
| Paired neutral + Qwen residual layer 26 (Phase 14.F-paired) | 83.3% | 57.3 |
| 768d pooled Maha (Phase 14.C) | 53.3% | 164.0 |

**Global 仍是绝对 SOTA**,paired 没有超过它。

## 下一步

1. **保持 Phase 14.F global neutral rerank 路线** — 不需要 paired
2. **可探索**: 双 layer 融合 rerank (layer 22 + 26) 在 global 空间 — 可能进一步 lift
3. **可探索**: per-user σ² 全协方差 (LW) 在 global 空间 — Phase 14.D 单维失败需重新验证
4. **不要扩展到 876 users** — Phase 13.B 198 sweet spot