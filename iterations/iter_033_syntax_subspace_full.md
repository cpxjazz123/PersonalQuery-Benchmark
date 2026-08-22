# iter 033: Syntax-only Subspace 完整结果 (Phase 1-4)

**日期**: 2026-08-22
**承接**: iter_032 (Phase 1 探索) + 用户反馈"review-space ≠ query-space"
**目的**: 验证 syntax-only Δh 是否能 (a) 提升 query syntax fidelity, (b) 降低 sentiment leakage

---

## Phase 1: PCA + correlation 探索 (iter_032)

发现: hidden residual 几乎被 sentiment/category 主导。**dim 0** 跨 layer 稳定
KEEP_SYNTAX (sentence_length 主信号); Layer 20 是 syntax-only dim 最多 layer
(5-dim: [0, 12, 15, 16, 17])。

## Phase 2: per-user Gaussian refit on syntax-only dims

20 users × (3-5 dim) syntax-only Gaussian。Δh_syntax norm:
- Δh_mean: 18-140 (mean ≈ 67)
- Δh_sample: 100-170 (与 mean 量级 2-3x → α 必须降到 0.1)

## Phase 3: query generation with syntax-only Δh

3 个 conditions 在同一 10 records × K=4 上:
- **D_off baseline** (no injection, strict_v8 prompt)
- **syntax_mean α=0.3** (z_u = mu_u deterministic)
- **syntax_sample α=0.1** (z_u ~ N(mu_u, Sigma_u) stochastic)

所有 conditions 都成功生成 query, 无 broken output (Phase 14 旧路径 α=1.0 有 broken)。

## Phase 4: 评估 (syntax_dist vs user self mean)

| condition                         | n  | syn_dist (lower better) | sent_leak (lower better) |
|-----------------------------------|----|--------------------------|---------------------------|
| D_off_baseline                    | 10 | 5.500                    | 0.358                     |
| syntax_L20_mean_a0.3              | 10 | 6.505 (+1.004, ✗)        | 0.210 (-41%, ✓)           |
| syntax_L20_sample_a0.1            | 10 | 5.724 (+0.223, ✗)        | 0.258 (-28%, ✓)           |
| **user-to-user natural floor**    | 20 | **1.849**                | —                         |

### 关键 takeaway

1. **Syntax-only Δh 显著降低 sentiment leakage** (mean 模式 -41%, sample 模式 -28%)
   → 验证用户假设: review-space residual 中的 sentiment/category 信号确实在
      syntax-only subspace 中被剥离。

2. **Syntax fidelity 没有提升** (反而略增)
   → 解释: query 的 syntax (sentence_length, POS ratios, function_word_ratio)
      主要由 generation pipeline + prompt 决定, hidden bias 是细微调整,
      不能 force 改变 query 句法骨架。
   → Query 离 user self sentence 的距离 (5-6) 仍然 ~3x 大于 user-to-user
      natural distance (1.849), 说明 generation 主导 query syntax。

3. **Sample vs Mean**: sample 模式 α=0.1 的 syntax_dist 比 mean 模式 α=0.3 更接近 baseline,
   但 sentiment_leak 也更高 (但仍优于 baseline)。Sample 模式不显著优于 mean。

## 结论 / 决策点

### Hypothesis 1 (syntax-only subspace 能 decorrelate sentiment) → ✓ CONFIRMED
   syntax-only Δh 注入比 direct residual 注入更"中性"。

### Hypothesis 2 (syntax-only Δh 能 force query 模仿 user syntax) → ✗ REFUTED
   注入 hidden bias 不能改变 query generation 的句法结构 (POS/length/function word 比例)。

### 实际意义

- **Direct residual injection (Phase 14 路线) 已经是从 hidden space 控制 query style 的极限**
- 想真正改变 query syntax, 需要的不是 Δh 注入, 而是 **in-context learning** (用 user
  exemplar 句作为 few-shot) 或 **fine-tuning** (把 user style 内化进模型)
- Phase 14.Q/Q10 路线 (exemplar + K-sample) 已经做到 18/30 rank-1 LOPO, 是该方向 SOTA

### 与已有 SOTA 对比

| 路线                                       | style fidelity | sentiment neutral | 备注 |
|--------------------------------------------|----------------|-------------------|------|
| **Phase 14.Q10 BoK-184 (enriched prompt + exemplars)** | SOTA 18/30 rank-1 | ✓ natural | in-context learning |
| **Phase 14.F rerank (Qwen residual space)** | 86.7% top-100 | ✓ | post-hoc selection |
| **syntax-only Δh injection (本 iter)**       | ~5-6 dist (no gain) | ✓✓ best | cleanest injection |
| **direct residual injection (旧路径)**       | ~5-6 dist (no gain) | ✗ leaks sentiment | noisy injection |

→ **Syntax-only injection 的价值是 clean (no sentiment leak), 不是 fidelity boost**.
→ Phase 14 路线 (Q10 BoK) 仍是 SOTA.

## 下一步建议

1. **NO Phase 5 production** — syntax-only Δh 没有显著优势
2. **如果继续 style injection 路线**: 在 syntax-only subspace 上加 **negative constraint**
   (强制 query 的 Δh 投影到 user style subspace), 但估计边际收益小
3. **切回 Phase 14.Q10/Q11 + Phase 14.F rerank**: 已被验证 SOTA, 不要再分散精力到注入路线

## 输出文件

- `syntax_subspace/phase1_explore.py` — Phase 1 (PCA + correlation)
- `syntax_subspace/phase2_refit_gaussian.py` — Phase 2 (per-user Gaussian)
- `syntax_subspace/phase3_query_gen.py` — Phase 3 (query generation)
- `syntax_subspace/phase4_evaluate.py` — Phase 4 (evaluation)
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/`:
  - syntax_features.npz, noise_features.npz
  - pca_layer_{16,20,24,26}.npz
  - correlation_layer_{16,20,24,26}.json
  - syntax_dims_summary.json
  - syntax_gaussian_layer_{16,20,24,26}.npz
  - syntax_gaussian_summary.json
  - phase4_results.json
- `result/query_records_with_query_inject_syntax_L20_a0.3_mean.json`
- `result/query_records_with_query_inject_syntax_L20_a0.1_sample.json`

## 最终 takeaway

> 用户假设 "review-space residual ≠ query-space style" 在数据上验证成立:
> - PCA correlation 表明 hidden residual 几乎被 sentiment/category 主导
> - syntax-only subspace 剥离 sentiment (Phase 4: -41% leakage) 验证 decorrelation 成功
> - 但 syntax-only injection 没有提升 syntax fidelity, 说明 hidden bias 不能
>   改变 query generation 的句法骨架
> → 真正可工作的路线是 Phase 14.Q/Q10 (in-context exemplars) 或 Phase 14.F
>   (Qwen residual rerank), 而不是 hidden state 注入
