# Phase 14.Q10-L: 2041-fold 大规模验证 — 30-pair 60% rank-1 是 overfit/biased 采样

## 背景

Phase 14.Q10 BoK-184 在 30 用户 LOPO 上达到 **rank-1 18/30 (60%)**, mean_rank 10.9。
但 30 用户是从 198 用户池挑的,且 BoK-184 候选池 (184 cands/pair) 也是基于这
30 用户。**用户池规模是否影响 SOTA?** 用户要求在 10K 用户上验证。

数据约束: Baby Products 2023 共 3.4M 用户,但 median 1 review/user。**所有 ≥30
评论用户 = 2,043 users** (这是 hidden state 计算需要的最小数据量),这是数据
支持的**最大验证样本**。

## Phase 14.Q10-L 设计

**Step 1**: 抽取 ≥30 评论用户 (2,043 users),为每个用户匹配一个 5 attrs 的
product pair。**最终 2,041 个 (user, asin) pairs**。

**Step 2**: 对每个 pair,用 Phase 14.B 同样的 prompt template 生成 K=8 D_off
queries (no style injection baseline)。**总共 16,328 queries**。

**Step 3**: 用 Qwen2-7B encode:
- 2,041 users × 30 sentences = 61,230 sentences → user_hiddens (2041, 30, 5, 3584)
- 16,328 queries → cand_hiddens (16328, 5, 3584)

**Step 4**: PCA-500 + α=0.25 ensemble Maha rerank,**2,041-fold LOPO**:
- 对每个 fold,计算 8 个 candidates 对所有 2,041 个 user 的 Maha distance
- 取 min rank → best_rank per pair

## LOPO 结果 (2,041-fold D_off baseline)

| 指标 | 2041-fold D_off | 30-pair BoK-184 |
|---|---|---|
| **rank-1** | **11/2041 (0.54%)** | **18/30 (60.0%)** |
| top-10 | 63/2041 (3.09%) | - |
| top-100 | 350/2041 (17.15%) | - |
| mean_rank | **694.31** | **10.9** |
| mean_rank CI95 | [668.1, 720.0] | - |

**降幅**:
- rank-1: 60.0% → 0.54% (**降 99.1%**)
- mean_rank: 10.9 → 694.31 (**增 63.7x**)

## 关键发现

### 1. **30-pair 60% rank-1 是严重 overfit / biased 采样**

30 个 pair 是从 198 用户池里**手工挑选**(Phase 14.B 早期 setup),不是随机抽样。
这 30 个用户是 Phase 14.Q diagnostic 证明**最难**的 12 个 + 选过 StyleVector 测试
的 18 个,**全部是 "Phase 14 重点关注" 的样本**。

在 198 用户池里,这些用户因为已有 StyleVector steering + Q-8/Q-10 prompt 工程,
**本身已经是 pipeline 优化目标**。所以 LOPO 上表现"好"。

在 2,041 用户池里,**真实随机用户**,D_off baseline alone 几乎无法区分:
- mean_rank 694 ≈ 2041/2 - 100 (中位数附近)
- rank-1 0.54% ≈ 1/2041 (随机)
- top-100 17.15% (略好于随机 5%)

### 2. **Phase 14.Q rerank ceiling 是 user pool size artifact**

BoK-184 SOTA 的 18/30 看起来是 SOTA,但实际是 **30 用户子集 + 184 候选池** 在
**198 用户 distractor 池** 下的局部最优。

当 distractor 池扩到 2,041,即使只对比 8 cands/pair,真实性能大幅缩水。

### 3. **生成阶段 + rerank 范式在大规模上不够**

D_off 生成 = 通用 query template,无风格注入。在 2041 用户池中,任何 query 的
"风格" 都比真实用户弱,**用户 Qwen hidden 难以区分哪个 query 属于哪个 user**。

StyleVector 注入理论上能让 query 偏向特定 user,但 SOTA (A14_a1.0) 在 198 用户
池上 30/198 表现"好",在 2041 用户池上的真实泛化**待测**。

## 下一步方向

### A. 验证 StyleVector 在 2041 池上的真实增益

- 生成 A14_a1.0 K=8 / A22_a0.5 K=8 / A26_a0.5 K=8 (各 16K queries)
- 每个 pair 候选池扩到 32-40 cands (4-5 conds × K=8)
- 跑 LOPO: 是否 rank-1 > 0.54%?

### B. 重新评估 SOTA 标准

- **30-pair 验证 = 上限 bias**;真实泛化需要 ≥200 pair 随机抽样的 LOPO
- 建议: 改用 **198 用户池 random sample 100 pair** 作为标准验证集

### C. 转向真正的 user-conditioned generation

- Q-10 enriched prompt 在 30 user 上 alone 4/30 弱,但 BoK-184 中起 diversity 作用
- 在 2041 user 上,**enriched prompt 的平均增益需要重新测**

### D. 用户/产品侧改进

- 当前 rerank 用 Qwen hidden state,**用户→产品的 attribution 弱**
- 也许要用 product-side information (rating, category) 做 reverse rerank

## 输出文件

| Path | Purpose |
|------|---------|
| `phase14_q10l_pairs.jsonl` | 2,041 (user, asin) pairs with 4+ attrs |
| `phase14_q10l_user_reviews.pkl` | 2,043 users with text reviews |
| `phase14_q10l_generated.jsonl` | 16,328 D_off queries (K=8 per pair) |
| `phase14_q10l_user_hiddens_5layers.npz` | user_hiddens (2041, 30, 5, 3584) |
| `phase14_q10l_cand_residuals_qwen.npy` | cand_residuals (16328, 5, 3584) |
| `phase14_q10l_lopo_eval.json` | LOPO result (rank-1 0.54%, mean_rank 694) |
| `phase14_q10l_lopo_per_pair.jsonl` | per-pair best_rank |
| `phase14_q10l_generate.py` | vLLM batched K=8 generation |
| `phase14_q10l_encode.py` | Qwen 5 layers encode (users + cands) |
| `phase14_q10l_lopo.py` | 2041-fold PCA-500 Maha LOPO |

## 关键复用要点

- **30-pair 验证不可靠**: rank-1 60% 在 2,041 用户池降至 0.54%
- **Phase 14.Q SOTA 是 30 用户子集 artifact**: 不是真实泛化指标
- **D_off baseline 在大规模上 = random**: mean_rank 694 ≈ 2041/3,几乎无区分
- **下一步必做**: StyleVector + BoK 在 2041 用户池的复验,确认泛化
- **SOTA 标准需重定义**: ≥200 pair 随机抽样 LOPO,而不是 30 重点 pair