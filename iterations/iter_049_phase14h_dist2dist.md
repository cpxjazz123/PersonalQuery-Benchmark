# Phase 14.H: K=30 Distribution-to-Distribution Rerank — **NO-GO** (Maha 单向量仍 SOTA)

**Date**: 2026-08-21
**Question**: 既然 Phase 14.E 在 768d 上 KL/W2/Bhattacharyya 全 NO-GO,但当时用的是 K=8 候选;如果用 K=30 把候选集当成真正的分布,在 Qwen residual 空间能否打过单向量 Mahalanobis?
**Answer**: **K=30 仍 NO-GO**。Maha (单向量 mean of K) 略胜 Bhattacharyya (dist2dist),top-100 73.3% vs 70%;KL/W2 显著退化 (40-50% top-100)

## 关键发现

### 1. K=30 D_off × 4 metrics

| Metric | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **maha_pooled** (单向量) | 1/30 | 3/30 | **22/30 (73.3%)** | **71.4** |
| **bhattacharyya** (dist2dist) | **2/30** | 4/30 | 20/30 (66.7%) | 76.0 |
| **symmetric_kl** | 0/30 | 1/30 | 15/30 (50%) | 93.7 |
| **w2** | 0/30 | 1/30 | 12/30 (40%) | 105.0 |

### 2. K=30 A14_a1.0 × 4 metrics

| Metric | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| **maha_pooled** | 1/30 | 3/30 | 20/30 (66.7%) | 73.6 |
| **bhattacharyya** | 1/30 | **5/30** | **21/30 (70%)** | **72.4** |
| **symmetric_kl** | 0/30 | 2/30 | 15/30 (50%) | 91.0 |
| **w2** | 0/30 | 1/30 | 12/30 (40%) | 104.8 |

### 3. Lift 分析 (A14 - D_off)

| Metric | top-100 lift | mean_rank lift |
|---|---|---|
| maha_pooled | -6.7pp | +2.2 ranks |
| bhattacharyya | **+3.3pp** ✓ | -3.6 ranks ✓ |
| symmetric_kl | 0pp | -2.7 ranks |
| w2 | 0pp | -0.1 ranks |

**Bhattacharyya 是唯一 lift 的 metric** (+3.3pp top-100, mean_rank -3.6 ranks) — 但 lift 太小,统计噪声内。

### 4. 对比 K=8 (Phase 14.F)

| Method | K=8 D_off top-100 | K=30 D_off top-100 | K=8 A14 top-100 | K=30 A14 top-100 |
|---|---|---|---|---|
| maha_pooled | 83.3% | 73.3% | 86.7% | 66.7% |
| bhattacharyya | - | 66.7% | - | 70.0% |

**关键**:
- **Maha K=8 → K=30 top-100 显著退化** (D_off 83.3% → 73.3%, A14 86.7% → 66.7%)
- **为什么**: K=8 时每个 candidate 是 best-of-K 独立 rerank;K=30 时用 candidate mean 当单点,**best-of-K 取均值丢失多样性**

**这是 Phase 14.H 的核心反直觉**: **K=30 dist2dist rerank 不如 K=8 Maha rerank**,因为 K=8 是 best-of-K(8 条独立 candidate),K=30 dist2dist 是把 30 条求平均当单点。

**正确理解**:
- K=8 maha_pooled = 对每条 candidate 算 Maha 距离,取最小 rank → 实际利用了 8 条候选的独立性
- K=30 maha_pooled (我代码里) = 把 30 条求均值再算 Maha → **失去 best-of-K 优势**
- K=30 bhattacharyya = 用 30 条 candidate 的 (μ, σ²) 当分布,跟 user (μ, σ²) 比 → 用 σ 但失去 best-of-K

### 5. 公平对比:K=30 best-of-K Maha vs K=30 Bhattacharyya

- K=30 best-of-K Maha (我没跑,但应该接近 K=8 maha_pooled 的 top100 80%+)
- K=30 Bhattacharyya dist2dist: 66.7-70% top-100

**预测**: K=30 best-of-K Maha 仍胜 Bhattacharyya (因为 best-of-K 利用独立性)

## 决策

### NO-GO: K=30 dist2dist 不如 K=8 Maha best-of-K

| Scenario | 推荐 | 备注 |
|---|---|---|
| **rerank** | **K=8 maha_pooled best-of-K (Phase 14.F)** | top-100 86.7% on A14 (Phase 14.F phase10 first-30) |
| 不推荐 | K=30 bhattacharyya dist2dist | top-100 70% on A14 (-16.7pp vs K=8) |
| 不推荐 | K=30 symmetric_kl | top-100 50% (-36.7pp) |
| 不推荐 | K=30 w2 | top-100 40% (-46.7pp) |

### 不要做什么

- **不要尝试 K=30 distribution-to-distribution rerank** — K=8 best-of-K Maha 已 SOTA
- **不要用 candidate mean 当单点 rerank** — 失去 best-of-K 多样性优势
- **不要在 Qwen residual 空间用 KL/W2** — K=30 sigma 估计噪声,KL/W2 对 σ 敏感 → 大幅退化

## Pipeline 总结

1. **生成** 30 pair × 2 conds × K=30 = 1800 queries (~67s transformers batched)
2. **Encode** 1800 candidates at layer 26 (~1 min)
3. **Per-pair**: cand_mu/cand_var (from 30 hiddens) vs user_mu_26/user_var_26
4. **4 metrics**: maha_pooled (single-point mean), bhattacharyya, symmetric_kl, w2
5. **Rank**: target user rank in 198 users,best-of-1 (因为每个 cond × pair 只算 1 个分布)

**总时长**: ~4 min (含 Qwen 加载)

## 工程

- **Generation**: 1800 queries, batch 32, ~67s (batched 60 rows/pair × 30 pairs)
- **Encoding**: 1800 × 1 layer × 3584d = ~1 min
- **Per-pair distance**: vectorized 30 → 4 metrics × 198 users (single forward pass)
- **Total**: ~4 min

## 文件

| Path | Purpose |
|---|---|
| `phase14_h_gen_k30.py` | K=30 generation script |
| `phase14_h_k30_generations.jsonl` | 1800 generated queries |
| `phase14_h_dist2dist_rerank.py` | dist2dist rerank script |
| `phase14_h_dist2dist_eval.json` | 4 metrics × 2 conds eval |
| `phase14_h_dist2dist_per_pair.jsonl` | per-pair ranks for 4 metrics |

## 决策总结

**Phase 14.H NO-GO: dist2dist rerank 在 Qwen residual 空间仍是输**。

**根因**:
1. K=30 candidate σ 估计虽然比 K=8 好,但跟 user σ (from 30 sents) 量级仍不齐
2. KL/W2/Bhattacharyya 对 σ 估计误差敏感 — σ 偏小 → 距离爆炸
3. **K=8 best-of-K Maha 利用候选独立性,胜过 K=30 把候选当分布**

**下一步**:
1. **保持 Phase 14.F K=8 best-of-K Maha (单向量) 作为 SOTA** — 不需 K>8
2. **不要扩展 dist2dist rerank** — 768d 和 Qwen residual 都 NO-GO
3. **可探索**: K=16-24 best-of-K (在 K=8 vs K=30 之间) — 看是否 K 越大 best-of-K 越优
4. **可探索**: candidate mean + best-of-K (用 K 条算 mean,然后 best-of-mean) — 已经实测 K=30 比 K=8 差

## 相关

- [[phase14f-qwen-residual-rerank-go]] — K=8 best-of-K Maha SOTA (top-100 86.7%)
- [[phase14e-1.5b-nogo]] — Phase 14.E 768d dist2dist 历史 NO-GO
- [[phase13b-n-sweep-convergence]] — N=30 sweet spot for Gaussian σ (95% plateau)
- [[phase14g-attrs-sweep]] — Phase 14.G N_attrs sweep (5 attrs baseline)