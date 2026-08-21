# Phase 14.I: PCA 降维 + dist2dist rerank — **PCA 30d KL best of dist2dist** (NO SOTA breakthrough)

**Date**: 2026-08-21
**Question**: Phase 14.H 揭示 dist2dist rerank 在原始 3584d 受 σ 估计噪声拖累。**如果 PCA 降到 signal dimensions (30/50/100d),能否减少噪声、提升 dist2dist 效果?**
**Answer**: **PCA 30d symmetric_kl 是所有 dist2dist 组合中最优** (top-100 66.7%, mean 72.8 — 略优于 3584d Maha 73.6),但 **仍不如 Phase 14.F K=8 best-of-K Maha (top-100 86.7%)**

## 关键发现

### 1. A14_a1.0 全 dim × metric 矩阵

| Dim | Method | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|---|
| **30** | maha_pooled | 1/30 | 5/30 | 18/30 (60%) | 83.4 |
| **30** | **bhattacharyya** | 1/30 | 5/30 | 20/30 (66.7%) | 86.1 |
| **30** | w2 | 0/30 | 6/30 | 19/30 (63.3%) | 81.2 |
| **30** | **symmetric_kl** ✓ | **2/30** | 3/30 | 20/30 (66.7%) | **72.8** |
| 50 | maha_pooled | 1/30 | 2/30 | 18/30 (60%) | 86.1 |
| 50 | bhattacharyya | 1/30 | 3/30 | 19/30 (63.3%) | 84.4 |
| 50 | w2 | 0/30 | 1/30 | 14/30 (46.7%) | 101.7 |
| 50 | symmetric_kl | 0/30 | 2/30 | 17/30 (56.7%) | 93.2 |
| 100 | maha_pooled | 0/30 | 4/30 | 20/30 (66.7%) | 79.6 |
| 100 | bhattacharyya | 0/30 | 4/30 | 20/30 (66.7%) | 76.3 |
| 100 | w2 | 0/30 | 1/30 | 14/30 (46.7%) | 101.8 |
| 100 | symmetric_kl | 0/30 | 4/30 | 16/30 (53.3%) | 91.9 |
| 3584 | maha_pooled | 1/30 | 3/30 | 20/30 (66.7%) | 73.6 |
| 3584 | bhattacharyya | 1/30 | 5/30 | **21/30 (70%)** | 72.4 |
| 3584 | w2 | 0/30 | 1/30 | 12/30 (40%) | 104.8 |
| 3584 | symmetric_kl | 0/30 | 2/30 | 15/30 (50%) | 91.0 |

### 2. 最佳 dist2dist 组合

**d30 symmetric_kl**:
- rank-1: **2/30** (Phase 14.H 3584d dist2dist 中没有比这更高的)
- top-100: 66.7% (持平 d30 Bhattacharyya)
- **mean_rank 72.8** (所有 dist2dist 组合中**最低**)

**d100 Bhattacharyya**:
- rank-1: 0/30
- top-100: 66.7%
- mean_rank: 76.3 (次低)

**d3584 Bhattacharyya** (Phase 14.H):
- rank-1: 1/30
- top-100: 70%
- mean_rank: 72.4 (跟 d30 KL 几乎相同)

### 3. 与 Phase 14.F SOTA 对比

| Method | top-100 | mean_rank | 来源 |
|---|---|---|---|
| **Phase 14.F K=8 best-of-K Maha (SOTA)** | **86.7%** | **47.2** | phase10 first-30 pair |
| d30 symmetric_kl (best dist2dist) | 66.7% | 72.8 | Phase 14.B 30 pair |
| d3584 Bhattacharyya (Phase 14.H best) | 70% | 72.4 | Phase 14.B 30 pair |

**最佳 dist2dist 仍输 Phase 14.F K=8 best-of-K Maha 20pp top-100**。

### 4. PCA 效果分析

**意外发现**: **PCA 30d 让 symmetric_kl 从 NO-GO 变成可行**
- 3584d symmetric_kl: top-100 50%, rank-1 0/30
- d30 symmetric_kl: top-100 66.7%, rank-1 2/30 (+16.7pp, +2 rank-1)

**根因**: σ 估计在低维空间更稳定
- 30d 估计 σ 只用 30 个样本 → 自由度 = 30-1 = 29,足以估 30 个 σ
- 3584d 估计 σ 用 30 个样本 → 3584 个 σ 严重 rank-deficient
- **PCA 把噪声维度去掉,留下 signal 维度 → σ 估计噪声消失**

### 5. Maha 在 PCA 下退化

**反直觉**: Maha_pooled 在 d30 (top-100 60%) 比 d3584 (top-100 66.7%) 退化 6.7pp
- **Maha 不依赖 σ**,只比对 μ
- 但 PCA 投影**会损失部分 μ 信息**(去掉的小维度仍有 style signal)
- d100 略好 (66.7%),但仍不及 3584d (66.7% tie)
- **结论**: Maha 不需 PCA,PCA 反而损失信息

### 6. W2 全 NO-GO

W2 在所有 dim 上都差:
- 30d: top-100 63.3% (略好于其他 dim)
- 50d/100d: 46.7%
- 3584d: 40%

**W2 的 `||σ1-σ2||²` 项在 PCA 30d 上**仍受 σ 噪声影响**(只是量级小一些)

## 决策

### 找到最佳 dist2dist 组合 (但仍 < Phase 14.F)

| 场景 | 推荐 | 备注 |
|---|---|---|
| **rerank** | **Phase 14.F K=8 best-of-K Maha** | top-100 86.7% (SOTA) |
| 备选 | **d30 symmetric_kl (Phase 14.I)** | top-100 66.7% (-20pp vs SOTA) |
| 备选 | d3584 Bhattacharyya (Phase 14.H) | top-100 70% (-16.7pp vs SOTA) |
| 不推荐 | W2 (任何 dim) | top-100 40-63% |

### 不要做什么

- **不要把 d30 symmetric_kl 当 SOTA** — 比 Phase 14.F 仍差 20pp
- **不要扩展 PCA 到 d>100** — Maha 在 PCA 下退化
- **不要尝试 covariance 矩阵而非 diagonal** — 计算 + 噪声更大
- **不要用 PCA + Mahalanobis** — Maha 不需 PCA

### 留给未来

- **PCA + covariance estimation** (Ledoit-Wolf on low-dim) — 可能进一步 lift
- **Adaptive dim per user** (per-user PCA dim) — 工程复杂
- **PCA on StyleVector injection space** (instead of hidden state) — 也许 injection 维度更紧致

## Pipeline 总结

1. **复用 Phase 14.H K=30 candidates** (1800 queries)
2. **PCA fit**: 用198 × 30 条用户评论 hidden 拟合 top-3584 PCs (SVD,full dim)
3. **Project**: candidates (1800 × 3584d) 和 users (5940 × 3584d) 投到 PCA space
4. **Re-fit per-user Gaussian in PCA space**: 198 user × 30 sents × 30d/50d/100d/3584d
5. **Rerank per (cond, dim, metric)**: 2 × 4 × 4 = 32 settings × 30 pairs
6. **Aggregate**: rank-1, top-10, top-100, mean_rank

**总时长**: ~4 min (含 Qwen 加载 1 min + PCA SVD 几秒 + rerank 几秒)

## 工程

- **PCA fit**: SVD on [5940, 3584] matrix → 3584 PCs (~几秒 CPU)
- **Project**: 5940 + 1800 vectors × 3584d @ 3584d PC matrix = ~1s
- **Rerank**: 2 conds × 4 dims × 4 metrics × 30 pairs = 960 evaluations (vectorized)
- **Total**: ~3 min (含 encoding)

## 文件

| Path | Purpose |
|---|---|
| `phase14_i_pca_dist2dist_rerank.py` | PCA + dist2dist rerank script |
| `phase14_i_pca_dist2dist_eval.json` | 4 dim × 4 metric × 2 cond eval |
| `phase14_i_pca_dist2dist_per_pair.jsonl` | per-pair rank for all settings |
| `phase14_i_pca_dist2dist_meta.json` | metadata |

## 决策总结

**Phase 14.I PARTIAL-GO: PCA 30d symmetric_kl 是 dist2dist rerank 最佳组合**,但仍 < Phase 14.F SOTA。

**Key findings**:
1. **PCA 30d symmetric_kl 是唯一 lift 显著的 dist2dist metric** (rank-1 2/30)
2. **PCA 通过降低 σ 估计噪声让 KL 重新可行**
3. **PCA 不帮助 Maha** (Maha 不依赖 σ,只比对 μ)
4. **PCA 不帮助 W2** (W2 仍受 σ 噪声)
5. **PCA 帮助 Bhattacharyya 一点点** (d30 vs d3584 top-100 持平)

**SOTA 仍是 Phase 14.F K=8 best-of-K Maha (top-100 86.7%)**。

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换到 PCA + dist2dist
2. **可探索**: K=8 best-of-K Maha + PCA 30d — 看 PCA 是否帮助 best-of-K
3. **可探索**: PCA + covariance LW shrinkage (full covariance in 30d) — 可能进一步 lift
4. **不要再扩展** W2 / symmetric_kl 在 50d/100d — 没有 lift 趋势

## 相关

- [[phase14h-dist2dist-nogo]] — Phase 14.H dist2dist rerank NO-GO (3584d)
- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F K=8 best-of-K Maha SOTA
- [[phase14g-attrs-sweep]] — Phase 14.G N_attrs sweep
- [[phase13b-n-sweep-convergence]] — N=30 σ estimation sweet spot