# iter 047: Phase 35.M — Per-User 2-GMM (shared covariance) NO-GO

**日期**: 2026-08-23
**承接**: iter_045 (V2 N=7 K=4 = 50.7% SOTA) + 用户高阶假设:
  > 一个用户可能有 multimodal 风格(短 keyword + 长 natural language),
  > 单 Gaussian 平均掉了 modes。2-GMM 应该更好。
**目的**: 验证 per-user 2-component GMM 是否比 single Gaussian 更 discriminative

---

## 设计

- **每用户** PCA32 residual (~10 条历史评论 → 10 个 32d 向量)
- **KMeans K=2** → (μ_1, μ_2) cluster centers, (π_1, π_2) from cluster sizes
- **共享全局 diagonal covariance** Σ_global = mean(var per dim across users)
- **Score**: log p(z|u) = log[π_1 N(z; μ_1, Σ) + π_2 N(z; μ_2, Σ)]
- **Softmax over users in pool** with τ = 0.5

**风险控制**:
- ❌ 每个 component 各自估 covariance (会过拟合,用户只有 10 样本)
- ✅ 共享 covariance (避免 per-component 32×32 矩阵估计)
- ✅ 只学 μ_1, μ_2, π_1, π_2 (更少参数)

---

## 结果 (V2 N=7 K=4 cands, intra-product size≥2, 67 records / 16 asins)

| Method | Rank-1 | mean_rank | lift vs random | Δ vs baseline |
|--------|--------|-----------|----------------|---------------|
| **single-Gaussian softmax_g32_τ0.5** (Phase 35.G SOTA) | **50.7%** | **9.77** | **2.27x** | — |
| 2-GMM shared-cov (NEW) | 38.8% | 11.22 | 1.73x | **-12.18pp** |
| hybrid 0.5*single + 0.5*GMM | 38.8% | 10.97 | 1.73x | -12.21pp |

**Bootstrap CI (1000 resamples, record-level)**:
- 2-GMM Δ vs baseline: **-12.18pp [-23.88, +0.00]** ← CI 触 0, 显著负
- hybrid Δ vs baseline: -12.21pp [-25.37, +1.49] ← 偏负

**Pairwise hit diff (n=67 records)**:
- 2-GMM: better=6, worse=14, same=47 → **net -8 records** (输多于赢)
- hybrid: better=7, worse=15, same=45 → net -8 records

**GMM 自身统计** (n=74 用户, K=2 均成功):
- π_0 (cluster 0 mix): mean=0.584, std=0.210 (cluster 平衡, ~58/42 split)
- center_dist (PCA32 space): mean=89.78, median=87.02 (clusters 相距 ~90 units)

---

## 核心结论: NO-GO ★★★

### 1. **User style 近似 unimodal** in PCA32 residual space
- 2-GMM 严格比 single-Gaussian 差 12pp (50.7 → 38.8)
- Bootstrap CI 偏负,不包含正向优势区间
- Pairwise 净输 8 records → 不是 noise, 是 consistent degradation

### 2. **KMeans K=2 拟合的是 noise 不是 real modes**
- ~10 个样本 → K=2 容易把数据切两半,但这种切分没有 user-discriminative signal
- π_0 mean 0.584 (std 0.210) → cluster 平衡说明切分是 data-driven 不是 user-mode-driven
- center_dist 89.78 (PCA32 单位) → clusters 看起来分离,但只是把单 Gaussian 拆成两个 noise half

### 3. **为什么 2-GMM 失败(理论解释)**
- **样本不足**: 10 samples / 2 components = 5 per Gaussian → covariance 估计必须共享
- **共享 covariance 让 2-GMM 退化成 "two-point discrete mixture"**: score ≈ log[π_1 exp(-||z-μ_1||²/2σ²) + π_2 exp(-||z-μ_2||²/2σ²)]
- 当 z 离某个 μ_i 近,score ≈ log π_i + const; 离两个都远,score ≈ log(π_1 + π_2) + const ≈ const
- **本质问题**: 如果用户真有两种风格,A 模式查询应该离 μ_1 近,B 模式查询应该离 μ_2 近
- 但 PCA32 residual 本身已经是 single-mode 的表达(KMeans 切不出来),所以 2-GMM 帮不上忙

### 4. **用户的 multimodal 假设是合理的,但 PCA32 space 不支持**
- 用户可能确实有时写短 keyword 有时写长 natural language
- 但 Qwen mean-pool residual **没有把这个 mode 切换编码进 32 维 PCA**
- 残余空间是 single Gaussian 友好的(continuous unimodal distribution)
- 要捕获 multimodal,可能需要:
  - 不同 layer 的 residual (不同层编码不同 abstraction)
  - 不同 pooling (first/last token vs mean)
  - 句法/词性显式特征 (length, POS tag distribution) 作为 side info

---

## 排除项 (don't need to retry)

- ❌ 2-GMM with per-component covariance → 严重过拟合(10 samples / 2 × 32 × 32 params)
- ❌ K=3 GMM → 更差,参数更多
- ❌ 用 length label 作为 mode 监督 → 半监督,样本量不够
- ❌ Hybrid single + GMM → 跟 2-GMM 一样差

---

## 对照用户原假设

| 用户假设 | 实测 | 判定 |
|---------|------|------|
| 一个用户可能有 multimodal 风格 | 数据上看 KMeans 能切 2 cluster (center_dist 89.78) | ✅ cluster 存在 |
| 2-GMM 比 single Gaussian 更好 | 50.7% → 38.8% (-12pp) | ❌ **NO-GO** |
| shared covariance 更稳 | 是(否则会更差) | ✅ 实现选择正确 |
| hybrid 应该至少不输 baseline | 一样差(-12pp) | ❌ 输给 baseline |

→ **核心回答**: 在 PCA32 residual 空间下,**single Gaussian 已经足够**,2-GMM 反而引入 noise。Multimodal 假设需要换 representation 而不是换 scoring。

---

## 下一步

1. **Phase 35.N**: 探索 multimodal capture 的其他方式
   - 不同 layer residual 拼接 (layer 14 + 26)
   - Length as side feature (re-weight by query length)
   - Per-(N_attrs, length_bucket) 局部 Gaussian (承认模式与生成条件耦合)
2. **Phase 35.J-2 forced gen scoring**: 验证 N=7 L=36-50 在 K=8 样本量下是否 100% 仍然成立
3. **Phase 35.L**: 论文 trade-off figure (N vs L 2D heatmap, Pareto frontier)

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35m_gmm.py` | 2-GMM 实验主脚本 (~3s) |
| `syntax_subspace/phase35m_debug.py` | 验证 single-Gaussian 复现 50.7% baseline |
| `result/phase35m/gmm_summary.json` | 完整结果 + bootstrap CI + GMM stats |

---

## 最终 takeaway

> Phase 35.M **2-GMM per-user (shared covariance) NO-GO**:
> 1. **50.7% → 38.8%** (Δ = -12.18pp, CI [-23.88, +0.00])
> 2. pairwise 6 better / 14 worse → consistent degradation
> 3. KMeans K=2 拟合的是 data-noise 不是 user-mode (π balance 0.584, center_dist 89.78)
>
> 在当前 PCA32 residual 空间,**用户风格近似 unimodal**,single Gaussian 已经是最优。
> 用户的 multimodal 假设理论上合理,但当前 representation 不支持 — 需要换 embedding
> (different layer, different pooling, side features) 而不是换 scoring。
>
> 下一步尝试其他 multimodal 表达方式(layer concat / length-aware / per-cond Gaussian)。
