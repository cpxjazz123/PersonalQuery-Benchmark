# Phase 11.A: Per-user Gaussian N(μ_u, Σ_u) at 768d — 总结

**Date**: 2026-08-20
**Status**: GO (核心信号极强)

## 实验动机

Phase 10.9–10.19 LLM 自由生成范式 5 次验证都不能突破 1.26% rank-1 ceiling。**用户洞察**: 用户风格本质是分布 N(μ_u, Σ_u) 而非点向量。新方向第一步: **从点向量升级到 Gaussian 分布**,验证 AnnaWegmann 768d 上 Σ 是否携带 user-discriminative 信号。

## 算法

1. **Per-sentence 768d**: AnnaWegmann encoder, 8760 句 (876 用户 × 10 train sentences),batch 128,~5s GPU
2. **Per-user mean μ_u**: 标准 mean-pool,split-half cos 0.717 (与 Phase 10.17 一致)
3. **Per-user covariance Σ_u with LW shrinkage**: 每个用户 n=10 sentences 在 768d → 样本 cov 严重 rank-deficient。LedoitWolf 收缩 → 平均 α=0.378 (sample cov 还有 62% signal)
4. **Cholesky 存为 sigma_lowrank[876, 768, 768]**: 2GB for high-fidelity sampling
5. **PCA 768→128 on user_mu matrix**: top 128 PC 解释 99.58% 方差(用户 mean 向量本质低秩)
6. **Validation**: hold-out 1 sentence/user,在 768d AnnaWegmann 空间里用 pooled LW-shrunk cov 算 Mahalanobis,看是否排在 target user 前面

## 关键结果

```
LW shrinkage α (mean/median/max): 0.3776 / 0.3745 / 0.6722  ← sample cov 有 signal (< 0.5)
Eigenvalue min (median/min):      5.29e-02 / 8.50e-03      ← PD (≥ 1e-6)
Split-half cos:                   0.717 ± 0.184            ← μ_u 稳定

Validation (held-out sentence identification, n=200):
  rank-1 coverage:                73.50%                   ← 远超 Phase 10.15 1.26%
  top-10 coverage:                96.00%
  top-100 coverage:               99.50%
  mean rank:                      3.41
  median rank:                    1.0
  random baseline mean rank:      438.0
```

**核心结论**: **AnnaWegmann 768d + LW-shrunk pooled cov 极强区分用户**。Held-out 一句属于 target user 的 rank-1 命中率 73.5%,mean rank 3.4(随机 baseline ~438)。

## 与 Phase 10.15 baseline 对比

| Metric | Phase 10.15 (318d, user-user) | Phase 11.A (768d AnnaWegmann, held-out sentence) |
|--------|-------------------------------|-------------------------------------------------|
| rank-1 coverage | 1.26% (876 pairs) | **73.50%** (200 pairs) |
| mean rank | 307.76 | **3.41** |
| 测试场景 | generated query vs all users | held-out sentence vs all users |

注意: 测试场景不同。Phase 10.15 是 **generated query** vs 用户 mean vectors (gap 大);Phase 11.A 是 **held-out 真实 sentence** vs 用户 mean vectors (gap 小)。后者是上限测试,验证 **AnnaWegmann 用户空间本身可分**。

## Phase 11.A 给下游 Phase 的输入

1. **Phase 11.B (sampling diversity)**:
   - 用 `mu_768[user]` + `sigma_lowrank[user] @ eps` (Cholesky) 采样 s_u ∈ R^768
   - 然后缩放到 TinyStyler 768d prefix (K=8, α=2.0)
2. **Phase 11.C (diffusion)**:
   - 用 `mu_128[user]` + PCA-projected Σ_128 (重建) 作为 s_u 的 128d 表示
   - Content embedding: 用 Qwen layer 14 hidden state 3584d → PCA 128d
3. **Phase 11.D (e2e)**:
   - 同 11.C,加 z → prefix head 训练

## 工程实现经验

1. **Per-user sentence groups**: uid_idx → [sentence indices],O(n) 构建
2. **LW shrinkage per user**: n=10 d=768 → α ~0.38 (sample cov 噪声大但还有信息)
3. **Cholesky for sampling**: `sigma_lowrank[user] @ eps` (eps ~ N(0, I_768))
4. **Pooled cov for identification**: 把所有 user_mu (876, 768) 一起 LW,作 dmat — 比 per-user cov 更稳定
5. **PCA on user_mu**: 128 PC 解释 99.58% 方差(用户风格本质低维)
6. **Cache 策略**: per-sentence 768d 单独 cache (~27MB),Gaussians 2GB (含 Cholesky),都在 hj82_scratch2

## 文件位置

- 引擎: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase11_a_per_user_gaussian.py` (~340 lines)
  - `phase11_a_per_sentence_embs_768d.npz` (~27MB,8760×768d)
  - `phase11_a_user_gaussians_768d.npz` (~2GB,含 Cholesky)
  - `phase11_a_user_gaussians_meta.json`
  - `phase11_a_validation_maha.json`
  - `phase11_a.log`
- 复用: phase10_user_embs_768d.npz (Phase 10.17.A),AnnaWegmann cache,vades_prototype_3000u_v6_raw_sentences.jsonl

## 下一步

**Phase 11.B (立即开始)**:
- 4-control A/B/C/D 用 TinyStyler
- A: sample s_u ~ N(μ_u, Σ_u) K=10,取 mean 作 prefix
- B: 用 μ_u (single mean) 作 prefix (baseline)
- C: shuffled-z (用其他 user 的 μ_u) sanity
- D: injection-off
- 验证: std(sampled) > 1.5× std(mean) AND mean rank NOT regress

## 决策: GO (Phase 11.B 开始)

- ✓ LW shrinkage α < 0.5: Σ 有 signal
- ✓ Eigenvalue min > 1e-6: PD
- ✓ Held-out rank-1 = 73.5%: AnnaWegmann 用户空间强可分
- ✓ μ_u 稳定 (split-half 0.717)
- ✓ Σ_u PD (8.5e-3 min eigenvalue)
- **下一步**: 11.B 验证 sampling diversity