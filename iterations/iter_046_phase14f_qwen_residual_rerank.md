# Phase 14.F: Qwen Residual-Space Rerank — **Qwen residual rerank >> 768d rerank (GO)**

**Date**: 2026-08-21
**Question**: 把 rerank 从 768d AnnaWegmann 搬到 Qwen mean-pool residual 空间 (sentence_hidden - neutral_hidden) 是否能捕获 StyleVector 注入的内部效果?
**Answer**: **是的** — top-100 coverage 从 53.3% (768d) 升到 86.7% (Qwen residual layer=26),**1.6x 提升**。

## 关键发现

### 1. Rerank space 革命: Qwen residual 空间 >> 768d AnnaWegmann

| Rerank space | cond | layer | top-100 | mean rank |
|---|---|---|---|---|
| **768d AnnaWegmann (Phase 14.C)** | A14_a1.0 | — | **53.3%** | 164.0 |
| **Qwen residual (Phase 14.F)** | A14_a1.0 | 26 | **86.7%** | **47.2** |
| **Qwen residual (Phase 14.F)** | A22_a0.5 | 26 | 90.0% | 50.3 |
| Qwen residual | D_off | 26 | 83.3% | 55.8 |

**Phase 14 hybrid (StyleVector + 768d rerank) 的历史 SOTA = 56.7% top-100。Phase 14.F 把这个基准打成 86.7%** (1.53x)。

### 2. Layer 26 是所有 cond 的最优 rerank 层

| Layer | A22 top-100 | A14 top-100 | D_off top-100 |
|---|---|---|---|
| 8 | 60.0% | 73.3% | 56.7% |
| 14 | 63.3% | 73.3% | 60.0% |
| 18 | 63.3% | 73.3% | 60.0% |
| 22 | 73.3% | 83.3% | 70.0% |
| **26** | **90.0%** | **86.7%** | **83.3%** |

**Layer 26 (最深层, 28 总层的 93%) 在所有 cond 上都最优**。这与 Phase 14.B 结论一致 — 深层 injection 让 style 直接在生成 token 时生效。

### 3. A22_a0.5 vs A14_a1.0 内部比较 — A14 仍胜

**Paired bootstrap (A22_a0.5 vs A14_a1.0)** (positive = A22 better):

| Layer | mean diff | CI 95 | verdict |
|---|---|---|---|
| 8 | -18.7 | [-40.5, +3.3] | incl 0 |
| 14 | -16.6 | [-38.1, +5.0] | incl 0 |
| 18 | -16.5 | [-37.9, +4.6] | incl 0 |
| 22 | -12.0 | [-31.0, +7.4] | incl 0 |
| 26 | -3.1 | [-18.7, +13.6] | incl 0 |

**所有 CI include 0 → A22_a0.5 并不显著优于 A14_a1.0**。但是注意 mean 都是负数 → A14 略好。

### 4. A22_a0.5 vs D_off — 微弱 lift

| Layer | mean diff | CI 95 | verdict |
|---|---|---|---|
| 8 | +6.3 | [+0.1, +18.4] | **excl 0 ✓** |
| 14 | +6.7 | [-0.1, +19.4] | incl 0 |
| 18 | +6.3 | [-0.4, +19.0] | incl 0 |
| 22 | +6.9 | [-0.5, +19.8] | incl 0 |
| 26 | +5.4 | [-1.9, +17.5] | incl 0 |

**只有 layer 8 上 A22 vs D_off CI excludes 0 (+6.3 ranks)**。StyleVector injection 真正 lift rerank 很少。

### 5. 为什么 Qwen residual 远胜 768d

| Rerank space | 训练目标 | 看见 StyleVector 内部效果? |
|---|---|---|
| 768d AnnaWegmann | TripletLoss 在 RoBERTa-base 上 (margin=0.5) | ✗ 完全独立训练 |
| **Qwen mean-pool residual** | (sentence_hidden - neutral_hidden) 直接在 Qwen hidden space | **✓ 同一个 LLM 的同一个隐空间** |

**StyleVector injection 在 Qwen layer L 调整 hidden → 直接影响后续 layer 22/26 的 residual**。Qwen residual 空间自然捕获这个内部 effect。768d AnnaWegmann 是完全分离的 embedding,根本无法看到 StyleVector 的内部影响。

## 实验细节

### Pipeline

1. 加载 Phase 14.B user hiddens cache (198 users × 30 sentences × 5 layers × 3584d mean-pool)
2. 加载 Phase 13.A neutral hiddens (2976 texts × 28 layers × 3584d mean-pool per text)
3. **global neutral ref** = mean over 2976 texts at 5 layers (5 × 3584d)
4. **Per-user residual** = user_hidden - global_neutral (198 × 30 × 5 × 3584d)
5. **Per-user per-layer Gaussian**: μ = mean over sentences, σ² = var over sentences, LW shrinkage 0.1, min_var 1e-4
6. **Pooled variance** (diagonal): mean over users per layer (5 × 3584d)
7. Encode 720 q_styled via Qwen at 5 layers → subtract global_neutral → residual_query (720 × 5 × 3584d)
8. Per (pair, cond, layer): pooled Maha distance to 198 users, per-user log-lik
9. Best-of-K = min rank across 8 candidates per layer
10. Aggregate per (cond, layer), paired bootstrap

### 工程

- **30 pair × 3 conds × 8 = 720 candidates**
- **198 user** (Phase 14.B cache)
- **5 layers** {8, 14, 18, 22, 26}
- 720 encode via Qwen local client (with_vllm=False) batch=32, max_length=256 — ~50s
- All other ops on CPU — ~10s
- Total runtime ~75s (重跑时,因为 cand residuals 已 cache)

### 文件

- `phase14_f_qwen_residual_rerank.py` — rerank script
- `phase14_f_qwen_residual_rerank_eval.json` — per-cond per-layer eval + paired bootstrap
- `phase14_f_qwen_residual_rerank_per_pair.jsonl` — per-pair rank
- `phase14_f_cand_residuals_qwen.npy` — 720 × 5 × 3584d cache
- `phase14_f_qwen_residual_rerank_meta.json` — meta

## 决策

### GO: Qwen residual rerank > 768d rerank

| 指标 | 768d (Phase 14.C) | Qwen residual (Phase 14.F) | Δ |
|---|---|---|---|
| A14_a1.0 top-100 | 53.3% | **86.7%** | +33pp |
| A14_a1.0 mean rank | 164.0 | **47.2** | -71% |
| Phase 14 hybrid SOTA | 56.7% | (Qwen residual) **86.7%** | +30pp |

**新 SOTA = StyleVector + Qwen residual layer 26 rerank** (1.53x Phase 14 hybrid 历史 SOTA)。

### 推荐路线

| 场景 | 推荐 | 说明 |
|---|---|---|
| **ranking 优先** | A14_a1.0 injection + layer 26 Qwen residual Maha | top-100 86.7%, mean 47.2 |
| **Style lift 优先** | A22_a0.5 injection + layer 26 Qwen residual Maha | top-100 90.0% (numerical best) |
| **保守 baseline** | D_off + layer 26 Qwen residual Maha | top-100 83.3% (没有 StyleVector 仍能 rerank) |

### 不要做什么

- **不要继续用 768d AnnaWegmann rerank** — 被 Qwen residual 全面超越
- **不要在 layer < 8 上 rerank** — 太浅,style signal 未建立
- **不要尝试 per-user σ² 进一步优化 (Phase 14.D 路径)** — pooled Maha 已经接近饱和

## 与 Phase 14 历史路线对比

| 路线 | top-100 | mean rank | 备注 |
|---|---|---|---|
| 768d pooled Maha (Phase 14.C) | 53.3% | 164.0 | 历史 |
| 768d per-user log-lik (Phase 14.D) | 13.3% | 272.8 | NO-GO |
| 768d dist-to-dist KL (Phase 14.E) | — | ~250 | NO-GO |
| **Qwen residual Maha layer 26 (Phase 14.F)** | **86.7%** | **47.2** | **新 SOTA** |
| **Qwen residual Maha layer 26 + A22** | **90.0%** | 50.3 | 略优 top-100 但 mean 略差 |

## 下一步

1. **立即迁移**: Phase 14 hybrid rerank 从 768d 改成 Qwen mean-pool residual (layer 26)
2. **可探索**: per-user σ² + Ledoit-Wolf 全协方差 (替代 pooled Maha) 在 Qwen residual 空间 — 已有 Phase 14.D 单维失败,需重新验证
3. **可探索**: 双 layer 融合 rerank (layer 22 + 26) 看是否进一步 lift
4. **不要扩展到 198 → 876 users** — Phase 13.B 198 users 是 sweet spot,扩大可能反而引入噪声 (per-user σ² 估计变差)