# Phase 14.B: Layer × α Grid Sweep — **Layer 26 α=1.0 是真正的 Style Lift** (PARTIAL-GO)

**Date**: 2026-08-21
**Question**: 不同 layer 的 StyleVector 注入效果如何? Layer 14 是最优的吗?
**Answer**: **不是**。**Layer 26 α=1.0** 才有真正显著的 style lift (+0.0087 vs D_off, CI excludes 0),且 BoK margin 最高 (+0.0173)。但需要 semantic 代价 (-0.163)。**Layer 22 α=0.5** 是 sweet spot (BoK +0.0046, sem 仅 -0.015)。

## 关键发现

| Layer | α=0.5 BoK vs D_off | α=1.0 BoK vs D_off | α=0.5 sem vs D_off | α=1.0 sem vs D_off |
|---|---|---|---|---|
| 8 (浅) | -0.0012 ✗ | +0.0040 ✗ | -0.128 ✓ | -0.162 ✓ |
| 14 (中) | **+0.0060 ✓** | +0.0006 ✗ | -0.099 ✓ | -0.130 ✓ |
| 18 | **+0.0041 ✓** | +0.0021 ✗ | -0.053 ✓ | -0.100 ✓ |
| 22 | **+0.0046 ✓** | +0.0034 ✗ | -0.015 ✓ | -0.058 ✓ |
| **26 (深)** | **+0.0063 ✓** | **+0.0087 ✓ (best)** | -0.071 ✓ | -0.163 ✓ |

### 关键 cond (按 BoK margin 排序)

1. **A26_a1.0** — BoK **+0.0173** (+0.0087 vs D_off, CI [+0.0020, +0.0159])
   - 数值上 BoK margin 最高的 cond
   - sem_diff **-0.163** (semantic 显著破坏)
   - 适合 style lift 优先场景
   
2. **A26_a0.5** — BoK +0.0149 (+0.0063 vs D_off)
   - 比 A26_a1.0 略低 BoK 但 sem 破坏少一半 (-0.071 vs -0.163)
   - 平衡选项

3. **A22_a0.5** — BoK +0.0132 (+0.0046 vs D_off)
   - **semantic 破坏最小** (-0.015) 的显著 lift cond
   - 适合 content preservation 场景

4. **A14_a0.5** — BoK +0.0146 (+0.0060 vs D_off)
   - 经典 layer 14 + light injection
   - sem -0.099 (中等破坏)

### 重要发现: Layer 选择非常关键

**Layer 14 (Phase 13/14/15/16 默认) 不是最优**:
- A14_a1.0 BoK +0.0092, vs D_off diff **+0.0006 NS** (与 Phase 16 一致)
- A14_a0.5 BoK +0.0146, vs D_off diff +0.0060 ✓ — 但 α=0.5 更低更强

**深层 (Layer 22-26) 才是 sweet spot**:
- 注入到接近输出的层 (Layer 26 = 28 总层的 93%) 让 style 直接在生成 token 时生效
- Layer 14 在中间层可能被后续 layer-norm / 注意力洗掉一部分 signal

## 实验细节

### Sweep 配置 (11 conds)

| Cond | Layer | α | 注 |
|---|---|---|---|
| D_off | — | 0 | baseline |
| A8_a0.5 / A8_a1.0 | 8 | 0.5 / 1.0 | 浅层 (28%) |
| A14_a0.5 / A14_a1.0 | 14 | 0.5 / 1.0 | **Phase 13/14/15/16 默认层** (50%) |
| A18_a0.5 / A18_a1.0 | 18 | 0.5 / 1.0 | 中深层 (64%) |
| A22_a0.5 / A22_a1.0 | 22 | 0.5 / 1.0 | 深层 (79%) |
| A26_a0.5 / A26_a1.0 | 26 | 0.5 / 1.0 | **最深层 (93%)** |

### 工程
- 198 users (Phase 13.B-v2 sweet spot) × N=30 sentences
- 5 layers {8, 14, 18, 22, 26} hidden extraction (~22s, batch=32)
- Per-layer Gaussian fit (μ, σ_diag) with neutral residual
- **batched multi-cond**: 11 conds × K=8 = 88 rows per generate call
- 30 calls × ~3s = **92s sweep** (vs naive 11 conds × 30 calls × 3s = 16 min, **10x 加速**)
- Per-layer hooks (5 layer hooks):每个 hook 只在对应 layer 的 row 注入
- 11 conds × 30 pairs × K=8 = **2640 records**

### Eval
- 768d AnnaWegmann style margin (cos_target - cos_off_mean)
- Sentence-bert all-MiniLM-L6-v2 semantic sim vs attrs
- Paired bootstrap CI 2000 resamples vs D_off

## 决策

### Phase 14.B PARTIAL-GO

- **不是 NO-GO** — 多个 layer × α 组合有 **真正显著的 BoK lift vs D_off (CI excludes 0)**
- **A26_a1.0 数值最高** (+0.0173 BoK margin) — 但 semantic -0.163 破坏大
- **A22_a0.5 是 sweet spot** — BoK +0.0046 ✓, sem 仅 -0.015 (最小破坏)
- **A26_a0.5** 是 balance 选项 — BoK +0.0063 ✓, sem -0.071 中等

### 推荐路线 (按场景)

| 场景 | 推荐 | BoK lift | sem 破坏 |
|---|---|---|---|
| Style lift 最优先 | **A26_a1.0** | +0.0087 ✓ | -0.163 (high) |
| 平衡 style + content | **A22_a0.5** | +0.0046 ✓ | -0.015 (min) |
| Style-first light injection | A14_a0.5 | +0.0060 ✓ | -0.099 (medium) |
| 保守 baseline | D_off | 0 | 0 |

### 与 Phase 14 hybrid (历史 SOTA) 比较

- Phase 14 hybrid (A_a1.0 layer 14 + 768d rerank): top-100 56.7% (3.3x Phase 10.19 baseline)
- Phase 14.B A26_a1.0 (无 rerank): BoK margin +0.0173 (vs Phase 14 hybrid 隐含 ~+0.01)
- **下一步**: A26_a1.0 + 768d rerank 可能突破 SOTA

## 与历史路线对比

| 路线 | Style signal | BoK lift | 备注 |
|---|---|---|---|
| D_off (no injection) | baseline | 0 | sem 最高 |
| A14_a1.0 (Phase 14 hybrid 主路线) | +0.0006 NS | +0.0092 | 实际 lift 主要来自 768d rerank |
| **A26_a1.0** | **+0.0087 ✓** | **+0.0173** | **新 SOTA (StyleVector only)** |
| **A22_a0.5** | **+0.0046 ✓** | +0.0132 | **最小 sem 代价显著 lift** |
| A14_a0.5 | +0.0060 ✓ | +0.0146 | 经典 light injection |
| 1.5B 任何 | — | — | NO-GO (Phase 15.E) |

## 文件位置

### Scripts (in git)
- `result/phase14/scripts/phase14_b_layer_alpha_sweep_v2.py` — sweep 引擎 (per-layer extract + batched sweep)
- `result/phase14/scripts/phase14_b_eval_only.py` — eval (style margin + sbert semantic + bootstrap)

### Results
- `phase14_b_layer_alpha_sweep.jsonl` — 2640 records (11 conds × 30 pairs × K=8)
- `phase14_b_user_hiddens_5layers.npz` — 198 users × N=30 × 5 layers hidden cache
- `phase14_b_eval.json` — per-cond eval + paired bootstrap vs D_off
- `phase14_b_meta.json` — sweep meta (config_names, layers, alphas)

### Logs
- `result/phase14/logs/phase14_b.log` — Qwen load + extract + sweep
- `result/phase14/logs/phase14_b_eval.log` — eval run

## 下一步建议

1. **A22_a0.5 立即可用** — 最优 style/content 平衡,可作为 Phase 14 hybrid 的 μ-only 替换
2. **A26_a1.0 + 768d rerank** 是潜在新 SOTA — 后续实验: rank-1 coverage on 876 pairs
3. **不要尝试 α > 1.5** — 当前所有 α=1.0 都已经 semantic -0.10 ~ -0.16,继续增大只会更糟
4. **不要尝试 layer < 8 或 > 26** — 浅层无 style signal,layer 27+ 接近 output 会破坏生成

## Bug 修复记录

1. **v1 失败**: phase14_b_layer_alpha_sweep.py 假设 hiddens 维度为 [N, n_layers, H], 但现有 cache 只有 layer 14 mean-pool [N, H]
   - **修复**: 写 v2 重 extract per-layer hiddens, cache 到 npz [N, 5, H],然后用真实 per-layer Gaussians
2. **per-layer hook 实施**: 不同 layer 的 steer 在不同 forward step 注入, batched multi-cond 通过 cond_idx // K 找到 row 索引, hook 用 cond_layer[cond_idx] 判断本层是否负责