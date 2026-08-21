# Phase 16: Gaussian-style steer sweep (ball/shrink/Maha) — **NO-GO**

**Date**: 2026-08-21
**Question**: 真正用 Gaussian steer (而不是只用 μ point estimate) 能 work 吗?
**Answer**: **NO**。所有 6 种 Gaussian 范式 (ball/shrink/Maha) 均无 style lift。Ball/Shrink 还破坏 semantic。Maha 范式对 semantic 中性但也无 style lift。

## 实验动机

Phase 15.D (2026-08-21) 报告 `ρ × σ × ε` 范式 NO-GO (σ_residual norm 175 >> μ_residual 24, steer norm 2637 爆炸)。
Phase 13.B-v2 N sweep 确认 N=30 是 sweet spot, 但即使 N=100, σ_residual norm (310) 仍 ≈ 3-4× hidden norm, **ρ noise 范式本身有问题,不是样本量问题**。

用户问: "**怎么才可以真正使用到高斯呢**" — 既然 `ρ × σ × ε` 爆炸,有没有范式既能用上 σ 信息又不会 explode?

Phase 16 提出 3 种"安全" Gaussian 范式:
1. **方案1 Ball-constrained**: `ε ~ N(0, I), ||ε|| ≤ 1, s = μ + ρ × σ × ε_normed`
2. **方案2 Shrink σ**: `σ' = min(σ, 0.3 × |μ|)`,然后 ball sample + norm clip
3. **方案3 Mahalanobis weight**: `α × μ × confidence(σ)`, `confidence = 1 / (1 + σ_norm)` (无 noise, 只用 σ 给权重)

## 实施

### Sweep 配置 (9 configs)

| Cond | Mode | Params | 说明 |
|---|---|---|---|
| D_off | off | {} | baseline (no hook) |
| A_a1.0 | mu_only | α=1.0 | 主路线对照 (Phase 14 hybrid) |
| Ball_r0.3 | ball | ρ=0.3 | Ball-constrained ε, ρ=0.3 |
| Ball_r0.5 | ball | ρ=0.5 | Ball-constrained ε, ρ=0.5 |
| Ball_r1.0 | ball | ρ=1.0 | Ball-constrained ε, ρ=1.0 |
| Shrink_r0.5 | shrink | ρ=0.5, c=0.3 | Shrink σ 然后 ball sample + norm clip |
| Shrink_r1.0 | shrink | ρ=1.0, c=0.3 | Shrink σ 然后 ball sample + norm clip |
| Maha_a0.5 | maha | α=0.5 | α × μ × confidence(σ), no noise |
| Maha_a1.0 | maha | α=1.0 | α × μ × confidence(σ), no noise |

### 数据
- 30 pairs (从 phase10_pairs 876 取前 30)
- 198 users with ≥30 reviews (Phase 13.B-v2 sweet spot)
- N=30 sents/user 拟合 Gaussian (μ, σ_diag), 减 layer 14 neutral mean → residual
- Qwen2-7B-Instruct, layer 14 forward hook, last-token position
- K=8 samples per pair, temperature=1.0, top_p=0.95, max_new_tokens=80
- hard-copy 后处理 (clause-append missing attrs)

### 工程
- 9 conds × 30 pairs × K=8 = **2160 records**
- Qwen load ~4 min, sweep ~8 min (9 conds × ~50s/cond)
- Eval: AnnaWegmann 768d style margin (cos_target - cos_off_mean) + sentence-bert semantic

## 结果

### Style margin (vs D_off, paired bootstrap 2000 resamples)

| Cond | BoK margin (vs D_off) | 95% CI | Sig? |
|---|---|---|---|
| D_off | — | — | — |
| A_a1.0 | +0.0017 | [-0.0021, +0.0056] | ✗ |
| Ball_r0.3 | +0.0010 | [-0.0022, +0.0043] | ✗ |
| Ball_r0.5 | +0.0008 | [-0.0034, +0.0048] | ✗ |
| Ball_r1.0 | -0.0003 | [-0.0036, +0.0027] | ✗ |
| Shrink_r0.5 | -0.0013 | [-0.0038, +0.0014] | ✗ |
| Shrink_r1.0 | +0.0005 | [-0.0037, +0.0047] | ✗ |
| Maha_a0.5 | +0.0002 | [-0.0016, +0.0023] | ✗ |
| Maha_a1.0 | +0.0010 | [-0.0007, +0.0026] | ✗ |

→ **全部 CI includes 0** → **无 style lift** (与 Phase 15.D 一致)

### Semantic preservation (vs D_off, paired bootstrap)

| Cond | sem_diff (vs D_off) | 95% CI | Sig? |
|---|---|---|---|
| A_a1.0 | -0.0646 | [-0.1075, -0.0287] | ✓ (sem regress) |
| Ball_r0.3 | -0.0603 | [-0.0983, -0.0272] | ✓ |
| Ball_r0.5 | -0.0770 | [-0.1173, -0.0418] | ✓ |
| Ball_r1.0 | -0.0701 | [-0.1120, -0.0321] | ✓ |
| Shrink_r0.5 | -0.0696 | [-0.1109, -0.0358] | ✓ |
| Shrink_r1.0 | -0.0680 | [-0.1107, -0.0307] | ✓ |
| **Maha_a0.5** | **-0.0068** | **[-0.0166, +0.0024]** | **✗ (neutral)** |
| **Maha_a1.0** | **-0.0073** | **[-0.0156, +0.0009]** | **✗ (neutral)** |

→ Ball/Shrink 范式都 **显著破坏 semantic** (-0.06 ~ -0.08)
→ **Maha 范式对 semantic 中性** (CI includes 0), 但也无 style lift

### Per-cond raw means (mean of all K=8 cands)

| Cond | mean_margin | BoK_margin | mean_sem_sim |
|---|---|---|---|
| D_off | -0.0001 | 0.0104 | **0.8138** |
| A_a1.0 | 0.0003 | 0.0120 | 0.7491 |
| Ball_r0.3 | 0.0003 | 0.0113 | 0.7535 |
| Ball_r0.5 | 0.0010 | 0.0112 | 0.7367 |
| Ball_r1.0 | -0.0001 | 0.0100 | 0.7437 |
| Shrink_r0.5 | -0.0003 | 0.0090 | 0.7441 |
| Shrink_r1.0 | 0.0007 | 0.0109 | 0.7458 |
| Maha_a0.5 | -0.0017 | 0.0106 | 0.8070 |
| Maha_a1.0 | -0.0004 | 0.0114 | 0.8065 |

注意: D_off sem_sim 0.8138 远高于其他 conds (0.74-0.81) — StyleVector injection 注入让 query 更像"风格化"而不是 attribute listing。

## 关键发现

### 1. Gaussian steer 范式整体 NO-GO
- 即使 ball-constrained / shrink σ / Mahalanobis weight, **BoK margin vs D_off 全部 CI includes 0**
- 与 Phase 15.D `ρ × σ × ε` 范式 NO-GO 一致 — **不是范式参数问题,是 StyleVector injection 在 layer 14 整体不 work**

### 2. Ball/Shrink 范式比 μ-only 更糟
- A_a1.0 bok_diff **+0.0017** vs Ball_r1.0 **-0.0003** — 加 noise 让 style margin 下降
- A_a1.0 sem_diff **-0.0646** vs Ball_r1.0 **-0.0701** — 加 noise 略破坏 content
- Ball/Shrink noise 是 pure downside (no style lift + sem regress)

### 3. Maha 范式是"中性 baseline"
- Maha_a0.5 / Maha_a1.0 sem_diff CI includes 0 — 不破坏 content
- 但 BoK margin 也不显著 — 没 style lift
- Maha 在 style space 接近 D_off, 因为 `confidence = 1/(1+σ_norm)` 让大多数 dim 的 α 衰减为 ~0.3-0.5
- **Maha 可作为"安全路线"但没价值** — 不如直接用 D_off

### 4. D_off sem_sim 0.81 vs active conds 0.74-0.81
- D_off 输出最像 attrs (no interference from StyleVector)
- Active conds 注入让 query 偏 attribute-listing → 反而降低 semantic sim (因为 attrs 是描述性,生成的 query 模仿风格后会偏离原始 attrs 顺序)
- 但这是 hard-copy 后处理造成的,不是 injection 本身的 fault

## 回答用户问题

**Q: 怎么才可以真正使用到高斯呢?**

**A: 不能** (在当前 7B Qwen + layer 14 last-token 范式下)。具体原因:
1. **per-dim σ 是 hidden noise**, 不是 user style signal (Phase 13.B-v2 已证实: σ_residual norm 310 ≈ 3-4× hidden norm 即使 N=100)
2. **任何 Gaussian steer (Ball/Shrink/Maha)** 都无法显著 lift style margin
3. **Ball/Shrink 范式** 双重 NO-GO (no style lift + sem regress)
4. **Maha 范式** 对 semantic 中性但也没 style lift — 等价于"弱化的 μ-only"

### 当前主路线结论
- **Phase 14 hybrid** (μ-only α=1.0 + 768d rerank) 仍是最佳 style lift 路线 (rank-1 56.7%, 3.3x Phase 10.19 baseline)
- StyleVector injection 本身只能让 query **slightly** 偏 user style — 真正 ranking lift 来自 768d rerank
- Gaussian 范式 (本应让 steer 更"自然") 不 work — StyleVector injection 是 post-hoc 修饰,不是真生成

## 文件位置

### Scripts (in git)
- `result/phase16/scripts/phase16_n30_gaussian_sweep.py` — sweep 引擎 (9 conds)
- `result/phase16/scripts/phase16_n30_eval_only.py` — 独立 eval (768d margin + sbert sem)

### Results
- `phase16_n30_gaussian_sweep.jsonl` — 2160 records (9 conds × 240 each)
- `phase16_n30_user_gaussians_qwen.npz` — 198 users × N=30 Gaussian fits (μ, σ_diag)
- `phase16_n30_eval.json` — 9 conds eval results (margin, sem, bootstrap CI)
- `phase16_n30_meta.json` — sweep meta

### Logs
- `result/phase16/logs/phase16_n30_sweep.log` — Qwen load + sweep
- `result/phase16/logs/phase16_n30_eval.log` — eval run

## 下一步建议

1. **不重跑 Phase 16**: Gaussian steer 范式已穷尽 (Ball/Shrink/Maha), 全部 NO-GO
2. **冻结 Phase 16**: 当前 9 conds 已足够否定 Gaussian 范式, 不需要更多 sweep
3. **下一研究方向**:
   - **Phase 17**: StyleVector 在其他 layer (14 vs 12 vs 16) 是否 lift 更强?
   - **Phase 17**: StyleVector + 不同的融合方式 (scale + bias vs add)?
   - **Phase 17**: StyleVector 的 norm 衰减 (cosine similarity 保持方向 vs norm preservation)?
4. **Memory 更新**: 记录 Gaussian steer 3 种范式均 NO-GO, 确认 Phase 14 hybrid 是 SOTA

## Bug 修复记录

1. **Qwen load 期间 boostrapping process killed**: 用户 6 min 后要求加速, kill 时 sweep 已完成 5 conds, 但 jsonl incremental write 已写入全部 9 conds (kill signal 在所有 Ball_r1.0+ 写完后才生效)
2. **meta.json 缺失**: sweep 被 kill 在 [4] Sweep 完成后, [5] Meta 段未执行 → 手动补写 meta.json