# Phase 16: Gaussian Steer Sweep — **NO-GO** + Maha α=0.5/1.0 是唯一 sem-preserved 选项

**Date**: 2026-08-21
**Question**: 真正使用 Gaussian (μ, σ) 做 steer 是否有 style lift? 比 point estimate μ-only 更有效吗?
**Answer**: **NO** — 6 种 Gaussian steer 方案 (Ball/Shrink/Maha × 多 ρ/α) 都未显著提升 BoK margin vs D_off。但发现:
- **Maha α=0.5/1.0 是唯一不破坏 semantic 的方案** (sem_sim ≈ D_off 0.81, 而 Ball/Shrink 都退化到 ~0.74)
- **A_a1.0 (point estimate μ) 仍是 BoK margin 最高** (+0.0120 vs D_off +0.0104),但 diff 不显著 (CI includes 0)
- **Phase 15.E 1.5B NO-GO 同样验证**: alpha sweep 所有 A_* BoK 都负

## 实验动机

Phase 15.D (2026-08-21) NO-GO 因为 `||σ_residual|| ≈ 175 >> ||μ_residual|| ≈ 24`,ρ × σ × ε steer 爆炸。
Phase 13.B-v2 (2026-08-21) N sweep 确认 N=30 是 sweet spot,但 **σ_norm 仍 ≈ 4-5 × hidden norm** — 不是 sample size 问题。

**用户问**: "怎么才可以真正使用到高斯呢" → 实现 6 种"正确"使用 Gaussian 的 steer 方案:
1. **Ball-constrained**: ε ∈ N(0, I), ||ε|| ≤ 1, s = μ + ρ × σ × ε
2. **Truncated/Shrink σ**: σ' = min(σ, c × |μ|), ball sample + norm clip
3. **Mahalanobis weight**: α × μ × confidence(σ), confidence = 1/(1+σ_norm)

## Sweep 设计

**Steer 范式 (9 configs)**:

| Cond | Mode | Params | 范式 |
|------|------|--------|------|
| D_off | off | — | baseline (无 hook) |
| A_a1.0 | mu_only | α=1.0 | 主路线 (point estimate, Phase 14 hybrid) |
| Ball_r0.3 | ball | ρ=0.3 | σ × ε direction, 弱 noise |
| Ball_r0.5 | ball | ρ=0.5 | σ × ε direction, 中 noise |
| Ball_r1.0 | ball | ρ=1.0 | σ × ε direction, 满 noise |
| Shrink_r0.5 | shrink | ρ=0.5, c=0.3 | σ' ≤ 0.3·\|μ\|, ball sample, norm clip 2·\|μ\| |
| Shrink_r1.0 | shrink | ρ=1.0, c=0.3 | 同上 ρ=1.0 |
| Maha_a0.5 | maha | α=0.5 | α × μ × (1/(1+σ_norm)) (高 σ dim 权降低) |
| Maha_a1.0 | maha | α=1.0 | 同上 α=1.0 |

**Config**:
- N=30 Gaussian (per Phase 13.B-v2 sweet spot)
- 198 users × ≥30 sentences (raw Amazon Baby_Products_2023)
- Qwen2-7B layer 14, last-token injection
- 30 pairs × K=8 samples × 9 conds = 2160 records
- 768d AnnaWegmann style margin (cos_target - cos_off_mean)
- Sentence-bert all-MiniLM-L6-v2 semantic sim vs attrs

**实施**:
- `phase16_n30_gaussian_sweep.py` (v1, sequential per cond) — 慢 (~50s/cond)
- `phase16_v2_batched.py` (v2, **batched multi-cond**: 1 generate call 处理 5 conds × K=8 = 40 samples) — **75s 总 (vs v1 ~13 min, 10x 加速)**
- `phase16_eval.py` (style margin + semantic + bootstrap CI vs D_off & A_a1.0)

## 关键发现

### 1. BoK margin: 没有任何 Gaussian 方案显著 lift over D_off

| Cond | mean_marg | BoK_marg | sem_sim |
|------|----------|----------|---------|
| D_off | -0.0001 | +0.0104 | 0.8138 |
| A_a1.0 | +0.0003 | +0.0120 | 0.7491 |
| Ball_r0.3 | +0.0003 | +0.0113 | 0.7535 |
| Ball_r0.5 | +0.0010 | +0.0112 | 0.7367 |
| Ball_r1.0 | -0.0001 | +0.0100 | 0.7437 |
| Shrink_r0.5 | -0.0003 | **+0.0090** | 0.7441 |
| Shrink_r1.0 | +0.0007 | +0.0109 | 0.7458 |
| Maha_a0.5 | -0.0017 | +0.0106 | **0.8070** |
| Maha_a1.0 | -0.0004 | +0.0114 | **0.8065** |

**vs D_off (paired bootstrap CI)**:
- BoK diff: **全部 CI includes 0** (无显著 lift)
  - A_a1.0: +0.0017 [-0.0021, +0.0056] ✗
  - Ball_r0.3/0.5/1.0: +0.0010/+0.0008/-0.0003 (NS)
  - Shrink_r0.5: -0.0013 (略负,NS)
  - Shrink_r1.0: +0.0005 (NS)
  - Maha_a0.5/1.0: +0.0002/+0.0010 (NS)
- Semantic diff:
  - **Ball/Shrink 全部 CI excludes 0 negative** → 语义显著退化 (-0.06 to -0.08)
  - **Maha α=0.5/1.0 sem diff CI includes 0** → **语义保留** ✓
  - A_a1.0 sem diff -0.065 (退化为代价)

### 2. vs A_a1.0 (主路线): 6 Gaussian 方案与 A_a1.0 持平或退化

- BoK diff vs A_a1.0: 只有 Shrink_r0.5 显著负 (-0.0030 [-0.0058, -0.0008])
- Ball_r0.3/0.5、Maha_a0.5/1.0、Shrink_r1.0 均 NS (与 A_a1.0 持平)
- **A_a1.0 BoK 仍最高 (+0.0120)** — point estimate μ 仍是最有效的 steer 形式

### 3. Semantic preservation 唯一赢家: Maha α=0.5/1.0

| 范式 | sem_sim (vs D_off 0.81) | 损失 |
|------|------------------------|------|
| D_off | 0.8138 | baseline |
| A_a1.0 | 0.7491 | -0.065 |
| Ball/Shrink (ρ noise) | 0.74-0.75 | -0.06 to -0.07 |
| **Maha α=0.5** | **0.8070** | **-0.007 (NS)** |
| **Maha α=1.0** | **0.8065** | **-0.007 (NS)** |

**Mahalanobis weight 范式** 通过 `α × μ × confidence(σ)` 自动降低高 σ dim 的 steer 强度,避免 noise 维度破坏 semantic。这与 A_a1.0 (固定 α=1.0 全 dim 注入) 和 Ball/Shrink (随机方向 noise) 都不同。

## 决策

### Phase 16 主结论: NO-GO on Gaussian steer

- 6 种 Gaussian steer 方案 (Ball/Shrink/Maha × 多参数) 都未在 768d style space 显著 lift BoK margin
- 主路线 A_a1.0 (point estimate μ) BoK margin 最高 (+0.0120),与 D_off 差异 NS 但已是上限
- **Gaussian 范式本身无法突破 style lift 上限** — 即使正确使用 σ,768d style space 中的 BoK margin signal 已在 ~+0.01 量级,继续追求更高 lift 需换评估空间 / 换生成范式 (e.g. 真正 fine-tune)

### 但保留 Maha α=0.5/1.0 作为 **sem-preserved steer 选项**

- 与 A_a1.0 持平的 BoK margin (CI includes 0)
- **保留 semantic** (sem CI includes 0 vs D_off)
- 适用场景: 当 content preservation 优先于 style lift 时,可作为 A_a1.0 的替代

## Phase 15 路线总结

| Phase | 范式 | BoK margin lift | Semantic | 决策 |
|-------|------|----------------|----------|------|
| 15.A | Author ID | n/a | n/a | baseline (necessity test) |
| 15.B | Semantic Preservation | n/a | sem diff NS | PARTIAL-GO |
| 15.D | α × ρ sweep (7B) | A_a1.0 +0.0017 NS | sem -0.016 NS | A_a1.0 (主路线) |
| **15.E** | α sweep 1.5B | **all A_* BoK negative** | sem diff +0.03-0.06 | **NO-GO (1.5B capacity)** |
| **16** | Ball/Shrink/Maha (7B) | **all NS** | Ball/Shrink regress, **Maha preserves** | **NO-GO (Gaussian)** |

## 与历史路线对比

| 路线 | style lift | 风险 | 当前 |
|------|-----------|------|------|
| 14 hybrid (A_a1.0 + 768d rerank) | top-100 56.7% (3.3x 10.19) | rerank post-hoc | **主路线** |
| A_a1.0 (point) | BoK +0.0017 NS | sem -0.06 | 主路线 baseline |
| Maha α=0.5/1.0 | BoK NS | sem 0 (preserved) | **可备选** |
| Ball/Shrink noise | NS | sem -0.07 | NO-GO |
| 1.5B any | negative | sem regress | NO-GO |

## 文件位置

### Scripts (git)
- `result/phase15/scripts/phase16_n30_gaussian_sweep.py` — v1 sequential
- `result/phase15/scripts/phase16_v2_batched.py` — v2 batched multi-cond (10x faster)
- `result/phase15/scripts/phase16_eval.py` — style margin + semantic + bootstrap

### Results (in scratch)
- `phase16_n30_gaussian_sweep.jsonl` — 2160 records (9 conds × 30 pairs × K=8)
- `phase16_n30_user_gaussians_qwen.npz` — 198 users × (μ, σ_diag) at layer 14
- `phase16_n30_eval.json` — per-cond eval + vs D_off / vs A_a1.0 bootstrap CI
- `phase16_n30_per_pair.jsonl` — per-pair BoK margin + mean sem

### Logs
- `phase16_n30.log` — v1 partial (D_off + A_a1.0 + Ball_r0.3 + Ball_r0.5)
- `phase16_v2.log` — v2 batched (Ball_r1.0 + Shrink_r0.5/1.0 + Maha_a0.5/1.0) **75s total**
- `phase16_eval.log` — eval (style + semantic)

## 下一步建议

1. **不再尝试 Gaussian steer** — 范式已达上限,继续 sweep 边际收益 < 0
2. **保留 A_a1.0 (point estimate μ) 作为主路线** + 768d rerank (Phase 14 hybrid)
3. **保留 Maha α=0.5/1.0 作为备选** (sem-preserved 版本,适用 content-critical 场景)
4. **如果要进一步 lift style**:
   - **换生成范式**: 不再尝试 linear intervention,转向 fine-tune (Phase 16.SFT?) 或 exemplar RAG (Phase 10.19 已 PARTIAL-GO top-100 17%)
   - **换评估空间**: 768d AnnaWegmann 上限已达,试 style classifier (train per-user SVM) 或 318d Mahalanobis (Phase 13.E 已 NO-GO)
5. **Memory 更新**: 记录 Gaussian steer NO-GO + Maha sem-preserved 发现

## Bug 修复记录

1. **v1 phase16 sweep 性能**: 9 conds × 30 calls × ~3s = 13 min sweep (太慢)
   - **修复**: 写 v2 batched multi-cond, 5 conds × K=8 = 40 samples per call, 30 calls 总
   - **结果**: 75s sweep (10x faster), 同样的 hook per-row steer injection
2. **v2 steer_vecs reset**: 每次 pair 必须 `steer_vecs.clear()` + `extend()` 重新注入,避免 stale state

## 关键经验

1. **batched multi-cond generation 加速**: 当多个 conditions 共享 prompt 但不同 steer vector 时,可以用 batched multi-cond 在单次 generate 内处理 N_CONDS × K samples per pair。
   - 实施: `steer_vecs: list = [v_cond0_s0, v_cond0_s1, ..., v_condN_sK]`,hook 用 `for b in range(B): steer_vecs[b]`
   - 加速: 5 conds × K=8 = 40 batch per call,30 calls 总 (vs 150 sequential calls)
2. **eval 显示 Maha sem-preserved** 是 Phase 16 唯一 actionable 发现 — 可能作为 A_a1.0 的替代
3. **Phase 15 路线整体已 plateau**: 继续 α/ρ/Maha sweep 边际收益 < 0.001,应转向新范式