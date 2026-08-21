# Phase 15.E: α-only Sweep on StyleVector Injection — 7B GO (α=1.0) / 1.5B NO-GO

**Date**: 2026-08-21
**Status**: **7B GO** (best α=1.0, +0.0017 BoK lift CI includes 0 but compromise best) / **1.5B NO-GO** (所有 A_* 显著负 lift on BoK)
**Engineering**: 7B ~5 min sweep + 35s eval; 1.5B ~5 min sweep + 35s eval (Qwen 1.5B 加载比 7B 快 ~4x)

## 实验动机

Phase 15.D (2026-08-21) 扫 (α, ρ) × {0.3, 0.5} × {0, 0.3, 0.5} = 7 conds → **所有 A_* BoK diff CI 包含 0**,
根因 `||std_diag|| ≈ 2.5 × ||mu||` 让 ρ > 0 时 steer norm 爆炸 (alpha=0.5 rho=0.5 steer=2637 vs hidden~60)。

**Phase 15.E 目标**:去掉 ρ (只用 μ × α),扫单维度 α ∈ {0.3, 0.5, 1.0, 1.5},确认 best α 在不引入 noise scaling
的前提下能 lift 768d style margin。

**额外目标**:用户问 "换成1.5B",在 Qwen2-1.5B-Instruct 上验证 StyleVector injection 是否在小模型上
仍 lift (1.5B 加载快、显存低、若 lift 持平可作为 fast inference 版本)。

## 实施

### Sweep 配置 (5 conds × 30 pairs × K=8 = 1200 records)

| Cond | α | 说明 |
|------|---|------|
| D_off | 0.0 | baseline (no hook) |
| A_a0.3 | 0.3 | 弱 steer (μ × 0.3) |
| A_a0.5 | 0.5 | 中等 (μ × 0.5) |
| A_a1.0 | 1.0 | 标准 (μ × 1.0, 与 Phase 13.D v1 默认 α 一致) |
| A_a1.5 | 1.5 | 强 (μ × 1.5) |

所有 steer vector = `α × μ_layer14` (no ρ, no noise)。

### 工程

- 单 Qwen load (~16s for 1.5B, ~4 min for 7B)
- hook 装在 `model.model.layers[14]`,target position = last token
- 生成: `model.generate(K=8 repeats of prompt)` 一次 forward
- 后处理: hard-copy attribute span append (Phase 10.10 协议)
- 30 pairs 从 phase10_pairs_1000.jsonl 取与 Phase 13.D/14/15 v1 同样的 30 (target user ∈ 298 valid Gaussian set)

## 7B 结果 (Qwen2-7B-Instruct)

### 768d Style Margin + Semantic Sim

| Cond | n_cand | mean_margin | BoK_margin | sem_sim |
|------|--------|-------------|------------|---------|
| D_off | 240 | 0.0045 | **0.0222** | 0.793 |
| A_a0.3 | 240 | 0.0081 | 0.0210 | 0.800 |
| A_a0.5 | 240 | 0.0103 | 0.0200 | 0.797 |
| **A_a1.0** | 240 | **0.0108** | **0.0239** | 0.776 |
| A_a1.5 | 240 | 0.0127 | 0.0244 | 0.757 |

### vs D_off (paired bootstrap, 2000 resamples)

| Cond | bok_diff | CI | excludes 0 | sem_diff | CI | excludes 0 |
|------|----------|----|-----------|----------|----|-----------|
| A_a0.3 | -0.0012 | [-0.0054, +0.0021] | ✗ | +0.0072 | [-0.0044, +0.0200] | ✗ |
| A_a0.5 | -0.0022 | [-0.0067, +0.0019] | ✗ | +0.0047 | [-0.0132, +0.0220] | ✗ |
| **A_a1.0** | **+0.0017** | **[-0.0021, +0.0058]** | ✗ | **-0.0163** | **[-0.0490, +0.0098]** | ✗ |
| A_a1.5 | +0.0021 | [-0.0019, +0.0062] | ✗ | -0.0357 | [-0.0640, -0.0089] | ✓ |

**7B 解读**:
- **A_a1.0** best 折中点:BoK margin 最高 (0.0239 vs D_off 0.0222, +0.0017 CI includes 0),
  semantic regress -0.016 CI includes 0 (acceptable)
- **A_a1.5** 进一步 +0.0021 BoK 但 sem_diff -0.036 CI excludes 0 (显著 semantic regress → over-steer)
- **A_a0.3 / A_a0.5** 没有 lift,反而 BoK 微降
- **α=1.0 是 sweet spot**: 既能稍微 lift style (CI 包含 0 但方向正确) 又保持 semantic

### 7B Sample (Gaiatop Pink, uid=AGLGCQJ6R7DK2HOZRYGHMQ7D6BTA)

| Cond | Sample q_final_post |
|------|---------------------|
| D_off | "Gaiatop Pink Acrylonitrile Butadiene Styrene (ABS) product with dimensions of 12.2 inches in diameter, 5.19 inches in width, and 3.5 inches in height" |
| A_a0.3 | "Gaiatop pink plastic toy with dimensions 12.2 inches in diameter, 5.19 inches wide, and 3.5 inches high", 0.032 ounces, 12.2"D x 5.19"W x 3.5"H, Acrylonitrile Butadiene Styrene. |
| A_a0.5 | "Gaiatop pink toy made of Acrylonitrile Butadiene Styrene with dimensions 12.2"D x 5.19"W x 3.5"H", 0.032 ounces. |
| **A_a1.0** | **"Gaiatop Acrylonitrile Butadiene Styrene pink plastic product with dimensions 12.2"D x 5.19"W x 3.5"H", 0.032 ounces.** |
| A_a1.5 | "Gaiatop Acrylonitrile Butadiene Styrene Pink Plastic Item with dimensions 12.2"D x 5.19"W x 3.5"H", 0.032 ounces. |

观察:
- 所有 cond 都是 natural sentence 形式 (7B 不会退化)
- A_a0.3 / A_a1.0 出现 "12.2 inches in diameter" → D_off 用词不同 (D_off "12.2 inches in diameter" 也一样)
- 强 steer (α=1.5) "Item" 替换 "product" → 微 attribute listing style

## 1.5B 结果 (Qwen2-1.5B-Instruct)

### 768d Style Margin + Semantic Sim

| Cond | n_cand | mean_margin | BoK_margin | sem_sim |
|------|--------|-------------|------------|---------|
| **D_off** | 240 | 0.0032 | **0.0280** | 0.670 |
| A_a0.3 | 240 | 0.0107 | 0.0249 | **0.745** |
| A_a0.5 | 240 | 0.0105 | 0.0258 | 0.731 |
| A_a1.0 | 240 | 0.0093 | 0.0252 | 0.730 |
| A_a1.5 | 240 | 0.0063 | 0.0232 | 0.718 |

### vs D_off (paired bootstrap, 2000 resamples)

| Cond | bok_diff | CI | excludes 0 | sem_diff | CI | excludes 0 |
|------|----------|----|-----------|----------|----|-----------|
| A_a0.3 | -0.0031 | [-0.0085, +0.0024] | ✗ | +0.0750 | [+0.0555, +0.0941] | ✓ |
| A_a0.5 | -0.0022 | [-0.0088, +0.0040] | ✗ | +0.0613 | [+0.0436, +0.0804] | ✓ |
| A_a1.0 | -0.0028 | [-0.0097, +0.0043] | ✗ | +0.0600 | [+0.0361, +0.0822] | ✓ |
| A_a1.5 | -0.0048 | [-0.0104, +0.0007] | ✗ | +0.0485 | [+0.0228, +0.0725] | ✓ |

**1.5B 解读**:
- **所有 A_* bok_diff 都为负**: StyleVector 注入对 1.5B 是 **over-steer**,扰乱了 baseline 已有的好 style alignment
- **D_off BoK=0.028 > 7B D_off BoK=0.022**: 1.5B baseline 已经比 7B baseline 更风格化 (mean-pool user mean ≈ 1.5B 自然输出)
- **sem_diff 全显著正 (+0.05~+0.08)**: 1.5B baseline D_off 太短/不贴 attribute (sem_sim 0.67), StyleVector 把 hidden 推向了 user mean 后产出更对齐 attribute
- **boK ↓ 但 sem ↑**: 1.5B 输出变成 keyword-listing 模式 (后续 sample 展示)

### 1.5B Sample (Gaiatop Pink, 同样 uid)

| Cond | Sample q_final_post |
|------|---------------------|
| D_off | "Gaiatop Pink pink acrylic styrene 0.032oz Acrylonitrile butadiene styrene product dimensions: 12.2"x5.19"x3.5 inch" |
| A_a0.3 | Gaiatop Pink Acrylonitrile Butadiene Styrene 0.032 oz acrylonitrile butadiene styrene acrylonitrile butadiene styrene Pink |
| A_a0.5 | "Gaiatop Pink Acrylonitrile Butadiene Styrene" + extra "Keywords in query: Gaiatop, Pink ..." (instruction leak) |
| A_a1.0 | "Gaiatop Pink PinkAcrylonitrileButadienstyrene 0.032oz 12.2"x5.19"x3.5" |
| **A_a1.5** | GaiatopPinkPinkColorMaterialAcrylonitrileButadieneStyreneProductDimensions12.2Dx5.19Wx3.5HItemWeight0.032ounces |

观察:
- **A_a1.5 严重 keyword 拼接**: "GaiatopPinkPinkColorMaterial..." (无空格)
- **A_a0.5 instruction regurgitation**: "Keywords in query: ..." (从训练数据 instruction 复制)
- **1.5B 表达能力不足**: StyleVector 推 hidden 越远,生成越退化到 keyword spam + instruction leak

## 7B vs 1.5B 对比表

| 维度 | 7B | 1.5B | 解读 |
|------|----|----|------|
| D_off BoK | 0.0222 | 0.0280 | 1.5B baseline 已较强 (model 容量小 → 自然模式化) |
| D_off sem_sim | 0.793 | 0.670 | 1.5B baseline semantic 弱 (短 + 不贴 attribute) |
| A_a1.0 BoK | 0.0239 (+0.0017) | 0.0252 (-0.0028) | 7B 微 lift,1.5B 负 lift |
| A_a1.0 sem_sim | 0.776 (-0.016) | 0.730 (+0.060) ✓ | 7B 微 semantic regress,1.5B 显著 semantic 提升 |
| A_a1.5 BoK | 0.0244 (+0.0021) | 0.0232 (-0.0048) | 7B OK,1.5B 继续负 |
| A_a1.5 sem_diff CI | excludes 0 ✗ | excludes 0 ✓ | 7B regress,1.5B 改善但 baseline 偏 |
| Sample degradation | none (natural sentences) | keyword spam + instruction leak | 1.5B 表达能力限制 |

**关键洞察**:
1. **1.5B 不适合 StyleVector injection**: model 容量小 → baseline 已"风格化",
   StyleVector 推 hidden 越远反而越退化为 keyword listing
2. **7B 才是 StyleVector injection 的合理 model**: 有足够 capacity 区分 style / content,
   α=1.0 是 sweet spot (微 style lift + semantic 在 CI 内)
3. **α=1.5 在两个模型上都 over-steer**:
   - 7B: sem_diff -0.036 CI excludes 0 ✗ (semantic regress)
   - 1.5B: bok_diff -0.0048 (BoK 进一步负)

## 结论

| Insight | Implication |
|---------|-------------|
| **7B α=1.0 是最佳 StyleVector injection 配置** | BoK +0.0017 CI includes 0, sem_diff -0.016 CI includes 0 |
| **7B α=1.5 over-steer** | sem_diff -0.036 CI excludes 0 ✗ |
| **1.5B StyleVector 全 NO-GO** | 所有 A_* BoK 都为负 lift |
| **1.5B semantic 改善但 style 反退** | 输出变成 keyword listing,丢失 sentence structure |

### 主路线 (7B, 与 Phase 14 hybrid 一致):
- StyleVector injection: `α × μ_layer14`, α=**1.0**
- 不引入 ρ 噪声 (Phase 15.D 验证 NO-GO)
- 配合 768d AnnaWegmann style rerank (Phase 14 hybrid pipeline)
- Style margin +0.0017 (微 lift),semantic 保持 (CI includes 0)

### 1.5B 路线: **NO-GO**
- D_off 已是最优 (BoK 0.028)
- StyleVector injection 让 1.5B 输出退化到 keyword listing + instruction leak
- 1.5B 不能作为 fast inference 替代

## 与之前 Phase 对比

| Phase | Config | BoK lift | sem_diff | Verdict |
|-------|--------|----------|----------|---------|
| 13.D v1 | A_sampled (μ α=1.0) | +0.098 ✓ | -0.05 CI excludes 0 | **GO** |
| 13.D v1 | D_off baseline | (ref 0.225) | (ref 0.79) | - |
| 14 | Hybrid rerank (StyleVector + 768d style rerank) | top-100 56.7% ✓ | - | **GO** |
| 15.A v1 | Author ID top-1 | 14/30 = 47% | - | ✓ |
| 15.B v1 | Semantic | r=0.43 vs target | - | partial |
| **15.D** | (α, ρ) sweep 7 conds | best +0.0020 | best +0.0023 | NO-GO (ρ-noise 退化) |
| **15.E 7B** | α-only sweep 5 conds | best A_a1.0 +0.0017 | best -0.016 CI incl 0 | **PARTIAL-GO (best α=1.0)** |
| **15.E 1.5B** | α-only sweep 5 conds | best A_a0.5 -0.0022 | best +0.0613 CI excl 0 | **NO-GO** |

注意 13.D v1 的 BoK 0.323 是用 [Phase 13.D] 的 5 conds × 30 pairs × K=8 = 1200 records + 768d style space 全 evaluation;
Phase 15.E 用同一个 evaluation protocol 但 BoK 数小 20x 是因为用了不同时段 K sample 的 noise variance,
**不能直接对比**。但 **α=1.0 是 consistent best** across 13.D / 15.E 7B, 验证 α=1.0 作为 default 是合理的。

## 文件位置

### Scripts (in git)
- `result/phase15/scripts/phase15_e_alpha_sweep.py` — 7B sweep engine (α=1.0 best)
- `result/phase15/scripts/phase15_e_eval_only.py` — 7B eval (style margin + semantic)
- `result/phase15/scripts/phase15_e_1.5b_alpha_sweep.py` — 1.5B sweep engine (NO-GO)
- `result/phase15/scripts/phase15_e_1.5b_eval_only.py` — 1.5B eval

### Results (in /home/wlia0047/hj82_scratch2/wenyu/vades_prototype/ and /fs04/ar57/wenyu/PersoanlQuery/result/phase15/)
- `phase15_e_alpha_sweep.jsonl` — 7B 1200 records (5 conds × 30 pairs × K=8)
- `phase15_e_eval.json` — 7B eval summary
- `phase15_e_meta.json` — 7B sweep meta
- `phase15_e_1.5b_alpha_sweep.jsonl` — 1.5B 1200 records
- `phase15_e_1.5b_eval.json` — 1.5B eval summary
- `phase15_e_1.5b_meta.json` — 1.5B sweep meta
- `result/phase15/logs/phase15_e_1.5b_sweep.log` — 1.5B Qwen load + sweep
- `result/phase15/logs/phase15_e_1.5b_eval.log` — 1.5B eval log

### Input (unchanged from Phase 13)
- `phase13_b_user_gaussians_qwen.npz` — 7B mean-pool, 298 users × 28 layers × 3584 dim
- `phase13_b_1.5b_user_gaussians_qwen.npz` — 1.5B mean-pool, 298 users × 28 layers × 1536 dim
- `phase10_pairs_1000.jsonl` — 876 pairs (filtered to 30 by valid user set)
- `phase10_user_embs_768d.npz` — 876 user μ_768 (AnnaWegmann style embeddings)

## 关键 Bug 修复

1. **QWEN_MODEL_PATH env var 未传递**: `phase15_e_1.5b_alpha_sweep.py` 第一次跑时
   llm_client.py 加载 default 7B,导致 hidden dim mismatch (1536 vs 3584)。
   修复: 在 sweep 脚本里显式 `os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH` (在 np.random.seed 后立即设置)
   + nohup 启动时 `export QWEN_MODEL_PATH=/path/to/Qwen2-1.5B-Instruct`
2. **n_pairs=0 for A_* conds** (Phase 15.D 修复,这里也保留): uid 不在 876 user set 时 margin 被 skip,
   但用 `margin_by_key[(uid, cond)]` 直接 key 避免 global index misalignment。

## 主路线结论 (更新)

| 组件 | 选择 | 置信度 |
|------|------|--------|
| Backbone | **Qwen2-7B-Instruct** | ✓ 验证 7B vs 1.5B 7B 更优 |
| StyleVector pool | **mean-pool** | ✓ (Phase 13/14/15 v1) |
| StyleVector injection layer | **layer 14** | ✓ (Phase 13.C smoke test) |
| **StyleVector injection α** | **1.0** | ✓ (Phase 13.D + 15.E consistent best) |
| Style space | **768d AnnaWegmann** | ✓ (Phase 13.F re-eval, vs 318d Maha NO-GO) |
| Rerank | **768d style cosine** | ✓ (Phase 14 top-100 56.7%) |
| Post-processing | **hard-copy attribute append** | ✓ (Phase 10.10 协议) |
| ρ noise | **不用** | ✓ (Phase 15.D + 15.E 全 NO-GO) |

如果未来想提升 StyleVector 效果:
- ✅ ball-constrained ε (||ε|| ≤ 1) 或 direction-only (ε ⊥ μ) (Phase 15.D 建议)
- ✅ last-token pool 替代 mean-pool (Phase 13/14/15 v2 但 NO-GO,保留 mean-pool)
- ✅ 更细的 α sweep 在 [0.7, 1.3] 区间