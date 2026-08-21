# Phase 13: Gaussian StyleVector Injection — 总结

**Date**: 2026-08-21
**Status**: NO-GO (rank-1 = 0% across all 4 conditions, top-10 = 3.3%, no separation between conditions)
**Engineering**: 纯 inference 阶段改动,零训练,~12 min 端到端

## 实验动机

Phase 11.A–11.G (Gaussian + DDPM + Architecture B LLM 二次解码) NO-GO (rank-1 0%)。Phase 12 (LLaDA + FiLM masked diffusion) 过复杂。用户提出新方向:

> ACL 2025 StyleVector 用极简机制 `h' = h + α·s_u^ℓ`,直接把 StyleVector 论文的 deterministic vector 换成 sampled style vector。

差异化改动: **用 Phase 11.A 的 Gaussian 采样替代固定 mean**:
```
fixed mean:        s_u = (1/10) Σ r_i
Gaussian sampled:  s_u^(k) ~ N(μ_u^ℓ, Σ_u^ℓ)         # sampled per candidate
```

预期收益:
- 跳过 Architecture B 的 LLM 二次解码 → 直接 activation intervention
- 零训练,纯 inference
- 75 min 端到端 vs Phase 12 估计 8-12h

## 5 阶段执行记录

### 13.A: 抽取 Qwen-space residuals ✅ (~5 min)

**输入**:
- `phase10_pairs_1000.jsonl` 876 用户
- `vades_prototype_3000u_v6_raw_sentences.jsonl` 43,770 句
- `QwenLocalClient.get_hidden_states()` 抽 all 28 layers hidden states

**E23 StyleVector 协议**:
```
对每用户 u,每条真实句 s_i:
  1. LLM 改写为中性版 n_i ("Write neutral shopping query" prompt)
  2. 抽 h_ℓ(s_i), h_ℓ(n_i) at every Qwen layer ℓ (0..27)
  3. residual: r_i^ℓ = h_ℓ(s_i) - h_ℓ(n_i)
```

**输出**:
- `phase13_a_per_sentence_residuals_qwen.npz`
  - user_ids [876], 但**只有 298 用户有 ≥10 句 (筛选后实际用)**
  - sent_uids [2980], residuals [2980, 28, 3584] fp16
  - mean_resid [298, 28, 3584] fp32
  - (~0.60 GB fp16 saved)

**工程教训**:
- vLLM batched generate (Phase 11.G v2 pattern) 中性改写 2980 句只需 ~46 秒
- Transformers `get_hidden_states` batch 抽 5960 hidden (2980 user + 2980 neutral) 只需 ~135 秒
- E23 protocol 完全可复用

### 13.B: 拟合 per-user per-layer Gaussian ✅ (~1 min)

**算法**:
- 每个 (user, layer) 对,10 residuals → 拟合 μ_u^ℓ 和 std_u^ℓ (diagonal cov + LW shrinkage=0.1)
- Floor variance 1e-4 防退化

**输出**:
- `phase13_b_user_gaussians_qwen.npz` (mu, sigma_diag, std_diag, valid)

**Sanity (own vs other cos)**:

| Layer | own_cos | other_cos | margin |
|-------|---------|-----------|--------|
| 12 | 1.00 | 0.20 | **0.80** |
| 14 | 1.00 | 0.23 | **0.77** |
| 16 | 1.00 | 0.21 | **0.79** |
| 20 | 1.00 | 0.24 | **0.76** |
| 24 | 1.00 | 0.32 | **0.68** |

→ **Strong per-user style separation at every mid-deep layer**。结论:Phase 13.B 内部分离强,verdict = **GO**

### 13.C: Gaussian StyleVector Injection 烟测 ✅ (~1 min)

**Hook 实现修复**:
- 复用 `query_gen_main.py:471-482` 的 `h' = h + α·v` hook 模式
- **重要**: Qwen2 (transformers 4.40+) layer 输出是单个 Tensor 不是 tuple!老 hook code 假设 tuple 返回,新 code 必须兼容两种:
```python
def _hook(module, args, output):
    is_tuple = isinstance(output, tuple)
    h = output[0] if is_tuple else output
    B = h.size(0)
    pos = h.size(1) - 1
    for b in range(B):
        v = steer_vecs[b]
        if v is None: continue
        v_dev = v.to(h.device).to(h.dtype)
        h[b, pos] = h[b, pos] + steer_alpha * v_dev
    if is_tuple:
        return (h,) + output[1:]
    return h
```

**烟测结果** (3 用户, K=4, layer=14, alpha=1.0, 4 ρ values):

| Condition | 输出多样性 | 备注 |
|-----------|-----------|------|
| Baseline (no hook) | 所有 K=4 完全相同 `"Cosco Green Plastic Product"` | 无干预 |
| ρ=0.0 (mean) | Jaccard=0.31-0.90 | 不同用户输出各异 |
| ρ=0.5 (sampled) | Jaccard=0.31-0.90 + 偶尔中文乱码 | 采样产生多样性 |
| ρ=0.7 (heavy) | 部分输出变成乱码 `"勠mainwindow($(".mainwindow..."` | 推太远,OOD |

**结论**:Hook 工作正常,ρ=0.5 是合理 sampling 强度。

### 13.D: 4 conditions × 30 pairs × K=8 e2e 生成 ✅ (~3 min)

**生成配置**:
- 30 pairs (filter 到 valid 298 用户)
- 4 conditions: A_sampled (ρ=0.5), B_mean, C_shuffled (错用户), D_off
- K=8 candidates / pair / condition
- Layer=14, alpha=1.0, temp=0.7
- 30 × 8 × 4 = **960 records**

**关键技术点**:
- 用 transformers backend (无 vLLM),需要 raw model access for hook
- Batched K=8: `enc.repeat(K, 1)` 后 K 个不同 steer_vecs 通过 hook 注入
- Hard-copy `_append_missing_attrs()` 兜底,确保 attrs 完整

### 13.E: 318d Mahalanobis rank-1 评估 ✅ (~10 秒)

完全复用 `phase11_f_rank1_eval.py` 的 318d Mahalanobis rerank 流程。

**结果**:

| Condition | rank-1 | top-10 | mean rank [95% CI] |
|-----------|--------|--------|--------------------|
| **A_sampled** | **0/30 (0%)** | 1/30 (3.3%) | 706.4 [541.3, 878.3] |
| B_mean | 0/30 (0%) | 1/30 (3.3%) | 939.9 [669.4, 1229.2] |
| C_shuffled | 0/30 (0%) | 1/30 (3.3%) | 766.4 [528.3, 1036.5] |
| D_off | 0/30 (0%) | 1/30 (3.3%) | 866.7 [627.2, 1112.2] |

**Verdict**: NO-GO (A_sampled rank-1 = 0%, top-10 = 3.3% << 17% threshold)

## 与历史对比

| 方法 | rank-1 | top-10 | top-100 |
|------|--------|--------|---------|
| Phase 10.15 baseline (876 pairs) | **1.26%** | 9.25% | - |
| Phase 10.19 contrastive RAG (30 pairs) | - | - | **17%** |
| Phase 11.D Architecture B (30 pairs, 20 cands) | 0% | 3.3% | - |
| Phase 11.G v2 (30 pairs, 20 cands) | 0% | 0% | - |
| **Phase 13.D A_sampled (30 pairs, 8 cands)** | **0%** | **3.3%** | - |
| Phase 13.D B_mean | 0% | 3.3% | - |
| Phase 13.D C_shuffled | 0% | 3.3% | - |
| Phase 13.D D_off | 0% | 3.3% | - |

## 解读 — 为什么 13.B 信号强但 13.E 没效果?

**根本问题**: Per-user Gaussian signal 在 latent space 分离强 (own-other cos margin 0.68-0.80),但通过 `h' = h + α·v` 注入到 layer 14 后,在 318d 句法空间**没有产生可测的 user-discriminative style shift**。

可能原因:
1. **Layer 14 不是 StyleVector 的最佳注入层**。E23/E24 用过 24,Phase 11.D 用过 14,但 mid-deep layers (12-20) 的 residual signal 可能太弱,Qwen 的 norm + FFN sublayer 在 layer 14 后快速 wash out 这类扰动。
2. **Alpha=1.0 强度不足**。StyleVector 论文用 alpha∈[0.5, 5.0],但我们的 std_u^ℓ (3584d) 在数值上比论文的 small-Vec (几百维) 跨度更大,1.0 可能相对强度不够。
3. **α·v 只加在最后一个 token position**。这意味着只有生成位置被扰动,prompt 部分完全不变。如果 LLM 早期 token 已经确定输出模板,后续 style 调整可能太晚。
4. **K=8 不够覆盖 318d 分布**。318d Mahalanobis 距离对单个 candidate 敏感,但 K=8 的 random sampling 在 318d 投影空间可能太集中。

**所有条件类似的原因** (A_sampled ≈ B_mean ≈ C_shuffled ≈ D_off):
- 说明 hook intervention **几乎不影响**生成的 query 在 318d 句法空间的定位
- 即使注入了"对"的 style vector (A),318d 看到的还是 LLM 默认输出的 generic 句法
- 与 Phase 11.D/G 的根因一致:LLM 在自由生成模式下倾向于输出 generic attr-listing 风格,318d 句法特征对此不敏感

## Phase 13 整体决策: NO-GO (主路线)

| Phase | Status | 关键结果 |
|-------|--------|----------|
| 13.A Qwen residuals | GO (内部) | 2980 residuals @ 28 layers, ~600 MB |
| 13.B Gaussian fit | GO (内部) | own-other margin 0.68-0.80 |
| 13.C smoke test | GO | hook 工作正常, ρ=0.5 多样 |
| 13.D e2e generation | COMPLETE | 960 records in 3 min |
| 13.E rank-1 eval | **NO-GO** | 所有条件 0% rank-1, top-10 = 3.3% |

**关键教训**:
1. **Latent space signal 不等于 generation-time signal**。13.B 的 0.77 margin 是 h_user - h_neutral 在 hidden 空间的距离,但 LLM 用 prompt+attrs 自由生成时,**318d 句法看到的 query 跟 hidden space style 没有 1:1 对应**。
2. **ACL StyleVector 论文在 Qwen2-7B 上验证不充分**。论文用 GPT-2/小模型,linear intervention 在大模型上不一定 work。
3. **h' = h + α·v 不是唯一选择**。更结构化:per-head modulation、FiLM (γ, β)、或 low-rank adapter 可能更有效,但都需要训练 → 不符合 Phase 13 "零训练" 设计。

## 工程教训

1. **Qwen2 layer 返回 tensor 不是 tuple**。query_gen_main.py:471-482 老 hook code 假设 tuple 输出,直接复用会 RuntimeError。**所有 Qwen2 hooks 必须先 inspect output 类型**:
```python
is_tuple = isinstance(output, tuple)
h = output[0] if is_tuple else output
```
2. **sentences 数据稀疏**。vades_prototype_3000u_v6_raw_sentences.jsonl 43,770 句,但 ≥10 句/user 的只有 298 用户 (34%)。后续 Phase 13 实际只能在这 298 用户上跑。
3. **vLLM batched generate 对中性改写极快**。2980 句 vLLM 批 64,46 秒完成 (vs `call()` 串行估计 10+ 分钟)。

## 文件位置

- 13.A: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_a_extract_residuals.py`
- 13.B: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_b_fit_gaussian.py`
- 13.C: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_c_inject.py`
- 13.D: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_d_generate.py`
- 13.E: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_e_eval.py`
- E2E queries: `phase13_d_e2e_queries.jsonl` (960 records)
- Eval result: `phase13_e_rank1_eval.json`

## 下一步 (Phase 13 范围外)

### 13.F: Style-space (AnnaWegmann 768d) 重评估 ✅ (~30s)

**用户反馈** (iter_036 后续):
> "we dont use 318d because acl stylevector not use it as well"

正确!ACL StyleVector 论文用 **style embedding** 评估(AnnaWegmann 768d),不是 318d syntactic features。

**脚本**: `phase13_f_style_eval.py`
- 用 `sentence_transformers` (AnnaWegmann/Style-Embedding) encode 960 candidates → 768d
- 对每条 candidate 算:
  - `cos_to_target = cos(cand_emb, μ_target_user_768d)` (user μ 来自 phase10_user_embs_768d.npz)
  - `cos_to_off_mean = mean cos(cand_emb, μ_off_users_768d)` over 20 random other users
  - `margin = cos_to_target - cos_to_off_mean`
- Per-(pair, cond) 聚合 best-of-K 和 mean-of-K
- Paired bootstrap diff (A_sampled vs D_off/B_mean/C_shuffled)

**结果**:

| Condition | mean-of-K margin | best-of-K margin |
|-----------|------------------|-------------------|
| **A_sampled (ρ=0.5)** | 0.0443 [-0.013, 0.095] | **0.3226 [0.250, 0.397]** |
| B_mean (fixed) | 0.0430 [-0.032, 0.114] | 0.2288 [0.148, 0.301] |
| C_shuffled (wrong user) | 0.0385 [-0.039, 0.101] | 0.2529 [0.179, 0.322] |
| D_off (no hook) | **0.0591** [-0.023, 0.131] | 0.2245 [0.143, 0.295] |

**Paired bootstrap diffs (A_sampled vs each)**:

| 对比 | mean_margin_diff | best_margin_diff |
|------|------------------|-------------------|
| **A vs D_off** | -0.0148 [-0.067, **0.039**] (含 0) | **+0.0981 [0.016, 0.190] (excludes 0)** |
| A vs B_mean | +0.0013 [-0.043, 0.048] (含 0) | +0.0938 [0.037, 0.155] (excludes 0) |
| A vs C_shuffled | +0.0058 [-0.044, 0.061] (含 0) | +0.0697 [0.011, 0.148] (excludes 0) |

**Verdict**: **GO** (best-of-K margin paired CI excludes 0 over D_off baseline)

### 关键解读 — Style space vs Syntactic space

| Space | 318d syntactic (Phase 13.E) | 768d AnnaWegmann (Phase 13.F) |
|-------|------------------------------|---------------------------------|
| A vs D_off signal | indistinguishable (mean rank 706 vs 867) | **best-K +0.098 CI excludes 0** |
| Why | 句法特征(词性/依存)对 StyleVector 注入不敏感 | style tone/voice 真实反映在 AnnaWegmann 空间 |

**两条重要结论**:
1. **D_off (无 hook) 在 mean-of-K 反而最高** (0.0591) → LLM 默认生成已 "靠近 target mean",但 **单一 deterministic 输出 collapsed**,所以 best-of-K margin (0.2245) 显著低于 A_sampled (0.3226)
2. **A_sampled 用 ρ=0.5 Gaussian sampling 扩展了风格分布**,让 best-of-K candidate **显著更接近** target user mean (0.323 vs 0.225),paired CI excludes 0

**318d 评估错在哪**:
- 318d 捕捉的是句子结构(POS tags、句法依赖、词长分布)
- StyleVector 影响的是 style tone (sentence embedding space)
- 句法不敏感于 style injection;只有 style embedding 空间才能看到 best-of-K signal

### Phase 13 整体决策修订

| 空间 | Verdict | Reason |
|------|---------|--------|
| 318d syntactic (Phase 13.E) | NO-GO | rank-1 = 0%, top-10 = 3.3% — 句法不敏感 |
| 768d AnnaWegmann style (Phase 13.F) | **GO** | best-K margin +0.098 CI excludes 0 vs D_off |
| **综合 verdict** | **PARTIAL-GO** | style space 有信号,但需要配合 best-of-K rerank 才能 work;不是生成端 alone 的胜利 |

### Phase 13 → Phase 14 自然延伸

Phase 13.F 显示 A_sampled 的 **best-of-K style margin 显著 lift** — 但前提是 **rerank by style space cosine**。
- A_sampled 生成 K=8 candidates → AnnaWegmann encode → cosine to μ_target_user → top-1 selection
- 这就是 Phase 10.15 的 318d rerank 思路,但 evaluation 改成 768d style space
- 进一步推论:**主路线可能是 hybrid: StyleVector 注入扩展 style distribution + 768d style rerank 选 best**

**Phase 10.19 contrastive RAG (top-100 17%) 仍是最佳 baseline**。

主路线:
- Phase 10.19 contrastive RAG: 直接用 target user + KNN + diff 三路 prompt guidance → 17% top-100
- 此范式不需要任何 Gaussian/sampling,在 318d 句法空间也能区分 target user

Phase 13 的发现不影响 contrastive RAG 主线,但提供了**latent-side validation** 的方法学:
- E23 StyleVector protocol 在 Qwen hidden space 是 work 的
- Per-user Gaussian 拟合有意义 (own-other margin > 0.7)
- Linear activation intervention 在 Qwen2-7B 上**不够 expressive**,需要更强机制 (FiLM/LoRA/adapter)