# Phase 11.B: Gaussian Sampling Diversity Test — 总结

**Date**: 2026-08-20
**Status**: GO (Diversity gain 1.57x > 1.5x PASS 阈值)

## 实验动机

Phase 11.A 验证了 AnnaWegmann 768d 用户空间信号极强 (held-out rank-1 73.5%)。下一步:**验证 Σ_u 携带的协方差信息是否能转化为生成多样性**——如果从 N(μ_u, Σ_u) 采样 s_u 比用单 μ_u 产生更多样的 query,Gaussian 范式就有价值。

## 4-Control 设计

每个 (pair, control) 产生 10 个 candidates:
- **A sampled-z**: 从 N(μ_u, Σ_u) 采 K=10 个 s_u,每个 s_u 生成 1 candidate
- **B mean-z**: μ_u 单 mean 作 prefix,10 个不同 seed
- **C shuffled-z**: μ_other (另一个 user) 作 prefix,sanity check
- **D injection-off**: 无 prefix,sanity baseline

30 pairs × 4 conds × 10 cands = **1200 generations** (T5-large fp16, ~7 分钟 GPU)

## 关键结果 (30 pairs)

```
Diversity (1 - mean_pair_cos among 10 candidates per pair):

  sampled-z      : diversity = 0.6999 (mean_pair_cos = 0.3001)  ← 最 diverse
  mean-z         : diversity = 0.4458 (mean_pair_cos = 0.5542)
  shuffled-z     : diversity = 0.4394 (mean_pair_cos = 0.5606)
  injection-off  : diversity = 0.5732 (mean_pair_cos = 0.4268)

Diversity gain (sampled-z / mean-z) = 1.57x  ← PASS (>= 1.5x)
```

## 解读

1. **Sampled-z 是最 diverse 的** (mean_pair_cos 仅 0.30,candidates 之间 cos sim 很低)。Σ 携带的信息确实让 prefix 沿不同 style subspace 移动。
2. **Mean-z 与 shuffled-z 几乎相同 diversity** (0.4458 vs 0.4394)。这表明 **single mean prefix 把所有 candidate 推向同一个 style anchor**,candidates 之间的差异主要来自 T5 的采样温度,不是 style 信号。
3. **Injection-off 也较高 diversity** (0.5732)。无 prefix 时 T5 自由发挥,diversity 反而更高——这反向说明:**任何 prefix injection 都在压缩 T5 的输出分布**。
4. **关键对比**: sampled-z (0.6999) > injection-off (0.5732) > mean-z (0.4458) ≈ shuffled-z (0.4394)。**Σ 让 prefix 变得 "not too prescriptive"**。

## 与原计划的 Acceptance 校核

- ✅ std/1-mean_cos(sampled) > 1.5× std/1-mean_cos(mean): **1.57x 满足**
- ⏳ Mean rank NOT regress > 5% vs B mean-z: **未在 11.B 评估**,计划放在 11.E (与 11.D 全 pipeline 对比)
- ✅ C (shuffled-z) < A,B: **0.4394 < 0.6999** ✓ (wrong user prefix 没有问题,但是这种条件相对 mean-z 几乎一样说明 single mean 本来就和 wrong user 类似——这是因为 single μ_u 和 random μ_other 都被 LLM 解释为 "某种 average style")
- ✅ D (injection-off) vs A,B: **diversity 介于 mean-z 和 sampled-z 之间**

## 工程实现经验

1. **Cholesky sampling**: `s_u = mu[None,:] + eps @ L.T`,其中 L = cholesky(Σ_u),eps ~ N(0, I_768)
2. **Per-pair sample unique to that pair**: 同一 user 在多个 pair 间共享同一组 K=10 s_u(避免 sample bias)
3. **TinyStyler prefix 是用 s_u 直接*2.0 作 K=8 prefix** (E30.34 已验证 K=8, α=2.0)
4. **BATCH=8 防止 OOM**: T5-large fp16 ~3GB + AnnaWegmann + projection 64GB ok

## 文件位置

- 引擎: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase11_b_gaussian_sampling.py` (~330 lines)
  - `phase11_b_smoke.py`
  - `phase11_b_gaussian_samples.jsonl` (~1200 records)
  - `phase11_b_diversity_report.json`
  - `phase11_b.log`
- 复用: phase11_a_user_gaussians_768d.npz (Cholesky),TinyStyler E30.34,AnnaWegmann cache

## 下一步

**Phase 11.C (立即开始)**: Conditional Latent Diffusion training
- 训练 DDPM denoiser ε_θ(z_t, t, c, s_u)
- 用 12000 phase10 sentence 的 Qwen layer 14 hidden states 作为 content
- 训练 30 epochs,~40 分钟 GPU
- 接受标准:training loss < 0.05 + style transfer 验证

**Phase 11.D + 11.E**: 端到端 pipeline + 4 conditions 评估
- C1 mean-z baseline vs C2 sampled-z (11.B) vs C3 diffusion-only vs C4 完整 pipeline
- 评估 rank-1 coverage + top-100 count

## 决策: GO (Phase 11.C 开始)

- ✅ Diversity gain 1.57x (> 1.5x): Σ 携带的协方差信息转化为生成多样性
- ✅ Sampled-z 是 4 个条件中最 diverse 的
- ✅ shuffled-z 和 injection-off sanity check 都通过
- **下一步**: 11.C 训练 diffusion model,验证 c 和 s_u 是否能解耦