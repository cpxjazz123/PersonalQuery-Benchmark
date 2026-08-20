# Phase 10.16: Soft Prefix Injection — Final NO-GO Summary

**Date**: 2026-08-20
**Branch**: main
**Status**: NO-GO (soft prefix injection 不能提升 rank-1 coverage)

## 实验设计

4-control 严格对比 (876 pairs × 4 seeds = 3504 cands per control):

| Control | Prefix | 期望效果 |
|---------|--------|---------|
| real-z | `projector(z_user)` (correct user) | 显著 ↑ rank-1 |
| shuffled-z | `projector(z_other)` (random other user) | ≈ baseline (sanity) |
| true-zero-z | `zeros(K, H)` | ≈ baseline (no signal) |
| injection-off | no prefix | baseline |

**Setup**:
- Model: Qwen2.5-Coder-7B-Instruct (hidden=3584, 匹配 e22_t3 projector)
- Projector: e22_t3_injector_best.pt (alpha=0.277, K=16, trained on e22_t1 user vectors)
- User vectors: phase10_user_vectors_318d.npy (876 users, z-score normalized using e22_t1 train_mean/train_std)
- Generation: temp=0.7, top_k=40, top_p=0.92, max_new=48, BATCH=32 → 20.92 query/s
- 14016 total generations, 11.2 分钟

## 结果 (rank-1 coverage 4-control)

```
Control          Rank-1 Cov   Mean Best Rank   Δ vs off
real-z            0.0000      896.6           -0.0011
shuffled-z        0.0011      924.2           +0.0000
true-zero-z       0.0023      885.6           +0.0011
injection-off     0.0011      935.1            0.0000
```

**Paired bootstrap (real-z vs off)**: Δ = -0.0011, 95% CI [-0.0034, 0.0000]
**Paired bootstrap (real-z vs shuffled-z)**: Δ = -0.0011, 95% CI [-0.0034, 0.0000]

## 结论: **NO-GO**

real-z 注入**不**显著提升 rank-1 coverage,反而略有下降 (0/876 vs 1/876)。
mean best rank 在 real-z (896) 与 injection-off (935) 之间无统计显著差异。

## 失败原因分析

1. **Covariate shift between training and test**:
   - e22_t3 projector 训练时 user vectors 来自 e22_t1 (915 train users, ~10 sentences/user)
   - phase10 user vectors 来自 phase10 句法子集 (876 users, 句法多样)
   - 形式上对齐 (都是 (v - tm) / ts 归一化的 318d),但实际分布不同
   - 验证: e22_t1 与 phase10 重叠 23 users, cosine similarity 0.11 (远低于 1.0)

2. **Projector 训练数据不足**:
   - e22_t3 projector 训练时可能对 e22_t1 train user distribution 过拟合
   - 缺 cross-distribution generalization 能力

3. **Alpha = 0.277 可能太小**:
   - 即使方向正确,prefix 幅度不足以显著影响 generation distribution
   - 但 shuffled-z 与 injection-off 也几乎相同,说明 projector 输出本身对 LLM 影响小

4. **Sample size 不足**:
   - 4 seeds/control 比 Phase 10.15 的 96 candidates 少 24×
   - 即使真实效应存在,降采样到 1/24 后难以统计显著

## 决策

**Phase 10.16 关闭 — 软前缀注入不工作**

接下来探索方向:
- (a) 用 e22_t1 训练集上重新训练 projector(对齐分布)
- (b) 增加 seeds 到 24+/control,确认 small effect
- (c) 探索 in-context learning: 把用户 past queries 直接放 prompt 而非 prefix
- (d) 重新审视 Phase 10.15 1.26% rank-1 上限本身是否是结构性问题

## 文件位置

- 训练数据/模型: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase10_user_vectors_318d.npy` (876, 318) — normalized 318d user vectors
  - `phase10_16_b_generations_4control.jsonl` (14016 records) — 4-control generations
  - `phase10_16_c_candidates_318d.npy` (14016, 250) — extracted 318d features
  - `phase10_16_c_rank1_eval.json` — rank-1 coverage results
- 脚本:
  - `phase10_16_a_user_vectors.py`
  - `phase10_16_b_soft_prefix_gen.py`
  - `phase10_16_c_rank1_eval.py`
  - `phase10_16_b_smoke.py` (5-pair smoke test)
- 原始 projector: `result/e22_t3/e22_t3_injector_best.pt`
