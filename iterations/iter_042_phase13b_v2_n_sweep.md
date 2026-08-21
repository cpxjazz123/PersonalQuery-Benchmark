# Phase 13.B-v2: N sweep on Gaussian σ fitting convergence — **N=30** is sweet spot

**Date**: 2026-08-21
**Question**: 多少 sentences/user 才能让 σ 估计不再嘈杂?
**Answer**: **N=30** (σ 95% plateau); **N=50** 完全收敛 (σ + μ 都稳定)

## 实验动机

Phase 15.D (2026-08-21) 验证 `||σ_residual|| ≈ 175 >> ||μ_residual|| ≈ 24` 让 ρ × σ × ε steer 爆炸。
用户问: **10 样本太少导致 σ 估计不可靠** —— 是否用更大 N 能让 ρ noise 范式 work?

**Phase 13.B-v2 目标**: 找 σ 估计收敛的最小 N。

## 实施

### 数据
- 6M raw Amazon reviews (Baby_Products_2023)
- 876 phase10 users 全部 matched in raw data
- **198 users 有 ≥50 reviews** (selected for N sweep, 因为需要 N=100)
- Sentence 切分: 5-60 words filter, 与 phase13_a 协议一致

### Sweep 范围
| N | 用途 |
|---|------|
| 10 | baseline (Phase 13.B v1 default) |
| 20 | small growth |
| 30 | moderate (≥30 review users) |
| 50 | substantial |
| 100 | upper limit (实际 183 users 有 ≥100 sents) |

### 工程
- Qwen 2-7B 抽 layer 14 hidden (mean-pool per sentence)
- 198 users × up to 100 sents = 19,349 forward passes
- Qwen load ~4 min, hidden extraction ~1 min (batch=32), fit <1 min
- **两次 fit**:
  1. Raw hidden (不减 neutral) — 看 σ hidden 的收敛
  2. Residual = hidden - neutral (layer 14 mean) — 与 Phase 15.D comparable

## 结果

### Raw hidden fit (σ/μ in raw hidden space, ~700 norm)

| N | n_users | mean_μ | mean_σ | mean_ratio | med_ratio |
|---|---------|--------|--------|------------|-----------|
| 10 | 198 | 709.8 | 282.3 | 0.403 | 0.394 |
| 20 | 198 | 701.1 | 297.7 | 0.431 | 0.427 |
| **30** | **198** | 704.2 | 304.6 | **0.440** | **0.434** |
| 50 | 198 | 705.4 | 310.0 | 0.448 | 0.446 |
| 100 | 183 | 703.7 | 313.1 | 0.453 | 0.455 |

→ **plateau at N=30** (σ change +1.7% from N=20, <5% threshold)

### Residual fit (σ/μ in residual space, comparable to Phase 15.D)

| N | n_users | mean_μ_resid | mean_σ_resid | mean_ratio | med_ratio | std_ratio |
|---|---------|--------------|--------------|------------|-----------|-----------|
| 10 | 198 | **135.3** | **282.3** | **4.21** | 2.60 | 4.85 |
| 20 | 198 | 121.7 | 297.7 | 6.00 | 2.83 | 7.70 |
| 30 | 198 | 113.6 | 304.6 | 6.37 | 3.28 | 7.95 |
| **50** | **196** | 110.2 | 309.9 | **6.40** | **3.29** | **7.94** |
| 100 | 183 | 103.6 | 309.6 | 6.19 | 3.65 | 7.00 |

→ **plateau at N=50** (mean_ratio change <5% from N=30)

## 关键发现

### 1. N=10 时 σ 确实嘈杂
- σ_N10 = 282 vs σ_N30 = 304 (低 7%)
- μ_N10 = 135 vs μ_N50 = 110 (高估 23%)
- mean_ratio: 4.21 (N=10) → 6.37 (N=30) → 6.40 (N=50)

**结论**: Phase 13.B v1 用 N=10, σ 估计偏低 ~7%, μ 偏高 ~23%, ratio 偏低 ~35%。

### 2. 收敛点
- **σ 在 N=30 后 95% 收敛** (304 → 310 仅 +2%)
- **μ 在 N=50 才 95% 收敛** (110 → 104 仍有 -5%)
- **N=30 是 sweet spot** (性价比最高)

### 3. Phase 15.D ρ noise 范式即使 N=100 仍 NO-GO
- N=100: σ_residual = 310, μ_residual = 104, **median ratio = 3.65** (mean 6.19)
- 即使用 100 句 sample, σ_residual norm (310) 仍 ≈ 4-5 × hidden state norm (~60-80)
- **ρ × σ × ε steer norm ≈ 2500-3500** (远超 hidden) → 仍会 overwrite
- 不是 N=10 太少的问题, **是 sample-level std 本质就 > mean** (per-dim std 是 hidden 噪声)

### 4. 分布右偏 (有 outlier user σ 极大)
- mean_ratio (6.4) ≈ 2× median_ratio (3.3) at N=30
- std_ratio = 7.95 at N=30
- 一些 user 的 per-dim std 远超 mean (Phase 15.D 报告 range 70-466 vs μ range 9-302)

## 回答用户问题

**Q: 多少样本数以上拟合的 σ 才不嘈杂?**

**A: N=30**。具体含义:
- 10 句 → σ 偏低 ~7%, μ 偏高 ~23% (嘈杂, Phase 13.B v1 的状态)
- **30 句 → σ 95% 收敛** (推荐 minimum)
- 50 句 → σ + μ 都 95% 收敛 (完全 plateau)
- 100 句 → 与 50 几乎一致 (无边际收益)

## Phase 15.D ρ-noise NO-GO 的根本原因

**不是样本量不足**,而是 **per-dim σ 范式**本身有问题:
- 即使 N=100, σ_residual norm (310) 仍 >> μ_residual (104), ratio 3.65
- 真要 work 需要: (a) ball-constrained ε, (b) direction-only ε, 或 (c) shrink σ
- 但这些不是 N sweep 能解决的 — 需要换范式

## 与 Phase 13.B v1 对比

| Metric | Phase 13.B v1 (N=10) | Phase 13.B v2 N=30 (current) | Phase 13.B v2 N=50 |
|--------|---------------------|------------------------------|---------------------|
| Users | 298 | 198 | 196 |
| mean_μ_residual | 24 | 113.6 | 110.2 |
| mean_σ_residual | 175 | 304.6 | 309.9 |
| mean_ratio | 7.3 | 6.37 | 6.40 |

**注**: v1 vs v2 数值差异因为:
- v1: 298 users (filter 后的 vades_proto 15 句限制)
- v2: 198 users (raw Amazon ≥50 reviews)
- v1 μ 更小 (24) 因为 v1 sample 更纯 (filter 过)
- v2 σ 更大 (304) 因为 raw reviews 含更多 noise

**核心结论**: **Phase 15.D 的 NO-GO 在更大 N 下不会改变** — 仍 NO-GO。问题在范式本身,不在样本量。

## 文件位置

### Scripts (in git)
- `result/phase15/scripts/phase13_b_v2_n_sweep.py` — Qwen extract + raw hidden fit
- `result/phase15/scripts/phase13_b_v2_fit_only.py` — 仅 fit (load cached hiddens)
- `result/phase15/scripts/phase13_b_v2_residual_fit.py` — residual fit (减 neutral)

### Results
- `phase13_b_v2_user_hiddens_n100.npz` — 198 users × up to 100 sents, hidden states (cached)
- `phase13_b_v2_ratio_per_user.csv` — raw hidden ratio per (user, N), 990 rows
- `phase13_b_v2_summary.json` — raw hidden aggregate + plateau detection
- `phase13_b_v2_residual_ratio_per_user.csv` — residual ratio per (user, N), 990 rows
- `phase13_b_v2_residual_summary.json` — residual aggregate

### Logs
- `result/phase15/logs/phase13_b_v2_sweep.log` — Qwen load + 19800 forward + raw fit
- `result/phase15/logs/phase13_b_v2_residual_fit.log` — residual fit (CPU only)

## 下一步建议

1. **不重跑 Phase 13.B with N=30**: 当前 mean-only (μ × α) 已经是 sweet spot, ρ 范式本身 NO-GO
2. **如果未来想用 Gaussian noise**: 改范式为 ball-constrained 或 direction-only, 而不是增 N
3. **Memory 更新**: 记录 N=30 是 minimum reliable σ estimate

## Bug 修复记录

1. **np.stack on ragged hiddens**: 第一版用 `np.stack` 拼 [198, N, H], 但 user sents 数不同 (min 34 max 3209),崩了。
   修复: 改用 `np.empty(..., dtype=object)` 每个 user 一个变长 array。
2. **Neutral npz key 错误**: 第一次跑 residual fit 用 `n_d["neutral_hidden"]` 但实际字段是 `vecs`, 字段 shape [N, 28, 3584] (per-sentence neutral mean-pool layer 14 hidden)。
   修复: 取 layer 14 mean → `[3584]`。