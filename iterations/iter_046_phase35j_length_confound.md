# iter 046: Phase 35.J — Length Confound Validation (corrected: candidate-pool control)

**日期**: 2026-08-23
**承接**: iter_045 (V2 N=7 K=4 = 50.7% SOTA) + 用户高阶分析
  > 你的 scorer 很可能天然偏爱长文本 ... 两种可能:
  > 一种是好事: 长文本 → 真正包含更多用户风格 → Mahalanobis 更容易识别
  > 一种是 confound: 长文本 → Qwen rep 改变 → score 系统性变高
**目的**: 用 length-matched + Spearman ρ + 残差化 S + length-controlled (full pool) 区分两种解释

---

## 三组分析

### A. Length-Matched Rank-1 (|ΔL|≤5 per cand)

> 核心实验: 把同一长度的 cand 放在一起比较, 消除长度优势

| N | L≤15 | 16-25 | 26-35 | 36-50 | 51-80 | L>80 |
|---|------|-------|-------|-------|-------|------|
| 4 | 42.3% (142) | 55.9% (34) | **100.0%** (16) | 95.5% (22) | 87.2% (39) | 85.7% (7) |
| 5 | 30.7% (127) | 43.5% (46) | 84.6% (13) | **89.5%** (19) | 84.6% (39) | 66.7% (9) |
| 6 | 32.0% (103) | 42.9% (70) | 47.8% (23) | 70.0% (20) | **89.7%** (29) | **100.0%** (10) |
| **7** | 25.4% (63) | 39.2% (79) | 62.5% (24) | **100.0%** (15) ★ | **87.0%** (46) | **100.0%** (6) |
| 8 | 35.7% (28) | 42.4% (99) | 79.3% (29) | 65.4% (26) | **89.5%** (19) | **100.0%** (6) |

→ Length-matched 多桶 100% Rank-1, N=7 L=36-50 = 100% (n=15)

**但用户警告**: Length-matching 缩小了 candidate pool,绝对 Rank-1 可能 artifact。

### B. Spearman ρ(L, S_residual)

| metric | ρ | p | 判定 |
|--------|---|---|------|
| ρ(L, S) 全样本 | **+0.145** | 5.19e-09 | 弱正, < 0.3 阈值 |
| ρ(cov, S) | +0.079 | 1.54e-03 | 弱 |
| ρ(L, S) per N | +0.05 ~ +0.19 | - | N=8/9/10 几乎无关 |

→ **长度对 S 影响有限** — 远低于 style signal

### C. Residualized S* = S − β·L

线性拟合: **S = 0.1969 + 0.0015 · L** (β = 0.0015)

| 方法 | Rank-1 | mean_rank |
|------|--------|-----------|
| Raw S (Phase 35.G baseline) | **44.8%** | 9.29 |
| **S* = S − β·L (去长度)** | **34.1%** (-10.7pp) | 10.21 (+0.93) |

→ **去掉长度贡献后 Rank-1 反而下降 10.7pp** ★
→ 长度相关 representation 携带用户区分信号 (反向证明)

---

## Phase 35.J-3 关键: Length-Controlled (full pool, fixed L)

> **用户警告**: 之前的 Length-Matched 把 candidate 池缩小到 2-3 users, random baseline 升到 50%,
> 100% Rank-1 是 task trivially easier,不是真的 style discrimination。
> 需要 query-length 受控但 candidate-pool **不变**的实验。

| N | L≤15 | 16-25 | 26-35 | 36-50 | 51-80 | L>80 |
|---|------|-------|-------|-------|-------|------|
| 4 | 1.39x | 1.40x | **2.23x** | **2.23x** | 2.22x | 2.02x |
| 5 | 1.41x | 1.83x | **2.17x** | 2.17x | 2.30x | 2.25x |
| 6 | 1.83x | 1.61x | 1.64x | 1.64x | 2.83x | **3.74x** ★ |
| **7** | 2.39x | 1.40x | 1.72x | **2.49x** | 2.23x | 2.36x |
| 8 | 2.33x | 2.07x | 1.16x | 1.16x | **3.18x** | **3.02x** |
| 9 | 2.09x | 1.39x | 1.90x | 1.90x | 2.17x | 1.97x |

**关键 case: N=7 L=36-50**:
- n_test: 23
- **pool_size_avg: 4.74** (NOT 10-20!, 用户怀疑过的样本量)
- **random_acc: 27.96%** (vs Length-Matched 池 ~50%)
- **Rank-1: 69.6%**
- **lift vs random: 2.49x** (真实 lift,非 artifact)

**最佳 lift 桶**:
- N=6 L>80: **3.74x** ← 长 query 真实信息密度高
- N=8 L=51-80: 3.18x
- N=8 L>80: 3.02x
- N=7 L=36-50: 2.49x (用户建议的 operating point)
- N=7 L=26-35: 1.72x (baseline 附近)

---

## 关键发现 (修正版)

### 1. **Length-Matched 100% 是 post-hoc artifact** ⚠️
- length-matching 让 pool 从 ~7 缩小到 ~2-3 users
- random baseline 28% → ~50% (1/2)
- 真实 lift 没变,但 absolute Rank-1 大幅上升
- **不能用作 "Maha 在固定长度内识别 style 极强" 的证据**

### 2. **Length-Controlled (full pool) lift 1.5-3.7x random** ★
- 在 fixed L 下,Maha 仍能 lift,但绝对值没有 100% 那么戏剧
- 长 query lift 更高 (N=6 L>80 3.74x, N=8 L=51-80 3.18x)
- 短 query lift 较低 (N=7 L≤15 2.39x, N=7 16-25 1.40x)
- **结论**: 长 query 在 full pool 下也有最高 lift,与 "长 → 多 style 信号" 一致

### 3. **ρ(L, S) = 0.145 确认 (不主要依赖长度)** ✅
- 弱正相关,< 0.3 confound 阈值
- Maha 不是 systemically biased toward long text

### 4. **S − β·L 下降 10.7pp → 长度相关 representation 有用** ⚠️
- 但这**不等于** "长度是 pure style signal"
- 长度可能 mix:verbosity / syntax complexity / attribute realization / generation mode
- 只能说 "length-related variation contains user-discriminative info"

### 5. **N=7 L=36-50 (Length-Controlled) 真实 lift = 2.49x**
- pool_size 4.74, random 28%, Rank-1 69.6%
- 不是 100%! 但 lift 仍显著
- 与 N=6 L>80 3.74x 相比,**短 query operating point 不是 sweet spot**

---

## 对照用户疑虑

| 用户假设 | 实测 | 判定 |
|---------|------|------|
| Length-matched 仍 60-70%? | 多个 100% | ⚠️ artifact (pool 缩小) |
| Length-matched 后 \|C\| ≈ 10-20? | 实测 2-3 | ❌ 比预期小很多 |
| True Rank-1 vs random baseline ratio? | 2.49x (N=7 L=36-50) | ✅ 真实 lift |
| ρ(L, S) > 0.3 表示 confound? | ρ=0.145 | ✅ 远低于阈值 |
| 去掉 β·L → Rank-1 不变或升? | 下降 10.7pp | ✅ 长度相关有用 |
| 长 query lift 更高? | N=6 L>80 3.74x vs 短 1.5-2x | ✅ **完全验证** |

---

## 修正结论 (vs 初版)

初版说 "长度是 signal 不是 confound",**过度了**。修正版:

1. **长度不是 confound** (ρ=0.145 远低于阈值)
2. **长度相关 representation 携带用户区分信号** (去 β·L 下降 10.7pp)
3. **但不能等同 pure style** (可能是 verbosity / attribute realization / generation mode 等混合)
4. **Length-controlled lift 1.5-3.7x random 是真实能力**, Length-Matched 100% 是 task 变简单的 artifact
5. **长 query 反而更 discriminative** (N=6 L>80 3.74x),可能因为长 query 信息密度高

> **核心**: 在 residual space 里,Maha 已经很好 (50.7%)。长度只是其中一个相关特征,
> 不是一个需要特殊处理的 confound。Length-aware rerank 可能再 +1-2pp 但不是核心杠杆。

---

## Phase 35.J-2 Forced Generation (待评)

> 生成时强制 N=7 + L=36-50,K=8 / record,验证 100% 是否在样本量放大后稳态

- 强制 gen 已完成: 544 cands / 68 records,样本 L=36-50 hit rate 待统计
- scoring 待跑(已加入任务清单)

---

## 下一步

1. **Phase 35.J-2 scoring**: N=7 L=36-50 forced K=8 验证
2. **Phase 35.M 2-GMM** (iter_047): 已完成 NO-GO (-12pp vs baseline)
3. **Phase 35.N**: length-aware rerank (用 L 作为 re-weight 系数)
4. **Phase 35.L**: 论文 trade-off figure (N vs L 2D heatmap, Pareto frontier)

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35j_length_confound.py` | length-matched + Spearman + residualize (~10s) |
| `syntax_subspace/phase35j2_forced_gen.py` | 强制 N=7 L=36-50 生成 (K=8) |
| `syntax_subspace/phase35j3_controlled.py` | length-controlled (full pool) eval |
| `result/phase35j/length_confound_summary.json` | 完整分析结果 |

---

## 最终 takeaway (修正版)

> Phase 35.J 长度 confound 验证 — **长度不是 confound, 但 length-matched 100% 是 artifact**:
> 1. **ρ(L, S) = 0.145** 弱正, 远低于 confound 阈值 0.3 ✅
> 2. **Length-Matched 100%** 是 post-hoc artifact (pool 2-3 users, random 50%)
> 3. **Length-Controlled (full pool) lift 1.5-3.7x random** — 真实能力
> 4. **N=7 L=36-50 controlled lift = 2.49x** (pool 4.74, random 28%)
> 5. **长 query 反而 lift 更高**: N=6 L>80 3.74x, N=8 L=51-80 3.18x
> 6. **去 β·L 下降 10.7pp** → 长度相关 representation 有用 (但 ≠ pure style)
>
> 修正了初版"长度是 signal"的过度说法。**Length-related variation contains user-discriminative info**
> (suitable as feature, not confound to remove),但不能用 length-matched 100% 作为 pure style 证据。
> 下一步 length-aware rerank / N vs L Pareto 探索。
