# Phase 15.D: α × ρ Sweep on StyleVector Injection — NO LIFT (saturated Gaussians)

**Date**: 2026-08-21
**Status**: NO-GO — 7 conds 全部没有显著 style lift;**std_diag 远大于 mu** 让 ρ > 0 把 hidden 完全 overwrite
**Engineering**: ~5 min sweep + 35s eval (Qwen load 4 min reused)
**Sweep configs**: D_off + α∈{0.3, 0.5} × ρ∈{0.0, 0.3, 0.5} = 7 conds × 30 pairs × K=8 = 1680 records

## 实验动机

Phase 14/15 v1 (mean-pool) 已经证明 StyleVector injection 有 **+0.098 style margin lift**
(Phase 13.F 768d best-K margin),但具体 **(α, ρ) 哪个组合生成最 natural** 没扫过。
Phase 11.G latent diffusion 用 ρ 让 steer 多样化,但这里 ρ 直接 scale std_diag (per-dim std)
而非一个统一的 Gaussian 球面 std。

**Phase 15.D**:在 v1 mean-pool user_gaussians 上扫 (α, ρ),找出:
- 强 style lift (vs D_off margin diff > +0.02)
- 语义保持 (|sem_diff| < 0.02)
- 文本 natural (no reviewer commentary / no 乱码)

## 实施

### Sweep 范围

| Cond | α | ρ | 含义 |
|------|---|---|------|
| D_off | 0.0 | 0.0 | baseline, no hook |
| A_a0.3_r0.0 | 0.3 | 0.0 | 只用 mu,温和 |
| A_a0.3_r0.3 | 0.3 | 0.3 | + ε × std × 0.3 |
| A_a0.3_r0.5 | 0.3 | 0.5 | + ε × std × 0.5 |
| A_a0.5_r0.0 | 0.5 | 0.0 | 较强 mu |
| A_a0.5_r0.3 | 0.5 | 0.3 | 较强 mu + ε × std × 0.3 |
| A_a0.5_r0.5 | 0.5 | 0.5 | 较强 mu + ε × std × 0.5 |

注: α=0.7 因 sweep 初期判定 too strong 被砍掉。

### Sweep 工程

- 单 Qwen load (~4 min),每 config 复用
- hook 装在 `model.model.layers[14]`,target position = last token
- steer vec: `s = mu + ρ · std_diag · ε`,然后 mean-zero per-position 正则
  (实际不影响,因为 ε mean ≈ 0)
- 生成: `model.generate(K repeats of prompt)` 一次 forward, K=8
- 后处理: hard-copy attribute span append (Phase 10.10 协议)

## 关键发现 — std_diag 远大于 mu

```
layer 14 (298 user mean over all):
  ||mu||     = 69.41  (range 9-302)
  ||std_diag|| = 172.50  (range 70-466)
```

**`||std|| ≈ 2.5 × ||mu||`**,sample size 10 拟合的 per-dim std 远大于 per-dim mean
→ Gaussian 噪声主导而非 user style signal。

实际 steer vector norm (alpha=0.5, rho=0.5):
```
||alpha * (mu + rho * std * eps)||
= ||0.5 * (mu + 0.5 * std * eps)||
≈ 0.5 * (||mu|| + 0.5 * ||std|| * ||eps||)
= 0.5 * (24 + 0.5 * 175 * 60)  (||eps|| ~ sqrt(3584) ~ 60)
= 0.5 * (24 + 5250)
= 2637  ← 完全 overwrite hidden state (norm 50-80)
```

**ρ > 0 时的实际 steer norm 远超预期**,等于把 hidden state 推到 noise direction,
生成退化 (α=0.5 ρ=0.5 直接乱码)。

## Eval 结果 — 7 conds

### 768d Style Margin

| Cond | n_cand | mean_margin | BoK_margin | sem_sim |
|------|--------|-------------|------------|---------|
| D_off | 240 | 0.0092 | 0.0152 | 0.7562 |
| A_a0.3_r0.0 | 240 | 0.0104 | 0.0171 | 0.7585 |
| A_a0.3_r0.3 | 240 | 0.0103 | 0.0161 | 0.7619 |
| A_a0.3_r0.5 | 240 | 0.0102 | 0.0158 | 0.7552 |
| A_a0.5_r0.0 | 240 | 0.0104 | 0.0168 | 0.7603 |
| A_a0.5_r0.3 | 240 | 0.0102 | 0.0151 | 0.7532 |
| A_a0.5_r0.5 | 240 | 0.0099 | 0.0161 | 0.7505 |

### vs D_off (paired bootstrap, 2000 resamples)

| Cond | bok_diff | CI | excludes 0 | sem_diff | CI | excludes 0 |
|------|----------|----|-----------|----------|----|-----------|
| A_a0.3_r0.0 | +0.0020 | [-0.0002, +0.0048] | ✗ | +0.0023 | [-0.0104, +0.0158] | ✗ |
| A_a0.3_r0.3 | +0.0009 | [-0.0021, +0.0043] | ✗ | +0.0057 | [-0.0037, +0.0166] | ✗ |
| A_a0.3_r0.5 | +0.0006 | [-0.0018, +0.0028] | ✗ | -0.0010 | [-0.0115, +0.0092] | ✗ |
| A_a0.5_r0.0 | +0.0016 | [-0.0007, +0.0043] | ✗ | +0.0041 | [-0.0096, +0.0190] | ✗ |
| A_a0.5_r0.3 | -0.0000 | [-0.0024, +0.0018] | ✗ | -0.0030 | [-0.0178, +0.0113] | ✗ |
| A_a0.5_r0.5 | +0.0010 | [-0.0028, +0.0045] | ✗ | -0.0057 | [-0.0192, +0.0085] | ✗ |

**所有 6 个 cond 的 BoK diff CI 都包含 0 → 没有任何 cond 显著 lift**。

## 与 Phase 13.D v1 对比

| Metric | Phase 13.D v1 (mean-pool) | Phase 15.D (best A) | Δ |
|--------|---------------------------|---------------------|---|
| A BoK_margin | 0.323 | 0.017 | -19x |
| D_off BoK_margin | 0.225 | 0.015 | -15x |
| A vs D lift | +0.098 CI excludes 0 ✓ | +0.002 CI includes 0 ✗ | -49x |

**v1 mean-pool sweep 完全胜出**,Phase 15.D v1 BoK = 0.323 而 Phase 15.D 任意 cond 都 ≤ 0.017。

## Sample 对比 (Milliard user)

| Cond | Sample q_final_post |
|------|---------------------|
| D_off | `Milliard grey foam mat with dimensions 40 inches wide by 7 inches high", with 2.07 pounds, 40"W x 7"H.` |
| A_a0.3_r0.0 | `Milliard grey foam padding 40 inch wide by 7 inch high", with 2.07 pounds, 40"W x 7"H.` |
| A_a0.3_r0.3 | `Find a Milliard grey foam mat with 40 inch width and 7 inch height, with 2.07 pounds, 40"W x 7"H.` |
| A_a0.3_r0.5 | `Milliard grey foam mat with dimensions 40 inches wide by 7 inches high", with 2.07 pounds, 40"W x 7"H.` |
| A_a0.5_r0.0 | `Milliard grey foam mat 40 inches wide by 7 inches high", with 2.07 pounds, 40"W x 7"H.` |
| A_a0.5_r0.3 | `Milliard grey foam mat 40 inch wide", with 2.07 pounds, 40"W x 7"H.` |
| A_a0.5_r0.5 | `Milliard grey foam 40x7 inch mattress", with 2.07 pounds, 40"W x 7"H.` |

观察:
- **α=0.3 ρ=0.3** 偶尔出现 "Find a ..." 前缀,有点 user-style
- **α=0.5 ρ=0.5** 有时把 "padding" 换成 "mattress" (同义替换,但失去 user 特征)
- 其他 cond 几乎与 D_off 一样 → steer 没产生 style signal

更糟样本 (A_a0.5_r0.5 second):
```
Bre breath, 7 oz, 2000 and 1200, 15 inches
11 34
2012
2035, with BreathableBaby, 7 ounces, 33"L x 15"W, Bassinet
```
**完全乱码** — steer 把 hidden state 完全推到 noise direction,生成崩溃。

## 结论 — NO-GO on ρ-std scaling

| Insight | Implication |
|---------|-------------|
| `||std|| ≈ 2.5 × ||mu||` | Per-dim std 远大于 per-dim mean, ρ 不能简单 scale std_diag |
| ρ > 0 steer norm 爆炸 | ρ 应 ≤ 0.05 或重新归一化 (std → σ/μ ratio) |
| 所有 cond 无显著 lift | (α, ρ) 不是问题;问题在 std scaling 范式本身 |
| α=0.5 ρ=0.5 乱码 | Steer 完全 overwrite hidden state,失去 model semantic |

**主路线仍为 Phase 13.D/14/15 v1 mean-pool StyleVector + 768d style rerank**:
- StyleVector injection strength α=1.0 (Phase 13.D v1 baseline)
- 不引入 ρ 噪声 (Phase 13.D v1 直接用 mu,无 std scaling)
- 配合 768d AnnaWegmann style rerank (Phase 14 hybrid pipeline)

## 路线调整

- ❌ **不再用 ρ > 0 std scaling 范式** — std_diag 主导会让 steer noise > signal
- ✅ 如果未来想引入多样性,应改用 **ball-constrained sampling** (e.g. ε ∈ N(0, I), ||ε|| ≤ 1)
  或 **direction-only** (μ normalized, ε orthogonal to μ)
- ✅ Phase 13.D/14/15 v1 mean-pool 路线已 commit + push (2d926c1)

## 文件位置

### Scripts
- `result/phase15/scripts/phase15_d_alpha_rho_sweep.py` — sweep engine
- `result/phase15/scripts/phase15_d_eval_only.py` — standalone eval (replaces buggy evaluate_all)

### Results (in /home/wlia0047/hj82_scratch2/wenyu/vades_prototype/)
- `phase15_d_alpha_rho_sweep.jsonl` — 1680 records (7 conds × 30 pairs × K=8)
- `phase15_d_samples.json` — 5 user samples per cond (35 total)
- `phase15_d_eval.json` — final eval summary with bootstrap CI
- `phase15_d.log` — full sweep log (Qwen load 4 min + sweep 5 min)
- `phase15_d_eval2.log` — eval log (35s)

### User Gaussians (input)
- `phase13_b_user_gaussians_qwen.npz` — v1 mean-pool, 298 users × 28 layers × 3584 dim
  - `mu`: mean residual per user per layer
  - `std_diag`: per-dim std from 10 sentences (过大,导致 ρ > 0 steer 爆炸)

## Bug 修复记录

1. **`NameError: torch` in evaluate_all**: sweep.py 调用 `evaluate_all()` 但函数内 `with torch.no_grad()` 没 import torch。
   修复: `phase15_d_eval_only.py` 独立脚本,首部 import torch。
2. **n_pairs=0 for all A_* conds**: uid 不在 876 user set 时 margin 被 skip,但 `by_key` 用全局 index 错位。
   修复: 改用 `margin_by_key = dict[(uid, cond)]` 直接 key,不用全局 index。
3. **NameError 影响 sweep 输出**: 1680 records 已成功生成并写入 jsonl,只是 eval 没跑。独立 eval 复用 jsonl。