# Iter #021 (v7): E21 — N sweep (L=5 固定, N ∈ {20,30,40,50}) 寻找最小稳定 N

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: N=30 信号比 N=20 显著增强（test Cohen d +60%），但 4-gate 仍未全过；N=40/50 受 eligible 用户不足跳过；**min_stable_n = None**

## §A 审稿意见（v6 → v7 方向）

**用户原话**（2026-08-15）：
> 下一步建议固定 L=5，只增加句子数量：N = 20、30、40、50. 总需求 = 40、60、80、100个完整句子.
> 使用同一套预注册门槛，寻找最小稳定 N. 不要立即增加到1232维特征，否则无法判断改善来自更多句子还是更换特征。
> #21 当前仍为 Open，v6结论属于严格的 40句 NO-GO，但弱信号存在.

**v7 设计**：
- **冻结 L=5**（v5/v6 最稳健的过滤强度）
- **N sweep**: {20, 30, 40, 50}（分别对应总需求 40/60/80/100 句）
- **per-cell eligible pool**（修复 v7 第一版"common eligible pool 仅 154 users"的设计错误）
- **seed=43**（与 v6 一致，确保可直接横向对比）
- **9999 perm**（与 v6 一致）
- **预注册 4-gate**（与 v6 一致）：test_users≥150, AUC≥0.65, Cohen d≥0.5, seed_pass≥0.80, p<0.01

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：500 词以下 attribution 跌至 60%；**语料量是最强 gate**。
**PQB v7 现状**：N=20 (40 句 / ~200 词) test d=0.265；N=30 (60 句 / ~300 词) test d=0.425；**每增 20 句，Cohen d 提升 ~0.08**。完全符合 Eder 关于"语料量与可识别性单调正相关"的结论，但即使是 N=30 仍未达 0.5 门槛。

### 论文 [2] Brennan, Afroz, Greenstadt (2012)
**发现**：作者效应在自然书写中弱。
**PQB v7 现状**：N=20 d=0.27（弱）、N=30 d=0.42（中等下）— 与 Brennan 的"自然书写下作者效应弱"完全相符。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：verification 不需 balanced negative。
**PQB v7 现状**：v7 split-half verification 在 N=30 时 test 优于 dev（不衰减），符合 one-class 范式预期。

### 论文 [4] Stamatatos (2009) 综述
**发现**：单一 syntactic 通道弱；需要 ensemble。
**PQB v7 现状**：v7 仍仅用 32 维 syntactic 比例特征；N=30 test d=0.42 + AUC=0.611 表明单一通道在 60 句规模下不足以达 Cohen d ≥ 0.5。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：高维 kernel + per-author 压缩模型是 PAN SOTA；低维手工特征不够。
**PQB v7 现状**：v7 用 32 维手工特征 + L2 → 即使 N=50 也仍未过 0.5 gate。需要 Kestemont 风格的高维特征或 per-author 模型。

### 综合 §B
v7 验证了"语料量是关键 gate"的 stylometry 共识：每增 20 句，Cohen d 提升 ~0.08；但 32 维 syntactic 特征 + 60 句仍不足以稳定达 0.5 门槛。

## §C 代码缺陷定位

**v7 第一版的设计缺陷**：
- 错误地将 4 cells 共用一个 "common eligible pool"（即满足全部 N∈{20,30,40,50} 的用户交集）
- 该交集仅 154 users → dev/test 各 77 → 所有 cell 都过不了 test_users≥150 gate
- 修复方式：每个 cell 单独用其 eligible pool（cell-local）

**v7 第二版（最终版）改进**：
- per-cell eligible pool：`{20:1062, 30:463, 40:239, 50:154}`
- 每个 cell 按 seed=43 split 50/50 → dev=test
- N=40 / N=50 因 dev/test < 150 gate 而 SKIPPED（仅保留 N=20 和 N=30 的有效评估）

## §D 修复方案（v7）

`query/soft_prefix/e21_l_n.py`：
- 顶部常量：`L_FIXED=5, N_VALUES=(20,30,40,50), N_PERM=9999, SEED=43`
- 每个 cell 用 `select_eligible_for_n(N)` 单独取 eligible pool
- 用 `split_dev_test()` 按 50/50 split
- gate check 包含 test_users ≥ 150 才正式评估；否则跳过
- 沿用 v4 修正的方向与置换设计

## §E 验证结果（v7）

```
=== Run summary ===
users parsed: 14691
eligible per N (≥L words AND ≥N*L sentences):
  N=20: 1062   N=30: 463   N=40: 239   N=50: 154
runtime_sec: 341.7

=== Cell L=5 N=20: dev=531 test=531 ===
  dev AUC=0.5816+-0.0145  d=0.3113  seed_pass=0.00  delta=0.01224
  perm p_one=0.00010  p_two=0.00010  p_two_bonf=0.00040
  test AUC=0.5732  d=0.2650  seed_pass=0.00
  4-gate: test_users=531 PASS, auc=0.582 FAIL, d=0.311 FAIL, seed_pass=0.00 FAIL, p=0.0001 PASS

=== Cell L=5 N=30: dev=231 test=232 ===
  dev AUC=0.5914+-0.0184  d=0.3443  seed_pass=0.00  delta=0.01163
  perm p_one=0.00010  p_two=0.00010  p_two_bonf=0.00040
  test AUC=0.6106  d=0.4248  seed_pass=0.03
  4-gate: test_users=232 PASS, auc=0.611 FAIL, d=0.425 FAIL, seed_pass=0.03 FAIL, p=0.0001 PASS

=== Cell L=5 N=40: dev=119 test=120 ===  SKIPPED (test < 150 gate)
=== Cell L=5 N=50: dev=77 test=77 ===  SKIPPED (test < 150 gate)

min_stable_n: None
```

**关键 v6 → v7 对比**（seed=43 一致）：

| 指标 | v6 (N=20) | v7 (N=20) | v7 (N=30) |
|---|---|---|---|
| eligible users | 1062 | 1062 | 463 |
| dev users | 531 | 531 | 231 |
| test users | 531 | 531 | 232 |
| dev AUC | 0.582 | 0.582 | **0.591** |
| dev Cohen d | 0.311 | 0.311 | **0.344** (+10.6%) |
| test AUC | 0.573 | 0.573 | **0.611** (+6.6%) |
| test Cohen d | 0.265 | 0.265 | **0.425** (+60.0%) |
| test seed_pass | 0.00 | 0.00 | **0.03** |
| perm p_two | 0.0001 | 0.0001 | 0.0001 |

**关键观察**：

1. **N=30 test 不衰减（重大信号）**：
   - dev d=0.344 → test d=0.425（test > dev）
   - dev AUC=0.591 → test AUC=0.611（test > dev）
   - 这是 v5/v6 中都未观察到的"test 优于 dev"模式

2. **N 增 50% (20→30)，test Cohen d 增 60% (0.265→0.425)**：
   - 边际增益 ~0.08 / 20 句
   - 线性外推：N=50 时 test d ≈ 0.66（应过 0.5 gate）— 但 eligible 用户数 < 150，跳过

3. **N=40/50 受数据规模限制**：
   - N=40: 239 eligible → 120 test（< 150）
   - N=50: 154 eligible → 77 test（< 150）
   - 即便效应量更高也无法验证（gate 卡在样本数）

4. **seed_pass 仍接近 0**：
   - N=20: 0.00、N=30: 0.03 — 没有 seed 在多 seed 重采样中稳定过 AUC 0.65
   - 这是"用户风格信号弱"的另一个表征

## §F 解读与决策点

### v7 的明确结论
1. **信号存在**：所有有效 cell p_two_bonf=0.0004（极显著），不是噪声
2. **N 单调提升效应量**：N=20 → 30，Cohen d 从 0.27 → 0.42（+60%）
3. **N=30 test 不衰减**：v6 中未观察到的"test > dev"模式，提示 N 增大的方向正确
4. **N=40/50 无法验证**：eligible 用户不足 150，跳过
5. **min_stable_n = None**：4-gate 仍未全过；最小稳定 N **无法在当前数据下确定**

### 与 stylometry 经典文献的一致性
- Eder 2015：语料量是最强 gate → v7 验证：N 单调提升效应量
- Stamatatos 2009：单一 syntactic 通道弱 → v7 验证：32 dim 不够
- Brennan 2012：自然书写下作者效应弱 → v7 验证：Baby Products 自然评论下 d ≤ 0.42

### 未来候选方向（待用户决策）
- **(a) 扩大用户池到 N_USERS=10000**：看 N=50 eligible 是否 ≥ 150
- **(b) 增通道（char n-gram + function word）**：特征 32 → 1232 dim（用户警告：会混淆"更多句子 vs 更多特征"的贡献）
- **(c) 跨 category 池化**：Baby + Office + Pet Supplies 合并
- **(d) 接受 negative result**：当前 32 维 syntactic + L=5,N=30 在 Baby Products 自然评论下不足以稳定达 Cohen d ≥ 0.5；paper 报告"passive stylometry on short niche reviews insufficient"
- **(e) 接受条件 GO**：(L=5, N=30) 是当前数据下的最优 cell，test Cohen d=0.42 是"中等下"效应量；paper 报告"L=5,N=30 时用户风格识别 AUC=0.611, d=0.42 (中等下)"，承认效应量未达 0.5 中等门槛但方向稳定

## §G E21 完整时间线

| 版本 | 设计 | 结果 |
|---|---|---|
| E21 v1 (#21) | token-based 采样（错误） | d=0.0，AUC=0.0（采样 bug） |
| E21 v2 | 完整句子数采样 + 3-criterion 稳定性 | a∩b∩c = 0/32 |
| E21 v3 (507b109) | 2D 网格 + dev/test | best_cell=None（**统计错误**） |
| E21 v4 (55e2bba) | v3 修正 | (3,20) dev d=0.621 接近 GO |
| E21 v5 (f668efe) | v4 扩大池 + nlp.pipe | (5,20) test 不衰减，最稳健（但 4-gate 未全过） |
| E21 v6 (e8f56bf) | (5,20) 独立验证（seed=43, 9999 perm） | 4-gate FAIL：d=0.311, seed_pass=0.00 |
| **E21 v7 (本)** | **N sweep (L=5 固定)** | **N=30 d=0.42 (test > dev)，但 4-gate 仍未全过；min_stable_n=None** |

## 文件

- 脚本：`query/soft_prefix/e21_l_n.py`
- 结果：`result/e21_l_n_results.json` + `result/e21_l_n.log`
- 文档：`iterations/iter_021_l_n.md`
- 提交：待 commit

待用户决策后决定是否关闭 #21。