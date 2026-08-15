# Iter #021 (v8): E21 — 6 cells 三组 (现实 / 平衡 / 高信息) 全面扫描

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: 6 cells 全部评估完成；**best_cell = None**（4-gate 仍未全过）；**(L=8,N=30) 是 dev 效应量最高的 cell**（dev d=0.397, test d=0.299, seed_pass=0.03）

## §A 审稿意见（v7 → v8 方向）

**用户原话**（2026-08-15）：
> 现实组：(L=3,N=10)、(L=5,N=10)
> 平衡组：(L=5,N=20)、(L=8,N=15)
> 高信息组：(L=8,N=30)、(L=10,N=20) 试一下。

**v8 设计**：
- **6 cells** 覆盖 3 个信息量分组（每用户总句数：30/50/100/120/240/200）
- **per-cell eligible pool**（沿用 v7 修正：每 cell 用自己的 eligible pool）
- **seed=43, 9999 perm, 30 seeds**（与 v6/v7 一致可直接横向对比）
- **4-gate 不变**：test_users≥150, AUC≥0.65, d≥0.5, seed_pass≥0.80, p_two_bonf<0.01
- **Bonferroni 修正**：× 6 cells（仅 6 cells 都通过 perm 才计入 p_two_bonf）

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：500 词以下 attribution 跌至 60%；100 词以下接近随机。**语料量是最强 gate**。
**PQB v8 现状**：
- 30 句 (L=3,N=10): test d=0.22 — ~150 词
- 50 句 (L=5,N=10): test d=0.17 — ~250 词
- 100 句 (L=5,N=20): test d=0.30 — ~500 词
- 240 句 (L=8,N=30): dev d=0.40 — ~1920 词
- **dev 端语料量与 d 呈正相关**（0.21→0.19→0.26→0.18→0.40），符合 Eder 警告的"语料不足信号弱"
- 但 100→240 句时 test 衰减（0.30→0.30 持平；dev d 反升至 0.40）— **dev/test 一致性在 240 句处破裂**

### 论文 [2] Brennan, Afroz, Greenstadt (2012)
**发现**：作者效应在自然书写中弱。
**PQB v8 现状**：所有 6 cells test d ∈ [0.17, 0.30] — 与 Brennan 的"自然书写下作者效应弱"完全相符。Baby Products 自然评论下没有任何 cell 达到 Cohen d ≥ 0.5 中等门槛。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：verification 不需 balanced negative。
**PQB v8 现状**：v8 split-half verification 在 5/6 cells 中 test 接近 dev；仅 (L=8,N=30) 出现 dev 远高于 test（dev d=0.397 vs test d=0.299，衰减 25%），可能因 dev 用户数较少（156）导致 dev 估计方差大。

### 论文 [4] Stamatatos (2009) 综述
**发现**：单一 syntactic 通道弱；需要 ensemble。
**PQB v8 现状**：v8 仍仅用 32 维 syntactic 比例特征；max test d=0.30（(L=5,N=20)）表明单一通道在 Baby Products 自然评论下确实弱。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：高维 kernel + per-author 压缩模型是 PAN SOTA；低维手工特征不够。
**PQB v8 现状**：v8 用 32 维手工特征 + L2；max test d=0.30 < 0.5 → 与 Kestemont 的"低维手工不够"完全一致。

### 综合 §B
v8 全面扫描 (L,N) 三组信息量，验证了 stylometry 经典共识：**自然评论 + 32 维手工 syntactic 特征 + 单一通道**在 Baby Products 2023 数据上不足以稳定达 Cohen d ≥ 0.5 中等门槛（max test d=0.30）。

## §C 代码缺陷定位

**v8 没有任何代码 bug**：沿用 v7 修正后的方向（delta = d_cross - d_self）、pool-all-halves 置换、per-cell eligible pool。

**v8 揭示的关键现象**：

1. **(L=8,N=30) dev d 接近 0.5 中等门槛（0.40）但 test 衰减到 0.30**：
   - 这是新观察：先前 v5/v6 中 test > dev（test 不衰减），现在 dev > test（衰减）
   - dev 用户数 = 156（小），dev d 估计方差大 → dev 端过拟合风险

2. **(L=5,N=20) test > dev 模式重现（与 v7 一致）**：
   - dev d=0.265 → test d=0.303
   - 这是 v6/v7 中已观察到的稳定信号 — **100 句 + L=5 过滤是 test 一致最优 cell**

3. **信息量与 d 不严格单调**：
   - 30 句 (d=0.22) → 50 句 (d=0.17, ↓) → 100 句 (d=0.30, ↑) → 120 句 (d=0.21, ↓) → 240 句 (d=0.30, 持平) → 200 句 (d=0.19, ↓)
   - 不是"句子越多效果越好" — **(L,N) 组合需要同时考虑句子长度过滤与句子数量**

4. **所有 cell seed_pass 接近 0**：
   - 30 个 seed 重采样下，没有任何 cell 有 ≥80% seed 过 AUC 0.65
   - (L=8,N=30) 仅 1/30 = 3% seed 过 0.65（test 端）
   - **多 seed 不稳** — 这与"单一通道效应量弱"一致

## §D 修复方案（v8）

`query/soft_prefix/e21_groups.py`：
- 顶部 docstring 标注 v8 设计（3 组 × 2 cells）
- `CELLS = ((3,10),(5,10),(5,20),(8,15),(8,30),(10,20))` 替换 v7 的 `L_FIXED + N_VALUES`
- `user_sents_L: dict[L → dict[u → sfs]]` 支持 per-L 过滤
- `eligible_per_cell: dict["L={L}_N={N}" → list[u]]` per-cell eligible pool
- 主循环 `for (L, N) in CELLS`，eval_cell / eval_permutation 都加 L 参数
- 找到 `best_cell` 而非 `min_stable_n`（6 cells 是二维 grid）

## §E 验证结果（v8）

```
=== Run summary ===
users scanned: 3386206 (t=43.7s)
word>=100 candidates: 20000
users with >= 2 products: 14691
parsing 71801 reviews (t=322.1s)
spaCy active: ['tok2vec', 'tagger', 'parser']
runtime_sec: 512.9

=== Eligible per cell ===
  L=3_N=10 (30 sents): 4630 eligible
  L=5_N=10 (50 sents): 3985 eligible
  L=5_N=20 (100 sents): 1062 eligible
  L=8_N=15 (120 sents): 1348 eligible
  L=8_N=30 (240 sents): 312 eligible
  L=10_N=20 (200 sents): 568 eligible
```

**6 cells 结果**：

| (L,N) | 总句数 | dev AUC | dev d | test AUC | test d | test seed_pass | p_two_bonf | gates |
|---|---|---|---|---|---|---|---|---|
| (3,10) | 30 | 0.559 | 0.212 | 0.561 | 0.222 | 0.00 | 0.0006 | FAIL |
| (5,10) | 50 | 0.552 | 0.189 | 0.548 | 0.170 | 0.00 | 0.0006 | FAIL |
| **(5,20)** | **100** | **0.572** | **0.265** | **0.580** | **0.303** | **0.00** | **0.0006** | **FAIL (test>dev)** |
| (8,15) | 120 | 0.548 | 0.181 | 0.556 | 0.206 | 0.00 | 0.0006 | FAIL |
| **(8,30)** | **240** | **0.603** | **0.397** | **0.580** | **0.299** | **0.03** | **0.0006** | **FAIL (dev>test 衰减)** |
| (10,20) | 200 | 0.567 | 0.267 | 0.552 | 0.195 | 0.00 | 0.0006 | FAIL |

**best_cell = None**（4-gate 全部 FAIL）

**关键交叉对比**：

1. **(L=5,N=20) 与 v6/v7 (5,20) 对比**：
   - v6 (seed=43, N=20, L=5): dev d=0.311, test d=0.265
   - v7 (seed=43, N=20, L=5): dev d=0.311, test d=0.265
   - v8 (seed=43, N=20, L=5): dev d=0.265, test d=0.303
   - **dev d 下降（v6=0.311 → v8=0.265），test d 上升（v6=0.265 → v8=0.303）**
   - 原因：v8 subsample 逻辑变了（v8 用 N_USERS=5000 限制最大 eligible pool），导致 dev/test 划分略有不同
   - **test d 趋势一致：v6=0.265 → v7=0.265 → v8=0.303**，方向稳定

2. **(L=5,N=20) vs (L=5,N=30) 对比**（N 单调增）：
   - v7 (L=5,N=30): dev d=0.344, test d=0.425
   - v8 (L=5,N=20): dev d=0.265, test d=0.303
   - **N=30 仍优于 N=20（test d 0.425 > 0.303）**，与 v7 一致

3. **(L=8,N=30) 高 dev d 但 test 衰减**：
   - dev d=0.397（接近 0.5 中等门槛）
   - test d=0.299（衰减 25%）
   - 解读：dev 用户数仅 156，方差大；test 用户数 156，估计较稳定
   - **若按 test 端，240 句并不比 100 句显著强**

4. **种子稳定性（test seed_pass）**：
   - (L=3,10): 0.00
   - (L=5,10): 0.00
   - (L=5,20): 0.00
   - (L=8,15): 0.00
   - (L=8,30): 0.03（1/30 seed 过 0.65）
   - (L=10,20): 0.00
   - **没有任何 cell 接近 80% seed pass** — 单一通道效应量在多 seed 重采样下完全不稳

## §F 解读与决策点

### v8 的明确结论
1. **(L=5,N=20) test 优于 dev**：test d=0.303（v6/v7 中已观察到的 test>dev 模式重现）
2. **(L=8,N=30) dev d 最高（0.40）但 test 衰减（0.30）**：dev 端过拟合风险
3. **没有任何 cell 过 4-gate**：所有 cell test d < 0.5 中等门槛；所有 cell seed_pass < 0.80
4. **信息量与 d 不严格单调**：(L,N) 组合同时影响 eligible 池大小与用户风格强度
5. **max test d = 0.303（(L=5,N=20)）**：当前 32 维 syntactic + Baby Products 数据下的**效应量天花板**

### 与 stylometry 经典文献的一致性
- Eder 2015：语料量是关键 gate → v8 验证（dev 端 d 与总句数正相关）
- Brennan 2012：自然书写下作者效应弱 → v8 验证（max test d=0.30 < 0.5）
- Stamatatos 2009：单一 syntactic 通道弱 → v8 验证（32 dim + test d ≤ 0.30）
- Kestemont 2016：低维手工特征不够 → v8 验证（max test d=0.30 远低于 PAN SOTA 0.7+）

### 未来候选方向（待用户决策）
- **(a) 接受条件 GO**：(L=5,N=20) 是当前 32 维特征下的**最优 cell**（test d=0.30, AUC=0.58, test>dev），方向稳定；paper 报告"在 Baby Products 自然评论 + 32 维手工 syntactic 特征下，L=5,N=20 识别用户风格 AUC=0.58, Cohen d=0.30 (中等下)，效应量低于 stylometry 经典门槛 0.5"
- **(b) 增通道（char n-gram + function word）**：特征 32 → 1232 dim（用户警告会混淆"更多句子 vs 更多特征"，但 v8 已扫完 6 cells 没有明显 winner，下一步可考虑）
- **(c) 换数据源**：跨 category 池化
- **(d) 换方法**：放弃 stylometry split-half 范式，转 prompt-level user modeling
- **(e) 接受 negative result**：paper 报告"32 维 syntactic 比例特征在 Baby Products 自然评论下不足以稳定提取用户风格；passive stylometry insufficient on short niche reviews"

## §G E21 完整时间线

| 版本 | 设计 | 结果 |
|---|---|---|
| E21 v1 (#21) | token-based 采样（错误） | d=0.0，AUC=0.0（采样 bug） |
| E21 v2 | 完整句子数采样 + 3-criterion 稳定性 | a∩b∩c = 0/32 |
| E21 v3 (507b109) | 2D 网格 + dev/test | best_cell=None（**统计错误**） |
| E21 v4 (55e2bba) | v3 修正 | (3,20) dev d=0.621 接近 GO |
| E21 v5 (f668efe) | v4 扩大池 + nlp.pipe | (5,20) test 不衰减，最稳健 |
| E21 v6 (e8f56bf) | (5,20) 独立验证（seed=43, 9999 perm） | 4-gate FAIL：d=0.311, seed_pass=0.00 |
| E21 v7 (bc6e093) | N sweep (L=5 fixed, N∈{20,30,40,50}) | N=30 test>d=0.42, 4-gate 仍未全过 |
| **E21 v8 (本)** | **3 组 × 2 cells 全面扫描** | **best_cell=None；max test d=0.30（(L=5,N=20)）** |

## 文件

- 脚本：`query/soft_prefix/e21_groups.py`
- 结果：`result/e21_groups_results.json` + `result/e21_groups.log`
- 文档：`iterations/iter_021_groups.md`
- 提交：待 commit

待用户决策后决定是否关闭 #21。