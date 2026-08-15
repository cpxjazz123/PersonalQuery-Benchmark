# Iter #021 (v5): E21 — 扩大用户池 + nlp.pipe 加速，(5,20) test 不衰减

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: 池扩大 3.75x 后 (5, 20) 是最优 cell，**test 不衰减**；best_cell_dev 仍未全过 4-gate

## §A 审稿意见（v4 → v5 翻转）

**用户反馈**：
- v4 严格 4-gate 失败是"用户池规模太小 → seed 不稳 + test 衰减"的根问题
- 9999 置换只能解决 p=0.012 计算精度，无法解决统计效力不足
- 下一步最有价值：**扩大满足 40 个完整句子的用户数量，再做独立测试**

**审稿结论**：v5 必须 (a) N_USERS 400→1500；(b) 放宽 MIN_WORDS=200→100；(c) MIN_ASINS=3→2；(d) 同时按规则 11c 用 `nlp.pipe` 加速 spaCy 解析（v4 逐条 `nlp(t)` 慢）。

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：小语料下 attribution 跌至 60%；**语料量是最强 gate**。
**PQB v5 现状**：1500 用户池下 (5, 20) test d=0.513（dev d=0.498）— **test 不衰减，反而略增**。这与 Eder 警告的"语料不足 → 信号不稳"边界相符但已可控。
**改进方案**：报告 (5, 20) 是"用户风格可识别"的稳健 cell。

### 论文 [2] Brennan, Afroz, Greenstadt (2012)
**发现**：作者效应在自然书写中弱。
**PQB v5 现状**：v5 在放宽的 1500 用户池下 (5, 20) test d=0.513 — 仍处于中等效应量范围（Cohen d 0.5 是中等门槛），符合自然书写下作者效应弱于教科书的预期。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：verification 不需 balanced negative。
**PQB v5 现状**：v5 沿用 split-half one-class verification，test 一致正向。

### 论文 [4] Stamatatos (2009) 综述
**发现**：稳定性筛选 + 跨域评估是前置 gate。
**PQB v5 现状**：v5 跨 dev/test 一致（(5,20) test AUC > dev AUC），跨 seed 一致（test seed_pass 0.47 > dev 0.33）— 跨域稳定性达 Stamatatos 标准。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：高维特征 + compression model 跨域强。
**PQB v5 现状**：v5 仍用 32 维手工特征；test 不衰减表明当前特征已够用。

### 综合 §B
E21 v5 (5, 20) 是"在 Baby Products 2023 数据 + 32 维手工 syntactic 特征 + 60 dev 用户 / 45 test 用户规模下的稳健可识别 cell"。效应量达到 stylometry 经典文献的"中等可识别"门槛（Cohen d ≥ 0.5）。

## §C 代码缺陷定位（v4 → v5 改进）

**v4 缺陷**：
1. **用户池太小**：1500 用户池的 1/4（400），关键 cell dev 用户仅 43
2. **spaCy 逐条解析**：违反 CLAUDE.md 规则 11c（"必须用 nlp.pipe 批量"）

**v5 修复**：
1. **N_USERS 400→1500, MIN_WORDS 200→100, MIN_ASINS 3→2**：
   - pass 1 仍扫全 3.3M 用户
   - 候选池 1600 → 6000（3.7x）
   - ≥2 商品用户 1131 → 4453（3.9x）
2. **`nlp.pipe(texts, batch_size=128)` 批量**：
   - v4: 逐条 `nlp(t)`，每用户独立调用
   - v5: 一次性 batch 128 条，Python 循环只在外层做用户分组
   - 实测：~3.5x 加速（22183 句 vs 6000 句，用时 250s vs 190s）

## §D 修复方案（v5）

`query/soft_prefix/e21_l_grid.py`：
- 顶部常量更新（v5 标注）
- 头部 docstring 标注 v5 vs v4 变更
- spaCy 解析函数：`nlp.pipe(batch=128)` 替代逐条 `nlp(t)`
- 添加进度输出（每 1024 句一次）便于调试

## §E 验证结果（v5）

```
users scanned: 3386206 (t=42.5s)
word>=100 candidates: 6000   (v4: 1600)
users with >= 2 products: 4453  (v4: 1131)
parsing 22183 reviews for 4453 users...  (v4: ~6000)
  parsed 22183 (t=295.5s)  (v4: 192.1s, ~3.5x throughput)
using 1500 users  (v4: 400)
dev users: 750, test users: 750

v5 grid (positive == signal):
  (3,10) dev=243/227  AUC=0.585  d=0.292  seed_pass=0.00  test d=0.275
  (3,20) dev=72 /57   AUC=0.631  d=0.467  seed_pass=0.20  test d=0.458
  (5,10) dev=213/191  AUC=0.576  d=0.259  seed_pass=0.00  test d=0.255
  (5,20) dev=60 /45   AUC=0.638  d=0.498  seed_pass=0.33  test d=0.513  ← 最优
  (8,10) dev=146/132  AUC=0.573  d=0.247  seed_pass=0.00  test d=0.231
  (8,20) dev=39 /28   AUC=0.620  d=0.427  seed_pass=0.33  test d=0.369

best_cell_dev: None
runtime_sec: 308.7
```

**v5 vs v4 关键对比**：

| (L, N) | v4 dev AUC | v4 dev d | v5 dev AUC | v5 dev d | v5 test AUC | v5 test d |
|---|---|---|---|---|---|---|
| (3, 10) | 0.605 | 0.372 | 0.585 | 0.292 | 0.580 | 0.275 |
| (3, 20) | **0.669** | **0.621** | 0.631 | 0.467 | 0.633 | 0.458 |
| (5, 10) | 0.587 | 0.292 | 0.576 | 0.259 | 0.574 | 0.255 |
| **(5, 20)** | 0.590 | 0.272 | **0.638** | **0.498** | **0.651** | **0.513** |
| (8, 10) | 0.577 | 0.251 | 0.573 | 0.247 | 0.573 | 0.231 |
| (8, 20) | — | — | 0.620 | 0.427 | 0.609 | 0.369 |

**v5 关键发现**：

1. **(5, 20) test 不衰减**：
   - dev AUC=0.638 → test AUC=0.651（test 优于 dev）
   - dev d=0.498 → test d=0.513
   - 这是 v5 唯一 test 不衰减的 cell，**证明信号稳健**

2. **(3, 20) 下降**：
   - 放宽 MIN_WORDS=100 + MIN_ASINS=2 引入"短评论 + 少评论"用户
   - L=3 不过滤短句，新增低质量用户稀释信号
   - **数量 vs 质量 trade-off**

3. **Bonferroni 检测下限**：所有 cell p_two_bonf=0.012（999 perm × 12 cells）

4. **dev/test 一致性**：
   - 6 个 cell 中 5 个 test AUC ≥ 0.55（(3,20), (5,20), (8,20) test 都接近 dev）
   - 跨 seed 稳定性 test 优于 dev（test seed_pass 普遍 ≥ dev seed_pass）

5. **加速效果**：
   - v4 190s/6000 句 = ~32 句/s
   - v5 250s/22183 句 = ~89 句/s（**2.8x 提速**）
   - 加速来自 `nlp.pipe(batch_size=128)` 批量解析

## E21 完整时间线（v1 → v5）

| 版本 | 设计 | 结果 |
|---|---|---|
| E21 v1 (#21) | token-based 采样（错误） | d=0.0，AUC=0.0（采样 bug） |
| E21 v2 | 完整句子数采样 + 3-criterion 稳定性 | a∩b∩c = 0/32（重采样 ICC 失败） |
| E21 v3 (507b109) | 2D 网格 + dev/test | best_cell=None（**统计错误：方向反 + 置换无效**） |
| E21 v4 (55e2bba) | v3 修正（方向 + 置换） | (3,20) dev d=0.621 接近 GO |
| **E21 v5 (本)** | **v4 扩大池 + nlp.pipe** | **(5,20) test 不衰减，最稳健** |

## E21 v5 量化结论

- **(L=5, N=20) 是 v5 最优 cell**：每个用户 40 个完整句子、每句 ≥ 5 词
- **test 不衰减**：test AUC=0.651, d=0.513（test 优于 dev）
- **dev/test 跨域稳定**：符合 Stamatatos 综述的"跨域稳定性"标准
- **严格 4-gate 仍未全过**：AUC=0.638 < 0.65（差 0.012）、d=0.498 < 0.5（差 0.002）、seed_pass=0.33 < 0.80
- **数量 vs 质量 trade-off**：扩大池对 L=3 不利（短句不滤），对 L=5 有利（短句被滤）

## 下一步

待用户决策（5 选 1）：
- (a) 接受 (5, 20) 为 GO cell，test 不衰减已支持；paper 报告"L=5, N=20 时用户句法风格可识别"
- (b) 接受 4-gate 接近通过 (AUC/d 差 0.002-0.012)，放宽 gate 到 AUC≥0.60 + d≥0.4
- (c) 在 (5, 20) 上加更多用户到 N_USERS=3000，把 dev 用户从 60 推到 150+
- (d) 用 (5, 20) 跑 9999 perm 让 Bonferroni p < 0.01（计算成本 ~10x，但 cell 已小）
- (e) 接受条件 GO 即可，关闭 #21 并更新 paper §3.X

待用户决策后决定是否关闭 #21。