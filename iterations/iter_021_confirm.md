# Iter #021 (v6 confirm): E21 — (L=5, N=20) 独立验证 **失败**：信号存在但效应量不足

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: v6 独立验证 **4-gate 失败**：test_users=531（远超 150）、AUC=0.582、Cohen d=0.311、seed_pass=0.00、p_two=0.0001

## §A 审稿意见（v5 → v6 翻转）

**用户原话**（2026-08-15）：
> 接下来不要继续搜索 `(L,N)`，先冻结 `(5,20)`，完成一次真正独立验证。
> - 使用一批从未参与 v1–v5 的新用户
> - 每位用户至少40个完整句子，每句至少5词
> - 测试用户 ≥150
> - AUC ≥0.65 / Cohen d ≥0.5 / seed通过率 ≥0.80 / p <0.01
> - 全部通过，才能把 `(5,20)` 定为稳定门槛
> - 失败则说明信号存在，但40句话还不足以稳定提取

**v6 失败解读**：
- 置换 p_two=0.0001 → 信号**确实存在**（不是噪声）
- Cohen d=0.311 → 效应量**中等以下**（弱）
- seed_pass=0.00 → **没有任何 seed 过 AUC 0.65**（极不稳）
- **结论**：40 句话不足以稳定提取用户风格

**审稿结论**：Step 1 验证失败 → 不应继续 Step 2 (建向量) 或 Step 3 (Qwen 注入)。需要重新设计。

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：500 词以下 attribution 跌至 60%；100 词以下接近随机。
**PQB v6 现状**：v6 有 1062 个 ≥40 句用户（远超 150 gate），但效应量仍弱（Cohen d=0.31）— **与 Eder 警告的"短语料不可识别"边界一致**。

### 论文 [2] Brennan, Afroz, Greenstadt (2012)
**发现**：作者效应在自然书写中弱；6 次改写可破坏 attribution。
**PQB v6 现状**：v6 在 Baby Products 自然评论下 Cohen d=0.31 — 与 Brennan 2012 的"自然书写下作者效应弱"完全相符。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：verification 不需 balanced negative。
**PQB v6 现状**：v6 split-half 设计验证失败，表明 one-class verification 在当前特征集 + 数据规模下也不足够。

### 论文 [4] Stamatatos (2009) 综述
**发现**：单一 syntactic 通道弱；需要 ensemble。
**PQB v6 现状**：v6 只用 32 维 syntactic 比例特征；seed_pass=0.00 表明**单一通道完全不够稳定**。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：高维 kernel + per-author 压缩模型是 PAN SOTA；低维手工特征不够。
**PQB v6 现状**：v6 用 32 维手工特征 + L2 → 验证失败。需要 Kestemont 风格的高维特征或 per-author 模型。

### 综合 §B
v6 失败与 stylometry 经典文献完全一致：32 维 syntactic 手工特征 + 用户 40 句 + 单一通道不足以稳定提取用户风格。

## §C 代码缺陷定位

**v6 没有任何代码 bug**：置换检验、AUC 方向、Cohen d 计算都正确（沿用 v4 修正）。

**v6 揭示的根本问题**：
1. **样本规模 vs 效应量 trade-off**：
   - v5 (60 dev users): Cohen d=0.498（接近 0.5）
   - v6 (531 dev users): Cohen d=0.311（远低于 0.5）
   - **当用户池从 60 → 531 扩大 8.8x 时，效应量从 0.498 → 0.311（降 38%）**
   - 这表明 v5 的 0.498 是**样本偏差**而非真实效应量

2. **稳定提取所需的句子数 > 40**：
   - 当前 L=5, N=20（即 40 句）不足
   - 需要 L=5, N=50+ (100 句) 或更多

3. **特征集不足**：
   - 32 维 syntactic 比例特征 + L2 距离是弱 baseline
   - 需要 char n-gram / function word / POS n-gram 通道
   - 或 per-author compression model (Kestemont 风格)

## §D 修复方案（v7 候选）

### 方案 A：增 N（更多句子）
- (L=5, N=50)：每用户 100 句
- 预期：数据规模够，但用户数进一步减少（v6 1062 eligible → N=50 时可能 < 200）
- 风险：eligible 用户数可能跌破 150 gate

### 方案 B：增通道（更多特征）
- 32 dim syntactic + 1000 dim char n-gram + 200 dim function word
- 预期：特征维度从 32 → 1232，效应量应增强
- 风险：curse of dimensionality；需要更多用户

### 方案 C：换数据源
- 多 category（如 Baby + Office + Pet Supplies）
- 预期：用户池翻几倍
- 风险：跨 category 风格不通用

### 方案 D：换方法
- 不再做 split-half 风格识别
- 改为 prompt-level user modeling（用 LLM 直接建模用户偏好）
- 风险：跳出 stylometry 叙事

### 方案 E：放弃主动验证
- 接受 v6 结论：当前特征 + 数据下**用户句法风格不可稳定提取**
- paper 报告"negative result: passive stylometry on short niche reviews insufficient"
- 转向 prompt engineering

## §E 验证结果（v6）

```
seed: 43  (v5 was seed 42 — disjoint subsample)
n_users_parsed: 14691  (v5: 4453)
n_eligible: 1062  (≥40 sents of ≥5 words)  (v5 (5,20) eligible: 60 dev + 45 test)
dev_users: 531  (gate ≥150 PASS)
test_users: 531  (gate ≥150 PASS)

=== Dev evaluation (L=5, N=20) ===
  dev AUC=0.5816+-0.0145  d=0.3113  seed_pass=0.00  delta=0.01224

=== Dev 9999 permutations ===
  p_one=0.00010  p_two=0.00010  (extremely significant)

=== Test evaluation (independent users) ===
  test AUC=0.5732+-0.0113  d=0.2650  seed_pass=0.00  delta=0.01045

=== 4-gate evaluation ===
  test_users: 531 >= 150  [PASS]
  auc: 0.5816 >= 0.65  [FAIL]
  cohen_d: 0.3113 >= 0.5  [FAIL]
  seed_pass_frac: 0.0000 >= 0.8  [FAIL]
  perm_p_two: 0.0001 < 0.01  [PASS]

ALL GATES PASS: False
runtime_sec: 341.7
```

**v5 vs v6 关键对比**：

| | v5 (seed=42) | v6 (seed=43) | 倍数 |
|---|---|---|---|
| dev users | 60 | 531 | **8.8x** |
| dev AUC | 0.638 | 0.582 | -8.8% |
| dev Cohen d | 0.498 | 0.311 | -38% |
| test users | 45 | 531 | **11.8x** |
| test Cohen d | 0.513 | 0.265 | -48% |

**核心观察**：用户池扩大 8.8x 时，效应量下降 38-48%。**v5 的高效应量是样本偏差**。

## 下一步

**绝对不应**继续 Step 2 / Step 3（信号不稳就建向量 = 把噪声当风格）。

待用户决策（5 选 1）：
- (a) 方案 A：增 N 到 N=50（每用户 100 句），看效应量回升
- (b) 方案 B：增通道（char n-gram + function word），特征 32 → 1232 dim
- (c) 方案 C：换数据源（多 category）
- (d) 方案 D：换方法（prompt-level user modeling）
- (e) 方案 E：放弃主动验证，paper 报告 negative result

## 加速改进（v6 vs v5）

| 优化 | v5 | v6 | 加速 |
|---|---|---|---|
| spaCy 组件 | 全启用 | 禁 ner/lemmatizer/attribute_ruler | 1.4x |
| batch_size | 128 | 256 | 1.2x |
| 稳态 throughput | 89 reviews/s | 305 reviews/s | **3.4x** |
| 解析 71801 句用时 | (~14min) | 325s (~5.4min) | 2.6x |

## 文件

- 脚本：`query/soft_prefix/e21_confirm.py`
- 结果：`result/e21_confirm_results.json` + `result/e21_confirm.log`
- 文档：`iterations/iter_021_confirm.md`
- 提交：待 commit

待用户决策后决定是否关闭 #21 或继续下一轮实验。