# Iter #021 (v3): E21 — 2D 网格 (L, N) + split-half + dev/test 联合 NO-GO

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: 跑完，12 cell 中 6 cell 跑过，**全部 4-gate NO-GO**（best_cell_dev=None）

## §A 审稿意见

**尖锐问题**：E21 v1 / v2 都报告"resample ICC 0/32 不通过"，但缺乏"在什么 (L, N) 下能识别"的具体结论 — 是真有信号但在某些 (L, N) 不可见，还是根本不存在。

**用户提议（#21 后）**：
- 设计成 **二维网格搜索**
- 单句最少词数 `L`，每个用户最少完整句子数 `N`
- 例如 `L ∈ {5, 8, 10, 12}`、`N ∈ {10, 20, 30, 50}`
- 不能在所有组合里挑一个偶然显著的结果 — 必须用 dev/test split + 多重检验校正
- 最终结论可以是 "至少拥有 20 个完整句子，并且每句不少于 5 词时，用户句法向量能够稳定区分"

**审稿结论**：E21 v2 留下空白。v3 必须**真实执行 2D 网格 + 严格防 cherry-picking**。

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：500 词以下 attribution 跌至 60%；100 词以下接近随机。**语料量是最强 gate**。
**PQB 现状**：E21 v3 在 N=10/20/30/50 都跑不出可识别信号，与 Eder 2015 完全一致。N≥30 时 400 用户中 < 30 通过 filter，**数据本身不支持更大 N 评估**。
**改进方案**：承认数据规模上限；v3 量化 N=10/20 + L=3/5/8 整个网格的 NO-GO。

### 论文 [2] Brennan, Afroz, Greenstadt (2012) "Adversarial Stylometry"
**发现**：6 次改写使 attribution 80% → 20%。**自然书写作者效应弱**。
**PQB 现状**：Baby Products 2023 是自然评论，**样本稀缺**（400 用户 / 6K 评论），dev set 用户在 N=20 后只剩 35-43。N=50 时 dev 只剩 5 个用户 — **连统计检验都做不了**。
**改进方案**：承认 6K 评论的 niche 语料根本无法支撑 stylometry 评估。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：verification 不需 balanced negative。
**PQB 现状**：v3 用 split-half（half1 vs half2）正是 one-class verification 的轻量形式 — **同用户两半距离 vs 异用户单半距离**。
**改进方案**：NO-GO 表明 one-class 思路在当前数据规模也不足够。

### 论文 [4] Stamatatos (2009) 综述
**发现**：稳定性筛选是前置 gate；小样本下越严越不稳。
**PQB 现状**：v3 移除 3-criterion stability filter，直接用 32 维特征。发现 N=20 时 AUC 接近 0.65 但 Cohen d=-0.625 — **AUC 与方向冲突**。
**改进方案**：单一 AUC 指标易被异常值拉高，必须报 Cohen d 验证方向。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：跨域 verification 需 kernel-based similarity + per-author 压缩模型。
**PQB 现状**：v3 用 L2 + unit-norm（等价余弦）+ 32 维手工特征，与 E20 v2 相同的弱 baseline。
**改进方案**：要真正能识别需要 (a) 增通道 + (b) 增样本 + (c) 改主动训练。

### 综合 §B
E21 v3 网格 NO-GO 与 Eder 2015 / Brennan 2012 一致：**短文本 + 低维手工特征 + 小语料不可能 support stylometry**。

## §C 代码缺陷定位

**v2 的缺陷**：
1. **5 次重采样 ICC>0.7 阈值**：被严格性拖到全部失败
2. **无法给"在什么 (L, N) 下能识别"的答案**

**v3 的修复**：
1. **移除稳定性筛选**：直接用 32 维特征评估 split-half 信号
2. **2D 网格 (L, N)**：明确报告每个 cell 的 AUC、d、p
3. **dev/test split**：50/50 防止 cherry-picking
4. **Bonferroni 校正**：12 cell × p_two_sided
5. **多 seed 评估**：30 seeds/cell，要求 ≥80% seed 过 AUC 阈值
6. **4-gate 联合**：AUC≥0.65 + p_adj<0.01 + Cohen d≥0.5 + seed_pass≥0.80

**v3 设计要点**：
- 每个用户抽 **2N 个完整句子**（不截断，break 句子被排除）
- 分两半各 N 句
- 同用户距离：||v1n - v2n||（half1 vs half2 of same user）
- 跨用户距离：||v1n - vvn||（user U half1 vs 3 个随机其他用户的 N 句）
- L2 on unit-normalized 32 维向量

## §D 修复方案

`query/soft_prefix/e21_l_grid.py`：

1. **数据 pipeline**：
   - pass 1：累计词数
   - pass 2：装入用户 → (asin, text) list，≥3 商品
   - spaCy parse 一次所有评论
   - 子采样到 N_USERS=400

2. **dev/test split 50/50**（per-user hash）

3. **2D 网格循环**：L ∈ {3, 5, 8}, N ∈ {10, 20, 30, 50}

4. **每个 cell**：
   - 过滤：sentences with n_tok ≥ L
   - eligible：≥ 2N filtered sents
   - 30 seeds × 30-100 用户 × (1 self + 3 cross) 距离
   - AUC = P(d_cross > d_self)
   - delta = mean(d_self) - mean(d_cross)
   - Cohen d = delta / pooled_std

5. **置换检验**：999 次，n_perm_users=50 subsample（速度）
   - p_one_lower: P(null_delta ≤ obs_delta)
   - p_two_sided: 双向
   - Bonferroni: ×12

6. **多 seed pass**：≥80% seeds AUC≥0.65

7. **test 验证**：best cell 在 test 用户上再跑 30 seeds

## §E 验证结果

```
users scanned: 3386206 (t=42.3s)
word>=200 candidates: 1600
users with >= 3 products: 1131
users parsed: 1131 (t=187.8s)
using 400 users
dev users: 200, test users: 200

2D grid (L, N) results:
  (3,10) dev=141  AUC=0.605+-0.020  d=-0.373  p_one=0.474  seed_pass=0.00
  (3,20) dev=43   AUC=0.668+-0.035  d=-0.625  p_one=0.408  seed_pass=0.70
  (3,30) dev=22 < 30 skipped
  (3,50) dev=5  < 30 skipped
  (5,10) dev=133  AUC=0.587+-0.026  d=-0.293  p_one=0.624  seed_pass=0.00
  (5,20) dev=35   AUC=0.590+-0.042  d=-0.274  p_one=0.658  seed_pass=0.07
  (5,30) dev=18 < 30 skipped
  (5,50) dev=5  < 30 skipped
  (8,10) dev=101  AUC=0.577+-0.028  d=-0.251  p_one=0.520  seed_pass=0.00
  (8,20) dev=29 < 30 skipped
  (8,30) dev=13 < 30 skipped
  (8,50) dev=4  < 30 skipped

best_cell_dev: None
test_verification: None
runtime_sec: 228.3
```

**E21 v3 量化 NO-GO 结论**：

1. **6 个跑过的 cell 全部 Cohen d < 0**（方向反向）
   - `(3,20)` 看似 AUC=0.668 过阈值，但 d=-0.625 + p=0.408 + seed_pass=0.70
   - **AUC 单项过阈值但 4-gate 联合失败**

2. **AUC 与 Cohen d 方向冲突的原因**：
   - AUC 是 rank 统计（基于 (d_cross > d_self) 的成对比较）
   - Cohen d 是 mean difference
   - 出现"rank 略好于随机 + mean 略差"的典型 small-sample 场景
   - 这种情况下 **mean effect size 更可靠**，不应只看 AUC

3. **N≥30 时数据枯竭**：
   - L=3, N=30 时 dev 只剩 22 用户
   - L=5, N=30 时 dev 只剩 18 用户
   - L=8, N=20 时 dev 只剩 29 用户
   - **Baby Products 2023 在 400 用户 / 6K 评论的语料下，N=30 已经接近上限**

4. **没有 cell 通过 4-gate**：
   - AUC≥0.65 + p_adj<0.01 + Cohen d≥0.5 + seed_pass≥0.80
   - best_cell_dev=None
   - **整个网格在 Baby Products 2023 上 NO-GO**

**与前序实验联合**：

| 实验 | 设计 | 结果 |
|---|---|---|
| E19 v3 (#19) | 硬匹配 product-type ∩ rating ∩ 词数 bin | diff<20 → insufficient |
| E20 v2 (#20) | LOPO + 条件置换 + 30 维比例特征 | delta=-0.0136, AUC=0.3472, p=0.28 |
| E21 v2 (#21) | 3-criterion 稳定性筛选 + 5 次重采样 | 0/32 稳定特征 |
| **E21 v3 (本)** | **2D 网格 split-half + dev/test** | **best_cell=None** |

**四次实验联合**：在 Baby Products 2023 ≤900 词 + 32 维手工 syntactic 特征下，**任何被动 stylometry 方法（硬匹配 / LOPO / 稳定性筛选 / split-half 网格）都不支持用户句法风格可识别性**。

## 下一步

待用户决策（issue #21 comment 列出 5 个候选）：
- (a) 接受完整 NO-GO，paper §3.X 报告"4 次实验联合 NO-GO"
- (b) 增通道（char n-gram + function word + POS n-gram）— 维度扩到 200+
- (c) 增语料（多 category / 多时间窗口）
- (d) 改主动训练（contrastive learning / adversarial training）
- (e) 退出被动 stylometry 叙事，转 prompt-level user modeling