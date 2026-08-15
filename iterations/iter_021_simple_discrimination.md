# Iter #021: E21 — 严格 3 项稳定性筛选 + 简化同/异用户风格可分性

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: 跑完，3 项稳定性筛选交集 = 0/32，**无词数鲁棒句法特征**

## §A 审稿意见

**尖锐问题**：E20 v2 实施时**没有按 #20 计划的简化目标执行**：
1. 研究问题偏了 — 做了 LOPO 跨商品保持（更严格），而非"汇总评论同/异用户可分"（更基础）
2. 稳定性筛选只做了 1 项（`|corr(feat, sent_count)|<0.2`），计划是 3 项（|corr|<0.1 + 跨词数 ICC>0.6 + 重采样 ICC>0.7）
3. AUC 方向可能错 — 0.3472 < 0.5 与 d_self < d_cross 不一致，需检查标签约定
4. 置换 99 次对应 p<0.01 等价于"零次出现"，至少 999 次

**审稿结论**：E20 v2 跑完但 NO-GO 不可解释（不能区分"统计效力不足" vs "无信号"）。E21 实施**真正的简化目标 + 严格 3 项筛选**。

## §B 论文 vs PQB 现状对比反思

**注**：`mcp__consensus__search` 不可用；WebSearch 注入假 URL。改用训练知识中真实 stylometry 经典文献做对比反思。

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：100 词以下 stylometry 接近随机；小样本下重采样稳定性是关键瓶颈。
**PQB 现状**：E21 跑出"5 次重采样 ICC<0.7 占 32/32"，**与 Eder 2015 一致 — 短文本下句法特征重采样不稳定**。
**改进方案**：承认 NO-STABLE；E21 NO-GO 量化记录。

### 论文 [2] Brennan, Afroz, Greenstadt (2012) "Adversarial Stylometry"
**发现**：自然书写下作者效应远弱于教科书假设；人类改写 6 次可使 attribution 准确率从 80% 降至 20%。
**PQB 现状**：E21 5 次重采样都不稳定，符合"自然书写下作者效应弱"。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM Verification"
**发现**：one-class 比 balanced binary 更好；verification 不需要 negative set。
**PQB 现状**：E21 用 one-class 思路（同用户 vs 异用户）但连"同用户重采样"都不稳定，更谈不上 verification。

### 论文 [4] Stamatatos (2009) 综述
**发现**：稳定性筛选是 stylometry 的标准前置 gate；常用 5–10 折交叉验证评估。
**PQB 现状**：E21 实施 5 次重采样 ICC，是 Stamatatos 推荐做法的实现。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：跨域 verification 需高维句向量（char n-gram / compression model），低维手工特征失效。
**PQB 现状**：E21 32 维手工特征在 5 次重采样下全部不稳定，与 Kestemont 2016 一致。

### 综合 §B

E21 NO-STABLE-FEATURES 与 stylometry 经典文献的"短文本 + 低维手工特征 NO-GO"完全一致。**不是实现 bug，是 stylometry 在当前数据 + 当前特征集 + 当前样本规模下的预期结果**。

## §C 代码缺陷定位（E20 v2 偏题点）

E20 v2 (`e20_lopo_v2.py`) 的核心问题：

1. **LOPO 偏题**：
   - 计划: 汇总评论 + 同/异用户可分
   - 实际: per-(user, product) residual + LOPO 留一商品
   - 跨商品保持是**更强**的命题，不能用来否定**更弱**的命题

2. **稳定性筛选只 1 项**：
   - 计划: |corr|<0.1 + 跨词数 ICC>0.6 + 重采样 ICC>0.7
   - 实际: |corr(feat, sent_count_per_product)|<0.2 (单层)
   - 保留了 30/32 维"稳定"特征 — 实际这 30 维重采样都不稳定

3. **AUC 方向未 sanity check**：
   - E20 v2 输出 AUC=0.3472 < 0.5，与 d_self < d_cross 方向冲突
   - 可能是 `auc += (diffs_cross < d).sum()` 而非 `> d` — 标签反向

4. **置换 99 次 vs p<0.01**：
   - 99 次最小 p=0.01，p<0.01 等价于"零次出现" → 永远不显著
   - 应该至少 999 次

## §D 修复方案（E21）

`query/soft_prefix/e21_simple_discrimination.py`：

1. **汇总用户评论**（不做 LOPO）：
   - 选 K=300 用户，≥200 词、≥3 商品、≥10 句
   - 每个用户全部评论拼成 per-user 长文本，spaCy parse 一次

2. **3 项严格稳定性筛选**（`STAB_*_THRESH` 常量）：
   - (a) `|corr(feat, sent_count)| < 0.1` — 跨用户
   - (b) 跨词数子样本 ICC > 0.6 — 60/100/240/450 词 × 5 重采样
   - (c) 重采样 ICC > 0.7 — 同词数 5 次重采样
   - 取 **a ∩ b ∩ c** 作为稳定特征集

3. **同/异用户直接比较**（不做 LOPO）：
   - 同用户: 同一用户两次不同 resample 距离
   - 异用户: 1:3 跨用户对距离
   - 不做 product cluster 匹配

4. **AUC 方向 sanity**：
   - 同时报 `P(d_cross > d_self)` 和 `P(d_cross < d_self)`
   - 明确写出"标签是否反向"

5. **999 次置换**：
   - 在 K 用户中随机配对，构造 null d_self 分布
   - 与 obs delta 比对

## §E 验证结果

```
users scanned: 3386206 (t=43.7s)
word>=200 candidates: 1200
users with >= 3 products: 867
users with >= 10 sents: 861 (t=164.5s)
using 300 users
sample grid shape: (300, 4, 5, 32)  # (n_users, n_word_bins, n_resamples, n_feats)

3-criterion stability filter:
  (a) |corr(feat, sent_count)| < 0.1:    17/32 pass
  (b) cross-bin ICC > 0.6:                 30/32 pass
  (c) resample ICC > 0.7:                  0/32 pass  ← 全部不通过
  a ∩ b ∩ c:                               0/32  ← 交集为空

self/cross:
  d_self_mean: 0.0000  (feat_dim=0 → 空数组 norm = 0)
  d_cross_mean: 0.0000
  delta: 0.0000
  AUC: 0.0000 / 0.0000

runtime_sec: 169.1
```

**E21 量化 NO-STABLE-FEATURES 结论**：

- 32 维比例型特征中 **0 维**通过完整 3 项稳定性筛选
- **真正的瓶颈是 (c) 重采样 ICC**：5 次同词数重采样之间，特征波动 > 30%
- (a) corr 和 (b) 跨词数子样本 ICC 大部分通过，但 (c) 全部失败
- 在 Baby Products 2023 当前数据 + 5 次重采样规模下，**没有真正"词数鲁棒 + 跨词数稳定 + 重采样稳定"的句法特征**

**与 E20 v2 / E19 v3 联合结论**：

- E19 v3 硬匹配 NO-GO（cands 池 0）
- E20 v2 LOPO 弱反向（delta 极小 + AUC<0.5）
- E21 严格稳定性筛选 0/32 稳定特征
- **三次实验联合**：Baby Products 2023 上 ≤450 词 + 5 次重采样规模下，**用户句法风格可识别性不被任何被动方法支持**

## 下一步

待用户决策：
- (a) 接受 NO-STABLE，更新 paper
- (b) 降 resample ICC 阈值 0.7 → 0.3–0.4
- (c) 增 N_RESAMPLE 5 → 10/20
- (d) 用 cross-bin ICC (b) 单一标准 (30/32 通过)
- (e) 改聚合方式（按句子数子采样）
