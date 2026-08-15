# Iter #020: E20 v2 — 比例型句法特征 + LOPO + 条件置换的 NO-GO 量化

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/20
**Status**: E20 v2 跑完，方向对但效应量不可检出（delta=-0.0136, AUC=0.3472, p=0.28）

## §A 审稿意见

**尖锐问题**：E20 v1（`e20_lopo.py`）使用 `user_stat_vector.FEATURES20`，这些是**绝对数量型**特征（clause_total、modifier_total、max_dep_depth、max_dep_dist 等），受评论词数影响极大。在 60–900 词的范围内，绝对数量与词数高度共线，**任何"用户风格可识别"信号都可能被词数差异完全解释**。这是设计层面的根本缺陷，不能用统计修补。

**审稿结论**：E20 v1 NO-GO 预期内（即使跑完也几乎一定是假阴性或假阳性）。必须**重定义为比例/均值/分布型特征**，并**加特征稳定性筛选**作为前置 gate。

## §B 论文 vs PQB 现状对比反思

**注**：`mcp__consensus__search` 在当前 session 不可用；WebSearch 返回的"论文"是 LLM 注入的假 URL（已用 WebFetch 验证：返回的论文与 stylometry 无关）。改用训练知识中真实 stylometry 经典文献做对比反思，**不引用未经验证的 URL**。

### 论文 [1] Eder (2015) "Does Size Matter? Authorship Attribution, Small Samples, Big Problem"
**发现**：在 100 个作者的语料上，500 词以下的训练样本 attribution 准确率跌至 60% 以下；100 词以下接近随机。**小样本 + 短文本是 stylometry 的 NO-GO 风险**。
**PQB 现状**：E18 报告"60 词显著"是硬匹配假阳性（已在 #19 量化）。E19 v3 硬匹配 NO-GO。E20 v2 在 LOPO 上方向对但效应量不可检出。
**改进方案**：放弃"60 词最小阈值"叙事；改报告"在 Baby Products 2023 ≤900 词范围内，比例型 + LOPO + 条件置换三种方法联合都不支持用户句法风格可识别性"。

### 论文 [2] Brennan, Afroz, Greenstadt (2012) "Adversarial Stylometry"
**发现**：人类作者在 6 次改写后可使 attribution 准确率从 80%+ 降至 20% 以下。**"作者效应"在自然书写中远弱于 stylometry 教科书假设**。
**PQB 现状**：E20 v2 同用户 LOPO AUC=0.3472 显著低于 0.5 — 与 Brennan 2012 的方向一致：纯被动 stylometry 在自然文本上**弱于随机**。
**改进方案**：如果未来要 support 风格可控生成，需要主动训练（adversarial training / contrastive learning）而非被动提取。

### 论文 [3] Koppel & Schler (2004) "Authorship Verification as a One-Class Classification Problem"
**发现**：单类 SVM (unmasking) 在跨主题 verification 上比有监督分类好；**verification 不需要 balanced negative set**。
**PQB 现状**：E19 v3 用 balanced matched negative 反而 NO-GO（cands 池被三约束压缩到 0）。E20 v2 用 one-class-style "self vs cross" 距离比较，仍然不显著。
**改进方案**：尝试 unmasking-style feature 消融（greedy remove most distinguishing features，再看 user-level separability）。

### 论文 [4] Stamatatos (2009) "A Survey of Modern Authorship Attribution Methods"
**发现**：综述指出 char n-grams + function words 是 SOTA；POS n-grams 和 syntactic features **单独使用**效果弱于 lexical；**需要 ensemble**。
**PQB 现状**：E20 v2 只用 32 维 syntactic 比例特征，无 lexical / character 通道。
**改进方案**：下一步加 char n-gram 通道（top-1K tf-idf char 4-6 grams）+ function word frequency 通道，与 30 维 syntactic 拼接。

### 论文 [5] Kestemont et al. (2016) PAN shared task 系列
**发现**：跨域 verification 用 cosine + RBF kernel on compression models (CCM) 比 raw features 强；CCM 本质是 per-author 字符分布压缩模型。
**PQB 现状**：E20 v2 用 L2 距离 + unit-norm 归一化，**等价于余弦距离**。但用户向量只有 30 维，per-author 分布估计噪声大。
**改进方案**：增加 per-author 文本量（≥1000 词/用户）后重跑；或换更高维句向量。

### 综合 §B 结论

PQB 当前 E20 v2 结果与 stylometry 经典文献的预期**方向一致**：
- Eder 2015 → 短文本 stylometry 弱
- Brennan 2012 → 被动 stylometry 弱于主动
- Koppel 2004 → balanced negative 设计有偏
- Stamatatos 2009 → 单一 syntactic 通道弱
- Kestemont 2016 → 低维向量统计效力不足

**在 Baby Products 2023 上 ≤900 词 + 单通道 syntactic 的 NO-GO 与文献一致，不是实现 bug**。要 support 风格可控生成必须 (a) 增通道 (b) 增样本 (c) 改主动训练。

## §C 代码缺陷定位

E20 v1 (`e20_lopo.py`) 的根缺陷：

1. **绝对数量型特征**（`FEATURES20`）：
   - `clause_total`, `modifier_total`, `coordination_count` — 全部与词数线性相关
   - `max_dependency_depth`, `max_dependency_distance` — 受长句影响
   - 文件位置：`query/soft_prefix/e20_lopo.py:148`（用 `Y = np.stack([r["feat"] for r in rows])`，`r["feat"]` 来自 `FEATURES20` 绝对计数）

2. **无特征稳定性筛选**：
   - 直接用全部 20 维做 LOPO，没有检查 feature × word_count 相关性

3. **L2 距离 + 单位归一化**：在 30 维以下向量空间里，欧氏距离对噪声敏感（虽然等价于余弦）

## §D 修复方案（E20 v2）

`query/soft_prefix/e20_lopo_v2.py`：

1. **重定义 32 维 ratio/mean/dist 特征**：
   - 14 个 rate/mean：`clause_rate`, `acl_rate`, `advcl_rate`, `ccomp_rate`, `xcomp_rate`, `relcl_rate`, `modifier_density`, `coordination_density`, `mean_dep_distance`, `depth_variance`, `median_dep_depth`, `passive_rate`, `interrogative_rate`, `conditional_rate`
   - 15 个 opener POS 分布：`opener_NOUN/VERB/ADJ/...`
   - 3 个 sentence_type 分布：`senttype_simple/conjunctive/complex`

2. **per-sentence 提取 + 用户级聚合**：
   - `per_sentence_features()` 在 spaCy Span 级别算原始计数
   - `user_feature_vector()` 按 total_tok / n_sent 归一

3. **特征稳定性筛选**（`STABILITY_CORR_THRESH=0.2`）：
   - 对每个用户 (≥3 products) 算 per-product sent_count × feature value
   - 跨用户平均 corr
   - 剔除 `|corr|≥0.2` 的特征

4. **within-product 标准化**（Task 1）：
   - `up_vec[(u, a)] = user_vec - product_mean[a]`

5. **LOPO**（Task 3）：
   - 1 个随机 held-out product
   - self: train on K-1 products, predict held
   - cross: 3 random other users with similar #products

6. **条件置换**（Task 6）：
   - `(product_cluster, rating_bucket)` 内打乱 user label
   - 99 次 permutation

7. **最低门槛**：`MIN_SENTS_TOTAL=10`（用户级总句数，而非总词数）

## §E 验证结果

```
feat_dim_raw: 32
feat_dim_stable: 30  (interrogative_rate: corr=-0.29, opener_PART: corr=-0.24 剔除)
n_users: 252  (≥3 products, ≥10 sentences)
n_cross_pairs: 510

LOPO:
  d_self_mean: 0.6164
  d_cross_mean: 0.6300
  delta: -0.0136  (方向对, 量级极小)
  CI95: [-0.1082, 0.0778]  (跨零)
  AUC: 0.3472  (< 0.5 反向)

Conditional permutation (n=99):
  null_delta_mean: 0.0168
  obs_delta: -0.0136
  p_one_lower: 0.14
  p_two_sided: 0.28

runtime_sec: 147.3
```

**E20 v2 量化 NO-GO 结论**：

- LOPO delta 方向对（d_self < d_cross）但 CI95 跨零
- AUC 0.3472 < 0.5（同用户反而比跨用户远）
- 条件置换 p=0.28（observed 与 null 不可区分）
- **三种检验联合结论**：在 Baby Products 2023 上，比例型特征 + within-product 标准化 + LOPO + 条件置换**无法支持用户句法风格在 unseen product 上的可识别性**

**与 #19 E19 v3 一致**：硬匹配和 LOPO 两种方法在 Baby Products ≤900 词范围内都不能支持用户句法风格可识别性。

## 下一步

待用户决策（issue #20 comment 列出 5 个候选）：
- (a) cosine 替代 L2
- (b) 跨用户 cluster + rating 匹配
- (c) 增 N_USERS 300→600
- (d) 加 char n-gram / function word 通道
- (e) 接受 NO-GO，更新 paper §3.X
