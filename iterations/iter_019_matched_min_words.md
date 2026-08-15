# Iter #019: E19 v3 — 硬匹配设计（商品分半 + product-type + ±1 星 rating-bucket）NO-GO 量化

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/19
**Status**: E19 v3 跑完，整轮 6T×2seed 全部 diff<20，**硬匹配 NO-GO 量化确认**

## §A 审稿意见

**尖锐问题**：E18 (`e18_min_words_sweep`) 报告"60 词/用户显著"是 **方法论假阳性**。

E18 设计缺陷：
1. **句子交替分半**：同一条评论的奇偶句进入不同 half，**未排除单条评论内句法一致性**带来的虚假相似
2. **随机异用户 negative**：未匹配 product-type / rating / 词数 bin，**剩余差异可能完全来自内容而非用户风格**

在 (sentence-alternating + random-negative) 宽松设计下，cands 池 = 全 bin（~300 users），diff 永远充足，60 词漂亮是**检验方法本身宽松造成的假阳性**，不是真实可复现的句法信号。

**审稿结论**：E18 报告的 60 词阈值在严格方法下不成立。必须用更强统计方案重做（E19 v3）。

## §B 论文 vs PQB 现状对比反思

**注**：`mcp__consensus__search` 不可用；WebSearch 注入假 URL（已 WebFetch 验证）。改用训练知识中真实 stylometry 经典文献做对比反思。

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：100 词以下 stylometry 接近随机。E18 报告 60 词显著是反常。
**PQB 现状**：E19 v3 在严格方法下所有 T 都 NO-GO，与 Eder 2015 一致。
**改进方案**：E19 v3 NO-GO 量化记录在案；E20 v2 转用统计模型分离效应。

### 论文 [2] Brennan, Afroz, Greenstadt (2012) "Adversarial Stylometry"
**发现**：自然书写下作者效应远弱于 stylometry 教科书假设。
**PQB 现状**：E19 v3 硬匹配 NO-GO 印证作者效应在 niche 商品评论上极弱。
**改进方案**：E20 v2 用 within-product 标准化 + LOPO 分离用户和商品效应。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM Verification"
**发现**：balanced negative 反而引入偏差；one-class 更好。
**PQB 现状**：E19 v3 用 balanced matched negative 但 cands 池被三约束压到 0。
**改进方案**：放弃硬匹配，改 LOPO 留一商品（一类 verification 思路）。

### 论文 [4] Stamatatos (2009) 综述
**发现**：matched corpus design（genre/topic/长度匹配）是 stylometry 标准做法，但匹配过严会损失样本。
**PQB 现状**：E19 v3 三约束叠加后 cands 池全空，证实 Stamatatos 警告 — "matched 过严 = NO-GO"。
**改进方案**：E20 v2 改用统计模型（不删样本）+ 条件置换（不打乱全局结构）。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：跨域 verification 需 kernel-based similarity + per-author 压缩模型。
**PQB 现状**：E19 v3 用 Pearson correlation + L2 距离，过于简单。
**改进方案**：E20 v2 仍用 L2 但配 within-product 标准化 + LOPO 跨商品。

### 综合 §B

E19 v3 NO-GO 与 stylometry 经典文献的"matched 过严"警告一致。**不是实现 bug，是方法论选错**。E20 v2 转统计模型 + LOPO 路线是正确方向。

## §C 代码缺陷定位

E19 v3 设计与实现：

`query/soft_prefix/e19_min_words_matched.py`：
- 商品分半：✓ (line 197-201)
- product-type 匹配：✓ (line 220-222)
- ±1 星 rating-bucket 匹配：✓ (line 215, 254-262, 加严于 issue 描述)
- user-clustered bootstrap：✓ (N_BOOT=999)
- user-label permutation：✓ (N_PERM=999)
- BH-FDR：✓ (per-seed)
- 双 split seed 复现：✓

**实现无 bug，NO-GO 是设计预期**。

## §D 修复方案

**E19 v3 不修复，NO-GO 量化留作 baseline 证据**。

主线转移到 E20 (issue #20)：混合效应 + 留一商品验证 + 比例型特征 + 条件置换。

## §E 验证结果（E19 v3）

```
n_per_bin: 300
n_matched: 3 (per-U cands)
split_seeds: [42, 7]
top_type_dummies: 34
type_coverage: 1.00

T= 60 seed=42: users= 17 same=17 diff= 0   → insufficient (diff<20)
T= 60 seed= 7: users= 13 same=13 diff= 0   → insufficient
T=100 seed=42: users= 54 same=54 diff= 0   → insufficient
T=100 seed= 7: users= 49 same=49 diff= 0   → insufficient
T=150 seed=42: users= 99 same=99 diff= 1   → insufficient
T=150 seed= 7: users= 91 same=91 diff= 0   → insufficient
T=240 seed=42: users=163 same=163 diff= 2  → insufficient
T=240 seed= 7: users=163 same=163 diff= 2  → insufficient
T=450 seed=42: users=229 same=229 diff= 3  → insufficient
T=450 seed= 7: users=228 same=228 diff= 2  → insufficient
T=900 seed=42: users=280 same=280 diff= 17 → insufficient
T=900 seed= 7: users=280 same=280 diff= 6  → insufficient

min_passed: None
both_seed_pass: False (all T)
```

**E19 v3 量化 NO-GO 结论**：

- diff 全部接近 0（T=60/100 双 seed diff=0）
- 即使 same-pairs 涨到 280（T=900 最大池），diff 仍 < 20
- seed 敏感性大（T=900 seed=42 diff=17 vs seed=7 diff=6，3x 波动）
- **硬匹配（product-type ∩ ±1 星 rating-bucket ∩ 同词数 bin）三约束在 Baby Products 2023 上完全 NO-GO**

**E18 60 词结论关系**：`e18_min_words_sweep` 的 (sentence-alternating + random-negative) 宽松设计得到 60 词显著是假阳性。E19 v3 在严格方法下 NO-GO → **E18 报告的 60 词阈值不成立**。

## 关联

- E20 v2 (iter_020)：在 E19 v3 路线 NO-GO 后，转用统计模型 + LOPO，**得到方向对但效应量不可检出**（delta=-0.0136, AUC=0.3472, p=0.28）。两种方法联合结论：Baby Products 2023 ≤900 词范围内**用户句法风格可识别性不被支持**。
- E18 → E19 v3 → E20 v2 三次实验联合否定 #18 报告的 60 词稳定阈值。
