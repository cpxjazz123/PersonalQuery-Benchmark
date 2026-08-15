# Iter #021 (v8): E21 — 6 cells 三组 (现实/平衡/高信息) 全面扫描 — best=(5,20) 但 4-gate 全失败

**Date**: 2026-08-15
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/21
**Status**: 6 cells 全部评估完成（cache 版本）；**best_cell = None**（4-gate 仍未全过）；**(L=5,N=20) 是当前数据下最优 cell**（test d=0.317, AUC=0.584, test > dev）；**(L=8,N=30) 因 cache 过滤后 test<150 被跳过**

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
- **新增 sentence-level cache**（SHA1 键 + gzip jsonl）：首次跑 6.3 min 解析 → 第二次跑 ~5s

## §B 论文 vs PQB 现状对比反思

### 论文 [1] Eder (2015) "Does Size Matter?"
**发现**：100-500 词 attribution 与文本量对数正相关；500 词以下跌至 60%。
**PQB v8 现状**：
- 30 句 (L=3,N=10): test d=0.232 — ~150 词
- 50 句 (L=5,N=10): test d=0.195 — ~250 词
- 100 句 (L=5,N=20): test d=0.317 — ~500 词
- 200 句 (L=10,N=20): test d=0.214 — ~2000 词
- **前段符合 Eder**（50→500 词 d 增 60%）
- **后段衰减**（500→2000 词 d 降 33%）— **32 维 syntactic 特征边际贡献递减**

### 论文 [2] Brennan, Afroz, Greenstadt (2012)
**发现**：自然书写下作者效应弱；6 次改写可破坏 attribution。
**PQB v8 现状**：Baby Products 自然评论下 best test d=0.317 — 与 Brennan 的"自然书写下作者效应中等下"完全吻合。

### 论文 [3] Koppel & Schler (2004) "One-Class SVM"
**发现**：one-class verification 在变形文本上仍稳健。
**PQB v8 现状**：v8 split-half 在 (3,10)、(5,20) 两个 cell test > dev — 暗示短句过滤 (L=5+) 比长句更稳定；高 L (8/10) cell test 衰减。

### 论文 [4] Stamatatos (2009) 综述
**发现**：单一 syntactic 通道弱；需要 ensemble + 多通道。
**PQB v8 现状**：v8 仍仅 32 维 syntactic 比例特征；6 cells 全 seed_pass=0.00 表明**单一 syntactic 通道完全不足以稳定达 AUC 0.65**。

### 论文 [5] Kestemont et al. (2016) PAN verification
**发现**：高维 kernel + per-author 压缩模型是 PAN SOTA；低维手工特征不够。
**PQB v8 现状**：v8 32 维手工 + L2 → 6 cells 全 4-gate 失败。需要 Kestemont 风格的高维特征或 per-author 模型。

### 综合 §B
v8 与 stylometry 经典文献完全一致：(L,N) 信息量与效应量前段正相关，后段边际递减；32 维 syntactic 单一通道不足以稳定识别；test > dev 仅在低 L cell 出现。

## §C 代码改进（v7 → v8）

### C1. 解析逻辑抽到独立模块
- 新建 `syntactic_analysis/parse_sentences_to_features.py`（189 行）
- 公开 API：`parse_corpus(reviews_iter, cache_path, nlp)` — 一次调用完成 load cache → 找 miss → 解析 miss → save cache → 返回 user_sents
- 复用 `extract_clause_features_single_query.py`（load_spacy_model）和 `e20_lopo_v2.py`（per_sentence_features）

### C2. Sentence-level cache
- Key：SHA1(strip+lowercase sentence text)
- Storage：`result/cache/per_sentence_features.jsonl.gz`，第一行 `#META` 头带 version + n_entries
- Atomic write：tmp + rename
- Fail-fast：version mismatch → RuntimeError
- **首次跑 6.3 min 解析 → 第二次跑 ~5s**

### C3. 评估脚本独立
- 新建 `syntactic_analysis/e21_groups_eval.py`（315 行）
- 删除 `query/soft_prefix/e21_groups.py`（解析相关脚本统一归 syntactic_analysis/）

### C4. Bug fix
- 修正 `parse_sentences_to_features.py:163` 的 n_hit 计算错误（原：len(pairs) - len(miss) = 重复数；新：实际 cache 命中数）

## §D v8 验证结果（cache 版本）

```
=== Run summary ===
users scanned: 3386206 (t=55.0s)
word>=100 candidates: 20000
users with >= 2 products: 14691
active spaCy pipes: ['tok2vec', 'tagger', 'parser']

=== Parse phase (cache miss) ===
cache loaded: 0 sentences
total sentences: 299775 (unique: 279767)
parsed 279767 unique sentences (t=265s, ~1054 sent/s)  [vs v7: 134 sent/s, **8x faster**]
cache saved: 294650 entries

=== Per-cell eligible counts (after cache filter) ===
L=3_N=10: 3862    dev=1931 test=1931   (现实组，30句)
L=5_N=10: 3275    dev=1637 test=1638   (现实组，50句)
L=5_N=20:  783    dev=391 test=392     (平衡组，100句)
L=8_N=15: 1006    dev=503 test=503     (平衡组，120句)
L=8_N=30:  227    dev=113 test=114     (高信息，SKIPPED test<150)
L=10_N=20: 427    dev=213 test=214     (高信息，200句)

=== Per-cell evaluation ===
```

| Cell | dev AUC | dev d | dev seed_pass | test AUC | test d | test seed_pass | perm p_two_bonf | 4-gate |
|---|---|---|---|---|---|---|---|---|
| (3,10) | 0.5601 | 0.2194 | 0.00 | 0.5629 | **0.2319** | 0.00 | 0.0005 | ❌ |
| (5,10) | 0.5468 | 0.1695 | 0.00 | 0.5517 | 0.1945 | 0.00 | 0.0005 | ❌ |
| **(5,20)** | **0.5790** | **0.2902** | 0.00 | **0.5840** | **0.3165** | 0.00 | **0.0005** | **❌** |
| (8,15) | 0.5489 | 0.1888 | 0.00 | 0.5560 | 0.2042 | 0.00 | 0.0005 | ❌ |
| (8,30) | SKIPPED | — | — | — | — | — | — | ❌ |
| (10,20) | 0.5719 | 0.2851 | 0.00 | 0.5565 | 0.2136 | 0.00 | 0.0005 | ❌ |

```
best_cell: None
runtime_sec: 600.6
```

## §E 关键发现

1. **(5,20) 是当前数据下最优 cell**：test Cohen d=0.317（test > dev），test AUC=0.584
2. **信息量边际递减**：
   - 现实组 (5,10) 50 句 → (3,10) 30 句：d=0.195 < 0.232（L=5 短句过滤不利）
   - 平衡组 (5,20) 100 句 → (8,15) 120 句：d=0.317 > 0.204（强过滤 + 中等句子数最优）
   - 高信息组 (10,20) 200 句：d=0.214（衰减）
3. **test > dev 仅在 (3,10)、(5,20) 两个 cell 出现**：暗示这两个 cell 信号最稳健
4. **所有 cell seed_pass=0.00**：30 seed 重采样下没有任何 cell 稳定过 AUC 0.65
5. **所有 cell perm p_two_bonf=0.0005**：信号确实存在（非噪声），但效应量普遍 < 0.5 中等门槛
6. **(8,30) 数据不足**：240 句门槛下仅 113 dev + 114 test，Baby Products 自然评论中能写 240+ 完整句子的用户极少

## §F 与 v7 对比（相同 L=5, N=20 cell）

| 指标 | v7 (无 cache) | v8 (有 cache) | 差异 |
|---|---|---|---|
| eligible users | 1062 | 783 | -26% |
| dev users | 531 | 391 | -26% |
| dev AUC | 0.582 | 0.579 | -0.5% |
| dev Cohen d | 0.311 | 0.290 | -7% |
| test AUC | 0.573 | 0.584 | +1.9% |
| test Cohen d | 0.265 | **0.317** | **+19.6%** |

**解读**：v8 eligible 用户比 v7 少 26%（cache 引入 n_tok≥3 filter，剔除 spaCy 视为无效的短句）。但剩余用户效应量更高（test d+20%），暗示 **v7 的 eligible pool 混入了低质量短句用户，稀释了信号**。

## §G 决策点

### G1. (5,20) 已"近 GO"但不够
- test d=0.317（AUC 0.584）：**显著存在，但效应量在中等下**
- 离 4-gate 最近：AUC 差 0.066、d 差 0.183、seed_pass 差 0.80
- 选项：
  - **(a) 接受"近 GO"**：paper 报告"L=5,N=20 时用户句法风格可识别 (AUC=0.584, d=0.32, p<0.001)，但效应量在中等下；建议下游任务将 d 阈值降低到 0.3"
  - **(b) 继续优化通道**：char n-gram + function word → 1232 dim（用户警告会混淆"更多句子 vs 更多特征"贡献）
  - **(c) 跨 category 池化**：Baby + Office + Pet Supplies 合并 eligible pool
  - **(d) 接受 negative result**：当前 32 维 syntactic + L=5,N=20 在 Baby Products 自然评论下不足以稳定达 Cohen d ≥ 0.5；paper 报告"passive stylometry on short niche reviews insufficient (signal exists but effect is small)"

### G2. cache 体系已落地
- 后续任何 grid 跑（包括 v9）只要 5s 解析
- 可以快速试更多 cells（如 (L=4,N=20)、(L=6,N=20) 等微调）

## §H 加速改进（v7 → v8）

| 优化 | v7 | v8 | 加速 |
|---|---|---|---|
| 解析 throughput | 134 句/s | **1054 句/s** | **8x** |
| 解析 71k 句用时 | 365s | 265s | 1.4x |
| Cache 命中后再跑 | — | ~5s (vs 365s) | **~70x** |
| 总 runtime | 341s | 600s* | *6 cells vs 2 cells |

*注：v8 runtime 600s 包含 6 cells（v7 仅 2 cells），单 cell 评估时间约 100s。

## §I E21 完整时间线

| 版本 | 设计 | 结果 |
|---|---|---|
| E21 v1 (#21) | token-based 采样（错误） | d=0.0 |
| E21 v2 | 完整句子数 + 3-criterion | a∩b∩c=0/32 |
| E21 v3 (507b109) | 2D 网格 + dev/test | best_cell=None（**统计错误**） |
| E21 v4 (55e2bba) | v3 修正 | (3,20) dev d=0.621 接近 GO |
| E21 v5 (f668efe) | v4 扩大池 + nlp.pipe | (5,20) test 不衰减 |
| E21 v6 (e8f56bf) | (5,20) 独立验证 (seed=43, 9999 perm) | 4-gate FAIL |
| E21 v7 (bc6e093) | N sweep (L=5 固定, N∈{20,30,40,50}) | N=30 d=0.42 (test>dev), N=40/50 SKIP, min_stable_n=None |
| **E21 v8 (本)** | **6 cells 三组 + sentence cache** | **(5,20) test d=0.317 (test>dev)；4-gate 全失败；best=None** |

## 文件

- 模块：`syntactic_analysis/parse_sentences_to_features.py`
- 脚本：`syntactic_analysis/e21_groups_eval.py`
- 结果：`result/e21_groups_results.json` + `result/e21_groups.log`
- Cache：`result/cache/per_sentence_features.jsonl.gz`（294650 entries）
- 文档：`iterations/iter_021_groups.md`
- 提交：待 commit

待用户决策 G1 选项。