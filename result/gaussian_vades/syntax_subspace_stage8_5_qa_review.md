# Stage 8.5 Query Quality Review — 6 维度评分 + 2 个 P0 问题

**Date**: 2026-08-25
**Source**: 综合 Issue 26 的实际 Query 样本 + Stage 8 sanity check + Stage 8.5 shared pool + Mahalanobis selection

---

## 总评

> **目前 Query：实验可用，但还没有达到"高质量自然用户 Query"水平。**

整体打分：**6.5–7/10**

最大问题不再是属性缺失，而是 **模板化、额外语义、以及部分表达不像真实搜索 Query**。

---

## 6 维度评价

### 1. 属性完整性：很好 ✓

Stage 8 检查的 strict Query 都满足：
- 同一个 ASIN 固定相同的 4 个属性
- `attrs_covered == 4`
- 属性值实际出现在 Query
- 没有明显非法标点问题

例：Infant Optics 这个商品固定 (Brand, Style, Item Weight, Main Category)，所有 strict Query 都完整覆盖。

**评价**：Attribute fidelity = 高，这一层基本没问题。

---

### 2. 语义一致性：总体不错，但存在额外语义漂移 ⚠️

虽然 4 个指定属性一致，但实际 Query 中会自己增加没有提供的描述，例如：
- `high-quality`
- `reliable`
- `lightweight`
- `newborn care`

这些词在原始 4 个属性里并不存在。Issue 里的实际样本已可见。

**当前 strict filter 实际验证的是**：
> "4 个属性有没有出现"

**但并没有验证**：
> "除了这 4 个属性之外，有没有增加新的商品需求"

**评价**：核心属性语义固定，但存在少量额外语义漂移。

**影响**：这个问题对 retrieval robustness 实验很重要，因为 `reliable`、`lightweight` 之类的词本身就可能影响 BM25/MiniLM 的检索分数。

---

### 3. 自然度：中等，有明显模板味 ⚠️

大量 Query 都以：
- `Searching for...`
- `Looking for...`
- `I am looking for...`
- `I am searching for...`

开头。Issue 26 的一个 ASIN 下几十条 Query 就大量重复这些 framing。

还有 `Infant Optics brand monitor style baby product ...` 这种表达虽然语法上勉强成立，但**不像真实用户会输入的搜索 Query**。

**评价**：
- Grammaticality: 不错
- Search-query naturalness: **一般**

LLM 生成得"语法正确"，不等于"像真实用户搜索"。这是两个不同问题。

---

### 4. 句法多样性：表面多样，结构多样性中等

从数据上看有明显变化：
- 一个 ASIN：Query 长度从 11 到 64 tokens
- PCA48 D_syn 有明显跨度
- 有短句、长句、不同开头、不同 clause framing

Stage 8 全体 5400 Query D_syn mean ≈ 7.87、std ≈ 3.96、max > 50。

**但**从实际文本看，一部分 diversity 来自：

> `Searching for` → `Looking for` → `I am looking for`

以及句子长度变化。真正像：
- 从句嵌套
- coordination
- relative clause
- modifier attachment
- NP-heavy vs clause-heavy

这种**结构层面的丰富变化**，目前没有文本表面看起来那么强。

**评价**：Measured diversity = 高；真正结构多样性 = 中等

---

### 5. 重复问题：不严重，但值得处理

Issue 26 已经确认存在少量完全重复 Query，不同 `k` 可能生成同样文本。

这不算严重 bug，但对于实验：
> 100 个候选里如果很多只是模板重复，那么所谓 candidate pool size 实际上被高估了。

正式生成建议加：
- normalized string dedup
- token Jaccard dedup
- sentence embedding cosine near-dup

---

### 6. 用户个性化选择：已正常工作 ✓

这部分 Stage 8.5 比 Stage 8 好很多：
- 每个 ASIN 一个共享 K=50 candidate pool
- 每个用户用自己的 PCA48 Gaussian
- Mahalanobis 选最接近的 Query

1000 个 item-user pairs、294 个 unique users，每个用户至少 49 条 review。

选择后 Mahalanobis distance：
- selected: 88.5
- random: 196
- farthest: 598

91.7% 的 selected 比 random 更接近目标用户。

**但**：这一项不能单独证明"生成出来真的像这个用户"，因为本来就是按 Mahalanobis 最小来选。

**真正还缺一个独立验证**：
> 不使用同一个 Mahalanobis objective，再判断 selected Query 是否确实更像目标用户

比如 human / LLM blind evaluation，或者 user-identification rank/margin。

---

## 2 个 P0 问题

### 问题 A：额外语义控制 (Extra Semantic Constraint)

**为什么是 P0**：你的核心 claim 是"同语义、不同 syntax"，但实际 Query 中还存在 `high-quality / reliable / lightweight` 这样的内容变化，这会直接影响 retrieval robustness 的因果解释。

**建议**：

```
除了给定属性，不允许加入新的质量、功能、用途、情感或偏好描述。
```

**实现**：
1. Prompt 强化约束（已部分存在 GEN_SYSTEM_TMPL_NATURAL，但实际效果不足）
2. LLM-as-judge 检查每条 Query：`有没有 unsupported requirement`
3. 维护 blacklist：quality / reliability / use case / emotion 关键词

---

### 问题 B：模板化（Template Saturation）

现在很多句子就是：
> Looking for X... / Searching for X... / I am looking for X...

**对 PCA48 产生差异**但 reviewer 一看样本会想：
> "这到底是 personalized syntax，还是 LLM template sampling？"

**建议**：
- 同一 ASIN 共享候选池中，任意一种 opening template 不超过 15–20%
- 用 rejection sampling：检测 opening template family，超过阈值的 variant 重生成
- 或者 prompt 加 "Avoid starting with 'Looking for' / 'Searching for' / 'I am looking for' / 'I am searching for'"

---

## 总评分卡

| 维度 | 当前评价 | 状态 |
|------|---------|------|
| 4 属性完整性 | 很好 | ✓ |
| 核心语义一致性 | 较好 | ✓ |
| 额外语义控制 | **需要加强** | **P0** |
| 语法正确性 | 较好 | ✓ |
| Search Query 自然度 | 中等 | - |
| 句法多样性 | 中等偏好 | - |
| 模板重复 | 明显存在 | - |
| 用户 Gaussian selection | 很好 | ✓ |
| 独立 personalization validation | 还需要补 | - |

---

## 建议路径

> 现在的数据已经足够支持实验开发和方法验证，但如果准备作为论文正式 benchmark，我不会直接锁定这批 Query。下一步最值得做的不是继续增加数量，而是加强**禁止额外语义 + 去模板化 + 独立自然度/个性化验证**。

优先级：
1. **P0**: 额外语义控制（直接因果解释问题）
2. **P0**: 模板去重（同上 + pool size 真实量）
3. **P1**: 独立 personalization validation（human / LLM blind）
4. **P2**: 更深的结构多样性（从句嵌套、coordination）