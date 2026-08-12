# Iteration #38 — 审稿人视角：论文 §1 Introduction + §2.2 方法论批判

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人（顶会级别）
**scope**: §1 Introduction 动机声明 + §2.2 GMM/VAE 方法论缺陷

---

## §A 审稿意见（尖锐批评）

---

### 问题 1: Major — 用户筛选标准（≥20 reviews, ≥15 words）完全缺乏消融实验支撑

**论文位置**: §2.1, line 46
**原文引用**:
> "To ensure the reliability of user profiles, PQB retains only users with at least 20 historical reviews and requires each review to contain at least 15 words."

**具体批评**:
这是论文最严重的方法论缺陷之一。**为什么恰好是 20 条 reviews？为什么每条 review 必须 ≥15 词？论文完全没有给出选择这个阈值的依据。**

这不是一个可以轻易忽略的细节——这个阈值直接决定了：
1. 最终数据集的规模（26,609 queries 的覆盖面）
2. 可用的用户数量（影响统计显著性）
3. GMM 训练时每个用户的样本量是否足够支持 2-component mixture

**一个合理的审稿问题**：
> "Reviewer asks: Why exactly 20 reviews and 15 words? Was any ablation study conducted to validate these thresholds? How sensitive are the final conclusions (e.g., SPLADE Δ=9.6) to changing this threshold to 15 or 30 reviews?"

**代码实现**: Stage 04 `syntax_depth_no_depth_check.py` line 308 直接硬编码 `NUM_USERS_TO_TEST`，但 Stage 02 的用户筛选阈值在代码中没有可配置的参数。

**证据支撑**:
- 论文没有报告任何关于此阈值的 ablation study
- 26,609 queries ÷ 3 domains = ~8,870 per domain——这个规模是否稳定在这个阈值下？
- 如果阈值变成 15 或 30，GMM components 是否仍然合理？

---

### 问题 2: Major — "Review writing style = Query behavior" 核心假设从未被验证

**论文位置**: §1, line 25-27
**原文引用**:
> "PQB is built on Amazon reviews and product metadata, extracts expression signals from users' historical reviews, and generates personalized queries aligned with product attributes and user styles."

**具体批评**:
论文的**整个方法论建立在一个从未被验证的假设之上**：用户的 Amazon review 写作风格可以迁移到用户的 query 搜索行为。

这是一个极其危险的假设：
- **Review 写作**：用户有充足时间组织句子，使用复杂句法，考虑产品各方面
- **Query 搜索**：用户通常是简短、碎片化的输入，追求快速找到目标产品

**没有任何证据表明**表达风格在这两个场景中是可迁移的。一个习惯写深度分析 review 的用户，搜索时可能完全使用"bottle nipple 3month"这样的碎片化 query。

**审稿人会问**：
> "What is the empirical evidence that review writing style correlates with query formulation behavior? This is the foundational assumption of the entire paper, yet no validation is provided. Without this, the entire pipeline (GMM style modeling + query generation) may be modeling the wrong behavior."

---

### 问题 3: Major — GMM prior "best fits" 结论是循环论证

**论文位置**: §1, line 33 + §3.4
**原文引用**:
> "The Section 3.4 experiment shows that this prior best fits the real distribution of user sentences."

**具体批评**:
这是一个**循环论证**：
1. 论文用 GMM 去 fit 用户句子的 syntactic features 分布
2. 然后在 §3.4 比较 GMM vs t-distribution vs Laplace vs Logistic
3. 结论：GMM "best fits"

但这个比较的本质是：**GMM 和 t-distribution/Laplace/Logistic 都是在 fit 同一个数据集，只是参数化方式不同**。GMM 赢是因为它有更多的自由参数（2-component mixture），而不是因为它真的更"正确"。

更严重的问题是：**没有任何 "ground truth" 分布来验证谁更正确**。论文只是比较了不同模型对同一数据的拟合优度（likelihood/BIC/AIC），这不能证明 GMM 更"真"。

**审稿人会问**：
> "The comparison in Section 3.4 compares GMM against t-distribution/Laplace/Logistic ALL FITTING THE SAME DATA. GMM winning (with more free parameters via 2-component mixture) is expected — this is not evidence of 'truth'. What is the gold-standard ground truth distribution that validates GMM's superiority?"

---

### 问题 4: Major — "Personalized expression differences" 定义不严谨，操作化定义缺失

**论文位置**: §1, line 25
**原文引用**:
> "personalized syntactic expression structure, which refers to how users organize product attributes, supplementary conditions, and modifiers"

**具体批评**:
论文声称定义了"personalized expression differences"，但这个定义完全是**描述性/直觉性的**，没有任何**可操作的定义**：
- "product attributes" —— 如何定义/提取？
- "supplementary conditions" —— 什么算 supplementary？状语？从句？
- "modifiers" —— 形容词/副词？

更重要的是，**论文声称表达风格在用户间有"stable individual differences"（引用 [2, 5]）**，但：
1. 引用 [2, 5] 的具体内容在附录中没有详细说明
2. 这种"稳定性"是否在 Amazon review 这个 domain 同样成立？Amazon 用户的 review 写作风格和 query 风格是否同样稳定？

**审稿人会问**：
> "How are 'product attributes', 'supplementary conditions', and 'modifiers' operationally defined? What is the feature extraction pipeline? The paper claims 'expression patterns exhibit stable individual differences' but this is asserted, not demonstrated on the Amazon review domain."

---

### 问题 5: Minor — LLM 生成 query 的 prompt 完全不透明

**论文位置**: §2.2, line 50
**原文引用**:
> "PQB uses an LLM to generate candidate queries containing five target product attribute values."

**具体批评**:
论文完全没有披露：
- 使用的是哪个 LLM？（GPT-4? Claude? 自己的模型？）
- Temperature 设置是多少？
- 系统 prompt 是什么？
- 如何保证 LLM 生成的 query 在风格上是"真实"的用户表达？

在 NLP/IR 领域，**LLM 生成的数据集**一直受到"是否真实反映人类行为"的质疑。论文声称 query 是"personalized"的，但实际上是 LLM 生成的——这与"personalized expression"的声称存在根本矛盾。

**审稿人会问**：
> "Since all queries are LLM-generated (not real user queries), how does this dataset evaluate 'personalized expression differences' in actual user behavior? LLM-generated queries may not reflect genuine human query formulation patterns."

---

### 问题 6: Minor — 26,609 queries 的规模声称有误导性

**论文位置**: §1, line 31
**原文引用**:
> "PQB covers three product domains and contains 26,609 personalized queries."

**具体批评**:
26,609 queries 听起来很多，但分解一下：
- 3 个 Amazon domains
- Baby_Products + Grocery_and_Gourmet_Food + Pet_Supplies

实际每个 domain 的 queries 数量约为 **8,870 queries per domain**。这个规模在真实检索系统中并不算大。而且，**queries 的分布是否均匀**？某些用户/产品类别是否有大量冗余？

更重要的是，如果每个用户有 20+ reviews，但最终只有 26,609 queries，这意味着**大量用户的 reviews 被丢弃**——论文没有说明有多少用户被筛选掉了，为什么。

---

## §B 对应代码缺陷

| 审稿问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1: 筛选阈值无依据 | `02_writing_analysis/common/extract_errors_common.py` | 无可配置的 review count/word count 阈值 |
| 问题1: 筛选阈值无依据 | `04_query/common/syntax_depth_no_depth_check.py:307` | `NUM_USERS_TO_TEST` 硬编码，无阈值消融 |
| 问题2: 假设未验证 | `04_query/common/syntax_depth_no_depth_check.py` | 假设 review 风格 = query 风格，无任何验证 |
| 问题3: 循环论证 | `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py` | GMM vs other distributions 比较在同一数据集上，无 ground truth |
| 问题5: LLM prompt 不透明 | `04_query/common/syntax_depth_no_depth_check.py:68-118` | `_system_base()` prompt 未文档化，LLM 选择/参数未披露 |

---

## §C 本轮代码优化

**无代码改动**（本轮为审稿分析轮）

---

## §D 验证

N/A

---

## §E Git Commit

N/A（审稿分析轮次）

---

## §F 最需要 rebuttal 的审稿意见

**Top 3 最致命的审稿意见**（按严重程度排序）：

1. **[Major] Review writing style ≠ Query behavior 假设未验证**
   - 这是整个论文的 foundation assumption，必须有实验证据
   - 需要设计一个 validation study：比较用户的 review 风格和 query 风格是否真的相关

2. **[Major] GMM prior 结论是循环论证**
   - 必须重新设计实验：与一个**完全不使用 GMM 的 baseline** 对比，而不是只比较不同的 prior 分布
   - 或者承认这不是"最佳拟合"而是"在约束下的最优"

3. **[Major] 用户筛选阈值（≥20 reviews, ≥15 words）无依据**
   - 需要 ablation study 验证阈值敏感性
   - 或者改为可配置参数，在实验中报告不同阈值下的结果

---

**下一步**: 进入 iter #39，以这 3 个 Major 审稿意见为依据，设计代码改进方案。
