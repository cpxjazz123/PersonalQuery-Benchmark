# Iteration #42 — 审稿人视角：§4 Related Work + §5 Conclusion + 全文复查（最终审稿意见汇总）

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: §4 Related Work 缺失 + §5 Conclusion 过于简短 + 全文 Top-3 致命问题汇总

---

## §A 审稿意见（尖锐批评）

---

### 问题 1: Major — Related Work §4 极度单薄，只有 2 句话，文献引用与叙述完全脱节

**论文位置**: §4, line 159-160
**原文引用**:
> "Prior work shows query expression, syntax, and character-level perturbations affect retrieval effectiveness [8-12, 15, 18, 19, 2329, 31, 36, 38, 40]. Yet these factors are usually treated as variation or noise, not user-level habits. PQB fills this gap by evaluating personalized syntactic expression structure and personalized writing errors."

**具体批评**:
Related Work 只有 **2 句话 + 41 个参考文献**，这是顶会论文的 Related Work 最短极限。

**§4 的 Related Work 应该涵盖但完全缺失的主题**：

| 缺失主题 | 现有引用是否相关 | 说明 |
|---------|----------------|------|
| **Personalized retrieval benchmarks** | ❌ 未引用 | 是否有其他 personalized product search benchmark？（[22] JDsearch 存在但未在 Related Work 中讨论） |
| **Query Performance Prediction (QPP)** | ❌ 未讨论 | QPP 预测查询难度，与用户表达风格影响检索的关系密切 |
| **User simulation for IR evaluation** | ❌ 未引用 | 模拟用户行为生成查询的 methodology，benchmark 设计缺少这一块 |
| **Conversational search / session-level IR** | ❌ 未引用 | 用户在多轮搜索中的表达风格变化 |
| **Text complexity / readability in IR** | ❌ 未引用 | 查询复杂度对检索的影响 |
| **Spelling error correction in retrieval** | ✅ 引用 [10, 28, 40] | 但只是罗列，未总结技术路线 |
| **LLM-generated queries for IR** | ✅ 引用 [5, 27] | 但未讨论 LLM 生成查询的保真度问题 |

**更重要的是**：论文引用了 [22] JDsearch: A Personalized Product Search Dataset，但完全未在 Related Work 中讨论。这个数据集与 PQB 最直接相关，但作者只把它当作 [14] 的引用。

**审稿人会问**:
> "Section 4 has only 2 sentences for 41 references. Why are the references not discussed in context? How does PQB differ from JDsearch [22] which is also a personalized e-commerce search dataset? What is the relationship between PQB and prior personalized search benchmarks?"

---

### 问题 2: Major — §5 Conclusion 只有 2 句话，不足以支撑一篇正式论文的 Conclusion

**论文位置**: §5, line 163-164
**原文引用**:
> "PQB evaluates how personalized expression affects product retrieval. Syntactic styles and user writing errors change retriever effectiveness and stability across architectures, so personalized retrieval should consider user needs and expression habits."

**具体批评**:
一个正式论文的 Conclusion 只有 **2 句话，~50 词**，这是极其罕见的。这不是"简洁"，是**缺乏实质内容**。

**Conclusion 应该总结但完全没有的要素**：
1. **核心贡献是什么？**（contribution statement）
2. **主要发现是什么？**（key findings）
3. **实验结论的启示是什么？**（implications）
4. **局限性是什么？**（limitations）
5. **未来工作方向是什么？**（future work）

**论文实际上有大量重要结论**，但全部堆在 §3 正文里：
- SPLADE Δ=9.6 最敏感（§3.2）
- E5 对 spelling error 最敏感（Δ=11.16，§3.2）
- GMM prior 最优（§3.4）
- LLM-human Spearman=0.81（§3.3）

**这些结论没有一个出现在 Conclusion 中。**

---

### 问题 3: Major — 全文缺乏任何 Limitations 声明

**论文位置**: 全文
**具体批评**:
顶会论文通常有专门一节讨论 Limitations。PQB 完全没有。这给审稿人强烈的负面信号：作者是否在回避承认局限性？

**PQB 明显存在的局限性**：
1. **数据集局限性**：只用了 3 个 Amazon 子类，结论能否泛化？
2. **评估指标单一**：只用了 Hit@10，没有 NDCG、MRR 等
3. **Ground truth 问题**：query 的 ground truth product 是 LLM 生成的，不是真实用户点击
4. **Regeneration 机制未完全实现**：论文描述了 10 轮 × 10 个 candidate 的 regeneration，代码只实现了 tracking
5. **User style modeling 假设未验证**：review 风格 = query 风格的假设从未被验证

**审稿人会问**:
> "The paper has no Limitations section. What are the threats to validity of the conclusions? Can the findings generalize beyond Amazon Baby, Grocery, and Pet categories? Is Hit@10 sufficient or are other metrics needed?"

---

## §B 全文 Top-3 致命审稿意见（最终汇总）

经过 iter #38-#42 的全面审稿，以下是论文**最需要 rebuttal 的 3 个问题**：

---

### 致命问题 #1: 核心假设未验证 — "Review writing style = Query search behavior"

**来源**: iter #38 §A 问题 2（Major）
**论文位置**: §1, line 25-27

**假设**: 论文假设用户的 Amazon review 写作风格可以迁移到 query 搜索行为。

**问题**: 这个假设从未被验证。一个习惯写详细 review 的用户，搜索时可能完全使用碎片化 query。

**需要的 rebuttal**: 提供实证证据（或者承认这是设计选择而非已验证假设）。

---

### 致命问题 #2: 统计显著性严重不足 — Δ 值是点估计，无置信区间

**来源**: iter #40-#41 §A 问题（Major）
**论文位置**: §3.1-3.4, Table 1-3

**问题**: Table 1-3 的所有结论（Δ 值、GMM 优势、LLM-human 一致性）都是点估计，没有统计显著性支撑。

**需要的 rebuttal**: 补充 bootstrap CI 或 significance test，或者明确承认 n=3 无法做统计推断。

---

### 致命问题 #3: Regeneration 机制描述与代码不符

**来源**: iter #39 §A 问题（Major）
**论文位置**: §2.2, line 106

**问题**: 论文描述"10 candidate sentences per round for a maximum of 10 iterations"，代码在 Stage 04 和 Stage 10 都只生成 10 个 candidates 后直接拒绝，无 regeneration 循环。

**需要的 rebuttal**: 在代码中实现真正的 regeneration 机制，或者修改论文描述以匹配当前实现。

---

## §C §4 Related Work 改进建议（代码层面无改动，但审稿意见有效）

Related Work 章节应该在论文层面扩展，以下是审稿人建议的文献覆盖：

| 主题 | 建议引用 |
|------|---------|
| Personalized e-commerce search | [22] JDsearch (Liu et al., SIGIR 2023) |
| Query performance prediction | [9] Carmel & Yom-Tov (SIGIR 2010); [11] Datta et al. (SIGIR 2022) |
| User simulation for IR | [27] Sannigrahi et al. (SIGIR 2024); [5] Alaofi et al. (SIGIR 2023) |
| Spelling robustness in dense retrieval | [28] Sidiropoulos & Kanoulas (SIGIR 2022) |
| LLM-generated queries for evaluation | [1] Abe et al. (SIGIR 2025); [5] Alaofi et al. (SIGIR 2023) |
| Query intent / style modeling | [2] Abu Onq et al. (SIGIR 2025); [34] Yetukuri & Khan (SIGIR 2025) |

---

## §D 本轮代码优化

**无代码改动**（本轮为全面审稿总结轮）

---

## §E 验证

N/A（审稿分析轮次）

---

## §F Git Commit

N/A（审稿分析轮次）

---

## §G 全文 Backlog 汇总（iter #38-#42 所有 Major 问题）

| # | 优先级 | 审稿意见 | 所在章节 | 是否需要代码改动 |
|---|--------|---------|---------|----------------|
| 1 | P0 | Review writing style = Query behavior 假设未验证 | §1 | 无（实验设计问题）|
| 2 | P0 | Δ 值无统计显著性（无 CI / significance test） | §3.1-3.4 | 无（论文/实验问题）|
| 3 | P0 | Regeneration 机制描述与代码不符 | §2.2 | 有（Stage 04 需要实现）|
| 4 | P0 | DeepSeek-v4 非独立 retriever，横向比较不公平 | §3.1 | 有（Stage 14 需要说明）|
| 5 | P0 | E5 Δ=11.16 subword 敏感性解释与代码不符 | §3.1 | 有（Stage 05 需要 subword 注入）|
| 6 | P0 | GMM prior 比较无 BIC/AIC 校正，循环论证 | §3.4 | 无（实验设计问题）|
| 7 | P0 | §3.3 LLM-human eval 代码缺失，Kappa/Spearman 无法复现 | §3.3 | 有（确认是否有独立代码）|
| 8 | P0 | Related Work 只有 2 句话，41 参考文献未在正文中讨论 | §4 | 无（论文写作问题）|
| 9 | P0 | Conclusion 只有 2 句话，缺乏贡献/发现/局限性总结 | §5 | 无（论文写作问题）|
| 10 | P0 | 全文无 Limitations 章节 | 全文 | 无（论文结构问题）|
| 11 | P1 | 用户筛选阈值（≥20 reviews）无 ablation | §2.1 | 无（实验设计问题）|
| 12 | P1 | "Personalized expression differences" 操作化定义缺失 | §1 | 无（概念定义问题）|
| 13 | P1 | LLM query generation prompt 完全不透明 | §2.2 | 无（论文披露问题）|

---

## §H 下一轮建议

**如果继续迭代**，建议从以下 P0 代码问题中选择一个实现：

| 优先级 | 行动 | 预计影响 |
|--------|------|---------|
| P0 | 在 Stage 04 `syntax_depth_no_depth_check.py` 中实现真正的 LLM-based regeneration 循环 | 匹配论文 §2.2 描述 |
| P0 | 在 Stage 05 中研究 BPE-aware error injection（模拟 E5 subword 敏感性） | 验证 E5 Δ=11.16 解释 |
| P1 | 确认 §3.3 LLM-human evaluation 的代码位置或外部实现 | 验证 Kappa=0.72 / Spearman=0.81 |
| P1 | 在 Stage 14 rerank_runner 添加多 first-stage retriever 支持 | 改善 DeepSeek-v4 非独立 retriever 问题 |
