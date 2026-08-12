# Iteration #45 — §3.3 LLM-Human Evaluation 代码确认

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: 确认 §3.3 LLM-human evaluation (Kappa=0.72, Spearman=0.81) 是否有对应代码实现

---

## §A 搜索结果

### 搜索范围

搜索了以下路径和关键词：

| 关键词 | 搜索路径 | 结果 |
|--------|---------|------|
| `fleiss` / `Fleiss` | 全 codebase `.py` 文件 | 无 |
| `kappa` | 全 codebase `.py` 文件 | 无（仅 import 语句）|
| `human.*score` / `score.*human` | 全 codebase `.py` 文件 | 无 |
| `annotator` | 全 codebase `.py` 文件 | 无 |
| `agreement` | 全 codebase `.py` 文件 | 无 |
| `spearman` | Stage 13 `13_analyze_template_vs_llm.py` | 存在，但用于 template vs LLM retrieval，不是 human-LLM agreement |
| `spearmanr` | Stage 10 `evaluate_review_query_alignment.py` | 存在，但用于 review 风格 vs query 风格 alignment |
| `fleiss` / `kappa` / `human` | 结果目录 `/result/personal_query/` | 无 |

### 结论

**§3.3 的 LLM-human evaluation 没有对应代码实现。**

论文 Table 2 报告的三个关键指标：
- **Fleiss' Kappa = 0.72**（标注者间一致性）
- **Spearman = 0.81**（LLM vs human 相关性）
- **MAE = 0.89**（评分差异）

这三个指标的计算逻辑完全不存在于代码库中。

---

## §B 代码现状分析

### Stage 13 有 relevance scoring，但不是 LLM-human eval

`persona_utils.py` 的 `classify_preference_relevance` 函数用于：
- 分类用户偏好（REQUIRED / RELEVANT / CONFLICTING / IRRELEVANT）
- 评估 product-query relevance
- **不是** query quality 的 LLM-human agreement 评估

### Stage 10 有 Spearman，但不是 LLM-human eval

`evaluate_review_query_alignment.py` 计算：
- `spearmanr(review_values, query_values)` — review 写作风格与 query 风格的 Spearman 相关性
- **不是** LLM judgment vs human judgment 的相关性

---

## §C 审稿意见确认

### [P0 Major] — §3.3 LLM-human eval 代码缺失

**论文位置**: §3.3, Table 2

**问题**: 论文报告了 Fleiss' Kappa=0.72 和 Spearman=0.81，但代码中没有实现这个评估的任何逻辑。

**可能性**:
1. 这个评估是在**代码 pipeline 之外**运行的（独立的 Python notebook 或手动实验）
2. 这个评估使用了**外部数据**（人工标注数据），没有在代码库中维护
3. 这个评估**从未被实现**，数字无法复现

**需要作者回答**:
> "Where is the code that computes Fleiss' Kappa and Spearman correlation for Table 2? Is there a separate repository or notebook that is not in the main codebase?"

---

## §D 本轮代码优化

**无代码改动**（本轮为确认轮）

---

## §E 验证

N/A

---

## §F Git Commit

N/A（审稿确认轮）

---

## §G 结论

§3.3 的 LLM-human evaluation **没有代码实现**，属于以下两种情况之一：
1. 外部实验（独立 notebook / 手动分析）
2. 从未实现，数字无法通过代码验证

**对论文可信度的影响**：Major。审稿人会质疑 Table 2 的数字是否可复现。

**建议在论文中补充**：
- 明确说明 LLM-human evaluation 是在哪个环境/ notebook 中完成的
- 或者提供一个独立的 `stage_XX/human_eval/` 目录，包含完整的 annotation 代码和结果
