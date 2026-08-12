# Iteration #37 — Stage 13 LLM evaluation prompt 与论文一致性检查

**日期**: 2026-07-20
**scope**: Stage 13 LLM evaluation prompt 与论文 Q3 一致性检查
**关联 stage**: 13_query_template

## §A 论文发现

### §3.3 Q3: LLM vs Human 评估一致性

论文关键结论：
- Spearman = 0.81
- MAE = 0.89

论文使用 LLM 评估 query quality，判断 query 是否在语义/句法上符合用户意图。Stage 13 的主要功能是 template noisy query 的 retrieval evaluation（通过 `eval_template_driver.py` 调用 08-stage 的评估逻辑），不是 query quality 的 LLM 评估。

## §B 代码审查：LLM evaluation prompt

### §B.1 `persona_utils.py` — Relevance Scoring Prompt

Stage 13 没有直接的 template noisy query quality 评估。真正包含 LLM evaluation prompt 的是 `06_retrieval/utils/persona_utils.py` `build_improved_prompt()`：

**Prompt 结构**：
```
You are an expert search relevance evaluator. Score how RELEVANT a product is
to a user query on 0.0-1.0 scale.

Scoring Rules:
- 0.8-1.0: Core requirements met (brand + category + compatibility + main item type)
- 0.5-0.7: Most core requirements met
- 0.3-0.5: Some core requirements met
- 0.0-0.3: Few or no core requirements met
```

**评估对象**：Query vs Product relevance（relevance scoring），不是 query quality 评估。

**Persona context**：可选的用户偏好上下文（classified by REQUIRED/RELEVANT/CONFLICTING/IRRELEVANT）。

### §B.2 问题：论文描述的 LLM 评估 vs 代码实现

论文的 Q3 LLM evaluation：
- **论文描述**：评估生成的 noisy query 是否在语义/句法上符合用户意图
- **代码实现**：`build_improved_prompt()` 评估的是 **retrieved product 对 query 的 relevance**
- **差距**：代码的 prompt 不评估 query 本身的质量，只评估 product-query match

这意味着论文中 Spearman=0.81, MAE=0.89 的 LLM evaluation **不在 Stage 13 的 pipeline 中**。该评估可能是：
1. 独立运行的脚本，不在主 pipeline 中
2. 使用了与代码中不同的 prompt
3. 或者根本没有在代码中实现

### §B.3 Prompt 质量分析

`build_improved_prompt()` 的 prompt 设计：

| 维度 | 评估 |
|------|------|
| 评分刻度 | 0.0-1.0（连续）✅ |
| 评分规则 | 有明确分层 ✅ |
| Persona context | 支持用户偏好分类 ✅ |
| 输出格式 | 要求 "Final Score: X.X" ✅ |
| Syntactic quality | ❌ 不评估 |
| Query 自然度 | ❌ 不评估 |
| 噪声影响 | ❌ 不评估 |

**缺失的评估维度**（对应论文 Q3）：
- Query 的 syntactic quality（语法正确性）
- Query 的 semantic alignment（是否表达用户意图）
- Noisy query 相比 clean query 的 quality degradation

## §C 本轮已实施改动

无（本轮为分析）

## §D 总结

| 检查项 | 状态 | 备注 |
|--------|------|------|
| Stage 13 有 LLM evaluation prompt | ⚠️ | 有 relevance scoring prompt，但用于 retrieval，非 query quality |
| Prompt 评估 query quality | ❌ | 不评估 query 本身质量 |
| 论文 Q3 LLM eval 在 pipeline 中 | ❌ | 可能独立运行或未实现 |
| persona_utils prompt 质量 | ✅ | 有结构化评分规则、persona context |

## §E 下轮建议

- 确认论文的 Q3 LLM evaluation 是在代码中实现还是独立运行
- 如果需要实现 query quality LLM evaluation，需要新增 prompt 评估 syntactic quality、semantic alignment 等维度
- 当前的 `persona_utils.py` prompt 专注于 product-query relevance，适合 retrieval evaluation 但不适合 query quality 评估
