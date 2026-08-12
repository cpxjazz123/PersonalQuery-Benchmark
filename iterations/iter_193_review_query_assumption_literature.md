# Iteration 193 — P0 Review ≠ Query Assumption: Literature Evidence from Consensus MCP

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: P0 #3 Review writing style ≠ Query behavior 假设 — 搜索相关论文验证

## §A 审稿意见（来自 iter #38）

### 问题: Review writing style ≠ Query behavior 假设未验证

**P0 claim**: "用户历史评论的 writing style 可以预测其在 query 中的 expression behavior"

- PQB 用 GMM 从用户历史 review 提取 expression style
- 假设这个 style 会映射到 query 的 syntactic structure
- **但从未验证过：review style 是否真的等于 query behavior？**

## §B 参考文献（Consensus MCP search，规则 7）

### 关键发现 1：Writing Style 影响检索一致性

- **[Writing Style Matters: An Examination of Bias and Fairness in Information Retrieval Systems](https://consensus.app/papers/details/55359ff1f98d5d528bd1b173e8398aed/?utm_source=claude_code)** [1] — Cao, 2024, WSDM, 14 citations. 发现 embedding models 对 query writing style 有偏好，会 match query style with retrieved document style [1]。这说明 writing style 确实影响 retrieval behavior，支持"review style → query style"映射的部分假设。

### 关键发现 2：Search + Writing Behavior 相互关联

- **[Predicting essay quality from search and writing behavior](https://consensus.app/papers/details/599c85bf277e54b5a26b2f529c8d3a82/?utm_source=claude_code)** [2] — Vakkari et al., 2021, J. Association for IS&T, 7 citations. 用 path analysis 发现 search process 对 essay quality 有显著贡献（direct + mediated effects）。更关键的是：不同 writing strategies（boil-down vs build-up）产生不同的 search behavior，说明 **writing behavior 和 search behavior 是 harmonize 的**[2]。这直接支持了"用户写作风格会影响其 query 行为"的假设。

### 关键发现 3：用户认知风格影响 Query Reformulation

- **[Modeling users' web search behavior and their cognitive styles](https://consensus.app/papers/details/b4bc73bf94585960891fbc3ba5001075/?utm_source=claude_code)** [3] — Kinley et al., 2014, J. Association for IS&T, 57 citations. 发现 cognitive styles 显著影响 query reformulation behavior 和 information-processing approaches [3]。说明用户稳定的个人特征会映射到 query 行为层面。

### 关键发现 4：Query Reformulation 可从历史预测

- **[Learning user reformulation behavior for query auto-completion](https://consensus.app/papers/details/110f2d4b3364542294e7621d228beef5/?utm_source=claude_code)** [4] — Jiang et al., SIGIR 2014, 113 citations. 提出用 term-level、query-level、session-level features 学习 user reformulation behavior，显著提升 prediction accuracy [4]。说明 query 行为有可预测的结构模式，与用户历史文本相关。

## §C 结论

**论文 [1-4] 总体上支持 PQB 的核心假设**（review style → query behavior），但也揭示了重要的 nuance：

| 论文 | 核心结论 | 对 PQB 的启示 |
|------|---------|--------------|
| [1] | Embedding models match query style with document style | 检索系统对 style 敏感，PQB 的 Δ Range 有意义 |
| [2] | Writing strategy 与 search behavior harmonize | review style → query behavior 映射有实证支撑 [2] |
| [3] | Cognitive style → query reformulation pattern | 个人稳定特征映射到 query 行为 [3] |
| [4] | Query reformulation behavior 可从历史预测 | 用户历史文本确实包含预测 query 行为的信号 [4] |

**但关键问题仍未解决**：以上研究都是西方/英文语境，PQB 的 GMM 是对 syntactic depth/clause structure 建模，与这些论文的 writing style（informal/formal、emotive 等）维度不同。PQB 需要验证的是：**syntactic structure style（dependency depth、modifier density 等）是否影响 retrieval behavior**，而不是 general writing style。

**iter #184 framework (Kendall τ + bootstrap CI + permutation test) 已完成**，但 end-to-end pilot 需要 Stage 1+6 真实数据 lineage。

## §D 后续 iter

- **iter #194**: 运行 iter #184 pilot 在真实 Stage 1+6 数据上，验证 review syntactic depth 与 query behavior 的相关性（blocked on Stage 1/6 data lineage）

## §E 更新 loop.md §11

- P0 #3 条目加入论文 [1-4] 引用作为"假设有 literature 支撑"的证据
- 同时标注：现有 literature 验证的是 general writing style vs query behavior，PQB 的 syntactic depth/clause structure 维度仍需专项验证
