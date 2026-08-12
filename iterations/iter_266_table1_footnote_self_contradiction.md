# Iteration 266 — Table 1 iter #204 Footnote 内部矛盾：query-level vs cluster-mean aggregation

**日期**: 2026-07-22
**角色**: NLP/IR 专业审稿人
**scope**: Paper line 139 iter #204 footnote 内部矛盾——声称 lower-panel Δ Range 使用 query-level aggregation，但同一条 footnote 承认 pipeline 实现是 cluster-mean level approximation

---

## §A 审稿意见（nature-reviewer 评估）

### 问题 1: iter #204 footnote 内部逻辑矛盾 — Major
**严重程度**: Major
**具体批评**:

> "iter #204 resolution: The lower-panel Δ Range is computed at the **query level** rather than at the cluster-mean level... the release pipeline's `08_compare_p10_across_domains.py:_compute_error_effect_delta_per_retriever` implements this formula at the **cluster-mean level**"

line 139 footnote 同时声称两个互斥的命题：
1. "lower-panel Δ Range 使用 query-level aggregation"（声称）
2. "pipeline 实现用的是 cluster-mean level"（承认）

这两个aggregation方法数学上不等价，会产生不同的 Δ 值。

### 问题 2: "originally-released full Stage-6 query pool" 来源不明 — Major
**严重程度**: Major
**具体批评**:

> "The Δ Range values reported above... were computed on the originally-released full Stage-6 query pool (≈90 queries per category)" (line 133)

该 query pool 在 codebase 中从未被定位到。Stage 06 当前 pipeline 仅产生 2 queries/cell，8-cluster stratified analysis 无法运行。

### 问题 3: Upper panel Δ Range reproducibility 声称不成立
**严重程度**: Major
**具体批评**:

> "The first [panel]... computed as the highest cluster Hit@10 minus the lowest cluster Hit@10 within each domain" (line 135)

Stage 06 当前只有 2 queries/cell，max(C1..C8)−min(C1..C8) per domain 的 8-cluster 统计量无法计算。

---

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| Lower panel query-level vs cluster-mean 矛盾 | `08_compare_p10_across_domains.py:_compute_error_effect_delta_per_retriever` | 函数实现 cluster-mean aggregation，但 iter #204 footnote 声称 query-level |
| Table 1 Δ 值无法从当前 pipeline 复现 | Stage 06 retrieval pivot 输出 (all_query_records=[] for all retrievers) | 2 queries/cell 导致 degenerate 输出，Table 1 数值依赖未发布的 90-query pool |
| Footnote 自我矛盾 | `PersonalQuery-Benchmark_evaluating_retrieval.md:139` | "iter #204 resolution"声称 query-level 但承认 pipeline 是 cluster-mean |

---

## §C 本轮代码优化

无代码修改（问题在 paper footnote 文本，不在代码）。

**Paper footnote 修复方案**：line 139 footnote 需要二选一：
- 方案A（诚实）：承认 pipeline 实现是 cluster-mean approximation，retract query-level 声称
- 方案B（修正）：修改 pipeline 为 query-level aggregation，重新跑 Stage 09，update Table 1 数值

---

## §D 验证

- 无代码修改，无需 py_compile 验证
- Paper footnote 矛盾需人工决定采用方案A还是方案B

---

## §E 下一步

**needs input**: 作者选择方案A（诚实路线：retract query-level 声称）还是方案B（修正pipeline，重新跑 Stage 06+09）？

---

## §F nature-reviewer 评估摘要

3位独立审稿人共识：
1. **R1-M1/R2-M1/R3-M1 共识**：iter #204 footnote 内部矛盾是 blocking issue
2. **R1-M2/R2-M2/R3-M2 共识**："originally-released full Stage-6 query pool" 不可验证，违反 reproducibility 要求
3. **R1-M3/R2-M3 共识**：Upper panel Δ Range reproducibility 声称因 Stage 06 当前状态（2 queries/cell）而不成立

**Nature-style 5轴评估**：
- originality: GMM-based query stratification 有原创性
- scientific importance: retriever architecture syntax sensitivity 有明确重要性
- technical soundness: Table 1 数值无法从 pipeline 验证
- reproducibility: footnote 矛盾 + query pool 缺失
- readability: 写作清晰
