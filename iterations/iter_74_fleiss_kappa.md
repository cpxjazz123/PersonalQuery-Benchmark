# Iteration #74 — Fleiss Kappa implementation for §3.3 LLM-human eval

**日期**: 2026-07-21
**scope**: 论文 §3.3 LLM-human eval / agreement metrics
**reviewer concern (P0)**: "[Major] §3.3 LLM-human eval 代码缺失，全库无 Fleiss Kappa/Spearman agreement 计算逻辑" (iter #45 / #41)

## §A 审稿意见(尖锐批评)

### 问题 1: 论文 §3.3 给出 Spearman=0.81 的同时无代码支撑
- 严重程度: Major
- 论文 `PersonalQuery-Benchmark_evaluating_retrieval.md` §3.3 声称 LLM-as-judge
  与人工评分 Spearman ρ=0.81, Fleiss κ=N(在 N 个 human evaluator 间), 这两个数字都
  在 paper table 中作为 LLM-as-judge 有效性的核心证据。
- 但整个 repo `grep -i fleiss kappa cohens` 完全无结果 — `iter #45` 已确认。
- 审稿人质询: "How can the paper claim reproducer-tier inter-rater agreement without
  any code to back it? 读者无法验证 N 个 evaluator / K 个 subject 的归一化随机一致性
  是按什么样的 Fleiss-Kappa 实现估计的。"

### 问题 2: Pearson + Spearman 已有(在 iter #72)但 Cohen/Fleiss 缺
- iter #72 pilot 已实现手写 Pearson + Spearman(无 scipy)
- 但 LLM-human 评估需要的核心 metrics:
  - **Fleiss Kappa** (多评估者, 固定 K raters per subject, N 个 subjects): 论文里
    衡量"多 annotator 间一致"
  - **Cohen's Kappa** (2 raters, 多类): 衡量"LLM vs 1 human"或 "human vs human 对照"
- 二者都缺。

## §B 实现

新增 `PersoanlQuery/agreement_metrics.py`:

- `fleiss_kappa(ratings: list[list[int]], n_categories: int) -> float`
  - 严格输入校验: K(每 subject rater 数) 必须固定且 >= 2
  - n_categories 必须 >= 2
  - 检查每个 rater score ∈ [0, n_categories)
  - P_e == 1.0 (单一类别) 时 raise, 不 fallback
- `cohens_kappa(rater_a: Sequence, rater_b: Sequence) -> float`
  - 长度匹配 + 每项严格转 int
- 公开公式注释, 引用 Fleiss (1971) / Cohen (1960)
- 内嵌自检 `_self_test` 用 Wikipedia worked example + 完美一致 test case
  验证: kappa 必须在 [-1, 1], 完美一致 → 1.0

## §C 验证

```
$ python3 -m py_compile PersoanlQuery/agreement_metrics.py
(no output)

$ python3 PersoanlQuery/agreement_metrics.py
agreement_metrics self-tests passed.
```

## §D 与论文 §3.3 对应

论文需要回答的 4 个 sub-question + 这个模块能直接接入的位置:

| 论文问题 | 需要 metric | 模块位置 |
|---------|-----------|----------|
| LLM vs Human correlation | Pearson / Spearman | iter #72 `pilot_review_vs_query_correlation.py` |
| Multi-human inter-rater agreement | Fleiss Kappa | iter #74 ✓ NEW |
| Single human vs LLM pair | Cohen's Kappa | iter #74 ✓ NEW |
| 评估者间一致性 | Fleiss + Cohen | iter #74 ✓ NEW |

## §E 后续 (iter #75+)

- 还没有 human evaluation 的 JSON 数据,所以无法算实际 Fleiss Kappa。
- 真正落地需要一个真实数据: 至少 3 human eval, 在同一个 query set 上独立标注
  (例如: 30-50 个 query, 每个 query 3 个评估者标 REL/PARTIAL/IRREL)。
- iter #75 候选: 写 mock human eval 数据 + 跑 agreement_metrics 跑出假 demo (验证
  数据 pipeline → agreement 输出 端到端跑通)。

## §F 文件 & 命令

- 模块: `PersoanlQuery/agreement_metrics.py` (137 行, 含 docstring + 自检)
- 命令:
  - `python3 -m py_compile PersoanlQuery/agreement_metrics.py`
  - `python3 PersoanlQuery/agreement_metrics.py` (跑 self-test)
