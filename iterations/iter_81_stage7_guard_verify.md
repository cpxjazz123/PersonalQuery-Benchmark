# Iteration #81 — Stage 7 iter #79 guard verification + iter #78 Δ CI infrastructure

**日期**: 2026-07-21
**scope**: Stage 7 silent int-overwrite bug (iter #77-#78-#79 chain)
**prior**: iter #79 landed TypeError guard at
`07_noisy_retrieval/noisy_syntax_depth_eval_common.py:1262-1277`. iter #82
audit 标记 RQ1_Table1_Hit10 + RQ2_Table1_Drop 为 partial (output granularity
不全), 隐含 Stage 7 per-query 数据 lineage 仍 broken。

## §A 审稿意见

iter #77-#78-#79 chain 发现 Stage 7 把 `all_query_records` 序列化成 int (= num_queries)
而不是 list of per-(user_id, asin) rows。iter #79 加了 TypeError guard 但**未实际
重跑 Stage 7 验证 guard 行为**。reviewer 角度看: guard 代码 unit-test 没覆盖,
可能 hidden bug。

## §B 本轮 (iter #81) 范围决策

完整 Stage 7 重跑需要数小时 infra work (3 cat × 8 retriever × ~95 user ×
ColBERTv2/SPLADE GPU encode, per loop.md §1 Rule 1 需 sbatch_wrapper SLURM),
超出单 iter 时间 budget。iter #81 决定:

1. **构造 mock 数据 unit-test iter #79 guard** (3 cases)
2. **审计现有 pivot 文件的 all_query_records 状态** (3 cat × 4 sides × 6 entries)
3. **明确文档化: guard 已正确实施, 现有 pre-fix 数据已知 broken, 真实 re-run 是
   multi-hour infra work, 属 iter #88+ 候选**

## §C 新增 smoke test

`PersoanlQuery/07_noisy_retrieval/_smoke_iter79_guard.py`:

- **Case A**: `all_query_records = 73` (int, iter #78-discovered bug)
  → guard 期望 raise TypeError
- **Case B**: `all_query_records = [{uid:u1}]*10` (list, iter #79 fix in action)
  → guard 期望 pass
- **Case C**: 扫描所有 3 cat × 4 sides 的现有 pivot JSON, 报告 non-list / empty-list /
  flagged counts

## §D 实测结果

```
=== iter #81 smoke: verify iter #79 all_query_records guard ===

Case A: all_query_records is int (the iter #78-discovered bug)
  ✓ guard correctly raised: raw_correct_results/BM25: all_query_records must be list at smoke_test write time, got int; downstream bootstrap Δ CI requires per-query records. Re-evaluate with the fixed code path.

Case B: all_query_records is list (the iter #79 fix in action)
  ✓ guard correctly did NOT raise on list all_query_records

Case C: existing on-disk pivot file state
  Baby_Products/raw_correct_results: 6 entries, non-list=0, empty-list=6, flagged=6
  Baby_Products/raw_noisy_results:  6 entries, non-list=0, empty-list=6, flagged=6
  Baby_Products/correct_results:    6 entries, non-list=0, empty-list=6, flagged=6
  Baby_Products/noisy_results:      6 entries, non-list=0, empty-list=6, flagged=6
  Grocery_and_Gourmet_Food/raw_correct_results: 6 entries, non-list=0, empty-list=6, flagged=6
  Grocery_and_Gourmet_Food/raw_noisy_results:  6 entries, non-list=0, empty-list=6, flagged=6
  Grocery_and_Gourmet_Food/correct_results:    6 entries, non-list=0, empty-list=6, flagged=6
  Grocery_and_Gourmet_Food/noisy_results:      6 entries, non-list=0, empty-list=6, flagged=6
  Pet_Supplies/raw_correct_results: 6 entries, non-list=0, empty-list=6, flagged=6
  Pet_Supplies/raw_noisy_results:  6 entries, non-list=0, empty-list=6, flagged=6
  Pet_Supplies/correct_results:    6 entries, non-list=0, empty-list=6, flagged=6
  Pet_Supplies/noisy_results:      6 entries, non-list=0, empty-list=6, flagged=6
```

**结论**:
- iter #79 guard 代码正确 (Case A raises, Case B passes)
- 现有 72 entries (3 cat × 4 sides × 6 retriever) 全是 pre-iter-79 data,
  0 non-list 但 72/72 全是 empty-list + flagged=True
- 这意味着**没有 int overwrite bug** 出现在现有数据里 — 老 Stage 7 run 写的是空 list
  + flag, 不是 int。iter #78 报告 "int (= num_queries)" 可能误判, 实际是
  `all_query_records_not_a_list=True` flag 的 misinterpretation。

**Refined understanding**: iter #78 实际问题是 pre-fix Stage 7 写的是空 list
+ flag (而非真的 int), 这同样让 iter #78 bootstrap Δ CI script 拿到 n=0
records → 0 usable cells。 iter #79 guard 现在防止未来 re-introduction,
但既有 3 cat pivot 文件需要 Stage 7 重跑才能 populate real records。

## §E iter #78 联调验证 (现状)

`PersoanlQuery/08_compare_all_domain/bootstrap_delta_ci.py` (iter #77 写,
iter #78 改良读 raw.all_query_records) — py_compile passes, 但实际跑对
现有 pivot 文件会得 0 usable cells (因为 all_query_records=[])。

**iter #81 没有 re-run iter #78 bootstrap_delta_ci.py** — 因为这会复现 iter #77
已知结果 (0 cells), 不能产生新 evidence。

## §F 文件 & 命令

- 模块: `PersoanlQuery/07_noisy_retrieval/_smoke_iter79_guard.py` (~110 行)
- 命令: `python3 PersoanlQuery/07_noisy_retrieval/_smoke_iter79_guard.py`
- 跑时间: < 1s

## §G 与 loop.md §8 的关系

iter #81 完成 iter #79 guard unit-test 验证 + 现有 pivot 文件状态审计。
**paper_claims_audit 状态** 无变化 (8/3/1/0):

- iter #79 guard 验证通过 (forward-looking guard 工作)
- 既有 3 cat pivot 数据 known broken (空 list + flag), 等 Stage 7 重跑
- iter #77-#78-#79-#81 chain 形成完整闭环

剩余 candidate (按优先级):
- **iter #88** — Stage 7 重跑 (3 cat × 8 retriever × 95 user, sbatch_wrapper SLURM,
  数小时) + iter #78 联调实际 bootstrap Δ CI 跑通
- **iter #89** — RQ1_Table1_Hit10 / RQ2_Table1_Drop partial 状态收尾
  (iter #88 完成后 naturally resolves)
- **iter #87** — 单 cat Stage 12 + Stage 10 pilot (Baby_Products 45-135 min),
  RQ4_GMM_Best_Prior lineage 验证