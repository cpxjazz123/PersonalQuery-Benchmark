# Iteration #79 — Stage 7 all_query_records-as-int invariant + protect downstream bootstrap

**日期**: 2026-07-21
**scope**: 07_noisy_retrieval / Stage 7 serialize correctness
**reviewer concern (P0)**: "[Major] Table 1 Δ 值无 CI, bootstrap 数据 lineage 断裂" (iter #41 + iter #77 + iter #78)

## §A 审稿意见（尖锐批评）

iter #77 综述: 我跑了 bootstrap Δ CI, 0 个 usable cell;
iter #78 (correction): 看了 source, 知道 `raw_*_results[i].all_query_records` 应该
是 per-query record list, 但 dump 里是 `[]` 配 `all_query_records_not_a_list=True`
flag.

iter #70 pivot script 的注释已记录这一现象:
> "the iter #68-produced noisy JSONs store `all_query_records` as an int (= num_queries)
> rather than a list of records. There is no way to recover the per-(user_id, asin)
> rows without re-running the eval."

这是 Stage 7 的 **silent serialization bug** — `all_query_records` 应当是
`List[Dict]`(每 query 一个 record 含 metrics), 实际写成了一个 int (= query 数)。
iter #68 之后所有 3 domains × 6 retrievers 的 outputs 都是这个形态。Bootstrap Δ
CI 没法跨这个数据 lineage 缺口。

## §B 本轮: 保护 invariant + 仍能 evidence 出下游必要性

### Patch (1 file)

`07_noisy_retrieval/noisy_syntax_depth_eval_common.py` 在原 JSON write site
(line 1263) 之前插入 invariant check:

```python
# iter #79: explicit invariant — all_query_records must be a list at JSON
# write time. Previous runs accidentally serialized this field as an int
# (lost per-query Δ data), so downstream bootstrap CI failed silently.
for side, items in (
    ("raw_correct_results", results_to_save["raw_correct_results"]),
    ("raw_noisy_results", results_to_save["raw_noisy_results"]),
    ("correct_results", results_to_save["correct_results"]),
    ("noisy_results", results_to_save["noisy_results"]),
):
    for entry in items:
        rec_field = entry.get("all_query_records")
        if not isinstance(rec_field, list):
            raise TypeError(
                f"{side}/{entry.get('retriever', '?')}: all_query_records must be list at "
                f"write time, got {type(rec_field).__name__}; downstream bootstrap Δ CI "
                f"(iter #78) requires per-query records. Re-evaluate the pair with the "
                f"fixed code path."
            )
```

符合 loop.md §1 ("缺失 → raise, 禁 fallback"): 不再 silent fallback, 直接 raise
指引 re-evaluate。

### Smoke test

`python3 -m py_compile` 通过。

## §C Bug 根因 (best-effort 推测)

iter #68 把 `filter_retriever_pair_results` 的 empty-filter raise 改成了
return-empty + `all_records_excluded=True` 路径 (lines 549-554), 这是已经修过的
关键修复。但 Stage 7 给某条特定 query 路径(e.g. `evaluate_dense_pair` for one
retriever in Baby with 2 queries where both have noisy_value >= correct_value)
走过这条 all-records-excluded path, 使 `filtered_*_results[i].all_query_records = []`
(filtered 端合法)。**但 raw 端的  `*_result` 是直接 reference `correct_results[i]`,
是不是某次 in-place mutation 让 raw 端也被改写 = [] ?** 这就是真正的根因, 但
需要 Stage 7 实际重跑才能 pinpoint (我这里只做了 invariant assertion, 没改
in-place mutation 行为)。

## §D 论文 §3.4 / Limitations 应加

> "We found during review that the Stage 7 noisy JSONs store `all_query_records`
> as an int instead of a per-query list, preventing per-query bootstrap Δ CI.
> We add an explicit TypeError invariant at the write site
> (iterations/iter_79_stage7_all_query_records_assertion.md) to ensure any future
> rerun of Stage 7 either produces per-query records or fails loudly. A full
> bootstrap-resampled CI Table 1 is therefore left for a Stage 7 re-run with the
> updated code path."

## §E 与 iter #78 关系

iter #78 = "boot script (sampling-pipeline 端) using raw.all_query_records".
iter #79 = "Stage 7 (writing 端) invariant assertion to prevent silent corruption".
两个 iter 闭环了 bootstrap Δ CI 的 end-to-end:
- iter #79 之后**任何 Stage 7 重跑**会大声 fail 或正常输出 list
- iter #78 **未来**能直接读 raw.all_query_records 跑真 bootstrap

## §F Remaining P0 items (更新)

| ID | 审稿意见 | 状态 |
|----|---------|------|
| iter #80 候选 | 实际跑 Stage 7 重生成 raw.*.all_query_records (需要 sbatch, 60min on GPU 0/1/2) | iter #80 todo |
| iter #81 候选 | iter #78 + iter #79 联调验证: Stage 7 重跑后 iter #78 跑出真 CI | iter #81 todo |
| iter #38 | GMM prior 循环论证 | iter #71 文档化 gap, blocked on Stage 10/12 |

## §G 文件 & 命令

- 修改: `PersoanlQuery/07_noisy_retrieval/noisy_syntax_depth_eval_common.py`
  - lines 1262-1277: invariant assertion 块插入到 json.dump 前
- 命令: `python3 -m py_compile PersoanlQuery/07_noisy_retrieval/noisy_syntax_depth_eval_common.py` 通过
- 静态行为: 下次 Stage 7 eval 跑过 raw.*.all_query_records 异常时会 raise 而非 silent overwrite → 文档化为"iter #68 之后此 bug 已存在", 实际修复需 Stage 7 重跑。
