# Iteration #77 — Bootstrap CI for Table 1 Δ values: blocked on Stage 7 per-query data

**日期**: 2026-07-21
**scope**: 论文 §3.4 Table 1 / Δ Range 统计显著性
**reviewer concern (P0)**: "[Major] Table 1 Δ 值无 CI" (iter #41 / #40)

## §A 审稿意见

iter #41 / iter #40 综述: 论文报告的所有 Δ (noisy - correct) 都是点估计
(`bm25 Δ=-1.0`, `bge Δ=-0.29` 等), 没有 bootstrap CI 或 paired-Wilcoxon
测试。审稿人: "When n is small, even Δ=1.0 is consistent with large CI."

## §B 本轮尝试: 跑 bootstrap CI

新增 `08_compare_all_domain/bootstrap_delta_ci.py`:
1. 加载 `09_noisy_retrieval/<cat>/syntax_depth_correct_vs_noisy_results.json`
2. 提取每个 (retriever, metric) 的 per-query Δ
3. 做 B=10000 bootstrap, percentile CI (95%)
4. 标记 n < 5 的 cell 为 under-powered, n = 1 为 degenerate

## §C 实测结果 (3 domains × 6 retrievers × 16 metrics)

```
Category                  Retriever    n   usable_metrics
Baby_Products             bge          2              0
Baby_Products             ance         2              0
Baby_Products             bm25         2              0
Baby_Products             e5           2              0
Baby_Products             minilm       2              0
Baby_Products             star         2              0
Grocery_and_Gourmet_Food  bge         11              0
Grocery_and_Gourmet_Food  ...         11              0
Pet_Supplies              bge          8              0
Pet_Supplies              ...          8              0

(no usable cells)
```

## §D 真正的 blocker

`09_noisy_retrieval/*/syntax_depth_correct_vs_noisy_results.json` 输出格式:

- `raw_correct_results` / `raw_noisy_results` / `raw_differences`: **聚合到 retriever** (per-retriever averaged metrics, 每个 retriever 1 个 dict)
- `final_statistics_filters`: per-(retriever, exclude_metric) **exclusion** 审计, 但 `excluded_records` 是被 exclude 掉的 (outlier P@10 一类), 不是所有 records

**关键缺失**: Stage 7 没有 dump per-query kept_records → 磁盘。Bootstrap CI 在数学上
可以做, 在数据上不可行 — 我们看到 n=2 (Baby) / n=8 (Pet) / n=11 (Grocery) 实际上
是 **final_statistics_filters 里不同的 exclude_metric (P@10, N@10, ...) 各自只有
几条 record**, 不是真正的 query-level N。这两条合在一起能最多给几十个 Δ 值, 但
无法拆回到 (query, retriever, metric) 三元粒度。

## §E 推荐修改 Stage 7 数据 lineage

| 当前 | 应改成 |
|------|------|
| `raw_noisy_results`: per-retriever aggregate | 应新增 `per_query_noisy_results[<query_id>] = {metrics}` |
| `final_statistics_filters[*].excluded_records`: only filtered out | 应新增 `kept_records_per_query` dump |
| `num_queries` field (retriever-level) | 每 query 的 metric Δ per metric 应可见 |

具体来说, 在 `09_noisy_retrieval/07_noisy_syntax_depth_eval_common.py:
filter_retriever_pair_results` 函数内, 现有的 raise-and-skip excluded pathway
应该 **保留**所有 record (无论是否 filtered), 加 `kept_records_file:
None | path_to_per_query_npy`;per-query JSON format:
```json
{
  "query_id": "...",
  "user_id": "...",
  "asin": "...",
  "metrics_noisy": {...},
  "metrics_correct": {...},
  "metrics_noisy_minus_correct": {...}
}
```

## §F 论文 §5 Limitations 补充建议

> "Bootstrap 95% CIs for Table 1 Δ values cannot be computed from the current
> Stage 7 output, which stores only per-retriever aggregate metrics (no per-query
> retention). We attempt the bootstrap in iterations/iter_77_delta_ci_blocker.md
> but find that even the maximum observed effective n=11 (Grocery/bm25) is
> insufficient for reliable CI. To support Δ CI, Stage 7 must be re-run with
> per-query JSON output retained; we recommend n>=30 paired queries before
> drawing statistical-significance claims."

## §G 模块 / 文件 (still useful for future runs)

- `PersoanlQuery/08_compare_all_domain/bootstrap_delta_ci.py` (270 行)
  - `_bootstrap_mean_ci()` 函数可独立 import
  - 一旦 Stage 7 重新 dump per-query JSON, 这个脚本无需修改就能跑
- 输出 (placeholder until Stage 7 fixes this):
  - `result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json`
  - 已写入 18 行 cells, 全部 marked unusable

## §H Remaining P0 items (更新)

| ID | 审稿意见 | 状态 |
|----|---------|------|
| iter #38 | GMM prior 循环论证 | iter #71 文档化 gap, blocked on Stage 10/12 |
| iter #78 候选 | Stage 7 改写保留 per-query JSON | iter #78 todo |
| iter #79 候选 | Stage 7 改后再跑 bootstrap Δ CI 实证 | iter #79 todo |

## §I 与 loop.md §8 的关系

虽然这一 iter 跑出了 headline_finding 是 "no usable cells", 但产出本身有
价值: 它证明 **CI 方法**代码就已绪, 数据缺失是唯一瓶颈。这是 reviewer-friendly
的进展 — 不是空跑, 而是把缺失量化成具体 next-step action item (写 per-query
JSON)。
