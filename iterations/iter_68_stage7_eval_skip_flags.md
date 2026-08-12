# Iteration #68 — Stage 7 eval: SKIP_COLBERTV2 + SKIP_SPLADE + 空集过滤器

**日期**: 2026-07-21
**scope**: 07_noisy_retrieval / eval 弹性 + 全部 3 个 domain 跑通

## §A 现状

Stage 7 eval 是 iter #58 Stage 08 Δ Range BIC/AIC 的前置。三个 domain
(Baby_Products / Pet_Supplies / Grocery_and_Gourmet_Food) 都跑不通,
逐一排查三类阻塞:

| 阻塞 | Root cause | 修复 |
|------|-----------|------|
| `colbertv2 correct query cache not found` | Stage 06 build 在 SKIP_COLBERTV2=1 下不生成 ColBERTv2 cache,但 eval 没感知 | 在 RETRIEVERS 列表前按 env var 过滤 ColBERTv2 |
| `SPLADE retriever cache not found` | Pet/Grocery Stage 06 SPLADE 编码在 60min budget 内超时,baby 和 build 半残缺 | 同模式加 SKIP_SPLADE env var |
| `X has no query records left after excluding noisy-better-or-equal cases` | 当 n=8-11 配对很小时,过滤后某 retriever 全部命中 noisy ≥ clean;原代码直接 raise 阻塞整个 eval | 把"硬失败"换成"返回空集 +all_records_excluded=True" |

三处都是 loop.md §1 的回退逻辑禁忌:
- 缺失 → None/空值
- 意料外失败 → raise

第一类是"缺失" (预期内, 没有 ColBERTv2 cache), 必须返回不跑。
第二类是"缺失", 同理。
第三类是"统计意义为零", 也不算"意料外失败", 按空集放过更合理。
(下游 BIC/AIC 需要能 iterate 全部 retriever, 空集就让它走到该函数自身去决定。)

## §B 改动

### noisy_syntax_depth_eval_common.py

第 53-63 行附近新增 SKIP_SPLADE 过滤, 跟既有的 SKIP_COLBERTV2 对称:

```python
DENSE_RETRIEVERS = ["bge", "e5", "minilm", "star", "ance"]
COLBERTV2_RETRIEVERS = ["colbertv2"]
SPARSE_RETRIEVERS = ["splade"]

_skip_colbert = os.environ.get("SKIP_COLBERTV2", "").lower() in {"1", "true", "yes", "on"}
if _skip_colbert:
    COLBERTV2_RETRIEVERS = []
_skip_splade = os.environ.get("SKIP_SPLADE", "").lower() in {"1", "true", "yes", "on"}
if _skip_splade:
    SPARSE_RETRIEVERS = []

RETRIEVERS = DENSE_RETRIEVERS + COLBERTV2_RETRIEVERS + SPARSE_RETRIEVERS + ["bm25"]
```

`filter_retriever_pair_results` 第 527-560 行附近, 把 raise 改为返回空集:

```python
if not filtered_correct_records:
    log(f"{retriever} 所有样本均为 noisy 不差于 clean, 跳过过滤（保留空集）")
    empty_filter_summary = {
        "retriever": correct_result["retriever"],
        "exclude_metric_name": exclude_metric_name,
        "kept_count": 0,
        "excluded_count": len(correct_records),
        "excluded_pair_ids": [e["pair_id"] for e in excluded_records],
        "all_records_excluded": True,
    }
    return (
        {**correct_result, "all_query_records": [], "final_statistics_filter": empty_filter_summary},
        {**noisy_result,   "all_query_records": [], "final_statistics_filter": empty_filter_summary},
        empty_filter_summary,
    )
```

### 同时删改 wrapper launcher

`/tmp/stage7_eval_{pet,grocery,baby}.sh` 通过 `sbatch --export=ALL,SKIP_COLBERTV2=1,SKIP_SPLADE=1` 注入。

## §C 结果

| Domain | 跑了哪些 retriever | 输出文件 | 行数 |
|--------|-------------------|---------|------|
| Baby_Products | bge e5 minilm star ance bm25 (SPLADE skipped) | `09_noisy_retrieval/Baby_Products/syntax_depth_correct_vs_noisy_results.json` | 70K |
| Pet_Supplies | bge e5 minilm star ance bm25 (SPLADE skipped) | `09_noisy_retrieval/Pet_Supplies/syntax_depth_correct_vs_noisy_results.json` | 231K |
| Grocery_and_Gourmet_Food | bge e5 minilm star ance bm25 (SPLADE skipped) | `09_noisy_retrieval/Grocery_and_Gourmet_Food/syntax_depth_correct_vs_noisy_results.json` | 351K |

注意: Pet/Grocery 没有 SPLADE 比较是已知缺口 (Stage 06 build 超时),
将来如要补 SPLADE,只需取消 SKIP_SPLADE + 等 SPLADE 索引 build 完。

## §D iter #58 下一步

3 个 domain JSON 都已就绪。下一步是 Stage 08 分析:
- Δ Range across retrievers (P10 / NDCG@10 ...)
- BIC / AIC over GMM components (需 Stage 10 latent representations)
- Bootstrap CI on Δ values

代码已经存在 (`compute_prior_bic_aic.py` + `print_08_delta_range_analysis`),
缺的是输入数据, 这正是 Stage 07 现在解锁的。

## §E 还没有完成的

- Stage 06 SPLADE 重建 (90+min on 603K docs, 等 GPU 空闲)
- 14 reranker 对应的 e5/cross-encoder first-stage cache (rerank_config 已支持多 first-stage)
- 12_prf 还没拆开来看, 跟 Stage 07 是不同 PRF 维度
