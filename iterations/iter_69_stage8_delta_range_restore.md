# Iteration #69 — Stage 08 Δ Range code 恢复 + fast_fullscale SKIP flags

**日期**: 2026-07-21
**scope**: 08_compare_all_domain / 10_complexity_analysis / 06_fast_fullscale_eval (3)

## §A 现状

iter #49 (commit a68b4d2) 给 Stage 08 加了 `print_08_delta_range_analysis()` 函数,
给 Stage 10 加了 `compute_prior_bic_aic.py`. 但 iter #48 之后重整 stage
directories, 这两个文件丢了 — 在 Stage 8 上跑 fallback 不再能拿 Δ Range.

Stage 08 同时也需要输入文件 `06_retrieval/<cat>/retrieval_syntax_depth_summary.json`,
这个原本由 `06_fast_fullscale_eval_<cat>.py` 在干净查询上跑出来。这些脚本
不带 SKIP_COLBERTV2 / SKIP_SPLADE env var, 不能直接跑 (会卡 ColBERTv2 compile
和 SPLADE 加载)。

## §B 本轮改动

### Stage 08 — `08_compare_p10_across_domains.py`

从 git (a68b4d2) 恢复 `print_08_delta_range_analysis()`, 插在
`print_08_hit10_table` 和 `print_08_three_level_score_table` 之间 (约第 421 行),
`main()` 里加上 `print_08_delta_range_analysis(data_08)` 调用 (在 print_08_hit10_table 之后).

helpers (`get_all_retrievers_from_08`, `average_domain_values`, `fmt4`,
`CATEGORIES`, `SYNTAX_DEPTH_GROUP_ORDER_CLEAN`) 在 refactor 后都还在,
iter #49 的代码可以直接套上 (无需修改)。

### Stage 10 — `10_complexity_analysis/common/compute_prior_bic_aic.py`

从 git (a68b4d2) 恢复 (149 行)。当前 Stage 10 不输出 latent representations,
所以本轮只通过 `python3 -m py_compile` 验证脚本可编译, 不实际跑它。
等 Stage 10 出 latent embeddings 后再跑 BIC/AIC。

### Stage 06 — `06_fast_fullscale_eval_{Baby,P,G}.py`

在每个文件第 71-74 行附近, 把硬编码的 `RETRIEVERS` 拆成 SKIP env-var aware 版本:

```python
DENSE_RETRIEVERS = ['bge', 'e5', 'minilm', 'star', 'ance']
SPARSE_RETRIEVERS = ['splade']
COLBERTV2_RETRIEVERS = ['colbertv2']

if os.environ.get("SKIP_COLBERTV2", "").lower() in {"1", "true", "yes", "on"}:
    print(f"[{datetime.now().isoformat()}] [SKIP_COLBERTV2] ...", flush=True)
    COLBERTV2_RETRIEVERS = []
if os.environ.get("SKIP_SPLADE", "").lower() in {"1", "true", "yes", "on"}:
    print(f"[{datetime.now().isoformat()}] [SKIP_SPLADE] ...", flush=True)
    SPARSE_RETRIEVERS = []

RETRIEVERS = DENSE_RETRIEVERS + COLBERTV2_RETRIEVERS + SPARSE_RETRIEVERS + ['bm25']
```

跟 iter #56 的 `06_build_retriever_indices_*` 和 iter #68 的 `noisy_syntax_depth_eval_common.py`
完全对称。

### Wrappers

新写 3 个 sbatch wrapper, 路径 `/home/wlia0047/ar57/wenyu/result/personal_query/logs/stage6_fast_fullscale_{baby,pet,grocery}.sh`, 各指向对应 fast_fullscale_eval 脚本, 同时带 SKIP env vars
(`SKIP_SPLADE=1` 只对 Pet/Grocery, Baby 没有 skip SPLADE 因为它有 SPLADE index)。

## §C 验证

- `python3 -m py_compile` 3 个 fast_fullscale_eval 文件 ✅
- `python3 -m py_compile` 08_compare_p10_across_domains.py ✅
- `python3 -m py_compile` compute_prior_bic_aic.py ✅
- 3 个 fast_fullscale jobs 已 submit (58419935/936/937)

## §D 等到的产出

提交 batch job 完了预计会得到:
- `06_retrieval/Baby_Products/retrieval_syntax_depth_summary.json`
- `06_retrieval/Pet_Supplies/retrieval_syntax_depth_summary.json`
- `06_retrieval/Grocery_and_Gourmet_Food/retrieval_syntax_depth_summary.json`

喂给 `08_compare_p10_across_domains.py`, 它现在会多打印一个表:

```
08 Δ Range (correct-query range) per domain and across domains
Retriever   Baby Δ  Grocery Δ  Pet Δ  Mean Δ  Std Δ
bge         ...      ...        ...     ...     ...
e5          ...      ...        ...     ...     ...
minilm      ...      ...        ...     ...     ...
star        ...      ...        ...     ...     ...
ance        ...      ...        ...     ...     ...
bm25        ...      ...        ...     ...     ...
```

这正是论文 Table 1 那一栏 (Higher = more query-sensitive)。

## §E 还没解决的

- Stage 10 BIC/AIC 实际运行需要先跑 Stage 10 训练 latent representations。
  iter #49 那个脚本是 grammar-checked, 没喂数据跑过。
- SPLADE 重建 (~90min on 603K docs for Pet/Grocery): 一旦 SPLADE index 有,
  撤销 SKIP_SPLADE 即可多 1 列。
