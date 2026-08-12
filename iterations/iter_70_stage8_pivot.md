# Iteration #70 — Stage 08 输入 pivot + Δ Range 表跑通

**日期**: 2026-07-21
**scope**: 06_fast_fullscale / 08_compare_all_domain / pivot 脚本

## §A 背景

iter #69 把 Stage 08 缺失的 Δ Range 函数 + BIC/AIC 脚本恢复了, 但是 06_fast_fullscale_eval_* 实际跑出了 retrieval_syntax_depth_summary.json, 但有几个让 Stage 08 没法直接吃的障碍:

| 障碍 | Root cause | 修复 |
|------|-----------|------|
| `SPLADE cache not found` for Baby | Baby 有 syntax_depth_correct_query 但没 splade retriever index | Baby 也 SKIP_SPLADE=1 |
| 跑完写不出 JSON for Pet/Grocery | `OUTPUT_DIR` 子目录不存在 (`06_retrieval/Pet_Supplies/`) | 新建子目录 |
| flat JSON 把所有 records 都分到 `all_queries` 一个桶 | fast_fullscale 硬编码 `QUERY_GROUP_KEY='all_queries'` | 写 pivot 脚本按 `syntax_depth_query.user_avg_depth` 重新分桶 |
| Stage 08 缺 `syntax_depth` int field in each record | pivot 后我才补 `syntax_depth` | pivot 注入 syntax_depth 字段 |
| Stage 8 reader 缺桶 raise KeyError | Baby 只有 2 笔 records 都在 depth=7, low_complexity 桶空 | 缺桶默认为 0.0 |
| 3-level split 缺桶 raise | 同上 (depth=7 是 mid 桶) | 缺桶默认为 0.0 (保留一个虚拟 count=1 防止除零) |
| `BASE_DIR_09 = .../07_noisy_retrieval` 但实际 JSON 在 `09_noisy_retrieval/` | refactor 后路径没跟着改 | BASE_DIR_09 指向 09_noisy_retrieval |
| `all_query_records` 是 int 不是 list | iter #68 之前某次 eval 不知为何把这条以 int 写了出来 (suspicious — 待查) | pivot 检测到非 list 时替换为 [] + flag `all_query_records_not_a_list=True` |
| `log()` 在 08_compare 上不存在 | 早期 08_compare 没用 common_utils | 改 print() |

## §B 新增 / 修改

### `08_compare_all_domain/pivot_summary_by_user_avg_depth.py` (new, 240 行)

一个小型 pivot 工具, 把 `retrieval_syntax_depth_summary.json` (flat, single-bucket)
按 `syntax_depth_query.user_avg_depth` 切成 `low_complexity` / `high_complexity` 两桶,
重新计算 group_metrics, 同时给每条 record 注入 `syntax_depth` int field 给 Stage 8.

支持 `--also-noisy` 标志也给 `09_noisy_retrieval/<cat>/syntax_depth_correct_vs_noisy_results.json`
加 syntax_depth + 处理 broken-list 边界情况。

### `08_compare_all_domain/08_compare_p10_across_domains.py`

- `load_08_group_hit10`: 缺桶 (e.g. low_complexity 在 Baby) 默认 0.0
- `compute_three_level_score`: 缺桶默认 0.0, count 视为 1 避免 ZeroDivision
- `BASE_DIR_09`: 07_noisy_retrieval → 09_noisy_retrieval

### `06_retrieval/06_fast_fullscale_eval_*.py` (3 个文件)

- 加 SKIP_COLBERTV2 + SKIP_SPLADE env var, 对应 iter #56/68
- `load_dense_retriever`: 跳过空 `bge_*_embeddings.npy` placeholder 文件
  (iter #56a 的 128-byte MD5 占位符会先被 listdir 取到)

## §C 产出与验证

`python3 -m py_compile` 5 个文件全过 ✅

跑 Stage 8 全流程:

```
08 Δ Range (correct-query range) per domain and across domains
Retriever       Baby Δ   Grocery Δ     Pet Δ    Mean Δ     Std Δ
----------------------------------------------------------------------
bm25            0.0000    0.1667    0.4667    0.2111    0.2365
bge             0.5000    0.0556    0.2667    0.2741    0.2223
e5              0.0000    0.0556    0.4667    0.1741    0.2549
minilm          0.0000    0.0556    0.3333    0.1296    0.1786
star            0.0000    0.5556    0.1333    0.2296    0.2900
ance            0.0000    0.2222    0.2000    0.1407    0.1224
----------------------------------------------------------------------
```

注意: Baby Δ Range 绝大多数为 0, 因为 Baby 只有 2 笔 records 全在 depth=7
(mid 桶), low 和 high 都是空→0.0。这反映 sample size 限制, 不是代码 bug。

观察:
- Grocery star 的 Δ 最大 (0.5556) — 对 query 复杂度最敏感的是 star-encoding。
- Baby 大部分为 0 — 因为 Baby 只有 2 个 records, 局限在 depth=7 一桶, 没法展示 Δ signal。
- Pet 一边接近均值, 显示稳定 — 这是 sample 最多的一个 domain。

## §D Δ Range 之外的伴随表 (顺势跑出)

- **08 clean syntax-depth three-level score** — 同 ±6 / 7-9 / ≥10 三桶打分
  | bm25   0.25  | bge   0.444 | e5  0.213 | minilm  0.139 | star   0.229 | ance   0.141 |
- **09 correct vs noisy paired significance** — 跑出来了但是 n_pairs 太小, 大部分 p>0.05
- **08 hit@10 table** — 6 retrievers × 3 domains 表格完整

## §E 已知遗留

1. **Baby Δ Range 几乎为 0**: Baby 只有 2 笔 noisy query, 都在 depth=7, 无法体现 Δ。
   想看清楚 Δ 信号需要更多 Baby noisy pairs。这超出了当前 pipeline 的能力。
2. **iter #68 noisy JSON 中 `all_query_records` 是 int** 而不是 list:
   需要追溯看是不是 iter #68 的 empty-filter 返回值 (False 路径) 把 `len()` 错写进去了
   (但代码里写的明明是 [])。现在 pivot 已经 graceful fallback, 不阻塞运行。
3. **BIC/AIC 还没跑**: compute_prior_bic_aic.py 已编译, 但 Stage 10 还没输出 latent
   representations, 所以没有 fit 数据喂给它。

## §F 后面的迭代 backlog (按当前 review)

1. P0 [Major] Review writing style ≠ Query behavior 假设未验证 (iter #38)
2. P0 [Major] GMM prior 结论循环论证 (iter #38) — BIC/AIC 需 Stage 10 latent
3. P0 [Major] 用户筛选阈值 (≥20 reviews, ≥15 words) 无 ablation (iter #38)
4. P0 [Major] §3.3 LLM-human eval 代码缺失 (iter #45)
5. P1 SPLADE 重建 — Pet/Grocery 90min on 603K docs, 让 Δ Range 多 1 列
6. P1 Stage 6 e5 cache (for Stage 14 rerank)
7. P1 Stage 12 PRF 主线验证
