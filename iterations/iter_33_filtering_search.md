# Iteration #33 — 论文§2.2 95th percentile filtering + §3.1 retriever架构分析

**日期**: 2026-07-20
**scope**: 确认95th percentile style filtering未实现 + retriever架构覆盖检查
**关联 stage**: 05_inject_noisy / 10_complexity_analysis

## §A 论文发现

### §2.2 Style Filtering — 确认缺失
论文 Section 2.2 描述：
> "the negative surrogate log-likelihood of the candidate query under the target user's mixture must not exceed the 95th percentile of the user's held-out review distribution"

全项目 grep 搜索结果：
- `evaluate_review_query_alignment.py`: percentile 用于 query score normalization（[0,1] 范围），非 GMM filtering
- `06_fast_fullscale_eval*.py`: percentile 用于 bootstrap confidence interval，非 filtering
- **结论：95th percentile style filtering 整个项目未实现**

这意味着当前 pipeline 的 candidate query filtering 机制可能不完整。

### §3.1 Retriever 架构分类

| 架构类型 | Retriever | 论文Δ(句法) | 论文Δ(误差) |
|---------|-----------|------------|------------|
| Lexical | BM25 | 4.3 | 8.22 |
| Sparse expansion | SPLADE | 9.6 | 6.51 |
| Dense bi-encoder | BGE | 5.6 | 6.65 |
| Dense bi-encoder | E5 | 6.6 | **11.16** |
| Dense bi-encoder | MiniLM | 2.5 | 2.91 |
| Dense bi-encoder | STAR | 5.1 | 7.06 |
| Dense bi-encoder | ANCE | 3.6 | 5.76 |
| Multi-vector late interaction | ColBERTv2 | 8.6 | 9.39 |
| LLM reranker | DeepSeek-v4 Reranker | 9.1 | 6.62 |

关键发现：
- **E5 对 spelling error 最敏感（Δ=11.16）**：说明 E5 的 subword tokenization 受错误拼写影响严重
- **SPLADE/DeepSeek/ColBERTv2 对句法结构最敏感**：说明显式 term expansion 和 token-level interaction 对结构更敏感

## §B 代码问题

- **95th percentile style filtering 完全缺失**：这可能导致生成的 queries 超出用户历史表达范围
- 需要确认 Stage 05 的 query 生成是否有其他 filtering 机制替代

## §C 本轮已实施改动

无（本轮为论文分析）

## §D 验证

N/A

## §E Git Commit

N/A（分析轮次）

## §F 下轮建议

- 在 Stage 05 `apply_lambdamart_userbased_noisy.py` 或 `generate_noisy_query_cache.py` 中找到 query generation 的 filtering 逻辑，确认是否使用了其他方式替代 95th percentile filtering
- 检查 Stage 07 的 noisy query cache 生成 pipeline 是否有类似 filtering
- 如果确实缺失，考虑实现 95th percentile filtering 作为 Stage 10 或 Stage 05 的补充
