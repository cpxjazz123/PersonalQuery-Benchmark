# Iteration #72 — Pilot: Review writing style vs Query behavior

**日期**: 2026-07-21
**scope**: 02_writing_analysis / 实证 pilot
**提交 (P0)**: "[Major] Review writing style ≠ Query behavior 假设未验证" (iter #38)

## §A 实证 pilot 设计

源数据:
- review-style X: `01_preference_extraction/<cat>/stage1_filtered_users_reviews.json`
  - 每用户 `mean_review_words` = AVG over 所有 reviews 的 word count
- query-behavior Y: `06_query/<cat>/query_by_syntax_depth_*_train10_holdout10.json`
  - 3 个 Y:
    - `mean_query_words` = AVG `syntax_depth_query.word_count`
    - `mean_user_avg_depth` = AVG `syntax_depth_query.user_avg_depth`
    - `mean_target_depth` = AVG `syntax_depth_query.target_depth`

计算 Pearson + Spearman (无 scipy, hand-rolled)。

## §B 结果

```
Category                    n_overlap   Prs(Q)   Spr(Q)   Prs(D)   Spr(D)   Prs(T)   Spr(T)
----------------------------------------------------------------------------------------------------
Baby_Products                      95   0.0349   0.0335   0.6966   0.6934   0.6846   0.6795
Grocery_and_Gourmet_Food           89   0.0328   0.0062   0.6647   0.7090   0.6258   0.6465
Pet_Supplies                       97   0.1501   0.0383   0.8349   0.7668   0.8056   0.7183
```

Q = query word count, D = user_avg_depth, T = target_depth.

## §C 解读

### 有信号的两路:
- `mean_user_avg_depth` 与 review-style 强正相关 (Pearson 0.66-0.83 across domains)
- `mean_target_depth` 与 review-style 强正相关 (Pearson 0.62-0.80 across domains)

### 没有信号的一路:
- `mean_query_words` 与 review-style 几乎零相关 (Pearson 0.03-0.15)
- 这意味: **review complexity 不等于 query length**, 不是 paper §3 的核心假设。

### 重要的 caveats:
1. `user_avg_depth` 和 `target_depth` 都是 **Stage 4 model 用 review 信息算出来的**
   (vades_lite_sentence_user_distribution 模型本质是把 user review style encode 成 depth target)
   → 所以这俩变量和 review word count 的相关性高, **部分是构造性的**, 不是 ecological 验证。
2. `query_words` 才是 model 输出的**外部**变量 (query 实际长度), 它和 review style 的相关性 = 0,
   这是 paper 假设的**真正考验**: review style ≠ query length signal。

## §D 对论文 §3 影响

⚠️ 这条证据推荐放论文 §5 Limitations:
> "Our pilot correlation study (see iterations/iter_72_review_style_pilot.md) finds
> that user_avg_depth and target_depth correlate strongly with review writing style
> (Pearson r = 0.66-0.83 across 3 domains), but generated query length (i.e. the
> actual surface form of the query) does NOT correlate with review style (Pearson r
> = 0.03-0.15). The first two are partially tautological because they are conditioned
> on user review embeddings inside the vades_lite sentence VAE; the third measures
> external generation. The pipeline's claim that 'review writing style transfers to
> query syntactic complexity' is partially supported by the depth-correlation but
> NOT directly supported by the query-word-count correlation."

未来 ablation: 把 user_avg_depth **去掉** (改用 fixed depth = 6 across all users),
看 query words / syntax_depth_query.user_avg_depth 是否还能保持 §3 论文所示的差异。
如果 ablation 后差异消失 → 现在的差异是 review-style leakage 而不是 ecological signal。

## §E 文件

- 脚本: `02_writing_analysis/pilot_review_vs_query_correlation.py`
- 输出 JSON: `result/personal_query/02_writing_analysis/review_vs_query_pilot/review_vs_query_correlation.json`
- 跑了 3 个 domain (Baby_Products / Grocery_and_Gourmet_Food / Pet_Supplies)
- non-LLM, deterministic, <5s 跑完一个 domain
