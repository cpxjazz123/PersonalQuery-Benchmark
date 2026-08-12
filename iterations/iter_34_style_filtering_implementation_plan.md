# Iteration #34 — Stage 05/07/10 filtering 机制详细检查

**日期**: 2026-07-20
**scope**: Stage 05/07/10 filtering 机制详细检查 + 95th percentile filtering 实现位置确认
**关联 stage**: 05_inject_noisy / 04_query / 10_complexity_analysis

## §A 论文发现

### §2.2 Style Filtering（核心发现）

论文 Section 2.2 原文：
> "the negative surrogate log-likelihood of the candidate query under the target user's mixture must not exceed the 95th percentile of the user's held-out review distribution"

约束公式：
```
-nll(x_q, u) ≤ percentile_95(nll(x_h, u) for x_h in H_u)
```

## §B 代码问题 — 重新核实结论

### §B.1 Stage 05 — 无 GMM filtering ✅（确认）

`apply_lambdamart_userbased_noisy.py`:
- 规则匹配注入错误（编辑距离、换位、视觉相似、键盘相邻）
- 无 GMM 过滤逻辑
- 所有 query 都注入错误，不管风格匹配度

### §B.2 Stage 04 — 无 GMM filtering ✅（确认）

`syntax_depth_no_depth_check.py`:
- 生成 10 个 candidate queries per user
- 仅验证 5 个 attribute 的使用
- 无 GMM style filtering

### §B.3 Stage 10 GMM filtering ✅ **已实现**（重要更正）

**关键发现：`train_vades_lite_sentence_latent_threshold.py` 中 `rank_and_select_queries` 函数已实现 95th percentile style filtering。**

实现细节：

1. **`calibrate_absolute_threshold_with_unseen_holdout`**（line 1626）：
   - 对每个用户的 held-out sentences，计算在用户 GMM 分布下的 log-likelihood
   - 取 `ABS_THRESHOLD_QUANTILE = 0.95` 即 95th percentile 作为阈值
   - 每个用户的 threshold 独立计算

2. **`rank_and_select_queries`**（line 1807）：
   - 对每个 candidate query，计算 `range_score = -log_p_xu`（negative log-likelihood）
   - 判断：`range_score <= abs_threshold[user_id]` → passes
   - 如果有多个通过，选择 range_score 最低的
   - 未通过的 query 记录到 `rejected_rows`

3. **Pipeline 流程确认**：
   - 输入：Stage 04 原始候选 queries（`query_by_syntax_depth_no_depth_check_10.json`，每用户 10 个 candidate）
   - 处理：Stage 10 VAE/GMM 训练 → 计算 95th percentile threshold → filtering
   - 输出：过滤后 queries（`query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json`）

这正好匹配论文的 pipeline：GMM filtering 在 Stage 10 中实现，在 GMM 训练完成后对 candidate queries 应用。

### §B.4 Stage 09/07 — retrieval-level 过滤 ✅（确认）

`noisy_syntax_depth_eval_common.py` `filter_retriever_pair_results()`:
- 过滤规则：noisy P@10 ≥ correct P@10 的样本被排除
- 这是 retrieval-level 过滤（过滤掉 noisy 反而更好的 case）
- 与论文要求的 GMM-based style filtering 是不同层次，互不冲突

### §B.5 结论（修正）

| 阶段 | 是否有 95th percentile GMM filtering | 备注 |
|------|--------------------------------------|------|
| Stage 04 query 生成 | ❌ | 仅 attribute 验证 |
| Stage 05 error injection | ❌ | 仅规则匹配，无风格过滤 |
| Stage 10 GMM training + filtering | ✅ **已实现** | `rank_and_select_queries` 实现完整 |
| Stage 07/09 evaluation | ❌ | retrieval-level 过滤 |

## §C 本轮已实施改动

无（本轮为分析）

## §D 待核实新问题

虽然 95th percentile filtering 在 Stage 10 中实现了，但发现了新的潜在问题：

1. **Stage 10 输出被 Stage 05 引用了吗？**
   - Stage 05 `apply_lambdamart_userbased_noisy.py` 的 query file 路径是：
     `query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json`
   - 这正是 Stage 10 的输出文件，说明 Stage 05 正确使用了经过 filtering 的 queries

2. **需要确认：Stage 10 的 filtering 是否只在 VAE encoder 可用时才工作？**
   - 如果 `COVARIANCE_MODE == "diagonal_gmm"`，则使用 GMM log-likelihood
   - 如果是其他模式（StudentT/Laplace/Logistic），则使用不同的 likelihood 函数
   - 论文推荐 GMM，所以代码选择了 `diagonal_gmm` 作为默认

3. **Regeneration 机制缺失？**
   - 论文说：如果 candidate 不满足 filtering，regenerate（每轮10个，最多10轮）
   - 代码中 Stage 10 似乎只是**选择**最好的，没有 regenerate 机制
   - Stage 04 生成了固定的 10 个 candidates，之后不再 regenerate
   - 这是**实现与论文描述的差异**

## §E 下轮建议

- 确认 Stage 04 是否实现了 regeneration 机制（每轮10个candidate，最多10轮）
- 如果没有，这可能是需要补充的论文实现差异
- 检查 Stage 10 filtering 输出是否正确传递到 Stage 05
