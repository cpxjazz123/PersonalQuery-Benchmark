# Iteration #35 — Stage 04/10 Regeneration 机制检查

**日期**: 2026-07-20
**scope**: Stage 04/10 是否实现论文描述的 regeneration 机制
**关联 stage**: 04_query / 10_complexity_analysis

## §A 论文描述的 Regeneration 机制

论文 Section 2.2 原文：
> "If a candidate query fails this check, the system regenerates it, generating 10 candidate sentences per round for a maximum of 10 iterations; samples that still fail after 10 rounds are excluded."

**核心要求**：
1. 每个 candidate query 在 GMM filtering 检查后，如果不满足 95th percentile 阈值
2. 系统 regenerate（重新生成）10 个 candidate sentences
3. 每轮最多 10 个 candidate
4. 最多 10 轮（100 个 total candidates）
5. 10 轮后仍不满足的样本 → excluded

## §B 代码实际情况

### §B.1 Stage 04 — 无 Regeneration ✅（确认缺失）

`process_one_user()`:
```python
response = call_llm_no_empty_retry(...)  # LLM 只调用一次
candidates = parse_syntax_depth_response(...)  # 解析 JSON
# 如果 candidates 不足 10 个 → 返回 None（用户失败）
# 如果 candidates 通过 attribute 验证但不足 10 个 → 返回 None
return None  # 无任何重试逻辑
```

**实际情况**：
- LLM **只调用一次**，生成固定 10 个 candidates
- 如果 LLM 返回格式错误或 candidates 不足 10 个 → 用户直接标记为 failed
- **完全没有 regeneration 机制**
- 如果 attribute validation 拒绝了一些 candidates，也**不会重新生成**

### §B.2 Stage 10 — 无 Regeneration ✅（确认缺失）

`rank_and_select_queries()`:
```python
for user_id in user_ids:
    candidates = grouped_candidates.get(user_id)  # 固定 10 个（来自 Stage 04）
    # 计算每个 candidate 的 range_score
    # 95th percentile filtering
    passed = [row for row in candidate_records if row["passes_abs_threshold"]]
    if passed:
        best = min(passed, key=lambda row: row["range_score"])  # 选择最好的
        selected_rows.append(best)
    else:
        best = min(candidate_records, key=lambda row: row["range_score"])  # 选择最差的
        rejected_rows.append(best)  # 直接 reject，无 regeneration
```

**实际情况**：
- 如果所有 10 个 candidates 都不满足 95th percentile threshold
- 系统选择 `range_score` 最小的那个（即使不满足）→ `rejected_rows.append(best)`
- **没有任何 regeneration 逻辑**

### §B.3 与论文的差异总结

| 论文要求 | 代码实现 | 差距 |
|---------|---------|------|
| candidate 不满足 filtering → regenerate 10 个 | 代码直接 reject 最差的 | Stage 10 缺失 regeneration |
| 每轮 10 个 candidate，最多 10 轮 | Stage 04 只生成 1 轮 10 个 | Stage 04 缺失 regeneration |
| 10 轮后仍不满足 → excluded | Stage 10 直接 excluded（但没有 regenerate） | 流程上匹配，但 regenerate 逻辑完全缺失 |

### §B.4 Pipeline 流程对比

**论文描述的 pipeline**：
```
for each user:
    for round in 1..10:
        candidates = generate_10_candidates()
        for each candidate:
            if -nll(candidate, user) <= 95th_percentile_threshold:
                accept
                break
            else:
                continue to next candidate
        if any candidate accepted:
            break
    if no candidate accepted after 10 rounds:
        excluded
```

**代码实现的 pipeline**：
```
Stage 04:  # (生成阶段，一次性)
    candidates = generate_10_candidates()  # 只生成一轮
    validate_attribute_usage()  # attribute 验证
    return 10 candidates (或 None 如果失败)

Stage 10:  # (过滤阶段)
    for each user:
        for each of 10 candidates:
            if -nll(candidate, user) <= 95th_percentile_threshold:
                accept (select best)
            else:
                reject
        if no candidate passes:
            reject (选择最差的进入 rejected_rows)
```

## §C 本轮已实施改动

无（本轮为分析）

## §D Regeneration 机制缺失的影响

1. **Query 质量下降**：如果 Stage 04 生成的 10 个 candidates 都不在用户历史表达风格范围内，Stage 10 只能"矮子里面挑将军"，无法真正 regenerate 到风格匹配的 queries
2. **用户覆盖度降低**：部分用户在 Stage 10 filtering 后可能只有很少的 queries 可用
3. **与论文描述不符**：这是实现与论文的显著差异

## §E 下轮建议

- 在 Stage 10 `rank_and_select_queries` 中实现 regeneration 循环
- 调用 Stage 04（或直接调用 LLM）重新生成 candidates
- 设计合理的最大轮数和 early-exit 条件
- 或者，在 Stage 04 中实现多轮 regeneration，将 regeneration 结果也写入 JSON
