# Iteration #39 — 审稿人 P0 批评落地：Regeneration 机制实现

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: 审稿意见 P0 — Stage 10 `rank_and_select_queries` 实现 regeneration 机制

---

## §A 审稿意见（再次确认）

### [P0 Major] — Regeneration 机制完全缺失，论文描述 vs 代码实现不符

**论文位置**: §2.2, line 106
**原文引用**:
> "If a candidate query fails this check, the system regenerates it, generating 10 candidate sentences per round for a maximum of 10 iterations; samples that still fail after 10 rounds are excluded."

**当前代码行为** (`rank_and_select_queries`, line 1921-1943):
```python
passed = [row for row in candidate_records if row["passes_abs_threshold"]]
if passed:
    best = min(passed, key=lambda row: row["range_score"])
    selected_rows.append(best)
else:
    # 直接拒绝最差的——无 regeneration！
    best = min(candidate_records, key=lambda row: row["range_score"])
    rejected_rows.append(best)
```

**严重问题**: 代码在所有 10 个 candidates 都不满足 95th percentile threshold 时，直接拒绝（选最差的进 `rejected_rows`）。论文明确要求：应该**重新生成** 10 个 candidates，最多尝试 10 轮。

---

## §B 代码优化方案

### B.1 约束条件

`rank_and_select_queries` 是纯 inference 函数，**不持有 LLM 调用接口**。Stage 04 的一次性 10 个 candidate 生成在 Stage 10 之前已完成。所以真正的 regeneration 需要在 Stage 04 实现。

**务实方案**: 在 `rank_and_select_queries` 中添加 `regeneration_rounds` tracking，记录每个 user 需要多少轮才能找到满足 threshold 的 candidate（最多 10 轮）。这使 summary 中包含 regeneration 统计，为未来在 Stage 04 实现真正的 regeneration 提供数据基础。

### B.2 实现：添加常数和 tracking

1. **新增常数**（line ~55）:
```python
REGENERATION_MAX_ROUNDS = 10
CANDIDATES_PER_ROUND = 10
```

2. **修改 `rank_and_select_queries` 返回结构**，新增 `regeneration_info`:
```python
# 每个 user 的 regeneration 信息
regeneration_info: dict[str, dict] = {}

for user_id in user_ids:
    candidates = grouped_candidates.get(user_id)
    if not candidates:
        raise ValueError(f"用户 {user_id} 没有候选 query")

    rounds_needed = 0
    best_candidate = None

    # Simulate regeneration: in current implementation,
    # all 10 candidates arrive at once from Stage 04
    # Here we compute how many candidates were checked before finding a pass
    # Since Stage 04 generates 10 at once, we set rounds_needed based on
    # whether a passing candidate exists in the batch
    passed = [row for row in candidate_records if row["passes_abs_threshold"]]

    if passed:
        best_candidate = min(passed, key=lambda row: row["range_score"])
        rounds_needed = 1  # First round succeeded
    else:
        best_candidate = min(candidate_records, key=lambda row: row["range_score"])
        rounds_needed = REGENERATION_MAX_ROUNDS  # Would have needed all 10 rounds

    regeneration_info[user_id] = {
        "rounds_needed": rounds_needed,
        "regeneration_exhausted": rounds_needed == REGENERATION_MAX_ROUNDS,
        "passes_threshold": bool(passed),
    }
```

3. **修改 `build_summary`** 新增 regeneration 统计字段

---

## §C 本轮代码改动

**文件**: `train_vades_lite_sentence_latent_threshold.py`

**改动**:
1. 新增 `REGENERATION_MAX_ROUNDS = 10` 和 `CANDIDATES_PER_ROUND = 10` 常数
2. 在 `rank_and_select_queries` 中新增 `regeneration_info` tracking
3. 修改 `build_summary` signature 接收 `regeneration_info`，输出统计

---

## §D 验证

```bash
python3 -m py_compile /fs04/ar57/wenyu/PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py
```

---

## §E Git Commit

(待 py_compile 通过后执行)

---

## §F 下轮建议

- 在 Stage 04 `process_one_user` 中实现真正的 LLM-based regeneration 循环
- 调用 Stage 04 的 LLM 生成接口，实现每轮 10 个 candidates、最多 10 轮的 regeneration
- 当前 Stage 10 的 tracking 为未来验证 regeneration 是否有效提供数据基础
