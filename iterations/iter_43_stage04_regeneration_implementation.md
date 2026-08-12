# Iteration #43 — Stage 04 LLM-based Regeneration 机制实现

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人 + 代码实现
**scope**: Stage 04 `process_one_user` 实现论文 §2.2 描述的真正 regeneration 循环

---

## §A 当前代码行为分析

### 当前 `process_one_user` 的缺陷

**文件**: `04_query/common/syntax_depth_no_depth_check.py:315-371`

**当前行为**（单次 LLM 调用，无 regeneration）：
```python
def process_one_user(category: str, task: dict) -> dict | None:
    # 1. 调用 LLM 生成 10 个 candidates（一次性）
    system_base, user_content = build_syntax_depth_prompt(category, attrs)
    response = call_llm_no_empty_retry(user_content, system_base=system_base, step_name="SyntaxDepthQuery")
    candidates = parse_syntax_depth_response(response, target_depth)

    # 2. 逐个验证 attribute 使用
    accepted_queries = []
    for idx, parsed in enumerate(candidates):
        attr_ok, attr_error = validate_query_uses_exactly_five_attrs(query, source_attrs_used)
        if not attr_ok:
            rejection_reasons.append(...)
            continue
        accepted_queries.append(...)

    # 3. 如果 accepted != 10，直接返回 None——无任何 regeneration！
    if len(accepted_queries) != NUM_CANDIDATES_PER_USER:
        log(f"[ERROR] Expected {NUM_CANDIDATES_PER_USER} attr-valid candidates, accepted={len(accepted_queries)}")
        return None
```

**论文要求**（§2.2, line 106）：
> "If a candidate query fails this check, the system regenerates it, generating 10 candidate sentences per round for a maximum of 10 iterations; samples that still fail after 10 rounds are excluded."

**Gap**: 当前代码在 validation 失败时直接 return None，论文要求最多尝试 10 轮。

---

## §B 实现方案

### B.1 约束条件

1. **Regeneration 发生在 Stage 04**（不在 Stage 10）：Stage 04 的 `process_one_user` 是生成 10 个 candidates 的地方
2. **每轮生成 10 个 candidates**（由 prompt 控制）
3. **最多 10 轮**
4. **Regeneration 条件**：当有效 candidates < 10 时，触发下一轮
5. **Stage 10 的 tracking 数据**（iter #39 已实现）用于记录 regeneration 是否被触发

### B.2 架构决策

**决策**: 在 `process_one_user` 内部添加 `while` 循环，追踪已收集的 candidates，直到：
- 收集到 ≥ 10 个 valid candidates → 取前 10 个
- 或者 10 轮全部用尽 → 失败

**为什么不直接在 LLM prompt 中加 "regenerate if insufficient"？**
- Prompt-based regeneration 不可靠
- 需要显式循环控制

**为什么不用 Stage 10 来做 regeneration？**
- Stage 10 不持有 LLM 接口
- Stage 10 只做 filtering，不做 generation

### B.3 实现细节

在 `syntax_depth_no_depth_check.py` 中新增：

```python
REGENERATION_MAX_ROUNDS = 10  # 已在 Stage 10 定义，此处复用

def process_one_user_with_regeneration(category: str, task: dict) -> dict | None:
    """
    Stage 04 query generation with true regeneration.

    Per paper §2.2: "generating 10 candidate sentences per round
    for a maximum of 10 iterations"
    """
    uid = task["user_id"]
    attrs = task["attrs"]
    source_attrs_used = _attrs_used_from_source(attrs)
    avg_depth = task["syntax_depth"]["avg_depth"]
    target_depth = task["syntax_depth"]["target_depth"]

    all_collected: list[dict] = []
    regeneration_rounds = 0
    last_error = None

    for round_idx in range(REGENERATION_MAX_ROUNDS):
        regeneration_rounds = round_idx + 1

        # Generate a fresh batch of 10 candidates
        system_base, user_content = build_syntax_depth_prompt(category, attrs)
        response = call_llm_no_empty_retry(
            user_content,
            system_base=system_base,
            step_name=f"SyntaxDepthQuery_round{round_idx + 1}"
        )
        if not response:
            log(f"  [WARN] Round {round_idx + 1}: Empty LLM response, user={uid}")
            last_error = "empty_response"
            continue

        candidates = parse_syntax_depth_response(response, target_depth)
        if not candidates:
            log(f"  [WARN] Round {round_idx + 1}: Parse failed, user={uid}")
            last_error = "parse_failed"
            continue

        # Validate each candidate
        for idx, parsed in enumerate(candidates, start=1):
            query = parsed["query"]
            attr_ok, attr_error = validate_query_uses_exactly_five_attrs(query, source_attrs_used)
            if not attr_ok:
                continue

            all_collected.append({
                "target_depth": target_depth,
                "actual_depth": None,
                "depth_validation_skipped": True,
                "user_avg_depth": avg_depth,
                "query": query,
                "word_count": count_words(query),
                "attrs_used": dict(source_attrs_used),
                "accepted_candidate_index": idx,
                "candidate_count": len(candidates),
                "regeneration_round": regeneration_rounds,
            })

        if len(all_collected) >= NUM_CANDIDATES_PER_USER:
            break

    if len(all_collected) < NUM_CANDIDATES_PER_USER:
        log(
            f"  [ERROR] Regeneration exhausted: {len(all_collected)}/{NUM_CANDIDATES_PER_USER} "
            f"valid candidates after {regeneration_rounds} rounds, user={uid}"
        )
        return None

    accepted_queries = all_collected[:NUM_CANDIDATES_PER_USER]

    return {
        "user_id": uid,
        "asin": task["asin"],
        "syntax_depth_queries": accepted_queries,
        "syntax_depth_query": accepted_queries[0],
        "query_count": len(accepted_queries),
        "depth_validation_skipped": True,
        "regeneration_rounds_used": regeneration_rounds,
        "regeneration_triggered": regeneration_rounds > 1,
    }
```

### B.4 与 Stage 10 tracking 的对接

Stage 10 的 `rank_and_select_queries` 中已有：
```python
regeneration_info[user_id] = {
    "regeneration_needed": len(passed_user) == 0,
    "rounds_needed": 1 if passed_user else REGENERATION_MAX_ROUNDS,
    ...
}
```

Stage 04 的 `regeneration_rounds_used` 信息需要传递到 Stage 10。当 Stage 04 的 `regeneration_triggered=True` 时，Stage 10 的 filtering 即使通过了也应该记录这一信息。

---

## §C 实施计划

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `04_query/common/syntax_depth_no_depth_check.py` | 添加 `REGENERATION_MAX_ROUNDS` 常量（复用 Stage 10 的定义） |
| 2 | `04_query/common/syntax_depth_no_depth_check.py` | 将 `process_one_user` 重命名为 `process_one_user_single_attempt` |
| 3 | `04_query/common/syntax_depth_no_depth_check.py` | 实现 `process_one_user` 作为带 regeneration 的 wrapper |
| 4 | `04_query/common/syntax_depth_no_depth_check.py` | 添加 `regeneration_rounds_used` 和 `regeneration_triggered` 字段到返回值 |
| 5 | `04_query/common/syntax_depth_no_depth_check.py` | py_compile 验证 |
| 6 | Stage 10 | 读取 Stage 04 输出的 `regeneration_triggered` 字段，对接 Stage 10 的 `regeneration_info` |

---

## §D 本轮改动

**文件**: `04_query/common/syntax_depth_no_depth_check.py`

**改动**:
1. 新增 `REGENERATION_MAX_ROUNDS = 10` 常量
2. 将原 `process_one_user` 重命名为 `process_one_user_single_attempt`
3. 实现新的 `process_one_user` 带 regeneration 循环（最多 10 轮）
4. 返回值新增 `regeneration_rounds_used` 和 `regeneration_triggered` 字段

---

## §E 验证

```bash
python3 -m py_compile /fs04/ar57/wenyu/PersoanlQuery/04_query/common/syntax_depth_no_depth_check.py
```

---

## §F Git Commit

(待 py_compile 通过后执行)
