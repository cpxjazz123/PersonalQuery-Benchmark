# Iteration #01 — common 模块去重

**日期**: 2026-07-20
**scope**: 公共模块去重 —— 各 stage `common/` 与根级 `common_utils.py` / `llm_client.py` 的函数重复审查
**关联 stage**: 跨 stage (02 / 04 / 05 / 07 / 11 / 12 / 13 / 14)

---

## §A 冗余代码发现

### [P0] `log` 函数重复定义 47+ 次

- **文件**:几乎每个 stage 脚本和 `common/` 都自行定义了 `def log(msg)` 或 `def log_with_timestamp`
- **根级**:`common_utils.py` 有 `log_with_timestamp` (line 517) 但**没有**简单 `def log`
- **问题**:47 份重复实现,签名不统一
- **建议**:在 `common_utils.py` 导出统一的 `def log(msg: str) -> None`

### [P1] `format_elapsed` 重复定义 2 次

- `02_writing_analysis/common/extract_errors_common.py:117`
- `02_writing_analysis/common/classify_writing_errors_common.py:148`

### [P1] `write_json_atomic` / `write_jsonl` 重复定义 6+ 次

### [P1] Stage 05 `edit_distance` 重复定义(两个不同实现)

### [P2] Stage 04 `attribute_helpers.py` 通用函数应归属根级

### [P2] Stage 02 两文件约 60% 重叠

---

## §B 不合理逻辑发现

### [P0] Stage 05 `common.py` JSON fallback 静默吞数据(违反 Rule 7)

### [P0] Stage 01 多处 `except Exception: pass`(违反 Rule 7)

### [P1] Stage 07 `noisy_syntax_depth_eval_common.py` monkey-patch 模式三层派生

### [P1] Stage 07 评估脚本 bare `except Exception` 无日志

### [P2] Stage 04 `_load_json_with_fallback` 命名误导

---

## §C 可合并文件发现

### [P1] Stage 11/12/13 `*_syntax_depth_eval_common.py` 三层派生 → 应统一为 1 份

### [P1] Stage 02 两文件 60% 重叠 → 合并

### [P2] Stage 13 `template_query.py` + `template_noisy_query.py` 公共函数合并

### [P2] Stage 05 两文件 `edit_distance` 重复

---

## §D 本轮已实施改动

- **删除**:无(本轮仅扫描)
- **合并**:无
- **修复**:无
- **结论**:发现 P0×4 / P1×6 / P2×4

---

## §E 验证

本轮仅扫描,无文件修改,待下一轮实施后验证。

---

## §F 下轮建议

1. **[iter #02] 在 `common_utils.py` 添加 `log`+`write_json_atomic`+`write_jsonl`,批量替换 47+ 处 `log` 定义**
2. **[iter #03] 修复 Stage 05 JSON fallback + Stage 01 `except Exception: pass`**
3. **[iter #04] Stage 07/11/12/13 eval common 统一**
4. **[iter #05] Stage 02 两文件合并**
5. **[iter #06] Stage 04 attribute_helpers 拆分**
