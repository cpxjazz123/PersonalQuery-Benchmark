# Iteration #31 — §3.1 dead code 复查 Stage 02/07

**日期**: 2026-07-20
**scope**: §3.1 dead code 复查（Stage 02/07/13/14 common/ 目录 pyflakes 复查）
**关联 stage**: 02_writing_analysis, 07_noisy_retrieval, 13_query_template, 14_consolidate

## §A 冗余代码发现

- `02_writing_analysis/common/extract_errors_common.py`:
  - `from datetime import datetime` 未使用
  - `from pathlib import Path` 重复 import（第 9 行和第 15 行相同）
  - `import string` 在函数内部（第 50 行），未使用
  - `APOSTROPHES` 字面量 tuple 用法（3处），应使用常量 APOSTROPHES
  - `LLMClient` 应为 `MiniMaxAnthropicClient`
  - `print(f"[LLM_ERROR] response为空!")` f-string 无占位符
  - `output_rows = []` 死变量
- `02_writing_analysis/common/classify_writing_errors_common.py`:
  - `from datetime import datetime` 未使用
  - `total_filtered` 死变量
  - `total_simple` 死变量
  - `simple_counts_total` 死变量
- `07_noisy_retrieval/noisy_syntax_depth_eval_common.py`:
  - `from collections import defaultdict` 未使用
  - `from typing import Optional` 未使用
- `13_query_template/common/template_query.py`: datetime、Optional 未使用（原 iter #29 已修复）
- `13_query_template/common/generate_template_cache.py`: Tuple、load_template_user_queries 未使用（原 iter #29 已修复）
- `13_query_template/common/cache_path_overrides.py`: Optional 未使用（原 iter #29 已修复）
- `13_query_template/common/template_noisy_query.py`: datetime 未使用（原 iter #29 已修复）
- `13_query_template/common/user_data_loader.py`: os、sys 未使用（原 iter #29 已修复）

## §B 不合理逻辑发现

无新发现。

## §C 可合并文件发现

无。

## §D 本轮已实施改动

- 修复 `02_writing_analysis/common/extract_errors_common.py`:
  - 删除 `from datetime import datetime`
  - 删除第 2 个重复 `from pathlib import Path`
  - 删除函数内 `import string`
  - 修正 APOSTROPHES 字面量 → 使用常量 APOSTROPHES（行 67 定义）
  - `LLMClient` → `MiniMaxAnthropicClient`
  - `print(f"[LLM_ERROR]...)` → `print("[LLM_ERROR]...)`
  - 删除 `output_rows = []` 死变量
- 修复 `02_writing_analysis/common/classify_writing_errors_common.py`:
  - 删除 `from datetime import datetime`
  - 删除 `total_filtered` 死变量
  - 删除 `total_simple` 死变量
  - 删除 `simple_counts_total` 死变量
- 修复 `07_noisy_retrieval/noisy_syntax_depth_eval_common.py`:
  - 删除 `from collections import defaultdict`，改为 `import collections`（pyflakes 仍报未使用，删除整行）
  - 删除 `Optional` from `from typing import Dict, List, Optional, Tuple`

## §E 验证

- `python3 -m py_compile` 全部 3 个文件通过
- `python3 -m pyflakes` 全部 3 个文件 0 warnings

## §F Git Commit

- Stage 02 extract_errors_common.py: 移除 datetime/重复Path/函数内import/APOSTROPHES字面量/LLMClient类型/f-string死变量/output_rows死变量
- Stage 02 classify_writing_errors_common.py: 移除 datetime/total_filtered/total_simple/simple_counts_total死变量
- Stage 07 noisy_syntax_depth_eval_common.py: 移除 defaultdict/Optional未使用import

## §G 下轮建议

- Stage 05/06/07 入口脚本 smoke test
- Stage 03 syntactic analysis 脚本 pyflakes 复查
