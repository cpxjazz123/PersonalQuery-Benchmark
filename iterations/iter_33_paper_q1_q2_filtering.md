# Iteration #33 — 论文 §3.1 retriever架构分析 + Stage 05 dead code

**日期**: 2026-07-20
**scope**: 论文§3.1 retriever架构 + Stage 05 pyflakes复查
**关联 stage**: 05_inject_noisy

## §A 论文发现

- **95th percentile filtering**：论文 §2.2 未找到 95th percentile style filtering 的具体实现代码；全项目 grep 未发现此逻辑
- **E5 对 spelling error 最敏感**：Δ=11.16（论文 §3.2），对应 Stage 07 error injection
- **SPLADE 对 syntactic structure 最敏感**：Δ=9.6（论文 §3.1），对应 Stage 07 noisy query 生成

## §B 代码问题

- `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py`：
  - `import json` / `import os` / `from collections import Counter` / `import lightgbm as lgb` 未使用
  - `get_user_error_words` / `predict_token_scores` 从 token_level 导入但未使用
  - `print(f"\n统计:")` f-string 无占位符
- `05_inject_noisy/common/token_level_lambdamart_user_based.py`：
  - `import sys` / `from pathlib import Path` / `from datetime import datetime` 未使用
  - `user_query_list = []` 死变量（赋值未使用）
  - 两处 `print(f"  跳过: ...")` f-string 无占位符

## §C 本轮已实施改动

- 修复 `apply_lambdamart_userbased_noisy.py`：移除 4 个未使用 import；修正 token_level import 只保留 `tokenize_query, remove_stop_words`；修复 f-string
- 修复 `token_level_lambdamart_user_based.py`：移除 sys/pathlib/datetime 未使用 import；删除 user_query_list 死变量；修复 2 处 f-string

## §D 验证

- `python3 -m py_compile` 两文件通过
- `python3 -m pyflakes` 两文件 0 warnings

## §E Git Commit

- Stage 05 common: apply_lambdamart_userbased_noisy.py + token_level_lambdamart_user_based.py 清理未使用import+死变量+f-string

## §F 下轮建议

- Stage 07/06 入口脚本 smoke test
- Stage 03 syntactic analysis 脚本 pyflakes 复查
- 95th percentile filtering 代码搜索（如果确实不存在则标记为缺失功能）
