# Iteration #24 — 全项目 dead code 复查（iter #23 后续）

**日期**: 2026-07-20
**scope**: 全项目 dead code 复查 + Stage 11 monkey-patch 残留清理
**关联 stage**: 跨 stage（11/llm_client）

## §A 冗余代码发现
- `llm_client.py`：`import json` 顶层未使用（仅 `import json` 顶层声明，但全程无 `json.` 调用）
- `11_preprocess_normalization/common/preprocessed_syntax_depth_eval_common.py`：与 Stage 12 相同的 monkey-patch 链消除残留，13 个从 `noisy_syntax_depth_eval_common` 导入但未使用的符号
- `11_preprocess_normalization/common/preprocessed_cache_generator.py`：`import importlib.util` 未使用
- `03_syntactic_analysis/03_count_acl_ccomp_levels.py`：4 处 f-string 无占位符警告（pyflakes 严格模式，不影响运行）

## §B 不合理逻辑发现
无新发现。

## §C 可合并文件发现
无新发现。

## §D 本轮已实施改动
- 修复: `llm_client.py`：删除未使用的 `import json`
- 修复: `11_preprocess_normalization/common/preprocessed_syntax_depth_eval_common.py`：删除未使用的 `pickle`、`time`、`Optional`；删除 12 个未使用的 `noisy_syntax_depth_eval_common` 导入项（SYNTAX_DEPTH_QUERY_CATEGORY/NOISY_QUERY_TYPE 等）
- 修复: `11_preprocess_normalization/common/preprocessed_cache_generator.py`：删除未使用的 `import importlib.util`

## §E 验证
- `python3 -m py_compile` llm_client.py, Stage 11 两个文件全部通过

## §F 下轮建议
- Stage 03 f-string 严格警告（4处 `f"\n{...}"` 改为普通字符串）
- Stage 05/07/10 入口脚本 smoke test（确认统一入口正常工作）

## §F Git Commit
- `git commit -m "iter #24: Stage 11 monkey-patch残留清理+llm_client未使用import清理"`
