# Iteration #14 — Stage 02/05 入口三域合并

**日期**: 2026-07-20
**scope**: Stage 02/05 入口三域合并
**关联 stage**: 02_writing_analysis / 05_inject_noisy

## §A 冗余代码发现
- Stage 02 `02_extract_errors_*` 三域仅 `CATEGORY` 不同
- Stage 05 `05_generate_noisy_queries_by_lambdamart_userbased_*` 三域仅 `CATEGORY` 不同

## §B 不合理逻辑发现
- 无

## §C 可合并文件发现
- Stage 02 → 合并为 `02_extract_errors.py` + `--category`（直接 import 方式）
- Stage 05 → 合并为 `05_generate_noisy_queries_by_lambdamart_userbased.py` + `--category`（runpy 方式）

## §D 本轮已实施改动
- 新增: `02_extract_errors.py` — 统一入口（`--category`），直接 import `extract_and_filter_errors`
- 新增: `05_generate_noisy_queries_by_lambdamart_userbased.py` — 统一入口（`--category`），`runpy` 委托
- 原 6 个域脚本保留

## §E 验证
- `python3 -m py_compile` 两个新文件全部通过
- 外部引用检查: 无任何外部引用

## §F 下轮建议
- Stage 09 query dataset 入口三域合并评估
