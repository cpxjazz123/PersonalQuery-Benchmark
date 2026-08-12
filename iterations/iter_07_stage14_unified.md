# Iteration #07 — Stage 14 入口统一

**日期**: 2026-07-20
**scope**: Stage 14 入口脚本三域合并
**关联 stage**: 14_llm_rerank

## §A 冗余代码发现
- Stage 14 原 21 个入口脚本（5 query_type × 3 域 + 3 build_bge_cache + 3 eval），结构高度重复，仅 `CATEGORY` / `query_types=` / `smoke_limit=` 参数不同

## §B 不合理逻辑发现
- `eval_runner.evaluate_for_category` 硬编码 `["correct", "noisy"]`，其他 query_type 无法评测

## §C 可合并文件发现
- 21 个入口脚本 → 3 个统一入口（`14_run_llm_rerank.py` / `14_eval_rerank.py` / `14_build_bge_query_cache.py`）+ `eval_runner.py` 加 `query_types` 参数

## §D 本轮已实施改动
- 删除: 20 个域专用入口脚本
- 新增: `14_run_llm_rerank.py` — 统一入口，`--category` + `--query-type` + `--smoke-limit`
- 新增: `14_eval_rerank.py` — 统一入口，`--category` + `--query-types`
- 新增: `14_build_bge_query_cache.py` — 统一入口，`--category`
- 修改: `common/eval_runner.py` — `evaluate_for_category` 加 `query_types` 参数（默认 `["correct", "noisy"]`）
- **净减少: 20 个文件**

## §E 验证
- `python3 -m py_compile` 全部通过
- 外部引用检查: 无任何外部引用

## §F 下轮建议
- Stage 07 评估 `bare except Exception` 分预期内外处理
- Stage 04 `attribute_helpers.py` 通用函数拆分到 `common_utils.py`
- 全项目 dead code 扫描
