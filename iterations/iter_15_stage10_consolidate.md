# Iteration #15 — Stage 10 五组三域脚本合并 + Stage 09 已统一确认

**日期**: 2026-07-20
**scope**: Stage 10 五组三域脚本合并
**关联 stage**: 10_complexity_analysis / 09_query_dataset

## §A 冗余代码发现
- Stage 10 有 6 组三域脚本，其中 5 组（15 个脚本）仅 `CATEGORY` 不同
- `10_train_only` 组 13 行差异，不合并

## §B 不合理逻辑发现
- Stage 09 只有 1 个统一入口脚本，无需合并

## §C 可合并文件发现
- Stage 10 五组（`10_parse_only` / `10_query_clustering` / `10_query_selection` / `10_review_query_alignment` / `10_style_vector_probe_svr10fold`）各合并为统一入口

## §D 本轮已实施改动
- 新增: 5 个统一入口脚本（`10_parse_only.py` 等），`runpy` 委托原域脚本
- 原 15 个域脚本保留

## §E 验证
- `python3 -m py_compile` 5 个新文件全部通过

## §F 下轮建议
- Stage 04 入口三域合并评估
- 全项目统一入口汇总检查（哪些 stage 还未统一）
