# Iteration #13 — Stage 03 三域脚本合并

**日期**: 2026-07-20
**scope**: Stage 03 三域脚本合并
**关联 stage**: 03_syntactic_analysis

## §A 冗余代码发现
- Stage 03 `03_syntactic_analysis_*` 三域仅 `CATEGORY` 不同
- Stage 03 `03_user_average_syntax_depth_*` 三域仅 `CATEGORY` 不同

## §B 不合理逻辑发现
- 无

## §C 可合并文件发现
- `03_syntactic_analysis_{Baby_Products,Grocery_and_Gourmet_Food,Pet_Supplies}.py` → 合并为 `03_syntactic_analysis.py` + `--category`
- `03_user_average_syntax_depth_{Baby_Products,Grocery_and_Gourmet_Food,Pet_Supplies}.py` → 合并为 `03_user_average_syntax_depth.py` + `--category`

## §D 本轮已实施改动
- 新增: `03_syntactic_analysis.py` — 统一入口（`--category`），`runpy` 委托原域脚本
- 新增: `03_user_average_syntax_depth.py` — 统一入口（`--category`），`runpy` 委托原域脚本
- 原 6 个域脚本保留（`runpy` 依赖它们存在）

## §E 验证
- `python3 -m py_compile` 两个新文件全部通过
- 外部引用检查: 无任何外部引用

## §F 下轮建议
- Stage 02/05 入口三域合并评估
- Stage 09 query dataset 入口三域合并评估
