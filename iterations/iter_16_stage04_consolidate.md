# Iteration #16 — Stage 04/01 合并 + 全项目三域入口汇总

**日期**: 2026-07-20
**scope**: Stage 04/01 入口三域合并
**关联 stage**: 04_query / 01_preference_extraction

## §A 冗余代码发现
- Stage 04 `04_generate_by_syntax_depth_no_depth_check_10_*` 仅 `CATEGORY` 不同
- Stage 01 `01_extract_preferences_*` 仅 `CATEGORY` 不同（Baby 含婴儿产品 filter 注释块属于数据集内容差异，非脚本逻辑差异）

## §B 不合理逻辑发现
- 无（用户要求即使仅域不同也强制合并）

## §C 可合并文件发现
- Stage 04 → `04_generate_by_syntax_depth_no_depth_check_10.py` + `--category`
- Stage 01 → `01_extract_preferences.py` + `--category`

## §D 本轮已实施改动
- 新增: `04_generate_by_syntax_depth_no_depth_check_10.py` — 统一入口
- 新增: `01_extract_preferences.py` — 统一入口

## §E 验证
- `python3 -m py_compile` 两个新文件全部通过

## §F 下轮建议
- Stage 12 三域脚本合并评估
- Stage 07/11 不可合并（已确认）
