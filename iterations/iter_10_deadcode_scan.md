# Iteration #10 — 全项目 dead code 扫描（pyflakes）

**日期**: 2026-07-20
**scope**: 全项目 dead code 扫描
**关联 stage**: 跨 stage

## §A 冗余代码发现
- [12_prf/common/prf_syntax_depth_eval_common.py:行 47-80] 9 个从 `noisy_syntax_depth_eval_common` 导入但未在本地使用的符号（iter #04 monkey-patch 链消除遗留）
- [14_llm_rerank/common/llm_rerank_common.py:行 25/41] `time` import 和 `get_category_config` import 未使用

## §B 不合理逻辑发现
- Stage 13 `template_noisy_query.py` 行 22/276：`sys` 重定义

## §C 可合并文件发现
- 无新发现

## §D 本轮已实施改动
- 修复: `12_prf/common/prf_syntax_depth_eval_common.py`：删除 9 个未使用 import
- 修复: `14_llm_rerank/common/llm_rerank_common.py`：删除 `import time`、`from config import get_category_config` 及空的 sys.path 追加

## §E 验证
- `python3 -m py_compile` Stage 12/14 两个文件全部通过，已同步 fs04

## §F 下轮建议
- Stage 13 `template_noisy_query.py` sys 重定义修复
- Stage 01 preference_extraction 入口三域合并
- Stage 06/08 检索公共逻辑合并
