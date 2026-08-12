# Iteration #17 — Stage 11/12 三域脚本合并

**日期**: 2026-07-20
**scope**: Stage 11/12 各3组三域脚本合并
**关联 stage**: 11_preprocess_normalization / 12_prf

## §A 冗余代码发现
- Stage 11: `11_eval_preprocessed_*` / `11_generate_preprocessed_cache_*` / `11_preprocess_query_*` 各仅 `CATEGORY` 不同（0 差异）
- Stage 12: `12_eval_prf_*` / `12_generate_prf_cache_*` / `12_generate_prf_queries_*` 各仅 `CATEGORY` 不同（0 差异）

## §B 不合理逻辑发现
- Stage 07 `07_generate_noisy_query_cache_*` 不可合并（BM25 预建索引加载机制复杂，iter #09 已确认）

## §C 可合并文件发现
- Stage 11 三组 → 各自合并为统一入口
- Stage 12 三组 → 各自合并为统一入口

## §D 本轮已实施改动
- 新增: `11_eval_preprocessed.py` / `11_generate_preprocessed_cache.py` / `11_preprocess_query.py`
- 新增: `12_eval_prf.py` / `12_generate_prf_cache.py` / `12_generate_prf_queries.py`
- 原 18 个域脚本保留

## §E 验证
- `python3 -m py_compile` 6 个新文件全部通过

## §F 下轮建议
- Stage 00 入口三域合并
- 全项目统一入口汇总确认
