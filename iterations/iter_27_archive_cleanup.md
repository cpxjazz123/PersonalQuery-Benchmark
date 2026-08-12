# Iteration #27 — §3.3 可合并文件复查 + dead code 清理

**日期**: 2026-07-20
**scope**: §3.3 可合并文件复查 + Stage 11 dead code 清理
**关联 stage**: 11_preprocess_normalization / 跨 stage

## §A 冗余代码发现
- `11_preprocess_normalization/common/preprocessed_cache_generator.py`：3 个未使用 import（`datetime`、`Tuple`、`PREPROCESSED_QUERY_TYPE`）+ 1 个死变量（`query_cache_base_dir`）
- `PersoanlQuery/03_syntactic_analysis/` 下有 6 个域专用脚本残留（`03_syntactic_analysis_Baby_Products.py` 等），iter #13 后应删除但未清理

## §B 不合理逻辑发现
- Stage 10 archive 目录 23 个 ablation 脚本已与 pipeline 分离，无新发现
- 各 stage 根目录无遗留废弃脚本

## §C 可合并文件发现
无。

## §D 本轮已实施改动
- 修复: `11_preprocess_normalization/common/preprocessed_cache_generator.py`：删除未使用 import（`datetime`、`Tuple`）和未使用导入项（`PREPROCESSED_QUERY_TYPE`）；删除死变量 `query_cache_base_dir`

## §E 验证
- `python3 -m py_compile` 通过

## §F 下轮建议
- 删除 `PersoanlQuery/03_syntactic_analysis/` 下 6 个残留域专用脚本（iter #13 遗留）
- 同步工作树和 fs04 共享目录（iter #23/24 改动尚未完全同步）

## §F Git Commit
- `git commit -m "iter #27: Stage 11清理4个未使用import和死变量;发现03_syntactic_analysis残留域脚本待清理"`
