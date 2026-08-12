# Iteration #25 — Stage 10 工具函数合并

**日期**: 2026-07-20
**scope**: Stage 10 重复工具函数合并到 stage10_common.py
**关联 stage**: 10_complexity_analysis

## §A 冗余代码发现
- `summarize_array`：在 4 个文件中完全相同（~14行 × 4 = ~56行）
- `load_json`：在 2 个文件中略有差异（有无验证），合并为最严格版本
- `load_jsonl`：在 2 个文件中相同（~14行 × 2 = ~28行）
- `write_jsonl`：在 4 个文件中基本相同（~6行 × 4 = ~24行）
- 合计约 ~120 行重复

## §B 不合理逻辑发现
无。

## §C 可合并文件发现
- `summarize_array`/`load_json`/`load_jsonl`/`write_jsonl` 在 Stage 10 的 4 个 common/ 文件中重复定义，应合并

## §D 本轮已实施改动
- 新建: `10_complexity_analysis/common/stage10_common.py`：统一 `summarize_array`、`load_json`、`load_jsonl`、`write_jsonl` 四个工具函数（48行）
- 修改: `cluster_strict5550_query_gmm_and_attach_retrieval.py`：删除 3 个本地定义，改为从 `stage10_common` 导入
- 修改: `evaluate_review_query_alignment.py`：删除 3 个本地定义，改为从 `stage10_common` 导入
- 修改: `evaluate_vades_style_vector_probe_svr10fold.py`：删除 3 个本地定义，改为从 `stage10_common` 导入
- 修改: `train_vades_lite_sentence_latent_threshold.py`：删除 3 个本地定义，改为从 `stage10_common` 导入

## §E 验证
- `python3 -m py_compile` 全部 5 个文件通过
- 代码行数净减少约 72 行（~120 - 48）

## §F 下轮建议
- 检查 `10_complexity_analysis/archive/` 目录是否有被 pipeline 引用的 ablation 脚本
- Stage 04/13 config.py 重复函数合并评估

## §F Git Commit
- `git commit -m "iter #25: Stage 10 四文件重复工具函数合并到stage10_common.py,净减少~72行"`
