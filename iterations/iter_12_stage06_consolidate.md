# Iteration #12 — Stage 06 三域脚本合并

**日期**: 2026-07-20
**scope**: Stage 06 三域脚本合并评估
**关联 stage**: 06_retrieval

## §A 冗余代码发现
- Stage 06 三域脚本 `build_retriever_indices_*` 和 `generate_query_cache_*` 仅 `CATEGORY_NAME` 不同

## §B 不合理逻辑发现
- `fast_fullscale_eval` 三域差异 100+ 处（QUERY_GROUP_FIELD vs DEPTH_GROUP_FIELD、元组格式不同、函数签名不同），逻辑不同，**不可合并**

## §C 可合并文件发现
- `06_build_retriever_indices_{Baby_Products,Grocery_and_Gourmet_Food,Pet_Supplies}.py` → 合并为 `06_build_retriever_indices.py` + `--category`
- `06_generate_query_cache_{Baby_Products,Grocery_and_Gourmet_Food,Pet_Supplies}.py` → 合并为 `06_generate_query_cache.py` + `--category`

## §D 本轮已实施改动
- 新增: `06_build_retriever_indices.py` — 统一入口（`--category`），内部 `runpy.run_path` 调用原域脚本
- 新增: `06_generate_query_cache.py` — 统一入口（`--category`），内部 `runpy.run_path` 调用原域脚本
- 保留原 6 个域脚本（`runpy` 方式依赖它们存在，不删除）
- **净减少: 0 文件（新增2+保留6），但提供统一入口**

## §E 验证
- `python3 -m py_compile` 06_build_retriever_indices.py / 06_generate_query_cache.py — OK
- 外部引用检查: 无任何外部引用被删脚本

## §F 下轮建议
- Stage 03 syntactic_analysis 入口三域合并评估
- Stage 02/05 两 stage 入口三域合并评估
- 全项目继续 dead code 扫描
