# Iteration #18 — 删除冗余域专用脚本 + Stage 13 合并

**日期**: 2026-07-20
**scope**: 删除已合并的域专用脚本
**关联 stage**: 跨 stage

## §A 冗余代码发现
- 约 80 个域专用脚本（`*_Baby_Products.py` / `*_Grocery_and_Gourmet_Food.py` / `*_Pet_Supplies.py`）在合并为统一入口后应删除

## §B 不合理逻辑发现
- Stage 07 `07_generate_noisy_query_cache_*` 不可合并（BM25 预建索引机制复杂，iter #09 已确认）
- Stage 00 `00_batch_prepare_data_*` 有 gzip 处理函数等差异，非简单 category 不同
- Stage 10 `10_train_only_*` 有 13 行逻辑差异

## §C 可合并文件发现
- Stage 13：6 组（18 个脚本）全部仅 CATEGORY 不同，已合并并删除原脚本

## §D 本轮已实施改动
- 删除：Stage 01-06/10部分/11-14 约 80 个域专用脚本
- 新增：Stage 13 六组统一入口（6 个文件）
- Stage 07/00/10 train_only 保留原脚本（功能性差异）

## §E 验证
- 各统一入口 `python3 -m py_compile` 通过
- 剩余域专用脚本确认：Stage 07（6个）/ Stage 00（3个）/ Stage 10 train_only（3个）

## §F 下轮建议
- 全项目汇总确认
- 检查是否有 pipeline 入口脚本仍引用已删除的域专用脚本
