# Iteration #09 — Stage 07 死代码清理 + 三域合并评估

**日期**: 2026-07-20
**scope**: Stage 07 三域脚本合并评估 + 死代码清理
**关联 stage**: 07_noisy_retrieval / 04_query

## §A 冗余代码发现
- Stage 07 `07_generate_noisy_query_cache_Baby_Products.py` 行 420-427：**死代码**，两段相同 return 块，第二段永不可达

## §B 不合理逻辑发现
- Stage 07 三域脚本结构**极为复杂**：包含 ColBERTv2 编译环境配置、stdout/stderr 重定向、预建 BM25 索引加载、torch 多卡配置等，与 `CATEGORY_NAME` 紧耦合
- 强制合并会破坏 BM25 预建索引加载逻辑（`load_retriever_for_query_encoding` 路径含 category 名）
- **结论：Stage 07 三域脚本暂不合并，保持现状**

## §C 可合并文件发现
- Stage 04 `attribute_helpers.py`：无通用函数（全部是 attr 规范化/变体匹配/5属性验证领域逻辑），不适合迁到 `common_utils.py`

## §D 本轮已实施改动
- 修复: `07_generate_noisy_query_cache_Baby_Products.py` 行 424-427：删除死代码（重复 return 块）

## §E 验证
- `python3 -m py_compile` 通过
- 已同步 fs04

## §F 下轮建议
- 全项目 dead code 扫描(vulture)
- Stage 01 preference_extraction 入口三域合并
