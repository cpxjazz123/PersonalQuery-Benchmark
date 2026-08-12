# Iteration #08 — Stage 07 评估 bare `except Exception` 整改

**日期**: 2026-07-20
**scope**: Stage 07 评估 bare except Exception 整改(Rule 7)
**关联 stage**: 07_noisy_retrieval

## §A 冗余代码发现
- 无

## §B 不合理逻辑发现
- [07_generate_noisy_query_cache_{Baby_Products,Grocery_and_Gourmet_Food,Pet_Supplies}.py:行 688] `except Exception as e: log` — 文件删除失败静默吞异常，违反 Rule 7（预期内失败应明确异常类型）

## §C 可合并文件发现
- 三域脚本结构完全相同（仅 `CATEGORY_NAME` 等少量常量不同），理论上可合并为统一入口 + `--category` 参数 —— 但 pipeline 入口依赖这些脚本的文件名，不改

## §D 本轮已实施改动
- 修改: `07_generate_noisy_query_cache_Baby_Products.py` 行 688: `except Exception as e:` → `except OSError as e:`
- 修改: `07_generate_noisy_query_cache_Grocery_and_Gourmet_Food.py` 行 688: 同上
- 修改: `07_generate_noisy_query_cache_Pet_Supplies.py` 行 688: 同上
- 其余 3 处 `except Exception as e: ... raise`（行 570/613/653）已含 `raise`，属预期外失败正确处理，保留不变

## §E 验证
- `python3 -m py_compile` 三文件全部通过
- fs04 验证: 行 688 确认为 OSError

## §F 下轮建议
- Stage 04 `attribute_helpers.py` 通用函数拆分到 `common_utils.py`
- 全项目 dead code 扫描(vulture / pyflakes / grep import)
- Stage 07 三个 `generate_noisy_query_cache_*.py` 合并为统一入口 + `--category` 参数
