# Iteration #06 — Stage 13 write_json* 替换 + Stage 10 瘦身归档

**日期**: 2026-07-20
**scope**: Stage 13 冗余 import 修复 + Stage 10 ablation/可视化脚本归档,向减少代码量方向推进
**关联 stage**: 13 / 10

---

## §A 冗余代码发现

### [P1] Stage 13 `template_query.py` 本地 `write_json_atomic` ✅ 已替换

- `13_query_template/common/template_query.py:223` 仍有本地 `write_json_atomic` 定义(iter 02 应替换但漏了)
- 处理:添加 `from common_utils import write_json_atomic`,删除本地定义

### [P2] Stage 10 ablation/可视化脚本冗余 ✅ 已归档

- 43 个脚本中 23 个为 ablation/可视化/private,归档到 `archive/`
- 归档文件:
  - `_debug_kl.py`, `_fit_quality.py`, `_kl_groundtruth.py`, `_save_per_user_fit_quality.py`, `_smoke_test_*.py` (7 个 private)
  - `10_plot_*.py`(5 个) + `10_interpretability_*.py`(4 个) + `10_ablation_*.py`(1 个) + `10_compare_*.py`(2 个) + `10_characterize_*.py`(1 个) + `10_smoke_test_gmm.py` + `10_validate_gaussian_assumption.py` (16 个)
- 剩余 20 个为 pipeline 脚本(3×parse_only + 3×query_clustering + 4×query_selection + 3×review_query_alignment + 3×style_vector_probe + 3×train_only + 1×cleanup)

---

## §B 不合理逻辑发现

**无本轮新增。**

---

## §C 可合并文件发现

**无本轮变更。**

---

## §D 本轮已实施改动

- **修改**:
  - `13_query_template/common/template_query.py`:添加 `from common_utils import write_json_atomic`,删除本地定义(约 10 行)
- **归档**:
  - `10_complexity_analysis/archive/`:移入 23 个非 pipeline 文件
  - Stage 10 从 43 → 20 个文件(-23)

---

## §E 验证

- `python3 -m py_compile` 抽检 Stage 10 pipeline 脚本 PASS
- Stage 13 `template_query.py` 语法 PASS

---

## §F 下轮建议

1. **[iter #07] Stage 14 LLM 重排 15 个入口脚本统一为 1 个 + `--query_type` 参数**
2. **[iter #08] 全项目 dead code 扫描**:`vulture` / `pyflakes` 跑一遍
3. **[iter #09] Stage 04 `attribute_helpers.py` 通用函数拆分到 `common_utils.py`**
