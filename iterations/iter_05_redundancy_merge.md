# Iteration #05 — Stage 02/05 冗余合并

**日期**: 2026-07-20
**scope**: 合并 Stage 02 两文件 + Stage 05 edit_distance 双实现,向减少代码量方向推进
**关联 stage**: 02 / 05

---

## §A 冗余代码发现

### [P1] Stage 02 两文件 60% 重叠 ✅ 已合并

- `extract_errors_common.py`(462 行) + `classify_writing_errors_common.py`(198 行) → `extract_errors_common.py` 作为规范实现,后者变为 4 行 stub
- `extract` 合并内容:
  - 补入 `from common_utils import log`(iter 02 未生效的 patch)
  - 补入 `classify` 的 `_edit_distance`(extract 原本没有,用了 Stage 02 `_edit_distance` 版本)
  - `is_simple_error` 升级为 `classify` 完整版本(含 `edit_dist_le2`/`quote_only`/`char_subset` 等分支)
- `classify` → 4 行 stub,完全向后兼容

### [P1] Stage 05 `edit_distance` 双实现合并 ✅ 已合并

- `apply_lambdamart_userbased_noisy.py` 和 `token_level_lambdamart_user_based.py` 的 `edit_distance` **完全相同**
- `apply` 中删除本地定义,改为从 `token_level` import,净减少 ~16 行

---

## §B 不合理逻辑发现

**无本轮新增。**

---

## §C 可合并文件发现

**无本轮变更。**

---

## §D 本轮已实施改动

- **修改**:
  - `02_writing_analysis/common/extract_errors_common.py`:+`_edit_distance`,升级`is_simple_error`,补`from common_utils import log`
  - `02_writing_analysis/common/classify_writing_errors_common.py`:→4 行 stub(向后兼容入口)
  - `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py`:删除本地`edit_distance`,改 import
- **删除**:Stage 05 `edit_distance` 重复定义 1 份(~16 行)
- **净减少**:约 194 行(classify 删 194 行 + extract 增完整版 16 行 + apply 删 16 行 ≈ 净减 194 行)

---

## §E 验证

- `python3 -m py_compile` 全部 PASS:
  - `02_writing_analysis/common/extract_errors_common.py`
  - `02_writing_analysis/common/classify_writing_errors_common.py`
  - `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py`
  - `05_inject_noisy/common/token_level_lambdamart_user_based.py`

---

## §F 下轮建议

1. **[iter #06] Stage 13 `template_query.py` + `template_noisy_query.py` 公共函数合并**
2. **[iter #07] Stage 10 ablation 脚本瘦身**:约 40+ 文件归档为 1 份 ablation 报告生成器
3. **[iter #08] Stage 14 LLM 重排 15 个入口脚本统一为 1 个 + `--query_type` 参数**
