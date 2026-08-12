# Iteration #04 — Stage 07/11/12/13 eval common 统一

**日期**: 2026-07-20
**scope**: 消除三层 monkey-patch 派生链,在 Stage 07 统一支持 4 种 query_type
**关联 stage**: 07 / 11 / 12

---

## §A 冗余代码发现

**无新增冗余。**

---

## §B 不合理逻辑发现

### [P1] Stage 07/11/12/13 三层 monkey-patch 派生链 ✅ 已修复

- **问题**:Stage 11 `_query_record_for_output_13` monkey-patch Stage 07,Stage 12 `_query_record_for_output_14` 再次 monkey-patch,形成 11→12→07 三层耦合
- **处理**:在 Stage 07 `query_record_for_output` 直接扩展,支持 `correct/noisy/preprocessed/prf` 4 种 query_type
- **Stage 07 改动**:
  - 新增常量 `PREPROCESSED_QUERY_TYPE = "preprocessed"`, `PRF_QUERY_TYPE = "prf"`
  - 扩展 `query_record_for_output` 的 query_text 提取分支
  - 扩展 `get_query_cache_path` 类型校验
- **Stage 11 改动**:删除行 100-136 monkey-patch 块(37 行)
- **Stage 12 改动**:删除行 100-142 monkey-patch 块(43 行)

---

## §C 可合并文件发现

**无本轮变更。**

---

## §D 本轮已实施改动

- **修改**:3 个文件,净减少 ~80 行 monkey-patch 代码
  - `07_noisy_retrieval/noisy_syntax_depth_eval_common.py`:+5 行(常量+分支),扩展 query_record_for_output
  - `11_preprocess_normalization/common/preprocessed_syntax_depth_eval_common.py`:-37 行(删除 monkey-patch)
  - `12_prf/common/prf_syntax_depth_eval_common.py`:-43 行(删除 monkey-patch)

---

## §E 验证

- `python3 -m py_compile` 07/11/12 三个文件 PASS

---

## §F 下轮建议

1. **[iter #05] Stage 02 两文件合并**:`extract_errors_common.py` + `classify_writing_errors_common.py` → `writing_errors_common.py`
2. **[iter #06] Stage 05 `edit_distance` 双实现合并**
3. **[iter #07] Stage 13 `template_query.py` + `template_noisy_query.py` 公共函数合并**
