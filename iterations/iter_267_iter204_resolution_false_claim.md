# Iteration 267 — iter #204 "resolution" 虚假：cluster-mean approximation 未解决 query-level 问题

**日期**: 2026-07-22
**角色**: NLP/IR 专业审稿人
**scope**: iter #204 自称 "resolved" lower-panel formula，但代码实现是 cluster-mean approximation，且因 Stage 09 all-NaN 而完全失效

---

## §A iter #204 做了什么 vs 声称做了什么

### iter #204 实际实现
`08_compare_p10_across_domains.py:_compute_error_effect_delta_per_retriever`:
```python
"""iter #204: Compute lower-panel error-effect Δ Range per retriever.
Cluster-mean approximation using Stage 9 pivot data."""  # ← 明确说 cluster-mean approximation
```

### iter #204 声称做了什么
Paper line 139 footnote: "**iter #204 resolution**: The lower-panel Δ Range is computed at the **query level**"

### 矛盾本质
1. 代码注释明确说是 "cluster-mean approximation"
2. footnote 声称是 "iter #204 resolution" — 暗示 query-level 问题被解决了
3. 实际上 iter #204 既没有实现 query-level aggregation，也没有解决原来的 mismatch

### iter #204 §D 虚假声明
iter #204 §D 影响表称:
> "Audit RQ2_Table1_Drop status: discrepant → **resolved-by-iter-#204-code-impl-plus-paper-text-clarification**"

但实际上：
- iter #253 发现 Stage 09 all-NaN → RQ2_Table1_Drop status 变为 **degenerate**
- "resolved" 状态被 iter #253 回滚了
- iter #204 的 cluster-mean approximation 实际上从未产生有效值（因为 Stage 09 all-NaN）

---

## §B 问题总结

| 问题 | 描述 |
|------|------|
| iter #204 resolution 虚假 | 代码是 cluster-mean approximation，footnote 声称 query-level resolved |
| RQ2_Table1_Drop 状态回滚 | iter #204 claimed resolved，iter #253 reverted to degenerate |
| footnote 前后矛盾 | footnote 自己承认 pipeline 是 cluster-mean，却声称 query-level resolution |
| 三个独立的 iter 都在产生"resolved"状态，但没有真正解决 | iter #204 → iter #253 → iter #266 footnote 矛盾 |

---

## §C §11 Backlog 条目更新建议

建议在 §11 添加新 P0 条目：
- **[Major] iter #204 "resolution" 虚假承诺**：iter #204 声称 resolved lower-panel formula，但代码是 cluster-mean approximation，且因 Stage 09 all-NaN 而完全失效；iter #204 §D 影响表声称 RQ2_Table1_Drop 从 discrepant → resolved，但 iter #253 回滚为 degenerate

---

## §D 下一步

**needs input**: 同 iter #266 — 方案A（retract iter #204 resolution 声称）还是方案B（真正实现 query-level aggregation 并重新跑 Stage 09）？
