# Iteration #118 — Dashboard status legend / help text

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add `_render_status_legend`)
**prior**: dashboard had 7 status colors but no inline description; this iter adds reviewer-facing help text

## §A 审稿意见

iter #108 dashboard 用 7 status colors (verified_value_match / verified /
discrepant / degenerate / partial / unverified / blocked), reviewer 第一次看
dashboard 不知道 "unverified" vs "partial" vs "degenerate" 区别。 需查
`paper_claims_audit.py` 源码才能 understand semantic。

iter #118 fix: collapsible `<details class='status-legend'>` block, 每个 status
列出 color dot + label + 一句话 semantic description。

## §B 本轮 (iter #118) 改动

### _generate_audit_dashboard.py

新增 `_render_status_legend` (~25 lines):

```python
def _render_status_legend() -> str:
    """[iter #118] Inline legend explaining each audit status."""
    legend_items = [
        ("verified_value_match", "File exists + extracted value matches paper within tolerance."),
        ("verified", "File exists; no numerical claim to validate (infrastructure / path claim)."),
        ("discrepant", "File exists + extracted value disagrees with paper (≥ rel_delta threshold)."),
        ("degenerate", "File exists but selector returned no non-NaN values (data lineage broken)."),
        ("partial", "Code referenced but no specific output paths enumerated for verification."),
        ("unverified", "Code referenced; no expected output glob matches actual files (Stage outputs missing)."),
        ("blocked", "Required Stage output missing; audit cannot proceed."),
    ]
    rows = "\n".join(
        f"<li><span class='legend-dot' style='background:{COLOR}'></span>"
        f"<strong>{LABEL}</strong> — {desc}</li>"
        for s, desc in legend_items
    )
    return f"<details class='status-legend'>...</details>"
```

`_render_html` 调用,插在 claims index 后 + summary 前。CSS `.status-legend` (蓝色背景) + `.legend-dot` (彩色圆点)。

## §C 关键改动点

1. **Reviewer self-service**: 不需离开 page 查 source code 或 paper §5 doc,
   inline legend 解释 7 statuses。
2. **Color dot + label + description**: 三层信息 — visual color 快速识别,
   label 知道叫什么,description 知道啥含义。
3. **Details collapsed by default**: 不占 dashboard 主体视觉空间, reviewer
   需要时 click expand。

## §D 测试

Dashboard HTML grep:
```
<details class='status-legend'>
  <li>verified (value match) — File exists + extracted value matches paper within tolerance.</li>
  <li>verified — File exists; no numerical claim to validate (infrastructure / path claim).</li>
  <li>discrepant — File exists + extracted value disagrees with paper (≥ rel_delta threshold).</li>
  <li>degenerate — File exists but selector returned no non-NaN values (data lineage broken).</li>
  <li>partial — Code referenced but no specific output paths enumerated for verification.</li>
  <li>unverified — Code referenced; no expected output glob matches actual files (Stage outputs missing).</li>
  <li>blocked — Required Stage output missing; audit cannot proceed.</li>
</details>
```

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+`_render_status_legend` + CSS + body insertion)
- 验证: legend renders 7 statuses + descriptions + dots ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 dashboard evolution 的关系

dashboard UX 演进:
- **#108** — sortable table + per-claim detail
- **#116** — anchors + claims index (navigation)
- **#117** — section breakdown (granular)
- **#118** — status legend (semantic help) ✓ 本轮

reviewer 看 dashboard 现在能:
1. **导航**: 17 anchors + index sidebar (iter #116)
2. **总览**: 7 summary badges + section breakdown (iter #108 + #117)
3. **语义**: legend 解释 7 statuses (iter #118)
4. **provenance**: expected_outputs + code_evidence + matched_files (iter #112 + #115)

完整自解释 (self-contained): reviewer 不需离开 page, 不需查 source, 不需读 paper §5。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample