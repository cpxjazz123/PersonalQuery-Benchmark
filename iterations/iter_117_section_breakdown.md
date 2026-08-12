# Iteration #117 — Dashboard per-section breakdown sub-table

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add `_render_section_breakdown`)
**prior**: aggregate summary (#108) showed total counts; this iter breaks down by paper section

## §A 审稿意见

iter #108 dashboard summary header 显示 7 个 aggregate count badges (6 verified
/ 4 discrepant / 1 degenerate / 1 partial / 5 unverified),但 reviewer 翻 paper §3.3
想知道 "§3.3 + Table 2 这一节里 audit 怎么样" 时,只能数 dashboard table 里
匹配 section string 的 row,慢。

需求: per-section breakdown sub-table,每行一个 paper section (e.g. §3.3 + Table 2),
每列一个 status (7 cols),cell 显示该 section 该 status 的 count + 颜色编码。
Reviewer 一眼看出 §3.3 + Table 2 行有 4 discrepant cells,说明这一节 audit
最 flaky。

## §B 本轮 (iter #117) 改动

### _generate_audit_dashboard.py

新增 `_render_section_breakdown` (~30 lines):

```python
def _render_section_breakdown(claims):
    """[iter #117] Per-section breakdown: which paper section has how many of each status."""
    from collections import Counter
    by_section = {}
    for c in claims:
        sec = c.get("section", "(unspecified)")
        by_section.setdefault(sec, Counter())[c["status"]] += 1
    rows = []
    for sec in sorted(by_section.keys()):
        cells = " ".join(
            f"<span class='sec-cell' style='color:{COLOR}' title='{s}'>{N}</span>"
            for s in STATUS_ORDER
        )
        rows.append(f"<tr><td><code>{sec}</code></td><td>{cells}</td></tr>")
    return f"<details class='section-breakdown' open>...</details>"
```

`_render_html` 调用 + 插在 Summary section 下方 + Claims table 上方。

CSS `.section-breakdown`, `.sec-table`, `.sec-cell` (table layout for 8 sections × 7 status cols)。

## §C 关键改动点

1. **Open by default (`<details open>`)**: reviewer 第一次看 dashboard 就看到
   breakdown,不需要 click expand;如果不需要可以 collapse。
2. **Sorted by section name**: deterministic order,便于 reviewer 知道 §3.x
   章节 cluster 在一起。
3. **Color-coded cells**: 每个 cell 用 status 颜色 + bold 数字,即使 0 也染色
   (与 SUMMARY badge 一致)。
4. **Compact display**: 8 sections × 7 status cols in one table,<details> 默认
   open 但可折叠。

## §D 测试

Dashboard 验证 (从 HTML grep):
```
§3.3 + Table 2:    0 verified_value_match / 0 verified / 4 discrepant / 0 degenerate / 0 partial / 0 unverified
§3.4 + Table 3:    0 verified_value_match / 0 verified / 0 discrepant / 0 degenerate / 0 partial / 1 unverified
§3.2 + Table 1:    0 / 0 / 0 / 1 degenerate / 0 / 0
§2.2:              0 / 3 verified / 0 / 0 / 0 / 4 unverified
(infrastructure):   0 / 0 / 0 / 0 / 1 partial / 0
```

Reviewer 现在能立即看出:
- **§3.3 + Table 2** 是最 flaky section (4 discrepant 全在这一节)
- **§3.4 + Table 3** 是 Stage 12 gap (1 unverified)
- **§2.2** split verified / unverified (Stage 12 derived)
- **§3.2 + Table 1** 1 degenerate (Stage 6 query pool size)

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+`_render_section_breakdown` + CSS + body insertion)
- 验证: 8 sections × 7 status cells ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 dashboard evolution 的关系

dashboard 演进:
- **#108** — summary header (aggregate)
- **#117** — per-section breakdown (granular) ✓ 本轮

reviewer 看 dashboard 现在能问两种问题:
1. "总体怎么样?" → summary badges (7 数字)
2. "某一节怎么样?" → section breakdown table (8×7 matrix)

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample