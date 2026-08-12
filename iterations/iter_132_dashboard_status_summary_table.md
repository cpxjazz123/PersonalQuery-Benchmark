# Iteration #132 — Dashboard `status_summary_table` panel

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add tabular status breakdown panel; extend `_smoke_audit_regression.py` to 15 cases (Case O)
**prior**: dashboard header had a single "summary" badge row with 7 colored chips showing counts only (no percentages, no visual bars, no tabular breakdown). Reviewer wanted to "see which status is flakiest at a glance" in a browser tab without scrolling or hovering. The compact 1-line `--status-summary` (iter #131) was shell-friendly but had no equivalent visual affordance in the dashboard.

## §A 审稿意见

Dashboard 现有 summary header 只 show 7 colored chips with counts (e.g. `verified 6` / `discrepant 4` / ...). Reviewer 想 browser scan "which status dominates?" 必须 hover + tooltip + mental math:

- No percentage column: relative weight of each status not immediately visible (e.g. `verified 6` looks comparable to `discrepant 4` until you count 17 total and compute 35.3% / 23.5%).
- No bar visualization: chip size alone doesn't communicate proportional weight.
- No tabular breakdown: all 7 statuses crammed into one badge row — visually competes with the per-claim table below it.
- `verified_value_match` 和 `blocked` always show 0 — visually identical to non-zero status badges, reviewer must read the number.

后果: reviewer 在 browser 看 dashboard 没法 quick scan "is `discrepant` > 30% of total?" — 必须 external calculator 或 grep audit JSON。

iter #132 fix: dashboard header 下方加 `<details class='status-summary-table' open>` 表格 panel,7 rows × 4 columns (status label + count + percent + bar),yellow `#fff8e1` background 让视觉区分 from blue summary chips。Row order freeze audit status order (`verified_value_match` / `verified` / `discrepant` / `degenerate` / `partial` / `unverified` / `blocked`),bar width = percent of 17 total。

## §B 本轮 (iter #132) 改动

### PersoanlQuery/_generate_audit_dashboard.py

`_render_status_summary_table(summary, n_total)` 新函数 (~30 lines),接 `(summary_dict, n_total_claims)` 返回 HTML fragment:

```python
def _render_status_summary_table(summary: dict, n_total: int) -> str:
    """iter #132: tabular breakdown of audit status × count × percent × bar."""
    if n_total <= 0:
        return ""
    rows = []
    for status in ["verified_value_match", "verified", "discrepant",
                   "degenerate", "partial", "unverified", "blocked"]:
        count = summary.get(status, 0)
        label, color = STATUS_COLORS.get(status, (status, "#9e9e9e"))
        pct = (count / n_total) * 100 if n_total > 0 else 0.0
        bar_width = max(0, min(100, pct))
        rows.append(
            f"<tr><td><span class='legend-dot' style='background:{color}'></span>"
            f"<strong>{label}</strong></td>"
            f"<td class='sst-count'>{count}</td>"
            f"<td class='sst-pct'>{pct:.1f}%</td>"
            f"<td class='sst-bar-cell'><div class='sst-bar' "
            f"style='width:{bar_width:.1f}%; background:{color}'></div></td></tr>"
        )
    return f"""<details class='status-summary-table' open>
<summary><strong>Status summary</strong> ({n_total} total claims)</summary>
<table class='sst-table'>
<thead><tr><th>Status</th><th class='sst-count'>Count</th>
<th class='sst-pct'>Percent</th><th class='sst-bar-cell'>Distribution</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

CSS additions (yellow `#fff8e1` background to differentiate from blue summary chips):
```css
.status-summary-table { background: #fff8e1; padding: 8px 12px;
    border-radius: 4px; margin: 12px 0; border: 1px solid #ffe082; }
.status-summary-table summary { cursor: pointer; font-size: 14px; }
.sst-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
.sst-table th { background: #ffecb3; padding: 4px 6px; text-align: left; }
.sst-table td { padding: 4px 6px; border-top: 1px solid #ffe082; }
.sst-count { width: 60px; text-align: right; font-weight: bold; }
.sst-pct { width: 60px; text-align: right; color: #555; }
.sst-bar-cell { width: 40%; }
.sst-bar { height: 12px; border-radius: 2px; min-width: 1px; }
```

Body wire (在 summary badges `<div class='summary'>{summary_html}</div>` 之后插入):
```python
status_summary_table_html = _render_status_summary_table(summary, n_total)
```

Generator footer 更新: `iter #108, #129, #132`。

### PersoanlQuery/_smoke_audit_regression.py

加 Case O freeze iter #132 behavior:
```python
def case_o_dashboard_status_summary_table():
    """Case O (iter #132): dashboard HTML embeds status_summary_table with 7 rows + counts + percents + bars."""
    html = dash.read_text(encoding="utf-8")
    assert "status-summary-table" in html
    assert "Status summary" in html
    assert "sst-table" in html
    for label in ["verified", "discrepant", "degenerate", "partial", "unverified"]:
        assert f"<strong>{label}</strong>" in html
    assert "35.3%" in html  # 6/17 verified
    assert "23.5%" in html  # 4/17 discrepant
    assert "29.4%" in html  # 5/17 unverified
    assert html.count("sst-bar") >= 7
```

docstring + main() 同步更新到 15 cases。

## §C 关键改动点

1. **Tabular not just chip**: 从 inline badge row 升级到 `<table>` with 4 columns (status / count / percent / bar). Reviewer 可以 sort/compare visually,no hover needed.

2. **Yellow background `#fff8e1`**: 视觉 distinct from blue summary chips (header) and white claim cards (body),panel 自成一个 focus zone。`#ffecb3` header row + `#ffe082` border 协调。

3. **`open` attribute**: `<details open>` 默认展开 — reviewer 第一次看 dashboard 立即看到 status breakdown,不用 click 展开。

4. **Bar width = percent**: `bar_width = min(100, pct)`,bar background = `STATUS_COLORS[status]` 让 bar 颜色跟 legend dot 一致。Reviewer 一眼看出 `verified` 35% bar 比 `discrepant` 23% bar 长。

5. **Frozen percent values in test**: `35.3%` (6/17) / `23.5%` (4/17) / `29.4%` (5/17) explicit assert — if audit JSON flips status counts,test fires as baseline flip detector。

6. **`verified_value_match` + `blocked` rows still rendered** even when count=0 — 表格完整性,不会因 count=0 skip row 让 reviewer 误以为 status missing。

7. **`min-width: 1px` on `.sst-bar`**: 即使 0% 也保留 1px 视觉 hint,不会消失成空白 cell。

## §D 测试

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'                                PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965               PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message                                 PASS
Case D: --strict --json-only (full audit) expects exit 1                                         PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)                             PASS
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)                        PASS
Case G: dashboard HTML has anchors + status legend (iter #121)                                   PASS
Case H: dashboard HTML <title> + <meta name='description'> (iter #122)                           PASS
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123, #126, #127)               PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                        PASS
Case K: dashboard <link rel=alternate> -> paper_audit_id_mapping.json (iter #125)                PASS
Case L: mapping sidecar reverse_section_index (iter #128)                                        PASS
Case M: dashboard reverse-section panel (iter #129)                                              PASS
Case N: --status-summary compact 1-line (iter #131)                                              PASS
Case O: dashboard status_summary_table panel (iter #132)                                         PASS

All 15 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~30 lines `_render_status_summary_table` + ~15 lines CSS + body wire + footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case O, +docstring, +main count to 15)
- 验证: 15/15 cases pass; pre-commit hook freeze 15 invariants ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh`

## §F 与 audit infra timeline 的关系

Dashboard features timeline:
- **iter #108** — initial 7 color-coded chips
- **iter #109** — rel_delta as percent
- **iter #110** — value_check panel with severity
- **iter #112** — provenance panels per claim
- **iter #113** — generated_at timestamp in header
- **iter #115** — matched_files for degenerate claims
- **iter #116** — HTML anchors + claims-index sidebar
- **iter #117** — per-section breakdown table
- **iter #118** — inline status legend
- **iter #122** — `<title>` + `<meta description>` with counts
- **iter #125** — `<link rel=alternate>` to mapping JSON
- **iter #129** — reverse-section panel
- **iter #132** — status_summary_table panel ✓ 本轮

regression test coverage timeline:
- **iter #131** — 14 cases
- **iter #132** — 15 cases (+ status_summary_table) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample