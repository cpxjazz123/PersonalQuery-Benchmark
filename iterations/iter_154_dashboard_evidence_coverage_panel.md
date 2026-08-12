# Iteration #154 — Dashboard evidence-coverage panel (4-bucket expected_outputs × code_evidence)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_evidence_coverage_panel()` function with 4-bucket classification (both/expected_only/evidence_only/neither) with claim ID anchor links; extend `_smoke_audit_regression.py` to 38 cases (Case AL)
**prior**: iter #152 (CLI `--evidence-coverage`) bucketed 17 claims into both(16)/expected_only(0)/evidence_only(1)/neither(0) for audit-scope planning. iter #153 dashboard severity-tier panel mirrored iter #150 CLI for browser triage. Reviewer 想要 dashboard mirror of iter #152 CLI for browser audit-scope planning (no shell needed).

## §A 审稿意见

Reviewer 想要 browser-side audit-scope planning — "open the dashboard, see evidence-coverage buckets, click to claim detail":
1. iter #152 CLI `--evidence-coverage` — works in shell, no anchor links
2. iter #153 dashboard severity-tier — orthogonal axis (severity not evidence coverage)
3. Manual: open audit JSON + mental Venn diagram

后果: audit-scope planning requires shell access; dashboard reviewers can't easily see coverage gaps.

iter #154 fix: add `_render_evidence_coverage_panel()` to `_generate_audit_dashboard.py` — mirror iter #152 logic exactly (`bool(expected_outputs) × bool(code_evidence)` 4-bucket classification),render as light-blue `#e1f5fe` collapsible panel with 4 bucket color dots (both=green, expected_only=blue, evidence_only=orange, neither=gray) + claim IDs as anchor links `<a href='#claim-{cid}'>` (matches iter #116 dashboard anchor format).

## §B 本轮 (iter #154) 改动

### PersoanlQuery/_generate_audit_dashboard.py

新加 `_render_evidence_coverage_panel(claims)` function (~55 lines):
```python
def _render_evidence_coverage_panel(claims: list) -> str:
    """iter #154: evidence-coverage panel (4-bucket expected_outputs × code_evidence).
    Mirrors iter #152 CLI --evidence-coverage. Each bucket has claim ID anchor links
    (iter #116) so reviewer can jump from coverage summary directly to claim detail."""
    n_total = len(claims)
    if n_total <= 0:
        return ""
    both: list = []
    expected_only: list = []
    evidence_only: list = []
    neither: list = []
    for c in claims:
        cid = c.get("id", "?")
        has_expected = bool(c.get("expected_outputs"))
        has_evidence = bool(c.get("code_evidence"))
        bucket_label = (
            "both" if has_expected and has_evidence else
            "expected_only" if has_expected else
            "evidence_only" if has_evidence else
            "neither"
        )
        locals()[bucket_label].append(c)
    bucket_colors = {
        "both": "#388e3c",
        "expected_only": "#1976d2",
        "evidence_only": "#fb8c00",
        "neither": "#9e9e9e",
    }
    rows = []
    for label in ("both", "expected_only", "evidence_only", "neither"):
        bucket = locals()[label]
        if not bucket:
            rows.append(...)  # empty row
        else:
            id_links = ", ".join(f"<a href='#claim-{c['id']}'><code>{c['id']}</code></a>" for c in bucket)
            rows.append(...)
    return f"""<details class='evidence-coverage-panel' open>
<summary><strong>Evidence coverage</strong> ({n_total} total; 4 buckets: expected_outputs × code_evidence) <span class='ec-hint'>(iter #154; mirror iter #152 CLI)</span></summary>
<table class='ec-table'>
<thead><tr><th>Bucket</th><th>Count</th><th>Claims</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

Wire 到 `_render_html` body:
```python
evidence_coverage_html = _render_evidence_coverage_panel(claims)
# ...
{severity_tier_html}
{evidence_coverage_html}    # iter #154
{reverse_section_html}
```

CSS block 加 10 lines:
```css
.evidence-coverage-panel { background: #e1f5fe; padding: 8px 12px; border-radius: 4px; margin: 12px 0; font-size: 13px; }
.evidence-coverage-panel summary { cursor: pointer; }
.ec-table { border-collapse: collapse; width: 100%; margin-top: 6px; }
.ec-table th, .ec-table td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
.ec-table th { background: #e1f5fe; font-weight: bold; }
.ec-count { width: 60px; text-align: right; font-weight: bold; }
.ec-ids a { text-decoration: none; }
.ec-ids a:hover { text-decoration: underline; }
.ec-hint { font-size: 11px; color: #888; margin-left: 6px; }
```

Generator footer updated: `iter #122, #125, #129, #132, #153, #154`.

### PersoanlQuery/_smoke_audit_regression.py

加 Case AL freeze iter #154 dashboard behavior:
- `evidence-coverage-panel` class present in dashboard HTML
- "Evidence coverage" panel header
- "iter #154" marker in panel hint
- All 4 bucket labels (both, expected_only, evidence_only, neither)
- `Pipeline_Skip_ColBERTv2_SPLADE` in evidence_only bucket (only partial claim without expected_outputs)
- `ec-table` class for panel table

docstring + main() 同步更新到 38 cases.

## §C 关键改动点

1. **Dashboard ↔ CLI affordance symmetry** extends iter #153 pattern (matches iter #132 ↔ #131, iter #153 ↔ #150):
   - iter #132 dashboard `status_summary_table` ↔ iter #131 CLI `--status-summary`
   - iter #153 dashboard `severity-tier-panel` ↔ iter #150 CLI `--severity-tier`
   - iter #154 dashboard `evidence-coverage-panel` ↔ iter #152 CLI `--evidence-coverage` ✓ 本轮
   
   Reviewer mental model: 3 paired affordances covering status / severity / evidence-coverage axes。

2. **Same classification logic as iter #152 CLI**: `bool(expected_outputs) × bool(code_evidence)` 4-bucket Venn → single source of truth (CLI + dashboard agree on both=16/expected_only=0/evidence_only=1/neither=0)。

3. **4-bucket color coding**:
   - `both` = green `#388e3c` (fully scoped = good)
   - `expected_only` = blue `#1976d2` (unusual, audit-source unclear)
   - `evidence_only` = orange `#fb8c00` (partial by design, actionable)
   - `neither` = gray `#9e9e9e` (audit-incomplete, warning)
   
   Matches dashboard severity-tier color palette convention (green=good, orange=actionable, red/gray=warning)。

4. **Anchor links to claim detail rows**: `<a href='#claim-{cid}'>` matches iter #116 dashboard anchor format → reviewer can click `Pipeline_Skip_ColBERTv2_SPLADE` in evidence_only bucket → jumps to claim detail row → see partial status reason + code_evidence (no expected_outputs by definition)。

5. **Light-blue `#e1f5fe` background**: 6th distinct panel color:
   - claim-index: white (default)
   - status-legend: light blue
   - section-breakdown: gray `#f5f5f5`
   - status-summary-table: yellow `#fff8e1`
   - reverse-section-panel: purple `#f3e5f5`
   - severity-tier-panel: orange `#ffe0b2`
   - **evidence-coverage-panel: light blue `#e1f5fe`** ✓ 本轮
   
   Visual distinction from iter #153 orange (severity) + iter #132 yellow (status) + iter #129 purple (reverse) → 7 distinct panel colors for 7 distinct info types。

6. **Empty bucket rendering**: `<em>(empty)</em>` text for empty bucket → defensive against future state changes (none expected in current frozen baseline but defensive)。

7. **Open by default** (`<details open>`): matches iter #117/132/129/153 patterns → reviewer sees coverage summary immediately on page load without clicking。

8. **Color-coding dual-purpose**: legend dots (small inline) + table th background (`#e1f5fe` matching panel bg) → consistent visual hierarchy。

9. **`iter #154` generator footer marker**: dashboard `<meta name='generator'>` updated to list all cumulative iters → reviewer can see dashboard is up-to-date with audit infra timeline。

10. **No re-run audit**: dashboard regeneration independent of audit re-run, only reads existing audit JSON → fast (~200ms) full dashboard regen including new panel。

11. **`#claim-{cid}` anchor format compat**: relies on iter #116 claim-row anchor → if iter #116 anchor format changes, this iter #154 panel breaks → single source of truth (anchor pattern) for dashboard deep-linking。

## §D 测试

```bash
$ python3 PersoanlQuery/_generate_audit_dashboard.py
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html

$ grep -c "evidence-coverage-panel\|ec-table\|ec-row" /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html
12   # 1 panel + 1 ec-table + 4 ec-row + 1 CSS + 4 inline references

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AL: dashboard evidence-coverage panel (iter #154)
  PASS  dashboard evidence-coverage panel renders 4 buckets with anchor links to evidence_only claim

All 38 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~75 lines: function + CSS + wire + generator footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AL, +docstring, +main count to 38)
- 验证: 38/38 cases pass; pre-commit hook freeze 38 invariants ✓
- 命令: `python3 PersoanlQuery/_generate_audit_dashboard.py` (~200ms full regen including new panel)

## §F 与 audit infra timeline 的关系

Dashboard panel timeline:
- **iter #108** — initial dashboard
- **iter #117** — per-section breakdown sub-table
- **iter #118** — status legend
- **iter #129** — reverse section index panel
- **iter #132** — `status_summary_table` panel
- **iter #133** — sort-by-severity toggle
- **iter #136** — filter-by-status click handler
- **iter #153** — severity-tier panel
- **iter #154** — evidence-coverage panel ✓ 本轮

Dashboard panels taxonomy (7 distinct background colors):
| panel | bg color | iter |
|-------|----------|------|
| claim-index sidebar | white (default) | #116 |
| status-legend | light blue | #118 |
| section-breakdown | gray `#f5f5f5` | #117 |
| status-summary-table | yellow `#fff8e1` | #132 |
| reverse-section-panel | purple `#f3e5f5` | #129 |
| severity-tier-panel | orange `#ffe0b2` | #153 |
| **evidence-coverage-panel** | **light blue `#e1f5fe`** | **#154** ✓ 本轮 |

Audit CLI ↔ dashboard affordance symmetry:
| view | CLI | dashboard |
|------|-----|-----------|
| raw status | `--status-summary` (iter #131) | `status_summary_table` panel (iter #132) |
| severity tier | `--severity-tier` (iter #150) | `severity-tier-panel` (iter #153) |
| evidence coverage | `--evidence-coverage` (iter #152) | `evidence-coverage-panel` (iter #154) ✓ 本轮 |

regression test coverage timeline:
- **iter #153** — 37 cases
- **iter #154** — 38 cases (+ dashboard evidence-coverage panel) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample