# Iteration #159 — Dashboard stats-by-section panel (mirror iter #157 CLI; 9th distinct panel color)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_stats_by_section_panel()` (per-section max/mean abs_delta + max/mean rel_delta_pct + claim count + vc count, sorted by max abs desc); extend `_smoke_audit_regression.py` to 43 cases (Case AQ)
**prior**: iter #157 (CLI `--stats-by-section`) prints per-paper-section numeric stats,iter #156 dashboard audit-stats panel mirrored iter #155,iter #154 dashboard evidence-coverage panel mirrored iter #152,iter #153 dashboard severity-tier panel mirrored iter #150 → per-section numeric aggregate affordance CLI-only,dashboard reviewers (PI / non-CLI-comfortable) want browser-side mirror

## §A 审稿意见

Iter #157 `--stats-by-section` 输出 §3.3 + Table 2 (4 claims, 6 vcs, max=0.7415 mean=0.2158) + 7 other sections (13 claims, 0 vcs),reviewer 想要 dashboard-side mirror,can't run shell at meeting → "is §3.3 worse than §2.2 by how much?" requires shell access。Iter #153 (severity tier) + iter #154 (evidence coverage) + iter #156 (audit stats) established CLI ↔ dashboard mirror pattern — iter #159 extends to per-section axis。

后果:
1. Paper-erratum meeting: PI opens dashboard on laptop, doesn't have shell → can't see §3.3 max abs / §2.2 (no vcs) in one glance
2. PI has to ask "is §3.3 worse than §2.2?" and someone has to shell out
3. 30+ sec disruption per query

iter #159 fix: 新增 `_render_stats_by_section_panel()` to dashboard — mirror iter #157 logic exactly:
- Per-section table: 8 sections × (Section name + Claims + #Vcs + Max abs + Mean abs + Max rel + Mean rel)
- Sorted by max abs desc (sections with no abs sink to bottom)
- 9th distinct panel background color: `#fff8e1` light amber (vs iter #156 green / iter #153 orange / iter #154 light blue / iter #132 yellow / iter #129 purple)
- Frozen baseline: 8 sections, 17 claims, 6 vcs with abs

## §B 本轮 (iter #159) 改动

### PersoanlQuery/_generate_audit_dashboard.py

#### _render_stats_by_section_panel() function

插入在 `_render_audit_stats_panel` (line 339-422) 之后,`_render_section_breakdown` (line 425) 之前:

```python
def _render_stats_by_section_panel(claims: list) -> str:
    """iter #159: per-paper-section numeric statistics panel (mirror iter #157 CLI).
    Same _safe_abs/_safe_rel logic + same max/mean abs + max/mean rel format."""
    n_total = len(claims)
    if n_total <= 0:
        return ""

    def _safe_abs(vc: dict):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else None

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    per_sec: dict = {}
    for c in claims:
        sec = c.get("section", "(unspecified)")
        bucket = per_sec.setdefault(sec, {"claims": 0, "abs": [], "rel": []})
        bucket["claims"] += 1
        for vc in c.get("value_check_results", []):
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                bucket["abs"].append(ad)
            if rd is not None:
                bucket["rel"].append(rd)

    rows_data = []
    for sec, b in per_sec.items():
        n_abs = len(b["abs"])
        n_rel = len(b["rel"])
        if n_abs:
            max_abs = max(b["abs"])
            mean_abs = sum(b["abs"]) / n_abs
        else:
            max_abs = mean_abs = 0.0
        if n_rel:
            max_rel = max(b["rel"])
            mean_rel = sum(b["rel"]) / n_rel
        else:
            max_rel = mean_rel = 0.0
        rows_data.append((sec, b["claims"], n_abs, n_rel, max_abs, mean_abs, max_rel, mean_rel))
    rows_data.sort(key=lambda r: (-r[4], r[0]))

    rows = []
    for sec, n_c, n_a, n_r, mx_a, mn_a, mx_r, mn_r in rows_data:
        sec_disp = sec if len(sec) <= 30 else sec[:27] + "..."
        mx_a_s = f"{mx_a:.4f}" if n_a else "—"
        mn_a_s = f"{mn_a:.4f}" if n_a else "—"
        mx_r_s = f"{mx_r * 100:.2f}%" if n_r else "—"
        mn_r_s = f"{mn_r * 100:.2f}%" if n_r else "—"
        rows.append(
            f"<tr class='sbs-row' data-section='{sec}'>"
            f"<td class='sbs-section'>{sec_disp}</td>"
            f"<td class='sbs-claims'>{n_c}</td>"
            f"<td class='sbs-vcs'>{n_a}</td>"
            f"<td class='sbs-maxabs'>{mx_a_s}</td>"
            f"<td class='sbs-meanabs'>{mn_a_s}</td>"
            f"<td class='sbs-maxrel'>{mx_r_s}</td>"
            f"<td class='sbs-meanrel'>{mn_r_s}</td></tr>"
        )

    n_sec = len(rows_data)
    n_total_vc_abs = sum(r[2] for r in rows_data)
    return f"""<details class='stats-by-section-panel' open>
<summary><strong>Stats by section</strong> ({n_sec} sections, {n_total} claims, {n_total_vc_abs} vcs with abs) <span class='sbs-hint'>(iter #159; mirror iter #157 CLI --stats-by-section)</span></summary>
<table class='sbs-table'>
<thead><tr><th>Section</th><th>Claims</th><th>#Vcs</th><th>Max abs</th><th>Mean abs</th><th>Max rel</th><th>Mean rel</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

#### _render_html() wiring

```python
# iter #159: stats-by-section panel (per-paper-section numeric statistics).
stats_by_section_html = _render_stats_by_section_panel(claims)
```

body insertion (after `{audit_stats_html}`):
```python
{evidence_coverage_html}
{audit_stats_html}
{stats_by_section_html}
```

#### CSS block (9th distinct panel color)

```css
.stats-by-section-panel {{ background: #fff8e1; padding: 8px 12px; border-radius: 4px; margin: 12px 0; font-size: 13px; }}
.stats-by-section-panel summary {{ cursor: pointer; }}
.sbs-table {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px; }}
.sbs-table th, .sbs-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
.sbs-table th {{ background: #fff8e1; font-weight: bold; }}
.sbs-claims, .sbs-vcs {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sbs-maxabs, .sbs-meanabs, .sbs-maxrel, .sbs-meanrel {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sbs-hint {{ font-size: 11px; color: #888; margin-left: 6px; }}
```

(Note: `#fff8e1` light amber chosen — distinct from iter #156 light green `#e8f5e9`, iter #153 orange `#ffe0b2`, iter #154 light blue `#e1f5fe`, iter #132 yellow `#ffecb3`, iter #129 purple `#f3e5f5`, iter #117 white-default, iter #125 white-default, iter #118 white-default, iter #136 white-default.)

#### meta generator footer

```html
<meta name='generator' content='_generate_audit_dashboard.py (iter #122, #125, #129, #132, #153, #154, #156, #159)'>
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AQ freeze iter #159 behavior:
- Dashboard HTML contains `stats-by-section-panel` CSS class
- "Stats by section" panel header present
- "iter #159" hint marker present
- `0.7415` (RQ3_LLM_Full_Set_94% preservation subclaim) in panel
- `78.38%` max rel in panel
- `sbs-row` per-section row class present
- §3.3 + Table 2 row visible (worst section)
- "Mean abs" + "Max rel" table headers present
- Em-dash for no-abs sections (7 other sections)

docstring + main() 同步更新到 43 cases.

### README.md

Dashboard features bullet +1 line:
```
- Stats-by-section panel (iter #159) — per-paper-section numeric statistics (max/mean abs + max/mean rel + claim count + vc count), 9th panel color (#fff8e1 light amber); mirror of `--stats-by-section` CLI flag (iter #157)
```

## §C 关键改动点

1. **CLI ↔ dashboard mirror symmetry**: iter #131 (status_summary) → iter #132 (dashboard table); iter #150 (severity-tier) → iter #153 (dashboard panel); iter #152 (evidence-coverage) → iter #154 (dashboard panel); iter #155 (audit-stats) → iter #156 (dashboard panel); **iter #157 (stats-by-section) → iter #159 (dashboard panel) ✓ 本轮**。
   
   Mental model: "if CLI flag has numeric aggregate / classification output worth a glance, dashboard should mirror it"。

2. **7-column table layout**: Section + Claims + #Vcs + Max abs + Mean abs + Max rel + Mean rel — same column set as iter #157 CLI output → mental model "for global aggregate → iter #156 panel; for per-section → iter #159 panel; per-section breakdown available in both CLI and dashboard"。

3. **Sort by max abs desc**: §3.3 + Table 2 (with abs) sinks to TOP (highest max=0.7415); other 7 sections (no abs) sink to BOTTOM alphabetically. Reviewer sees worst section first → matches iter #145 `--worst-by-section` mental model.

4. **Defensive `_safe_abs` / `_safe_rel` accessors**: identical to iter #157 CLI — return `None` (not 0.0) when missing → degenerate vc (RQ1_Table1_Hit10 abs_delta=None) correctly excluded from per-section abs aggregate。

5. **Monospace font + tabular-nums**: `ui-monospace, 'SF Mono', Menlo, monospace` + `font-variant-numeric: tabular-nums` → numeric columns align perfectly for visual scan。Same convention as iter #156 audit-stats panel。

6. **`—` em-dash for empty cells**: matches iter #156 audit-stats panel + iter #157 CLI convention → no false `0.0000` implying "we computed mean of zero"。

7. **9th distinct panel background color**: `#fff8e1` light amber (vs iter #156 light green `#e8f5e9` / iter #153 orange `#ffe0b2` / iter #154 light blue `#e1f5fe` / iter #132 yellow `#ffecb3` / iter #129 reverse-section light purple `#f3e5f5`). Semantic alignment: section breakdown = warm/neutral tone for paper organization。

8. **`<details open>` default open**: matches iter #153 + iter #154 + iter #156 — first-load reviewers see per-section stats without needing to click。

9. **Same per-section aggregation logic as iter #157 CLI**: single source of truth → if iter #157 logic changes (e.g. add median abs), dashboard panel can be regenerated with matching logic。

10. **Section name truncation at 30 chars**: `sec_disp = sec if len(sec) <= 30 else sec[:27] + "..."` — handles long section names like "§3.2 + Table 1 (lower panel)" (29 chars, fits; longer would truncate). Matches iter #157 CLI convention。

11. **Frozen baseline renders correctly**: dashboard now shows §3.3 + Table 2 row with max abs=0.7415 + max rel=78.38% + 6 vcs + 4 claims → matches CLI output exactly。Cross-validate Case AQ: html grep finds `0.7415` + `78.38%` + `§3.3 + Table 2` → reviewable in browser。

12. **`#Vcs` header**: explicit "#Vcs" (not ambiguous "vcs") to clarify "count of vcs WITH abs_delta". Matches iter #157 CLI `#vcs` convention。

13. **`data-section` attribute on each row**: `<tr class='sbs-row' data-section='§3.3 + Table 2'>` → enables future JS filtering / sorting hooks if reviewer wants client-side interactivity (currently no JS hooks, but data attribute preserves option)。

14. **8 sections alphabetical sort within "no abs" group**: when max_abs is 0 (no abs), the secondary sort key is alphabetical by section name → deterministic output regardless of claim ordering in JSON. Matches iter #157 CLI convention。

15. **Generator footer updated**: `iter #122, #125, #129, #132, #153, #154, #156, #159` → breadcrumb of all dashboard-affecting iters → reviewer can see dashboard evolution。

16. **No new CLI flag**: pure dashboard affordance (mirror of existing iter #157 CLI) → keeps CLI flag count frozen at 25+2 = 27 (last added iter #158) → no new argparse flag burden。

## §D 测试

```bash
$ python3 PersoanlQuery/_generate_audit_dashboard.py
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html

$ grep -c 'stats-by-section-panel' result/personal_query/iterations/paper_claims_audit_dashboard.html
3

$ python3 PersoanlQuery/_smoke_audit_regression.py
...
Case AQ: dashboard stats-by-section panel (iter #159)
  PASS  dashboard stats-by-section panel shows 8 sections with §3.3 max abs=0.7415 + 78.38% + em-dash for no-abs sections

All 43 cases passed. Audit CLI frozen baseline verified.

$ python3 PersoanlQuery/paper_claims_audit.py --stats-by-section
Stats by section (8 sections, 17 claims, 6 vcs with abs):
  section                         claims  #vcs    max abs   mean abs    max rel   mean rel
  §3.3 + Table 2                       4     6     0.7415     0.2158     78.38%     23.88%
  (infrastructure)                     1     0          —          —          —          —
  §2.1                                 1     0          —          —          —          —
  §2.2                                 7     0          —          —          —          —
  §3.2                                 1     0          —          —          —          —
  §3.2 + Table 1                       1     0          —          —          —          —
  §3.2 + Table 1 (lower panel)         1     0          —          —          —          —
  §3.4 + Table 3                       1     0          —          —          —          —

$ bash PersoanlQuery/_run_audit_ci.sh
=== AUDIT CI PASSED ===
```

Dashboard now shows in browser (anchored after audit-stats panel):
```
▾ Stats by section (8 sections, 17 claims, 6 vcs with abs) (iter #159; mirror iter #157 CLI --stats-by-section)
  Section                       Claims  #Vcs   Max abs   Mean abs   Max rel   Mean rel
  §3.3 + Table 2                    4     6   0.7415    0.2158     78.38%    23.88%
  (infrastructure)                  1     0      —         —          —         —
  §2.1                              1     0      —         —          —         —
  §2.2                              7     0      —         —          —         —
  §3.2                              1     0      —         —          —         —
  §3.2 + Table 1                    1     0      —         —          —         —
  §3.2 + Table 1 (lower panel)      1     0      —         —          —         —
  §3.4 + Table 3                    1     0      —         —          —         —
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~95 lines `_render_stats_by_section_panel` + wiring + CSS + footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AQ, +docstring, +main count to 43)
- 文档: `README.md` (+1 line stats-by-section panel feature)
- 验证: 43/43 cases pass; pre-commit hook freeze 43 invariants ✓
- 命令: `python3 PersoanlQuery/_generate_audit_dashboard.py` (~1s, regenerates dashboard with new panel)

## §F 与 dashboard ↔ CLI mirror timeline 的关系

| iter | CLI flag | dashboard panel | axis |
|------|----------|-----------------|------|
| #131 | `--status-summary` | iter #132 status_summary_table | count by status |
| #133 | `--top N` | iter #133 sort toggle | worst-N sort |
| #150 | `--severity-tier` | iter #153 severity_tier_panel | HIGH/MEDIUM/LOW |
| #152 | `--evidence-coverage` | iter #154 evidence_coverage_panel | 4-bucket coverage |
| #155 | `--audit-stats` | iter #156 audit_stats_panel | global numeric aggregate |
| #157 | `--stats-by-section` | iter #159 stats_by_section_panel ✓ 本轮 | per-paper-section numeric stats |

dashboard panel color timeline:
| iter | panel | bg color |
|------|-------|----------|
| #117 | per-section breakdown | white default |
| #118 | status legend | white default |
| #129 | reverse section index | `#f3e5f5` light purple |
| #132 | status_summary_table | `#ffecb3` yellow |
| #153 | severity_tier_panel | `#ffe0b2` orange |
| #154 | evidence_coverage_panel | `#e1f5fe` light blue |
| #156 | audit_stats_panel | `#e8f5e9` light green |
| #159 | stats_by_section_panel | `#fff8e1` light amber ✓ 本轮 |

regression test coverage timeline:
- **iter #158** — 42 cases
- **iter #159** — 43 cases (+ dashboard stats-by-section panel) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample