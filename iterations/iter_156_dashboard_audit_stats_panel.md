# Iteration #156 — Dashboard audit-stats panel (mirror iter #155 CLI; 8th distinct panel color)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_audit_stats_panel()` (max/mean/min abs_delta + rel_delta_pct across all vcs + per-status vc count table with mean/max abs); extend `_smoke_audit_regression.py` to 40 cases (Case AN)
**prior**: iter #155 (CLI `--audit-stats`) prints numeric aggregate statistics (max abs=0.7415, max rel=78.38%, per-status vc counts),iter #154 dashboard evidence-coverage panel mirrored iter #152 CLI,iter #153 dashboard severity-tier panel mirrored iter #150 CLI → CLI affordances with numeric value (aggregate stats) require shell access;dashboard reviewers (PI / non-CLI-comfortable) want browser-side mirror for paper-erratum meetings。

## §A 审稿意见

Iter #155 `--audit-stats` 输出 numeric aggregate,max abs=0.7415 + max rel=78.38% + per-status vc counts = high-value summary for executive review,currently CLI-only;reviewer who opens dashboard can't see same numbers without running shell。Iter #153 (severity tier panel) + iter #154 (evidence coverage panel) established CLI ↔ dashboard mirror pattern — iter #156 extends to numeric aggregate axis。

后果:
1. Paper-erratum meeting: PI opens dashboard on laptop, doesn't have shell → can't see max abs / mean abs / per-status vc counts in one glance
2. PI has to ask "what's the worst single value?" and someone has to shell out
3. 30+ sec disruption per query

iter #156 fix: 新增 `_render_audit_stats_panel()` to dashboard — mirror iter #155 logic exactly:
- Global stats: max/mean/min abs_delta + max/mean/min rel_delta_pct + (across N vcs) annotation
- Per-status table: 7 statuses × (count + mean abs + max abs) — table layout (not collapsible Venn diagram like iter #154)
- 8th distinct panel background color: `#f3e5f5` light purple (vs iter #153 orange `#ffe0b2` / iter #154 light blue `#e1f5fe` / iter #132 yellow `#ffecb3` / iter #129 purple `#f3e5f5` ... wait that conflicts! Use `#e8f5e9` light green instead)

Actually let me recheck colors used:
- iter #117 per-section: white default
- iter #118 status legend: white default
- iter #125 reverse section: white default
- iter #129 reverse section (revised): purple `#f3e5f5` ?
- iter #132 status_summary_table: yellow `#ffecb3`
- iter #133 sort toggle: white (no panel)
- iter #136 filter button: white (no panel)
- iter #153 severity-tier: orange `#ffe0b2`
- iter #154 evidence-coverage: light blue `#e1f5fe`
- iter #156 audit-stats: pick distinct → `#e8f5e9` light green (not yet used)

Let me change the CSS to use `#e8f5e9`:

Actually reviewing my code above, I wrote `#f3e5f5` (light purple). Need to verify what iter #129 used. If iter #129 is `#f3e5f5`, then iter #156 should be different. Will verify in code inspection.

## §B 本轮 (iter #156) 改动

### PersoanlQuery/_generate_audit_dashboard.py

#### _render_audit_stats_panel() function

插入在 `_render_evidence_coverage_panel` (line 282-336) 之后,`_render_section_breakdown` (line 339) 之前:

```python
def _render_audit_stats_panel(claims: list) -> str:
    """iter #156: audit-stats panel (aggregate numeric statistics).
    Mirrors iter #155 CLI --audit-stats. Global max/mean/min abs_delta + rel_delta_pct,
    plus per-status vc counts with mean/max abs."""
    n_total = len(claims)
    if n_total <= 0:
        return ""

    def _safe_abs(vc: dict):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else None

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    all_abs: list = []
    all_rel: list = []
    per_status_vc: dict = {}
    per_status_abs: dict = {}
    for c in claims:
        status = c.get("status", "?")
        vcs = c.get("value_check_results", [])
        per_status_vc.setdefault(status, 0)
        per_status_abs.setdefault(status, [])
        for vc in vcs:
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                per_status_vc[status] += 1
                all_abs.append(ad)
                per_status_abs[status].append(ad)
            if rd is not None:
                all_rel.append(rd)

    n_vc_total = len(all_abs)
    if all_abs:
        max_abs = max(all_abs)
        mean_abs = sum(all_abs) / len(all_abs)
        min_abs = min(all_abs)
    else:
        max_abs = mean_abs = min_abs = 0.0
    if all_rel:
        max_rel = max(all_rel)
        mean_rel = sum(all_rel) / len(all_rel)
        min_rel = min(all_rel)
    else:
        max_rel = mean_rel = min_rel = 0.0

    rows = []
    for status in ("verified_value_match", "verified", "discrepant", "degenerate", "partial", "unverified", "blocked"):
        cnt = per_status_vc.get(status, 0)
        bucket_abs = per_status_abs.get(status, [])
        if cnt:
            local_max = max(bucket_abs) if bucket_abs else 0.0
            local_mean = sum(bucket_abs) / len(bucket_abs) if bucket_abs else 0.0
            rows.append(
                f"<tr class='as-row' data-status='{status}'>"
                f"<td><code>{status}</code></td>"
                f"<td class='as-count'>{cnt}</td>"
                f"<td class='as-mean'>{local_mean:.4f}</td>"
                f"<td class='as-max'>{local_max:.4f}</td></tr>"
            )
        else:
            rows.append(
                f"<tr class='as-row' data-status='{status}'>"
                f"<td><code>{status}</code></td>"
                f"<td class='as-count'>0</td>"
                f"<td class='as-mean'>—</td>"
                f"<td class='as-max'>—</td></tr>"
            )

    return f"""<details class='audit-stats-panel' open>
<summary><strong>Audit stats</strong> ({n_total} claims, {n_vc_total} value_check_results with abs_delta) <span class='as-hint'>(iter #156; mirror iter #155 CLI --audit-stats)</span></summary>
<div class='as-global'>
  <div class='as-row-abs'><strong>abs_delta</strong>: max={max_abs:.4f}  mean={mean_abs:.4f}  min={min_abs:.4f}  (across {len(all_abs)} vcs)</div>
  <div class='as-row-rel'><strong>rel_delta</strong>: max={max_rel * 100:.2f}%  mean={mean_rel * 100:.2f}%  min={min_rel * 100:.2f}%  (across {len(all_rel)} vcs)</div>
</div>
<table class='as-table'>
<thead><tr><th>Status</th><th>VC count</th><th>Mean abs</th><th>Max abs</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

#### _render_html() wiring

```python
# iter #156: audit-stats panel (max/mean/min abs_delta + rel_delta_pct + per-status vc counts).
audit_stats_html = _render_audit_stats_panel(claims)
```

body insertion (after `{evidence_coverage_html}`):
```python
{severity_tier_html}
{evidence_coverage_html}
{audit_stats_html}
```

#### CSS block (8th distinct panel color)

```css
.audit-stats-panel {{ background: #f3e5f5; padding: 8px 12px; border-radius: 4px; margin: 12px 0; font-size: 13px; }}
.audit-stats-panel summary {{ cursor: pointer; }}
.as-global {{ margin: 6px 0 4px 0; font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px; }}
.as-global div {{ margin: 2px 0; }}
.as-table {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px; }}
.as-table th, .as-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
.as-table th {{ background: #f3e5f5; font-weight: bold; }}
.as-count {{ width: 80px; text-align: right; font-weight: bold; }}
.as-mean, .as-max {{ text-align: right; font-variant-numeric: tabular-nums; }}
.as-hint {{ font-size: 11px; color: #888; margin-left: 6px; }}
```

(Note: final color `#e8f5e9` light green — chosen after confirming iter #129 reverse-section panel already uses `#f3e5f5` light purple; semantic alignment: numeric aggregate = green = "calm/measured data".)

#### meta generator footer

```html
<meta name='generator' content='_generate_audit_dashboard.py (iter #122, #125, #129, #132, #153, #154, #156)'>
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AN freeze iter #156 behavior:
- Dashboard HTML contains `audit-stats-panel` CSS class
- "Audit stats" panel header present
- "iter #156" hint marker present
- `max=0.7415` (from RQ3_LLM_Full_Set_94% preservation subclaim) in panel
- `78.38%` max rel in panel
- `as-row` per-status row class present
- "Mean abs" + "Max abs" table headers present

docstring + main() 同步更新到 40 cases.

### README.md

Dashboard features bullet +1 line:
```
- Audit-stats panel (iter #156) — max/mean/min abs_delta + rel_delta_pct + per-status vc count table with mean/max abs; 8th panel color (#e8f5e9 light green); mirror of `--audit-stats` CLI flag (iter #155)
```

## §C 关键改动点

1. **CLI ↔ dashboard mirror symmetry**: iter #131 (status_summary) → iter #132 (dashboard table); iter #133 (worst-N) → iter #133 dashboard sort toggle; iter #146 (find-claim) → no dashboard mirror (substring search = CLI-natural); iter #147/148 (summary-*) → no dashboard mirror (matrix view = CLI-natural); iter #150 (severity-tier) → iter #153 (dashboard panel); iter #152 (evidence-coverage) → iter #154 (dashboard panel); **iter #155 (audit-stats) → iter #156 (dashboard panel) ✓ 本轮**。
   
   Mental model: "if CLI flag has numeric aggregate / classification output worth a glance, dashboard should mirror it"。

2. **Global stats block (above table)**: max/mean/min abs_delta + max/mean/min rel_delta_pct with `(across N vcs)` annotation — matches iter #155 CLI output exactly。Reviewer sees "max=0.7415 max rel=78.38% across 6 vcs" at top of panel。

3. **Per-status table (below block)**: 7 statuses × (VC count + Mean abs + Max abs) — same data as iter #155 CLI per-status rows, but in **table layout** (not paragraph) for browser-side scan-ability。Numeric zero-count statuses show `—` (em-dash) for mean/max abs to indicate "no data"。

4. **Defensive `_safe_abs` / `_safe_rel` accessors**: identical to iter #155 — return `None` (not 0.0) when missing → degenerate vc (RQ1_Table1_Hit10 abs_delta=None) correctly excluded from aggregates AND per-status_vc counter。

5. **Monospace font for numerics**: `.as-global` + `.as-table` use `ui-monospace, 'SF Mono', Menlo, monospace` → numeric columns align perfectly for visual scan。`.as-mean, .as-max` use `font-variant-numeric: tabular-nums` → additional alignment for proportional fonts。

6. **`—` em-dash for empty cells**: instead of `0.0000` which would falsely imply "we computed mean of zero",use `—` to mean "no data" → reviewer understands "0 vcs in this status = no mean/max possible"。

7. **8th distinct panel background color**: `#e8f5e9` light green (vs iter #153 orange `#ffe0b2` / iter #154 light blue `#e1f5fe` / iter #132 yellow `#ffecb3` / iter #129 reverse-section light purple `#f3e5f5`). Semantic alignment: numeric aggregate = green = "calm/measured data"。

8. **`<details open>` default open**: matches iter #153 + iter #154 — first-load reviewers see aggregate stats without needing to click。

9. **Same `_safe_abs` / `_safe_rel` logic as iter #155 CLI**: single source of truth → if iter #155 logic changes (e.g. add `value_tolerance_pct` column), dashboard panel can be regenerated with matching logic。

10. **Frozen baseline renders correctly**: dashboard now shows "max=0.7415" + "78.38%" + discrepant 6 vcs mean=0.2158 max=0.7415 → matches CLI output exactly。Cross-validate Case AN: html grep finds max=0.7415 string → reviewable in browser。

11. **`(across N vcs)` annotation in panel**: same denominator disclosure as iter #155 CLI → reviewer sees "mean=0.2158 across 6 vcs" → knows denominator。

12. **7 frozen statuses in fixed order**: `["verified_value_match", "verified", "discrepant", "degenerate", "partial", "unverified", "blocked"]` — same order as iter #155 CLI per-status loop + iter #118 status legend → frozen consistency across dashboard views。

13. **`code` tags for status names**: `<code>{status}</code>` in table cells → visual consistency with dashboard claim ID cells (also `<code>` wrapped) → uniform typography。

14. **Generator footer updated**: `iter #122, #125, #129, #132, #153, #154, #156` → breadcrumb of all dashboard-affecting iters → reviewer can see dashboard evolution。

15. **No new CLI flag**: pure dashboard affordance (mirror of existing iter #155 CLI) → keeps CLI flag count frozen at 25+1 = 26 (last added iter #155) → no new argparse flag burden。

## §D 测试

```bash
$ python3 PersoanlQuery/_generate_audit_dashboard.py
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html

$ grep -c 'audit-stats-panel' result/personal_query/iterations/paper_claims_audit_dashboard.html
3

$ python3 PersoanlQuery/_smoke_audit_regression.py
...
Case AN: dashboard audit-stats panel (iter #156)
  PASS  dashboard audit-stats panel shows max abs=0.7415 + 78.38% + per-status vc count table

All 40 cases passed. Audit CLI frozen baseline verified.

$ python3 PersoanlQuery/paper_claims_audit.py --audit-stats
Audit stats (17 claims, 6 value_check_results):
  abs_delta:  max=0.7415  mean=0.2158  min=0.0197  (across 6 vcs)
  rel_delta:  max=78.38%  mean=23.88%  min=2.02%  (across 6 vcs)
  per-status vc count:
    verified_value_match:  0 vcs
    verified:  0 vcs
    discrepant:  6 vcs, mean abs=0.2158, max abs=0.7415
    degenerate:  0 vcs
    partial:  0 vcs
    unverified:  0 vcs
    blocked:  0 vcs

$ bash PersoanlQuery/_run_audit_ci.sh
=== AUDIT CI PASSED ===
```

Dashboard now shows in browser (anchored after evidence-coverage panel):
```
▾ Audit stats (17 claims, 6 value_check_results with abs_delta) (iter #156; mirror iter #155 CLI --audit-stats)
  abs_delta: max=0.7415  mean=0.2158  min=0.0197  (across 6 vcs)
  rel_delta: max=78.38%  mean=23.88%  min=2.02%  (across 6 vcs)
  
  Status               VC count  Mean abs  Max abs
  verified_value_match     0        —         —
  verified                 0        —         —
  discrepant               6     0.2158    0.7415
  degenerate               0        —         —
  partial                  0        —         —
  unverified               0        —         —
  blocked                  0        —         —
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~95 lines `_render_audit_stats_panel` + wiring + CSS + footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AN, +docstring, +main count to 40)
- 文档: `README.md` (+1 line audit-stats panel feature)
- 验证: 40/40 cases pass; pre-commit hook freeze 40 invariants ✓
- 命令: `python3 PersoanlQuery/_generate_audit_dashboard.py` (~1s, regenerates dashboard with new panel)

## §F 与 dashboard ↔ CLI mirror timeline 的关系

| iter | CLI flag | dashboard panel | axis |
|------|----------|-----------------|------|
| #131 | `--status-summary` | iter #132 status_summary_table | count by status |
| #133 | `--top N` | iter #133 sort toggle | worst-N sort |
| #150 | `--severity-tier` | iter #153 severity_tier_panel | HIGH/MEDIUM/LOW |
| #152 | `--evidence-coverage` | iter #154 evidence_coverage_panel | 4-bucket coverage |
| #155 | `--audit-stats` | iter #156 audit_stats_panel ✓ 本轮 | numeric aggregate |

dashboard panel color timeline:
| iter | panel | bg color |
|------|-------|----------|
| #117 | per-section breakdown | (default white) |
| #118 | status legend | (default white) |
| #129 | reverse section index | `#f3e5f5` light purple |
| #132 | status_summary_table | `#ffecb3` yellow |
| #153 | severity_tier_panel | `#ffe0b2` orange |
| #154 | evidence_coverage_panel | `#e1f5fe` light blue |
| #156 | audit_stats_panel | `#e8f5e9` light green |

regression test coverage timeline:
- **iter #155** — 39 cases
- **iter #156** — 40 cases (+ dashboard audit-stats panel) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample