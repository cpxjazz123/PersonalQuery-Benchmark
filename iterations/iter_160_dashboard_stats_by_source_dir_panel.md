# Iteration #160 — Dashboard stats-by-source-dir panel (mirror iter #158 CLI; 10th distinct panel color)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_stats_by_source_dir_panel()` (per-source-dir max/mean abs_delta + max/mean rel_delta_pct + claim count + vc count, sorted by max abs desc); extend `_smoke_audit_regression.py` to 44 cases (Case AR)
**prior**: iter #158 (CLI `--stats-by-source-dir`) prints per-source-dir numeric stats (02_writing_analysis max=0.7415 + agreement_metrics.py max=0.2700 + 6 other dirs 0 vcs),iter #159 dashboard stats-by-section panel mirrored iter #157,iter #156 dashboard audit-stats panel mirrored iter #155 → per-source-dir numeric aggregate affordance CLI-only,dashboard reviewers want browser-side mirror for code-side erratum triage.

## §A 审稿意见

Iter #158 `--stats-by-source-dir` 输出 02_writing_analysis (1 claim, 3 vcs, max=0.7415 mean=0.2783) + agreement_metrics.py (3 claims, 3 vcs, max=0.2700 mean=0.1534) + 6 other dirs (13 claims, 0 vcs),reviewer 想要 dashboard-side mirror,can't run shell at meeting → "fix Stage 02 first or agreement_metrics.py first?" requires shell access。Iter #156 (audit-stats global) + iter #159 (stats-by-section) + iter #160 (stats-by-source-dir) complete the CLI↔dashboard mirror for numeric-stats axes。

后果:
1. Code-side erratum meeting: PI opens dashboard on laptop, doesn't have shell → can't see 02_writing_analysis vs agreement_metrics.py in one glance
2. PI has to ask "which dir should we fix first?" and someone has to shell out
3. 30+ sec disruption per query

iter #160 fix: 新增 `_render_stats_by_source_dir_panel()` to dashboard — mirror iter #158 logic exactly:
- 7-column table (Source dir + Claims + #Vcs + Max abs + Mean abs + Max rel + Mean rel)
- Sorted by max abs desc (worst dir first, no-abs sink to bottom)
- 10th distinct panel background color: `#fce4ec` light pink (vs iter #159 amber / iter #156 green / iter #153 orange / iter #154 light blue / iter #132 yellow / iter #129 purple)
- Frozen baseline: 8 dirs, 17 claims, 6 vcs with abs

## §B 本轮 (iter #160) 改动

### PersoanlQuery/_generate_audit_dashboard.py

#### _render_stats_by_source_dir_panel() function

插入在 `_render_stats_by_section_panel` (line 425-497) 之后,`_render_section_breakdown` (line 500) 之前:

```python
def _render_stats_by_source_dir_panel(claims: list) -> str:
    """iter #160: per-source-dir numeric statistics panel (mirror iter #158 CLI).
    Reuses iter #149 _primary_dir extraction. Sorted by max abs desc. Em-dash for no-abs dirs."""
    n_total = len(claims)
    if n_total <= 0:
        return ""

    def _primary_dir(c: dict) -> str:
        evidences = c.get("code_evidence") or []
        for ref in evidences:
            parts = ref.split("/")
            if len(parts) >= 2 and parts[0] in ("PersoanlQuery", "personelquery"):
                d = parts[1].split(":")[0]
                return d if d else "(top-level)"
        return "(no source dir)"

    def _safe_abs(vc: dict):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else None

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    per_dir: dict = {}
    for c in claims:
        d = _primary_dir(c)
        bucket = per_dir.setdefault(d, {"claims": 0, "abs": [], "rel": []})
        bucket["claims"] += 1
        for vc in c.get("value_check_results", []):
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                bucket["abs"].append(ad)
            if rd is not None:
                bucket["rel"].append(rd)

    rows_data = []
    for d, b in per_dir.items():
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
        rows_data.append((d, b["claims"], n_abs, n_rel, max_abs, mean_abs, max_rel, mean_rel))
    rows_data.sort(key=lambda r: (-r[4], r[0]))

    rows = []
    for d, n_c, n_a, n_r, mx_a, mn_a, mx_r, mn_r in rows_data:
        d_disp = d if len(d) <= 32 else d[:29] + "..."
        mx_a_s = f"{mx_a:.4f}" if n_a else "—"
        mn_a_s = f"{mn_a:.4f}" if n_a else "—"
        mx_r_s = f"{mx_r * 100:.2f}%" if n_r else "—"
        mn_r_s = f"{mn_r * 100:.2f}%" if n_r else "—"
        rows.append(
            f"<tr class='sbd-row' data-dir='{d}'>"
            f"<td class='sbd-dir'><code>{d_disp}</code></td>"
            f"<td class='sbd-claims'>{n_c}</td>"
            f"<td class='sbd-vcs'>{n_a}</td>"
            f"<td class='sbd-maxabs'>{mx_a_s}</td>"
            f"<td class='sbd-meanabs'>{mn_a_s}</td>"
            f"<td class='sbd-maxrel'>{mx_r_s}</td>"
            f"<td class='sbd-meanrel'>{mn_r_s}</td></tr>"
        )

    n_dir = len(rows_data)
    n_total_vc_abs = sum(r[2] for r in rows_data)
    return f"""<details class='stats-by-source-dir-panel' open>
<summary><strong>Stats by source dir</strong> ({n_dir} dirs, {n_total} claims, {n_total_vc_abs} vcs with abs) <span class='sbd-hint'>(iter #160; mirror iter #158 CLI --stats-by-source-dir)</span></summary>
<table class='sbd-table'>
<thead><tr><th>Source dir</th><th>Claims</th><th>#Vcs</th><th>Max abs</th><th>Mean abs</th><th>Max rel</th><th>Mean rel</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

#### _render_html() wiring

```python
# iter #160: stats-by-source-dir panel (per-source-dir numeric statistics).
stats_by_source_dir_html = _render_stats_by_source_dir_panel(claims)
```

body insertion (after `{stats_by_section_html}`):
```python
{stats_by_section_html}
{stats_by_source_dir_html}
```

#### CSS block (10th distinct panel color)

```css
.stats-by-source-dir-panel {{ background: #fce4ec; padding: 8px 12px; border-radius: 4px; margin: 12px 0; font-size: 13px; }}
.stats-by-source-dir-panel summary {{ cursor: pointer; }}
.sbd-table {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px; }}
.sbd-table th, .sbd-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
.sbd-table th {{ background: #fce4ec; font-weight: bold; }}
.sbd-claims, .sbd-vcs {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sbd-maxabs, .sbd-meanabs, .sbd-maxrel, .sbd-meanrel {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sbd-hint {{ font-size: 11px; color: #888; margin-left: 6px; }}
```

(Note: `#fce4ec` light pink chosen — distinct from iter #159 light amber `#fff8e1`, iter #156 light green `#e8f5e9`, iter #153 orange `#ffe0b2`, iter #154 light blue `#e1f5fe`, iter #132 yellow `#ffecb3`, iter #129 purple `#f3e5f5`.)

#### meta generator footer

```html
<meta name='generator' content='_generate_audit_dashboard.py (iter #122, #125, #129, #132, #153, #154, #156, #159, #160)'>
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AR freeze iter #160 behavior:
- Dashboard HTML contains `stats-by-source-dir-panel` CSS class
- "Stats by source dir" panel header present
- "iter #160" hint marker present
- `0.7415` (02_writing_analysis LLM_Full_Set_94% preservation) in panel
- `78.38%` max rel in panel
- `02_writing_analysis` row visible (worst dir)
- `agreement_metrics.py` row visible (#2 worst dir)
- `sbd-row` per-dir row class present
- "Mean abs" + "Max rel" table headers present
- Em-dash for no-abs dirs (6 other dirs)

docstring + main() 同步更新到 44 cases.

### README.md

Dashboard features bullet +1 line:
```
- Stats-by-source-dir panel (iter #160) — per-source-dir numeric statistics (max/mean abs + max/mean rel + claim count + vc count), 10th panel color (#fce4ec light pink); mirror of `--stats-by-source-dir` CLI flag (iter #158)
```

## §C 关键改动点

1. **Completes CLI↔dashboard mirror symmetry for numeric-stats axes**: iter #155 (global) → iter #156 (dashboard panel); iter #157 (per-section) → iter #159 (dashboard panel); **iter #158 (per-source-dir) → iter #160 (dashboard panel) ✓ 本轮**。
   
   Mental model: "for global numeric aggregate → iter #156 panel; for per-section → iter #159 panel; for per-source-dir → iter #160 panel"。

2. **Reuse iter #149 `_primary_dir()` helper**: extracts first `PersoanlQuery/X/...` ref, strips `:function_name` suffix → top-level utility files (e.g. `agreement_metrics.py`) bucket together. Same logic as iter #158 CLI / iter #149 / iter #144 → consistent source-dir labeling。

3. **7-column table layout**: Source dir + Claims + #Vcs + Max abs + Mean abs + Max rel + Mean rel — same column set as iter #158 CLI output → mental model "per-source-dir breakdown available in both CLI and dashboard"。

4. **Sort by max abs desc**: 02_writing_analysis (max=0.7415) sinks to TOP; agreement_metrics.py (max=0.2700) #2; 6 other dirs (no abs) sink to BOTTOM alphabetically. Reviewer sees worst dir first → matches iter #149 `--worst-by-source-dir` mental model。

5. **Defensive `_safe_abs` / `_safe_rel` accessors**: identical to iter #158 CLI — return `None` (not 0.0) when missing → degenerate vc correctly excluded from per-dir abs aggregate。

6. **Monospace font + tabular-nums**: `ui-monospace, 'SF Mono', Menlo, monospace` + `font-variant-numeric: tabular-nums` → numeric columns align perfectly for visual scan。Same convention as iter #156 / iter #159 panels。

7. **`—` em-dash for empty cells**: matches iter #158 CLI + iter #156 / iter #159 dashboard convention → no false `0.0000` implying "we computed mean of zero"。

8. **10th distinct panel background color**: `#fce4ec` light pink (vs iter #159 amber `#fff8e1` / iter #156 green `#e8f5e9` / iter #153 orange `#ffe0b2` / iter #154 light blue `#e1f5fe` / iter #132 yellow `#ffecb3` / iter #129 purple `#f3e5f5`). Semantic alignment: source-dir breakdown = warm/neutral tone for code organization。

9. **`<code>` tags wrap dir names**: `<code>{d_disp}</code>` in table cells → visual consistency with dashboard claim ID cells + iter #156 audit-stats status name cells → uniform typography。

10. **`<details open>` default open**: matches iter #153 + iter #154 + iter #156 + iter #159 — first-load reviewers see per-source-dir stats without needing to click。

11. **Same per-source-dir aggregation logic as iter #158 CLI**: single source of truth → if iter #158 logic changes (e.g. add median abs), dashboard panel can be regenerated with matching logic。

12. **Dir name truncation at 32 chars**: `d_disp = d if len(d) <= 32 else d[:29] + "..."` — handles long dir names like `PersoanlQuery/10_complexity_analysis` (already short at 22 chars; longer would truncate). Matches iter #158 CLI convention。

13. **Frozen baseline renders correctly**: dashboard now shows 02_writing_analysis row with max abs=0.7415 + max rel=78.38% + 3 vcs + 1 claim + agreement_metrics.py row with max abs=0.2700 + max rel=30.34% + 3 vcs + 3 claims → matches CLI output exactly。Cross-validate Case AR: html grep finds `0.7415` + `78.38%` + `02_writing_analysis` + `agreement_metrics.py` → reviewable in browser。

14. **`#Vcs` header**: explicit "#Vcs" (not ambiguous "vcs") to clarify "count of vcs WITH abs_delta". Matches iter #158 CLI `#vcs` convention。

15. **`data-dir` attribute on each row**: `<tr class='sbd-row' data-dir='02_writing_analysis'>` → enables future JS filtering / sorting hooks if reviewer wants client-side interactivity (currently no JS hooks, but data attribute preserves option)。

16. **8 dirs alphabetical sort within "no abs" group**: when max_abs is 0 (no abs), the secondary sort key is alphabetical by dir name → deterministic output regardless of claim ordering in JSON. Matches iter #158 CLI convention。

17. **Generator footer updated**: `iter #122, #125, #129, #132, #153, #154, #156, #159, #160` → breadcrumb of all dashboard-affecting iters → reviewer can see dashboard evolution。

18. **No new CLI flag**: pure dashboard affordance (mirror of existing iter #158 CLI) → keeps CLI flag count frozen at 25+3 = 28 (last added iter #158) → no new argparse flag burden。

## §D 测试

```bash
$ python3 PersoanlQuery/_generate_audit_dashboard.py
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html

$ grep -c 'stats-by-source-dir-panel' result/personal_query/iterations/paper_claims_audit_dashboard.html
3

$ python3 PersoanlQuery/_smoke_audit_regression.py
...
Case AR: dashboard stats-by-source-dir panel (iter #160)
  PASS  dashboard stats-by-source-dir panel shows 8 dirs with 02_writing_analysis max abs=0.7415 + 78.38% + em-dash for no-abs dirs

All 44 cases passed. Audit CLI frozen baseline verified.

$ python3 PersoanlQuery/paper_claims_audit.py --stats-by-source-dir
Stats by source dir (8 dirs, 17 claims, 6 vcs with abs):
  source dir                        claims  #vcs    max abs   mean abs    max rel   mean rel
  02_writing_analysis                    1     3     0.7415     0.2783     78.38%     29.34%
  agreement_metrics.py                   3     3     0.2700     0.1534     30.34%     18.43%
  00_data_preparation                    1     0          —          —          —          —
  04_query                               2     0          —          —          —          —
  05_inject_noisy                        1     0          —          —          —          —
  07_noisy_retrieval                     1     0          —          —          —          —
  08_compare_all_domain                  3     0          —          —          —          —
  10_complexity_analysis                 5     0          —          —          —          —

$ bash PersoanlQuery/_run_audit_ci.sh
=== AUDIT CI PASSED ===
```

Dashboard now shows in browser (anchored after stats-by-section panel):
```
▾ Stats by source dir (8 dirs, 17 claims, 6 vcs with abs) (iter #160; mirror iter #158 CLI --stats-by-source-dir)
  Source dir                  Claims  #Vcs   Max abs   Mean abs   Max rel   Mean rel
  02_writing_analysis             1     3   0.7415    0.2783     78.38%    29.34%
  agreement_metrics.py            3     3   0.2700    0.1534     30.34%    18.43%
  00_data_preparation             1     0      —         —          —         —
  04_query                        2     0      —         —          —         —
  05_inject_noisy                 1     0      —         —          —         —
  07_noisy_retrieval              1     0      —         —          —         —
  08_compare_all_domain           3     0      —         —          —         —
  10_complexity_analysis          5     0      —         —          —         —
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~95 lines `_render_stats_by_source_dir_panel` + wiring + CSS + footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AR, +docstring, +main count to 44)
- 文档: `README.md` (+1 line stats-by-source-dir panel feature)
- 验证: 44/44 cases pass; pre-commit hook freeze 44 invariants ✓
- 命令: `python3 PersoanlQuery/_generate_audit_dashboard.py` (~1s, regenerates dashboard with new panel)

## §F 与 dashboard ↔ CLI mirror timeline 的关系

| iter | CLI flag | dashboard panel | axis |
|------|----------|-----------------|------|
| #131 | `--status-summary` | iter #132 status_summary_table | count by status |
| #133 | `--top N` | iter #133 sort toggle | worst-N sort |
| #150 | `--severity-tier` | iter #153 severity_tier_panel | HIGH/MEDIUM/LOW |
| #152 | `--evidence-coverage` | iter #154 evidence_coverage_panel | 4-bucket coverage |
| #155 | `--audit-stats` | iter #156 audit_stats_panel | global numeric aggregate |
| #157 | `--stats-by-section` | iter #159 stats_by_section_panel | per-paper-section numeric stats |
| #158 | `--stats-by-source-dir` | iter #160 stats_by_source_dir_panel ✓ 本轮 | per-source-dir numeric stats |

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
| #159 | stats_by_section_panel | `#fff8e1` light amber |
| #160 | stats_by_source_dir_panel | `#fce4ec` light pink ✓ 本轮 |

regression test coverage timeline:
- **iter #159** — 43 cases
- **iter #160** — 44 cases (+ dashboard stats-by-source-dir panel) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample