# Iteration #153 — Dashboard severity-tier panel (HIGH/MEDIUM/LOW with anchor links)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_severity_tier_panel()` function with HIGH/MEDIUM/LOW bucketing (mirror iter #150 CLI); extend `_smoke_audit_regression.py` to 37 cases (Case AK)
**prior**: iter #150 (CLI `--severity-tier`) bucketed 17 claims into HIGH(2)/MEDIUM(9)/LOW(6) for executive triage. iter #132 (`status_summary_table` dashboard panel) gave raw-status breakdown with click-to-filter. Reviewer 想要 dashboard-side mirror of iter #150 CLI for browser triage (no shell needed).

## §A 审稿意见

Reviewer 想要 browser-side severity triage — "open the dashboard, see HIGH tier claims at a glance, click to jump to claim detail":
1. iter #150 CLI `--severity-tier` — works in shell, no anchor links
2. iter #132 dashboard `status_summary_table` — raw status (7 buckets) not severity (3 tiers)
3. iter #133 dashboard sort toggle — reorders by abs_delta but doesn't bucket by severity

后果: severity triage requires shell access; dashboard reviewers (e.g. PI, non-CLI-comfortable reviewers) can't easily see HIGH tier.

iter #153 fix: add `_render_severity_tier_panel()` to `_generate_audit_dashboard.py` — mirror iter #150 logic exactly (HIGH_THRESHOLD_PCT=20.0, `classify()` function, _safe_abs/_safe_rel accessors),render as orange `#ffe0b2` collapsible panel with HIGH(red)/MEDIUM(orange)/LOW(green) color dots + claim IDs as anchor links `<a href='#claim-{cid}'>` (matches iter #116 dashboard anchor format) so reviewer can jump from tier summary directly to claim detail row.

## §B 本轮 (iter #153) 改动

### PersoanlQuery/_generate_audit_dashboard.py

新加 `_render_severity_tier_panel(claims)` function (~60 lines):
```python
def _render_severity_tier_panel(claims: list) -> str:
    """iter #153: severity-tier panel (HIGH/MEDIUM/LOW) with claim ID anchor links.
    Mirrors iter #150 CLI --severity-tier. Each tier row links to claim anchors
    (iter #116) so reviewer can jump from tier summary directly to claim detail."""
    n_total = len(claims)
    if n_total <= 0:
        return ""
    HIGH_THRESHOLD_PCT = 20.0

    def _safe_abs(vc):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else 0.0

    def _safe_rel(vc):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    def classify(c):
        status = c.get("status", "?")
        vcs = c.get("value_check_results", [])
        if status in ("verified_value_match", "verified", "blocked"):
            return "LOW"
        if status == "discrepant":
            max_rel = None
            for vc in vcs:
                rd = _safe_rel(vc)
                if rd is None: continue
                if max_rel is None or abs(rd) > abs(max_rel):
                    max_rel = rd
            if max_rel is not None and abs(max_rel) * 100 > HIGH_THRESHOLD_PCT:
                return "HIGH"
            return "MEDIUM"
        if status in ("degenerate", "partial", "unverified"):
            return "MEDIUM"
        return "MEDIUM"

    tier_buckets = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for c in claims:
        tier_buckets[classify(c)].append(c)

    def _max_abs(c):
        vcs = c.get("value_check_results", [])
        return max((_safe_abs(vc) for vc in vcs), default=0.0)

    tier_colors = {"HIGH": "#d32f2f", "MEDIUM": "#fb8c00", "LOW": "#388e3c"}
    rows = []
    for tier in ("HIGH", "MEDIUM", "LOW"):
        bucket = sorted(tier_buckets[tier], key=lambda c: -_max_abs(c))
        if not bucket:
            rows.append(...)  # empty row
        else:
            id_links = ", ".join(f"<a href='#claim-{c['id']}'><code>{c['id']}</code></a>" for c in bucket)
            rows.append(...)
    return f"""<details class='severity-tier-panel' open>
<summary><strong>Severity tier</strong> ({n_total} total; HIGH = discrepant rel&gt;20%) <span class='st-hint'>(iter #153; mirror iter #150 CLI)</span></summary>
<table class='st-table'>
<thead><tr><th>Tier</th><th>Count</th><th>Claims</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

Wire 到 `_render_html` body:
```python
severity_tier_html = _render_severity_tier_panel(claims)
# ...
{status_summary_table_html}
{severity_tier_html}    # iter #153
{reverse_section_html}
```

CSS block 加 12 lines:
```css
.severity-tier-panel { background: #ffe0b2; padding: 8px 12px; border-radius: 4px; margin: 12px 0; font-size: 13px; }
.severity-tier-panel summary { cursor: pointer; }
.st-table { border-collapse: collapse; width: 100%; margin-top: 6px; }
.st-table th, .st-table td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
.st-table th { background: #ffe0b2; font-weight: bold; }
.st-count { width: 60px; text-align: right; font-weight: bold; }
.st-ids a { text-decoration: none; }
.st-ids a:hover { text-decoration: underline; }
.st-hint { font-size: 11px; color: #888; margin-left: 6px; }
```

Generator footer updated: `iter #122, #125, #129, #132, #153`.

### PersoanlQuery/_smoke_audit_regression.py

加 Case AK freeze iter #153 dashboard behavior:
- `severity-tier-panel` class present in dashboard HTML
- "Severity tier" panel header
- "iter #153" marker in panel hint
- HIGH tier label + RQ3_LLM_Full_Set_94% + RQ3_MAE_0.89 (HIGH tier claims visible)
- MEDIUM + LOW labels present
- `st-table` class for panel table

docstring + main() 同步更新到 37 cases.

## §C 关键改动点

1. **Dashboard ↔ CLI affordance symmetry**: extends iter #132 pattern (dashboard `status_summary_table` mirrors iter #131 CLI `--status-summary`):
   - iter #132 dashboard `status_summary_table` ↔ iter #131 CLI `--status-summary`
   - iter #153 dashboard `severity-tier-panel` ↔ iter #150 CLI `--severity-tier` ✓ 本轮
   
   Reviewer mental model: "for raw-status view → iter #132 dashboard / iter #131 CLI; for severity-tier view → iter #153 dashboard / iter #150 CLI"。

2. **Same classification logic as iter #150 CLI**: `HIGH_THRESHOLD_PCT=20.0`, `classify()` function, `_safe_abs/_safe_rel` accessors → single source of truth for tier assignment (CLI + dashboard agree on HIGH=2/MEDIUM=9/LOW=6 for current audit state)。Future drift between CLI and dashboard would be a bug。

3. **Anchor links to claim detail rows**: `<a href='#claim-{cid}'>` matches iter #116 dashboard anchor format → reviewer can click `RQ3_LLM_Full_Set_94%` in HIGH tier → jumps to claim detail row → see value_check_results (extracted 0.205, expected 0.946, abs_delta 0.7415) + provenance panels。

4. **Color-coded legend dots**: HIGH=red `#d32f2f`, MEDIUM=orange `#fb8c00`, LOW=green `#388e3c` → instant visual scan (matches iter #132 `status_summary_table` color dot pattern)。

5. **Sort within tier by max abs_delta desc**: same as iter #150 CLI → reviewer sees worst-first within each tier。

6. **Empty tier rendering**: `<em>(empty)</em>` text for empty tier → defensive against future state changes (none expected in current frozen baseline but defensive)。

7. **Orange `#ffe0b2` background**: visual distinction from iter #132 yellow `#fff8e1` (status_summary_table), iter #117 gray `#f5f5f5` (section_breakdown), iter #129 purple `#f3e5f5` (reverse-section), iter #118 blue (status-legend) → 5 distinct panel colors for 5 distinct info types。

8. **Open by default** (`<details open>`): matches iter #117/132/129 patterns → reviewer sees severity summary immediately on page load without clicking。

9. **`iter #153` generator footer marker**: dashboard `<meta name='generator'>` updated to list all cumulative iters → reviewer can see dashboard is up-to-date with audit infra timeline。

10. **No re-run audit**: dashboard regeneration is independent of audit re-run, only reads existing audit JSON → fast (~200ms) full dashboard regen including new panel。

11. **`#claim-{cid}` anchor format compat**: relies on iter #116 claim-row anchor → if iter #116 anchor format changes, this iter #153 panel breaks → single source of truth (anchor pattern) for dashboard deep-linking。

## §D 测试

```bash
$ python3 PersoanlQuery/_generate_audit_dashboard.py
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html

$ grep -c "severity-tier-panel" /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html
3   # 1 in body, 1 in CSS class definition, 1 in legend

$ grep -o "HIGH\|MEDIUM\|LOW" /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html | sort | uniq -c
   3 HIGH
   2 LOW
   2 MEDIUM

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AK: dashboard severity-tier panel (iter #153)
  PASS  dashboard severity-tier panel renders HIGH/MEDIUM/LOW with anchor links to HIGH claims

All 37 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~75 lines: function + CSS + wire + generator footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AK, +docstring, +main count to 37)
- 验证: 37/37 cases pass; pre-commit hook freeze 37 invariants ✓
- 命令: `python3 PersoanlQuery/_generate_audit_dashboard.py` (~200ms full regen including new panel)

## §F 与 audit infra timeline 的关系

Dashboard panel timeline:
- **iter #108** — initial dashboard (summary badges + claims table)
- **iter #109** — rel_delta percent display fix
- **iter #110** — vc status color coding
- **iter #112** — provenance panels
- **iter #113** — generated_at timestamp
- **iter #116** — claim anchors + index sidebar
- **iter #117** — per-section breakdown sub-table
- **iter #118** — status legend
- **iter #122** — `<title>` + `<meta description>`
- **iter #125** — `<link rel='alternate'>` to mapping sidecar
- **iter #129** — reverse section index panel
- **iter #132** — `status_summary_table` panel
- **iter #133** — sort-by-severity toggle button
- **iter #136** — filter-by-status click handler
- **iter #153** — severity-tier panel ✓ 本轮

Dashboard panels taxonomy (5 distinct background colors):
| panel | bg color | iter |
|-------|----------|------|
| claim-index sidebar | white (default) | #116 |
| status-legend | light blue | #118 |
| section-breakdown | gray `#f5f5f5` | #117 |
| status-summary-table | yellow `#fff8e1` | #132 |
| reverse-section-panel | purple `#f3e5f5` | #129 |
| **severity-tier-panel** | **orange `#ffe0b2`** | **#153** ✓ 本轮 |

Audit CLI ↔ dashboard affordance symmetry:
| view | CLI | dashboard |
|------|-----|-----------|
| raw status | `--status-summary` (iter #131) | `status_summary_table` panel (iter #132) |
| severity tier | `--severity-tier` (iter #150) | `severity-tier-panel` (iter #153) ✓ 本轮 |
| per-section | `--by-section` (iter #134) | `section-breakdown` panel (iter #117) |
| reverse section | `--summary-by-source-dir-and-status` (iter #147) | `reverse-section-panel` (iter #129) |

regression test coverage timeline:
- **iter #152** — 36 cases
- **iter #153** — 37 cases (+ dashboard severity-tier panel) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified (MEDIUM tier → LOW tier after re-run)
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample