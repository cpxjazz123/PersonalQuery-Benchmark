# Iteration #136 — Dashboard "filter by status" (click status_summary_table row)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` make status_summary_table rows clickable filters; add `clear-filter` button + JS handler; add `data-status` attribute to claim tbodies; extend `_smoke_audit_regression.py` to 20 cases (Case T)
**prior**: status_summary_table (iter #132) showed 7 status × count × percent × bar with static rendering — reviewer could see "4 discrepant" but to focus on discrepant claims had to scroll through 17 mixed-status rows. Sort toggle (iter #133) reorders by abs_delta but doesn't filter by status. Reviewer wanting "show only discrepant" must eyeball-filter or open JSON.

## §A 审稿意见

Dashboard status_summary_table iter #132 已经 give 7-row breakdown, 但 reviewer 想 focus on "discrepant only" 必须 scroll past 13 verified/unverified claims。常见 reviewer workflow:

1. Open dashboard HTML
2. See status_summary_table: "4 discrepant (23.5%)"
3. 想 jump to those 4 → must scroll through 17 claims mixed
4. Mental mark which rows have orange `discrepant` badge
5. 实际 only 4/17 visible attention

后果: reviewer triage discrepant claims 浪费 visual scanning effort。Sort by abs_delta (iter #133) helps but not filter — still must scan 17 rows for orange badges。

iter #136 fix: 让 status_summary_table 每 row clickable,click → filter claims table 到只 show that status。Add `clear-filter` button visible when filter active. Default = no filter (show all 17)。Vanilla JS,no external library。

## §B 本轮 (iter #136) 改动

### PersoanlQuery/_generate_audit_dashboard.py

`_render_status_summary_table` 加 `data-status` attribute + `sst-row` class on each `<tr>` (clickable); add `sst-hint` + `clear-filter` button:
```python
def _render_status_summary_table(summary: dict, n_total: int) -> str:
    if n_total <= 0:
        return ""
    rows = []
    for status in ["verified_value_match", "verified", "discrepant", "degenerate", "partial", "unverified", "blocked"]:
        count = summary.get(status, 0)
        label, color = STATUS_COLORS.get(status, (status, "#9e9e9e"))
        pct = (count / n_total) * 100 if n_total > 0 else 0.0
        bar_width = max(0, min(100, pct))
        rows.append(
            f"<tr class='sst-row' data-status='{status}' title='Click to filter claims to this status'>"
            f"<td><span class='legend-dot' style='background:{color}'></span><strong>{label}</strong></td>"
            f"<td class='sst-count'>{count}</td>"
            f"<td class='sst-pct'>{pct:.1f}%</td>"
            f"<td class='sst-bar-cell'><div class='sst-bar' style='width:{bar_width:.1f}%; background:{color}'></div></td></tr>"
        )
    return f"""<details class='status-summary-table' open>
<summary><strong>Status summary</strong> ({n_total} total claims)
<span class='sst-hint'>(click a row to filter claims; iter #136)</span>
<button id='clear-filter' class='clear-filter-btn' type='button' style='display:none'>Clear filter</button>
</summary>
<table class='sst-table'>
<thead><tr><th>Status</th><th>Count</th><th>Percent</th><th>Distribution</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

`_render_claim_row` 加 `data-status` on tbody:
```python
return f"""
<tbody class='claim-tbody' data-claim-id='{c['id']}' data-abs-delta='{max_abs:.4f}' data-status='{status}'>
...
</tbody>"""
```

CSS additions:
```css
.sst-row { cursor: pointer; transition: background-color 0.15s ease; }
.sst-row:hover { background: #fff3c4; }
.sst-row.active-filter { background: #ffe082; font-weight: 600; }
.sst-hint { font-size: 11px; color: #888; font-weight: normal; margin-left: 8px; }
.clear-filter-btn { background: #d32f2f; color: white; border: none; padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; margin-left: 12px; }
.clear-filter-btn:hover { background: #b71c1c; }
```

JS filter handler (separate `<script>` block after sort-toggle):
```javascript
(function() {
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr.sst-row'));
  var table = document.getElementById('claims-table');
  var clearBtn = document.getElementById('clear-filter');
  if (!rows.length || !table) return;

  function applyFilter(status) {
    var tbodies = Array.prototype.slice.call(table.querySelectorAll('tbody.claim-tbody'));
    rows.forEach(function(r) { r.classList.remove('active-filter'); });
    if (status === null) {
      tbodies.forEach(function(tb) { tb.style.display = ''; });
      if (clearBtn) clearBtn.style.display = 'none';
      return;
    }
    tbodies.forEach(function(tb) {
      tb.style.display = (tb.getAttribute('data-status') === status) ? '' : 'none';
    });
    var activeRow = rows.find(function(r) { return r.getAttribute('data-status') === status; });
    if (activeRow) activeRow.classList.add('active-filter');
    if (clearBtn) clearBtn.style.display = '';
  }

  rows.forEach(function(row) {
    row.addEventListener('click', function() {
      var status = row.getAttribute('data-status');
      applyFilter(status);
    });
  });
  if (clearBtn) {
    clearBtn.addEventListener('click', function() { applyFilter(null); });
  }
})();
```

Generator footer: `iter #108, #129, #132, #133, #136`。

### PersoanlQuery/_smoke_audit_regression.py

加 Case T freeze iter #136 behavior:
- Dashboard HTML has `sst-row` class + `data-status` attribute
- All 7 status values present (`verified_value_match` / `verified` / `discrepant` / `degenerate` / `partial` / `unverified` / `blocked`)
- ≥17 claim tbodies have `data-status` attribute (one per claim)
- `clear-filter` button + `sst-hint` label
- JS `applyFilter` function + CSS `active-filter` class + `.sst-row` / `.clear-filter-btn` rules

docstring + main() 同步更新到 20 cases。

## §C 关键改动点

1. **`data-status` on both rows AND tbodies**: Two-way linking — status_summary_table rows say "I'm discrepant", claim tbodies say "I'm discrepant". JS matches by attribute equality (`tb.getAttribute('data-status') === status`)。

2. **`tb.style.display = ''` (clear) vs `'none'` (hide)**: 用 `display` toggle 而不是 DOM remove — `display=''` reset to default (table-row-group for tbody),claims 立即 reappear without flicker。

3. **`active-filter` CSS class**: 视觉 indicate currently selected status row — yellow `#ffe082` background + bold,跟 hover state (`#fff3c4` lighter yellow) 区分。Clear-filter 时 remove class。

4. **Clear-filter button hidden by default**: `style='display:none'` in HTML — only show when filter active。如果 always visible,screen reader / keyboard nav 用户 see "Clear filter" button with no function (no filter to clear)。

5. **`title='Click to filter claims to this status'` on rows**: HTML native tooltip — mouse hover 显示 hint,no need for separate help text。

6. **Sort + filter compose**: 排序 (iter #133) 跟 filter (iter #136) 是 orthogonal operations — reviewer 可 "filter to discrepant + sort by abs_delta desc" 拿 worst discrepant claims first。两种 affordance 一起用 → 1 click filter + 1 click sort → 全 focused view。

7. **Vanilla JS**: 跟 iter #133 sort toggle 同样 no external library,`Array.prototype.slice.call` + `querySelectorAll` + `addEventListener` + DOM property manipulation。

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
Case P: --top N CLI flag (iter #133)                                                             PASS
Case Q: dashboard sort-by-severity toggle (iter #133)                                            PASS
Case R: --by-section CLI flag (iter #134)                                                        PASS
Case S: --audit-age CLI flag (iter #135)                                                         PASS
Case T: dashboard filter-by-status (iter #136)                                                   PASS

All 20 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~30 lines `_render_status_summary_table` clickable rows + `data-status` on tbodies + CSS + JS + clear-filter button + footer)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case T, +docstring, +main count to 20)
- 验证: 20/20 cases pass; pre-commit hook freeze 20 invariants ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh`

## §F 与 audit infra timeline 的关系

Dashboard interactive affordances timeline:
- **iter #116** — HTML anchors (deep-linking from paper footnotes)
- **iter #117** — per-section breakdown table (static)
- **iter #129** — reverse-section panel (static)
- **iter #132** — status_summary_table panel (static)
- **iter #133** — sort-by-severity toggle (JS interactive) ✓
- **iter #136** — filter-by-status (JS interactive) ✓ 本轮

regression test coverage timeline:
- **iter #135** — 19 cases
- **iter #136** — 20 cases (+ filter-by-status) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample