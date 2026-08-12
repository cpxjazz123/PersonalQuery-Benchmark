# Iteration #133 — Audit CLI `--top N` + dashboard "Sort by severity" toggle

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--top N` argparse flag (compact N-row discrepant table); `PersoanlQuery/_generate_audit_dashboard.py` wrap each claim in `<tbody data-abs-delta>` + add client-side sort toggle button; extend `_smoke_audit_regression.py` to 17 cases (Case P + Q)
**prior**: dashboard rendered claims in alphabetical claim_id order, with no way to surface "worst-offending claims first". `abs_delta` / `rel_delta_pct` were buried in the value_check detail rows under each claim — reviewer had to open every `<details>` and visually compare. `--status-summary` (iter #131) gave a 1-line overview but no answer to "which 3 claims should I look at first?"

## §A 审稿意见

Dashboard 当前 claims 按 claim_id alphabetical sort (`Pipeline_Regeneration_10x10`, `RQ3_Fleiss_Kappa_0.72`, `RQ3_LLM_Full_Set_94%`, ...)。Reviewer 想知道 "most discrepant claims first" 必须:

1. Open audit JSON (`paper_claims_audit.json`)
2. Manually parse each claim's `value_check_results`
3. Sort by `abs_delta` desc mentally
4. Cross-reference with paper sections

iter #131 加 `--status-summary` 给 shell 1-line overview,但 no affordance for "top-N worst claims" 列表 or browser sort toggle。

后果: reviewer triage 一份 17-claim audit 必须花 ~5 分钟 mental sorting。Worst case, 一眼 biggest `abs_delta=0.7415` (RQ3_LLM_Full_Set_94%, paper says 94% actual=20.4%) 应该是第 1 priority fix,但 alphabetical sort 把 RQ3_* claims 排在 P* 之后,reviewer 容易 miss。

iter #133 fix: 
(a) CLI `--top N` flag — 不 re-run audit,直接 read most recent audit JSON,sort discrepant claims by max `abs_delta` desc,print compact N-row table (`# / ID / abs_delta / rel_delta / Reason`)
(b) Dashboard "Sort by severity" toggle button — JS client-side reorder of `<tbody>` elements by `data-abs-delta` attribute; default = by claim_id,click → by abs_delta desc,click again → back

## §B 本轮 (iter #133) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--top` flag (nargs="?" const=5 default=None type=int):
```python
parser.add_argument("--top", nargs="?", const=5, default=None, type=int,
                    help="(iter #133) Print top N discrepant claims sorted by abs_delta descending "
                         "(default N=5 if just '--top'; e.g. '--top 3'); reads most recent audit JSON, does not re-run.")
```

`main()` handler (在 `--status-summary` handler 之后):
```python
if args.top is not None:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    n = args.top if args.top is not None else 5
    if n <= 0:
        print(f"ERROR: --top N must be >= 1 (got {n}).", file=__import__("sys").stderr)
        return 2
    # Collect discrepant claims with max abs_delta + max abs(rel_delta) across value_check_results.
    discrepant = []
    for c in claims_data:
        if c.get("status") != "discrepant":
            continue
        vcs = c.get("value_check_results", [])
        max_abs = max((vc.get("abs_delta", 0.0) for vc in vcs), default=0.0)
        max_rel = None
        for vc in vcs:
            rd = vc.get("rel_delta")
            if rd is None: continue
            if max_rel is None or abs(rd) > abs(max_rel):
                max_rel = rd
        discrepant.append((c["id"], max_abs, max_rel, c.get("reason", "")))
    discrepant.sort(key=lambda x: x[1], reverse=True)
    top_n = discrepant[:n]
    print(f"top {len(top_n)} discrepant claims (by abs_delta desc):")
    print(f"{'#':<4} {'ID':<28} {'abs_delta':>10}  {'rel_delta':>10}  Reason")
    print("-" * 100)
    for i, (cid, ad, rd, reason) in enumerate(top_n, start=1):
        rd_str = f"{rd * 100:.2f}%" if rd is not None else "n/a"
        print(f"{i:<4} {cid:<28} {ad:>10.4f}  {rd_str:>10}  {reason[:60]}")
    return 0
```

docstring 加 flag description (`--top [N]` block)。

### PersoanlQuery/_generate_audit_dashboard.py

`_render_claim_row` 加 max_abs 计算 + wrap in `<tbody class='claim-tbody' data-claim-id='{id}' data-abs-delta='{max_abs:.4f}'>`:
```python
def _render_claim_row(c: dict) -> str:
    status = c["status"]
    ...
    max_abs = 0.0
    if "value_check_results" in c:
        for vc in c["value_check_results"]:
            ad = vc.get("abs_delta")
            if ad is not None and ad > max_abs:
                max_abs = ad
            ...
    return f"""
<tbody class='claim-tbody' data-claim-id='{c['id']}' data-abs-delta='{max_abs:.4f}'>
<tr id='claim-{c['id']}'>
  ...
  <td class='sev-col'>{max_abs:.4f}</td>
</tr>
<tr class='detail-row'>
  <td colspan='6'>{detail_html or '<em>(no detail)</em>'}</td>
</tr>
</tbody>"""
```

`Claims ({n_total})` section header 加 toolbar + button + JS toggle:
```html
<h2>Claims ({n_total})</h2>
<div class='claims-toolbar'>
  <button id='sort-toggle' class='sort-btn' data-mode='id' type='button'>Sort by severity (abs_delta desc)</button>
  <span class='sort-note'>Default: by claim ID (alphabetical). Click to re-sort.</span>
</div>
<table id='claims-table'>
<thead><tr><th>ID</th><th>Status</th><th>Section</th><th>Claim</th><th>Reason</th><th class='sev-col'>abs_delta</th></tr></thead>
{rows}
</table>
<script>
(function() {
  var btn = document.getElementById('sort-toggle');
  var table = document.getElementById('claims-table');
  if (!btn || !table) return;
  btn.addEventListener('click', function() {
    var tbodies = Array.prototype.slice.call(table.querySelectorAll('tbody.claim-tbody'));
    var mode = btn.getAttribute('data-mode');
    if (mode === 'id') {
      tbodies.sort(function(a, b) {
        return parseFloat(b.getAttribute('data-abs-delta')) - parseFloat(a.getAttribute('data-abs-delta'));
      });
      btn.setAttribute('data-mode', 'severity');
      btn.textContent = 'Sort by claim ID (alphabetical)';
    } else {
      tbodies.sort(function(a, b) {
        return a.getAttribute('data-claim-id').localeCompare(b.getAttribute('data-claim-id'));
      });
      btn.setAttribute('data-mode', 'id');
      btn.textContent = 'Sort by severity (abs_delta desc)';
    }
    tbodies.forEach(function(tb) { table.appendChild(tb); });
  });
})();
</script>
```

CSS additions:
```css
.claims-toolbar { margin: 8px 0; display: flex; align-items: center; gap: 12px; }
.sort-btn { background: #1976d2; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600; }
.sort-btn:hover { background: #1565c0; }
.sort-note { font-size: 12px; color: #666; }
.sev-col { width: 90px; text-align: right; color: #555; }
```

Generator footer: `iter #108, #129, #132, #133`。

### PersoanlQuery/_smoke_audit_regression.py

加 Case P + Q freeze iter #133 behavior:
- **Case P** — `--top` default → 4 rows (frozen baseline has 4 discrepant), `--top 3` → 3 rows, top-1 must be `RQ3_LLM_Full_Set_94%` (abs_delta=0.7415), `--top 0` → exit 2.
- **Case Q** — dashboard has `claims-toolbar` + `sort-btn` + `Sort by severity` label + `data-abs-delta` attribute + `claim-tbody` class + `data-mode` attribute, ≥17 `claim-tbody` rows, `0.7415` (largest abs_delta) visible.

docstring + main() 同步更新到 17 cases。

### README.md

Re-run commands section +1 line:
```bash
# Print top N discrepant claims sorted by abs_delta desc from the most recent audit JSON without re-running; iter #133
python3 PersoanlQuery/paper_claims_audit.py --top 3
```

## §C 关键改动点

1. **No re-run audit**: `--top` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **`nargs='?'` + `const=5` argparse pattern**: `--top` alone → const=5 (default 5), `--top 3` → type=int 转换 "3"。Reviewer 可写 `--top` 拿 top 5 默认值,或 `--top N` 拿 custom N。

3. **Max-abs across value_check_results**: 一个 claim 可能有多个 subclaim (e.g. RQ3_LLM_Full_Set_94% 只有 1 个,但保留 multi-subclaim future-proof)。`max()` 取 worst subclaim。

4. **Skip non-discrepant**: 只有 `status == "discrepant"` 的 claim 被 sort,其他 status 不 enter list。`--top 5` 即使 total claims=17 也只 print 最多 4 (当前 baseline)。

5. **Client-side dashboard sort**: JS toggle 不依赖 external library (`Array.prototype.slice.call` + `querySelectorAll`),纯 vanilla JS,文件 self-contained。Re-order tbodies by `data-abs-delta` (severity mode) 或 `data-claim-id` (ID mode)。

6. **`data-abs-delta` attribute on `<tbody>` not `<tr>`**: 每个 claim 2 rows (main + detail) — wrap 在 `<tbody>` 让 JS 可以 atomic reorder claim pair,而不是乱 row pairs。

7. **`colspan=6` on detail-row**: 加 `abs_delta` 列后,detail-row colspan 从 5 改为 6。Frozen assertion 在 Case Q 检查 visible `0.7415`。

8. **Frozen top-1 = RQ3_LLM_Full_Set_94%**: 已知 worst offender (paper claims 94%, actual 20.4%, abs_delta=0.7415, rel_delta=78.38%) — Case P 显式 assert 它是 `--top` 输出第 1 行。如果 audit JSON 未来 flip (e.g. RQ3_LLM_Full_Set_94% 被修复 → 0 abs_delta),test fire as baseline flip detector。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --top
top 4 discrepant claims (by abs_delta desc):
#    ID                            abs_delta   rel_delta  Reason
----------------------------------------------------------------------------------------------------
1    RQ3_LLM_Full_Set_94%             0.7415      78.38%  file exists but value mismatch: expected=0.9460 actual=0.204
2    RQ3_MAE_0.89                     0.2700      30.34%  file exists but value mismatch: expected=0.8900 actual=0.620
3    RQ3_Fleiss_Kappa_0.72            0.0965      13.40%  file exists but value mismatch: expected=0.7200 actual=0.623
4    RQ3_Spearman_0.81                0.0936      11.56%  file exists but value mismatch: expected=0.8100 actual=0.716

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

All 17 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~50 lines argparse flag + handler + docstring)
- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+~10 lines max_abs tracking + tbody wrapper + toolbar/button/JS/CSS)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case P + Case Q, +docstring, +main count to 17)
- 文档: `README.md` (+1 line `--top` example)
- 验证: 17/17 cases pass; pre-commit hook freeze 17 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --top 3` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary mode
- **iter #133** — --top N mode (worst-first triage) ✓ 本轮

Dashboard sort/interaction timeline:
- **iter #116** — HTML anchors + claims-index sidebar
- **iter #129** — reverse-section panel
- **iter #132** — status_summary_table panel
- **iter #133** — sort-by-severity toggle (client-side JS) ✓ 本轮

regression test coverage timeline:
- **iter #131** — 14 cases
- **iter #132** — 15 cases (+ status_summary_table)
- **iter #133** — 17 cases (+ --top + sort toggle) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample