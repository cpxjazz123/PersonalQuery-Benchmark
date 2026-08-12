# Iteration #137 — Audit CLI `--md-table` flag (markdown table for PR/Slack)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--md-table` argparse flag (markdown table sorted by severity, for PR comments / Slack / issue trackers); extend `_smoke_audit_regression.py` to 21 cases (Case U)
**prior**: reviewer's only shareable audit format was raw JSON (5 KB, unreadable in PR comments) or copy-paste from dashboard HTML (manual formatting, error-prone). GitHub PR comments, Slack threads, and issue trackers all expect markdown tables. Iter #131/133/134/135 quick-check CLI flags covered status / worst-N / per-section / temporal axes but not shareable-format axis.

## §A 审稿意见

Reviewer 想 share audit state with collaborators (e.g. tag author about discrepant claim, file GitHub issue listing infra gaps, post Slack summary) 当前必须:

1. Open dashboard HTML
2. Manually copy each claim row to markdown
3. Manually sort by severity
4. Format each cell correctly (`|` escaping, `code` backticks)
5. Hope no typos in 17-row table

后果: review-to-author turnaround 浪费 ~10 min/audit cycle on formatting。`paper_claims_audit.json` raw paste 到 GitHub PR 会触发 "JSON too long to render" warning,必须 format first。

iter #137 fix: 新增 `--md-table` CLI flag — 不 re-run audit,直接 read most recent audit JSON,sort by severity (discrepant first → degenerate → partial → unverified → blocked → verified → verified_value_match) tiebreak by abs_delta desc,print markdown table:

```
# paper_claims_audit (17/17 claims, generated 2026-07-21T22:50:00+00:00)

| ID | Status | Section | abs_delta | rel_delta_pct | Reason |
|----|--------|---------|-----------|---------------|--------|
| `RQ3_LLM_Full_Set_94%` | discrepant | §3.3 + Table 2 | 0.7415 | 78.38% | file exists but value mismatch: ... |
| `RQ3_MAE_0.89` | discrepant | §3.3 + Table 2 | 0.2700 | 30.34% | file exists but value mismatch: ... |
...
```

Reviewer 一行 pipe 到 clipboard (`--md-table | pbcopy`) 或 直接 paste 到 GitHub PR comment。

## §B 本轮 (iter #137) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--md-table` flag:
```python
parser.add_argument("--md-table", action="store_true",
                    help="(iter #137) Print claims as markdown table (sorted by status severity, then abs_delta desc) "
                         "for pasting into GitHub PR comments / Slack / issue trackers. Reads audit JSON, no re-run.")
```

docstring 加 flag description:
```
--md-table       Print claims as markdown table (sorted by status severity, then abs_delta desc)
                 for pasting into GitHub PR comments / Slack / issue trackers. Iter #137.
```

`main()` handler (在 `--audit-age` handler 之后):
```python
if args.md_table:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    severity_order = {s: i for i, s in enumerate(
        ["discrepant", "degenerate", "partial", "unverified", "blocked", "verified", "verified_value_match"])}
    def _safe_abs(vc: dict) -> float:
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else 0.0
    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None
    def sort_key(c: dict) -> tuple:
        vcs = c.get("value_check_results", [])
        max_abs = max((_safe_abs(vc) for vc in vcs), default=0.0)
        return (severity_order.get(c.get("status"), 99), -max_abs, c.get("id", ""))
    sorted_claims = sorted(claims_data, key=sort_key)
    print(f"# paper_claims_audit ({recent.get('n_audited', len(claims_data))}/{recent.get('n_claims', len(claims_data))} claims, generated {recent.get('generated_at', 'unknown')})")
    print()
    print("| ID | Status | Section | abs_delta | rel_delta_pct | Reason |")
    print("|----|--------|---------|-----------|---------------|--------|")
    for c in sorted_claims:
        vcs = c.get("value_check_results", [])
        max_abs = max((_safe_abs(vc) for vc in vcs), default=0.0)
        max_rel = None
        for vc in vcs:
            rd = _safe_rel(vc)
            if rd is None: continue
            if max_rel is None or abs(rd) > abs(max_rel):
                max_rel = rd
        sec = (c.get("section") or "").replace("|", "\\|")
        reason = (c.get("reason") or "")[:60].replace("|", "\\|").replace("\n", " ")
        abs_str = f"{max_abs:.4f}" if max_abs else "n/a"
        rel_str = f"{max_rel * 100:.2f}%" if max_rel is not None else "n/a"
        cid = c.get("id", "")
        status = c.get("status", "")
        print(f"| `{cid}` | {status} | {sec} | {abs_str} | {rel_str} | {reason} |")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case U freeze iter #137 behavior:
- `--md-table` exit 0
- Title line: `# paper_claims_audit (<n_audited>/<n_total> claims, generated <ISO8601>)` with `17/17 claims`
- Markdown header: `| ID | Status | Section | abs_delta | rel_delta_pct | Reason |`
- Separator: `|----|...`
- Exactly 17 data rows (one per claim)
- Top-1 row contains `RQ3_LLM_Full_Set_94%` + `0.7415`
- Top-4 rows contain all 4 discrepant claim IDs

docstring + main() 同步更新到 21 cases。

### README.md

Re-run commands section +1 line:
```bash
# Print claims as markdown table (sorted by severity) for pasting into GitHub PR comments / Slack; iter #137
python3 PersoanlQuery/paper_claims_audit.py --md-table | pbcopy
```

## §C 关键改动点

1. **No re-run audit**: `--md-table` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **Severity sort key**: `[discrepant, degenerate, partial, unverified, blocked, verified, verified_value_match]` — discrepant first (concrete value mismatch, actionable), verified last (no reviewer action needed)。

3. **Tiebreak abs_delta desc**: 同 severity 内 worst first (e.g. discrepant claims sorted by abs_delta desc — RQ3_LLM_Full_Set_94% (0.7415) before RQ3_MAE_0.89 (0.2700))。

4. **`_safe_abs` / `_safe_rel` defensive access**: `vc.get("abs_delta")` may return None if key exists but value is null (corrupted JSON / schema drift). Defensive `isinstance(v, (int, float))` check + 0.0 default avoids `TypeError: bad operand type for unary -: 'NoneType'` in sort。

5. **`|` → `\|` escape**: markdown table separator `|` collision — if a Section value contains `|`, must escape to `\|` 否则 markdown parser break column count. Same for Reason truncation。

6. **Reason truncated to 60 chars**: full Reason strings can be 200+ chars (e.g. "file exists but value mismatch: expected=0.9460 actual=0.204"). 60-char truncation keeps table readable while preserving key info (expected vs actual values)。

7. **Reason newline strip**: `.replace("\n", " ")` 防止 multi-line reason break markdown row into multiple visual rows (looks bad in GitHub PR render)。

8. **`# title` line + `print()` separation**: title 用 `# heading` syntax → GitHub / Slack 会 render 成 H1,清楚 mark "this is the audit snapshot"。Blank line `print()` 后 table 保证 markdown parser 正确识别。

9. **ID column backticked**: `\`RQ3_LLM_Full_Set_94%\`` 用 backtick 包起来 → render 成 monospace inline code,跟 paper 里的 backtick convention (iter #123) 一致。

10. **`--output` flag respected**: reviewer 想 `--md-table --output /custom/path.json` 跟其他 mode 语义一致。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --md-table | head -10
# paper_claims_audit (17/17 claims, generated 2026-07-20T23:00:45+00:00)

| ID | Status | Section | abs_delta | rel_delta_pct | Reason |
|----|--------|---------|-----------|---------------|--------|
| `RQ3_LLM_Full_Set_94%` | discrepant | §3.3 + Table 2 | 0.7415 | 78.38% | file exists but value mismatch: expected=0.9460 actual=0.204 |
| `RQ3_MAE_0.89` | discrepant | §3.3 + Table 2 | 0.2700 | 30.34% | file exists but value mismatch: expected=0.8900 actual=0.620 |
| `RQ3_Fleiss_Kappa_0.72` | discrepant | §3.3 + Table 2 | 0.0965 | 13.40% | file exists but value mismatch: expected=0.7200 actual=0.623 |
| `RQ3_Spearman_0.81` | discrepant | §3.3 + Table 2 | 0.0936 | 11.56% | file exists but value mismatch: expected=0.8100 actual=0.716 |
| `RQ1_Table1_Hit10` | degenerate | §3.2 + Table 1 | n/a | n/a | file exists but content degenerate: selector 'rows[*].per_me |

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
Case U: --md-table CLI flag (iter #137)                                                          PASS

All 21 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~50 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case U, +docstring, +main count to 21)
- 文档: `README.md` (+1 line `--md-table | pbcopy` example)
- 验证: 21/21 cases pass; pre-commit hook freeze 21 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --md-table` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary (global 1-line)
- **iter #133** — --top N (worst-claim triage)
- **iter #134** — --by-section (per-section concentration)
- **iter #135** — --audit-age (temporal staleness)
- **iter #137** — --md-table (shareable markdown format) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format | `--md-table` (iter #137) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #136** — 20 cases
- **iter #137** — 21 cases (+ --md-table) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample