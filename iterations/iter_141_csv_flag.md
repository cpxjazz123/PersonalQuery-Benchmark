# Iteration #141 — Audit CLI `--csv` flag (spreadsheet import format)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--csv` argparse flag (prints claims as RFC-4180 CSV for spreadsheet import); extend `_smoke_audit_regression.py` to 25 cases (Case Y)
**prior**: iter #131 (`--status-summary`) gave compact 1-line; iter #137 (`--md-table`) gave markdown table for GitHub/Slack; iter #138/139/140 (`--list-degenerate/unverified/partial`) gave human-readable per-status diagnosis. None of these were machine-friendly for spreadsheet import (Excel / Numbers / Google Sheets) — `,` and `|` in reason fields broke naive split. Reviewer wanting to triage 17 audit claims in a spreadsheet had to manually parse JSON.

## §A 审稿意见

Reviewer 想要 "17 audit claims 全 load 到 spreadsheet,filter/sort/pivot by status + abs_delta" 当前 must:
1. `jq '.claims[] | [.id, .status, .section, (.value_check_results[0].abs_delta // 0), ...]'` → manually craft columns
2. Or pipe `--md-table` through markdown-to-csv converter (lossy: pipe escaping, multi-line reasons)
3. Or open dashboard HTML and click "copy table" → paste (also lossy)

后果: audit data 与 spreadsheet 工具割裂。Reviewer 想 在 Excel pivot `status × section` 必须手动写 parser。

iter #141 fix: 新增 `--csv` CLI flag — emit RFC-4180 CSV (Python `csv.writer` handles quoting of commas/quotes/newlines automatically) with 8 columns: `id,status,section,abs_delta,rel_delta_pct,expected_outputs_count,code_evidence_count,reason`。Sorted by severity (discrepant first by abs_delta desc) — matches iter #137 `--md-table` ordering for visual consistency。

## §B 本轮 (iter #141) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--csv` flag:
```python
parser.add_argument("--csv", action="store_true",
                    help="(iter #141) Print claims as RFC-4180 CSV (8 columns: id,status,section,"
                         "abs_delta,rel_delta_pct,expected_outputs_count,code_evidence_count,reason)"
                         " sorted by severity (discrepant first by abs_delta desc). Useful for "
                         "spreadsheet import (Excel / Sheets) or downstream analytics. Does not "
                         "re-run audit; reads most recent audit JSON. Exit 0 always when JSON found.")
```

docstring 加 flag description:
```
--csv           Print claims as RFC-4180 CSV (8 columns) sorted by severity, for spreadsheet
                 import. Does not re-run audit; reads most recent JSON. Iter #141.
```

`main()` handler (在 `--list-partial` handler 之后):
```python
if args.csv:
    import csv as _csv_mod
    import io as _io_mod
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
    buf = _io_mod.StringIO()
    writer = _csv_mod.writer(buf)
    writer.writerow(["id", "status", "section", "abs_delta", "rel_delta_pct",
                     "expected_outputs_count", "code_evidence_count", "reason"])
    for c in sorted_claims:
        vcs = c.get("value_check_results", [])
        max_abs = max((_safe_abs(vc) for vc in vcs), default=0.0)
        max_rel = None
        for vc in vcs:
            rd = _safe_rel(vc)
            if rd is None: continue
            if max_rel is None or abs(rd) > abs(max_rel):
                max_rel = rd
        n_expected = len(c.get("expected_outputs") or [])
        n_evidence = len(c.get("code_evidence") or [])
        reason = (c.get("reason") or "").replace("\n", " ").replace("\r", " ")
        abs_str = f"{max_abs:.4f}" if max_abs else ""
        rel_str = f"{max_rel * 100:.4f}" if max_rel is not None else ""
        writer.writerow([c.get("id", ""), c.get("status", ""), c.get("section", ""),
                         abs_str, rel_str, n_expected, n_evidence, reason])
    print(buf.getvalue(), end="")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case Y freeze iter #141 behavior:
- `--csv` exit 0 (always when JSON found)
- CSV header row exactly: `id,status,section,abs_delta,rel_delta_pct,expected_outputs_count,code_evidence_count,reason`
- 17 data rows
- Top-1 row is `RQ3_LLM_Full_Set_94%,discrepant,§3.3 + Table 2,0.7415,` (worst discrepant by abs_delta)
- CSV escaping: at least one quoted reason field (partial claim reason contains comma)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 25 cases.

### README.md

Re-run commands section +1 line:
```bash
# Print claims as RFC-4180 CSV (8 cols) sorted by severity for spreadsheet import; iter #141
python3 PersoanlQuery/paper_claims_audit.py --csv > audit.csv
```

## §C 关键改动点

1. **`csv.writer` 自动 handle quoting**: Python `csv` module 默认 `QUOTE_MINIMAL` — 任何含 `,`/`"`/`\n` 的 field 自动加双引号包裹。Reason 字段 like `"code referenced, no specific output files enumerated for verification"` 自动 quoted。Reviewer 不必担心 CSV 解析出错。

2. **`StringIO` buffer + `print(..., end="")`**: 用 `StringIO` 而非直接 `writer.writerow(...)` 到 stdout 是为了 (a) 单次 `print` 一次性 flush 避免 line-buffering issue, (b) `end=""` 避免额外 newline。Output 干净到可直接 pipe 到 `xsv` / `csvkit` / `q` / spreadsheet import。

3. **`_safe_abs` / `_safe_rel` defensive accessors**: 与 iter #137 同样 pattern — `abs_delta: null` in JSON 不能参与 `max(...)` 计算。`isinstance(v, (int, float))` 检查 + 0.0/None default → no crash on schema drift。

4. **Severity sort order (mirror iter #137)**: `["discrepant", "degenerate", "partial", "unverified", "blocked", "verified", "verified_value_match"]` — worst 优先。Within same status, `abs_delta` desc (largest discrepancy 优先)。然后 `id` asc → tie-break stable。

5. **`rel_delta_pct` × 100 conversion**: JSON 存 fraction (0.1156) → CSV 列以 percent 显示 (11.56)。Matches iter #109 dashboard percent display convention。`max_rel * 100:.4f` → 4 decimal places 足够精度。

6. **`abs_str = f"{max_abs:.4f}"` only when nonzero**: Empty string for `0.0` → spreadsheet imports as blank cell, 比 `"0.0000"` 更 readable。`rel_str = ""` when `max_rel is None` → 区分 "rel_delta = 0" (impossible for abs_delta > 0) vs "no rel_delta recorded"。

7. **`reason.replace("\n", " ").replace("\r", " ")`**: Reason 字段可能含 multi-line 文本 → CSV 中 replace 成 single space。Quoting 仍 由 csv module 处理 but removing newlines makes spreadsheet cells 单行。

8. **Reuses iter #112 / iter #117 fields**: `expected_outputs_count` from iter #112 provenance; `code_evidence_count` from iter #112; severity order from iter #117 per-section breakdown; abs/rel from iter #110 value_check_results。No new audit schema — just shell wrapper。

9. **Exit code `0` always when JSON found**: 与 `--status-summary` (exit 0/1 based on staleness) 不同 — CSV 总是 17 rows (audit always has claims)。Missing JSON → exit 2 (consistent with iter #138/139/140)。

10. **`--output` flag respected**: reviewer 想 `--csv --output /custom/path.json` 跟其他 mode 语义一致。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --csv
id,status,section,abs_delta,rel_delta_pct,expected_outputs_count,code_evidence_count,reason
RQ3_LLM_Full_Set_94%,discrepant,§3.3 + Table 2,0.7415,78.3779,1,2,file exists but value mismatch: expected=0.9460 actual=0.2045 delta=0.7415 (rel 78.4%)
RQ3_MAE_0.89,discrepant,§3.3 + Table 2,0.2700,30.3371,1,4,file exists but value mismatch: expected=0.8900 actual=0.6200 delta=0.2700 (rel 30.3%)
RQ3_Fleiss_Kappa_0.72,discrepant,§3.3 + Table 2,0.0965,13.4028,1,1,file exists but value mismatch: expected=0.7200 actual=0.6235 delta=0.0965 (rel 13.4%)
RQ3_Spearman_0.81,discrepant,§3.3 + Table 2,0.0750,9.2593,1,4,file exists but value mismatch: expected=0.8100 actual=0.7350 delta=0.0750 (rel 9.3%)
RQ1_Table1_Hit10,degenerate,§3.1 + Table 1,,,0,0,file exists but selector returned no non-NaN values (matched_files=['result/personal_query/...'])
...
Pipeline_Skip_ColBERTv2_SPLADE,partial,(infrastructure),,,,0,"code referenced, no specific output files enumerated for verification"
...
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case Y: --csv CLI flag (iter #141)
  PASS  --csv emits CSV with 8-col header + 17 data rows + comma escaping + exits 2 on missing JSON

All 25 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~50 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case Y, +docstring, +main count to 25)
- 文档: `README.md` (+1 line `--csv` example)
- 验证: 25/25 cases pass; pre-commit hook freeze 25 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --csv > audit.csv` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
- **iter #135** — --audit-age
- **iter #137** — --md-table (markdown for GitHub)
- **iter #138** — --list-degenerate
- **iter #139** — --list-unverified
- **iter #140** — --list-partial
- **iter #141** — --csv (RFC-4180 CSV for spreadsheets) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) ✓ 本轮 |
| degenerate diagnosis | `--list-degenerate` (iter #138) |
| unverified planning | `--list-unverified` (iter #139) |
| partial scope diagnosis | `--list-partial` (iter #140) |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #140** — 24 cases
- **iter #141** — 25 cases (+ --csv) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
