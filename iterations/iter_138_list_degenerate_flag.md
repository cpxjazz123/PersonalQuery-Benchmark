# Iteration #138 — Audit CLI `--list-degenerate` flag (diagnosis for value-extraction failures)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--list-degenerate` argparse flag (prints degenerate claims with id + section + matched_files); extend `_smoke_audit_regression.py` to 22 cases (Case V)
**prior**: degenerate status means "file exists but selector returned no non-NaN values" (e.g. wrong JSON path glob, stale file schema, broken aggregator). Reviewer wanting to diagnose WHY a claim is degenerate had to run `--claim-id <id> --verbose` and visually scan the full value_check_results + matched_files output. Iter #115 dashboard added `matched_files` list per degenerate claim, but no equivalent CLI affordance for shell-side diagnosis.

## §A 审稿意见

Reviewer 看到 "1 degenerate" 在 `--status-summary` 输出,想 diagnose "why is RQ1_Table1_Hit10 degenerate? Which files did the selector try? Which paths matched?" 当前 must:

1. `python3 paper_claims_audit.py --claim-id RQ1_Table1_Hit10 --verbose`
2. Visually parse the verbose output (15+ lines)
3. Find `matched_files` list within value_check_results
4. Mental note which paths the selector tried

后果: reviewer diagnosis cycle ~3-5 min/claim。Common case: iter #88 Stage 6/9 re-run fix expected to flip 1 degenerate → verified,reviewer wants to confirm "RQ1_Table1_Hit10 is now verified, no other degenerate claims appeared" — currently must grep audit JSON `status == "degenerate"` + parse matched_files manually。

iter #138 fix: 新增 `--list-degenerate` CLI flag — 不 re-run audit,直接 read most recent audit JSON,filter `status == "degenerate"`,print compact per-claim block:

```
Degenerate claims (1 total):
  RQ1_Table1_Hit10  [§3.2 + Table 1]
    reason      : file exists but content degenerate: selector 'rows[*].per_metric.*.mean' returne
    matched_files: result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json, result/personal_query/06_retrieval/Grocery_and_Gourmet_Food/retrieval_syntax_depth_summary.json, result/personal_query/09_noisy_retrieval/Grocery_and_Gourmet_Food/syntax_depth_correct_vs_noisy_results_pivot.json
```

Reviewer 一眼看出: 3 matched files, 但 selector `'rows[*].per_metric.*.mean'` 返回 0 non-NaN — diagnose path glob mismatch。

## §B 本轮 (iter #138) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--list-degenerate` flag:
```python
parser.add_argument("--list-degenerate", action="store_true",
                    help="(iter #138) Print just the 'degenerate' claims with id, section, and matched_files "
                         "(one per line, comma-separated). Exit 0 if any degenerate found, exit 1 if none, "
                         "exit 2 if JSON missing. Useful for diagnosing value-extraction failures.")
```

docstring 加 flag description:
```
--list-degenerate Print just the 'degenerate' claims with id, section, and matched_files
                   (one per line, comma-separated). Exit 0 if any degenerate, exit 1 if none.
                   Useful for diagnosing value-extraction failures. Iter #138.
```

`main()` handler (在 `--md-table` handler 之后):
```python
if args.list_degenerate:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    degen = [c for c in claims_data if c.get("status") == "degenerate"]
    if not degen:
        print("No degenerate claims. Audit JSON has no value-extraction failures.")
        return 1
    print(f"Degenerate claims ({len(degen)} total):")
    for c in degen:
        cid = c.get("id", "?")
        sec = c.get("section", "?")
        reason = (c.get("reason") or "")[:80]
        vcs = c.get("value_check_results", [])
        matched = []
        for vc in vcs:
            for mf in vc.get("matched_files") or []:
                matched.append(mf)
        matched_str = ", ".join(matched) if matched else "(no matched files recorded)"
        print(f"  {cid}  [{sec}]")
        print(f"    reason      : {reason}")
        print(f"    matched_files: {matched_str}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case V freeze iter #138 behavior:
- `--list-degenerate` exit 0 (1 degenerate in frozen baseline)
- "Degenerate claims" header
- `RQ1_Table1_Hit10` row present (only frozen degenerate)
- `§3.2 + Table 1` section visible
- `matched_files` field visible
- ≥1 `result/personal_query/` path in matched_files
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 22 cases。

### README.md

Re-run commands section +1 line:
```bash
# List just the 'degenerate' claims with id, section, and matched_files for diagnosis; iter #138
python3 PersoanlQuery/paper_claims_audit.py --list-degenerate
```

## §C 关键改动点

1. **No re-run audit**: `--list-degenerate` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **Triple exit code**:
   - `0` (degenerate found) — actionable, claim exists
   - `1` (no degenerate) — all value-extractions succeeded (post-iter #88 ideal state)
   - `2` (JSON missing) — distinct from "no degenerate", shell scripts can tell apart

3. **`(no matched files recorded)` fallback**: 如果 degenerate claim 没 populate matched_files 字段 (e.g. older audit JSON before iter #115),print clear placeholder string instead of empty output — reviewer knows data is missing not claim has 0 files。

4. **`reason` truncated to 80 chars**: full reason strings can be 200+ chars (e.g. "file exists but content degenerate: selector 'rows[*].per_metric.*.mean' returned only NaN values (3 files matched but all fields NaN)..."). 80-char truncation 保留 key info (selector path) 同时 keep block readable。

5. **Compact block format**: `  <id>  [<section>]` / `    reason      : ...` / `    matched_files: ...` — indent 让视觉 grouping clear,8-space indent for field labels aligns with each other。

6. **Multi-line `matched_files` comma-join**: 3 matched files → single line joined with `, ` 让 reviewer 容易 copy-paste single line into shell command (e.g. `ls -la result/personal_query/.../bootstrap_delta_ci.json`)。

7. **`--output` flag respected**: reviewer 想 `--list-degenerate --output /custom/path.json` 跟其他 mode 语义一致。

8. **Reuses iter #115 `matched_files` field**: 不需 re-implement degenerate diagnosis logic — iter #115 dashboard already extracts matched_files per degenerate claim; `--list-degenerate` 只是 shell wrapper 读 same JSON field。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --list-degenerate
Degenerate claims (1 total):
  RQ1_Table1_Hit10  [§3.2 + Table 1]
    reason      : file exists but content degenerate: selector 'rows[*].per_metric.*.mean' returne
    matched_files: result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json, result/personal_query/06_retrieval/Grocery_and_Gourmet_Food/retrieval_syntax_depth_summary.json, result/personal_query/09_noisy_retrieval/Grocery_and_Gourmet_Food/syntax_depth_correct_vs_noisy_results_pivot.json
$ echo $?
0

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
Case V: --list-degenerate CLI flag (iter #138)                                                   PASS

All 22 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case V, +docstring, +main count to 22)
- 文档: `README.md` (+1 line `--list-degenerate` example)
- 验证: 22/22 cases pass; pre-commit hook freeze 22 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --list-degenerate` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
- **iter #135** — --audit-age
- **iter #137** — --md-table
- **iter #138** — --list-degenerate (diagnose value-extraction failures) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format | `--md-table` (iter #137) |
| degenerate diagnosis | `--list-degenerate` (iter #138) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #137** — 21 cases
- **iter #138** — 22 cases (+ --list-degenerate) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample