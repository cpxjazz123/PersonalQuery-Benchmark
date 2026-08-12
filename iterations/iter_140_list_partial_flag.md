# Iteration #140 — Audit CLI `--list-partial` flag (diagnose incomplete audit scope)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--list-partial` argparse flag (prints partial claims with id + section + expected_outputs); extend `_smoke_audit_regression.py` to 24 cases (Case X)
**prior**: partial status means "code referenced but no specific output files enumerated for verification" (e.g. `Pipeline_Skip_ColBERTv2_SPLADE` — code that explicitly skips two retrievers, audit JSON can't enumerate outputs that don't exist). Reviewer wanting to diagnose incomplete audit scope had to grep audit JSON. Iter #138 (`--list-degenerate`) and iter #139 (`--list-unverified`) established the per-status list pattern; this iter completes the trifecta with `--list-partial`.

## §A 审稿意见

Partial status 是 audit 自身的"已知未知" (known unknown) — claim code exists 在 repo,但 audit JSON 没法 pin 哪个 output file 应被 produce。常见 case:
- `Pipeline_Skip_ColBERTv2_SPLADE` — code 显式 skip 这两个 retrievers,所以 "no output" 才是 expected output
- 一些 infrastructure claims (e.g. ablation studies) 跟 main pipeline 不产 specific artifacts

Reviewer 想 see "which claims are partial + what scope they cover" 当前 must:
1. `jq '.claims[] | select(.status == "partial") | {id, section, expected_outputs}'`
2. Or `--claim-id <id> --verbose` × N

后果: audit scope 边界不可见。Reviewer 不 知道哪些 claims 是 "code implemented but expected-output ambiguous"。Common use case: 写 audit doc / paper section "limitations of automated audit" → 引用 partial list。

iter #140 fix: 新增 `--list-partial` CLI flag (mirror iter #138/139 pattern) — 不 re-run audit,直接 read most recent audit JSON,filter `status == "partial"`,print compact per-claim block with id + section + expected_outputs bullet list。

## §B 本轮 (iter #140) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--list-partial` flag:
```python
parser.add_argument("--list-partial", action="store_true",
                    help="(iter #140) Print just the 'partial' claims with id, section, and expected_outputs "
                         "(one bullet per glob). Exit 0 if any partial found, exit 1 if none, exit 2 if JSON missing. "
                         "Useful for diagnosing incomplete audit scope (code referenced but no specific output enumerated).")
```

docstring 加 flag description:
```
--list-partial    Print just the 'partial' claims with id, section, and expected_outputs
                   (one bullet per glob). Exit 0 if any partial, exit 1 if none. Useful for diagnosing
                   incomplete audit scope. Iter #140.
```

`main()` handler (在 `--list-unverified` handler 之后):
```python
if args.list_partial:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    partial = [c for c in claims_data if c.get("status") == "partial"]
    if not partial:
        print("No partial claims. All code-referenced claims have specific output globs enumerated.")
        return 1
    print(f"Partial claims ({len(partial)} total):")
    for c in partial:
        cid = c.get("id", "?")
        sec = c.get("section", "?")
        reason = (c.get("reason") or "")[:80]
        expected = c.get("expected_outputs", []) or []
        print(f"  {cid}  [{sec}]")
        print(f"    reason      : {reason}")
        if expected:
            print(f"    expected_outputs:")
            for gl in expected:
                print(f"      - {gl}")
        else:
            print(f"    expected_outputs: (none recorded)")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case X freeze iter #140 behavior:
- `--list-partial` exit 0 (1 partial in frozen baseline)
- "Partial claims" header
- `Pipeline_Skip_ColBERTv2_SPLADE` row present (only frozen partial)
- `(infrastructure)` section visible
- `expected_outputs` field visible (placeholder `(none recorded)` allowed)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 24 cases。

### README.md

Re-run commands section +1 line:
```bash
# List just the 'partial' claims with id, section, and expected_outputs for audit-scope diagnosis; iter #140
python3 PersoanlQuery/paper_claims_audit.py --list-partial
```

## §C 关键改动点

1. **No re-run audit**: `--list-partial` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **Triple exit code (mirror iter #138/139)**:
   - `0` (partial found) — actionable, audit scope ambiguous for these claims
   - `1` (no partial) — all code-referenced claims have specific output globs (full coverage ideal state)
   - `2` (JSON missing) — distinct from "no partial"

3. **`(none recorded)` fallback**: most partial claims have empty `expected_outputs` by definition (e.g. `Pipeline_Skip_ColBERTv2_SPLADE` explicitly skips output production)。print clear placeholder 让 reviewer 知道这是 expected "empty list" 不是 missing data。

4. **Bullet list for expected_outputs**: 每个 glob 前缀 `      - ` (6 spaces + dash) → 视觉对齐,可直接 copy-paste to shell `ls <glob>` 检查 file exists。

5. **Reason + expected_outputs both shown**: `reason` 说 "code referenced, no specific output files enumerated" (症状), `expected_outputs` 说 "哪些 globs 已 attempted (可能 empty)" (audit-scope detail)。Reviewer 可 diagnose "为何这个 claim 是 partial"。

6. **Reuses iter #112 `expected_outputs` field**: 不需 re-implement — iter #112 dashboard 加了 provenance panels 包含 expected_outputs;`--list-partial` 是 shell wrapper 读 same field。

7. **`--output` flag respected**: reviewer 想 `--list-partial --output /custom/path.json` 跟其他 mode 语义一致。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --list-partial
Partial claims (1 total):
  Pipeline_Skip_ColBERTv2_SPLADE  [(infrastructure)]
    reason      : code referenced, no specific output files enumerated for verification
    expected_outputs: (none recorded)
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
Case W: --list-unverified CLI flag (iter #139)                                                   PASS
Case X: --list-partial CLI flag (iter #140)                                                      PASS

All 24 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case X, +docstring, +main count to 24)
- 文档: `README.md` (+1 line `--list-partial` example)
- 验证: 24/24 cases pass; pre-commit hook freeze 24 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --list-partial` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
- **iter #135** — --audit-age
- **iter #137** — --md-table
- **iter #138** — --list-degenerate
- **iter #139** — --list-unverified
- **iter #140** — --list-partial (completes per-status list trifecta) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format | `--md-table` (iter #137) |
| degenerate diagnosis | `--list-degenerate` (iter #138) |
| unverified planning | `--list-unverified` (iter #139) |
| partial scope diagnosis | `--list-partial` (iter #140) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #139** — 23 cases
- **iter #140** — 24 cases (+ --list-partial) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample