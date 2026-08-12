# Iteration #142 — Audit CLI `--list-verified` flag (confirm evidence-backed claims)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--list-verified` argparse flag (prints verified claims with id + section + expected_outputs — the "evidence-backed" set); extend `_smoke_audit_regression.py` to 26 cases (Case Z)
**prior**: iter #138 (`--list-degenerate`), iter #139 (`--list-unverified`), iter #140 (`--list-partial`) established per-status list pattern covering the 3 actionable failure axes. Reviewer 想 see "what does the audit actually consider evidence-backed?" 当前 must `jq '.claims[] | select(.status == "verified") | ...'` — and iter #131 `--status-summary` 1-line gives count but no per-claim detail. Verified listing 是 the per-status list trifecta 的 natural symmetry-completer (3 failure axes + 1 success axis → full status coverage)。

## §A 审稿意见

Reviewer 想 confirm "audit considers these 6 claims evidence-backed" 当前 must:
1. `jq '.claims[] | select(.status == "verified") | {id, section, expected_outputs}'`
2. Or open dashboard HTML + scroll to non-discrepant claims + mental mark green badges
3. Or `--claim-id <id> --verbose` × 6

后果: audit "positive coverage" 不可见。Reviewer 想要 quick sanity check "what evidence supports the verified claims?" 没有 CLI affordance。

iter #142 fix: 新增 `--list-verified` CLI flag (mirror iter #138/139/140 pattern) — 不 re-run audit,直接 read most recent audit JSON,filter `status == "verified"`,print compact per-claim block with id + section + expected_outputs bullet list。Complete the per-status list quadrants:
- iter #138: degenerate (matched_files for value-extraction failures)
- iter #139: unverified (code_evidence for code-reference-but-no-output)
- iter #140: partial (expected_outputs for audit-scope-ambiguous)
- iter #142: verified (expected_outputs for evidence-backed set) ✓ 本轮

Note: 与 partial 不同 (partial means "audit scope ambiguous"), verified means "audit explicitly confirms evidence". Both 展示 expected_outputs but semantic opposite:
- partial = "code exists, no specific outputs claimed" → `(none recorded)` 是 expected
- verified = "code exists, all output globs match" → each glob is a proven file path

## §B 本轮 (iter #142) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--list-verified` flag:
```python
parser.add_argument("--list-verified", action="store_true",
                    help="(iter #142) Print just the 'verified' claims with id, section, and expected_outputs "
                         "(one bullet per glob). Exit 0 always when JSON found; exit 2 if JSON missing. "
                         "Useful for confirming which claims the audit considers evidence-backed.")
```

docstring 加 flag description:
```
--list-verified   Print just the 'verified' claims with id, section, and expected_outputs
                   (one bullet per glob). Exit 0 always when JSON found; exit 2 if JSON missing.
                   Useful for confirming what the audit considers evidence-backed. Iter #142.
```

`main()` handler (在 `--list-partial` handler 之后, `--csv` handler 之前):
```python
if args.list_verified:
    # iter #142: --list-verified — print verified claims with expected_outputs (evidence-backed set).
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    verified = [c for c in claims_data if c.get("status") == "verified"]
    print(f"Verified claims ({len(verified)} total, evidence-backed):")
    for c in verified:
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

加 Case Z freeze iter #142 behavior:
- `--list-verified` exit 0 (always when JSON found, even if 0 verified)
- "Verified claims (6 total, evidence-backed)" header (frozen baseline: 6 verified)
- All 6 frozen verified IDs present: `RQ1_Delta_Range`, `RQ2_Table1_Drop`, `Pipeline_Regeneration_10x10`, `BPE_aware_Error_Injection`, `UserFilter_20_reviews_15_words`, `Sec2_5_attrs_per_query`
- `expected_outputs` field label present
- `evidence-backed` qualifier in header
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 26 cases.

### README.md

Re-run commands section +1 line:
```bash
# List just the 'verified' claims (the audit considers evidence-backed) with id, section, expected_outputs; iter #142
python3 PersoanlQuery/paper_claims_audit.py --list-verified
```

## §C 关键改动点

1. **Per-status list quadrants complete**: 4 flags covering all 4 actionable statuses:
   - `--list-degenerate` (iter #138) — value-extraction failures (matched_files field)
   - `--list-unverified` (iter #139) — code-reference-but-no-output (code_evidence field)
   - `--list-partial` (iter #140) — audit-scope-ambiguous (expected_outputs field)
   - `--list-verified` (iter #142) — evidence-backed (expected_outputs field) ✓ 本轮
   
   Note: `verified_value_match` 是另一个 status (file exists, values match paper) 但 frozen baseline = 0 → no need flag yet. If iter #87/#88 flips any non-verified → verified_value_match, 可以加 `--list-verified-value-match` 后续。

2. **Mirror iter #140 pattern exactly (expected_outputs field)**: Both partial and verified claim 用 expected_outputs as primary diagnostic field,但语义 opposite:
   - partial = "code exists, audit couldn't enumerate outputs" → `(none recorded)` 是 norm
   - verified = "code exists, all output globs match" → 每个 glob 是 a concrete file path

3. **Exit code 0 always (not 0/1/2 triple)**: 与 iter #138/139/140 不同 (which exit 1 if no claims in that status), iter #142 always exits 0 when JSON found。Reason: "0 verified" 意味着 audit catastrophically broken (e.g. audit JSON has wrong shape) — not actionable info for reviewer。Verified listing is informational, not diagnostic。Safe to use in shell scripts without `|| true`。

4. **`evidence-backed` qualifier in header**: 显式说明语义 "what audit considers evidence-backed" → 与 `partial` 的 "audit-scope-ambiguous" 形成对照。Reviewer 一眼分清 positive vs negative list。

5. **Reuses iter #112 `expected_outputs` field**: 不需 re-implement — iter #112 dashboard 加了 provenance panels 包含 expected_outputs;`--list-verified` 是 shell wrapper 读 same field 但语义 opposite to `--list-partial`。

6. **`--output` flag respected**: reviewer 想 `--list-verified --output /custom/path.json` 跟其他 mode 语义一致。

7. **No re-run audit**: `--list-verified` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --list-verified
Verified claims (6 total, evidence-backed):
  RQ1_Delta_Range  [§3.2]
    reason      : 1/1 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/06_retrieval/<cat>/retrieval_syntax_depth_summary_pivot.json
  RQ2_Table1_Drop  [§3.2 + Table 1 (lower panel)]
    reason      : 2/2 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/09_noisy_retrieval/<cat>/syntax_depth_correct_vs_noisy_results_pivot.json
      - result/personal_query/07_inject_noisy/<cat>/noisy_query.json
  Pipeline_Regeneration_10x10  [§2.2]
    reason      : 1/1 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/04_query/<cat>/query_by_syntax_depth_*.json
  BPE_aware_Error_Injection  [§2.2]
    reason      : 1/1 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/07_inject_noisy/<cat>/noisy_query.json
  UserFilter_20_reviews_15_words  [§2.1]
    reason      : 1/1 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/00_data_preparation/ablation/long_sentence_threshold_ablation.json
  Sec2_5_attrs_per_query  [§2.2]
    reason      : 1/1 output globs match (no numerical claim to validate)
    expected_outputs:
      - result/personal_query/04_query/<cat>/query_by_syntax_depth_*.json
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case Z: --list-verified CLI flag (iter #142)
  PASS  --list-verified lists 6 verified claims with expected_outputs + exits 2 on missing JSON

All 26 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case Z, +docstring, +main count to 26)
- 文档: `README.md` (+1 line `--list-verified` example)
- 验证: 26/26 cases pass; pre-commit hook freeze 26 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --list-verified` (~100ms, 不 re-run audit)

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
- **iter #141** — --csv (RFC-4180 CSV for spreadsheets)
- **iter #142** — --list-verified (per-status list symmetry-completer) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| degenerate diagnosis | `--list-degenerate` (iter #138) |
| unverified planning | `--list-unverified` (iter #139) |
| partial scope diagnosis | `--list-partial` (iter #140) |
| verified confirmation | `--list-verified` (iter #142) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

Per-status list quadrants (4/4 complete):
- `--list-degenerate` (iter #138) — value-extraction failures
- `--list-unverified` (iter #139) — code-reference-but-no-output
- `--list-partial` (iter #140) — audit-scope-ambiguous
- `--list-verified` (iter #142) — evidence-backed ✓ 本轮

regression test coverage timeline:
- **iter #141** — 25 cases
- **iter #142** — 26 cases (+ --list-verified) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
