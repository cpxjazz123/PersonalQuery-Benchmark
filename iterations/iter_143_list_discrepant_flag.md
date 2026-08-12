# Iteration #143 — Audit CLI `--list-discrepant` flag (enumerate audit-mismatch claims)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--list-discrepant` argparse flag (prints discrepant claims with id + section + max abs_delta + max rel_delta_pct + reason, sorted by abs_delta desc); extend `_smoke_audit_regression.py` to 27 cases (Case AA)
**prior**: iter #133 (`--top N`) covers worst-N discrepant (configurable N), iter #141 (`--csv`) covers all 17 claims (machine-friendly), iter #137 (`--md-table`) covers all 17 claims (markdown shareable). But reviewer 想要 just the discrepant subset for cross-checking paper Section 3.3 had to filter manually. iter #138-140 (`--list-degenerate/unverified/partial`) and iter #142 (`--list-verified`) cover 4 of 5 per-status axes; this iter completes the per-status list set with discrepant — the most actionable status (numerical mismatches requiring paper text vs code cross-check).

## §A 审稿意见

Discrepant status = "file exists but values disagree with paper" (4 frozen claims, all in §3.3 + Table 2 — worst 78.4% relative delta for RQ3_LLM_Full_Set_94%). Reviewer 想要 enumerate all 4 claims for:
1. **Paper text cross-check**: each discrepant claim → check paper §3.3 footnote + actual code output file → decide if paper needs erratum
2. **Pre-meeting prep**: before discussing RQ3 with co-authors, list "exactly which metrics disagree by how much"
3. **Post-iter #87/#88 verification**: after Stage 6/9/12 re-run, check if any discrepancy flipped → verified_value_match

Current must:
1. `--top 4` (gives 4 discrepant but no per-claim id+section structured block)
2. `--csv` + grep `"discrepant"` (machine-friendly but loses readability)
3. `--md-table` + grep `discrepant` (readable but mixed with verified rows)

后果: 没有 dedicated CLI affordance for "just the 4 discrepant claims" → reviewer 用 3-step manual filter。

iter #143 fix: 新增 `--list-discrepant` CLI flag (mirror iter #138-140/142 pattern) — 不 re-run audit,直接 read most recent audit JSON,filter `status == "discrepant"`,print compact per-claim block: `  <id>  [<section>]  abs_delta=<max-abs>  rel_delta_pct=<max-rel-pct>` + `    reason      : <80-char>`. Sort by `-max_abs_delta desc` (matches `--top N` / `--csv` ordering for visual consistency)。`max_abs` 取自 `value_check_results[*].abs_delta` 的 max — handles claims with multiple value_checks (e.g. RQ3_LLM_Full_Set_94% has 3 value_checks, max=0.7415)。

## §B 本轮 (iter #143) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--list-discrepant` flag:
```python
parser.add_argument("--list-discrepant", action="store_true",
                    help="(iter #143) Print just the 'discrepant' claims with id, section, max abs_delta, "
                         "max rel_delta_pct, and reason (sorted by abs_delta desc). Exit 0 always when JSON "
                         "found; exit 2 if JSON missing. Useful for enumerating all audit-mismatch claims.")
```

docstring 加 flag description:
```
--list-discrepant Print just the 'discrepant' claims with id, section, max abs_delta, max rel_delta_pct,
                   and reason (sorted by abs_delta desc). Exit 0 always when JSON found; exit 2 if
                   JSON missing. Useful for enumerating all audit-mismatch claims. Iter #143.
```

`main()` handler (在 `--list-verified` handler 之后, `--csv` handler 之前):
```python
if args.list_discrepant:
    # iter #143: --list-discrepant — print discrepant claims with max abs_delta + rel_delta_pct, sorted desc.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    discrepant = [c for c in claims_data if c.get("status") == "discrepant"]

    def _safe_abs(vc: dict) -> float:
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else 0.0

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    def _claim_max_abs(c: dict) -> float:
        vcs = c.get("value_check_results", [])
        return max((_safe_abs(vc) for vc in vcs), default=0.0)

    discrepant_sorted = sorted(discrepant, key=_claim_max_abs, reverse=True)
    print(f"Discrepant claims ({len(discrepant_sorted)} total, audit-mismatch):")
    for c in discrepant_sorted:
        cid = c.get("id", "?")
        sec = c.get("section", "?")
        vcs = c.get("value_check_results", [])
        max_abs = _claim_max_abs(c)
        max_rel = None
        for vc in vcs:
            rd = _safe_rel(vc)
            if rd is None:
                continue
            if max_rel is None or abs(rd) > abs(max_rel):
                max_rel = rd
        reason = (c.get("reason") or "")[:80]
        abs_str = f"{max_abs:.4f}" if max_abs else ""
        rel_str = f"{max_rel * 100:.2f}%" if max_rel is not None else ""
        print(f"  {cid}  [{sec}]  abs_delta={abs_str}  rel_delta_pct={rel_str}")
        print(f"    reason      : {reason}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AA freeze iter #143 behavior:
- `--list-discrepant` exit 0 (always when JSON found, even if 0 discrepant)
- "Discrepant claims (4 total, audit-mismatch)" header (frozen baseline: 4 discrepant)
- All 4 frozen discrepant IDs present: `RQ3_LLM_Full_Set_94%`, `RQ3_MAE_0.89`, `RQ3_Fleiss_Kappa_0.72`, `RQ3_Spearman_0.81`
- Top-1 row in output must be `RQ3_LLM_Full_Set_94%` (max abs_delta=0.7415 across all value_checks)
- `abs_delta=0.7415` visible in top-1 row
- `rel_delta_pct=78.38%` visible (matches iter #109 dashboard percent display)
- All 4 rows sorted desc by abs_delta (0.7415, 0.2700, 0.0965, 0.0936)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 27 cases.

### README.md

Re-run commands section +1 line:
```bash
# List just the 'discrepant' claims (audit-mismatch) with id, section, max abs_delta, rel_delta_pct; iter #143
python3 PersoanlQuery/paper_claims_audit.py --list-discrepant
```

## §C 关键改动点

1. **Per-status list set complete (5/5)**: 5 flags covering all 5 actionable statuses:
   - `--list-degenerate` (iter #138) — value-extraction failures (matched_files field)
   - `--list-unverified` (iter #139) — code-reference-but-no-output (code_evidence field)
   - `--list-partial` (iter #140) — audit-scope-ambiguous (expected_outputs field)
   - `--list-verified` (iter #142) — evidence-backed (expected_outputs field)
   - `--list-discrepant` (iter #143) — audit-mismatch (max abs_delta + max rel_delta_pct) ✓ 本轮

2. **`max_abs_delta` aggregation**: discrepant claims may have multiple `value_check_results` (e.g. RQ3_LLM_Full_Set_94% has 3 — each comparing one paper-cited number against one code-output number). `--list-discrepant` shows the MAX abs_delta (most actionable single number per claim), 与 `--top N` / `--csv` ordering 一致。All 3 view modes (top, csv, list) sort by max abs_delta desc → reviewer mental model 一致 across modes。

3. **`max_rel_delta_pct` aggregation (mirror max_abs)**: 取自同一 group of value_checks,与 max_abs 的 position may differ (claim A might have max abs in vc[0] but max rel in vc[1])。We 独立 compute max_rel over all value_checks rather than indexing the same vc as max_abs。Rationale: rel and abs may not rank-correlate (e.g. small absolute diff at large denominator = large rel diff)。

4. **`abs_str = f"{max_abs:.4f}"` 4 decimal places** (mirror iter #141): sufficient precision for 78.4% rel_delta reproducibility。Empty string when max_abs == 0.0 → blank cell。

5. **`rel_str = f"{max_rel * 100:.2f}%"` 2 decimal places + percent sign**: 2 decimals sufficient for rel_delta (different from abs_delta 4 decimals because rel is bounded by percent display convention)。`%` suffix explicit 让 reviewer 一眼看出 "this is percentage not raw fraction"。

6. **`abs_delta=` / `rel_delta_pct=` key=value format (not table)**: key=value 让 reviewer grep-friendly:
   ```
   grep "abs_delta=0.7" → only RQ3_LLM_Full_Set_94%
   grep "rel_delta_pct=30" → only RQ3_MAE_0.89
   ```
   Whereas table format (mirroring iter #133 `--top`) would be: `1   RQ3_LLM_Full_Set_94%   0.7415   78.4%   ...` — harder to grep precisely。

7. **Exit code 0 always (not 0/1/2 triple)**: 与 iter #142 (`--list-verified`) 一致 — "0 discrepant" 是 ideal post-iter #87/#88 state,但 not actionable info。Verified/discrepant are informational not diagnostic → safe in shell scripts without `|| true`。

8. **`--output` flag respected**: reviewer 想 `--list-discrepant --output /custom/path.json` 跟其他 mode 语义一致。

9. **No re-run audit**: `--list-discrepant` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

10. **Reuses iter #110 `value_check_results` field + iter #109 percent display convention**: 不需 re-implement — iter #110 audit script 加 value_check panel with abs_delta + rel_delta; iter #109 dashboard 加 percent display × 100;`--list-discrepant` 是 shell wrapper presenting same data in key=value format。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --list-discrepant
Discrepant claims (4 total, audit-mismatch):
  RQ3_LLM_Full_Set_94%  [§3.3 + Table 2]  abs_delta=0.7415  rel_delta_pct=78.38%
    reason      : file exists but value mismatch: expected=0.9460 actual=0.2045 delta=0.7415 (rel 
  RQ3_MAE_0.89  [§3.3 + Table 2]  abs_delta=0.2700  rel_delta_pct=30.34%
    reason      : file exists but value mismatch: expected=0.8900 actual=0.6200 delta=0.2700 (rel 
  RQ3_Fleiss_Kappa_0.72  [§3.3 + Table 2]  abs_delta=0.0965  rel_delta_pct=13.40%
    reason      : file exists but value mismatch: expected=0.7200 actual=0.6235 delta=0.0965 (rel 
  RQ3_Spearman_0.81  [§3.3 + Table 2]  abs_delta=0.0936  rel_delta_pct=11.56%
    reason      : file exists but value mismatch: expected=0.8100 actual=0.7164 delta=0.0936 (rel 
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AA: --list-discrepant CLI flag (iter #143)
  PASS  --list-discrepant lists 4 discrepant claims sorted by abs_delta desc + exits 2 on missing JSON

All 27 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~50 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AA, +docstring, +main count to 27)
- 文档: `README.md` (+1 line `--list-discrepant` example)
- 验证: 27/27 cases pass; pre-commit hook freeze 27 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --list-discrepant` (~100ms, 不 re-run audit)

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
- **iter #142** — --list-verified
- **iter #143** — --list-discrepant ✓ 本轮

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
| verified confirmation | `--list-verified` (iter #142) |
| discrepant enumeration | `--list-discrepant` (iter #143) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

Per-status list set complete (5/5 actionable statuses):
- `--list-degenerate` (iter #138) — value-extraction failures
- `--list-unverified` (iter #139) — code-reference-but-no-output
- `--list-partial` (iter #140) — audit-scope-ambiguous
- `--list-verified` (iter #142) — evidence-backed
- `--list-discrepant` (iter #143) — audit-mismatch ✓ 本轮

regression test coverage timeline:
- **iter #142** — 26 cases
- **iter #143** — 27 cases (+ --list-discrepant) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
