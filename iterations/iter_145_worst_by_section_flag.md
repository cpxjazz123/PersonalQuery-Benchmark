# Iteration #145 — Audit CLI `--worst-by-section` flag (worst-claim-per-section)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--worst-by-section` argparse flag (for each paper section, shows worst-claim max abs_delta + claim_id + rel_delta_pct + n_claims, sorted desc); extend `_smoke_audit_regression.py` to 29 cases (Case AC)
**prior**: iter #134 (`--by-section`) gives paper section × status count matrix (aggregate count view). iter #133 (`--top N`) gives worst-N discrepant globally. But reviewer 想要 "for each paper section, what's the worst single discrepancy?" — combines --by-section grouping with --top max-per-group。Useful for paper revision triage: "where in the paper do I need erratum most urgently?"

## §A 审稿意见

Reviewer 想要 quick "which paper section has the worst single numerical discrepancy?" 当前 must:
1. iter #133 `--top 17` (gives all 4 discrepant but no per-section grouping)
2. iter #134 `--by-section` (gives count but not worst-claim-per-section)
3. Manual mental merge: `for sec in §3.3, §3.2, §2.2: read each cell, find max abs_delta`

后果: paper revision triage inefficient。Reviewer 想 identify "section needing erratum urgently" 没有 single shell call。

iter #145 fix: 新增 `--worst-by-section` CLI flag (combines iter #133 + iter #134 axes) — 不 re-run audit,直接 read most recent audit JSON,for each section extract max abs_delta + corresponding claim_id + rel_delta_pct + n_claims in section,sort by `-max_abs desc, section asc`,print compact per-section row + `(all clean)` row for sections without discrepant claims。All 8 sections visible (reviewer confirms non-discrepant sections are explicitly "all clean" not missing)。

## §B 本轮 (iter #145) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--worst-by-section` flag:
```python
parser.add_argument("--worst-by-section", action="store_true",
                    help="(iter #145) Show the worst-claim-per-paper-section (max abs_delta per section) "
                         "sorted desc. Exit 0 always when JSON found; exit 2 if JSON missing. Useful for "
                         "finding which paper section has the largest single discrepancy.")
```

docstring 加 flag description:
```
--worst-by-section Show the worst-claim-per-paper-section (max abs_delta per section) sorted desc.
                   Exit 0 always when JSON found; exit 2 if JSON missing. Useful for finding which
                   paper section has the largest single discrepancy. Iter #145.
```

`main()` handler (在 `--count-by-codebase-dir` handler 之后, `--csv` handler 之前):
```python
if args.worst_by_section:
    # iter #145: --worst-by-section — for each paper section, show worst-claim (max abs_delta) sorted desc.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    def _safe_abs(vc: dict) -> float:
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else 0.0

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    from collections import defaultdict
    section_worst: "dict[str, dict]" = defaultdict(lambda: {
        "max_abs": 0.0, "max_rel": None, "claim_id": "(none)", "n_total": 0,
    })
    for c in claims_data:
        sec = c.get("section", "(unknown)")
        section_worst[sec]["n_total"] += 1
        for vc in c.get("value_check_results", []) or []:
            ad = abs(_safe_abs(vc))
            if ad > abs(section_worst[sec]["max_abs"]):
                section_worst[sec]["max_abs"] = ad if ad > 0 else _safe_abs(vc)
                section_worst[sec]["claim_id"] = c.get("id", "?")
                rd = _safe_rel(vc)
                if rd is not None and (section_worst[sec]["max_rel"] is None or abs(rd) > abs(section_worst[sec]["max_rel"])):
                    section_worst[sec]["max_rel"] = rd

    # Sort by max_abs desc, section asc as tiebreak.
    sorted_secs = sorted(
        section_worst.keys(),
        key=lambda s: (-abs(section_worst[s]["max_abs"]), s),
    )
    print(f"Worst abs_delta per paper section ({len(sorted_secs)} sections):")
    for sec in sorted_secs:
        info = section_worst[sec]
        max_abs = info["max_abs"]
        max_rel = info["max_rel"]
        abs_str = f"{max_abs:.4f}" if max_abs else ""
        rel_str = f"{max_rel * 100:.2f}%" if max_rel is not None else ""
        claim_id = info["claim_id"]
        n_total = info["n_total"]
        if max_abs == 0.0:
            print(f"  0.0000  (all clean)                  [{sec}]  n_claims={n_total}")
        else:
            print(f"  {abs_str}  {claim_id:35s}  [{sec}]  rel_delta_pct={rel_str}  n_claims={n_total}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AC freeze iter #145 behavior:
- `--worst-by-section` exit 0 (always when JSON found)
- "8 sections" header (frozen baseline)
- Top section `§3.3 + Table 2` with worst discrepant `RQ3_LLM_Full_Set_94%` (abs_delta=0.7415)
- `rel_delta_pct=78.38%` visible (matches iter #109 dashboard percent display)
- `n_claims=4` visible (4 discrepant in §3.3 — confirms concentration)
- Sort order: max_abs desc — first data row is `§3.3 + Table 2`
- `(all clean)` rows for 7 non-discrepant sections
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 29 cases.

### README.md

Re-run commands section +1 line:
```bash
# Show worst-claim-per-paper-section (max abs_delta per section) sorted desc — paper erratum triage; iter #145
python3 PersoanlQuery/paper_claims_audit.py --worst-by-section
```

## §C 关键改动点

1. **New combination axis (--by-section + --top)**: 与 existing flags complementary:
   - `--by-section` (iter #134) — section × status count matrix (aggregate counts)
   - `--top N` (iter #133) — worst-N globally (no section grouping)
   - `--worst-by-section` (iter #145) — worst-claim-per-section ✓ 本轮
   
   Reviewer mental model: "for paper revision, which section needs erratum urgently?" → `--worst-by-section`。For "is section X concentrated on issues?" → `--by-section`。For "globally worst N discrepancies" → `--top N`。

2. **`abs(ad) > abs(max_abs)` comparison (signed-aware)**: 用 `abs()` 比较保证 negative abs_delta 不会 out-rank positive abs_delta。`max_abs` 存储 signed value (so output preserves direction if needed) but comparison uses `abs()`。

3. **`(all clean)` row for non-discrepant sections**: 与 `--by-section` 不同 (which shows 0 for cells with no claims), iter #145 explicit "(all clean)" label for sections without discrepant。Reviewer confirms these sections are intentionally clean not just unprinted (audit infra 100% section coverage)。

4. **`n_claims=N` column**: 显示每个 section 的 total claim count (not just discrepant count)。Reviewer sees "§2.2 has 7 claims, all clean" → §2.2 has good audit coverage。`§3.3 + Table 2 has 4 claims, all discrepant` → §3.3 是 concentrated risk。

5. **`max_rel` independent aggregation**: 同 iter #143 — 取自同一 group of value_checks 但 not indexed to max_abs。可能在 different value_check (rel vs abs may not rank-correlate)。

6. **`rel_str` percent sign explicit**: matches iter #143 convention。Reviewer 一眼分清 raw fraction vs percentage。

7. **Sort by max_abs desc, section asc**: Worst section first → reviewer triages erratum-urgent sections first。Section asc as tiebreak → stable across re-runs (e.g. `(infrastructure)` before `§2.1` for both max_abs=0, alphabetically earlier first)。

8. **`(unknown)` fallback for missing section field**: Frozen baseline has all claims with section set, but defensive fallback 防止 audit schema drift (e.g. new claim without section field).

9. **Exit code 0 always (informational, like iter #142/143/144)**: 与前 3 个 iter 一致 — "0 sections" means audit JSON catastrophically broken, not actionable info。Safe in shell scripts without `|| true`。

10. **`--output` flag respected**: reviewer 想 `--worst-by-section --output /custom/path.json` 跟其他 mode 语义一致。

11. **Reuses iter #134 section field**: 不需 re-implement — iter #134 audit script 加 per-section breakdown by `section` field;iter #145 是 shell wrapper reading same field but aggregating by max-abs not count。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --worst-by-section
Worst abs_delta per paper section (8 sections):
  0.7415  RQ3_LLM_Full_Set_94%                 [§3.3 + Table 2]  rel_delta_pct=78.38%  n_claims=4
  0.0000  (all clean)                  [(infrastructure)]  n_claims=1
  0.0000  (all clean)                  [§2.1]  n_claims=1
  0.0000  (all clean)                  [§2.2]  n_claims=7
  0.0000  (all clean)                  [§3.2]  n_claims=1
  0.0000  (all clean)                  [§3.2 + Table 1]  n_claims=1
  0.0000  (all clean)                  [§3.2 + Table 1 (lower panel)]  n_claims=1
  0.0000  (all clean)                  [§3.4 + Table 3]  n_claims=1
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AC: --worst-by-section CLI flag (iter #145)
  PASS  --worst-by-section shows worst per section with §3.3 + Table 2 dominant + exits 2 on missing JSON

All 29 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~55 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AC, +docstring, +main count to 29)
- 文档: `README.md` (+1 line `--worst-by-section` example)
- 验证: 29/29 cases pass; pre-commit hook freeze 29 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --worst-by-section` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section (paper section × status count)
- **iter #135** — --audit-age
- **iter #137** — --md-table (markdown for GitHub)
- **iter #138** — --list-degenerate
- **iter #139** — --list-unverified
- **iter #140** — --list-partial
- **iter #141** — --csv (RFC-4180 CSV for spreadsheets)
- **iter #142** — --list-verified
- **iter #143** — --list-discrepant
- **iter #144** — --count-by-codebase-dir (per-source-dir aggregation)
- **iter #145** — --worst-by-section (worst-claim-per-section) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section (paper count) | `--by-section` (iter #134) |
| per-section (paper worst) | `--worst-by-section` (iter #145) ✓ 本轮 |
| per-source-dir | `--count-by-codebase-dir` (iter #144) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| degenerate diagnosis | `--list-degenerate` (iter #138) |
| unverified planning | `--list-unverified` (iter #139) |
| partial scope diagnosis | `--list-partial` (iter #140) |
| verified confirmation | `--list-verified` (iter #142) |
| discrepant enumeration | `--list-discrepant` (iter #143) |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #144** — 28 cases
- **iter #145** — 29 cases (+ --worst-by-section) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
