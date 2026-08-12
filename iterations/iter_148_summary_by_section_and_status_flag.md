# Iteration #148 — Audit CLI `--summary-by-section-and-status` flag (paper-section 2D matrix)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--summary-by-section-and-status` argparse flag (2D matrix: paper section rows × audit status columns with per-cell counts + footer TOTAL row); extend `_smoke_audit_regression.py` to 32 cases (Case AF)
**prior**: iter #147 (`--summary-by-source-dir-and-status`) gives source-dir × status 2D matrix. iter #134 (`--by-section`) gives section × status count matrix but cells just show counts (no per-row alignment). Reviewer 想要 same 2D matrix visual layout for paper sections as iter #147 gives for source dirs — symmetry between paper-side and code-side matrix views。

## §A 审稿意见

Reviewer 想要 quick "for each paper section, how many claims per audit status?" 当前 must:
1. iter #134 `--by-section` — gives section × status count matrix but columns are status-labels (e.g. `disc / degen / part / unver`) and rows are sections — visual scan works but not as crisp as iter #147 layout (which has full status names + `total` column + TOTAL footer)
2. iter #147 `--summary-by-source-dir-and-status` — same matrix layout but for source dirs → reviewer can compare but axes differ
3. Manual pivot: extract from iter #134 output + manually arrange into matrix

后果: paper-side matrix view 不 match code-side iter #147 layout → reviewer mental load when switching axes。

iter #148 fix: 新增 `--summary-by-section-and-status` CLI flag (paper-section counterpart to iter #147) — 不 re-run audit,直接 read most recent audit JSON,group claims by paper section (read from `section` field) + by audit status,render 2D matrix with rows = paper sections (sorted by total desc),columns = 7 audit statuses (verified_value_match / verified / discrepant / degenerate / partial / unverified / blocked),per-cell = count + footer `TOTAL` row with column sums + grand total。Mirrors iter #147 structure exactly → reviewer can swap axes (paper section vs source dir) with no cognitive load。

## §B 本轮 (iter #148) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--summary-by-section-and-status` flag:
```python
parser.add_argument("--summary-by-section-and-status", action="store_true",
                    help="(iter #148) 2D matrix: paper section (rows) × audit status (columns) with "
                         "per-cell counts and footer TOTAL. Exit 0 always when JSON found; exit 2 if "
                         "JSON missing. Paper-section counterpart to iter #147.")
```

docstring 加 flag description:
```
--summary-by-section-and-status
                     2D matrix: paper section (rows) × audit status (columns) with per-cell counts
                     and footer TOTAL. Exit 0 always when JSON found; exit 2 if JSON missing.
                     Paper-section counterpart to iter #147. Iter #148.
```

`main()` handler (在 `--summary-by-source-dir-and-status` handler 之后, `--csv` handler 之前):
```python
if args.summary_by_section_and_status:
    # iter #148: --summary-by-section-and-status — 2D matrix paper section × status with counts + footer TOTAL.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    # Same frozen status order as iter #134 / #147.
    STATUSES = ["verified_value_match", "verified", "discrepant",
                "degenerate", "partial", "unverified", "blocked"]

    from collections import defaultdict
    matrix: "dict[str, dict[str, int]]" = defaultdict(lambda: defaultdict(int))
    for c in claims_data:
        sec = c.get("section", "(unknown)")
        matrix[sec][c.get("status", "?")] += 1

    sorted_secs = sorted(matrix.keys(), key=lambda s: (-sum(matrix[s].values()), s))
    col_width = max(len(s) for s in STATUSES)
    sec_width = max(len(s) for s in sorted_secs)
    header = f"{'paper section':<{sec_width}}  " + "  ".join(f"{s:>{col_width}}" for s in STATUSES) + "  total"
    print(f"Paper section × status matrix ({len(claims_data)} claims across {len(sorted_secs)} sections):")
    print(header)
    col_totals = {s: 0 for s in STATUSES}
    grand_total = 0
    for sec in sorted_secs:
        cnts = matrix[sec]
        row_total = sum(cnts.values())
        cells = "  ".join(f"{cnts.get(s, 0):>{col_width}d}" for s in STATUSES)
        print(f"{sec:<{sec_width}}  {cells}  {row_total}")
        for s in STATUSES:
            col_totals[s] += cnts.get(s, 0)
        grand_total += row_total
    footer_cells = "  ".join(f"{col_totals[s]:>{col_width}d}" for s in STATUSES)
    print(f"{'TOTAL':<{sec_width}}  {footer_cells}  {grand_total}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AF freeze iter #148 behavior:
- `--summary-by-section-and-status` exit 0 (always when JSON found)
- "17 claims across 8 sections" header (frozen baseline)
- All 7 status column headers visible (full names, not abbreviated)
- Top section `§2.2` visible (7 claims: 3 verified + 4 unverified)
- `§3.3 + Table 2` visible (4 discrepant = concentrated risk)
- Footer `TOTAL` row with per-status sums + grand total = 17
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 32 cases.

### README.md

Re-run commands section +1 line:
```bash
# 2D matrix paper section (rows) × audit status (columns) with counts + footer TOTAL; iter #148
python3 PersoanlQuery/paper_claims_audit.py --summary-by-section-and-status
```

## §C 关键改动点

1. **Paper-section counterpart to iter #147**: 与 existing flags complementary:
   - iter #134 `--by-section` — section × status count matrix (column-label format)
   - iter #147 `--summary-by-source-dir-and-status` — source dir × status 2D matrix (full status names + TOTAL footer)
   - iter #148 `--summary-by-section-and-status` — paper section × status 2D matrix ✓ 本轮
   
   Reviewer mental model: "for paper-section view with crisp layout → iter #148; for source-Stage matrix view → iter #147; for compact count view → iter #134"。

2. **Mirrors iter #147 structure exactly**: Same column layout (7 statuses in frozen order), same `TOTAL` footer row, same right-aligned cell counts, same sort order (row total desc, label asc)。Visual symmetry between `--summary-by-source-dir-and-status` and `--summary-by-section-and-status` → reviewer can switch axes with zero cognitive load。Footer TOTAL row matches `--status-summary` baseline (0/6/4/1/1/5/0 = 17) — same cross-validation property as iter #147。

3. **Top section: §2.2 with 7 claims (3 verified + 4 unverified)**: Confirms §2.2 is "concentrated infrastructure" — 4 of 5 unverified claims are in §2.2 (Sec2_*) + 3 verified (BPE_aware_Error_Injection, Pipeline_Regeneration_10x10, Sec2_5_attrs_per_query)。Section audit coverage spans both "verified working" and "unverified pending Stage 10/12 re-run" → §2.2 is mixed signal。

4. **Concentrated-risk section: §3.3 + Table 2 with 4 discrepant**: All 4 RQ3 metric discrepancies are in §3.3 + Table 2 → confirms iter #145 finding that this is the worst-claim-per-section. Paper erratum triage: §3.3 + Table 2 needs erratum urgently。

5. **`(unknown)` fallback for missing section field**: matches iter #145 (`--worst-by-section`) behavior → defensive against audit schema drift (claim without section field falls into `(unknown)` row)。Frozen baseline = 0 such claims → not displayed in current matrix。

6. **Sort by row total desc, section asc as tiebreak**: Largest section first → reviewer triages biggest bucket first。`sec_width = max(len(s) for s in sorted_secs)` auto-compute → fits longest section label (`§3.2 + Table 1 (lower panel)` = 30 chars)。

7. **Reuses iter #147 STATUSES frozen order**: Same 7-status column header (verified_value_match/verified/discrepant/degenerate/partial/unverified/blocked) → visual consistency across iter #147 and iter #148 matrices。

8. **Exit code 0 always (informational, like iter #144/145/146/147)**: 与前 4 iter 一致 — "0 sections" means audit JSON catastrophically broken, not actionable info。Safe in shell scripts without `|| true`。

9. **`--output` flag respected**: reviewer 想 `--summary-by-section-and-status --output /custom/path.json` 跟其他 mode 语义一致。

10. **No re-run audit**: `--summary-by-section-and-status` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

11. **Reuses iter #134 section field**: 不需 re-implement — iter #134 audit script 加 per-section breakdown by `section` field;iter #148 是 shell wrapper reading same field but rendering as 2D matrix with full status names + TOTAL footer。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --summary-by-section-and-status
Paper section × status matrix (17 claims across 8 sections):
paper section                 verified_value_match              verified            discrepant            degenerate               partial            unverified               blocked  total
§2.2                                             0                     3                     0                     0                     0                     4                     0  7
§3.3 + Table 2                                   0                     0                     4                     0                     0                     0                     0  4
(infrastructure)                                 0                     0                     0                     0                     1                     0                     0  1
§2.1                                             0                     1                     0                     0                     0                     0                     0  1
§3.2                                             0                     1                     0                     0                     0                     0                     0  1
§3.2 + Table 1                                   0                     0                     0                     1                     0                     0                     0  1
§3.2 + Table 1 (lower panel)                     0                     1                     0                     0                     0                     0                     0  1
§3.4 + Table 3                                   0                     0                     0                     0                     0                     1                     0  1
TOTAL                                            0                     6                     4                     1                     1                     5                     0  17
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AF: --summary-by-section-and-status CLI flag (iter #148)
  PASS  --summary-by-section-and-status renders 2D matrix with §2.2 + §3.3 + Table 2 + exits 2 on missing JSON

All 32 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~45 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AF, +docstring, +main count to 32)
- 文档: `README.md` (+1 line `--summary-by-section-and-status` example)
- 验证: 32/32 cases pass; pre-commit hook freeze 32 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --summary-by-section-and-status` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section (paper section × status count matrix)
- **iter #135** — --audit-age
- **iter #137** — --md-table
- **iter #138** — --list-degenerate
- **iter #139** — --list-unverified
- **iter #140** — --list-partial
- **iter #141** — --csv
- **iter #142** — --list-verified
- **iter #143** — --list-discrepant
- **iter #144** — --count-by-codebase-dir
- **iter #145** — --worst-by-section
- **iter #146** — --find-claim PAT
- **iter #147** — --summary-by-source-dir-and-status
- **iter #148** — --summary-by-section-and-status ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| paper section × status (compact) | `--by-section` (iter #134) |
| paper section × status (matrix) | `--summary-by-section-and-status` (iter #148) ✓ 本轮 |
| source dir × status (matrix) | `--summary-by-source-dir-and-status` (iter #147) |
| source dir (1D text) | `--count-by-codebase-dir` (iter #144) |
| paper section worst | `--worst-by-section` (iter #145) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| list by status | `--list-{degenerate,unverified,partial,verified,discrepant}` (iter #138-143) |
| search by substring | `--find-claim` (iter #146) |
| flip detection | `--diff` (iter #114) |

Audit CLI matrix views taxonomy (now 2D complete):
| view | axis | flag |
|------|------|------|
| paper-side matrix (compact) | section × status (count cells) | `--by-section` (iter #134) |
| paper-side matrix (full) | section × status (full status names + TOTAL footer) | `--summary-by-section-and-status` (iter #148) ✓ 本轮 |
| code-side matrix (full) | source dir × status (full status names + TOTAL footer) | `--summary-by-source-dir-and-status` (iter #147) |

regression test coverage timeline:
- **iter #147** — 31 cases
- **iter #148** — 32 cases (+ --summary-by-section-and-status) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
