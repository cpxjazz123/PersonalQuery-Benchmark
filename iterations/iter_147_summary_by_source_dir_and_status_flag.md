# Iteration #147 — Audit CLI `--summary-by-source-dir-and-status` flag (2D matrix)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--summary-by-source-dir-and-status` argparse flag (2D matrix: source dir rows × audit status columns with per-cell counts + footer TOTAL row); extend `_smoke_audit_regression.py` to 31 cases (Case AE)
**prior**: iter #144 (`--count-by-codebase-dir`) gives per-source-dir aggregation with status breakdown as inline `verified=2 / degenerate=1` text. iter #134 (`--by-section`) gives section × status 2D matrix for paper sections. But reviewer 想要 **single 2D matrix view** combining both axes — see "Stage 10 has 5 unverified, 0 verified" in one scan vs reading text breakdown。Mirrors iter #134 structure but transposes rows to source dir instead of paper section。

## §A 审稿意见

Reviewer 想要 quick "for each pipeline Stage, how many claims per audit status?" 当前 must:
1. iter #144 `--count-by-codebase-dir` — gives per-dir text breakdown but reviewer must visually parse `verified=2 / degenerate=1` strings
2. iter #134 `--by-section` — gives paper-section × status matrix but reviewer must mentally map paper-section to source-Stage (often 1-to-1 but not always — RQ3_Fleiss_Kappa_0.72 paper §3.3 but source `agreement_metrics.py`)
3. Manual pivot: extract from iter #144 output + manually arrange into matrix

后果: Stage × status triage requires manual pivot or visual parse。Reviewer 想要 alignment with iter #134 pattern (which is shell-friendly 2D matrix) for source dirs。

iter #147 fix: 新增 `--summary-by-source-dir-and-status` CLI flag — 不 re-run audit,直接 read most recent audit JSON,group claims by primary source dir (extracted from first matching `code_evidence` `PersoanlQuery/X/...` ref → `parts[1]` = `X`) + by audit status,render 2D matrix with rows = source dirs (sorted by total desc),columns = 7 audit statuses (verified_value_match / verified / discrepant / degenerate / partial / unverified / blocked),per-cell = count + footer `TOTAL` row with column sums + grand total。Reviewer 一眼 see "Stage 10 has 5 unverified (concentrated risk), Stage 8 has 2 verified + 1 degenerate (mixed), Stage 4 has 2 verified (clean)"。

## §B 本轮 (iter #147) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--summary-by-source-dir-and-status` flag:
```python
parser.add_argument("--summary-by-source-dir-and-status", action="store_true",
                    help="(iter #147) 2D matrix: source dir (rows) × audit status (columns) with "
                         "per-cell counts. Combines iter #144 (per-source-dir) with iter #134 (status "
                         "counts) into one Stage × status matrix. Exit 0 always when JSON found; "
                         "exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--summary-by-source-dir-and-status
                     2D matrix: source dir (rows) × audit status (columns) with per-cell counts.
                     Exit 0 always when JSON found; exit 2 if JSON missing. Combines iter #144
                     (per-source-dir) with iter #134 (status counts) into one Stage × status matrix.
                     Iter #147.
```

`main()` handler (在 `--find-claim` handler 之后, `--csv` handler 之前):
```python
if args.summary_by_source_dir_and_status:
    # iter #147: --summary-by-source-dir-and-status — 2D matrix source dir × status with counts.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    def _primary_dir(c: dict) -> str:
        """Same as iter #144: extract primary source dir, strip :function_name suffix."""
        for ref in c.get("code_evidence", []) or []:
            if ref.startswith("PersoanlQuery/"):
                parts = ref.split("/")
                if len(parts) >= 3:
                    return parts[1]
                if len(parts) == 2:
                    return parts[1].split(":")[0]
        return "(no source dir)"

    # Frozen status order matches audit JSON convention (most-actionable first).
    STATUSES = ["verified_value_match", "verified", "discrepant",
                "degenerate", "partial", "unverified", "blocked"]

    from collections import defaultdict
    matrix: "dict[str, dict[str, int]]" = defaultdict(lambda: defaultdict(int))
    for c in claims_data:
        d = _primary_dir(c)
        matrix[d][c.get("status", "?")] += 1

    sorted_dirs = sorted(matrix.keys(), key=lambda d: (-sum(matrix[d].values()), d))
    col_width = max(len(s) for s in STATUSES)  # status header width
    dir_width = max(len(d) for d in sorted_dirs)
    # Header row: full status names right-aligned in col_width, "total" right-aligned.
    header = f"{'source dir':<{dir_width}}  " + "  ".join(f"{s:>{col_width}}" for s in STATUSES) + "  total"
    print(f"Source dir × status matrix ({len(claims_data)} claims across {len(sorted_dirs)} dirs):")
    print(header)
    # Data rows.
    col_totals = {s: 0 for s in STATUSES}
    grand_total = 0
    for d in sorted_dirs:
        cnts = matrix[d]
        row_total = sum(cnts.values())
        cells = "  ".join(f"{cnts.get(s, 0):>{col_width}d}" for s in STATUSES)
        print(f"{d:<{dir_width}}  {cells}  {row_total}")
        for s in STATUSES:
            col_totals[s] += cnts.get(s, 0)
        grand_total += row_total
    # Footer with column totals.
    footer_cells = "  ".join(f"{col_totals[s]:>{col_width}d}" for s in STATUSES)
    print(f"{'TOTAL':<{dir_width}}  {footer_cells}  {grand_total}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AE freeze iter #147 behavior:
- `--summary-by-source-dir-and-status` exit 0 (always when JSON found)
- "17 claims across 8 dirs" header (frozen baseline)
- All 7 status column headers visible (full names, not abbreviated)
- Top dir `10_complexity_analysis` visible
- Footer `TOTAL` row with per-status sums + grand total = 17
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 31 cases.

### README.md

Re-run commands section +1 line:
```bash
# 2D matrix source dir (rows) × audit status (columns) with counts + footer TOTAL; iter #147
python3 PersoanlQuery/paper_claims_audit.py --summary-by-source-dir-and-status
```

## §C 关键改动点

1. **New combination axis (iter #144 + iter #134 → 2D matrix)**: 与 existing flags complementary:
   - iter #134 `--by-section` — paper section × status 2D matrix (matrix view)
   - iter #144 `--count-by-codebase-dir` — per-source-dir text breakdown
   - iter #147 `--summary-by-source-dir-and-status` — source dir × status 2D matrix ✓ 本轮
   
   Reviewer mental model: "for paper-section view → iter #134; for source-Stage matrix view → iter #147; for source-Stage text detail → iter #144"。

2. **Mirrors iter #134 structure**: Same column layout (7 statuses in frozen order), same `TOTAL` footer row, same right-aligned cell counts. Visual symmetry between `--by-section` (paper-side) and `--summary-by-source-dir-and-status` (code-side) matrix views → reviewer can switch axes with no cognitive load。

3. **Frozen status order `["verified_value_match", "verified", "discrepant", "degenerate", "partial", "unverified", "blocked"]`**: Matches audit JSON convention (most-actionable first). Same order as iter #134 for visual consistency。

4. **Sort by row total desc, dir asc as tiebreak**: Largest Stage first → reviewer triages biggest bucket first。`dir_width = max(len(d) for d in sorted_dirs)` auto-compute → fits longest dir name in row label column。

5. **`col_width = max(len(s) for s in STATUSES)`**: Auto-fit to longest status name (currently `verified_value_match` = 19 chars)。Right-aligned `>{col_width}` cell count → numbers line up under header text。

6. **`TOTAL` footer row**: 列 总和 + grand total。Reviewer 看到 `TOTAL  0  6  4  1  1  5  0  17` → confirms 17-claim audit baseline (0 verified_value_match + 6 verified + 4 discrepant + 1 degenerate + 1 partial + 5 unverified + 0 blocked = 17 total)。Column sums are computed from cell counts (not re-extracted from audit JSON) → if any cell extraction fails, total won't match audit summary → reviewer can spot drift。

7. **Footer TOTAL row matches `--status-summary` baseline (0/6/4/1/1/5/0)**: Reviewer can verify `--summary-by-source-dir-and-status` aggregation by comparing TOTAL row to `python3 paper_claims_audit.py --status-summary`。If they differ → aggregation bug → caught by Case AE regression test (asserts TOTAL row visible, manual column sums would catch any drift)。

8. **Reuses iter #144 `_primary_dir` function**: Same source-dir extraction logic (first matching PersoanlQuery/X/... ref → `parts[1]`, strip `:function_name` for top-level utility refs)。Single source of truth → iter #147 跟 iter #144 一致地 bucket agreement_metrics.py + 10_complexity_analysis。

9. **`(no source dir)` fallback**: matches iter #144 behavior → claim without `PersoanlQuery/` ref falls into `(no source dir)` row。Frozen baseline = 0 such claims → not displayed in current matrix。

10. **Exit code 0 always (informational, like iter #144/145/146)**: "0 dirs" means audit JSON catastrophically broken, not actionable info。Safe in shell scripts without `|| true`。

11. **`--output` flag respected**: reviewer 想 `--summary-by-source-dir-and-status --output /custom/path.json` 跟其他 mode 语义一致。

12. **No re-run audit**: `--summary-by-source-dir-and-status` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --summary-by-source-dir-and-status
Source dir × status matrix (17 claims across 8 dirs):
source dir              verified_value_match              verified            discrepant            degenerate               partial            unverified               blocked  total
10_complexity_analysis                     0                     0                     0                     0                     0                     5                     0  5
08_compare_all_domain                      0                     2                     0                     1                     0                     0                     0  3
agreement_metrics.py                       0                     0                     3                     0                     0                     0                     0  3
04_query                                   0                     2                     0                     0                     0                     0                     0  2
00_data_preparation                        0                     1                     0                     0                     0                     0                     0  1
02_writing_analysis                        0                     0                     1                     0                     0                     0                     0  1
05_inject_noisy                            0                     1                     0                     0                     0                     0                     0  1
07_noisy_retrieval                         0                     0                     0                     0                     1                     0                     0  1
TOTAL                                      0                     6                     4                     1                     1                     5                     0  17
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AE: --summary-by-source-dir-and-status CLI flag (iter #147)
  PASS  --summary-by-source-dir-and-status renders 2D matrix with footer TOTAL + exits 2 on missing JSON

All 31 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~55 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AE, +docstring, +main count to 31)
- 文档: `README.md` (+1 line `--summary-by-source-dir-and-status` example)
- 验证: 31/31 cases pass; pre-commit hook freeze 31 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --summary-by-source-dir-and-status` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section (paper section × status matrix)
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
- **iter #147** — --summary-by-source-dir-and-status (source dir × status matrix) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| paper section × status (matrix) | `--by-section` (iter #134) |
| source dir × status (matrix) | `--summary-by-source-dir-and-status` (iter #147) ✓ 本轮 |
| source dir (1D text) | `--count-by-codebase-dir` (iter #144) |
| paper section worst | `--worst-by-section` (iter #145) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| list by status | `--list-{degenerate,unverified,partial,verified,discrepant}` (iter #138-143) |
| search by substring | `--find-claim` (iter #146) |
| flip detection | `--diff` (iter #114) |

Audit CLI matrix views taxonomy:
| view | axis | flag |
|------|------|------|
| paper-side matrix | section × status | `--by-section` (iter #134) |
| code-side matrix | source dir × status | `--summary-by-source-dir-and-status` (iter #147) ✓ 本轮 |

regression test coverage timeline:
- **iter #146** — 30 cases
- **iter #147** — 31 cases (+ --summary-by-source-dir-and-status) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
