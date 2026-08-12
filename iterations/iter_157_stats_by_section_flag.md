# Iteration #157 — Audit CLI `--stats-by-section` (per-paper-section numeric statistics)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--stats-by-section` argparse flag (per-section max/mean abs_delta + max/mean rel_delta_pct + claim count + vc count, sorted by max abs desc); extend `_smoke_audit_regression.py` to 41 cases (Case AO)
**prior**: iter #155 (`--audit-stats`) gives GLOBAL numeric aggregate (max abs=0.7415 across all 17 claims),iter #148 (`--summary-by-section-and-status`) gives per-section × status COUNT matrix (no numeric),iter #145 (`--worst-by-section`) gives worst-claim-per-section (single abs_delta per section) → reviewer 想要 per-section numeric aggregate "which paper section has the worst aggregate drift? §3.3 with 6 vcs mean=0.2158 max=0.7415 or §2.2 with 7 claims all unverified (no abs)?"

## §A 审稿意见

Iter #155 `--audit-stats` 输出 GLOBAL max abs=0.7415 + mean=0.2158,across 6 vcs all in §3.3 + Table 2,reviewer 想要 per-section decomposition "for §3.3 specifically: max=0.7415 mean=0.2158, for §2.2: max=— mean=— (no vcs)".当前 must:
1. iter #155 `--audit-stats` — global aggregate, doesn't break down by section
2. iter #145 `--worst-by-section` — single max abs per section (just `max` not `mean`)
3. iter #143 `--list-discrepant` — list all discrepant with per-row abs + section, then mental arithmetic to get per-section aggregate
4. Manual: `jq -r '.claims[] | .section as $s | .value_check_results[]? | .abs_delta // empty | "\($s) \(.)"' | awk '{sum[$1]+=$2; cnt[$1]++; if($2>max[$1])max[$1]=$2} END {for(s in sum) print s, max[s], sum[s]/cnt[s]}'`

后果: paper-erratum meetings "is §3.3 worse than §2.2 by how much?" requires manual jq + awk — 5+ min per query; §2.2 with 7 unverified claims is invisible in iter #155 output (gets averaged into nothing).

iter #157 fix: 新增 `--stats-by-section` argparse flag — 不 re-run audit,直接 read most recent audit JSON,aggregate per paper section (key = `c["section"]`):
- Per-section: claim count + vc count (with abs) + max/mean abs_delta + max/mean rel_delta_pct
- Sorted by max abs desc (sections with no abs sink to bottom, alphabetical within)
- Frozen baseline: 8 sections, 17 claims, 6 vcs with abs
- §3.3 + Table 2: 4 claims, 6 vcs, max=0.7415 mean=0.2158 max rel=78.38% mean=23.88%
- All other 7 sections: 13 claims, 0 vcs (all verified/degenerate/partial/unverified/blocked = no numeric delta)

## §B 本轮 (iter #157) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--stats-by-section` flag (after `--audit-stats`, before `--csv`):

```python
parser.add_argument("--stats-by-section", action="store_true",
                    help="(iter #157) Per-paper-section numeric statistics: max/mean abs_delta + "
                         "max/mean rel_delta_pct + vc count + claim count. Sections sorted by max "
                         "abs desc. Extends iter #155 numeric-aggregate axis from global to "
                         "per-section. Exit 0 always when JSON found; exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--stats-by-section
                     Per-paper-section numeric statistics: max/mean abs_delta + max/mean
                     rel_delta_pct + vc count + claim count. Sections sorted by max abs desc.
                     Extends iter #155 numeric-aggregate axis from global to per-section.
                     Iter #157.
```

`main()` handler (在 `--audit-stats` handler 之后, `--csv` handler 之前):
```python
if args.stats_by_section:
    # iter #157: --stats-by-section — per-paper-section numeric statistics.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    def _safe_abs(vc):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else None

    def _safe_rel(vc):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    # Bucket by paper section (key = section, value = {claim_count, abs[], rel[]}).
    per_sec: dict = {}
    for c in claims_data:
        sec = c.get("section", "(unspecified)")
        bucket = per_sec.setdefault(sec, {"claims": 0, "abs": [], "rel": []})
        bucket["claims"] += 1
        for vc in c.get("value_check_results", []):
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                bucket["abs"].append(ad)
            if rd is not None:
                bucket["rel"].append(rd)

    # Build summary rows; sort by max abs desc (sections with no abs sink to bottom).
    rows = []
    for sec, b in per_sec.items():
        n_abs = len(b["abs"])
        n_rel = len(b["rel"])
        if n_abs:
            max_abs = max(b["abs"])
            mean_abs = sum(b["abs"]) / n_abs
        else:
            max_abs = mean_abs = 0.0
        if n_rel:
            max_rel = max(b["rel"])
            mean_rel = sum(b["rel"]) / n_rel
        else:
            max_rel = mean_rel = 0.0
        rows.append((sec, b["claims"], n_abs, n_rel, max_abs, mean_abs, max_rel, mean_rel))
    # Sort: max_abs desc (no-abs sections sink to bottom, preserve alphabetical order within).
    rows.sort(key=lambda r: (-r[4], r[0]))

    n_sec = len(rows)
    n_total_vc_abs = sum(r[2] for r in rows)  # r[2] = n_abs
    n_total_vc_rel = sum(r[3] for r in rows)  # r[3] = n_rel
    print(f"Stats by section ({n_sec} sections, {sum(r[1] for r in rows)} claims, {n_total_vc_abs} vcs with abs):")
    print(f"  {'section':<30}  {'claims':>6}  {'#vcs':>4}  {'max abs':>9}  {'mean abs':>9}  {'max rel':>9}  {'mean rel':>9}")
    for sec, n_c, n_a, n_r, mx_a, mn_a, mx_r, mn_r in rows:
        sec_disp = sec if len(sec) <= 30 else sec[:27] + "..."
        mx_a_s = f"{mx_a:.4f}" if n_a else "—"
        mn_a_s = f"{mn_a:.4f}" if n_a else "—"
        mx_r_s = f"{mx_r * 100:.2f}%" if n_r else "—"
        mn_r_s = f"{mn_r * 100:.2f}%" if n_r else "—"
        print(f"  {sec_disp:<30}  {n_c:>6}  {n_a:>4}  {mx_a_s:>9}  {mn_a_s:>9}  {mx_r_s:>9}  {mn_r_s:>9}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AO freeze iter #157 behavior:
- `--stats-by-section` exit 0 (always when JSON found)
- "Stats by section" + "8 sections" + "17 claims" + "6 vcs with abs" header (frozen baseline)
- §3.3 + Table 2 row with `0.7415` (max abs) + `78.38%` (max rel)
- Em-dash `—` for no-abs sections (7 other sections)
- §2.2 row visible (7 claims, 0 vcs)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 41 cases.

### README.md

Re-run commands section +1 line:
```bash
# Per-paper-section numeric statistics (extends iter #155 from global to per-section); iter #157
python3 PersoanlQuery/paper_claims_audit.py --stats-by-section
```

## §C 关键改动点

1. **Extends iter #155 axis from global to per-section**: same `_safe_abs` / `_safe_rel` accessors + same max/mean abs + max/mean rel format → mental model "for global aggregate → iter #155; for per-section → iter #157". Reviewer can spot which paper section is responsible for the aggregate max.

2. **Section key**: `c.get("section", "(unspecified)")` — same default fallback as iter #117 per-section breakdown table in dashboard → consistent section labeling。

3. **Sort by max abs desc**: §3.3 + Table 2 (with abs) sinks to TOP (highest max=0.7415); other 7 sections (no abs) sink to BOTTOM alphabetically. Reviewer sees worst section first → matches iter #145 `--worst-by-section` mental model ("worst section first").

4. **`—` em-dash for no-abs sections**: matches iter #156 dashboard audit-stats panel convention → no false `0.0000` implying "we computed mean of zero". Reviewer sees explicit "no numeric data here".

5. **Section name truncation at 30 chars**: `sec_disp = sec if len(sec) <= 30 else sec[:27] + "..."` — handles long section names like "§3.2 + Table 1 (lower panel)" (29 chars, fits; longer would truncate). 

6. **`#vcs` column header**: explicit "#vcs" (not ambiguous "vcs") to clarify "count of vcs WITH abs_delta". Matches iter #155 per-status `6 vcs` style.

7. **Column alignment**: f-string `>9` width for numeric columns → max abs and mean abs columns align across rows. Reviewer can scan column-wise to spot highest max abs.

8. **Frozen baseline shows section-distribution**: §3.3 + Table 2 = 4 claims with 6 vcs (mean=0.2158 max=0.7415) → that's THE only section with numeric drift; all other 7 sections have 0 vcs. Future iter #87/88 re-runs that flip unverified→verified will populate §2.2 rows with abs values, changing the table.

9. **13 of 17 claims have 0 vcs**: §2.2 (7 claims, all verified/unverified no abs) + §2.1 (1 verified) + §3.2 (1 degenerate) + §3.2+Table1 (1 partial) + §3.2+Table1 lower panel (1 unverified) + §3.4+Table3 (1 unverified) + (infrastructure) (1 verified) = 13 claims 0 vcs + §3.3+Table2 (4 discrepant) 6 vcs = 17 total ✓。

10. **Defensive `_safe_abs` / `_safe_rel` accessors**: identical to iter #155 — return `None` (not 0.0) when missing → degenerate vc (RQ1_Table1_Hit10 abs_delta=None) correctly excluded from per-section abs aggregate.

11. **Section name `(unspecified)` fallback**: claims without `section` field default to `(unspecified)` — defensive against future claims added without section field. Frozen baseline all claims have section so no row shows `(unspecified)`.

12. **Sum `{n_sec} sections` + `{sum(r[1] for r in rows)} claims` + `{n_total_vc_abs} vcs with abs` in header**: triple-counter annotation → reviewer sees both claim count and vc count (which can diverge if a claim has multiple vcs).

13. **8 sections alphabetical sort within "no abs" group**: when max_abs is 0 (no abs), the secondary sort key is alphabetical by section name → deterministic output regardless of claim ordering in JSON.

14. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s).

15. **Exit code 0 always (informational, like iter #144-156)**: 与前 13 iter 一致。Safe in shell scripts without `|| true`。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --stats-by-section
Stats by section (8 sections, 17 claims, 6 vcs with abs):
  section                         claims  #vcs    max abs   mean abs    max rel   mean rel
  §3.3 + Table 2                       4     6     0.7415     0.2158     78.38%     23.88%
  (infrastructure)                     1     0          —          —          —          —
  §2.1                                 1     0          —          —          —          —
  §2.2                                 7     0          —          —          —          —
  §3.2                                 1     0          —          —          —          —
  §3.2 + Table 1                       1     0          —          —          —          —
  §3.2 + Table 1 (lower panel)         1     0          —          —          —          —
  §3.4 + Table 3                       1     0          —          —          —          —
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AO: --stats-by-section CLI flag (iter #157)
  PASS  --stats-by-section shows 8 sections with §3.3 row max abs=0.7415 + 78.38% + em-dash for no-abs sections + exits 2 on missing JSON

All 41 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~75 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AO, +docstring, +main count to 41)
- 文档: `README.md` (+1 line `--stats-by-section` example)
- 验证: 41/41 cases pass; pre-commit hook freeze 41 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --stats-by-section` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
- **iter #135** — --audit-age
- **iter #137** — --md-table
- **iter #138-143** — --list-{degenerate,unverified,partial,verified,discrepant}
- **iter #144** — --count-by-codebase-dir
- **iter #145** — --worst-by-section
- **iter #146** — --find-claim PAT
- **iter #147** — --summary-by-source-dir-and-status
- **iter #148** — --summary-by-section-and-status
- **iter #149** — --worst-by-source-dir
- **iter #150** — --severity-tier
- **iter #151** — --claim-by-id-prefix
- **iter #152** — --evidence-coverage
- **iter #155** — --audit-stats
- **iter #157** — --stats-by-section ✓ 本轮

Audit CLI per-section axes taxonomy (filling the per-section x-axis):
| view | axis | flag |
|------|------|------|
| count by status | 7 audit statuses | `--status-summary` (iter #131) |
| count by section × status | section × status matrix | `--summary-by-section-and-status` (iter #148) |
| worst-claim-per-section | max abs per section | `--worst-by-section` (iter #145) |
| **stats per section** | **max/mean abs + max/mean rel + vc count per section** | **`--stats-by-section` (iter #157) ✓ 本轮** |

regression test coverage timeline:
- **iter #156** — 40 cases
- **iter #157** — 41 cases (+ --stats-by-section) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample