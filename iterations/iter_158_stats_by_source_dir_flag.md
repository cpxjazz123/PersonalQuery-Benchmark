# Iteration #158 — Audit CLI `--stats-by-source-dir` (per-source-dir numeric statistics, mirror iter #157)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--stats-by-source-dir` argparse flag (per-source-dir max/mean abs_delta + max/mean rel_delta_pct + claim count + vc count, sorted by max abs desc); extend `_smoke_audit_regression.py` to 42 cases (Case AP)
**prior**: iter #157 (`--stats-by-section`) gives per-paper-section numeric stats,iter #155 (`--audit-stats`) gives GLOBAL numeric stats,iter #149 (`--worst-by-source-dir`) gives worst-claim-per-source-dir (single max abs only),iter #147 (`--summary-by-source-dir-and-status`) gives per-source-dir × status COUNT matrix → reviewer 想要 per-source-dir numeric aggregate "is agreement_metrics.py or 02_writing_analysis worse by how much?" — code-side erratum triage complement to iter #157 paper-section view.

## §A 审稿意见

Iter #157 `--stats-by-section` outputs per-section aggregate: §3.3 + Table 2 (4 claims, 6 vcs, max=0.7415 mean=0.2158). Reviewer 想要 parallel CODE-SIDE decomposition "for `02_writing_analysis` dir (where the LLM_Full_Set script lives): 1 claim with 3 subclaim vcs max=0.7415 mean=0.2783; for `agreement_metrics.py` (where Fleiss/Spearman/MAE live): 3 claims with 3 vcs max=0.2700 mean=0.1534".当前 must:
1. iter #149 `--worst-by-source-dir` — single max abs per dir, no mean or vc count
2. iter #147 `--summary-by-source-dir-and-status` — per-dir × status COUNT matrix, no numeric
3. iter #157 `--stats-by-section` — paper-section view, doesn't break down by source dir
4. Manual: `jq -r '.claims[] | .code_evidence[0]? // "(none)" | split(":")[0] | ... '` + manual arithmetic

后果: code-side erratum meetings "should we fix Stage 02 first or agreement_metrics.py first?" requires manual jq + awk — 5+ min per query; Stage 02 vs Stage 08 impact unranked.

iter #158 fix: 新增 `--stats-by-source-dir` argparse flag — mirror iter #157 for source-dir axis:
- Per-source-dir: claim count + vc count (with abs) + max/mean abs_delta + max/mean rel_delta_pct
- Source-dir extraction: iter #149 `_primary_dir()` helper (first `PersoanlQuery/X/...` ref, strip `:function_name` suffix)
- Sorted by max abs desc (dirs with no abs sink to bottom, alphabetical within)
- Frozen baseline: 8 source dirs, 17 claims, 6 vcs with abs
  - `02_writing_analysis`: 1 claim (RQ3_LLM_Full_Set_94% with 3 subclaim vcs), max=0.7415 mean=0.2783 max rel=78.38%
  - `agreement_metrics.py`: 3 claims (Fleiss + Spearman + MAE, each with 1 vc), max=0.2700 mean=0.1534 max rel=30.34%
  - 6 other dirs: 13 claims, 0 vcs (verified/unverified/degenerate/partial/blocked = no numeric delta)

## §B 本轮 (iter #158) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--stats-by-source-dir` flag (after `--stats-by-section`, before `--csv`):

```python
parser.add_argument("--stats-by-source-dir", action="store_true",
                    help="(iter #158) Per-source-dir numeric statistics: max/mean abs_delta + "
                         "max/mean rel_delta_pct + vc count + claim count. Source dirs sorted by "
                         "max abs desc. Mirror iter #157 for source-dir axis (uses iter #149 "
                         "_primary_dir extraction). Exit 0 always when JSON found; exit 2 if JSON "
                         "missing.")
```

docstring 加 flag description:
```
--stats-by-source-dir
                     Per-source-dir numeric statistics: max/mean abs_delta + max/mean
                     rel_delta_pct + count of vcs. Source dirs sorted by max abs desc.
                     Mirror iter #157 for source-dir axis (uses iter #149 _primary_dir
                     extraction). Exit 0 always when JSON found; exit 2 if JSON missing.
                     Iter #158.
```

`main()` handler (在 `--stats-by-section` handler 之后, `--csv` handler 之前):
```python
if args.stats_by_source_dir:
    # iter #158: --stats-by-source-dir — per-source-dir numeric statistics (mirror iter #157).
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    # Reuse iter #149 _primary_dir extraction: first `PersoanlQuery/X/...` ref, strip
    # `:function_name` suffix. Top-level utility files bucket together.
    def _primary_dir(c: dict) -> str:
        evidences = c.get("code_evidence") or []
        for ref in evidences:
            parts = ref.split("/")
            if len(parts) >= 2 and parts[0] in ("PersoanlQuery", "personelquery"):
                d = parts[1].split(":")[0]
                return d if d else "(top-level)"
        return "(no source dir)"

    def _safe_abs(vc):
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else None

    def _safe_rel(vc):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    # Bucket by source dir (key = primary dir, value = {claim_count, abs[], rel[]}).
    per_dir: dict = {}
    for c in claims_data:
        d = _primary_dir(c)
        bucket = per_dir.setdefault(d, {"claims": 0, "abs": [], "rel": []})
        bucket["claims"] += 1
        for vc in c.get("value_check_results", []):
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                bucket["abs"].append(ad)
            if rd is not None:
                bucket["rel"].append(rd)

    rows = []
    for d, b in per_dir.items():
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
        rows.append((d, b["claims"], n_abs, n_rel, max_abs, mean_abs, max_rel, mean_rel))
    # Sort: max_abs desc, then alphabetical by dir name.
    rows.sort(key=lambda r: (-r[4], r[0]))

    n_dir = len(rows)
    n_total_vc_abs = sum(r[2] for r in rows)
    print(f"Stats by source dir ({n_dir} dirs, {sum(r[1] for r in rows)} claims, {n_total_vc_abs} vcs with abs):")
    print(f"  {'source dir':<32}  {'claims':>6}  {'#vcs':>4}  {'max abs':>9}  {'mean abs':>9}  {'max rel':>9}  {'mean rel':>9}")
    for d, n_c, n_a, n_r, mx_a, mn_a, mx_r, mn_r in rows:
        d_disp = d if len(d) <= 32 else d[:29] + "..."
        mx_a_s = f"{mx_a:.4f}" if n_a else "—"
        mn_a_s = f"{mn_a:.4f}" if n_a else "—"
        mx_r_s = f"{mx_r * 100:.2f}%" if n_r else "—"
        mn_r_s = f"{mn_r * 100:.2f}%" if n_r else "—"
        print(f"  {d_disp:<32}  {n_c:>6}  {n_a:>4}  {mx_a_s:>9}  {mn_a_s:>9}  {mx_r_s:>9}  {mn_r_s:>9}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AP freeze iter #158 behavior:
- `--stats-by-source-dir` exit 0 (always when JSON found)
- "Stats by source dir" + "8 dirs" + "17 claims" + "6 vcs with abs" header (frozen baseline)
- `02_writing_analysis` row with `0.7415` (max abs) + `78.38%` (max rel)
- `agreement_metrics.py` row with `0.2700` (max abs for MAE)
- Em-dash `—` for no-abs dirs (6 other dirs)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 42 cases.

### README.md

Re-run commands section +1 line:
```bash
# Per-source-dir numeric statistics (mirror iter #157 for source-dir axis, uses iter #149 _primary_dir); iter #158
python3 PersoanlQuery/paper_claims_audit.py --stats-by-source-dir
```

## §C 关键改动点

1. **Mirror iter #157 for source-dir axis**: same `_safe_abs` / `_safe_rel` accessors + same max/mean abs + max/mean rel format → mental model "for paper-section aggregate → iter #157; for source-dir aggregate → iter #158". Reviewer can spot which CODE dir is responsible for the aggregate max.

2. **Reuse iter #149 `_primary_dir()` helper**: extracts first `PersoanlQuery/X/...` ref, strips `:function_name` suffix → top-level utility files (e.g. `agreement_metrics.py`) bucket together rather than splitting into one bucket per function. Same logic as iter #149 `--worst-by-source-dir` and iter #144 `--count-by-codebase-dir` → consistent source-dir labeling across all audit views。

3. **Single-level dir (just stage number)**: `02_writing_analysis` not `PersoanlQuery/02_writing_analysis/...` → matches iter #149 / #144 / #147 mental model (compact stage labels)。

4. **Sort by max abs desc**: `02_writing_analysis` (max=0.7415) sinks to TOP; `agreement_metrics.py` (max=0.2700) #2; 6 other dirs (no abs) sink to BOTTOM alphabetically. Reviewer sees worst dir first → matches iter #149 `--worst-by-source-dir` mental model ("worst dir first").

5. **`—` em-dash for no-abs dirs**: matches iter #157 / iter #156 convention → no false `0.0000` implying "we computed mean of zero".

6. **Dir name truncation at 32 chars**: `d_disp = d if len(d) <= 32 else d[:29] + "..."` — handles long dir names like `PersoanlQuery/10_complexity_analysis` (already short at 22 chars; longer would truncate).

7. **`#vcs` column header**: explicit "#vcs" (not ambiguous "vcs") to clarify "count of vcs WITH abs_delta". Matches iter #157 / iter #155 per-status style.

8. **Column alignment**: f-string `>9` width for numeric columns → max abs and mean abs columns align across rows. Reviewer can scan column-wise to spot highest max abs.

9. **Frozen baseline shows source-dir distribution**: 
   - `02_writing_analysis`: 1 claim (RQ3_LLM_Full_Set_94% with 3 subclaim vcs: plausibility 0.0737 + structure 0.0197 + preservation 0.7415 = mean 0.2783) → THE dir with preservation worst subclaim (max=0.7415)
   - `agreement_metrics.py`: 3 claims (Fleiss 0.0965 + Spearman 0.0936 + MAE 0.2700 = mean 0.1534) → THE dir with 3 RQ3 metric claims
   - 6 other dirs (00_data_preparation + 04_query + 05_inject_noisy + 07_noisy_retrieval + 08_compare_all_domain + 10_complexity_analysis): 13 claims, 0 vcs (verified/unverified/degenerate/partial/blocked)
   - Total: 8 dirs, 17 claims, 6 vcs with abs ✓

10. **Single-level dir granularity**: reviewer sees "Stage 02" vs "Stage 08" vs "agreement_metrics.py" without diving into individual scripts. Code-side erratum triage: "fix Stage 02 first" or "fix agreement_metrics.py first" → reviewer makes call at dir level。

11. **Defensive `_safe_abs` / `_safe_rel` accessors**: identical to iter #157 / #155 — return `None` (not 0.0) when missing → degenerate vc (RQ1_Table1_Hit10 abs_delta=None) correctly excluded from per-dir abs aggregate.

12. **`(no source dir)` fallback**: claims without `PersoanlQuery/X/...` evidence refs default to `(no source dir)` — defensive against future claims added without code_evidence field. Frozen baseline all 17 claims have code_evidence so no row shows `(no source dir)`.

13. **8 dirs vs 10 dirs (Stage-level) discrepancy**: iter #158 uses single-level (just `02_writing_analysis`) not Stage-level (`PersoanlQuery/02_writing_analysis`). Frozen baseline: 8 dirs at single-level (00, 02, 04, 05, 07, 08, 10 + agreement_metrics.py) vs 10 at Stage-level (8 dirs + 2 different groupings). This matches iter #149 / #144 / #147 behavior → consistency across audit views.

14. **Sum `{n_dir} dirs` + `{sum(r[1] for r in rows)} claims` + `{n_total_vc_abs} vcs with abs` in header**: triple-counter annotation → reviewer sees both claim count and vc count (which can diverge if a claim has multiple vcs).

15. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s).

16. **Exit code 0 always (informational, like iter #144-157)**: 与前 14 iter 一致。Safe in shell scripts without `|| true`。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --stats-by-source-dir
Stats by source dir (8 dirs, 17 claims, 6 vcs with abs):
  source dir                        claims  #vcs    max abs   mean abs    max rel   mean rel
  02_writing_analysis                    1     3     0.7415     0.2783     78.38%     29.34%
  agreement_metrics.py                   3     3     0.2700     0.1534     30.34%     18.43%
  00_data_preparation                    1     0          —          —          —          —
  04_query                               2     0          —          —          —          —
  05_inject_noisy                        1     0          —          —          —          —
  07_noisy_retrieval                     1     0          —          —          —          —
  08_compare_all_domain                  3     0          —          —          —          —
  10_complexity_analysis                 5     0          —          —          —          —
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AP: --stats-by-source-dir CLI flag (iter #158)
  PASS  --stats-by-source-dir shows 8 dirs with 02_writing_analysis (max abs=0.7415) + agreement_metrics.py (max abs=0.2700) + em-dash for no-abs dirs + exits 2 on missing JSON

All 42 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~80 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AP, +docstring, +main count to 42)
- 文档: `README.md` (+1 line `--stats-by-source-dir` example)
- 验证: 42/42 cases pass; pre-commit hook freeze 42 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --stats-by-source-dir` (~100ms, 不 re-run audit)

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
- **iter #157** — --stats-by-section
- **iter #158** — --stats-by-source-dir ✓ 本轮

Audit CLI per-source-dir axes taxonomy:
| view | axis | flag |
|------|------|------|
| count by source dir × status | source dir × status matrix | `--summary-by-source-dir-and-status` (iter #147) |
| worst-claim-per-source-dir | max abs per dir | `--worst-by-source-dir` (iter #149) |
| **stats per source dir** | **max/mean abs + max/mean rel + vc count per dir** | **`--stats-by-source-dir` (iter #158) ✓ 本轮** |

regression test coverage timeline:
- **iter #157** — 41 cases
- **iter #158** — 42 cases (+ --stats-by-source-dir) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample