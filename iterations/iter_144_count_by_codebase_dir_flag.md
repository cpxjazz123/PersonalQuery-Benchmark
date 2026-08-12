# Iteration #144 — Audit CLI `--count-by-codebase-dir` flag (identify coverage-gap stages)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--count-by-codebase-dir` argparse flag (aggregates audit claims by source directory with per-status breakdown, sorted by count desc); extend `_smoke_audit_regression.py` to 28 cases (Case AB)
**prior**: Per-status list set is 5/5 complete (degenerate/unverified/partial/verified/discrepant). Existing flags cover per-claim / per-status / per-section axes,但缺 **per-source-dir aggregation** — reviewer 想 identify "which pipeline Stage has the most coverage gaps" must `jq '.claims[] | .code_evidence[0]'` + manual bucket. iter #87/#88 planning (Stage 12 + Stage 6/9 re-run) needs explicit Stage-by-Stage coverage map to confirm backlog ordering.

## §A 审稿意见

Reviewer / co-author 想要 quick "which Stage has the most audit issues?" 当前 must:
1. `jq -r '.claims[] | .code_evidence[]' | xargs -I{} dirname {} | sort | uniq -c | sort -rn` — manual script
2. Open dashboard HTML + scroll per-claim `provenance panels` (iter #112) + mental bucket
3. iter #134 `--by-section` (paper section × status) — orthogonal axis, doesn't help identify Stage

后果: audit infra 不能 tell reviewer "Stage 10 has 5 unverified, Stage 2 has 4 discrepant" in one shell call。Backlog prioritization (iter #87 → Stage 10 + Stage 12; iter #88 → Stage 6/9) requires this exact aggregation manually。

iter #144 fix: 新增 `--count-by-codebase-dir` CLI flag — 不 re-run audit,直接 read most recent audit JSON,group claims by primary source directory (extracted from `code_evidence[0]` PersoanlQuery/* path),aggregate per-status counts + claim IDs,sort by count desc (largest bucket first → reviewer sees worst-covered Stage first)。Top-level utility files (e.g. `PersoanlQuery/agreement_metrics.py`) get bucketed together via strip `':function_name'` suffix (iter #112 file:function reference format)。

## §B 本轮 (iter #144) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--count-by-codebase-dir` flag:
```python
parser.add_argument("--count-by-codebase-dir", action="store_true",
                    help="(iter #144) Group audit claims by source directory (e.g. 10_complexity_analysis) "
                         "with per-status breakdown. Exit 0 always when JSON found; exit 2 if JSON missing. "
                         "Useful for identifying which pipeline Stage has the most coverage gaps.")
```

docstring 加 flag description:
```
--count-by-codebase-dir Group audit claims by source directory (e.g. 10_complexity_analysis) with
                         per-status breakdown. Exit 0 always when JSON found; exit 2 if JSON missing.
                         Useful for identifying which pipeline Stage has the most coverage gaps. Iter #144.
```

`main()` handler (在 `--list-discrepant` handler 之后, `--csv` handler 之前):
```python
if args.count_by_codebase_dir:
    # iter #144: --count-by-codebase-dir — aggregate audit claims by source dir with per-status breakdown.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    def _primary_dir(c: dict) -> str:
        """Extract the primary source directory for a claim from its code_evidence refs.
        Fall back to '(no source dir)' if no PersoanlQuery/ ref present. Strip
        ':function_name' suffix from iter #112 file:function references so all
        refs to a top-level utility (e.g. PersoanlQuery/agreement_metrics.py) bucket together."""
        for ref in c.get("code_evidence", []) or []:
            if ref.startswith("PersoanlQuery/"):
                parts = ref.split("/")
                if len(parts) >= 3:
                    return parts[1]  # e.g. '10_complexity_analysis'
                if len(parts) == 2:
                    # Top-level file like PersoanlQuery/agreement_metrics.py — strip :function_name
                    return parts[1].split(":")[0]
        return "(no source dir)"

    from collections import defaultdict
    dir_status: "dict[str, dict[str, int]]" = defaultdict(lambda: defaultdict(int))
    dir_claims: "dict[str, list[str]]" = defaultdict(list)
    for c in claims_data:
        d = _primary_dir(c)
        dir_status[d][c.get("status", "?")] += 1
        dir_claims[d].append(c.get("id", "?"))

    sorted_dirs = sorted(dir_claims.keys(), key=lambda d: (-len(dir_claims[d]), d))
    print(f"Claims by source directory ({len(claims_data)} claims across {len(sorted_dirs)} dirs):")
    for d in sorted_dirs:
        n = len(dir_claims[d])
        status_cnts = dir_status[d]
        breakdown = " / ".join(
            f"{s}={status_cnts[s]}" for s in sorted(status_cnts.keys(), key=lambda x: (-status_cnts[x], x))
        )
        ids = ", ".join(dir_claims[d][:5])
        if len(dir_claims[d]) > 5:
            ids += f", ... (+{len(dir_claims[d]) - 5} more)"
        print(f"  {n:3d}  {d:35s}  {breakdown}")
        print(f"        ids: {ids}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AB freeze iter #144 behavior:
- `--count-by-codebase-dir` exit 0 (always when JSON found)
- "17 claims across 8 dirs" header (frozen baseline)
- Top dir `10_complexity_analysis` (count=5, status `unverified=5`)
- `agreement_metrics.py` bucket present (stripped :function_name suffix)
- `RQ3_Fleiss_Kappa_0.72` claim ID visible in ids list
- Sort order: count desc, dir asc — first row is `10_complexity_analysis` (count=5)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 28 cases.

### README.md

Re-run commands section +1 line:
```bash
# Group audit claims by source directory with per-status breakdown (identify coverage-gap stages); iter #144
python3 PersoanlQuery/paper_claims_audit.py --count-by-codebase-dir
```

## §C 关键改动点

1. **New aggregation axis (per-source-dir)**: 与 existing flags orthogonal:
   - `--by-section` (iter #134) — paper section × status matrix
   - `--list-*` (iter #138-143) — per-status claim listing
   - `--count-by-codebase-dir` (iter #144) — source directory × status ✓ 本轮
   
   Source directory 是 git-tracked code location, mapping directly to pipeline Stage。Reviewer sees "Stage 10 has 5 unverified → iter #87 scope confirmed" 一眼。

2. **Primary source directory extraction (not all refs)**: A claim may reference multiple files (e.g. RQ3_Fleiss_Kappa_0.72 references both `PersoanlQuery/agreement_metrics.py:fleiss_kappa` and `PersoanlQuery/02_writing_analysis/...`). For Stage-level aggregation, we use first matching `PersoanlQuery/X/...` reference (`parts[1]`) → maps to "primary Stage that owns this claim's computation logic"。Reviewer mental model: "this claim is primarily about Stage X's output"。

3. **`agreement_metrics.py` top-level bucket (strip `:function_name`)**: iter #112 introduced `PersoanlQuery/agreement_metrics.py:fleiss_kappa` format for cross-stage utility references。Without stripping the `:function_name` suffix, each function would create a separate bucket (5 functions = 5 rows). Strip → 1 row `agreement_metrics.py` containing all 3 RQ3_* discrepant claims (Fleiss_Kappa_0.72 + Spearman_0.81 + MAE_0.89) + others。

4. **`(no source dir)` fallback**: For claims with no `PersoanlQuery/` ref (e.g. paper-only claims), fall back to `(no source dir)` literal bucket。Frozen baseline has 0 such claims → bucket empty → not displayed。

5. **Sort by count desc, dir asc**: Worst-covered stage first (largest bucket = most claims to triage)。Tiebreak by dir asc → stable across re-runs (e.g. `08_compare_all_domain` before `agreement_metrics.py` for both count=3, alphabetically earlier first)。

6. **Status breakdown format `verified=2 / degenerate=1`**: Within each dir, show per-status counts as `status=N` joined with `" / "`. Sort by count desc → most frequent status first。Reviewer 一眼 see "08_compare_all_domain has 2 verified + 1 degenerate = degenerate needs investigation"。

7. **First-5 IDs inline + `+N more` overflow**: 显示 each dir's claim IDs inline (truncated to 5 + overflow count) → reviewer 不必 click into dashboard to see which IDs are in each bucket。Matches iter #134 `--by-section` cell format。

8. **`from collections import defaultdict` lazy import**: Avoid import overhead for flags that don't need it。Existing audit script uses this pattern sparingly。

9. **Exit code 0 always (informational, like iter #142/143)**: 与 iter #142 (`--list-verified`) + iter #143 (`--list-discrepant`) 一致 — "0 claims across 0 dirs" means audit JSON catastrophically broken, not actionable info。Safe in shell scripts without `|| true`。

10. **`--output` flag respected**: reviewer 想 `--count-by-codebase-dir --output /custom/path.json` 跟其他 mode 语义一致。

11. **Reuses iter #112 `code_evidence` field**: 不需 re-implement — iter #112 audit schema 加 `code_evidence` list with `PersoanlQuery/...` references;iter #144 是 shell wrapper reading same field but grouping by dir not per-claim。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --count-by-codebase-dir
Claims by source directory (17 claims across 8 dirs):
    5  10_complexity_analysis               unverified=5
        ids: RQ4_GMM_Best_Prior, Sec2_20dim_syntactic_features, Sec2_K8_clusters_BIC_AIC_silhouette, Sec2_K2_GMM_per_user, Sec2_95pct_holdout_threshold
    3  08_compare_all_domain                verified=2 / degenerate=1
        ids: RQ1_Table1_Hit10, RQ1_Delta_Range, RQ2_Table1_Drop
    3  agreement_metrics.py                 discrepant=3
        ids: RQ3_Fleiss_Kappa_0.72, RQ3_Spearman_0.81, RQ3_MAE_0.89
    2  04_query                             verified=2
        ids: Pipeline_Regeneration_10x10, Sec2_5_attrs_per_query
    1  00_data_preparation                  verified=1
        ids: UserFilter_20_reviews_15_words
    1  02_writing_analysis                  discrepant=1
        ids: RQ3_LLM_Full_Set_94%
    1  05_inject_noisy                      verified=1
        ids: BPE_aware_Error_Injection
    1  07_noisy_retrieval                   partial=1
        ids: Pipeline_Skip_ColBERTv2_SPLADE
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AB: --count-by-codebase-dir CLI flag (iter #144)
  PASS  --count-by-codebase-dir aggregates 17 claims across 8 dirs + exits 2 on missing JSON

All 28 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~55 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AB, +docstring, +main count to 28)
- 文档: `README.md` (+1 line `--count-by-codebase-dir` example)
- 验证: 28/28 cases pass; pre-commit hook freeze 28 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --count-by-codebase-dir` (~100ms, 不 re-run audit)

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
- **iter #143** — --list-discrepant
- **iter #144** — --count-by-codebase-dir (per-source-dir aggregation) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section (paper) | `--by-section` (iter #134) |
| per-section (code) | `--count-by-codebase-dir` (iter #144) ✓ 本轮 |
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
- **iter #143** — 27 cases
- **iter #144** — 28 cases (+ --count-by-codebase-dir) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
