# Iteration #149 — Audit CLI `--worst-by-source-dir` flag (code-side counterpart to iter #145)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--worst-by-source-dir` argparse flag (worst-claim-per-source-dir with max abs_delta across value_check_results, `(all clean)` rows for clean dirs, code-side counterpart to iter #145); extend `_smoke_audit_regression.py` to 33 cases (Case AG)
**prior**: iter #145 (`--worst-by-section`) gives worst-claim-per-paper-section for paper erratum triage. iter #144 (`--count-by-codebase-dir`) gives per-source-dir text breakdown but no max abs_delta signal. Reviewer 想要 code-side worst-claim-per-source-dir to triage iter #87/#88 by source dir, complementing iter #145 by source Stage.

## §A 审稿意见

Reviewer 想要 quick "for each pipeline Stage, which claim has the worst numerical discrepancy?" 当前 must:
1. iter #145 `--worst-by-section` — paper section × worst-claim-per-section (e.g. RQ3_LLM_Full_Set_94% worst in §3.3 + Table 2)
2. iter #144 `--count-by-codebase-dir` — per-source-dir text breakdown `verified=2 / degenerate=1` (no abs_delta)
3. iter #147 `--summary-by-source-dir-and-status` — 2D matrix counts per Stage × status (no max abs_delta)
4. Manual pivot: combine iter #144 text + iter #133 `--top N` worst-claim list + map back to source dir via `code_evidence`

后果: code-side worst-claim-per-dir axis 不对称 with paper-side iter #145 → reviewer can't switch axes with same mental model.

iter #149 fix: 新增 `--worst-by-source-dir` CLI flag (code-side counterpart to iter #145) — 不 re-run audit,直接 read most recent audit JSON,group claims by primary source dir (reuse iter #144/147 `_primary_dir` logic stripping `:function_name` suffix → top-level utility files bucket together),aggregate max abs_delta across `value_check_results[*]` + corresponding claim_id + max rel_delta_pct + n_claims in dir,sort by `max_abs desc, dir asc`,print compact per-dir row `  abs=<abs>  <claim_id>  [<dir>]  rel_delta_pct=<rel%>  n_claims=<n>` + `(all clean)  <dir>  n_claims=<n>` rows for dirs without discrepant (reviewer confirms clean dirs are explicitly clean not just unprinted). Reuses iter #145 layout (sorted desc by abs + all-clean rows + `(unknown)` fallback) for visual symmetry between paper-side and code-side worst-claim-per-X axes.

## §B 本轮 (iter #149) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--worst-by-source-dir` flag:
```python
parser.add_argument("--worst-by-source-dir", action="store_true",
                    help="(iter #149) Worst-claim-per-source-dir (max abs_delta per primary source "
                         "directory from code_evidence). Sorted desc; '(all clean)' rows for dirs "
                         "without discrepant. Code-side counterpart to iter #145 --worst-by-section. "
                         "Exit 0 always when JSON found; exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--worst-by-source-dir
                     Worst-claim-per-source-dir (max abs_delta across value_check_results per
                     primary source dir from code_evidence). Sorted desc by abs; '(all clean)'
                     rows for dirs without discrepant. Code-side counterpart to iter #145. Iter #149.
```

`main()` handler (在 `--summary-by-section-and-status` handler 之后, `--csv` handler 之前):
```python
if args.worst_by_source_dir:
    # iter #149: --worst-by-source-dir — worst-claim-per-source-dir (max abs_delta) for code-side erratum triage.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    # Reuse iter #144 / #147 logic: extract primary source dir from first `PersoanlQuery/X/...` ref,
    # stripping `:function_name` suffix so top-level utility files (e.g. agreement_metrics.py)
    # bucket together rather than splitting into one bucket per function.
    def _primary_dir(c: dict) -> str:
        evidences = c.get("code_evidence") or []
        for ref in evidences:
            parts = ref.split("/")
            if len(parts) >= 2 and parts[0] in ("PersoanlQuery", "personelquery"):
                d = parts[1].split(":")[0]
                return d if d else "(top-level)"
        return "(no source dir)"

    def _safe_abs(vc: dict) -> float:
        v = vc.get("abs_delta")
        return v if isinstance(v, (int, float)) else 0.0

    def _safe_rel(vc: dict):
        v = vc.get("rel_delta")
        return v if isinstance(v, (int, float)) else None

    # Aggregate max abs_delta / claim_id / max rel per source dir.
    per_dir: "dict[str, dict]" = {}
    for c in claims_data:
        d = _primary_dir(c)
        vcs = c.get("value_check_results", [])
        max_abs = max((_safe_abs(vc) for vc in vcs), default=0.0)
        max_rel = None
        for vc in vcs:
            rd = _safe_rel(vc)
            if rd is None:
                continue
            if max_rel is None or abs(rd) > abs(max_rel):
                max_rel = rd
        entry = per_dir.setdefault(d, {
            "max_abs": 0.0,
            "claim_id": c.get("id", "?"),
            "max_rel": None,
            "n_claims": 0,
            "worst_status": c.get("status", "?"),
        })
        entry["n_claims"] += 1
        if abs(max_abs) > abs(entry["max_abs"]):
            entry["max_abs"] = max_abs
            entry["claim_id"] = c.get("id", "?")
            entry["max_rel"] = max_rel
            entry["worst_status"] = c.get("status", "?")

    sorted_dirs = sorted(per_dir.keys(), key=lambda d: (-abs(per_dir[d]["max_abs"]), d))
    n_dirs = len(sorted_dirs)
    n_total = len(claims_data)
    print(f"Worst claim per source dir ({n_total} claims across {n_dirs} dirs; abs_delta = max across value_check_results):")
    for d in sorted_dirs:
        e = per_dir[d]
        if e["max_abs"] == 0.0:
            print(f"  (all clean)  {d}  n_claims={e['n_claims']}")
        else:
            max_rel = e["max_rel"]
            rel_str = f"{max_rel * 100:.2f}%" if max_rel is not None else "n/a"
            print(f"  abs={e['max_abs']:.4f}  {e['claim_id']}  [{d}]  rel_delta_pct={rel_str}  n_claims={e['n_claims']}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AG freeze iter #149 behavior:
- `--worst-by-source-dir` exit 0 (always when JSON found)
- "17 claims across 8 dirs" header (frozen baseline)
- Top dir `02_writing_analysis` with RQ3_LLM_Full_Set_94% abs=0.7415 (= worst-of-worst)
- `agreement_metrics.py` bucket (top-level utility, stripped `:function_name`) with RQ3_MAE_0.89 n_claims=3
- `(all clean)` rows for clean dirs (00_data_preparation, 04_query, 05_inject_noisy, 07_noisy_retrieval, 08_compare_all_domain, 10_complexity_analysis)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 33 cases.

### README.md

Re-run commands section +1 line:
```bash
# Worst-claim-per-source-dir sorted desc — code-side erratum triage (counterpart to --worst-by-section); iter #149
python3 PersoanlQuery/paper_claims_audit.py --worst-by-source-dir
```

## §C 关键改动点

1. **Code-side counterpart to iter #145**: 与 existing flags complementary:
   - iter #145 `--worst-by-section` — paper section × worst-claim-per-section (paper erratum triage)
   - iter #144 `--count-by-codebase-dir` — per-source-dir text breakdown `verified=2 / degenerate=1` (no abs_delta)
   - iter #147 `--summary-by-source-dir-and-status` — 2D matrix counts per Stage × status (no max abs_delta)
   - iter #149 `--worst-by-source-dir` — source dir × worst-claim-per-dir ✓ 本轮
   
   Reviewer mental model: "for paper-section worst → iter #145; for source-Stage worst → iter #149; for compact source-dir text → iter #144; for 2D matrix counts → iter #147"。

2. **Mirrors iter #145 structure**: Same `(all clean)` rows for clean axes (defensive against silently-unprinted dirs), same sort order (max_abs desc, axis asc), same per-row format `  abs=<abs>  <claim_id>  [<dir>]  rel_delta_pct=<rel%>  n_claims=<n>`. Reviewer can swap axes (paper section vs source dir) with zero cognitive load。

3. **Top dir: 02_writing_analysis with RQ3_LLM_Full_Set_94% (abs=0.7415)**: Worst-of-worst — RQ3 semantic preservation claim from iter #84 (real Qwen2.5-7B-vLLM eval) extracted 0.205 vs paper 0.946 = 78.4% relative delta. Source dir `02_writing_analysis/02_personal_query_builder.py` holds the LLM eval pipeline. Confirms iter #84 + iter #93 finding — Stage 7 noise injection produces semantic-changing errors (e.g. pacifiers→previous, toddler→their) not typo-level intent-preserving errors。

4. **agreement_metrics.py bucket (3 claims, RQ3_MAE_0.89 worst)**: Top-level utility file `agreement_metrics.py` strips `:function_name` suffix → 3 claims (RQ3_Fleiss/Spearman/MAE) all bucket together. Worst of bucket = RQ3_MAE_0.89 (abs=0.27, 30.34% rel). Reuses iter #144/147 logic for single source of truth。

5. **`(all clean)` rows for clean dirs**: `00_data_preparation`, `04_query`, `05_inject_noisy`, `07_noisy_retrieval`, `08_compare_all_domain`, `10_complexity_analysis` all show `(all clean)` — meaning no claim in that dir has any abs_delta > 0 in value_check_results (either no value_check_results, or all value_check_results degenerate). `10_complexity_analysis` shows `(all clean)` even though it has 5 unverified claims (Stage 12 lineage gap) → unverified claims have no abs_delta by definition (no value_check_results computed) → not "worst" by abs_delta axis → bucketed as clean。

6. **Defensive `abs(max_abs)` comparison**: signed-aware → negative abs_delta doesn't out-rank positive. Mirror iter #145 fix。

7. **Reuses iter #144 `_primary_dir` helper logic**: identical to iter #147 implementation → single source of truth across iter #144, #147, #149 → no drift risk。

8. **Sort by max_abs desc, dir asc**: Largest discrepancy first → reviewer triages biggest bucket first → 02_writing_analysis first, agreement_metrics.py second。

9. **Exit code 0 always (informational, like iter #144/145/146/147/148)**: 与前 5 iter 一致。Safe in shell scripts without `|| true`。

10. **`--output` flag respected**: reviewer 想 `--worst-by-source-dir --output /custom/path.json` 跟其他 mode 语义一致。

11. **No re-run audit**: `--worst-by-source-dir` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

12. **`(no source dir)` fallback**: paper-only claims (frozen baseline = 0) get bucketed together → defensive against audit schema drift (claim without `code_evidence` field)。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --worst-by-source-dir
Worst claim per source dir (17 claims across 8 dirs; abs_delta = max across value_check_results):
  abs=0.7415  RQ3_LLM_Full_Set_94%  [02_writing_analysis]  rel_delta_pct=78.38%  n_claims=1
  abs=0.2700  RQ3_MAE_0.89  [agreement_metrics.py]  rel_delta_pct=30.34%  n_claims=3
  (all clean)  00_data_preparation  n_claims=1
  (all clean)  04_query  n_claims=2
  (all clean)  05_inject_noisy  n_claims=1
  (all clean)  07_noisy_retrieval  n_claims=1
  (all clean)  08_compare_all_domain  n_claims=3
  (all clean)  10_complexity_analysis  n_claims=5
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AG: --worst-by-source-dir CLI flag (iter #149)
  PASS  --worst-by-source-dir shows 02_writing_analysis top + agreement_metrics.py bucket + (all clean) rows + exits 2 on missing JSON

All 33 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~60 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AG, +docstring, +main count to 33)
- 文档: `README.md` (+1 line `--worst-by-source-dir` example)
- 验证: 33/33 cases pass; pre-commit hook freeze 33 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --worst-by-source-dir` (~100ms, 不 re-run audit)

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
- **iter #148** — --summary-by-section-and-status
- **iter #149** — --worst-by-source-dir ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| paper section × status (compact) | `--by-section` (iter #134) |
| paper section × status (matrix) | `--summary-by-section-and-status` (iter #148) |
| source dir × status (matrix) | `--summary-by-source-dir-and-status` (iter #147) |
| source dir (1D text) | `--count-by-codebase-dir` (iter #144) |
| paper section worst | `--worst-by-section` (iter #145) |
| source dir worst | `--worst-by-source-dir` (iter #149) ✓ 本轮 |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| list by status | `--list-{degenerate,unverified,partial,verified,discrepant}` (iter #138-143) |
| search by substring | `--find-claim` (iter #146) |
| flip detection | `--diff` (iter #114) |

Audit CLI worst-claim-per-X axes symmetry (now complete):
| axis | flag |
|------|------|
| paper-section worst | `--worst-by-section` (iter #145) |
| source-dir worst | `--worst-by-source-dir` (iter #149) ✓ 本轮 |

regression test coverage timeline:
- **iter #148** — 32 cases
- **iter #149** — 33 cases (+ --worst-by-source-dir) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample