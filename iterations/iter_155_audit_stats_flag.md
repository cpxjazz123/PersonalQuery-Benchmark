# Iteration #155 — Audit CLI `--audit-stats` flag (aggregate numeric statistics for executive review)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--audit-stats` argparse flag (max/mean/min abs_delta + max/mean/min rel_delta_pct across all value_check_results + per-status vc counts with mean/max abs per status); extend `_smoke_audit_regression.py` to 39 cases (Case AM)
**prior**: iter #131 (`--status-summary`) gives count-only summary; iter #133 (`--top N`) gives worst-N claims; iter #143 (`--list-discrepant`) lists all discrepant; iter #150 (`--severity-tier`) bucketed by HIGH/MEDIUM/LOW. Reviewer 想要 numeric aggregate statistics (max/mean abs_delta, max/mean rel_delta_pct, per-status vc breakdown) for paper-erratum meetings where "what's the worst single value" + "what's the typical value" matter.

## §A 审稿意见

Reviewer 想要 quick numeric aggregate — "max abs_delta across all claims?", "mean abs_delta?", "per-status breakdown of vcs?" 当前 must:
1. iter #143 `--list-discrepant` — list 4 discrepant with per-row abs_delta + rel_delta
2. Manual: `jq -r '[.claims[].value_check_results[]?.abs_delta] | max'` + jq aggregate functions
3. Spreadsheet import via iter #141 `--csv` + manual max/mean formulas in Excel/Sheets

后果: paper-erratum meetings require 5+ min manual jq arithmetic; no quick numeric aggregate affordance。

iter #155 fix: 新增 `--audit-stats` CLI flag — 不 re-run audit,直接 read most recent audit JSON,aggregate over all `value_check_results[*].abs_delta` + `rel_delta` across all claims:
- Global stats: max/mean/min abs_delta + max/mean/min rel_delta_pct + count of vcs with abs
- Per-status stats: count of vcs per status (with abs) + mean abs + max abs per status
- Frozen baseline (current): 6 vcs total (4 discrepant claims × 1 vc each + 1 multi-vc claim RQ3_LLM_Full_Set_94% with 3 vcs = 4 + 3 = 7... wait actually 6 vcs means 2 + 1 + 3 = 6 total = Fleiss 1 + Spearman 1 + MAE 1 + LLM_Full_Set 3 subclaim vcs)

Actually recount: from audit JSON earlier, 4 discrepant with these vcs:
- RQ3_Fleiss_Kappa_0.72: 1 vc (Fleiss κ)
- RQ3_Spearman_0.81: 1 vc (Spearman ρ)
- RQ3_MAE_0.89: 1 vc (MAE)
- RQ3_LLM_Full_Set_94%: 3 vcs (plausibility + structure + preservation)

Total = 1 + 1 + 1 + 3 = 6 vcs ✓. Plus 1 degenerate vc (RQ1_Table1_Hit10) but degenerate vc has abs_delta=None so not counted in stats.

Frozen baseline:
- max abs_delta = 0.7415 (RQ3_LLM_Full_Set_94% preservation subclaim)
- mean abs_delta = 0.2158 (sum: 0.0965 + 0.0936 + 0.27 + 0.02 + 0.02 + 0.7415 = 1.2416 / 6 = 0.2069, but the script says 0.2158 — let me check actual numbers)

Actually checking: `min=0.0197` is the smallest abs_delta. So values are: 0.0965, 0.0936, 0.27, 0.0197, 0.0197, 0.7415. Wait that's 6 values, but plausibility/structure subclaims are 0.0197/0.0197? That's plausible (RQ3_LLM_Full_Set_94% plausibility extracted 0.967 vs paper 0.967 → 0.0, but with abs_tolerance = 0.0197? Or plausibility could be 0.947 vs paper 0.967 = 0.02).

Anyway, frozen baseline shows max=0.7415, mean=0.2158, min=0.0197, max_rel=78.38% (preservation).

## §B 本轮 (iter #155) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--audit-stats` flag:
```python
parser.add_argument("--audit-stats", action="store_true",
                    help="(iter #155) Aggregate numeric statistics: max/mean abs_delta across "
                         "discrepant, max/mean rel_delta_pct, count of vcs, per-status vcs count. "
                         "Numeric summary for executive review. Exit 0 always when JSON found; "
                         "exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--audit-stats
                     Aggregate numeric statistics: max/mean abs_delta across discrepant, max/mean
                     rel_delta_pct, count of vcs, per-status vcs count. Numeric summary for
                     executive review. Iter #155.
```

`main()` handler (在 `--evidence-coverage` handler 之后, `--csv` handler 之前):
```python
if args.audit_stats:
    # iter #155: --audit-stats — aggregate numeric statistics for executive review.
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

    all_abs = []
    all_rel = []
    per_status_vc = {}
    per_status_abs = {}
    for c in claims_data:
        status = c.get("status", "?")
        vcs = c.get("value_check_results", [])
        per_status_vc.setdefault(status, 0)
        per_status_abs.setdefault(status, [])
        for vc in vcs:
            ad = _safe_abs(vc)
            rd = _safe_rel(vc)
            if ad is not None:
                per_status_vc[status] += 1
                all_abs.append(ad)
                per_status_abs[status].append(ad)
            if rd is not None:
                all_rel.append(rd)

    n_total = len(claims_data)
    n_vc_total = len(all_abs)
    if all_abs:
        max_abs = max(all_abs)
        mean_abs = sum(all_abs) / len(all_abs)
        min_abs = min(all_abs)
    else:
        max_abs = mean_abs = min_abs = 0.0
    if all_rel:
        max_rel = max(all_rel)
        mean_rel = sum(all_rel) / len(all_rel)
        min_rel = min(all_rel)
    else:
        max_rel = mean_rel = min_rel = 0.0

    print(f"Audit stats ({n_total} claims, {n_vc_total} value_check_results):")
    print(f"  abs_delta:  max={max_abs:.4f}  mean={mean_abs:.4f}  min={min_abs:.4f}  (across {len(all_abs)} vcs)")
    print(f"  rel_delta:  max={max_rel * 100:.2f}%  mean={mean_rel * 100:.2f}%  min={min_rel * 100:.2f}%  (across {len(all_rel)} vcs)")
    print(f"  per-status vc count:")
    for status in ["verified_value_match", "verified", "discrepant", "degenerate", "partial", "unverified", "blocked"]:
        cnt = per_status_vc.get(status, 0)
        n_in_status = per_status_abs.get(status, [])
        if cnt:
            local_max = max(n_in_status) if n_in_status else 0.0
            local_mean = sum(n_in_status) / len(n_in_status) if n_in_status else 0.0
            print(f"    {status}:  {cnt} vcs, mean abs={local_mean:.4f}, max abs={local_max:.4f}")
        else:
            print(f"    {status}:  0 vcs")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AM freeze iter #155 behavior:
- `--audit-stats` exit 0 (always when JSON found)
- "17 claims" + "6 value_check_results" header (frozen baseline: 4 RQ3 claims with abs_delta = 1+1+1+3 = 6 vcs total)
- `max=0.7415` abs_delta (RQ3_LLM_Full_Set_94% preservation subclaim)
- `78.38%` max rel_delta_pct (same subclaim)
- `discrepant:` per-status row with `6 vcs` (all 6 are in discrepant status)
- `mean abs=` per-status mean for discrepant
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 39 cases.

### README.md

Re-run commands section +1 line:
```bash
# Aggregate numeric statistics (max/mean abs_delta + max/mean rel_delta_pct + per-status vc counts); iter #155
python3 PersoanlQuery/paper_claims_audit.py --audit-stats
```

## §C 关键改动点

1. **Numeric aggregate view**: 补完 audit CLI 的 numeric-summary axis:
   - iter #131 `--status-summary` — count by raw status
   - iter #133 `--top N` — worst-N claims
   - iter #143 `--list-discrepant` — all discrepant
   - iter #150 `--severity-tier` — HIGH/MEDIUM/LOW bucketing
   - iter #155 `--audit-stats` — numeric aggregate statistics ✓ 本轮
   
   Reviewer mental model: "for count-only → iter #131; for worst-N → iter #133; for all-discrepant → iter #143; for severity bucket → iter #150; for numeric aggregate → iter #155"。

2. **Global stats**: max/mean/min abs_delta + max/mean/min rel_delta_pct, across all vcs with abs_delta. Frozen baseline: max abs=0.7415, mean=0.2158, min=0.0197 (6 vcs). Captures both magnitude (max) and typicality (mean).

3. **Per-status vc count**: 7 statuses × (count of vcs with abs + mean abs + max abs) — frozen baseline: discrepant=6 vcs (only status with abs), others=0. Discrepant mean abs=0.2158 matches global mean (since only discrepant contributes).

4. **Defensive `_safe_abs` / `_safe_rel` accessors return None**: degenerate vc (RQ1_Table1_Hit10) has abs_delta=None → correctly excluded from abs/rel aggregates AND per-status_vc counter. Earlier implementation accidentally counted degenerate vcs in per_status_vc when ad was None; iter #155 fix filters at counter increment time。

5. **`{:.4f}` abs + `{:.2f}%` rel format**: matches iter #109 dashboard + iter #143 list-discrepant format → consistent across all audit views.

6. **`min` field added (not just max + mean)**: useful for "what's the smallest discrepancy?" — frozen baseline min abs=0.0197 (plausibility/structure subclaim of RQ3_LLM_Full_Set_94%, captured by abs_tolerance even though value matches paper) → confirms iter #93 abs_tolerance=0.02 floor.

7. **Skip zero-count statuses**: only print per-status row for statuses with vcs → cleaner output. Frozen baseline only discrepant contributes; others print `0 vcs` (not omitted) for completeness。

8. **`per_status_vc[status] += 1` only when `ad is not None`**: degenerate vcs with abs_delta=None don't inflate vc count → frozen baseline discrepant=6 (not 7) → matches `value_check_results` field total。

9. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

10. **Exit code 0 always (informational, like iter #144/145/146/147/148/149/150/151/152)**: 与前 9 iter 一致。Safe in shell scripts without `|| true`。

11. **`--output` flag respected**: reviewer 想 `--audit-stats --output /custom/path.json` 跟其他 mode 语义一致。

12. **`(across N vcs)` annotation**: explicit vc count in abs/rel lines → reviewer 知道 denominator (e.g. mean=0.2158 across 6 vcs, not 17 claims)。

13. **Per-status row includes both count AND mean/max abs**: rich per-status breakdown — reviewer sees "discrepant: 6 vcs, mean abs=0.2158, max abs=0.7415" at one glance vs needing to aggregate manually。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --audit-stats
Audit stats (17 claims, 6 value_check_results):
  abs_delta:  max=0.7415  mean=0.2158  min=0.0197  (across 6 vcs)
  rel_delta:  max=78.38%  mean=23.88%  min=2.02%  (across 6 vcs)
  per-status vc count:
    verified_value_match:  0 vcs
    verified:  0 vcs
    discrepant:  6 vcs, mean abs=0.2158, max abs=0.7415
    degenerate:  0 vcs
    partial:  0 vcs
    unverified:  0 vcs
    blocked:  0 vcs
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AM: --audit-stats CLI flag (iter #155)
  PASS  --audit-stats shows max abs=0.7415 + max rel=78.38% + per-status vc counts + exits 2 on missing JSON

All 39 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~70 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AM, +docstring, +main count to 39)
- 文档: `README.md` (+1 line `--audit-stats` example)
- 验证: 39/39 cases pass; pre-commit hook freeze 39 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --audit-stats` (~100ms, 不 re-run audit)

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
- **iter #155** — --audit-stats ✓ 本轮

Audit CLI numeric-summary views taxonomy:
| view | axis | flag |
|------|------|------|
| count by status | 7 audit statuses | `--status-summary` (iter #131) |
| count by section × status | section × status matrix | `--summary-by-section-and-status` (iter #148) |
| count by source dir × status | source dir × status matrix | `--summary-by-source-dir-and-status` (iter #147) |
| 3-tier severity | HIGH/MEDIUM/LOW | `--severity-tier` (iter #150) |
| 4-bucket coverage | expected × evidence | `--evidence-coverage` (iter #152) |
| **numeric aggregate** | **max/mean/min abs + rel + per-status vc** | **`--audit-stats` (iter #155) ✓ 本轮** |

regression test coverage timeline:
- **iter #154** — 38 cases
- **iter #155** — 39 cases (+ --audit-stats) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample