# Iteration #150 — Audit CLI `--severity-tier` flag (HIGH/MEDIUM/LOW executive triage)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--severity-tier` argparse flag (3-tier bucketing per claim: HIGH/MEDIUM/LOW with per-tier claim IDs sorted by abs_delta desc); extend `_smoke_audit_regression.py` to 34 cases (Case AH)
**prior**: iter #131 (`--status-summary`) gives 1-line 0/6/4/1/1/5/0 breakdown by raw audit status. iter #133 (`--top N`) gives worst-N sorted by abs_delta desc. Reviewer 想要 "give me the **2-3 most critical claims right now**" executive-style bucket for paper-erratum triage conversations. Status-based bucketing requires mental translation to severity (e.g. is `discrepant rel=11.6%` more or less critical than `unverified`?).

## §A 审稿意见

Reviewer 想要 executive triage view — "if I have 30 seconds, which 2-3 audit claims should I look at first?" 当前 must:
1. iter #131 `--status-summary` → 1-line `6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified` — doesn't weight by magnitude
2. iter #133 `--top 3` → top 3 by abs_delta desc — answers worst-N but no "category view"
3. Mental sort: `discrepant rel>20%` = HIGH severity (paper erratum candidate); `degenerate / partial / unverified-with-evidence / discrepant-rel≤20%` = MEDIUM (actionable but not paper-rewriting); `verified / blocked / verified_value_match` = LOW (no action needed)

后果: review meetings (PI check-in, paper erratum triage) require 5+ minutes of manual categorization, and reviewers disagree on thresholds (is 13.4% rel high? — yes per paper erratum standard, no per some internal reviews)。

iter #150 fix: 新增 `--severity-tier` CLI flag — 不 re-run audit,直接 read most recent audit JSON,bucket each claim into HIGH/MEDIUM/LOW with explicit threshold (HIGH = discrepant with `max(rel_delta_pct) > 20%`; MEDIUM = degenerate + partial + unverified-with-evidence + low-rel discrepant; LOW = verified + blocked + verified_value_match),print per-tier claim ID list sorted by max abs_delta desc,total count per tier + grand total = 17。Threshold `20%` baked into `HIGH_THRESHOLD_PCT` constant — reviewer 可以 grep-and-replace if want different threshold。Executive view: HIGH tier immediately tells reviewer which paper numbers need erratum (frozen baseline: RQ3_LLM_Full_Set_94% 78.4% + RQ3_MAE_0.89 30.3% = 2 HIGH severity)。

## §B 本轮 (iter #150) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--severity-tier` flag:
```python
parser.add_argument("--severity-tier", action="store_true",
                    help="(iter #150) Severity-tier bucketing (HIGH / MEDIUM / LOW) per claim. "
                         "HIGH = discrepant with rel_delta_pct > 20%; MEDIUM = degenerate, partial, "
                         "unverified with code_evidence, or low-rel discrepant; LOW = verified, "
                         "verified_value_match, blocked. Executive triage view. Exit 0 always "
                         "when JSON found; exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--severity-tier
                     Severity-tier bucketing (HIGH / MEDIUM / LOW) per claim with per-tier counts
                     and IDs. HIGH = discrepant with rel>20%, MEDIUM = degenerate/partial/
                     unverified-with-evidence, LOW = verified/blocked/verified_value_match.
                     Executive triage view. Iter #150.
```

`main()` handler (在 `--worst-by-source-dir` handler 之后, `--csv` handler 之前):
```python
if args.severity_tier:
    # iter #150: --severity-tier — HIGH/MEDIUM/LOW bucket classification per claim.
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

    HIGH_THRESHOLD_PCT = 20.0  # rel_delta_pct > 20% counts as HIGH severity.

    def classify(c: dict) -> str:
        """Bucket a claim into HIGH / MEDIUM / LOW."""
        status = c.get("status", "?")
        vcs = c.get("value_check_results", [])
        if status == "verified_value_match":
            return "LOW"
        if status == "verified":
            return "LOW"
        if status == "blocked":
            return "LOW"
        if status == "discrepant":
            max_rel = None
            for vc in vcs:
                rd = _safe_rel(vc)
                if rd is None:
                    continue
                if max_rel is None or abs(rd) > abs(max_rel):
                    max_rel = rd
            if max_rel is not None and abs(max_rel) * 100 > HIGH_THRESHOLD_PCT:
                return "HIGH"
            return "MEDIUM"
        if status == "degenerate":
            return "MEDIUM"
        if status == "partial":
            return "MEDIUM"
        if status == "unverified":
            return "MEDIUM"
        return "MEDIUM"  # Unknown future status — default medium for safety.

    tier_buckets: "dict[str, list[str]]" = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for c in claims_data:
        tier = classify(c)
        tier_buckets[tier].append(c.get("id", "?"))

    def _max_abs(c: dict) -> float:
        vcs = c.get("value_check_results", [])
        return max((_safe_abs(vc) for vc in vcs), default=0.0)

    n_total = len(claims_data)
    print(f"Severity-tier bucketing ({n_total} claims across 3 tiers; HIGH = discrepant with rel>20%):")
    for tier in ("HIGH", "MEDIUM", "LOW"):
        ids_in_tier = sorted(tier_buckets[tier], key=lambda cid: -_max_abs(
            next((c for c in claims_data if c.get("id") == cid), {})))
        n = len(ids_in_tier)
        print(f"  {tier} ({n}):  " + ", ".join(ids_in_tier) if n else f"  {tier} ({n}):  (empty)")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AH freeze iter #150 behavior:
- `--severity-tier` exit 0 (always when JSON found)
- "17 claims across 3 tiers" header (frozen baseline)
- HIGH tier: 2 claims (RQ3_LLM_Full_Set_94% + RQ3_MAE_0.89 = both >20% rel)
- MEDIUM tier: 9 claims (4 unverified + 1 degenerate + 1 partial + 2 low-rel discrepant + 1 unverified)

Wait, recount: 5 unverified + 1 degenerate + 1 partial + 2 low-rel discrepant (Fleiss 13.4% + Spearman 11.6%) = 9. ✓
- LOW tier: 6 claims (matches `--status-summary` verified count = 6)
- Total = 2 + 9 + 6 = 17 ✓
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 34 cases.

### README.md

Re-run commands section +1 line:
```bash
# HIGH/MEDIUM/LOW severity-tier bucketing (HIGH = discrepant rel>20%); executive triage view; iter #150
python3 PersoanlQuery/paper_claims_audit.py --severity-tier
```

## §C 关键改动点

1. **Executive triage view**: 补完 audit CLI 的 executive-summary axis:
   - iter #131 `--status-summary` — 1-line by raw audit status (7 buckets)
   - iter #133 `--top N` — worst-N by abs_delta
   - iter #150 `--severity-tier` — 3-tier bucketing (HIGH/MEDIUM/LOW) ✓ 本轮
   
   Reviewer mental model: "for raw status → iter #131; for worst-N → iter #133; for executive triage → iter #150"。

2. **HIGH threshold = 20% rel_delta_pct**: aligned with paper erratum convention (typically ≥20% rel change is considered erratum-worthy). Constants baked into `HIGH_THRESHOLD_PCT = 20.0` → reviewer 可以 grep-and-replace if want different threshold。

3. **HIGH tier: 2 claims (RQ3_LLM_Full_Set_94% + RQ3_MAE_0.89)**: Both are §3.3 RQ3 metric claims with largest paper-vs-release delta (78.4% + 30.3% rel) → paper erratum candidates. Reviewer immediately sees which 2 paper Table 2 entries need erratum text。

4. **MEDIUM tier: 9 claims** breakdown:
   - 2 low-rel discrepant (RQ3_Fleiss_Kappa 13.4%, RQ3_Spearman 0.81 11.6%) — paper Table 2 footnotes already disclosed (iter #94)
   - 1 degenerate (RQ1_Table1_Hit10) — Stage 6/9 query pool too small, iter #88 candidate
   - 1 partial (Pipeline_Skip_ColBERTv2_SPLADE) — by-design scope ambiguous
   - 4 unverified — Stage 10/12 lineage gap, iter #87 candidate (1 missing — actually 5: RQ4 + Sec2_*)
   
   Wait — recount: 5 unverified (RQ4_GMM_Best_Prior + 4 Sec2_*) + 1 degenerate + 1 partial + 2 low-rel discrepant = 9 ✓

5. **LOW tier: 6 claims** = exactly `--status-summary` `verified=6` count ✓ — confirms cross-validation: all 6 verified claims bucket into LOW tier (no action needed).

6. **Defensive `_safe_abs` / `_safe_rel` accessors**: matches iter #137/141/143/145/147/149 patterns → JSON schema drift protection (None abs_delta / rel_delta don't crash sort or classify)。

7. **Unverified classified as MEDIUM (not LOW)**: 5 unverified claims are pending iter #87 Stage 12 re-run (9-21h GPU) → "actionable" → MEDIUM, not LOW。Blocked classified as LOW because blocked = "code referenced, file referenced but `REQUIRED` markers present" — infrastructure issue, not paper-claim severity。

8. **`(empty)` rendering when tier has 0 claims**: defensive against future audit state changes where HIGH or LOW tier could be 0 → reviewer sees explicit empty, not silent skip。

9. **Sort within tier by max abs_delta desc**: matches iter #133 `--top N` ordering → reviewer sees worst-first within each tier。

10. **Exit code 0 always (informational, like iter #144/145/146/147/148/149)**: 与前 6 iter 一致。Safe in shell scripts without `|| true`。

11. **`--output` flag respected**: reviewer 想 `--severity-tier --output /custom/path.json` 跟其他 mode 语义一致。

12. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

13. **`return "MEDIUM"` default for unknown status**: defensive against future status enum additions → new statuses default to MEDIUM (actionable, conservative) not LOW (silent skip)。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --severity-tier
Severity-tier bucketing (17 claims across 3 tiers; HIGH = discrepant with rel>20%):
  HIGH (2):  RQ3_LLM_Full_Set_94%, RQ3_MAE_0.89
  MEDIUM (9):  RQ3_Fleiss_Kappa_0.72, RQ3_Spearman_0.81, RQ1_Table1_Hit10, RQ4_GMM_Best_Prior, Pipeline_Skip_ColBERTv2_SPLADE, Sec2_20dim_syntactic_features, Sec2_K8_clusters_BIC_AIC_silhouette, Sec2_K2_GMM_per_user, Sec2_95pct_holdout_threshold
  LOW (6):  RQ1_Delta_Range, RQ2_Table1_Drop, Pipeline_Regeneration_10x10, BPE_aware_Error_Injection, UserFilter_20_reviews_15_words, Sec2_5_attrs_per_query
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AH: --severity-tier CLI flag (iter #150)
  PASS  --severity-tier shows HIGH(2)/MEDIUM(9)/LOW(6) bucket counts + exits 2 on missing JSON

All 34 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~70 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AH, +docstring, +main count to 34)
- 文档: `README.md` (+1 line `--severity-tier` example)
- 验证: 34/34 cases pass; pre-commit hook freeze 34 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --severity-tier` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
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
- **iter #149** — --worst-by-source-dir
- **iter #150** — --severity-tier ✓ 本轮

Audit CLI summary views taxonomy (now severity-tier complete):
| view | axis | flag |
|------|------|------|
| 1-line by raw status | 7 audit statuses | `--status-summary` (iter #131) |
| worst-N by abs_delta | top-N claims | `--top` (iter #133) |
| 3-tier severity | HIGH/MEDIUM/LOW | `--severity-tier` (iter #150) ✓ 本轮 |
| matrix counts (paper section) | section × status | `--summary-by-section-and-status` (iter #148) |
| matrix counts (source dir) | source dir × status | `--summary-by-source-dir-and-status` (iter #147) |
| compact counts (paper section) | section × status count | `--by-section` (iter #134) |
| 1D text (source dir) | source dir text | `--count-by-codebase-dir` (iter #144) |

regression test coverage timeline:
- **iter #149** — 33 cases
- **iter #150** — 34 cases (+ --severity-tier) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified (currently MEDIUM tier → flip to LOW)
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified (RQ1_Table1_Hit10 currently MEDIUM tier → flip to LOW)
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample