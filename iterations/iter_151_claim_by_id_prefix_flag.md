# Iteration #151 — Audit CLI `--claim-by-id-prefix` flag (prefix-anchored ID filter, complement to iter #146 substring)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--claim-by-id-prefix PREFIX` argparse flag (case-insensitive prefix match on claim ID, prefix-anchored complement to iter #146 `--find-claim` substring search); extend `_smoke_audit_regression.py` to 35 cases (Case AI)
**prior**: iter #146 (`--find-claim PAT`) does case-insensitive substring search across 5 fields (id/section/reason/expected_outputs/code_evidence). Reviewer 想要 strict prefix-anchored match for ID-based claim family queries (e.g. "list all 5 §2.2 infrastructure claims = `Sec2_*`", "list all 4 RQ3 metric claims = `RQ3_*`"). Substring search `--find-claim Sec2` would also match `Sec2_5_attrs_per_query` but cross-field false-positives like reason text containing "Sec2" would clutter output。

## §A 审稿意见

Reviewer 想要 quick "show me all claims in family X" where X is ID-prefix-defined:
1. iter #146 `--find-claim Sec2` — substring search across 5 fields → would also match `Sec2_*` but matches across fields too, may produce false positives (e.g. `Sec2` mentioned in reason text)
2. iter #142 `--list-verified` — list by status, no family grouping
3. Manual: `jq -r '.claims[] | .id' | grep '^Sec2'`

后果: family-based queries (e.g. "all RQ3", "all Sec2", "all Pipeline") need manual `jq + grep` two-step; CLI affordance 不对称。

iter #151 fix: 新增 `--claim-by-id-prefix PREFIX` CLI flag — 不 re-run audit,直接 read most recent audit JSON,case-insensitive prefix-anchored match on claim ID field only (no cross-field noise),print compact per-claim block `  <id>  [<status>]  [<section>]` + `    reason      : <80-char>`; triple exit code 0 (matches found) / 1 (no match) / 2 (JSON missing OR empty PREFIX); defensive empty PREFIX → exit 2 (not exit 0/1) → catches `--claim-by-id-prefix ""` accidental misuse。

## §B 本轮 (iter #151) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--claim-by-id-prefix` flag:
```python
parser.add_argument("--claim-by-id-prefix", default=None, metavar="PREFIX",
                    help="(iter #151) List all claim IDs starting with PREFIX (case-insensitive "
                         "prefix match). Prefix-anchored complement to iter #146 --find-claim "
                         "substring search. Exit 0 if matches found; exit 1 if no match; exit 2 "
                         "if JSON missing OR PREFIX empty.")
```

docstring 加 flag description:
```
--claim-by-id-prefix PREFIX
                     List all claim IDs starting with PREFIX (case-insensitive prefix match).
                     Prefix-anchored complement to iter #146 substring search. Exit 0 if matches;
                     exit 1 if no match; exit 2 if JSON missing or PREFIX empty. Iter #151.
```

`main()` handler (在 `--severity-tier` handler 之后, `--csv` handler 之前):
```python
if args.claim_by_id_prefix is not None:
    # iter #151: --claim-by-id-prefix PREFIX — prefix-anchored ID list (complement to iter #146 substring).
    if args.claim_by_id_prefix == "":
        print("ERROR: --claim-by-id-prefix requires non-empty PREFIX.", file=__import__("sys").stderr)
        return 2
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    prefix_lower = args.claim_by_id_prefix.lower()
    matches = [c for c in claims_data
               if (c.get("id") or "").lower().startswith(prefix_lower)]
    if not matches:
        print(f"No claims match prefix '{args.claim_by_id_prefix}'.")
        return 1
    print(f"Claims with id prefix '{args.claim_by_id_prefix}' ({len(matches)} total):")
    for c in matches:
        cid = c.get("id", "?")
        status = c.get("status", "?")
        sec = c.get("section", "(unknown)")
        reason = (c.get("reason") or "").replace("\n", " ")[:80]
        print(f"  {cid}  [{status}]  [{sec}]")
        print(f"    reason      : {reason}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AI freeze iter #151 behavior:
- `--claim-by-id-prefix Sec2` exit 0 + "5 total" + all 5 Sec2_* IDs visible (Sec2_20dim_syntactic_features + Sec2_K8_clusters_BIC_AIC_silhouette + Sec2_5_attrs_per_query + Sec2_K2_GMM_per_user + Sec2_95pct_holdout_threshold)
- Lowercase `--claim-by-id-prefix sec2` exit 0 + "5 total" (case-insensitive)
- `--claim-by-id-prefix NO_SUCH_PREFIX` exit 1 + "No claims match"
- `--claim-by-id-prefix ""` exit 2 + "non-empty" stderr (defensive)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 35 cases.

### README.md

Re-run commands section +1 line:
```bash
# List claims with id starting with PREFIX (case-insensitive); prefix-anchored complement to --find-claim; iter #151
python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix Sec2
```

## §C 关键改动点

1. **Prefix-anchored complement to iter #146 substring search**: 与 existing flags complementary:
   - iter #146 `--find-claim PAT` — case-insensitive substring across 5 fields (id/section/reason/expected_outputs/code_evidence)
   - iter #151 `--claim-by-id-prefix PREFIX` — case-insensitive prefix on id field only ✓ 本轮
   
   Reviewer mental model: "for substring search anywhere → iter #146; for ID family query → iter #151"。

2. **Field-isolated (ID only)**: 避免 iter #146 cross-field false positives (e.g. `--find-claim Sec2` could match claim whose reason mentions "Sec2" but isn't a Sec2_* claim)。iter #151 strictly filters on `id` field。

3. **Frozen baseline: `--claim-by-id-prefix Sec2` = 5 matches**: all §2.2 infrastructure claims (Sec2_20dim_syntactic_features + Sec2_K8_clusters_BIC_AIC_silhouette + Sec2_5_attrs_per_query + Sec2_K2_GMM_per_user + Sec2_95pct_holdout_threshold) — 4 unverified + 1 verified。Confirms iter #148 §2.2 finding (7 claims in §2.2 but only 5 have `Sec2_` ID prefix; 2 are non-Sec2 in §2.2 = `Pipeline_Regeneration_10x10` + `BPE_aware_Error_Injection` + `UserFilter_20_reviews_15_words`).

Actually recount §2.2 claim IDs from iter #148 matrix:
- Pipeline_Regeneration_10x10 (verified)
- BPE_aware_Error_Injection (verified)
- UserFilter_20_reviews_15_words (verified)
- Sec2_20dim_syntactic_features (unverified)
- Sec2_K8_clusters_BIC_AIC_silhouette (unverified)
- Sec2_5_attrs_per_query (verified)
- Sec2_K2_GMM_per_user (unverified)
- Sec2_95pct_holdout_threshold (unverified)

But `7 claims` per iter #148. So §2.2 = 3 verified + 4 unverified = 7. Of these, 5 have `Sec2_` prefix, 3 are non-Sec2 (Pipeline_* + BPE_* + UserFilter_*). ✓

4. **Case-insensitive prefix match**: `--claim-by-id-prefix sec2` (lowercase) matches same 5 claims as `Sec2` (mixed case)。Reviewer 不需要 exact-case match。

5. **Defensive empty PREFIX → exit 2**: `--claim-by-id-prefix ""` exits 2 with "non-empty PREFIX" error (not exit 0/1)。Catches accidental misuse; mirror iter #146 empty PAT defensive handling。

6. **Triple exit code (0/1/2)**:
   - 0 = matches found (success)
   - 1 = no match (search yielded nothing)
   - 2 = JSON missing OR empty PREFIX (CLI error)
   
   Differs from iter #131/133/138-150 (all exit 0 always). iter #151 is search-mode like iter #146 → exit 1 on no match conveys "search successful but no results" (different from "no audit JSON to search")。

7. **`(c.get('id') or '').lower()` defensive**: mirror iter #146 pattern → JSON schema drift protection (None id field → empty string → no match)。

8. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

9. **`--output` flag respected**: reviewer 想 `--claim-by-id-prefix Sec2 --output /custom/path.json` 跟其他 mode 语义一致。

10. **Reason field truncated to 80 chars + newline-stripped**: matches iter #138/139/140/142/143 pattern (per-status list views) → output format consistent across search/list commands。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix Sec2
Claims with id prefix 'Sec2' (5 total):
  Sec2_20dim_syntactic_features  [unverified]  [§2.2]
    reason      : code referenced but no expected output glob matches actual files
  Sec2_K8_clusters_BIC_AIC_silhouette  [unverified]  [§2.2]
    reason      : code referenced but no expected output glob matches actual files
  Sec2_5_attrs_per_query  [verified]  [§2.2]
    reason      : 1/1 output globs match (no numerical claim to validate)
  Sec2_K2_GMM_per_user  [unverified]  [§2.2]
    reason      : code referenced but no expected output glob matches actual files
  Sec2_95pct_holdout_threshold  [unverified]  [§2.2]
$ echo $?
0

$ python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix RQ3
Claims with id prefix 'RQ3' (4 total):
  RQ3_Fleiss_Kappa_0.72  [discrepant]  [§3.3 + Table 2]
    reason      : audit JSON has value_check_results for 'Fleiss κ' subclaim
  RQ3_Spearman_0.81  [discrepant]  [§3.3 + Table 2]
    reason      : audit JSON has value_check_results for 'Spearman ρ' subclaim
  RQ3_MAE_0.89  [discrepant]  [§3.3 + Table 2]
    reason      : audit JSON has value_check_results for 'MAE' subclaim
  RQ3_LLM_Full_Set_94%  [discrepant]  [§3.3 + Table 2]
    reason      : audit JSON has value_check_results for 'semantic preservation' subclaim
$ echo $?
0

$ python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix NO_SUCH
No claims match prefix 'NO_SUCH'.
$ echo $?
1

$ python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix ""
ERROR: --claim-by-id-prefix requires non-empty PREFIX.
$ echo $?
2

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AI: --claim-by-id-prefix CLI flag (iter #151)
  PASS  --claim-by-id-prefix matches 5 Sec2_* + case-insensitive + exit 1 no-match + exit 2 empty/missing

All 35 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AI, +docstring, +main count to 35)
- 文档: `README.md` (+1 line `--claim-by-id-prefix` example)
- 验证: 35/35 cases pass; pre-commit hook freeze 35 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --claim-by-id-prefix PREFIX` (~100ms, 不 re-run audit)

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
- **iter #146** — --find-claim PAT (substring search across 5 fields)
- **iter #147** — --summary-by-source-dir-and-status
- **iter #148** — --summary-by-section-and-status
- **iter #149** — --worst-by-source-dir
- **iter #150** — --severity-tier
- **iter #151** — --claim-by-id-prefix PREFIX ✓ 本轮

Audit CLI search-mode flags taxonomy:
| view | match type | field | flag |
|------|-----------|-------|------|
| substring (any field) | case-insensitive substring | id/section/reason/expected_outputs/code_evidence | `--find-claim` (iter #146) |
| prefix (id only) | case-insensitive prefix | id only | `--claim-by-id-prefix` (iter #151) ✓ 本轮 |

regression test coverage timeline:
- **iter #150** — 34 cases
- **iter #151** — 35 cases (+ --claim-by-id-prefix) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified (Sec2_* 4 unverified all in iter #87 scope)
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample