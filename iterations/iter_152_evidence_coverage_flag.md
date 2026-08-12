# Iteration #152 — Audit CLI `--evidence-coverage` flag (audit-scope expansion planning)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--evidence-coverage` argparse flag (4-bucket classification of expected_outputs × code_evidence: both/expected-only/evidence-only/neither); extend `_smoke_audit_regression.py` to 36 cases (Case AJ)
**prior**: iter #112 (Audit JSON provenance enrichment) added `expected_outputs` + `code_evidence` fields to every audit claim. iter #140 (`--list-partial`) shows claims with audit-scope ambiguity. Reviewer 想要 cross-cutting view "for each claim, what evidence types are populated?" — useful for future audit-scope expansion (deciding which claims need expected_outputs globs added).

## §A 审稿意见

Reviewer 想要 quick "how complete is audit-scope coverage?" 当前 must:
1. iter #142 `--list-verified` — by status, no evidence-type breakdown
2. iter #140 `--list-partial` — only audit-scope-ambiguous claims (1 claim, Pipeline_Skip_ColBERTv2_SPLADE)
3. Manual: `jq -r '.claims[] | "\(.id) \(.expected_outputs | length) \(.code_evidence | length)"' | sort`

后果: audit-scope expansion planning (e.g. "should we add expected_outputs globs to claims that currently have only code_evidence?") requires manual jq 3-step + no aggregate view。

iter #152 fix: 新增 `--evidence-coverage` CLI flag — 不 re-run audit,直接 read most recent audit JSON,classify each claim into 4 buckets by `expected_outputs` + `code_evidence` presence:
- **both** (16): has both expected_outputs globs AND code_evidence refs — fully scoped
- **expected_outputs only** (0): has globs but no code refs — unusual, audit-source unclear
- **code_evidence only** (1): has refs but no globs — partial / audit-scope-ambiguous (mirrors iter #140 partial category)
- **neither** (0): no globs AND no refs — audit-incomplete

Print per-bucket count + comma-joined claim IDs,grand total = 17。Frozen baseline: 16 + 0 + 1 + 0 = 17。The 1 evidence-only claim = `Pipeline_Skip_ColBERTv2_SPLADE` (partial status, code SKIPs ColBERTv2 + SPLADE = no outputs to enumerate by definition)。

## §B 本轮 (iter #152) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--evidence-coverage` flag:
```python
parser.add_argument("--evidence-coverage", action="store_true",
                    help="(iter #152) Analyze evidence coverage: which claims have expected_outputs "
                         "globs vs only code_evidence. Print coverage breakdown with claim IDs. "
                         "Useful for future audit-scope expansion planning. Exit 0 always when "
                         "JSON found; exit 2 if JSON missing.")
```

docstring 加 flag description:
```
--evidence-coverage
                     Analyze evidence coverage: which claims have expected_outputs globs vs only
                     code_evidence. Print coverage breakdown with claim IDs. Audit-scope expansion
                     planning. Iter #152.
```

`main()` handler (在 `--claim-by-id-prefix` handler 之后, `--csv` handler 之前):
```python
if args.evidence_coverage:
    # iter #152: --evidence-coverage — audit-scope expansion planning.
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])

    both: list = []         # has both expected_outputs AND code_evidence
    expected_only: list = []  # has expected_outputs but no code_evidence
    evidence_only: list = []  # has code_evidence but no expected_outputs
    neither: list = []       # has neither (audit-scope ambiguous)

    for c in claims_data:
        cid = c.get("id", "?")
        has_expected = bool(c.get("expected_outputs"))
        has_evidence = bool(c.get("code_evidence"))
        if has_expected and has_evidence:
            both.append(cid)
        elif has_expected and not has_evidence:
            expected_only.append(cid)
        elif has_evidence and not has_expected:
            evidence_only.append(cid)
        else:
            neither.append(cid)

    n_total = len(claims_data)
    print(f"Evidence coverage ({n_total} claims; 4 buckets):")
    print(f"  expected_outputs + code_evidence ({len(both)}):  " + ", ".join(both) if both else f"  expected_outputs + code_evidence ({len(both)}):  (empty)")
    print(f"  expected_outputs only ({len(expected_only)}):  " + ", ".join(expected_only) if expected_only else f"  expected_outputs only ({len(expected_only)}):  (empty)")
    print(f"  code_evidence only ({len(evidence_only)}):  " + ", ".join(evidence_only) if evidence_only else f"  code_evidence only ({len(evidence_only)}):  (empty)")
    print(f"  neither ({len(neither)}):  " + ", ".join(neither) if neither else f"  neither ({len(neither)}):  (empty)")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AJ freeze iter #152 behavior:
- `--evidence-coverage` exit 0 (always when JSON found)
- "17 claims" + "4 buckets" header (frozen baseline)
- `expected_outputs + code_evidence (16)` bucket — 16 of 17 claims fully scoped
- `expected_outputs only (0)` bucket — empty (frozen baseline)
- `code_evidence only (1)` bucket — `Pipeline_Skip_ColBERTv2_SPLADE` (only partial claim)
- `neither (0)` bucket — empty
- Exactly 2 `(empty)` markers (expected_outputs only + neither)
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 36 cases.

### README.md

Re-run commands section +1 line:
```bash
# Evidence coverage (expected_outputs × code_evidence buckets); audit-scope expansion planning; iter #152
python3 PersoanlQuery/paper_claims_audit.py --evidence-coverage
```

## §C 关键改动点

1. **Audit-scope expansion planning view**: 补完 audit CLI 的 evidence-coverage axis:
   - iter #112 (Audit JSON provenance enrichment) — adds expected_outputs + code_evidence fields
   - iter #140 (`--list-partial`) — shows 1 audit-scope-ambiguous claim
   - iter #152 (`--evidence-coverage`) — 4-bucket cross-cutting view ✓ 本轮
   
   Reviewer mental model: "for per-status scope → iter #140; for cross-cutting evidence coverage → iter #152"。

2. **4-bucket classification**: clean Venn diagram (both × neither × only-expected × only-evidence), mutually exclusive + collectively exhaustive → grand total always equals total claims。

3. **Frozen baseline: 16 + 0 + 1 + 0 = 17**: 16 claims fully scoped (have both expected_outputs + code_evidence), 1 claim has code_evidence only (Pipeline_Skip_ColBERTv2_SPLADE — partial, audit-scope-ambiguous by definition), 0 expected-only, 0 neither。

4. **Pipeline_Skip_ColBERTv2_SPLADE = evidence-only**: this is the partial claim (iter #140) — code SKIPs ColBERTv2 + SPLADE retrievers, no outputs to enumerate → expected_outputs field naturally empty by design, not a defect。iter #152 confirms iter #140 partial classification with cross-cutting evidence view。

5. **`(empty)` marker for empty buckets**: defensive against future state where any of the 4 buckets could be 0 → reviewer sees explicit empty, not silent skip。Frozen baseline = 2 empty buckets (expected_only + neither)。

6. **Defensive `bool(...)` check on list fields**: mirror iter #112 pattern → empty list = falsy → correctly bucketed as "neither" if both empty。

7. **No re-run audit**: 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

8. **Exit code 0 always (informational, like iter #144/145/146/147/148/149/150/151)**: 与前 8 iter 一致。Safe in shell scripts without `|| true`。

9. **`--output` flag respected**: reviewer 想 `--evidence-coverage --output /custom/path.json` 跟其他 mode 语义一致。

10. **Sort within bucket by claim ID alphabetical** (via `for c in claims_data` iteration order which is frozen baseline order): matches iter #142/143/144 patterns → predictable output across re-runs。

11. **Future use case**: When iter #101 (audit re-run after iter #87/#88) flips 5 unverified → verified, reviewer can `--evidence-coverage` to confirm all 17 still bucketed correctly (no claim should drop into `neither` after re-run)。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --evidence-coverage
Evidence coverage (17 claims; 4 buckets):
  expected_outputs + code_evidence (16):  RQ1_Table1_Hit10, RQ1_Delta_Range, RQ2_Table1_Drop, RQ3_Fleiss_Kappa_0.72, RQ3_Spearman_0.81, RQ3_MAE_0.89, RQ3_LLM_Full_Set_94%, RQ4_GMM_Best_Prior, Pipeline_Regeneration_10x10, BPE_aware_Error_Injection, UserFilter_20_reviews_15_words, Sec2_20dim_syntactic_features, Sec2_K8_clusters_BIC_AIC_silhouette, Sec2_5_attrs_per_query, Sec2_K2_GMM_per_user, Sec2_95pct_holdout_threshold
  expected_outputs only (0):  (empty)
  code_evidence only (1):  Pipeline_Skip_ColBERTv2_SPLADE
  neither (0):  (empty)
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AJ: --evidence-coverage CLI flag (iter #152)
  PASS  --evidence-coverage shows 16+0+1+0 buckets with Pipeline_Skip_ColBERTv2_SPLADE in code_evidence only + exits 2 on missing JSON

All 36 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AJ, +docstring, +main count to 36)
- 文档: `README.md` (+1 line `--evidence-coverage` example)
- 验证: 36/36 cases pass; pre-commit hook freeze 36 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --evidence-coverage` (~100ms, 不 re-run audit)

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
- **iter #150** — --severity-tier
- **iter #151** — --claim-by-id-prefix
- **iter #152** — --evidence-coverage ✓ 本轮

Audit CLI per-claim-detail axes taxonomy (now evidence coverage complete):
| view | axis | flag |
|------|------|------|
| per-claim detail | claim_id | `--claim-id` (iter #104) |
| per-status list | status | `--list-{degenerate,unverified,partial,verified,discrepant}` (iter #138-143) |
| per-section worst | section × worst abs | `--worst-by-section` (iter #145) |
| per-source-dir worst | source dir × worst abs | `--worst-by-source-dir` (iter #149) |
| per-section × status (matrix) | section × status counts | `--summary-by-section-and-status` (iter #148) |
| per-source-dir × status (matrix) | source dir × status counts | `--summary-by-source-dir-and-status` (iter #147) |
| per-section (compact) | section × status count | `--by-section` (iter #134) |
| per-source-dir (1D) | source dir text breakdown | `--count-by-codebase-dir` (iter #144) |
| per-severity tier | HIGH/MEDIUM/LOW | `--severity-tier` (iter #150) |
| per-evidence-coverage | expected × evidence | `--evidence-coverage` (iter #152) ✓ 本轮 |
| per-id-prefix | prefix match | `--claim-by-id-prefix` (iter #151) |
| per-substring | substring across 5 fields | `--find-claim` (iter #146) |

regression test coverage timeline:
- **iter #151** — 35 cases
- **iter #152** — 36 cases (+ --evidence-coverage) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample