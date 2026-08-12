# Iteration #139 — Audit CLI `--list-unverified` flag (planning aid for iter #87 pilot)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--list-unverified` argparse flag (prints unverified claims with id + section + code_evidence); extend `_smoke_audit_regression.py` to 23 cases (Case W)
**prior**: unverified status means "code referenced but no expected output glob matches actual files" (e.g. Stage 12 lineage gap — code expects Stage 12 latent representations but Stage 12 GPU run never happened, so fallback paths produce output but globs don't match). Reviewer wanting to plan iter #87 Stage 12 + Stage 10 pilot had to grep audit JSON or run `--claim-id` 5 times. Iter #138 added `--list-degenerate` for the value-extraction failure axis; this iter covers the code-reference-but-no-output axis.

## §A 审稿意见

Reviewer 想 plan iter #87 Stage 12 + Stage 10 pilot (9-21h GPU) 必须 know:

1. Which 5 unverified claims exist
2. Each claim's paper section (§2.2 vs §3.4)
3. Each claim's code_evidence — which scripts implement the claim (so reviewer knows which files to inspect)

当前 must:
1. `jq '.claims[] | select(.status == "unverified") | {id, section, code_evidence}' paper_claims_audit.json`
2. Or `--claim-id <id>` × 5 iterations, each printing full value_check_results detail

后果: planning cycle ~5 min/iter for code-reference triage。Common case: pre-iter #87 pilot, reviewer 想 "which 5 scripts do I need to confirm Stage 12 outputs feed into?"

iter #139 fix: 新增 `--list-unverified` CLI flag — 不 re-run audit,直接 read most recent audit JSON,filter `status == "unverified"`,print compact per-claim block with id + section + code_evidence bullet list。Mirror of iter #138 pattern (`--list-degenerate`) but with `code_evidence` field instead of `matched_files`.

```
Unverified claims (5 total):
  RQ4_GMM_Best_Prior  [§3.4 + Table 3]
    reason      : code referenced but no expected output glob matches actual files
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:945-968 (...)
      - PersoanlQuery/10_complexity_analysis/common/extract_clause_features_single_query.py (...)
  Sec2_20dim_syntactic_features  [§2.2]
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/extract_clause_features_single_query.py:167-188 (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:44 (LATENT_DIM=20)
  ...
```

Reviewer 一眼看出: 5 unverified claims,4 in §2.2 (Sec2_*) + 1 in §3.4 (RQ4_GMM_Best_Prior),all reference `10_complexity_analysis/common/` scripts → iter #87 pilot scope is Stage 10 + Stage 12 line。

## §B 本轮 (iter #139) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--list-unverified` flag:
```python
parser.add_argument("--list-unverified", action="store_true",
                    help="(iter #139) Print just the 'unverified' claims with id, section, and code_evidence "
                         "(one bullet per script reference). Exit 0 if any unverified found, exit 1 if none, "
                         "exit 2 if JSON missing. Useful for planning iter #87/#88 stage re-runs.")
```

docstring 加 flag description:
```
--list-unverified Print just the 'unverified' claims with id, section, and code_evidence
                   (one bullet per script reference). Exit 0 if any unverified, exit 1 if none.
                   Useful for planning iter #87/#88 stage re-runs. Iter #139.
```

`main()` handler (在 `--list-degenerate` handler 之后):
```python
if args.list_unverified:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    unv = [c for c in claims_data if c.get("status") == "unverified"]
    if not unv:
        print("No unverified claims. All code-referenced claims have matching output files.")
        return 1
    print(f"Unverified claims ({len(unv)} total):")
    for c in unv:
        cid = c.get("id", "?")
        sec = c.get("section", "?")
        reason = (c.get("reason") or "")[:80]
        code_ev = c.get("code_evidence", []) or []
        print(f"  {cid}  [{sec}]")
        print(f"    reason      : {reason}")
        if code_ev:
            print(f"    code_evidence:")
            for ev in code_ev:
                print(f"      - {ev}")
        else:
            print(f"    code_evidence: (none recorded)")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case W freeze iter #139 behavior:
- `--list-unverified` exit 0 (5 unverified in frozen baseline)
- "Unverified claims" header
- All 5 expected unverified claim IDs present (`RQ4_GMM_Best_Prior`, `Sec2_20dim_syntactic_features`, `Sec2_K2_GMM_per_user`, `Sec2_K8_clusters_BIC_AIC_silhouette`, `Sec2_95pct_holdout_threshold`)
- `code_evidence` field visible + ≥1 `PersoanlQuery/10_complexity_analysis/common/` path
- Missing JSON via `--output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 23 cases。

### README.md

Re-run commands section +1 line:
```bash
# List just the 'unverified' claims with id, section, and code_evidence for iter #87 planning; iter #139
python3 PersoanlQuery/paper_claims_audit.py --list-unverified
```

## §C 关键改动点

1. **No re-run audit**: `--list-unverified` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **Triple exit code (mirror iter #138)**:
   - `0` (unverified found) — actionable, plan iter #87/#88
   - `1` (no unverified) — all code-referenced claims have outputs (post-iter #87 ideal state)
   - `2` (JSON missing) — distinct from "no unverified"

3. **`(none recorded)` fallback**: if a claim has no `code_evidence` field (e.g. older audit JSON before iter #112 added provenance),print clear placeholder instead of empty list — reviewer knows data is missing not claim has 0 evidence。

4. **Bullet list for code_evidence**: 每个 script reference 前缀 `      - ` (6 spaces + dash + space) → 视觉对齐,可直接 copy-paste to planning doc。

5. **Reason + code_evidence both shown**: 完整 diagnostic block — `reason` 说 "no expected output glob matches" (症状), `code_evidence` 说 "which scripts implement" (root cause candidates)。Reviewer 把两个 fields 一起 read 可 diagnose "Stage 12 缺失 → glob 不 match"。

6. **Reuses iter #112 `code_evidence` field**: 不需 re-implement — iter #112 dashboard 加了 provenance panels 包含 code_evidence;`--list-unverified` 是 shell wrapper 读 same field。

7. **`--output` flag respected**: reviewer 想 `--list-unverified --output /custom/path.json` 跟其他 mode 语义一致。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --list-unverified
Unverified claims (5 total):
  RQ4_GMM_Best_Prior  [§3.4 + Table 3]
    reason      : code referenced but no expected output glob matches actual files
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:945-968 (...)
      - PersoanlQuery/10_complexity_analysis/common/extract_clause_features_single_query.py (...)
  Sec2_20dim_syntactic_features  [§2.2]
    reason      : code referenced but no expected output glob matches actual files
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/extract_clause_features_single_query.py:167-188 (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:44 (LATENT_DIM=20)
  Sec2_K8_clusters_BIC_AIC_silhouette  [§2.2]
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py:24 (...)
      - PersoanlQuery/10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py:134-174 (...)
  Sec2_K2_GMM_per_user  [§2.2]
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:82 (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:536-563 (...)
  Sec2_95pct_holdout_threshold  [§2.2]
    code_evidence:
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:57 (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:1803 (...)
      - PersoanlQuery/10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:1909-1910 (...)
$ echo $?
0

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'                                PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965               PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message                                 PASS
Case D: --strict --json-only (full audit) expects exit 1                                         PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)                             PASS
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)                        PASS
Case G: dashboard HTML has anchors + status legend (iter #121)                                   PASS
Case H: dashboard HTML <title> + <meta name='description'> (iter #122)                           PASS
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123, #126, #127)               PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                        PASS
Case K: dashboard <link rel=alternate> -> paper_audit_id_mapping.json (iter #125)                PASS
Case L: mapping sidecar reverse_section_index (iter #128)                                        PASS
Case M: dashboard reverse-section panel (iter #129)                                              PASS
Case N: --status-summary compact 1-line (iter #131)                                              PASS
Case O: dashboard status_summary_table panel (iter #132)                                         PASS
Case P: --top N CLI flag (iter #133)                                                             PASS
Case Q: dashboard sort-by-severity toggle (iter #133)                                            PASS
Case R: --by-section CLI flag (iter #134)                                                        PASS
Case S: --audit-age CLI flag (iter #135)                                                         PASS
Case T: dashboard filter-by-status (iter #136)                                                   PASS
Case U: --md-table CLI flag (iter #137)                                                          PASS
Case V: --list-degenerate CLI flag (iter #138)                                                   PASS
Case W: --list-unverified CLI flag (iter #139)                                                   PASS

All 23 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~35 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case W, +docstring, +main count to 23)
- 文档: `README.md` (+1 line `--list-unverified` example)
- 验证: 23/23 cases pass; pre-commit hook freeze 23 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --list-unverified` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary
- **iter #133** — --top N
- **iter #134** — --by-section
- **iter #135** — --audit-age
- **iter #137** — --md-table
- **iter #138** — --list-degenerate (degenerate diagnosis)
- **iter #139** — --list-unverified (unverified planning aid) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section | `--by-section` (iter #134) |
| temporal | `--audit-age` (iter #135) |
| shareable format | `--md-table` (iter #137) |
| degenerate diagnosis | `--list-degenerate` (iter #138) |
| unverified planning | `--list-unverified` (iter #139) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

regression test coverage timeline:
- **iter #138** — 22 cases
- **iter #139** — 23 cases (+ --list-unverified) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample