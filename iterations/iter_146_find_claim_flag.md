# Iteration #146 — Audit CLI `--find-claim PAT` flag (substring search across fields)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--find-claim PAT` argparse flag (case-insensitive substring search across claim id / section / reason / expected_outputs / code_evidence, with matched_in indicator); extend `_smoke_audit_regression.py` to 30 cases (Case AD)
**prior**: All existing flags are list-mode (--list-*) or aggregate-mode (--count-*, --by-section, --worst-by-section). Reviewer 想要 ad-hoc query "show me all claims referencing GMM" 或 "which claims touch Stage 02 writing analysis?" 当前 must `jq` + manual field filter。Search-mode affordance 是 fundamentally new dimension — lets reviewer follow a single concept (code path, paper concept, file basename) across the entire audit JSON。

## §A 审稿意见

Reviewer 想要 investigate a specific concept (e.g. "GMM" — appears in Stage 10 complexity analysis + Stage 12 GMM/Bayesian clause features) 当前 must:
1. `jq '.claims[] | select(.id | test("GMM"; "i"))'` (id-only search, misses code_evidence refs to gmm_per_user.py)
2. `jq '.claims[] | select(.code_evidence[]? | test("GMM"; "i"))'` (code_evidence only, misses id matches)
3. Manual grep across multiple fields

后果: ad-hoc concept search requires multi-step jq + reviewer remembers which fields to scan。Search mode 不 exist → reviewer 信息 finding 效率 low。

iter #146 fix: 新增 `--find-claim PAT` CLI flag — case-insensitive substring search across 5 fields (id / section / reason / expected_outputs / code_evidence),print compact per-claim block: `  <id>  [<status>]  [<section>]` + `    matched_in: <comma-joined field indicators>`。Triple exit code 0 (match found) / 1 (no match) / 2 (missing JSON OR empty PAT)。`break` after first match per field category 避免 `matched_in: code_evidence, code_evidence` for claims with multiple GMM refs。

## §B 本轮 (iter #146) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--find-claim PAT` flag:
```python
parser.add_argument("--find-claim", default=None, metavar="PAT",
                    help="(iter #146) Substring search (case-insensitive) across claim id, section, "
                         "reason, expected_outputs, and code_evidence. Exit 0 if any match found, "
                         "exit 1 if no match, exit 2 if JSON missing or PAT empty.")
```

docstring 加 flag description:
```
--find-claim PAT   Substring search (case-insensitive) across claim id, section, reason,
                   expected_outputs, and code_evidence. Exit 0 if any match found, exit 1 if no
                   match, exit 2 if JSON missing or PAT empty. Iter #146.
```

`main()` handler (在 `--worst-by-section` handler 之后, `--csv` handler 之前):
```python
if args.find_claim is not None:
    # iter #146: --find-claim PAT — substring search across id/section/reason/expected_outputs/code_evidence.
    pattern = args.find_claim
    if not pattern:
        print("ERROR: --find-claim requires a non-empty PAT argument.", file=__import__("sys").stderr)
        return 2
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    pat_lower = pattern.lower()
    matches = []
    for c in claims_data:
        found_in = []
        if pat_lower in (c.get("id") or "").lower():
            found_in.append("id")
        if pat_lower in (c.get("section") or "").lower():
            found_in.append("section")
        if pat_lower in (c.get("reason") or "").lower():
            found_in.append("reason")
        for gl in c.get("expected_outputs", []) or []:
            if pat_lower in (gl or "").lower():
                found_in.append("expected_outputs")
                break  # one indicator per field category
        for ref in c.get("code_evidence", []) or []:
            if pat_lower in (ref or "").lower():
                found_in.append("code_evidence")
                break  # one indicator per field category
        if found_in:
            matches.append((c, found_in))
    if not matches:
        print(f"No claims match pattern: {pattern}")
        return 1
    print(f"Claims matching '{pattern}' ({len(matches)} total):")
    for c, found_in in matches:
        cid = c.get("id", "?")
        status = c.get("status", "?")
        sec = c.get("section", "?")
        print(f"  {cid}  [{status:11s}]  [{sec}]")
        print(f"    matched_in: {', '.join(found_in)}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case AD freeze iter #146 behavior:
- `--find-claim gmm` exit 0 + "Claims matching 'gmm' (3 total)" header (frozen baseline: 3 GMM matches)
- All 3 frozen matching IDs: `RQ4_GMM_Best_Prior`, `Sec2_K2_GMM_per_user`, `Sec2_K8_clusters_BIC_AIC_silhouette`
- `matched_in:` field label visible
- `code_evidence` matched_in indicator present (cross-field search: `Sec2_K8_clusters_BIC_AIC_silhouette` matches via `gmm_per_user.py` in code_evidence)
- `unverified` status present (all 3 GMM claims are Stage 10/12 → unverified)
- `--find-claim NO_SUCH_PATTERN_XYZ` exit 1 + "No claims match pattern" message
- `--find-claim ""` exit 2 + "non-empty" stderr error
- `--find-claim gmm --output /tmp/_does_not_exist.json` exit 2 + "not found" stderr

docstring + main() 同步更新到 30 cases.

### README.md

Re-run commands section +1 line:
```bash
# Substring search across id / section / reason / expected_outputs / code_evidence; iter #146
python3 PersoanlQuery/paper_claims_audit.py --find-claim GMM
```

## §C 关键改动点

1. **New search-mode axis (vs list-mode / aggregate-mode)**: 与 existing flags orthogonal:
   - list-mode (iter #138-143) — filter by status, return matching claims
   - aggregate-mode (iter #144-145) — group claims by dir/section, return counts/worst
   - search-mode (iter #146) — substring match across fields, return matching claims ✓ 本轮
   
   Reviewer mental model: "what claims reference concept X?" — single search affordance vs 5-step jq script。

2. **5 search fields**: id (claim_id), section (paper section label), reason (audit reason text), expected_outputs (file globs), code_evidence (script:function refs)。Cross-field search enables finding claims that mention concept in code_evidence path but not in id (e.g. `Sec2_K8_clusters_BIC_AIC_silhouette` matches `gmm` via `10_complexity_analysis/.../gmm_per_user.py` in code_evidence, not via id)。

3. **`break` after first match per field category**: 防止 `matched_in: code_evidence, code_evidence, code_evidence` 当 a claim references multiple files matching pattern (e.g. Sec2_K2_GMM_per_user has multiple GMM refs in code_evidence)。Single indicator per field → cleaner output。

4. **`pat_lower = pattern.lower()` + `.lower()` on each field**: 全 case-insensitive。`--find-claim gmm` 和 `--find-claim GMM` 和 `--find-claim Gmm` 全部 match same claims。

5. **`matched_in` indicator with 5 field categories**: Reviewer sees "this claim matches because of X field" → diagnostic hint。例如 `RQ4_GMM_Best_Prior  [unverified]  [§3.4 + Table 3]` + `matched_in: id` → id is the source of match. `Sec2_K8_clusters_BIC_AIC_silhouette  [unverified]  [§2.2]` + `matched_in: code_evidence` → the id 不 contain GMM but the code ref does → reviewer knows to investigate Stage 10 GMM code。

6. **Triple exit code 0/1/2**: 与 iter #138-139 pattern 一致 — 0 (match found = success), 1 (no match = different from missing JSON), 2 (missing JSON OR empty PAT)。Empty PAT exit 2 (not 0 or 1) → defensive against `--find-claim ""` accidentally returning all 17 claims as "match everything"。

7. **Pattern NOT a regex (literal substring)**: Matches reviewer expectation from `grep` / browser search。`--find-claim .` 不 match everything (only claims with literal `.` character).If we wanted regex, would need `--find-claim-regex` separate flag → out of scope。

8. **`is not None` guard (not truthy check)**: 空 PAT 单独 exit 2 (with explicit error message) rather than silently returning no matches。`if not pattern:` catches empty string after argparse accepts `--find-claim ""`。

9. **`(c.get('id') or '')` defensive None**: iter #137 defensive None-handling pattern。JSON schema drift protection — claim without id field → treat as empty string, no crash。

10. **`--output` flag respected**: reviewer 想 `--find-claim GMM --output /custom/path.json` 跟其他 mode 语义一致。

11. **No re-run audit**: `--find-claim PAT` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

12. **Reuses iter #112 `code_evidence` field + iter #110 `reason` field**: 不需 re-implement — iter #112 audit schema 加 code_evidence list with script:function refs;iter #110 audit script 加 reason text explaining each claim's audit outcome;`--find-claim` 是 shell wrapper searching same fields。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --find-claim gmm
Claims matching 'gmm' (3 total):
  RQ4_GMM_Best_Prior  [unverified ]  [§3.4 + Table 3]
    matched_in: id
  Sec2_K8_clusters_BIC_AIC_silhouette  [unverified ]  [§2.2]
    matched_in: code_evidence
  Sec2_K2_GMM_per_user  [unverified ]  [§2.2]
    matched_in: id, code_evidence
$ echo $?
0

$ python3 PersoanlQuery/paper_claims_audit.py --find-claim NO_SUCH_PATTERN_XYZ
No claims match pattern: NO_SUCH_PATTERN_XYZ
$ echo $?
1

$ python3 PersoanlQuery/paper_claims_audit.py --find-claim ""
ERROR: --find-claim requires a non-empty PAT argument.
$ echo $?
2

$ python3 PersoanlQuery/paper_claims_audit.py --find-claim gmm --output /tmp/_does_not_exist_audit.json
ERROR: audit JSON not found at /tmp/_does_not_exist_audit.json; run audit first.
$ echo $?
2

$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
...
Case AD: --find-claim CLI flag (iter #146)
  PASS  --find-claim substring search with 3 gmm matches + exits 0/1/2/2

All 30 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~45 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case AD, +docstring, +main count to 30)
- 文档: `README.md` (+1 line `--find-claim` example)
- 验证: 30/30 cases pass; pre-commit hook freeze 30 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --find-claim GMM` (~100ms, 不 re-run audit)

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
- **iter #146** — --find-claim PAT (substring search) ✓ 本轮

Audit CLI affordance axes covered:
| axis | flag |
|------|------|
| per-claim | `--claim-id` (iter #104) |
| status global | `--status-summary` (iter #131) |
| worst-N | `--top` (iter #133) |
| per-section (count) | `--by-section` (iter #134) |
| per-section (worst) | `--worst-by-section` (iter #145) |
| per-source-dir | `--count-by-codebase-dir` (iter #144) |
| temporal | `--audit-age` (iter #135) |
| shareable format (md) | `--md-table` (iter #137) |
| shareable format (csv) | `--csv` (iter #141) |
| list by status | `--list-{degenerate,unverified,partial,verified,discrepant}` (iter #138-143) |
| search by substring | `--find-claim` (iter #146) ✓ 本轮 |
| flip detection | `--diff` (iter #114) |

Audit CLI mode taxonomy:
| mode | flags |
|------|-------|
| list-mode (filter by status) | `--list-*` (iter #138-143) |
| aggregate-mode (group by axis) | `--by-section`, `--count-by-codebase-dir`, `--worst-by-section` (iter #134, #144, #145) |
| search-mode (substring across fields) | `--find-claim` (iter #146) ✓ 本轮 |

regression test coverage timeline:
- **iter #145** — 29 cases
- **iter #146** — 30 cases (+ --find-claim) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
