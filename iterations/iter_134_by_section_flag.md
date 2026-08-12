# Iteration #134 — Audit CLI `--by-section` flag (paper section × status matrix)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--by-section` argparse flag (matrix of paper section × audit status counts); extend `_smoke_audit_regression.py` to 18 cases (Case R)
**prior**: dashboard had a per-section breakdown table (iter #117, gray `#f5f5f5`) but no equivalent CLI affordance. Reviewer triaging which paper section was flakiest had to open dashboard HTML or manually aggregate the audit JSON. Combined with iter #131 (`--status-summary`) and iter #133 (`--top N`), the only missing axis was "where in the paper are the issues concentrated?"

## §A 审稿意见

Dashboard `per-section breakdown` (iter #117) render 8 sections × 7 status cells — useful for browser inspection but reviewer 想 shell quick check "which §X has most discrepant?" 必须:

1. Open `paper_audit_id_mapping.json` or audit JSON
2. Filter claims by `c['section']` 
3. Count statuses per section
4. Mental sort by discrepant desc

iter #131 (`--status-summary`) 给全局 1-line,iter #133 (`--top N`) 给 worst-claim triage,缺 axis #3: **per-section concentration**。

后果: reviewer 看到 `--top` 输出 "4 discrepant in RQ3_* claims" 但不知道 "这些 4 个 claims 全部 in §3.3" 还是 "分散在 §3.3 + §3.4" — section-level concentration 不可见。

iter #134 fix: 新增 `--by-section` CLI flag — 不 re-run audit,直接 read most recent audit JSON,aggregate `claims` by `c['section']` × `c['status']` count,sort by discrepant desc (tiebreak: unverified desc, then section name asc),print matrix:

```
paper section × audit status count matrix (17 claims across 8 sections):
Section                       verified value match  verified        discrepant      degenerate      partial         unverified      blocked         total
---------------------------------------------------------------------------------------------------------------------------------------------------------
§3.3 + Table 2                0               0               4               0               0               0               0               4
§2.2                          0               3               0               0               0               4               0               7
§3.4 + Table 3                0               0               0               0               0               1               0               1
(infrastructure)              0               0               0               0               1               0               0               1
§2.1                          0               1               0               0               0               0               0               1
§3.2                          0               1               0               0               0               0               0               1
§3.2 + Table 1                0               0               0               1               0               0               0               1
§3.2 + Table 1 (lower panel)  0               1               0               0               0               0               0               1
```

Reviewer 一眼看出: §3.3 has 4/4 discrepant (concentrated risk), §2.2 has 4/7 unverified (latent risk if not re-run),其他 sections 1 claim each mostly verified。

## §B 本轮 (iter #134) 改动

### PersoanlQuery/paper_claims_audit.py

argparse 加 `--by-section` flag:
```python
parser.add_argument("--by-section", action="store_true",
                    help="(iter #134) Print paper section × audit status count matrix from the most "
                         "recent audit JSON (no re-run). Sections sorted by discrepant count desc.")
```

docstring 加 flag description:
```
--by-section     Print paper section × audit status count matrix from the most recent
                 audit JSON (no re-run); sections sorted by discrepant count desc,
                 tiebreak by unverified desc. Iter #134.
```

`main()` handler (在 `--top` handler 之后):
```python
if args.by_section:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=__import__("sys").stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    claims_data = recent.get("claims", [])
    status_keys = ["verified_value_match", "verified", "discrepant", "degenerate",
                   "partial", "unverified", "blocked"]
    matrix: dict[str, dict[str, int]] = {}
    for c in claims_data:
        sec = c.get("section") or "(unknown)"
        row = matrix.setdefault(sec, {k: 0 for k in status_keys})
        st = c.get("status", "unverified")
        row[st] = row.get(st, 0) + 1
    rows = sorted(
        matrix.items(),
        key=lambda kv: (-kv[1].get("discrepant", 0), -kv[1].get("unverified", 0), kv[0]),
    )
    col_w = max(20, max((len(sec) for sec, _ in rows), default=20))
    header = f"{'Section':<{col_w}}  " + "  ".join(f"{k.replace('_', ' '):<14}" for k in status_keys) + "  total"
    print(f"paper section × audit status count matrix ({sum(sum(r.values()) for _, r in rows)} claims across {len(rows)} sections):")
    print(header)
    print("-" * len(header))
    for sec, row in rows:
        total = sum(row.values())
        cells = "  ".join(f"{row.get(k, 0):<14}" for k in status_keys)
        print(f"{sec:<{col_w}}  {cells}  {total}")
    return 0
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case R freeze iter #134 behavior:
- `--by-section` exit 0
- Header "paper section ... audit status count matrix"
- All 7 status columns (verified_value_match / verified / discrepant / degenerate / partial / unverified / blocked) in output
- §3.3 + Table 2 row present (4 discrepant — frozen)
- §2.2 row present (7 infrastructure claims — frozen)
- "17 claims across 8 sections" totals line

docstring + main() 同步更新到 18 cases。

### README.md

Re-run commands section +1 line:
```bash
# Print paper section × audit status count matrix from the most recent audit JSON without re-running; iter #134
python3 PersoanlQuery/paper_claims_audit.py --by-section
```

## §C 关键改动点

1. **No re-run audit**: `--by-section` 跳过 evaluate_claim loop,直接 read JSON file。Runtime <100ms vs full audit (~1-2s)。

2. **Sort key**: `-discrepant desc, -unverified desc, section asc` — reviewer 关注 "worst first"。Discrepant 是 value-mismatch 显式 evidence,unverified 是 latent risk (no output yet),所以 discrepant 优先,但 unverified 是 tiebreak 因为 next-worst。

3. **Raw section labels**: dashboard iter #117 用 raw labels (`§3.3 + Table 2` vs `§3.2 + Table 1`) — iter #134 跟 dashboard 保持一致,不做 normalize (e.g. 不 merge `§3.2 + Table 1` with `§3.2`)。

4. **Column width auto**: `col_w = max(20, max(len(sec) for sec in rows))` — section label 长度可变 (`§3.2 + Table 1 (lower panel)` 是 30 chars)。Max 20 防止 short section 浪费 horizontal space。

5. **Total column at right**: 每 row 末尾 `total` 计数,reviewer 一眼看出 "§2.2 has 7 claims, §3.3 has 4 claims"。

6. **`(unknown)` fallback**: claim JSON missing `section` 字段时 fall back to `(unknown)` 而不是 crash — defensive against future audit JSON schema drift。

7. **`--output` flag respected**: reviewer 想 `--by-section --output /custom/path.json` 跟其他 mode 语义一致。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --by-section
paper section × audit status count matrix (17 claims across 8 sections):
Section                       verified value match  verified        discrepant      degenerate      partial         unverified      blocked         total
---------------------------------------------------------------------------------------------------------------------------------------------------------
§3.3 + Table 2                0               0               4               0               0               0               0               4
§2.2                          0               3               0               0               0               4               0               7
§3.4 + Table 3                0               0               0               0               0               1               0               1
(infrastructure)              0               0               0               0               1               0               0               1
§2.1                          0               1               0               0               0               0               0               1
§3.2                          0               1               0               0               0               0               0               1
§3.2 + Table 1                0               0               0               1               0               0               0               1
§3.2 + Table 1 (lower panel)  0               1               0               0               0               0               0               1

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

All 18 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+~30 lines argparse flag + handler + docstring)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case R, +docstring, +main count to 18)
- 文档: `README.md` (+1 line `--by-section` example)
- 验证: 18/18 cases pass; pre-commit hook freeze 18 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --by-section` (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode
- **iter #131** — --status-summary mode (1-line global)
- **iter #133** — --top N mode (worst-claim triage)
- **iter #134** — --by-section mode (per-section concentration) ✓ 本轮

Dashboard per-section affordances:
- **iter #117** — per-section breakdown table (8 sections × 7 status cells)
- **iter #128** — paper_audit_id_mapping.json reverse_section_index
- **iter #129** — reverse-section panel (paper section → audit claim count)
- **iter #134** — CLI --by-section (complements dashboard iter #117) ✓ 本轮

regression test coverage timeline:
- **iter #133** — 17 cases
- **iter #134** — 18 cases (+ --by-section) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample