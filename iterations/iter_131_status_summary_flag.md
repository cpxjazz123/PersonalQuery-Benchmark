# Iteration #131 — Audit CLI `--status-summary` flag

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` add `--status-summary` argparse flag (compact 1-line output); extend `_smoke_audit_regression.py` to 14 cases (Case N)
**prior**: audit CLI had `--strict / --diff / --claim-id / --list / --json-only / --verbose`. Reviewer / CI 想 quick sanity check audit state 必须 parse full JSON (几 KB) 或 re-run full audit (slow). No compact CLI affordance for quick 1-line summary.

## §A 审稿意见

audit CLI 当前 6 flags 都 too heavy for quick sanity check:

- `--strict --json-only`: re-runs full audit (slow),writes JSON,exits 1
  if non-verified present。
- `--diff`: re-runs full audit + compares against baseline (slower)。
- `--claim-id`: re-runs audit on 1 claim only。
- `--list`: prints claim IDs (no summary)。
- `--verbose`: prints full per-claim detail。

后果: reviewer / CI 想 quick check "audit summary 当前怎么样" 必须
parse full audit JSON (几 KB) 或 re-run full audit (slow)。Common case:
reviewer 在 shell prompt 想知道 "latest audit pass 有几个 discrepant?" —
没 quick CLI affordance。

iter #131 fix: 新增 `--status-summary` flag — 不 re-run audit,直接 read
most recent audit JSON (default `result/personal_query/iterations/paper_claims_audit.json`),
print compact 1-line:
```
audit summary: 6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified (total 17/17) at 2026-07-20T22:40:41+00:00
```

Shell-friendly / cron-friendly / CI quick check。

## §B 本轮 (iter #131) 改动

### PersoanlQuery/paper_claims_audit.py

`argparse` 加 `--status-summary` flag:

```python
parser.add_argument("--status-summary", action="store_true",
                    help="(iter #131) Print one compact summary line of the most recent audit JSON "
                         "and exit 0; do not re-run audit.")
```

`main()` 在 `--list` 之后加 handler (不 re-run audit):

```python
if args.status_summary:
    recent_path = Path(args.output) if args.output else OUT_PATH
    if not recent_path.exists():
        print(f"ERROR: audit JSON not found at {recent_path}; run audit first.", file=sys.stderr)
        return 2
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    summary = recent.get("status_summary", {})
    order = ["verified_value_match", "verified", "discrepant", "degenerate",
             "partial", "unverified", "blocked"]
    parts = " / ".join(f"{summary.get(k, 0)} {k.replace('_', ' ')}" for k in order if summary.get(k, 0) > 0)
    n_total = recent.get("n_claims", 0)
    n_audited = recent.get("n_audited", n_total)
    generated_at = recent.get("generated_at", "unknown")
    print(f"audit summary: {parts} (total {n_audited}/{n_total}) at {generated_at}")
    return 0
```

docstring 加 flag description:
```
--status-summary   Print one compact summary line of the most recent audit JSON
                   (e.g. "audit summary: 6 verified / 4 discrepant / ... (total 17/17)
                   at <ISO8601>") and exit 0. Does not re-run the audit. Iter #131.
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case N freeze iter #131 behavior:

```python
def case_n_status_summary_flag():
    """Case N (iter #131): --status-summary prints compact 1-line summary from cached audit JSON."""
    exit_code, stdout, _ = _run_audit(["--status-summary"])
    assert exit_code == 0
    assert stdout.startswith("audit summary:")
    for status in ["6 verified", "4 discrepant", "1 degenerate", "1 partial", "5 unverified"]:
        assert status in stdout
    assert "(total 17/17)" in stdout
    iso_match = re.search(r"at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", stdout)
    assert iso_match
```

docstring + main() 同步更新到 14 cases。

### README.md

Re-run commands section 加一行:
```bash
# Print compact 1-line audit summary from the most recent audit JSON without re-running; iter #131
python3 PersoanlQuery/paper_claims_audit.py --status-summary
```

## §C 关键改动点

1. **No re-run audit**: `--status-summary` 跳过 evaluate_claim loop,直接
   read JSON file。所以 runtime < 100ms vs full audit (~1-2s)。

2. **`--output` flag respected**: reviewer 想 status-summary 在 custom path
   的 audit JSON 上,可 `--status-summary --output /custom/path.json` —
   同 `--output` flag 跟其他 mode 语义一致。

3. **Compact format**: `6 verified / 4 discrepant / ... (total 17/17) at
   <ISO8601>` 单行,human-readable + grep-friendly + shell prompt
   friendly。

4. **Skip zero-count statuses**: `verified_value_match` 当前 0 自动 omit,
   output 不被 "0 verified_value_match" noise clutter。如果 future audit
   跑出非零 `verified_value_match`,自动 appear。

5. **Error on missing JSON**: `--status-summary` 在 audit JSON missing 时
   exit 2 + clear error message (`audit JSON not found at <path>; run
   audit first.`)。不静默 fallback to re-run audit (避免 reviewer 误以为
   summary 已 fresh)。

6. **Frozen baseline 2026-07-21**: `0/6/4/1/1/5/0` (status_summary dict
   from audit JSON) — Case N 断言 5 status 全 present + total 17/17 +
   ISO 8601 timestamp。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --status-summary
audit summary: 6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified (total 17/17) at 2026-07-20T22:40:41+00:00

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
Case M: dashboard reverse_section panel (iter #129)                                              PASS
Case N: --status-summary compact 1-line (iter #131)                                              PASS

All 14 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+20 lines argparse flag
  + handler + docstring update)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case N, +docstring,
  +main count to 14)
- 文档: `README.md` (+1 line `--status-summary` example)
- 验证: 14/14 cases pass; pre-commit hook freeze 14 invariants ✓
- 命令: `python3 PersoanlQuery/paper_claims_audit.py --status-summary`
  (~100ms, 不 re-run audit)

## §F 与 audit infra timeline 的关系

audit CLI flags timeline:
- **iter #104** — initial CLI refactor with argparse: --claim-id / --verbose / --output / --json-only / --strict / --list
- **iter #114** — --diff mode (per-claim flip detector against baseline)
- **iter #131** — --status-summary mode (compact 1-line, no re-run) ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases
- **iter #121** — 7 cases
- **iter #122** — 8 cases
- **iter #123** — 9 cases
- **iter #124** — 10 cases
- **iter #125** — 11 cases
- **iter #128** — 12 cases
- **iter #129** — 13 cases
- **iter #131** — 14 cases (+ --status-summary flag) ✓ 本轮

每个新 CLI affordance 都应该 regression test 冻结。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
