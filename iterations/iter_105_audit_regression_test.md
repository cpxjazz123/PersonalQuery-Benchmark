# Iteration #105 — paper_claims_audit regression smoke test

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_smoke_audit_regression.py` (NEW, ~140 lines)
**prior**: iter #104 added CLI; this iter freezes the audit baseline via regression test

## §A 审稿意见

iter #104 完成 paper_claims_audit CLI refactor (6 args + exit code 0/1/2)。 但
没有 regression test 防止 audit baseline silent flip:
1. 有人误删 claim — audit 0 → -1 claim
2. 有人改 tolerance — discrepant → verified
3. Stage 12 重跑 — unverified → verified (期望变化但需 explicit 文档)
4. 有人改 evaluator code — status silent 变化

iter #105 加 regression test 冻结 baseline, 任何 silent flip 都会被 CI 抓到。

## §B 本轮 (iter #105) 改动

### §B.1 PersoanlQuery/_smoke_audit_regression.py (NEW)

4-case smoke test:
- **Case A**: `--claim-id Pipeline_Regeneration_10x10` → status `verified`
- **Case B**: `--claim-id RQ3_Fleiss_Kappa_0.72` → status `discrepant`, extracted=0.6235, abs_delta=0.0965 (frozen)
- **Case C**: `--claim-id DOES_NOT_EXIST` → exit 2 + error message in stderr
- **Case D**: `--strict --json-only` (full audit) → exit 1 + "STRICT MODE: 11 non-verified" in stderr

### §B.2 Audit script 修复 (--strict + --json-only stderr)

原 `--strict --json-only` exit 1 但没 print STRICT MODE message。 Fix: 在
`--json-only` early return path 也加 STRICT MODE stderr print, 跟非 --json-only
path 行为一致。

## §C 关键改动点

1. **Regression test freezes baseline**: extracted=0.6235, expected=0.72,
   abs_delta=0.0965 — 任何 silent drift 会 fail test
2. **Source file check**: Case B assert `source_file` ends with
   `llm_human_eval_cross_domain.json` — 防止有人改 output path 让 regression test
   失效
3. **CLI error path tested**: Case C 验证 --claim-id INVALID 真的 return exit 2
   + error message (防止 audit script 退化 silently swallow invalid id)
4. **--strict behavior tested**: Case D 验证 11 non-verified 被检测 (4 discrepant +
   1 degenerate + 1 partial + 5 unverified = 11) — 防止有人改 --strict 逻辑
   silent bypass

## §D 测试结果

```
=== paper_claims_audit regression smoke test (iter #105) ===
Audit script: /home/wlia0047/ar57/wenyu/PersoanlQuery/paper_claims_audit.py
Frozen baseline: 2026-07-21 (audit summary 0/6/4/1/1/5/0)

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'
  PASS  status=verified, reason='1/1 output globs match (no numerical claim to validate)'

Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965
  PASS  status=discrepant, extracted=0.6235, abs_delta=0.0965

Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message
  PASS  exit_code=2, stderr="ERROR: claim id 'DOES_NOT_EXIST' not found. Use --list to se"

Case D: --strict --json-only (full audit) expects exit 1
  PASS  exit_code=1, stderr='STRICT MODE: 11 non-verified claim(s) detected (exit 1)'

All 4 cases passed. Audit CLI frozen baseline verified.
```

## §E 与下游 iter 的关系

| iter | Usage |
|------|-------|
| #107 | CI hook pre-commit: `python3 PersoanlQuery/_smoke_audit_regression.py` 应该 pass |
| #87 (post Stage 12 re-run) | Regression test 需要 update baseline (5 unverified → verified) |
| #88 (post Stage 6/9 re-run) | Regression test 需要 update baseline (1 degenerate → verified) |

regression test baseline 是 living document — 当 audit state 预期改变时
(iter #87 / #88 完成), test 必须 explicit update 反映 new baseline。

## §F 文件 & 命令

- 模块: `PersoanlQuery/_smoke_audit_regression.py` (NEW, ~140 lines)
- 模块: `PersoanlQuery/paper_claims_audit.py` (--strict + --json-only stderr fix)
- 命令: `python3 PersoanlQuery/_smoke_audit_regression.py`
- 验证: `python3 -m py_compile PersoanlQuery/_smoke_audit_regression.py` (Rule 10) ✓
- 跑时间: < 2s (4 个 subprocess run + JSON parse)

## §G 与 loop.md §8 的关系

iter #105 完成 paper_claims_audit regression test:
- 4-case smoke test 冻结 baseline (2026-07-21: 0/6/4/1/1/5/0)
- 任何 silent flip 会 fail test, CI hook ready (iter #107)
- audit script minor fix: --strict + --json-only 也 print STRICT MODE stderr

后续 candidate:
- **iter #107** — Add pre-commit CI hook `python3 _smoke_audit_regression.py` 在每次 commit 前跑
- **iter #87** — Stage 12 re-run (9-21 h) — flip 5 unverified → verified (需 update regression test baseline)
- **iter #88** — Stage 6/9 re-run (1.5-3 h) — flip 1 degenerate → verified (需 update regression test baseline)
- **iter #108** — paper_claims_audit HTML dashboard