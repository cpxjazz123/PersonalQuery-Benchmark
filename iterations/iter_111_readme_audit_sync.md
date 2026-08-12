# Iteration #111 — README.md audit infrastructure sync (iter #106 realization)

**日期**: 2026-07-21
**scope**: `README.md` (add "Reproducibility Audit" section)
**prior**: iter #106 backlog; this iter surfaces the audit infra (iter #93–#110) to the top-level project README

## §A 审稿意见

iter #93–#110 累积了完整 audit infrastructure (paper_claims_audit.py CLI +
regression smoke test + pre-commit CI + HTML dashboard + iter #93-#99 paper
documentation),但 README.md 只描述 dataset + pipeline overview,没有提到
audit infra。 Reviewer 第一次 clone repo 看 README,**不知道有 paper claim
audit**,也不会 run `python3 PersoanlQuery/paper_claims_audit.py --list` 或
看 dashboard HTML。

iter #111 把 audit infra 提到 README 顶层,作为 "Reproducibility Audit" 段:
- 现状 summary table (17 claims / 6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified)
- 指向 audit JSON + dashboard HTML
- re-run 命令 (3 种粒度)
- pre-commit hook setup + bypass + uninstall
- re-execution cost table (iter #87/#88 预计时间)

## §B 本轮 (iter #111) 改动

### README.md (add §Reproducibility Audit after §Dataset Statistics)

```markdown
## Reproducibility Audit

Every quantitative claim in the companion paper has been audited...

### Current Audit State (frozen 2026-07-21)
| Status | Count | Meaning |
| verified_value_match | 0 | file exists, values match paper |
| verified | 6 | file exists, no numerical claim |
| discrepant | 4 | file exists, values disagree (11.6–78.4% rel delta) |
| degenerate | 1 | file exists, selector returned no non-NaN values |
| partial | 1 | code referenced, no specific outputs enumerated |
| unverified | 5 | code referenced but no expected outputs (Stage 12 gap) |
| **Total** | **17** | |

For full per-claim audit: paper_claims_audit.json + paper_claims_audit_dashboard.html

### Re-run
bash PersoanlQuery/_run_audit_ci.sh
python3 PersoanlQuery/paper_claims_audit.py --claim-id RQ3_Fleiss_Kappa_0.72 --verbose
python3 PersoanlQuery/paper_claims_audit.py --strict --json-only

### Pre-commit hook
bash PersoanlQuery/install_audit_precommit.sh

### Re-execution cost
| iter | Stage | Time | Flips |
| iter #87 | Stage 12 + Stage 10 | 9-21 h GPU | 5 unverified → verified |
| iter #88 | Stage 6/9 rebuild | 1.5-3 h GPU | 1 degenerate → verified |
```

## §C 关键改动点

1. **Top-level visibility**: README 是 reviewer 第一次接触项目的地方,audit infra
   必须在这里出现,不能只埋在 `PersoanlQuery/_AUDIT_CI_README.md` (那是
   developer-facing,不是 reviewer-facing)。
2. **Re-execution cost table**: reviewer 想知道 "要多久才能让 audit 全绿",
   给出 iter #87/#88 预计时间 (9-21 h + 1.5-3 h) + 哪些 status 会 flip。
3. **3-tier re-run commands**: full CI / 单 claim inspect / strict mode — 不同
   reviewer 需求覆盖。
4. **Cross-link to PersoanlQuery/_AUDIT_CI_README.md**: 详细文档不重复,
   reviewer 从 README 跳到详细 README。

## §D 测试

CI full run PASS (README 改动不影响 audit script,但确保 pipeline 不 break):
```
[1/3] Regenerating audit JSON...  PASS
[2/3] Running regression smoke test...  4/4 cases pass
[3/3] Regenerating HTML dashboard...  PASS
=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `README.md` (+56 lines: §Reproducibility Audit section)
- 验证: README 现在 contain audit infra references (verified by reviewer spot check)
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra 累积的关系

完整 audit infra stack:
- **iter #93** value-validation layer (file+value)
- **iter #94-#99** paper inline documentation (Table 1/2/3 footnotes + §5)
- **iter #103** §5 Conclusion Reproducibility subsection (paper 内部)
- **iter #104** CLI refactor (argparse)
- **iter #105** regression smoke test
- **iter #107** pre-commit CI hook
- **iter #108** HTML dashboard
- **iter #109** rel_delta percent fix
- **iter #110** vc status color coding
- **iter #111** README.md sync (project-level visibility) ✓ 本轮

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
- **iter #112+** — additional reviewer-driven audit infra improvements
  (e.g. add `expected_outputs` + `code_evidence` to audit JSON for full provenance)