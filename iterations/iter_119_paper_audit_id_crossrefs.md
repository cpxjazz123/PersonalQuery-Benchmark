# Iteration #119 — Paper §3.3 + Table 2 footnote + §5 audit ID cross-references

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` (add audit IDs to Table 2 footnote + §5 Limitations #5)
**prior**: iter #94 added Table 2 footnote but without audit IDs; iter #116 added dashboard anchors; this iter bridges the two

## §A 审稿意见

iter #94 加 Table 2 footnote 列出 4 discrepant metric (Fleiss 0.72 / Spearman 0.81
/ MAE 0.89 / preservation 94.6%) 与 release value (0.6235 / 0.7164 / 0.62 / 20.5%),
但 footnote **不引用 audit claim IDs** (`RQ3_Fleiss_Kappa_0.72` 等)。

后果: reviewer 看 paper Table 2 footnote 知道 "paper 0.72 vs release 0.6235",
但不能:
- `grep` audit JSON 找到对应 claim entry
- 直接跳到 dashboard claim row (`dashboard.html#claim-RQ3_Fleiss_Kappa_0.72`)
- 区分 "Table 2 footnote 提到的 0.72" vs "audit JSON 中某个其他 claim 也用 0.72"

iter #119 fix: Table 2 footnote + §5 Limitations #5 各加 4 个 audit claim ID in
backticks (renders as code),明确 4 个 metric → 4 个 audit claim 1:1 mapping。

## §B 本轮 (iter #119) 改动

### §B.1 Table 2 footnote (paper.md line 161-164)

```diff
- See `result/personal_query/iterations/paper_claims_audit.json`
- for the full per-claim audit (iter #93 value-validation layer).
+ Each metric corresponds to a specific audit claim in
+ `result/personal_query/iterations/paper_claims_audit.json`: Fleiss κ → `RQ3_Fleiss_Kappa_0.72`,
+ Spearman ρ → `RQ3_Spearman_0.81`, MAE → `RQ3_MAE_0.89`, semantic preservation →
+ `RQ3_LLM_Full_Set_94%` (subclaim: semantic preservation). Reviewer can grep the JSON
+ or jump directly to the dashboard anchor (e.g. `paper_claims_audit_dashboard.html#claim-RQ3_Fleiss_Kappa_0.72`,
+ iter #116). See `result/personal_query/iterations/paper_claims_audit.json` for the
+ full per-claim audit (iter #93 value-validation layer).
```

### §B.2 §5 Limitations #5 (paper.md line 204)

```diff
- (0.205 vs 0.946); see Table 2 footnote for details.
+ (0.205 vs 0.946); see Table 2 footnote for details. Each of these four metrics is
+ also individually enumerated in the audit JSON as `RQ3_Fleiss_Kappa_0.72`,
+ `RQ3_Spearman_0.81`, `RQ3_MAE_0.89`, and `RQ3_LLM_Full_Set_94%` (the last as a
+ subclaim of the LLM full-set semantic-evaluation claim).
```

### §B.3 §5 Reproducibility (paper.md line 206)

补 `--diff` mode (iter #114) + dashboard (iter #108) + anchors (iter #116) reference:

```diff
- change to verify reproducibility.
+ change to verify reproducibility; the `--diff` mode (`python3 PersoanlQuery/paper_claims_audit.py
+ --diff result/personal_query/iterations/paper_claims_audit.json`, iter #114) flags per-claim
+ flips vs a frozen baseline, and the dashboard (`result/personal_query/iterations/paper_claims_audit_dashboard.html`,
+ iter #108) renders the same audit as a self-contained HTML with deep-link anchors per
+ claim (iter #116).
```

## §C 关键改动点

1. **Paper ↔ audit JSON ↔ dashboard 三向 cross-link**: 现在 reviewer 任何
   起点都能跳:
   - **paper → audit**: grep audit ID, or follow URL `dashboard.html#claim-{id}`
   - **paper → release values**: 看 Table 2 footnote
   - **dashboard → paper**: claim row 显示 section (§3.3 + Table 2),可手动 navigate 回 paper
2. **Code-formatted IDs**: backticks render as `<code>RQ3_Fleiss_Kappa_0.72</code>`
   在 markdown viewer (GitHub, IDE, paper preview) 都 visually distinct。
3. **Subclaim disambiguation**: `RQ3_LLM_Full_Set_94%` 有 3 subclaims
   (semantic plausibility / target structure / semantic preservation),footnote
   明确 "subclaim: semantic preservation" 让 reviewer 知道是哪一个。

## §D 测试

Paper grep verification:
```
$ grep "RQ3_Fleiss_Kappa_0.72" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md
161:`result/personal_query/iterations/paper_claims_audit.json`: Fleiss κ → `RQ3_Fleiss_Kappa_0.72`,
204:also individually enumerated in the audit JSON as `RQ3_Fleiss_Kappa_0.72`, ...
```

4 IDs × 2 paper locations (Table 2 footnote + §5 Limitations #5) = 8 references。
audit JSON 中这 4 IDs status 全是 `discrepant`, reviewer 现在 paper → audit 链路
完整。

Full audit CI 测试 PASS (paper 改动不影响 audit script / dashboard,纯 doc 改动)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` (+9 lines: audit IDs in 3 places)
- 验证: paper grep 显示 8 audit ID references ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra cross-link 的关系

paper ↔ audit 双向 cross-link timeline:
- **#94-#99** — paper inline footnote (audit findings → paper)
- **#103** — §5 Reproducibility subsection
- **#111** — README sync (project-level visibility)
- **#119** — paper references audit claim IDs (paper → audit, machine-grepable) ✓ 本轮

reviewer 现在可以:
1. 读 paper §3.3 / Table 2 → 见 audit IDs (`RQ3_Fleiss_Kappa_0.72`)
2. Open dashboard `dashboard.html#claim-RQ3_Fleiss_Kappa_0.72` → 跳到该行
3. 看 dashboard detail panel (value_mismatch + 11.56% rel_delta + matched_files)
4. 决定是否 run `--diff baseline.json` ad-hoc

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample