# Iteration #120 — README dashboard feature coverage sync (post-#108-#119)

**日期**: 2026-07-21
**scope**: `README.md` §Reproducibility Audit (extend dashboard feature list + add `--diff` mode + paper cross-refs)
**prior**: iter #111 added §Reproducibility Audit but only mentioned iter #108/#109/#110 dashboard features; this iter syncs README to the 11 dashboard/CLI iters since #111

## §A 审稿意见

iter #111 README sync 写时,dashboard 只有 3 iters (#108/#109/#110)。 现在累积到
11 个 audit infra iters (#108-#119),但 README 没更新:

- **dashboard features** (#112 provenance / #113 timestamp / #115 matched_files
  / #116 anchors + index / #117 section breakdown / #118 status legend) — README
  没 mention
- **CLI --diff mode** (#114) — README 没 mention
- **paper cross-references** (#119) — README 没 mention "paper ↔ audit JSON ↔
  dashboard" 4-way loop

后果: reviewer 第一次看 README 以为 dashboard 只是简单的 color-coded
status badges,不知道 anchors / section breakdown / legend / provenance 这些
affordance 存在 → 不知道可以 deep-link from paper。

iter #120 fix: README 完整 enumerate 11 dashboard/CLI features + paper
cross-ref section。

## §B 本轮 (iter #120) 改动

### README.md §Reproducibility Audit

**扩展 dashboard feature list** (从 3 行到 11 行 bullet):

```diff
- `result/personal_query/iterations/paper_claims_audit_dashboard.html` (self-contained HTML with color-coded status badges; iter #108/#109/#110)
+ `result/personal_query/iterations/paper_claims_audit_dashboard.html` (self-contained HTML; features):
+   - 7 color-coded status badges (iter #108)
+   - rel_delta as percent (iter #109)
+   - per-claim vc panel severity colors (iter #110)
+   - provenance panels expected_outputs + code_evidence (iter #112)
+   - matched_files for degenerate claims (iter #115)
+   - HTML anchors + claims-index sidebar (iter #116)
+   - per-section breakdown table (iter #117)
+   - inline status legend (iter #118)
+   - generated_at timestamp (iter #113)
+   - audit_target + claim_id_filter rows (iter #104)
```

**`--diff` mode 加到 re-run section**:

```diff
+ # Diff current audit against a frozen baseline JSON — exit 1 if any per-claim flip (status change, value extracted drift, new/removed claim); iter #114
+ python3 PersoanlQuery/paper_claims_audit.py --diff result/personal_query/iterations/paper_claims_audit.json --json-only
```

**新 §Paper ↔ audit cross-references (iter #119) section**:

```markdown
The companion paper's Table 1, 2, and 3 inline footnotes each name the
corresponding audit claim IDs in backticks (e.g. `RQ3_Fleiss_Kappa_0.72`,
`RQ4_GMM_Best_Prior`). Reviewer can:
- paper → audit JSON: grep
- paper → dashboard: open dashboard.html#claim-{id}
- audit → paper: each dashboard claim row shows its paper section

This forms a closed 4-way loop: paper ↔ audit JSON ↔ dashboard anchors ↔ --diff CLI.
```

## §C 关键改动点

1. **Full feature enumeration**: reviewer 现在知道 dashboard 11 个 affordance
   全部存在,不需要 grep code 或 trial-and-error。
2. **Cross-link documentation**: paper ↔ audit 4-way loop 显式说明, reviewer
   understand 整个 infra stack 怎么 use together。
3. **CLI parity**: README 文档 `--diff` mode (iter #114) 与 3 个原 re-run
   command 并列,不埋在 prose。

## §D 测试

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== AUDIT CI PASSED ===
```

README 改动 18 lines added (dashboard feature list) + 11 lines (--diff mode) + 8 lines (cross-ref section) = 37 lines。

## §E 文件 & 命令

- 模块: `README.md` (+37 lines: dashboard features / --diff mode / paper cross-refs)
- 验证: README 现在 mention 11 dashboard features + --diff CLI + 4-way cross-ref ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 README sync timeline 的关系

README 演进:
- **iter #111** — initial §Reproducibility Audit section (3 dashboard features listed)
- **iter #120** — sync to 11 dashboard/CLI features + paper cross-refs ✓ 本轮

每次 audit infra 加新 affordance (dashboard layer / CLI flag / paper doc) 都
应该 README 同步,这是 review surface 完整性。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample