# Iteration #126 — §1 contribution bullets inline audit ID cross-link

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` §1 contribution bullets (lines 35-38) each get inline backticked audit IDs in parentheses; extend regression Case I to verify 4 contribution bullets present
**prior**: iter #123 added §1 reproducibility-audit paragraph enumerating 6 §2.2 infrastructure claim IDs. The 4 numbered contribution bullets above (lines 35-38) had no inline audit ID cross-link — reviewer reading top-down through contributions couldn't navigate to corresponding audit claims without grepping the rest of paper.

## §A 审稿意见

iter #123 完成 §1 reproducibility-audit paragraph 6 §2.2 infrastructure claim
IDs enumeration,但 §1 contribution bullets (lines 35-38) 仍无 inline cross-link:

- (1) Systemic Personalized Retrieval Evaluation Study → 应该 cite
  `RQ1_Table1_Hit10` + `RQ1_Delta_Range` + `RQ2_Table1_Drop` +
  `Pipeline_Skip_ColBERTv2_SPLADE` (§3.2 + Table 1)
- (2) Personalized Query Dataset → 应该 cite
  `Sec2_5_attrs_per_query` + `UserFilter_20_reviews_15_words`
  (§2.1 + §2.2)
- (3) Multivariate Gaussian Mixture Style Modeling → 应该 cite
  `RQ4_GMM_Best_Prior` + `Sec2_K2_GMM_per_user` +
  `Sec2_20dim_syntactic_features` + `Sec2_K8_clusters_BIC_AIC_silhouette`
  (§2.2 + §3.4 + Table 3)
- (4) Quality Guarantee → 应该 cite
  `RQ3_Fleiss_Kappa_0.72` + `RQ3_Spearman_0.81` + `RQ3_MAE_0.89` +
  `RQ3_LLM_Full_Set_94%` + `BPE_aware_Error_Injection` (§3.3 + Table 2)

后果: reviewer 第一次 read paper 从 §1 开始 — 4 contribution bullets 是
high-level summary,但 back-link 到具体 audit claims 必须跳到 Table
footnotes + §5 Limitations 才能 trace。contribution 跟 audit 没 inline
link。

iter #126 fix: 每个 contribution bullet 末尾加 `*(Audited via `...`,
§X; iter #126.)*` inline parenthetical,3-5 audit IDs backticked。

## §B 本轮 (iter #126) 改动

### PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md

§1 contribution bullets (lines 35-38) each get trailing parenthetical:

```diff
- (1) Systemic Personalized Retrieval Evaluation Study. ... when the queries have personalized
-     syntactic expression differences.
+ (1) Systemic Personalized Retrieval Evaluation Study. ... when the queries have personalized
+     syntactic expression differences. *(Audited via `RQ1_Table1_Hit10` + `RQ1_Delta_Range` +
+     `RQ2_Table1_Drop` + `Pipeline_Skip_ColBERTv2_SPLADE`, §3.2 + Table 1; iter #126.)*

- (2) Personalized Query Dataset. ... covering three Amazon domains.
+ (2) Personalized Query Dataset. ... covering three Amazon domains. *(Audited via
+     `Sec2_5_attrs_per_query` + `UserFilter_20_reviews_15_words`, §2.1 + §2.2; iter #126.)*

- (3) Multivariate Gaussian Mixture Style Modeling. ... during the modelling process.
+ (3) Multivariate Gaussian Mixture Style Modeling. ... during the modelling process.
+     *(Audited via `RQ4_GMM_Best_Prior` + `Sec2_K2_GMM_per_user` +
+     `Sec2_20dim_syntactic_features` + `Sec2_K8_clusters_BIC_AIC_silhouette`, §2.2 + §3.4 + Table 3;
+     iter #126.)*

- (4) Quality Guarantee. ... preservation of injected error patterns.
+ (4) Quality Guarantee. ... preservation of injected error patterns. *(Audited via
+     `RQ3_Fleiss_Kappa_0.72` + `RQ3_Spearman_0.81` + `RQ3_MAE_0.89` + `RQ3_LLM_Full_Set_94%` +
+     `BPE_aware_Error_Injection`, §3.3 + Table 2; iter #126.)*
```

Total: 15 audit IDs backticked across 4 contribution bullets (4 + 2 + 4 + 5)。

### PersoanlQuery/_smoke_audit_regression.py

Case I 扩展,加 4 contribution bullet 存在性检查:

```python
contribution_bullets = [
    "(1) Systemic Personalized Retrieval Evaluation Study",
    "(2) Personalized Query Dataset",
    "(3) Multivariate Gaussian Mixture Style Modeling",
    "(4) Quality Guarantee",
]
# ... after per-claim coverage loop ...
for bullet in contribution_bullets:
    assert bullet in paper, f"§1 contribution bullet missing: {bullet!r}"
```

## §C 关键改动点

1. **Inline parenthetical 用 `*(...)*`**: Markdown italic + 圆括号 —
   视觉上区分 audit cross-link 跟正文。reviewer 扫读 contributions 时
   可 skip,需要 trace 时立即看到。
2. **Audit IDs 用 + 分隔**: 每个 contribution 列 3-5 IDs, 用 `+` 而非
   逗号 — Markdown 列表更易 parse 且与 `RQ3_Fleiss_Kappa_0.72 +
   RQ3_Spearman_0.81` 等 multi-ID contexts 一致。
3. **Citation section 跟在 IDs 后**: `(audit IDs, §X; iter #126.)` —
   section reference 让 reviewer 知道该 contribution evidence 在 paper
   哪节。
4. **Mapping sidecar 自动扩展**: iter #124 regex `_AUDIT_ID_PREFIX` 扫
   全 paper;新 §1 contribution bullet backticks 自动被收录到
   `paper_audit_id_mapping.json` (`n_paper_locations` 从 26 增到 41,
   `+15` audit ID citations across 4 contribution bullets)。无需手工
   update sidecar — CI [4/4] step regenerate 自动 capture。
5. **Case I dual check**: 不仅 17/17 audit IDs 至少一处 backticked,
   还检查 4 contribution bullets 都 present。catches paper refactor
   删 bullet 但保留 footnote。

## §D 测试

```bash
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
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123, #126)                     PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                        PASS
Case K: dashboard <link rel=alternate> -> paper_audit_id_mapping.json (iter #125)                PASS

All 11 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...
  n_audit_claims: 17
  n_paper_locations: 41   (was 26 before iter #126; +15 from contribution bullets)
  unmapped_audit_ids: []
  unmapped_paper_ids: []
  summary: {'degenerate_covered': 1, 'discrepant_covered': 4, 'partial_covered': 1, 'unverified_covered': 5, 'verified_covered': 6}
  PASS  paper_audit_id_mapping.json regenerated

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`
  §1 contribution bullets (lines 35-38) +15 backticked audit IDs
  across 4 contribution bullets
- 测试: `PersoanlQuery/_smoke_audit_regression.py` Case I extended
  with contribution_bullets list (4 bullets) + assert each present in paper
- 验证: 11/11 cases pass; pre-commit hook freeze 11 invariants (4 + 3 + 1 + 1 + 1 + 1) ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra timeline 的关系

paper ↔ audit cross-link timeline:
- **iter #93** — paper_claims_audit.py + JSON 输出
- **iter #108** — dashboard HTML 化
- **iter #111** — README §Reproducibility Audit section
- **iter #116** — dashboard HTML anchors per claim (`#claim-{id}`)
- **iter #119** — paper Table 2 footnote 4 RQ3 claim IDs backticked
- **iter #122** — dashboard browser metadata (`<title>` + `<meta description>`)
- **iter #123** — paper Table 1/3 footnotes + §1 paragraph: 17/17 audit
  IDs cross-referenced (human-readable inline)
- **iter #124** — paper_audit_id_mapping.json machine-readable index
- **iter #125** — dashboard `<link rel='alternate'>` to sidecar
- **iter #126** — §1 contribution bullets 4×3-5 audit IDs inline
  cross-link (reviewer top-down navigation) ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta)
- **iter #123** — 9 cases (+ paper cross-ref coverage)
- **iter #124** — 10 cases (+ paper-audit mapping sidecar)
- **iter #125** — 11 cases (+ dashboard link rel alternate)
- **iter #126** — 11 cases + Case I extended with contribution_bullets
  assertion ✓ 本轮 (no new case, Case I 增强)

每个新 cross-link affordance 都应该 regression test 冻结,否则 reviewer
第一次 read paper 时发现 paper-to-dashboard link 失效不会被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
