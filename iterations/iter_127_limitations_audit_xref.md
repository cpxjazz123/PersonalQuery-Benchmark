# Iteration #127 — §5 Limitations numbered items inline audit ID cross-link

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` §5 Limitations (line 204) 5 numbered items (First/Second/Third/Fourth/Fifth) each get trailing `*(Audit: ...; iter #127.)*` parenthetical; extend regression Case I to verify 5 numbered Limitations
**prior**: iter #123 cross-linked Table 1/2/3 footnotes + §1 paragraph; iter #126 cross-linked §1 4 contribution bullets. §5 Limitations 5 numbered items (First-Fifth) describing domain coverage / GMM prior assumption / regen mechanism / Stage 12 lineage / human-eval pilot still had no inline audit IDs — reviewer reading limitations top-down couldn't trace to specific audit claims.

## §A 审稿意见

iter #123/#126 cross-linked Tables 1/2/3 footnotes + §1 paragraph + §1
contribution bullets,但 §5 Limitations 5 numbered items 仍缺 inline cross-link:

- **First** (dataset domain coverage) — 没 audit ID; 是 §3.1 setup,
  没 specific claim ID
- **Second** (GMM prior assumption) — 没 audit ID; 应 cite `RQ4_GMM_Best_Prior` +
  `Sec2_K2_GMM_per_user`
- **Third** (regen mechanism + Stage 7 TypeError) — 没 audit ID; 应 cite
  `Pipeline_Regeneration_10x10` + `RQ1_Table1_Hit10`
- **Fourth** (Stage 12 lineage gap) — 部分 cite `Sec2_*` unverified 在
  Table 3 footnote,但 Limitations 末尾没 inline backticks
- **Fifth** (human-eval pilot) — partial cite 4 RQ3 IDs 已经在 paragraph
  末尾,但 用 prose form 不是 inline parenthetical

后果: reviewer read Limitations 想 trace 到 audit claims 必须跳到 Table
footnotes + §1 paragraph — 不 inline navigation。

iter #127 fix: 每个 First/Second/Third/Fourth/Fifth 末尾加
`*(Audit: `...`; iter #127.)*` inline parenthetical (First dataset-domain
没有 specific claim ID,只标 "covered by §3.1 setup")。

## §B 本轮 (iter #127) 改动

### PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md

§5 Limitations (line 204) 5 numbered items each get trailing parenthetical:

```diff
- First, experiments are conducted within three Amazon product subcategories (...); ...
+ First, experiments are conducted within three Amazon product subcategories (...); ...
+ *(Audit: dataset-domain claim covered by §3.1 setup; iter #127.)*

- Second, the core assumption—that Amazon review writing style correlates with query formulation behavior—
+ Second, the core assumption—that Amazon review writing style correlates with query formulation behavior—
+ *(Audit: `RQ4_GMM_Best_Prior` + `Sec2_K2_GMM_per_user`; iter #127.)*

- Third, the 10-round × 10-candidate regeneration mechanism ...
+ Third, the 10-round × 10-candidate regeneration mechanism ...
+ See Table 1 footnote (line 129) for the quantitative impact on bootstrap CI.
+ *(Audit: `Pipeline_Regeneration_10x10` + `RQ1_Table1_Hit10`; iter #127.)*

- Fourth, while Stage 12 (PRF clause-features) outputs required for Table 3 ...
+ Fourth, while Stage 12 (PRF clause-features) outputs required for Table 3 ...
+ See Table 3 footnote (line 173) for the full list of 5 unverified audit claims and re-execution cost breakdown.
+ *(Audit: `Sec2_20dim_syntactic_features` + `Sec2_K8_clusters_BIC_AIC_silhouette` + `Sec2_95pct_holdout_threshold`; iter #127.)*

- Fifth, the human-evaluation agreement reported in Table 2 ...
+ Fifth, the human-evaluation agreement reported in Table 2 ...
+ as a subclaim of the LLM full-set semantic-evaluation claim).
+ *(Audit: `RQ3_Fleiss_Kappa_0.72` + `RQ3_Spearman_0.81` + `RQ3_MAE_0.89` + `RQ3_LLM_Full_Set_94%`; iter #127.)*
```

Total: +11 audit IDs backticked across 5 Limitations items (0 + 2 + 2 + 3 + 4)。
First item dataset-domain 是 qualitative scope claim,无 specific audit ID
(对应 audit JSON 没 Sec3_3_Domain_Scope claim),所以用 prose "covered by
§3.1 setup" 而非 audit IDs。

### PersoanlQuery/_smoke_audit_regression.py

Case I 扩展,加 5 Limitations numbered item 存在性检查:

```python
limitations_items = ["First,", "Second,", "Third,", "Fourth,", "Fifth,"]
# ... after contribution_bullets loop ...
for item in limitations_items:
    assert item in paper, f"§5 Limitations numbered item missing: {item!r}"
```

## §C 关键改动点

1. **First item 用 prose 而非 audit IDs**: dataset-domain coverage 是
   qualitative scope claim,audit JSON 没 specific claim ID (Stage 03
   query generation 跨 3 domains,但 audit 只 file-existence check 没
   per-domain claim)。用 "covered by §3.1 setup" 显式说 limitation 是
   paper 自身 §3.1 描述 (reviewer 跟 reviewer)。

2. **每个 numbered item 用 + 分隔 audit IDs**: 跟 iter #126 contribution
   bullets 一致 — `RQ4_GMM_Best_Prior` + `Sec2_K2_GMM_per_user` 让
   multi-ID contexts 视觉一致。

3. **Mapping sidecar 自动扩展**: iter #124 regex 抓 inline backticks;
   §5 Limitations +11 audit IDs 自动收录到
   `paper_audit_id_mapping.json` (`n_paper_locations` 从 41 增到 52,
   +11)。无需 manual sidecar update。

4. **Case I 现在检查 5 个 anchor points**:
   - 17/17 audit IDs 至少一处 backticked (§1 paragraph OR footnote)
   - 4 §1 contribution bullets present
   - 5 §5 Limitations numbered items present (First/Second/.../Fifth)
   总 26 anchor points 全部 required present in paper。

5. **No new audit claims**: iter #127 只是 cross-link indexing 改进,
   不改 audit JSON, 不改 status, 不改 frozen baseline (0/6/4/1/1/5/0)。

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
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123, #126, #127)               PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                        PASS
Case K: dashboard <link rel=alternate> -> paper_audit_id_mapping.json (iter #125)                PASS

All 11 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...
  n_audit_claims: 17
  n_paper_locations: 52   (was 41 before iter #127; +11 from Limitations)
  unmapped_audit_ids: []
  unmapped_paper_ids: []
  summary: {'degenerate_covered': 1, 'discrepant_covered': 4, 'partial_covered': 1, 'unverified_covered': 5, 'verified_covered': 6}
  PASS  paper_audit_id_mapping.json regenerated

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` §5
  Limitations 5 numbered items +11 backticked audit IDs (Second=2 + Third=2
  + Fourth=3 + Fifth=4 = 11; First 是 qualitative scope, 无 specific audit ID)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` Case I extended
  with limitations_items list (5 numbered items) + assert each present in paper
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
  cross-link
- **iter #127** — §5 Limitations 5 numbered items +11 audit IDs inline
  cross-link ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta)
- **iter #123** — 9 cases (+ paper cross-ref coverage)
- **iter #124** — 10 cases (+ paper-audit mapping sidecar)
- **iter #125** — 11 cases (+ dashboard link rel alternate)
- **iter #126** — 11 cases + Case I extended with 4 contribution bullets
- **iter #127** — 11 cases + Case I extended with 5 Limitations items ✓ 本轮

每个新 cross-link affordance 都应该 regression test 冻结,否则 reviewer
第一次 read paper 时发现 paper-to-dashboard link 失效不会被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
