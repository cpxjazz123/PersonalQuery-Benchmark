# Iteration #125 — Dashboard `<link rel="alternate">` to mapping sidecar

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add `<link rel='alternate' type='application/json'>` to `<head>`); `_smoke_audit_regression.py` extended to 11 cases (Case K)
**prior**: iter #124 generated `paper_audit_id_mapping.json` machine-readable sidecar but dashboard HTML had no machine-readable pointer to it. Reviewer / browser tooling had no way to discover the sidecar from the dashboard.

## §A 审稿意见

iter #124 generated `paper_audit_id_mapping.json` (17/17 audit IDs ×
26 paper locations),but dashboard HTML `<head>` 没 machine-readable pointer
到 sidecar。后果:

- reviewer 在 browser 打开 dashboard,想 programmatically discover mapping
  sidecar → 必须已知文件名 + 手动 fetch。
- browser dev tools / CI tooling 想 auto-discover related JSON resources →
  无 `<link rel='alternate'>` 提示。
- Static site analyzer / link checker 想 verify "dashboard 跟所有 related
  artifacts 有 machine-readable link" → 缺。

iter #125 fix: dashboard `<head>` 加
`<link rel='alternate' type='application/json' title='paper_audit_id_mapping' href='paper_audit_id_mapping.json'>`
— 浏览器/RSS reader/CI tooling 看到这 tag 立即知道 sidecar 存在。

## §B 本轮 (iter #125) 改动

### PersoanlQuery/_generate_audit_dashboard.py

`_render_html` `<head>` 加 1 行:

```diff
 <meta charset='utf-8'>
 <meta name='description' content='{meta_desc}'>
-<meta name='generator' content='_generate_audit_dashboard.py (iter #122)'>
+<meta name='generator' content='_generate_audit_dashboard.py (iter #122, #125)'>
+<link rel='alternate' type='application/json' title='paper_audit_id_mapping' href='paper_audit_id_mapping.json'>
 <title>{title_str}</title>
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case K freeze iter #125 behavior:

```python
def case_k_dashboard_link_alternate_mapping():
    """Case K (iter #125): dashboard HTML <link rel=alternate type=application/json> points to
    paper_audit_id_mapping.json so browser dev tools / CI tooling can discover the sidecar."""
    link_match = re.search(
        r"<link\s+rel='alternate'\s+type='application/json'\s+title='paper_audit_id_mapping'\s+href='([^']+)'>",
        html,
    )
    assert link_match, "missing <link rel=alternate type=application/json> for paper_audit_id_mapping"
    href = link_match.group(1)
    assert href == "paper_audit_id_mapping.json", f"unexpected href: {href}"
    sidecar = REPO_ROOT / "result/personal_query/iterations" / href
    assert sidecar.exists(), f"mapping sidecar not found: {sidecar}"
```

docstring + main() 同步更新到 11 cases。

## §C 关键改动点

1. **`rel='alternate'` (不是 `stylesheet` / `icon`)**: alternate 是 HTML spec
   for "this resource is alternative representation of current doc"。reviewer
   tooling 看到这 tag 知道 sidecar is semantically equivalent (machine-readable
   audit claim index) 不是 styling/icon dependency。

2. **`type='application/json'`**: 显式声明 MIME type,工具知道这是 JSON
   不是 HTML。HTML spec 允许 application/json type 跟 alternate rel 组合。

3. **`title='paper_audit_id_mapping'`**: human-readable hint,跟 href 一致。
   RSS reader 用 title 显示 link text。

4. **Href 跟 sidecar 同目录**: `paper_audit_id_mapping.json` (no path
   prefix),因为 dashboard 跟 sidecar 在同 dir
   (`result/personal_query/iterations/`)。这是 relative URL 优势:即使
   整个 dir 被 move/rename,link 仍有效。

5. **Case K 三个 assertion**:
   - `<link>` tag present (regex 精确匹配避免 false positive)
   - href 准确指向 sidecar (catches rename)
   - sidecar file exists (catches broken link)

## §D 测试

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
[1/4] Regenerating audit JSON...                                    PASS
[2/4] Running regression smoke test...
=== paper_claims_audit regression smoke test (iter #105, #121, #122, #123, #124, #125) ===

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'                                PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965               PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message                                 PASS
Case D: --strict --json-only (full audit) expects exit 1                                         PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)                             PASS
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)                        PASS
Case G: dashboard HTML has anchors + status legend (iter #121)                                   PASS
Case H: dashboard HTML <title> + <meta name='description'> (iter #122)                           PASS
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123)                           PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                        PASS
Case K: dashboard <link rel=alternate> -> paper_audit_id_mapping.json (iter #125)                PASS

All 11 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...          PASS

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+1 line `<link rel='alternate'>`,
  generator meta updated to "iter #122, #125")
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case K, +docstring,
  +main count to 11)
- 验证: 11/11 cases pass; pre-commit hook freeze 11 invariants (4 + 3 + 1 + 1 + 1 + 1) ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra timeline 的关系

dashboard cross-link affordance timeline:
- **iter #108** — basic status badges + summary
- **iter #112** — provenance panels (expected_outputs + code_evidence)
- **iter #116** — HTML anchors per claim (`#claim-{id}`)
- **iter #118** — inline status legend
- **iter #122** — browser metadata (`<title>` + `<meta description>`)
- **iter #124** — paper_audit_id_mapping.json machine-readable sidecar
- **iter #125** — dashboard `<link rel='alternate'>` to sidecar ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta)
- **iter #123** — 9 cases (+ paper cross-ref coverage)
- **iter #124** — 10 cases (+ paper-audit mapping sidecar)
- **iter #125** — 11 cases (+ dashboard link rel alternate) ✓ 本轮

每个新 cross-link affordance 都应该 regression test 冻结,否则
machine-readable link 失效不会被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
