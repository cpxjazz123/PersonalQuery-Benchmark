# Iteration #129 — Dashboard reverse section index panel

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` add `_render_reverse_section_panel()` and wire into body; extend `_smoke_audit_regression.py` to 13 cases (Case M)
**prior**: iter #128 added reverse_section_index to `paper_audit_id_mapping.json` sidecar (section → audit IDs), but dashboard HTML didn't surface this view. Reviewer saw per-claim rows but no paper-section-level reverse view. iter #129 embeds the reverse index as a dashboard panel with bidirectional anchor links.

## §A 审稿意见

iter #128 在 mapping sidecar 加 `reverse_section_index` (paper section →
audit IDs cited),但 dashboard HTML 不显示这 view。后果:

- reviewer 在 dashboard 看到 audit claim 列表,但看不到 "§3.4 cite 哪些 audit claims"。
- 想查 reverse 必须 jq mapping sidecar 或 grep paper text,dashboard
  没 affordance。
- Sidecar 现在 bidirectional 但 dashboard 单向。

iter #129 fix: dashboard 加 `Paper sections by audit claim count` panel —
从 `paper_audit_id_mapping.json` load `reverse_section_index`,render 为
table 每行: section label + claim count + audit IDs as clickable anchor links
to `#claim-{id}`。reviewer 看到 "§3.4 Prior Realism of the User Style
Distribution — 6 claims: RQ4_GMM_Best_Prior, Sec2_20dim_syntactic_features,
..." 直接 click anchor jump 到 claim row。

## §B 本轮 (iter #129) 改动

### PersoanlQuery/_generate_audit_dashboard.py

新增 `_render_reverse_section_panel()` (~25 lines):

```python
def _render_reverse_section_panel(reverse_index: dict[str, list[str]]) -> str:
    """iter #129: render paper section → audit claim count reverse index panel."""
    if not reverse_index:
        return ""
    rows = []
    for sec_label, cids in reverse_index.items():
        # Each audit_id is rendered as a clickable anchor link to claim-{id}
        cid_links = " ".join(
            f"<a class='rev-cid' href='#claim-{cid}'><code>{cid}</code></a>"
            for cid in cids
        )
        rows.append(
            f"<tr><td class='rev-sec'><code>{sec_label}</code></td>"
            f"<td class='rev-count'>{len(cids)}</td>"
            f"<td class='rev-cids'>{cid_links}</td></tr>"
        )
    return f"""<details class='reverse-section' open>
<summary><strong>Paper sections by audit claim count</strong> ({len(reverse_index)} sections)</summary>
<table class='rev-table'>
<thead><tr><th>Section</th><th># claims</th><th>Audit claim IDs</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</details>"""
```

`_render_html` wire load + render:

```python
reverse_section_html = ""
try:
    mapping_path = REPO_ROOT / "result/personal_query/iterations/paper_audit_id_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        reverse_section_html = _render_reverse_section_panel(mapping.get("reverse_section_index", {}))
except (OSError, json.JSONDecodeError):
    pass
```

Body 加 `{reverse_section_html}` 在 Summary section 后。

CSS 新增 purple-tinted `.reverse-section` panel (跟 section-breakdown blue
跟 status-legend blue 视觉区分 — purple = reverse index):

```css
.reverse-section { background: #f3e5f5; padding: 8px 12px; ... }
.rev-table th { background: #e1bee7; padding: 4px 6px; }
.rev-cids a { text-decoration: none; color: #555; margin-right: 4px; }
.rev-cids a:hover { text-decoration: underline; color: #1976d2; }
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case M freeze iter #129 behavior:

```python
def case_m_dashboard_reverse_section_panel():
    """Case M (iter #129): dashboard HTML embeds reverse_section panel from paper_audit_id_mapping.json."""
    html = dash.read_text(encoding="utf-8")
    assert "reverse-section" in html
    assert "Paper sections by audit claim count" in html
    assert "3.4 Prior Realism of the User Style Distribution" in html
    assert "href='#claim-RQ4_GMM_Best_Prior'" in html
    assert "href='#claim-Sec2_20dim_syntactic_features'" in html
```

docstring + main() 同步更新到 13 cases。

## §C 关键改动点

1. **Purple-tinted panel**: reverse-section background `#f3e5f5` (light purple)
   跟 section-breakdown `#f5f5f5` (gray) 跟 status-legend `#e3f2fd` (blue) 视觉
   区分 — reviewer 一眼能 identify panel purpose。

2. **Bidirectional anchor links**: 每个 audit_id in panel row 是
   `<a href='#claim-{cid}'>` — reviewer click §3.4 列表的 `RQ4_GMM_Best_Prior`
   直接 scroll 到 claim row in Claims table。这是 paper section → claim 的
   一跳导航 (反向 iter #116 claim → paper section)。

3. **Try-except graceful fallback**: 如果 sidecar missing 或 JSON malformed,
   dashboard 仍 render (empty panel),不 fail。CI [4/4] step 确保 sidecar
   always fresh,但 dashboard 不应 hard-depend on sidecar presence。

4. **Anchor href 跟 dashboard claim-{id} format 一致**: iter #116 anchor
   format `id='claim-{cid}'`,reverse panel href 同样 `#claim-{cid}` —
   browser native scroll-to-anchor,无 JS dependency。

5. **5 sections rendered (frozen baseline)**:
   - §1 Introduction: 17 audit IDs
   - §3.1 Experimental Setup: 3
   - §3.3 LLM and Human Quality Validation: 4
   - §3.4 Prior Realism of the User Style Distribution: 6
   - §5 Conclusion: 11

6. **No new audit claims**: iter #129 只是 dashboard UI 改进,不改 audit JSON,
   不改 sidecar structure (除 dashboard load),不改 frozen baseline (0/6/4/1/1/5/0)。

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
Case L: mapping sidecar reverse_section_index (iter #128)                                        PASS
Case M: dashboard reverse_section panel (iter #129)                                              PASS

All 13 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...
  n_audit_claims: 17
  n_paper_locations: 52
  n_sections_with_audit_refs: 5
  PASS  paper_audit_id_mapping.json regenerated

=== AUDIT CI PASSED ===
```

Dashboard panel §3.4 row click 体验:

```
3.4 Prior Realism of the User Style Distribution    6   RQ4_GMM_Best_Prior  Sec2_20dim_syntactic_features
                                                          Sec2_5_attrs_per_query  Sec2_95pct_holdout_threshold
                                                          Sec2_K2_GMM_per_user  Sec2_K8_clusters_BIC_AIC_silhouette
```

Click `RQ4_GMM_Best_Prior` → browser scrolls to `#claim-RQ4_GMM_Best_Prior`
in Claims table below。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+50 lines:
  `_render_reverse_section_panel` + try-except load mapping + body wire
  + CSS `.reverse-section` + `.rev-table` + `.rev-cids`)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case M, +docstring,
  +main count to 13)
- 验证: 13/13 cases pass; pre-commit hook freeze 13 invariants ✓
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
  IDs cross-referenced
- **iter #124** — paper_audit_id_mapping.json machine-readable forward index
- **iter #125** — dashboard `<link rel='alternate'>` to sidecar
- **iter #126** — §1 contribution bullets inline cross-link
- **iter #127** — §5 Limitations inline cross-link
- **iter #128** — sidecar reverse_section_index (section → audit IDs)
- **iter #129** — dashboard reverse_section panel (bidirectional anchor
  links paper §X ↔ claim-{id}) ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases
- **iter #121** — 7 cases
- **iter #122** — 8 cases
- **iter #123** — 9 cases
- **iter #124** — 10 cases
- **iter #125** — 11 cases
- **iter #128** — 12 cases
- **iter #129** — 13 cases (+ dashboard reverse section panel) ✓ 本轮

每个新 cross-link affordance 都应该 regression test 冻结。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
