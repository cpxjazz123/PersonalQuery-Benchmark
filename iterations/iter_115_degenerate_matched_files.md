# Iteration #115 — Degenerate vc `matched_files` provenance listing

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` (store matched_files in vc result) + `_generate_audit_dashboard.py` (render file list)
**prior**: dashboard provenance (iter #112) showed expected_outputs + code_evidence but not the actual files that the audit attempted to extract from

## §A 审稿意见

iter #112 加 provenance (`expected_outputs` + `code_evidence`) 到 dashboard,
但 degenerate vc 仍然只显示 `selector ... returned no non-NaN values
across 3 file(s)` — reviewer 不知道是哪 3 个文件。

RQ1_Table1_Hit10 是仅有的 degenerate claim (Stage 6 query pool 仅 2 queries
不够 bootstrap,导致 `rows[*].per_metric.*.mean` selector 返回空)。 Reviewer
需要知道:
1. 哪 3 个文件被 selector 试过
2. 这 3 个文件**实际存在**(证明 selector 真的执行了,不是 audit script bug)
3. 这 3 个文件的 path 是否 match `expected_outputs` 里的 glob patterns

iter #115 fix: 把 `matched_files` (relative paths list) 存到 degenerate vc
result,audit CLI `--verbose` + dashboard 都 print list。

## §B 本轮 (iter #115) 改动

### §B.1 paper_claims_audit.py — store matched_files in degenerate result

```python
if extracted is None:
    return {
        "status": "degenerate",
        ...
        "note": f"selector '{check['selector']}' returned no non-NaN values across {len(matched_files)} file(s)",
        "matched_files": [str(f.relative_to(REPO_ROOT)) for f in matched_files],  # iter #115
    }
```

### §B.2 paper_claims_audit.py — print matched_files in --verbose

```python
if vc.get("matched_files"):  # iter #115: list degenerate-matched files
    for mf in vc["matched_files"]:
        print(f"    matched_file={mf}")
```

### §B.3 _generate_audit_dashboard.py — render matched_files in vc panel

```python
if vc.get("matched_files"):  # iter #115: list degenerate-matched files
    detail_html += "<div class='vc-matched'><strong>matched_files (degenerate):</strong><ul>"
    for mf in vc["matched_files"]:
        detail_html += f"<li><code>{mf}</code></li>"
    detail_html += "</ul></div>"
```

+ CSS `.vc-matched` (灰色小字 ul).

## §C 关键改动点

1. **Reviewer 三层 audit trace** (现在完整): claim → expected_outputs glob → 实际 matched files
   → selector 提取失败原因。Before iter #115, reviewer 看到 `3 file(s)`
   是 abstract count,无法 trace。
2. **Empty list when not degenerate**: 只在 `extracted is None` (degenerate path)
   才存 matched_files,其他情况 (value_match / value_mismatch) 用
   source_file 单文件即可,不重复。
3. **CLI + dashboard 一致**: --verbose 和 dashboard 显示同样的 matched_files 列表,
   不需要 reviewer 切换工具。

## §D 测试

```bash
$ python3 PersoanlQuery/paper_claims_audit.py --claim-id RQ1_Table1_Hit10 --verbose
[RQ1_Table1_Hit10]  status=degenerate
  ...
  value_check: degenerate
    matched_file=result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json
    matched_file=result/personal_query/06_retrieval/Grocery_and_Gourmet_Food/retrieval_syntax_depth_summary.json
    matched_file=result/personal_query/09_noisy_retrieval/Grocery_and_Gourmet_Food/syntax_depth_correct_vs_noisy_results_pivot.json
    note=selector 'rows[*].per_metric.*.mean' returned no non-NaN values across 3 file(s)
```

Dashboard 验证 (HTML grep):
```html
<div class='vc-matched'><strong>matched_files (degenerate):</strong><ul>
<li><code>result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json</code></li>
<li><code>result/personal_query/06_retrieval/Grocery_and_Gourmet_Food/retrieval_syntax_depth_summary.json</code></li>
<li><code>result/personal_query/09_noisy_retrieval/Grocery_and_Gourmet_Food/syntax_depth_correct_vs_noisy_results_pivot.json</code></li>
</ul></div>
```

Reviewer 现在能 trace 完整 audit failure chain:
- expected_outputs globs (3 patterns with `<cat>` template)
- 实际 matched files (3 concrete paths)
- selector pattern 失败 (`rows[*].per_metric.*.mean` 全 NaN)

Full audit CI 测试 PASS (regression test + dashboard regen)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+1 line in `_validate_value_check` +3 lines in main verbose)
- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+5 lines render +2 lines CSS)
- 验证: degenerate claim shows 3 matched files ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra timeline 的关系

dashboard provenance evolution:
- **#112** — expected_outputs + code_evidence (audit-scope metadata)
- **#113** — generated_at timestamp (temporal provenance)
- **#115** — matched_files in degenerate vc (audit-execution trace) ✓ 本轮

audit JSON 现在有 3 层 provenance:
1. **Audit scope** (what was checked): expected_outputs + code_evidence
2. **Audit execution** (what actually happened): matched_files + extracted value
3. **Temporal** (when it ran): generated_at

Reviewer 看到 RQ1_Table1_Hit10 degenerate row 现在能完全诊断:
- 期望哪些 output (`expected_outputs` globs)
- 实际哪些文件被 selector 试过 (`matched_files`)
- selector pattern 失败原因 (`note`)
- 期望 files 路径的 template (`<cat>` placeholder → 实际 Grocery_and_Gourmet_Food)

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample