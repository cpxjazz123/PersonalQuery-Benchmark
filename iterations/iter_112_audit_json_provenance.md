# Iteration #112 — Audit JSON provenance enrichment (expected_outputs + code_evidence)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` (extend JSON output) + `PersoanlQuery/_generate_audit_dashboard.py` (render provenance)
**prior**: iter #110 deferred finding; this iter surfaces audit provenance to reviewer

## §A 审稿意见

iter #108–#110 dashboard 视觉化已经很好 (status badges + rel_delta +
vc color coding),但 reviewer 翻 dashboard 还是不知道 **audit 期望哪些
output 文件** 或 **哪些 script 实现这个 claim**。

iter #93 时 paper_claims_audit.py 的 CLAIMS 数组就已经定义了 `expected_outputs`
(glob patterns 列表) 和 `code_evidence` (script:function references 列表),
但 main() 的 entry dict 只写 6 个 key:
```python
entry = {"id": ..., "section": ..., "claim": ..., "status": ..., "reason": ...}
if result["value_check_results"]:
    entry["value_check_results"] = ...
if "paper_text_caveat" in claim:
    entry["paper_text_caveat"] = ...
```
→ `expected_outputs` + `code_evidence` 永远不进 JSON, dashboard 也就
无法显示。

后果: reviewer 看 dashboard 知道 RQ1_Table1_Hit10 是 degenerate,但不知道
audit 期望检查的是 `bootstrap_delta_ci.json` 还是
`retrieval_syntax_depth_summary*.json`,也不知道是哪个 script 实现的。
→ 缺 audit scope provenance。

## §B 本轮 (iter #112) 改动

### §B.1 paper_claims_audit.py (extend JSON entry dict)

```python
entry = {"id": ..., "section": ..., "claim": ..., "status": ..., "reason": ...}
if result["value_check_results"]:
    entry["value_check_results"] = ...
if "paper_text_caveat" in claim:
    entry["paper_text_caveat"] = ...
# iter #112: include provenance so dashboard can show full audit scope.
if claim.get("expected_outputs"):
    entry["expected_outputs"] = claim["expected_outputs"]
if claim.get("code_evidence"):
    entry["code_evidence"] = claim["code_evidence"]
```

### §B.2 _generate_audit_dashboard.py (render provenance)

```python
if c.get("expected_outputs"):
    detail_html += "<div class='prov'><strong>expected_outputs:</strong><ul>"
    for gl in c["expected_outputs"]:
        detail_html += f"<li><code>{gl}</code></li>"
    detail_html += "</ul></div>"
if c.get("code_evidence"):
    detail_html += "<div class='prov'><strong>code_evidence:</strong><ul>"
    for ev in c["code_evidence"]:
        detail_html += f"<li><code>{ev}</code></li>"
    detail_html += "</ul></div>"
```

+ CSS `.prov` (purple left border, matching partial status color #7b1fa2,
与 partial claim type 区分开 — 但这里用紫色是代表 provenance 而非
partial status, 让 reviewer 知道是 audit metadata 不是 claim state)。

## §C 关键改动点

1. **JSON 层加字段, dashboard 层 render**: clean separation, dashboard 不必
   infer provenance from elsewhere。
2. **Globs + script references**: `expected_outputs` 是 glob 列表 (e.g.
   `result/personal_query/06_retrieval/<cat>/retrieval_syntax_depth_summary*.json`),
   `code_evidence` 是 `script:function` references (e.g.
   `PersoanlQuery/08_compare_all_domain/08_compare_p10_across_domains.py:print_08_hit10_table`)。
3. **16/17 + 17/17 coverage**: 所有 claim 都有 code_evidence (script
   references), 16/17 有 expected_outputs (1 claim `Pipeline_Skip_ColBERTv2_SPLADE`
   只有 flag config 没有具体 output paths)。

## §D 测试

```bash
$ python3 -c "
import json
doc = json.load(open('result/personal_query/iterations/paper_claims_audit.json'))
n_eo = sum(1 for c in doc['claims'] if c.get('expected_outputs'))
n_ce = sum(1 for c in doc['claims'] if c.get('code_evidence'))
print(f'expected_outputs: {n_eo}/{len(doc[\"claims\"])}')
print(f'code_evidence:   {n_ce}/{len(doc[\"claims\"])}')
"
expected_outputs: 16/17
code_evidence:   17/17

$ grep -oE "expected_outputs:" result/personal_query/iterations/paper_claims_audit_dashboard.html | wc -l
16
```

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+4 lines: include provenance fields in JSON)
- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+12 lines: render prov panels + CSS)
- 验证: 16/17 + 17/17 coverage ✓; dashboard 16 expected_outputs panels ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 iter #108–#111 的关系

dashboard 演进 timeline:
- **#108** — self-contained HTML, sortable table, per-claim detail
- **#109** — rel_delta percent fix (off-by-100)
- **#110** — vc status color coding (severity)
- **#112** — provenance panels (audit scope metadata) ✓ 本轮

完整 audit infra stack (12 iters):
- value-validation (#93), paper doc (#94-#99), §5 reproducibility (#103),
  CLI (#104), regression test (#105), pre-commit CI (#107), HTML dashboard
  (#108), rel_delta fix (#109), vc colors (#110), README sync (#111),
  provenance (#112)

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample