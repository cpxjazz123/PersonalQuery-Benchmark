# Iteration #128 — Section-level reverse index sidecar

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_paper_audit_mapping.py` add `reverse_section_index` (paper section → audit IDs cited); extend `_smoke_audit_regression.py` to 12 cases (Case L)
**prior**: iter #124 generated `paper_audit_id_mapping.json` mapping audit_id → paper_locations. Reverse index missing — reviewer wanted "which audit claims does §3.4 mention" without grepping; iter #128 adds section-level reverse index alongside existing forward index.

## §A 审稿意见

iter #124 mapping sidecar 是 forward index (audit_id → paper_locations)。reviewer
想反向查 ("paper section X 提到哪些 audit claims") 必须 grep coverage dict
或 search paper text:

- reviewer 第一次 read §3.4 (Prior Realism) 想 trace 所有 audit claims 在该 section
  → 必须 list all 17 audit IDs + check coverage[cid].paper_locations 哪些 line
  落在 §3.4 line range (170-173)。手工繁琐。
- reviewer 想 "§5 Limitations 涉及哪些 audit IDs" → 同样手工。
- bidirectional navigation 没 machine-readable form。

iter #128 fix: 在 `paper_audit_id_mapping.json` 同 sidecar 加 `reverse_section_index`,
key 是 paper section label (e.g. "3.4 Prior Realism of the User Style Distribution"),
value 是 sorted list of audit IDs cited 在该 section 内 (deduped)。
reviewer 直接查 `"3.4 Prior Realism of the User Style Distribution"` → 6 audit IDs。

## §B 本轮 (iter #128) 改动

### PersoanlQuery/_generate_paper_audit_mapping.py

新增 3 helper functions (~50 lines):

```python
_SECTION_HEADER = re.compile(r"^(#{1,3})\s+(.+?)\s*$")

def _build_section_spans(paper):
    """Return list of (section_label, start_line, end_line) for each header."""
    headers = []
    for line_no, line in enumerate(paper.splitlines(), start=1):
        m = _SECTION_HEADER.match(line)
        if m:
            label = m.group(2).strip()
            headers.append((line_no, label))
    spans = []
    for i, (line_no, label) in enumerate(headers):
        end = headers[i + 1][0] - 1 if i + 1 < len(headers) else len(paper.splitlines())
        spans.append((label, line_no, end))
    return spans

def _line_to_section(line_no, spans):
    """Map 1-indexed line to enclosing section label (last match wins)."""
    label = None
    for sec_label, start, end in spans:
        if start <= line_no <= end:
            label = sec_label
    return label

def _build_reverse_section_index(coverage, spans):
    """For each section label, list audit IDs cited within it (sorted, deduped)."""
    out = {}
    for cid, info in coverage.items():
        for loc in info["paper_locations"]:
            sec = _line_to_section(loc["line"], spans)
            if sec is None:
                continue
            out.setdefault(sec, set()).add(cid)
    return {sec: sorted(ids) for sec, ids in sorted(out.items())}
```

`main()` wire reverse section index into output doc:

```python
spans = _build_section_spans(paper)
reverse_section_index = _build_reverse_section_index(coverage, spans)
out_doc = {
    ...
    "coverage": coverage,
    "reverse_section_index": reverse_section_index,
    "n_sections_with_audit_refs": len(reverse_section_index),
    ...
}
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case L freeze iter #128 behavior:

```python
def case_l_reverse_section_index():
    """Case L (iter #128): paper_audit_id_mapping.json reverse_section_index maps each
    paper section to the audit IDs cited within it (bidirectional navigation)."""
    mapping = json.load(open(mapping_path))
    assert "reverse_section_index" in mapping
    idx = mapping["reverse_section_index"]
    assert mapping["n_sections_with_audit_refs"] == len(idx)
    total_citations = sum(len(ids) for ids in idx.values())
    assert total_citations <= mapping["n_paper_locations"]  # dedup makes it <=
    # §3.3 (LLM and Human Quality Validation) must contain RQ3_Fleiss_Kappa_0.72
    sec33_keys = [k for k in idx if k.startswith("3.3 ")]
    assert len(sec33_keys) == 1
    assert "RQ3_Fleiss_Kappa_0.72" in idx[sec33_keys[0]]
```

docstring + main() 同步更新到 12 cases。

## §C 关键改动点

1. **Forward + reverse index 共 sidecar**: 不 split 到两个 file,
   因为 sidecar regeneration 已经 atomic (CI [4/4] step 单 write)。
   reviewer 用 jq `.coverage` 或 `.reverse_section_index` 任意 access。

2. **`_line_to_section` 用 last-match-wins**: nested sections (## §3 inside
   ### §3.1) 让 line 可能 match 多个 section span; last match 是最 specific
   (deepest header)。当前 paper 没有 nested subsection,所以实际只 match
   1 header / line,但 logic 支持 future nested structure。

3. **`total_citations <= n_paper_locations`**: reverse index dedupes
   per-section,所以 multiple locations 同 audit_id 在同 section 只 count 1。
   `n_paper_locations` 是 forward index 总 location count。两者关系:
   `total_citations = n_paper_locations - (audit IDs cited multiple times in same section)`。
   Case L 用 `<=` 而不是 `==` 防止 future paper refactor 增加 same-section
   duplicate citations break test。

4. **Section label 是 header text 去掉 `#`**: `"3.4 Prior Realism of the
   User Style Distribution"` 直接对应 paper §3.4 标题。reviewer 可 copy-paste
   sidecar key 到 paper toc 验证 section 存在。

5. **5 sections indexed (frozen baseline 2026-07-21)**:
   - **§1 Introduction**: 17 audit IDs (跨 4 contribution bullets + 9 §1 paragraph + 4 footnotes)
   - **§3.1 Experimental Setup**: 3 audit IDs (`RQ1_Delta_Range` + `RQ1_Table1_Hit10` + `RQ2_Table1_Drop`,从 contribution bullet 1)
   - **§3.3 LLM and Human Quality Validation**: 4 RQ3 audit IDs
   - **§3.4 Prior Realism of the User Style Distribution**: 6 (`RQ4_GMM_Best_Prior` + 5 `Sec2_*`)
   - **§5 Conclusion**: 11 audit IDs (Limitations cross-link 回溯)

6. **No new audit claims**: iter #128 只是 cross-link indexing 改进,
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
Case L: mapping sidecar reverse_section_index (iter #128)                                        PASS

All 12 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...                                  PASS
[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...
  n_audit_claims: 17
  n_paper_locations: 52
  n_sections_with_audit_refs: 5
  unmapped_audit_ids: []
  unmapped_paper_ids: []
  summary: {'degenerate_covered': 1, 'discrepant_covered': 4, 'partial_covered': 1, 'unverified_covered': 5, 'verified_covered': 6}
  PASS  paper_audit_id_mapping.json regenerated

=== AUDIT CI PASSED ===
```

§3.4 reverse index: `["RQ4_GMM_Best_Prior", "Sec2_20dim_syntactic_features",
"Sec2_5_attrs_per_query", "Sec2_95pct_holdout_threshold",
"Sec2_K2_GMM_per_user", "Sec2_K8_clusters_BIC_AIC_silhouette"]`

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_paper_audit_mapping.py` (+50 lines:
  `_SECTION_HEADER` regex + `_build_section_spans` + `_line_to_section`
  + `_build_reverse_section_index`; wire into main + output)
- 输出: `result/personal_query/iterations/paper_audit_id_mapping.json`
  +`reverse_section_index` + `n_sections_with_audit_refs` (gitignored,
  regenerated by CI)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case L, +docstring,
  +main count to 12)
- 验证: 12/12 cases pass; pre-commit hook freeze 12 invariants (4 + 3 + 1 + 1 + 1 + 1 + 1) ✓
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
- **iter #124** — paper_audit_id_mapping.json machine-readable forward index
  (audit_id → paper_locations)
- **iter #125** — dashboard `<link rel='alternate'>` to sidecar
- **iter #126** — §1 contribution bullets inline cross-link
- **iter #127** — §5 Limitations inline cross-link
- **iter #128** — sidecar reverse_section_index (section → audit IDs)
  bidirectional navigation ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta)
- **iter #123** — 9 cases (+ paper cross-ref coverage)
- **iter #124** — 10 cases (+ paper-audit mapping sidecar forward)
- **iter #125** — 11 cases (+ dashboard link rel alternate)
- **iter #126** — 11 cases + Case I extended (4 contribution bullets)
- **iter #127** — 11 cases + Case I extended (5 Limitations items)
- **iter #128** — 12 cases (+ reverse_section_index) ✓ 本轮

每个新 cross-link affordance 都应该 regression test 冻结,否则 reviewer
第一次 read paper 时发现 paper-to-dashboard link 失效不会被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
