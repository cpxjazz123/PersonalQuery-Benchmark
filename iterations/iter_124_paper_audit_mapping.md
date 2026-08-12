# Iteration #124 — Paper ↔ audit claim ID machine-readable mapping sidecar

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_paper_audit_mapping.py` (new, ~125 lines); `_run_audit_ci.sh` adds [4/4] mapping regeneration step; `_smoke_audit_regression.py` extended to 10 cases (Case J)
**prior**: iter #119/#123 added inline backticked cross-refs in paper (Table 1/2/3 footnotes + §1 paragraph). Cross-refs 现在 human-readable 但 machine-parse 不便。reviewer 想 grep / diff / programmatically verify paper-to-audit link 需要 inline prose parse,易错。

## §A 审稿意见

iter #119/#123 把 17/17 audit IDs 加 inline backtick cross-refs 进 paper,
但 cross-ref 索引是 prose-only:

- reviewer 想 "paper section X 提到哪些 audit IDs" → 必须用 regex 扫 paper,
  没有 cached index。
- reviewer 想 "audit claim Y 在 paper 哪里引用" → 必须 grep audit ID name
  全文,易错(`RQ1_Table1_Hit10` 跟 prose variable 同形)。
- CI 想 verify "新增 audit claim 自动加 paper cross-ref" → 必须人工 enforce。

后果: cross-ref coverage 是 iter #123 frozen baseline 的一部分 (Case I),
但 cross-ref *index* 是 implicit,reviewer/CI 没法 query。

iter #124 fix: generate `paper_audit_id_mapping.json` machine-readable sidecar,
每个 audit ID 列出所有 paper line numbers + 50-char context snippets。
reviewer 可 grep / diff / jq;CI 可 verify unmapped_audit_ids / unmapped_paper_ids
都为空。

## §B 本轮 (iter #124) 改动

### PersoanlQuery/_generate_paper_audit_mapping.py (NEW, ~125 lines)

```python
_AUDIT_ID_PREFIX = re.compile(
    r"`(RQ[1-9]_[A-Za-z0-9_.%]+|Sec[1-9]_[A-Za-z0-9_]+"
    r"|Pipeline_[A-Za-z0-9_]+|BPE_[A-Za-z0-9_]+"
    r"|UserFilter_[A-Za-z0-9_]+)`"
)

def _extract_audit_ids_with_locations(paper):
    """For each audit ID found in backticks, record line numbers + context snippets."""
    for line_no, line in enumerate(paper.splitlines(), start=1):
        for m in _AUDIT_ID_PREFIX.finditer(line):
            cid = m.group(1)
            ctx = line[max(0, m.start()-25):min(len(line), m.end()+25)].strip()
            out.setdefault(cid, []).append({"line": line_no, "context": ctx})
```

Output schema:

```json
{
  "generated_at": "2026-07-20T22:04:42+00:00",
  "paper_path": "PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md",
  "audit_json_path": "result/personal_query/iterations/paper_claims_audit.json",
  "n_audit_claims": 17,
  "n_paper_locations": 26,
  "coverage": {
    "<audit_id>": {
      "claim_status": "verified|discrepant|...",
      "claim_section": "<paper section>",
      "paper_locations": [{"line": <int>, "context": "<snippet>"}, ...]
    }
  },
  "unmapped_audit_ids": [],
  "unmapped_paper_ids": [],
  "summary": {"verified_covered": 6, "discrepant_covered": 4, ...}
}
```

§1 paragraph coverage (iter #123) 用 explicit annotation 在 output:
对 6 §2.2 infrastructure claim IDs 无 inline backtick 但在 §1 paragraph 提到,
加 `{"line": 40, "context": "§1 reproducibility-audit paragraph (iter #123)"}`。

### PersoanlQuery/_run_audit_ci.sh

CI 加 [4/4] step:

```bash
# Step 4: regenerate paper ↔ audit claim ID mapping sidecar (iter #124)
echo "[4/4] Regenerating paper ↔ audit claim ID mapping sidecar..."
if python3 "${MAPPING_SCRIPT}"; then
    echo "  PASS  paper_audit_id_mapping.json regenerated"
```

Header / step count 同步 3→4。

### PersoanlQuery/_smoke_audit_regression.py

加 Case J freeze iter #124 behavior:

```python
def case_j_paper_audit_mapping_sidecar():
    """Case J (iter #124): paper_audit_id_mapping.json covers all 17 audit IDs,
    no unmapped in either direction."""
    mapping = json.load(open(mapping_path))
    assert mapping["n_audit_claims"] == 17
    assert len(mapping["coverage"]) == 17
    assert mapping["unmapped_audit_ids"] == []
    assert mapping["unmapped_paper_ids"] == []
    for status in ["verified_covered", "discrepant_covered", "degenerate_covered",
                   "partial_covered", "unverified_covered"]:
        assert status in mapping["summary"] and mapping["summary"][status] >= 1
    assert "RQ1_Table1_Hit10" in mapping["coverage"]
    assert len(mapping["coverage"]["RQ1_Table1_Hit10"]["paper_locations"]) >= 1
```

docstring + main() 同步更新到 10 cases。

### README.md

dashboard feature list 上方加一行:

```diff
+ - `result/personal_query/iterations/paper_audit_id_mapping.json` (machine-readable paper ↔ audit claim ID index; iter #124 — for each audit ID, list every paper line that cites it; flags `unmapped_audit_ids` and `unmapped_paper_ids`)
```

## §C 关键改动点

1. **Two-pass extraction**: (1) regex 找 inline backticks,记录 line + context;
   (2) §1 paragraph enumeration explicit annotation (因为不是 backticked inline)。
   这样 17/17 audit IDs 都至少 1 paper location。

2. **Bidirectional validation**: `unmapped_audit_ids` (audit claims 没 paper cross-ref)
   + `unmapped_paper_ids` (paper cross-ref 不在 audit JSON) 双方向 verify。
   Case J 断言两者都为空。

3. **Status-keyed summary**: `summary.{verified,discrepant,degenerate,partial,unverified}_covered`
   让 reviewer 一眼看出每个 status 至少 1 cross-ref。如果 future 加 audit claim
   新 status (e.g. `regression_detected`),Case J 会 catch 缺 cross-ref。

4. **Context snippet 50 chars**: 25-char before + 25-char after,够 reviewer
   看 audit ID context 但不长到 noise。

5. **§1 annotation explicit**: 不伪装成 inline backtick,用 `{"line": 40,
   "context": "§1 reproducibility-audit paragraph (iter #123)"}` 显式标注,
   reviewer 知道这是 paragraph-level reference 不是 inline backtick。

6. **No new audit claims**: iter #124 只是 cross-link indexing 改进,不改
   audit JSON,不改 status,不改 frozen baseline (0/6/4/1/1/5/0)。

## §D 测试

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108 + #124) ===
Repo root: /home/wlia0047/ar57/wenyu

[1/4] Regenerating audit JSON...
  PASS  audit JSON regenerated

[2/4] Running regression smoke test...
=== paper_claims_audit regression smoke test (iter #105, #121, #122, #123, #124) ===

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'                                PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965               PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message                                 PASS
Case D: --strict --json-only (full audit) expects exit 1                                         PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)                             PASS
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)                        PASS
Case G: dashboard HTML has anchors + status legend (iter #121)                                   PASS
Case H: dashboard HTML <title> + <meta name='description'> (iter #122)                           PASS
Case I: paper backticked cross-refs cover all 17 audit IDs (iter #123)                           PASS
Case J: paper_audit_id_mapping.json covers 17/17 + 0 unmapped (iter #124)                         PASS

All 10 cases passed. Audit CLI frozen baseline verified.

[3/4] Regenerating HTML dashboard...
  PASS  dashboard regenerated

[4/4] Regenerating paper ↔ audit claim ID mapping sidecar...
Wrote: /home/wlia0047/ar57/wenyu/result/personal_query/iterations/paper_audit_id_mapping.json
  n_audit_claims: 17
  n_paper_locations: 26
  unmapped_audit_ids: []
  unmapped_paper_ids: []
  summary: {'degenerate_covered': 1, 'discrepant_covered': 4, 'partial_covered': 1, 'unverified_covered': 5, 'verified_covered': 6}
  PASS  paper_audit_id_mapping.json regenerated

=== AUDIT CI PASSED ===
```

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_paper_audit_mapping.py` (NEW, ~125 lines)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case J, +docstring,
  +main count to 10)
- CI: `PersoanlQuery/_run_audit_ci.sh` (+[4/4] step, header / step count 3→4)
- 输出: `result/personal_query/iterations/paper_audit_id_mapping.json`
  (gitignored — regenerated by CI)
- 文档: `README.md` (dashboard feature list 上方加 mapping sidecar 行)
- 验证: 10/10 cases pass; pre-commit hook freeze 10 invariants (4 + 3 + 1 + 1 + 1) ✓
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
  (programmable cross-ref coverage + bidirectional validation) ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta)
- **iter #123** — 9 cases (+ paper cross-ref coverage)
- **iter #124** — 10 cases (+ paper-audit mapping sidecar) ✓ 本轮

每个新 cross-link affordance 都应该 regression test 冻结,否则 reviewer
第一次 read paper 时发现 paper-to-dashboard link 失效不会被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
