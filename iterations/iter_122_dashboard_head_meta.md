# Iteration #122 — Dashboard `<title>` + `<meta description>` summary counts

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add `<title>` + `<meta description>` with summary counts + generated_at); extend `_smoke_audit_regression.py` to 8 cases
**prior**: iter #108-#121 built dashboard visualization stack (7 status badges / section breakdown / status legend / anchors / provenance / timestamp),but browser tab title was generic `paper_claims_audit dashboard` and `<head>` had no `<meta description>` — reviewer bookmark/tab/link-preview 看不到 audit state

## §A 审稿意见

iter #108-#121 累积了 11 个 dashboard affordance,但 browser-facing metadata 没做:

- 浏览器 tab 显示 generic `paper_claims_audit dashboard` — reviewer 同时开
  几个 dashboard/HTML 文件搞混。
- bookmark 没有 description — 三个月后再看 bookmark 不知道是哪个 audit
  pass。
- link-preview (Slack / Notion / email) 没有 description snippet — 同事
  click 之前不知道 dashboard 当前 state。
- `<head>` 缺 `meta generator` — 不知道是用哪个 script version 生成的。

后果: reviewer-facing metadata 比 on-page summary 还弱,affordance 11 个
里面 0 个是 browser-level。

iter #122 fix: dashboard `<title>` + `<meta description>` + `<meta generator>`
embed summary counts + generated_at ISO 8601 timestamp + script version。
Browser tab / bookmark / link-preview 全部立即显示 audit state。

## §B 本轮 (iter #122) 改动

### PersoanlQuery/_generate_audit_dashboard.py

`_render_html` 新增 short_counts 拼接 + title_str / meta_desc:

```python
status_order = ["verified_value_match", "verified", "discrepant",
                "degenerate", "partial", "unverified", "blocked"]
short_counts = " / ".join(
    f"{summary.get(s, 0)} {s.replace('_', ' ')}" for s in status_order
    if summary.get(s, 0) > 0
)
title_str = f"paper_claims_audit — {len(claims)} claims ({short_counts})"
meta_desc = (
    f"PersonalQuery paper-claims audit dashboard. {len(claims)} claims audited. "
    f"{short_counts}. Generated at {generated_at}."
)
```

`<head>` 改写:

```html
<head>
<meta charset='utf-8'>
<meta name='description' content='{meta_desc}'>
<meta name='generator' content='_generate_audit_dashboard.py (iter #122)'>
<title>{title_str}</title>
```

实际渲染:

```html
<title>paper_claims_audit — 17 claims (6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified)</title>
<meta name='description' content='PersonalQuery paper-claims audit dashboard. 17 claims audited. 6 verified / 4 discrepant / 1 degenerate / 1 partial / 5 unverified. Generated at 2026-07-20T22:04:42+00:00.'>
<meta name='generator' content='_generate_audit_dashboard.py (iter #122)'>
```

### PersoanlQuery/_smoke_audit_regression.py

加 Case H freeze iter #122 behavior (8/8 pass):

```python
def case_h_dashboard_head_meta():
    """Case H (iter #122): dashboard <title> + <meta description> embed summary counts + generated_at."""
    title_match = re.search(r"<title>([^<]+)</title>", html)
    assert "17 claims" in title
    assert "6 verified" in title
    assert "4 discrepant" in title
    assert "1 degenerate" in title
    desc_match = re.search(r"<meta name='description' content='([^']+)'>", html)
    assert "17 claims audited" in desc
    assert "Generated at" in desc
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", desc)
```

docstring + main() 同步更新到 8 cases。

### README.md

dashboard feature bullet list 加 1 行:

```diff
+ - `<title>` + `<meta name='description'>` embed summary counts + ISO 8601 generated_at (iter #122) — so browser tab / bookmark / link-preview shows concrete audit state
```

## §C 关键改动点

1. **Short counts 只列非零 status**: `verified_value_match` 当前 0,所以不
   显示。Reader 不会被 "0 verified_value_match" 噪声干扰。如果 future
   audit 跑出非零 `verified_value_match`,自动 appear。
2. **Title 用 em-dash (`—`)**: 与 paper / README / loop.md 一致;区别于
   hyphen `-`。
3. **Status name 下划线换成空格**: `verified_value_match` → `verified value match`,
   human-readable。
4. **ISO 8601 timestamp**: 用 `generated_at` 已有 field (iter #113),不重
   新生成 timestamp,dashboard 跟 JSON 完全同步。
5. **meta `generator` field**: 含 iter 编号,reviewer 知道 dashboard 由
   哪个 script version 渲染,方便 blame。
6. **Regression Case H 三个独立 assertion**: `<title>` 出现 + 计数出现
   + meta description 出现 + ISO timestamp 出现 — 任一退化都能 catch。

## §D 测试

```bash
$ python3 PersoanlQuery/_smoke_audit_regression.py
=== paper_claims_audit regression smoke test (iter #105, #121, #122) ===
Audit script: /home/wlia0047/ar57/wenyu/PersoanlQuery/paper_claims_audit.py
Frozen baseline: 2026-07-21 (audit summary 0/6/4/1/1/5/0)

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'                    PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965   PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message                     PASS
Case D: --strict --json-only (full audit) expects exit 1                             PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)                 PASS
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)            PASS
Case G: dashboard HTML has anchors + status legend (iter #121)                       PASS
Case H: dashboard HTML <title> + <meta name='description'> (iter #122)               PASS

All 8 cases passed. Audit CLI frozen baseline verified.
```

Full audit CI (regenerate + 8 regression + dashboard) PASS。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+9 lines: short_counts
  computation + 3 meta tags)
- 测试: `PersoanlQuery/_smoke_audit_regression.py` (+Case H, +docstring, +main count)
- 文档: `README.md` (+1 line dashboard feature)
- 验证: 8/8 cases pass; pre-commit hook 现在 freeze 8 invariants (4 + 3 + 1) ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra timeline 的关系

audit infra dashboard features timeline:
- **iter #108** — 7 status badges + summary header
- **iter #109** — rel_delta as percent
- **iter #110** — per-claim vc severity colors
- **iter #112** — provenance panels (expected_outputs + code_evidence)
- **iter #113** — generated_at timestamp
- **iter #115** — matched_files for degenerate
- **iter #116** — anchors + claims-index sidebar
- **iter #117** — section breakdown matrix
- **iter #118** — status legend
- **iter #122** — `<title>` + `<meta description>` browser-facing metadata ✓ 本轮

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend)
- **iter #122** — 8 cases (+ dashboard head meta) ✓ 本轮

每个新 affordance 都应该 regression test 冻结,这是 audit infra 不
silent-flip 的核心保证。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9–21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5–3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
