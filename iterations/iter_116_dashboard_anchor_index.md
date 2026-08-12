# Iteration #116 — Dashboard deep-link anchors + claims index sidebar

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add anchors + index)
**prior**: dashboard had 17 claim rows + detail rows but no deep links; this iter enables paper → dashboard jump

## §A 审稿意见

iter #108–#115 dashboard 视觉化完整 (status badges, provenance, vc colors,
matched_files, generated_at),但 reviewer 从 paper §3 跳到 dashboard 找特定
claim 没快速通道:

- 17 claim 主行 + 17 detail 行,每个长 ~10 lines,scroll 找 `RQ3_Fleiss_Kappa_0.72`
  要翻很久
- paper §3 引用 claim 时,reviewer 想 verify "paper 怎么说 → dashboard 怎么验"
  没有 deep link
- reviewer 想 "哪些 claim 是 unverified?" 需肉眼 scan 5 个灰色 badge

iter #116 fix:
1. 每个主 claim row 加 `id="claim-{claim_id}"` (HTML anchor)
2. dashboard header 加 collapsible claims index,每条 link 跳到对应 anchor
3. link 旁边显示 status color dot,一眼定位 type

## §B 本轮 (iter #116) 改动

### _generate_audit_dashboard.py

**`_render_claim_row`**: 主 row 加 `id`:

```python
return f"""
<tr id='claim-{c['id']}'>
  <td><code>{c['id']}</code></td>
  ...
```

**`_render_html`**: 生成 claims index HTML:

```python
claim_index_items = "\n".join(
    f"<li><a href='#claim-{c['id']}'><code>{c['id']}</code> <span class='idx-status' style='color:{STATUS_COLORS.get(c['status'], (c['status'], '#9e9e9e'))[1]}'>●</span></li>"
    for c in claims
)
claim_index_html = f"<details class='claim-index'><summary><strong>Claims index ({len(claims)})</strong></summary><ul>{claim_index_items}</ul></details>"
```

**CSS**: `.claim-index` (折叠 sidebar, 3-column layout for 17 claims):

```css
.claim-index { background: #f5f5f5; padding: 8px 12px; border-radius: 4px; ... }
.claim-index ul { columns: 3; ... }
.idx-status { font-size: 14px; }
```

## §C 关键改动点

1. **`<details>` collapsible**: 默认折叠,reviewer 不需要时不影响 scroll;
   click expand 后 17 claims 以 3-column layout 排列。
2. **Color dot per claim**: index 列表每条 link 旁边一个 `●` colored by status
   (verified=green, discrepant=orange, degenerate=red, unverified=gray)。
   reviewer 一眼能 scan "5 unverified, 4 discrepant, 6 verified, ..."。
3. **Standard HTML anchors**: no JS required, browser native deep linking
   (`paper_claims_audit_dashboard.html#claim-RQ3_Fleiss_Kappa_0.72` works
   directly via URL)。
4. **CSS columns**: 17 claims 在 3-column layout 下很 compact (~6 rows tall),
   比 inline list 节省 vertical space。

## §D 测试

```bash
$ grep -oE 'href=.#claim-[A-Za-z0-9_]+.' result/personal_query/iterations/paper_claims_audit_dashboard.html | head -3
href='#claim-RQ1_Table1_Hit10'
href='#claim-RQ1_Delta_Range'
href='#claim-RQ2_Table1_Drop'

$ grep -oE 'id=.claim-[A-Za-z0-9_]+.' result/personal_query/iterations/paper_claims_audit_dashboard.html | head -3
id='claim-RQ1_Table1_Hit10'
id='claim-RQ1_Delta_Range'
id='claim-RQ2_Table1_Drop'
```

17 anchors + 17 index links match。Reviewer 现在能:
- 浏览器地址栏 `dashboard.html#claim-RQ3_Fleiss_Kappa_0.72` 直接跳到该行
- dashboard 顶部 expand "Claims index (17)" 看所有 claim 一览
- index 每条 claim 旁边 color dot 一眼区分 status

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+anchor on main row + index sidebar + CSS)
- 验证: 17 anchors + 17 index links ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 dashboard evolution 的关系

dashboard UX 演进:
- **#108** — sortable table + per-claim detail
- **#109** — rel_delta percent fix
- **#110** — vc status colors
- **#112** — provenance panels
- **#113** — generated_at timestamp
- **#115** — matched_files for degenerate
- **#116** — anchors + index sidebar (navigation) ✓ 本轮

dashboard 现在 enable 4 类 navigation:
1. **Top-down**: Claims index → click claim → jump to detail
2. **Bottom-up**: scroll → claim → read detail
3. **Direct URL**: `dashboard.html#claim-{id}` (browser native anchor)
4. **Cross-reference**: paper §3 mention "RQ3_Fleiss" → dashboard deep link

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample