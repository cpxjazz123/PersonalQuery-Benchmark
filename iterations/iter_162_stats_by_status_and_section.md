# Iteration #162 — Audit CLI `--stats-by-status-and-section` + dashboard mirror panel (2D numeric matrix)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: 完成 audit infra 2D numeric matrix axis (status × section), 1D stats triangle (iter #157/158/161) 升级为 2D

## §A 审稿意见（尖锐批评）

### 问题 1: [iter #155-#161 audit infra] — 1D stats 已完整,缺 2D numeric cross-axis matrix
**严重程度**: Suggestion
**具体批评**:
> Iter #157 (`--stats-by-section`) + iter #158 (`--stats-by-source-dir`) + iter #161 (`--stats-by-status`) 完成 1D stats triangle; 但 iter #148 (`--summary-by-section-and-status`) 仅有 COUNT matrix,缺 NUMERIC matrix — reviewer 想 "which (status × section) intersection has worst max abs?" 必须 jq 拼装。 2D numeric matrix 是自然下一步。

**证据支撑**:
- iter #148 2D count matrix 是 paper section × status,但每格只有 count,无 max/mean abs
- iter #161 单 axis per-status numeric 完整但缺 cross-axis
- 7-status × 8-section = 56 cell matrix,可一眼看出"discrepant × §3.3+Table 2 = 0.7415"是 worst cell,其他 55 cells 都是 em-dash → 集中风险 confirmed

### 问题 2: [iter #161 dashboard panel color discipline] — 12 panel colors 必须 distinct
**严重程度**: Suggestion
**具体批评**:
> Iter #132/153/154/156/159/160/161 共 11 个 panel bg color,iter #162 第 12 个 panel 选 `#ede7f6` light purple/lavender (语义: 2D matrix = 复杂结构 → 紫色联想); 视觉与 11 个既有 color (warm tones + cyan + green + amber + pink) 区分; numeric cell 高亮 `#c62828` 红色吸引 attention on max abs cells。

**证据支撑**:
- grep 现有 11 panel bg color → 无 `#ede7f6`
- 2D matrix 是 complex view,选 cool tone `#ede7f6` 与暖色 alert panels (severity/discrepant) 区分
- `sbs-cell-num` 红色 class 仅 apply 到 max abs cells (`color: #c62828`),其他 cells 默认 text color → reviewer 视觉聚焦 worst cell

### 问题 3: [iter #161 frozen status order extension] — 2D matrix 用同一 taxonomy 保持一致性
**严重程度**: Suggestion
**具体批评**:
> Iter #162 2D matrix rows = iter #161 frozen 7-status taxonomy (verified_value_match/verified/discrepant/degenerate/partial/unverified/blocked); sections sorted by max abs desc → worst section (discrepant × §3.3+Table 2) appears as first non-empty column; 与 iter #161 iter #134/147/148 共享 frozen order → 视觉对称。

**证据支撑**:
- 1D stats-by-status 用 frozen status order (verified_value_match first → blocked last)
- 2D matrix 也用 same order → reviewer mental model 单 source of truth
- sections 按 max abs desc sort → 视觉 worst-on-left → 眼动扫描自然路径

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1 | `paper_claims_audit.py:1889` | 缺 2D status × section numeric matrix |
| 问题2 | `_generate_audit_dashboard.py` | 缺 12th panel + 12th distinct bg color |
| 问题3 | `_generate_audit_dashboard.py` | 缺 2D matrix HTML table |

## §C 本轮代码优化

1. **paper_claims_audit.py — `--stats-by-status-and-section` CLI flag**:
   - docstring section + argparse add_argument (line 815-825)
   - handler block (~80 lines): bucket claims by (status, section), compute max abs per cell, print 7-row × 8-col matrix with status taxonomy rows + section columns sorted by max abs desc; em-dash for empty cells; cell format `mx=0.7415 (n=4)` for numeric cells, `n=4 (no abs)` for count-only cells

2. **_generate_audit_dashboard.py — `_render_stats_by_status_and_section_panel()`**:
   - 新函数 mirror iter #162 CLI exactly (same status taxonomy + same per-cell max abs + same em-dash for empty)
   - 12th distinct panel bg color `#ede7f6` light purple/lavender (cool tone, distinct from iter #161 cyan)
   - `sbs-cell-num` 红色 `#c62828` highlight 仅 apply 到 numeric cells → 视觉聚焦 worst cell
   - `<code>` tags wrap status names + section names
   - monospace font + `font-variant-numeric: tabular-nums`
   - wire 到 `_render_html` body after `stats_by_status_html`
   - generator footer updated to `iter #122, #125, #129, #132, #153, #154, #156, #159, #160, #161, #162`

3. **_smoke_audit_regression.py — Case AU + Case AV**:
   - Case AU (CLI): 验证 2D matrix header + 7 statuses × 8 sections + 17 claims + 0.7415 in discrepant × §3.3 + Table 2 cell + em-dash for empty + exits 2 on missing JSON
   - Case AV (dashboard): 验证 panel class `stats-by-status-and-section-panel` + 12th distinct color `#ede7f6` + `sbs-cell-num` highlight + section column header
   - main count + docstring sync 到 48 cases

## §D 验证

- `python3 -m py_compile paper_claims_audit.py _generate_audit_dashboard.py _smoke_audit_regression.py` 全部通过
- 业务脚本 smoke test:
  - `python3 paper_claims_audit.py --stats-by-status-and-section` → 7×8 matrix, discrepant row × §3.3 + Table 2 column = `mx=0.7415 (n=4)`, 其他 55 cells em-dash or count-only
  - `python3 _generate_audit_dashboard.py` → HTML 含 12th panel + `sbs-cell-num` highlight
  - `python3 _smoke_audit_regression.py` → All 48 cases passed

## §E Git Commit

- `git commit -m "iter #162: Audit CLI --stats-by-status-and-section (2D numeric matrix) + dashboard mirror panel"` (待 commit)

## §F 剩余审稿意见（待后续迭代）

- iter #163 candidate: `--stats-by-source-dir-and-section` 2D matrix (cross paper-section × code-dir)
- iter #163 candidate: dashboard 2D matrix sortable columns (click section header → sort by max abs)
- iter #164 candidate: dashboard 2D matrix color-coded cells (heatmap by max abs)