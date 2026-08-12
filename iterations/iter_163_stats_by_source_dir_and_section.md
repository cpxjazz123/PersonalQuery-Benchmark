# Iteration #163 — Audit CLI `--stats-by-source-dir-and-section` + dashboard mirror panel (2D numeric matrix, source-dir × section)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: 完成 2D numeric matrix axis cross-cut: source-dir (code) × paper-section (paper)

## §A 审稿意见（尖锐批评）

### 问题 1: [iter #162 audit infra] — 2D matrix axes 不完整
**严重程度**: Suggestion
**具体批评**:
> Iter #162 完成 2D status × section numeric matrix,但仍缺 source-dir × section cross-cut — reviewer 想 "which code-dir produces worst discrepancies in which paper section?" 必须 jq 拼装。 Iter #147 (source-dir × status COUNT) + iter #148 (section × status COUNT) + iter #162 (status × section NUMERIC) 已存在, 缺 axis: source-dir × section NUMERIC 是 cross-axis (paper ↔ code) for code-erratum triage。

**证据支撑**:
- 2D matrix trio 已有 status × section (iter #162) → 需 source-dir × section 补全
- 8 source-dirs × 8 sections = 64 cell matrix,worst cell = `02_writing_analysis × §3.3 + Table 2 = 0.7415` (n=1)
- cross-axis 让 reviewer 一眼看出"02_writing_analysis 是 §3.3 paper-claim 实测最差 dir;agreement_metrics.py 第二差(0.27, n=3);其他 dir 全是 n>0 但 no abs (未触发 value check)"

### 问题 2: [iter #162 dashboard panel color discipline] — 13 panel colors 必须 distinct
**严重程度**: Suggestion
**具体批评**:
> Iter #132/153/154/156/159/160/161/162 共 12 panel bg color,iter #163 第 13 个 panel 选 `#fff3e0` light orange (暖色 but distinct from iter #153 `#ffe0b2` 较深的 orange); 原计划 `#f3e5f5` light purple 与 iter #129 reverse-section 重复 → 改用 `#fff3e0` (light orange 比 iter #153 `#ffe0b2` 更浅,与 11 个既有 color 区分)。

**证据支撑**:
- grep 12 panel bg color → 无 `#fff3e0`
- `#fff3e0` 比 iter #153 `#ffe0b2` 更浅 (alpha 较低),两者虽同 warm tone 但视觉 contrast 够
- 2D matrix code-side 视角 → 选 lighter tone 与 iter #162 paper-side matrix 视觉区分 (paper side 深一些 / code side 浅一些)

### 问题 3: [iter #149 _primary_dir reuse] — single source of truth
**严重程度**: Suggestion
**具体批评**:
> Iter #163 CLI + dashboard 都复用 iter #149 `_primary_dir()` 提取逻辑 (first `PersoanlQuery/X/...` ref, strip `:function_name` suffix → top-level utility files bucket together); 单 source of truth → CLI 与 dashboard 对 source-dir aggregation 完全一致; 避免 iter #149 → iter #158 → iter #163 三个 helper 函数漂移。

**证据支撑**:
- iter #149 `_primary_dir()` 已 exist 并被 iter #158 stats-by-source-dir CLI 复用
- iter #163 直接调用同一个函数,无新 helper
- frozen status taxonomy (iter #161) + frozen `_primary_dir` extraction → reviewer mental model 单 source of truth

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1 | `paper_claims_audit.py:1986` | 缺 2D source-dir × section numeric matrix |
| 问题2 | `_generate_audit_dashboard.py` | 缺 13th panel + 13th distinct bg color |
| 问题3 | `_generate_audit_dashboard.py` | 缺 2D matrix HTML table (code-side) |

## §C 本轮代码优化

1. **paper_claims_audit.py — `--stats-by-source-dir-and-section` CLI flag**:
   - docstring section + argparse add_argument (line 826-837)
   - handler block (~80 lines): bucket claims by (source-dir, section) via iter #149 `_primary_dir()`, compute max abs per cell + claim count, print 8-row × 8-col matrix with source-dirs sorted by max abs desc + sections sorted by max abs desc (worst axes on top-left); em-dash for empty cells; cell format `mx=0.7415 (n=1)` for numeric cells, `n=4 (no abs)` for count-only cells

2. **_generate_audit_dashboard.py — `_render_stats_by_source_dir_and_section_panel()`**:
   - 新函数 mirror iter #163 CLI exactly (reuse iter #149 `_primary_dir` extraction + same per-cell max abs + same em-dash for empty)
   - 13th distinct panel bg color `#fff3e0` light orange (warm but distinct from iter #153 `#ffe0b2` which is deeper orange)
   - `sbs-cell-num` 红色 `#c62828` highlight 仅 apply 到 numeric cells → 视觉聚焦 worst cells (02_writing_analysis × §3.3=0.7415 + agreement_metrics.py × §3.3=0.2700)
   - `<code>` tags wrap dir names + section names
   - `data-dir=` attribute on each `<tr>` for future JS hooks
   - wire 到 `_render_html` body after `stats_by_status_and_section_html`
   - generator footer updated to `iter #122, #125, #129, #132, #153, #154, #156, #159, #160, #161, #162, #163`

3. **_smoke_audit_regression.py — Case AW + Case AX**:
   - Case AW (CLI): 验证 2D matrix header + 8 dirs × 8 sections + 17 claims + 0.7415 in 02_writing_analysis × §3.3 cell + em-dash for empty + exits 2 on missing JSON
   - Case AX (dashboard): 验证 panel class `stats-by-source-dir-and-section-panel` + 13th distinct color `#fff3e0` + 02_writing_analysis/agreement_metrics.py/10_complexity_analysis 三个 dir rows 可见 + `data-dir=` attribute
   - main count + docstring sync 到 50 cases

## §D 验证

- `python3 -m py_compile paper_claims_audit.py _generate_audit_dashboard.py _smoke_audit_regression.py` 全部通过
- 业务脚本 smoke test:
  - `python3 paper_claims_audit.py --stats-by-source-dir-and-section` → 8×8 matrix, top rows 02_writing_analysis + agreement_metrics.py 各含 §3.3+Table 2 cell `mx=0.7415/0.2700`
  - `python3 _generate_audit_dashboard.py` → HTML 含 13th panel + `sbs-cell-num` highlight
  - `python3 _smoke_audit_regression.py` → All 50 cases passed

## §E Git Commit

- `git commit -m "iter #163: Audit CLI --stats-by-source-dir-and-section (2D numeric matrix code×paper) + dashboard mirror panel"` (待 commit)

## §F 剩余审稿意见（待后续迭代）

- iter #164 candidate: `--stats-by-source-dir-and-status` 2D matrix (mirror iter #147 count to numeric)
- iter #164 candidate: 2D matrix sortable columns (click section header → sort by max abs)
- iter #165 candidate: 2D matrix heatmap (cell bg color = max abs value)