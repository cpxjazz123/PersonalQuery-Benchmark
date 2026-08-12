# Iteration #164 — Audit CLI `--stats-by-source-dir-and-status` + dashboard mirror panel (2D numeric matrix, source-dir × status)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: 完成 2D numeric matrix trio 最后一个 axis: source-dir (code) × status (frozen 7-status taxonomy)

## §A 审稿意见（尖锐批评）

### 问题 1: [iter #163 audit infra] — 2D matrix trio 仍缺一轴
**严重程度**: Suggestion
**具体批评**:
> Iter #163 完成 source-dir × section numeric matrix,但 2D matrix trio (status×section / source-dir×section / source-dir×status) 仍缺 source-dir × status NUMERIC axis (mirror iter #147 COUNT version)。 Reviewer 想 "which code-dir 在 discrepant status 下实测量级最差?" 必须 jq 拼装。 三 axis 完整 → code-erratum triage (which dir 在 which status 上 worst) + 趋势 monitoring (是否某 dir 突然在某 status 集中爆发) 都 trivial。

**证据支撑**:
- 2D matrix trio 已 status × section (iter #162) + source-dir × section (iter #163),缺 source-dir × status NUMERIC
- 8 source-dirs × 7 statuses = 56 cell matrix,worst cell = `02_writing_analysis × discrepant = 0.7415` (n=1)
- axis 三轴齐 → "discrepant status 是 02_writing_analysis 全 paper-claim 实测最差 status (n=1)" 一眼可见

### 问题 2: [iter #163 dashboard panel color discipline] — 14 panel colors 必须 distinct
**严重程度**: Suggestion
**具体批评**:
> Iter #163 共 13 panel bg color,iter #164 第 14 个 panel 选 `#e0f2f1` light teal/mint (冷色, 与 13 既有 color 全部 contrast 够); 原计划 `#fce4ec` light pink 与 iter #160 `#fff8e1` 较近 → 改用 `#e0f2f1` (mint 与既有 blue/teal 不同) 。

**证据支撑**:
- grep 13 panel bg color → 无 `#e0f2f1`
- `#e0f2f1` 冷色 (mint) 与既有 11 个 warm/cool color 形成 contrast,视觉区分足够
- 2D matrix status-side 视角 → 选 mint 与 iter #162 `#ede7f6` (status×section paper-side matrix, light purple) 视觉区分 (mint 偏 green vs purple 偏 blue)

### 问题 3: [iter #149 _primary_dir reuse + iter #161 status_order reuse] — single source of truth x2
**严重程度**: Suggestion
**具体批评**:
> Iter #164 CLI + dashboard 都复用 iter #149 `_primary_dir()` 提取 (first `PersoanlQuery/X/...` ref, strip `:function_name` suffix → top-level utility files bucket together) + iter #161 `status_order` 排序 (frozen 7-status taxonomy); 单 source of truth x2 → CLI 与 dashboard 对 source-dir aggregation + status taxonomy 完全一致; 避免 iter #149 → #158 → #163 → #164 helper 漂移 + iter #161 → #162 → #164 status_order 漂移。

**证据支撑**:
- iter #149 `_primary_dir()` 已 exist 并被 iter #158 + iter #163 复用
- iter #161 `status_order` 已 exist 并被 iter #162 复用
- iter #164 直接调用两函数,无新 helper
- frozen `_primary_dir` + frozen `status_order` → reviewer mental model 单 source of truth x2

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1 | `paper_claims_audit.py:1986` | 缺 2D source-dir × status numeric matrix (最后 2D axis) |
| 问题2 | `_generate_audit_dashboard.py` | 缺 14th panel + 14th distinct bg color |
| 问题3 | `_generate_audit_dashboard.py` | 缺 2D matrix HTML table (status-side 视角) |

## §C 本轮代码优化

1. **paper_claims_audit.py — `--stats-by-source-dir-and-status` CLI flag**:
   - docstring section + argparse add_argument (line 838-849)
   - handler block (~80 lines): bucket claims by (source-dir, status) via iter #149 `_primary_dir()` + iter #161 `status_order`, compute max abs per cell + claim count, print 8-row × 7-col matrix with source-dirs sorted by max abs desc + status_order (frozen taxonomy) for columns; em-dash for empty cells; cell format `mx=0.7415 (n=1)` for numeric cells, `n=4 (no abs)` for count-only cells

2. **_generate_audit_dashboard.py — `_render_stats_by_source_dir_and_status_panel()`**:
   - 新函数 mirror iter #164 CLI exactly (reuse iter #149 `_primary_dir` + iter #161 `status_order` + same per-cell max abs + same em-dash for empty)
   - 14th distinct panel bg color `#e0f2f1` light teal/mint (cool color contrast 够 vs 13 既有)
   - `sbs-cell-num` 红色 `#c62828` highlight 仅 apply 到 numeric cells → 视觉聚焦 worst cells (02_writing_analysis × discrepant=0.7415)
   - `<code>` tags wrap dir names + status names
   - `data-dir=` attribute on each `<tr>` for future JS hooks
   - wire 到 `_render_html` body after `stats_by_source_dir_and_section_html`
   - generator footer updated to `iter #122, #125, #129, #132, #153, #154, #156, #159, #160, #161, #162, #163, #164`

3. **_smoke_audit_regression.py — Case AY + Case AZ**:
   - Case AY (CLI): 验证 2D matrix header + 8 dirs × 7 statuses + 17 claims + 0.7415 in 02_writing_analysis × discrepant cell + 7 frozen status taxonomy columns + em-dash for empty + exits 2 on missing JSON
   - Case AZ (dashboard): 验证 panel class `stats-by-source-dir-and-status-panel` + 14th distinct color `#e0f2f1` + 02_writing_analysis/agreement_metrics.py rows 可见 + 7 frozen status taxonomy columns + `data-dir=` attribute
   - main count + docstring sync 到 52 cases + iter #164 加入 header

## §D 验证

- `python3 -m py_compile paper_claims_audit.py _generate_audit_dashboard.py _smoke_audit_regression.py` 全部通过
- 业务脚本 smoke test:
  - `python3 paper_claims_audit.py --stats-by-source-dir-and-status` → 8×7 matrix, top rows 02_writing_analysis + agreement_metrics.py 各含 discrepant cell `mx=0.7415/0.2700`
  - `python3 _generate_audit_dashboard.py` → HTML 含 14th panel + `sbs-cell-num` highlight
  - `python3 _smoke_audit_regression.py` → All 52 cases passed

## §E Git Commit

- `git commit -m "iter #164: Audit CLI --stats-by-source-dir-and-status (2D numeric matrix code×status) + dashboard mirror panel"` (待 commit)

## §F 剩余审稿意见（待后续迭代）

- iter #165 candidate: 2D matrix trio heatmap (cell bg color = max abs value)
- iter #165 candidate: 2D matrix sortable columns (click dir header → sort by max abs)
- iter #165 candidate: 反向 2D matrix (per-claim × per-section contribution detail)