# Iteration #165 — 2D matrix heatmap (CLI `--heatmap-2d-matrices` + dashboard per-cell bg color)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: 给 2D matrix trio (status×section / source-dir×section / source-dir×status) 加 visual heatmap — reviewer 一眼 cross-validate 三 axis 数值一致性

## §A 审稿意见（尖锐批评）

### 问题 1: [iter #164 audit infra] — 2D matrix trio 缺 visual reinforcement
**严重程度**: Suggestion
**具体批评**:
> Iter #162/163/164 完成 2D numeric matrix trio,但 cell 内部仅 `mx=0.7415` 数字 + em-dash;reviewer 浏览器看 dashboard 时 0.7415 与 0.27 视觉权重相等(都是红字),需扫读 + mental compare → "which cell is worst?" 不是 at-a-glance; fix: 加 3-tier heatmap bg color (light yellow / orange / red) → worst cells (≥0.5) 红底立刻跳出来 → 视觉聚焦 zero-overhead。

**证据支撑**:
- 2D matrix trio 已有 3 surface (status×section / source-dir×section / source-dir×status)
- 三 axis worst cell 都 = `02_writing_analysis × §3.3 = 0.7415` (cross-validate 一致性)
- second-worst = `agreement_metrics.py × §3.3 = 0.2700` (medium tier)
- 其他 dir 全是 n>0 但 no abs (count-only cells,无 tier 染色)

### 问题 2: [iter #164 heatmap tier threshold rationale] — 3-tier 选择 vs 5-tier 或 continuous gradient
**严重程度**: Suggestion
**具体批评**:
> Heatmap tier threshold 选 3-tier (light < 0.1 / medium 0.1-0.5 / dark ≥ 0.5) 而非 5-tier 或 continuous gradient (e.g., colorbrewer RdYlGn reverse 11 bins);rationale: 3-tier 与 §3.3 "0.7415 worst" + "0.27 second" + "all others near-zero" 三个 cluster 自然对齐;5-tier 在 17 value-checks 小样本下浪费 bins; continuous gradient 需 CSS `background: hsl(...)` interpolation + legend gradient bar (复杂度不必要)。

**证据支撑**:
- 17 value-checks total → 3-tier 饱和度刚好
- threshold 0.1 与 paper-claim "rel_delta_pct above 20 pct" (= 0.2 abs approx) 对齐 (light < 0.1 = negligible,medium 0.1-0.5 = actionable,dark ≥ 0.5 = blocking)
- iter #150 severity-tier (HIGH/MEDIUM/LOW) 3-tier 已存在 → heatmap 3-tier 与已有 mental model 一致

### 问题 3: [iter #149 + iter #161 reuse] — single source of truth x2 续延
**严重程度**: Suggestion
**具体批评**:
> Iter #165 CLI handler 复用 iter #149 `_primary_dir()` 提取 + iter #161 `status_order` (frozen 7-status taxonomy),与 iter #162/163/164 完全共享 source-dir extraction + status taxonomy; heatmap tier 计算 (`_heatmap_tier(mx)`) 是新逻辑,但 matrix 构造复用;单 source of truth x2 → CLI 与 dashboard 与 iter #162/163/164 cross-axis 完全一致。

**证据支撑**:
- iter #149 `_primary_dir()` 仍 exist,iter #165 直接复用
- iter #161 `status_order` 仍 exist,iter #165 直接复用
- 新 `_heatmap_tier` 函数 → 在 CLI + dashboard 各一份 (各 ~6 行,parallel 实现)

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1 | `_generate_audit_dashboard.py:715,795,867` | 2D matrix cell 缺 visual heatmap tier (numeric cells 仅 sbs-cell-num 红字) |
| 问题2 | `_generate_audit_dashboard.py:1140-1141` | CSS 缺 .hm-light/.hm-medium/.hm-dark tier 样式 |
| 问题3 | `paper_claims_audit.py:2180` | 缺 CLI flag 一次性打印 3 个 2D matrix with heatmap tier |

## §C 本轮代码优化

1. **_generate_audit_dashboard.py — 3-tier heatmap CSS + per-cell class**:
   - 新 helper `heatmap_class(mx)` (~9 lines): mx < 0.1 → `hm-light`, 0.1-0.5 → `hm-medium`, ≥ 0.5 → `hm-dark`, None → `hm-empty`
   - 修改 3 个 2D matrix cell-rendering 行 (line 715, 795, 867): 加 ` {heatmap_class(mx)}` suffix → cell class 变成 `sbs-cell sbs-cell-num hm-dark` etc.
   - 新 CSS rules (~10 lines): `.hm-light` `#fff3cd` light yellow / `.hm-medium` `#ffcc80` orange / `.hm-dark` `#e57373` red (`color: #b71c1c` 深红 text 增强对比) / `.hm-empty` `transparent` / `.hm-legend` + `.hm-legend-swatch` for legend strip
   - 3 个 panel footer 各加 `<div class='hm-legend'>Heatmap (iter #165): ...</div>` strip 显示 tier 含义
   - generator footer updated to `iter #122, #125, #129, #132, #153, #154, #156, #159, #160, #161, #162, #163, #164, #165`

2. **paper_claims_audit.py — `--heatmap-2d-matrices` CLI flag**:
   - docstring section + argparse add_argument (line 866-872)
   - handler block (~120 lines): 复用 iter #149 `_primary_dir` + iter #161 `status_order`;新 `_heatmap_tier(mx)` helper 返 `L/M/D` tier letter;`_build_matrix()` helper with row/col key lambda → 通用 3 matrix 构造;`_row_max_abs` / `_col_max_abs` / `_sort_rows` / `_sort_cols` 4 helpers 正确处理 row/col dim 各自的 max-abs 聚合 (修复 initial KeyError bug);`_print_matrix()` helper with heatmap tier prefix `[D] mx=0.7415 (n=4)` 等;一次打印 3 个 matrix (status×section / source-dir×section / source-dir×status) + legend strip;exit 0 when JSON found,exit 2 on missing JSON
   - 修复 initial bug: `_sort_by_max_abs` 错误按 row key 索引 (matrix 是 row-major → `_matrix[col_key]` 不存在) → 重构为 `_row_max_abs(matrix, rk)` 与 `_col_max_abs(matrix, rks, ck)` 双 helper + `_sort_rows` / `_sort_cols` 双 sort 正确处理不同 dim

3. **_smoke_audit_regression.py — Case BA + Case BB**:
   - Case BA (CLI): 验证 3 matrix headers + heatmap legend `[L]/[M]/[D]` + `[D] mx=0.7415` worst cell + `[M] mx=0.2700` second-worst + "value-checks" label + exits 2 on missing JSON
   - Case BB (dashboard): 验证 3-tier CSS classes + bg colors `#fff3cd` / `#ffcc80` / `#e57373` + `hm-dark` count ≥ 5 (≥3 matrices + 2 CSS rule occurrences) + `sbs-cell-num hm-dark` combined class + 3 legend strips
   - main count + docstring sync 到 54 cases + iter #165 加入 header

## §D 验证

- `python3 -m py_compile paper_claims_audit.py _generate_audit_dashboard.py _smoke_audit_regression.py` 全部通过
- 业务脚本 smoke test:
  - `python3 paper_claims_audit.py --heatmap-2d-matrices` → 3 matrices with `[D] mx=0.7415` + `[M] mx=0.2700` + legend strip;fix iteration: 修复 initial `_sort_by_max_abs` KeyError bug
  - `python3 _generate_audit_dashboard.py` → HTML 含 3 panel 各自的 heatmap legend strip + 7 `hm-dark` + 6 `hm-medium` + 4 `hm-light` + 1 `hm-empty` tier applications
  - `python3 _smoke_audit_regression.py` → All 54 cases passed

## §E Git Commit

- `git commit -m "iter #165: 2D matrix heatmap (CLI --heatmap-2d-matrices + dashboard hm-light/medium/dark tiers)"` (待 commit)

## §F 剩余审稿意见（待后续迭代）

- iter #166 candidate: heatmap sortable columns (click dir/section header → sort by max abs + heatmap rerenders)
- iter #166 candidate: 4th tier "very-dark" (≥0.8) for outlier cells + per-tier cell count badge in legend
- iter #166 candidate: heatmap export (per-matrix PNG via matplotlib for paper figures)