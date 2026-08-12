# Iteration #161 — Audit CLI `--stats-by-status` + dashboard mirror panel (stats triangle completion)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: 完成 audit infra stats triangle: section (iter #157) / source-dir (iter #158) / status (iter #161)

## §A 审稿意见（尖锐批评）

### 问题 1: [iter #155-#160 audit infra] — Per-axis numeric aggregate axis matrix不完整
**严重程度**: Suggestion
**具体批评**:
> 现有 audit infra 有 per-section (iter #157) + per-source-dir (iter #158) numeric aggregate，但缺 per-status axis — reviewer 想 "for each audit status, what's the worst abs_delta + mean abs?" 必须 jq 拼装。

**证据支撑**:
- iter #155 `--audit-stats` global aggregate + per-status vc counts 但 no per-status max/mean abs
- iter #157 per-section, iter #158 per-source-dir,缺 axis #3
- stats triangle (section × source-dir × status) 应 complete 才能 cross-validate per-axis consistency

### 问题 2: [paper_claims_audit.py:780 pre-existing] — `--help` 已 broken since iter #150
**严重程度**: Minor (discovered during iter #161)
**具体批评**:
> iter #150 severity-tier help text 含 `> 20%;` literal — argparse help formatter 把 `%` 当 format char，遇到 `;` (0x3b) 报 ValueError "unsupported format character"; `--help` 自 iter #150 commit 起不可用; iter #161 add_argument 后才暴露 (run flag 通过, 但 `--help` 失败)。

**证据支撑**:
- `python3 paper_claims_audit.py --help` 抛 ValueError at index 111
- `> 20%;` 第 110 字符是 `%`,第 111 字符是 `;` → 完美匹配 error message
- 修复: `> 20%;` → `above 20 pct;`(iter #161 顺手修复,避免 review 阻塞)

### 问题 3: [iter #160 dashboard panel color discipline] — 11 panel colors 必须 distinct
**严重程度**: Suggestion
**具体批评**:
> iter #132/153/154/156/159/160 dashboard panel colors 都是 warm tones (#ffecb3/#ffe0b2/#e1f5fe/#e8f5e9/#fff8e1/#fce4ec); 第 11 个 panel (iter #161) 选 cold tone `#e0f7fa` light cyan 视觉区分 + numeric 语义 calm 不会误读。

**证据支撑**:
- grep 现有 10 panel bg color → 无 `#e0f7fa`
- 数字统计 panel 选冷色 → 与暖色 alert panel (severity-tier/discrepant) 视觉对立

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1 | `paper_claims_audit.py:1650-1785` | 缺 per-status axis numeric aggregate |
| 问题2 | `paper_claims_audit.py:780` | iter #150 severity-tier help text `%` format bug |
| 问题3 | `_generate_audit_dashboard.py` | 缺 11th panel + 11th distinct bg color |

## §C 本轮代码优化

1. **paper_claims_audit.py — `--stats-by-status` CLI flag**:
   - docstring section (line 699-708) + argparse add_argument (line 810-815)
   - handler block (line 1801-1866): bucket claims by status (7-status taxonomy frozen order),aggregate max/mean abs + max/mean rel + claim count + vc count,print 7-row table
   - defensive fallback: unknown status → `(unknown)` bucket sink to bottom
   - em-dash for statuses with no abs (zero-count buckets preserved in output)

2. **paper_claims_audit.py:780 — fix pre-existing iter #150 help text `%` bug**:
   - `rel_delta_pct > 20%;` → `rel_delta_pct above 20 pct;`
   - `--help` 现在能跑(自 iter #150 修复前 broken)

3. **_generate_audit_dashboard.py — `_render_stats_by_status_panel()`**:
   - 新函数 mirror iter #161 CLI exactly (same `_safe_abs`/`_safe_rel` accessors + same per-status bucket aggregation + same max/mean abs + max/mean rel format + same em-dash for no-abs)
   - 11th distinct panel bg color `#e0f7fa` light cyan
   - monospace font + `font-variant-numeric: tabular-nums` for numeric column alignment
   - `<code>` tags wrap status names → typography uniform with claim ID cells
   - wire 到 `_render_html` body after `stats_by_source_dir_html`
   - generator footer updated to `iter #122, #125, #129, #132, #153, #154, #156, #159, #160, #161`

4. **_smoke_audit_regression.py — Case AS + Case AT**:
   - Case AS (CLI): 验证 `--stats-by-status` 7-row table 含 7 statuses (frozen order) + discrepant max abs=0.7415 + max rel=78.38% + em-dash for no-abs + exits 2 on missing JSON
   - Case AT (dashboard): 验证 panel class `stats-by-status-panel` + 11th distinct color `#e0f7fa` + 7 status rows + `data-status=` attribute
   - main count + docstring sync 到 46 cases

## §D 验证

- `python3 -m py_compile paper_claims_audit.py _generate_audit_dashboard.py _smoke_audit_regression.py` 全部通过
- 业务脚本 smoke test:
  - `python3 paper_claims_audit.py --stats-by-status` → 7-row table 输出正确,discrepant row = 4 claims, 6 vcs, max abs=0.7415, mean abs=0.2158, max rel=78.38%, mean rel=23.88%
  - `python3 paper_claims_audit.py --help` 现在能跑(顺手修复 iter #150 bug)
  - `python3 _smoke_audit_regression.py` → All 46 cases passed
  - `python3 _generate_audit_dashboard.py` → 14316+ bytes HTML, 含 stats-by-status-panel

## §E Git Commit

- `git commit -m "iter #161: Audit CLI --stats-by-status + dashboard mirror panel (completes stats triangle)"` (待 commit)

## §F 剩余审稿意见（待后续迭代）

- iter #162 candidate: dashboard sortable status rows (like iter #133 sort by abs_delta + filter by status)
- iter #162 candidate: per-status × per-section 2D matrix (`--stats-by-status-and-section`) for cross-axis breakdown
- iter #163 candidate: stats-by-status export to CSV (mirror iter #141 CSV export)