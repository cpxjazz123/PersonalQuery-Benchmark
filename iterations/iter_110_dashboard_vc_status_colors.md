# Iteration #110 — Dashboard value_check status color coding (reviewer audit pass)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (add `VC_STATUS_COLORS` + apply to vc panels)
**prior**: iter #108 dashboard + iter #109 rel_delta fix; iter #110 reviewer pass for remaining display issues

## §A 审稿意见

iter #108 dashboard 加成 + iter #109 修了 off-by-100 rel_delta bug,但 reviewer
人工 spot check dashboard HTML 时发现另一个问题: 17 claim detail row 里每个
value_check 的 status 文本 (`value_match` / `value_mismatch` / `degenerate`)
全部都是 **bold 黑色**,无视觉区分。 Reviewer 扫 RQ3_LLM_Full_Set_94% 的 3 个
subclaim detail 时,要靠读文字判断哪个 mismatch,慢且易 miss。

需求: vc panel 的 left border + 背景色应该匹配 value_check status 严重程度:
- `value_match` → green (健康)
- `value_mismatch` → orange (需要 attention)
- `degenerate` → red (阻塞)

这样 reviewer 一眼能定位 mismatch subclaim 在哪。

## §B 本轮 (iter #110) 改动

### PersoanlQuery/_generate_audit_dashboard.py

新增 `VC_STATUS_COLORS` map + apply 到 vc panel inline style:

```python
VC_STATUS_COLORS = {
    "value_match": ("#2e7d32", "#e8f5e9"),
    "value_mismatch": ("#f57c00", "#fff3e0"),
    "degenerate": ("#c62828", "#ffebee"),
}
```

`_render_claim_row` 用 `vc_status` 查颜色 + 应用到 `<div class='vc'>` border-left
+ background + `<strong>` text color。

## §C 关键改动点

1. **Severity-coded color**: 3 个 value_check status 各一颜色,reviewer 一眼区分
   match / mismatch / degenerate。
2. **Border + background + text**: 三处都用同一颜色 hue,确保即使色盲 reviewer
   也能靠 contrast 区分。
3. **Status text 染色**: vc panel 内 `<strong>value_mismatch</strong>` 现在是
   orange,不是黑色 — 与外层 claim row 的 status badge 颜色呼应。

## §D 验证

dashboard HTML 验证 (3 个 distinct style):
```
class='vc' style='border-left-color:#2e7d32;background:#e8f5e9'  # value_match (green)
class='vc' style='border-left-color:#c62828;background:#ffebee'  # degenerate (red)
class='vc' style='border-left-color:#f57c00;background:#fff3e0'  # value_mismatch (orange)
```

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 与 iter #108/#109 的关系

iter #108: dashboard 加成 (latent off-by-100 bug)
iter #109: rel_delta percent fix
iter #110: vc status color coding (reviewer-audit pass for remaining display issues)

后续 dashboard 可以做的改进 (deferred):
- 把 audit script 的 `expected_outputs` + `code_evidence` 加到 JSON → dashboard
  可以 show "audit checked these N files: ..." 给 reviewer 完整 provenance
- 对 `reason[:200]` 改 full text (现在 0/17 truncated 所以不动)

## §F 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+8 lines: `VC_STATUS_COLORS` map + style apply)
- 验证: dashboard HTML 3 distinct vc styles ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §G backlog (未变)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
- **iter #106** — README.md sync to reflect audit infrastructure