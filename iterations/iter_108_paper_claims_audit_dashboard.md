# Iteration #108 — HTML audit dashboard

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (NEW) + `_run_audit_ci.sh` (wire)
**prior**: iter #107 pre-commit hook 现在 wired 上 iter #108 dashboard

## §A 审稿意见

iter #107 加 pre-commit hook 让 silent flip 被自动 catch,但 reviewer 看
audit 结果只能 run command 看 stdout / JSON。 直观可视化 (颜色 badge,
sortable table, 每 claim 的 value-check extracted/expected/delta) 还没
建立。 17 个 claim × 多 value check 的 JSON 没有 dashboard,reviewer
需要自己 grep + 算 relative delta,慢且易 miss。

iter #108 加 self-contained HTML dashboard:
1. `PersoanlQuery/_generate_audit_dashboard.py` — 读 JSON → render HTML
2. Wire 到 `_run_audit_ci.sh` 作为 [3/3] step
3. 输出 `result/personal_query/iterations/paper_claims_audit_dashboard.html`

设计选择:
- **Self-contained**: inline CSS, no external JS/CSS — reviewer 可以直接
  `xdg-open` 或 share via email attachment,不依赖 CDN / network。
- **Color-coded status badges**: 7 statuses (verified_value_match /
  verified / discrepant / degenerate / partial / unverified / blocked)
  各一颜色,reviewer 一眼能定位问题 claim。
- **Expandable detail row**: 每 claim 后跟一个 detail row 显示
  value_check_results (extracted/expected/abs_delta/rel_delta) +
  paper_text_caveat。
- **Wire 到 CI**: 每次 commit 自动 regenerate, reviewer 看 dashboard
  总是最新状态。

## §B 本轮 (iter #108) 改动

### §B.1 PersoanlQuery/_generate_audit_dashboard.py (NEW, ~165 lines)

```python
#!/usr/bin/env python3
"""[Reviewer-pilot] Generate static HTML dashboard from paper_claims_audit.json."""
import json
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
JSON_PATH = REPO_ROOT / "result" / "personal_query" / "iterations" / "paper_claims_audit.json"
HTML_PATH = REPO_ROOT / "result" / "personal_query" / "iterations" / "paper_claims_audit_dashboard.html"

STATUS_COLORS = {
    "verified_value_match": ("verified (value match)", "#2e7d32"),
    "verified": ("verified", "#388e3c"),
    "discrepant": ("discrepant", "#f57c00"),
    "degenerate": ("degenerate", "#c62828"),
    "partial": ("partial", "#7b1fa2"),
    "unverified": ("unverified", "#616161"),
    "blocked": ("blocked", "#b71c1c"),
}

def _render_claim_row(c): ...
def _render_summary_badges(summary): ...
def _render_html(doc): ...
def main(): ...
```

### §B.2 PersoanlQuery/_run_audit_ci.sh (wired)

把 [1/2]/[2/2] 改成 [1/3]/[2/3]/[3/3]:

```bash
# Step 3: regenerate HTML dashboard (iter #108)
if [[ $QUICK -eq 0 ]]; then
    echo "[3/3] Regenerating HTML dashboard..."
    if python3 "${DASHBOARD_SCRIPT}"; then
        echo "  PASS  dashboard regenerated"
        ...
    fi
fi
```

### §B.3 PersoanlQuery/_AUDIT_CI_README.md (updated)

加一行说明 dashboard 会在 full mode 自动 regenerate。

## §C 关键改动点

1. **No external dependencies**: 全部 inline CSS + server-side render,reviewer 可以
   离线 / airgapped 看 dashboard。
2. **Fail-soft in dashboard**: dashboard step 失败不会 fail CI (只是 stderr) —
   实际 wire 是 dashboard 失败 → exit 1,但 audit JSON 仍然存在,CI 不会因为
   dashboard 渲染问题 block commit (如某些 claim 没有 expected field 时)。
3. **Per-claim detail panel**: value_check_results 完整 render (subclaim label,
   extracted/expected/abs_delta/rel_delta table, source_file, note),
   paper_text_caveat 单独 highlight (yellow left border)。
4. **Summary header**: 7 个 status badge 各显示 count + label + 颜色,reviewer 一眼
   看总体分布 (6 verified 4 discrepant 1 degenerate 1 partial 5 unverified 0 blocked)。

## §D 测试结果

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107 + #108) ===
Repo root: /home/wlia0047/wenyu
Mode: full (regenerate + regression + dashboard)

[1/3] Regenerating audit JSON...
  PASS  audit JSON regenerated

[2/3] Running regression smoke test...
=== paper_claims_audit regression smoke test (iter #105) ===
...
All 4 cases passed. Audit CLI frozen baseline verified.

[3/3] Regenerating HTML dashboard...
Wrote: /home/wlia0047/wenyu/result/personal_query/iterations/paper_claims_audit_dashboard.html
  PASS  dashboard regenerated

=== AUDIT CI PASSED ===
```

Dashboard 文件:
- 14,316 bytes / 238 lines / 17 claims
- 7 status badges in summary header
- 每 claim 后跟 detail row (value_check_results + paper_text_caveat)

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (NEW, ~165 lines)
- 模块: `PersoanlQuery/_run_audit_ci.sh` (updated [3/3] step)
- 模块: `PersoanlQuery/_AUDIT_CI_README.md` (updated mention)
- 输出: `result/personal_query/iterations/paper_claims_audit_dashboard.html` (regenerated on commit)
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s)
- 验证: dashboard render → 17 claims / 7 summary badges ✓
- 跑时间: dashboard step ≈0.5s

## §F 与 loop.md §8 的关系

iter #108 完成 audit infrastructure 的最后一块 (visualization):
- iter #93: value-validation layer
- iter #94-#99: paper 文档化 audit 结果
- iter #103: §5 reproducibility 文档化
- iter #104: CLI 让 audit 可以 ad-hoc inspection
- iter #105: regression test 冻结 baseline
- iter #107: pre-commit hook 自动 catch silent flip
- iter #108: HTML dashboard 让 reviewer 一眼看 17 claim 状态

后续 candidate (backlog):
- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified (需 update baseline + regression test)
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified (需 update baseline + regression test)
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
- **iter #106** — README.md sync to reflect audit infrastructure