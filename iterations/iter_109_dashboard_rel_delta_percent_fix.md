# Iteration #109 — Dashboard rel_delta percentage display fix

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_generate_audit_dashboard.py` (1-line fix in `_render_claim_row`)
**prior**: iter #108 dashboard rendering; this iter fixes an off-by-100 display bug

## §A 审稿意见

iter #108 dashboard 加成后,reviewer 检查 HTML 输出发现
RQ3_Spearman detail row 显示 `rel_delta=0.12%`,但 audit 脚本的 reason text
说 `(rel 11.6%)`。 在阅读 paper Table 2 footnote 时,reviewer 会期待 detail
row 与 reason text 一致 (因为两个数字应该描述同一个 value-check)。

根因: `paper_claims_audit.py` 把 `rel_delta` 存为 fraction (0.1156) 而
非 percent (11.56%) — audit 脚本自己 compute reason text 时手动 `* 100`
并加 `%`,但 JSON field 是 raw fraction。 dashboard `_render_claim_row`
原本直接拿 fraction 加 `%` 后缀 → 显示 `0.12%`(显示成小数点后的 percent,
根本不是相对变化)。

数学验证 (RQ3_Spearman):
- extracted=0.7164, expected=0.81, abs_delta=0.0936
- rel_delta (fraction) = 0.0936 / 0.81 = 0.1156
- rel_delta (percent) = 11.56% ← 这才是相对变化

修复: dashboard 在 render `rel_delta` 前 `* 100`,然后再加 `%` 后缀。

## §B 本轮 (iter #109) 改动

### PersoanlQuery/_generate_audit_dashboard.py (1-line fix)

```diff
- if vc.get('rel_delta') is not None else '?'
+ rd = vc.get('rel_delta')
+ rd_pct = f"{rd * 100:.2f}%" if rd is not None else "?"
...
- <td>{_fmt_float(vc.get('rel_delta'), 2) if vc.get('rel_delta') is not None else '?'}%</td>
+ <td>{rd_pct}</td>
```

## §C 关键改动点

1. **Display 层 fix,不改 storage**: paper_claims_audit.py 存 fraction 没问题
   (machine-readable, percentage-agnostic); dashboard 在 render 层做单位转换
   是正确分层。
2. **一致性**: 现在 detail row 的 rel_delta 与 reason text 的 `(rel X%)` 完全一致。
3. **Triple-verify**: Spearman 11.56% / MAE 30.34% / preservation 78.38% —
   全部与 audit script 的 reason text 对齐。

## §D 测试结果

修复前 (iter #108):
```
RQ3_Spearman_0.81  → rel_delta: 0.12%  ❌ 与 reason "11.6%" 不一致
RQ3_MAE_0.89       → rel_delta: 0.30%  ❌ 与 reason "30.3%" 不一致
RQ3_LLM_Full_Set preservation → rel_delta: 0.78%  ❌ 与 reason "78.4%" 不一致
```

修复后 (iter #109):
```
RQ3_Spearman_0.81  → rel_delta: 11.56%  ✓
RQ3_MAE_0.89       → rel_delta: 30.34%  ✓
RQ3_LLM_Full_Set preservation → rel_delta: 78.38%  ✓
```

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (1-line fix in `_render_claim_row`)
- 验证: dashboard 显示 11.56% / 30.34% / 78.38% ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 loop.md §8 的关系

iter #109 完成 dashboard correctness fix:
- iter #108 加 dashboard (off-by-100 display bug 潜伏)
- iter #109 fix display (reviewer 第一次 review dashboard 时应被 catch,
  但 user 在 iter #108 commit 前没人工 spot check HTML)

这种 bug 在 production 会被 reviewer 第一眼就抓到 (rel_delta 显示 0.12%
明显 wrong),所以 fix 必须立刻,不能 deferred。

后续 candidate (未变):
- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
- **iter #106** — README.md sync to reflect audit infrastructure