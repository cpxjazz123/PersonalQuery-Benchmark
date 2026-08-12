# Iteration #121 — Regression test coverage extension (--diff + dashboard anchors)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_smoke_audit_regression.py` (add 3 cases E/F/G)
**prior**: iter #105 froze 4 cases; iter #114 added --diff mode but no regression coverage; iter #116/#118 added dashboard features but no regression coverage

## §A 审稿意见

iter #105 regression test 4 cases 冻结 audit baseline,但 iter #114 (--diff mode)
+ iter #116 (anchors) + iter #118 (status legend) 加了 3 个新 affordance 没
regression coverage:

- 某 developer 不小心改 `_diff_against_baseline()` 让 self-diff exit 1 → silent flip
  detector broken,baseline 失效。
- 某 developer 改 dashboard `_render_claim_row` 去掉 `id='claim-{id}'` → anchors
  失效,paper → dashboard cross-link broken。
- 某 developer 改 dashboard 去掉 status legend block → reviewer 没 inline help,
  需查 source。

iter #121 fix: 加 3 cases 冻结 iter #114/#116/#118 behavior:
- **Case E**: `--diff <current JSON>` → exit 0 + "NO FLIP" message (self-diff)
- **Case F**: `--diff <fake-flip-baseline>` → exit 1 + "SILENT FLIP DETECTED" + status_flip kind
- **Case G**: dashboard HTML has 17 anchors (regex allows `.` and `%` in IDs) + status-legend block

## §B 本轮 (iter #121) 改动

### PersoanlQuery/_smoke_audit_regression.py

```python
def case_e_diff_self():
    """Case E: --diff self returns exit 0 + 'NO FLIP' message."""
    current = REPO_ROOT / "result/personal_query/iterations/paper_claims_audit.json"
    exit_code, stdout, stderr = _run_audit(["--diff", str(current), "--json-only"])
    assert exit_code == 0
    assert "NO FLIP" in stdout
    assert "17 claims match baseline" in stdout

def case_f_diff_flip_detection():
    """Case F: --diff with fake-flip baseline detects status flip, exits 1."""
    cur = json.load(open(current_json))
    cur["claims"][0]["status"] = "verified"  # mutate first claim
    with tempfile.NamedTemporaryFile(...) as f:
        json.dump(cur, f)
        fake_path = f.name
    exit_code, stdout, stderr = _run_audit(["--diff", fake_path, "--json-only"])
    assert exit_code == 1
    assert "SILENT FLIP DETECTED" in stdout
    assert "status_flip" in stdout

def case_g_dashboard_anchors_and_legend():
    """Case G: dashboard HTML has per-claim anchors + status legend."""
    import re
    anchors = re.findall(r"id='claim-([A-Za-z0-9_.%]+)'", html)
    assert len(anchors) == 17  # 17 claim IDs (some have . and % like RQ3_Fleiss_Kappa_0.72, RQ3_LLM_Full_Set_94%)
    assert "status-legend" in html
    assert "verified (value match)" in html
    assert "discrepant" in html and "degenerate" in html
```

`main()` call 3 new cases, print "All 7 cases passed".

## §C 关键改动点

1. **Anchor regex covers `.` and `%`**: claim IDs like `RQ3_Fleiss_Kappa_0.72` and
   `RQ3_LLM_Full_Set_94%` contain these chars; old regex `[A-Za-z0-9_]+` missed 4
   anchors (only 13 captured). Fix: `[A-Za-z0-9_.%]+` → 17 captured.
2. **Case F mutates first claim**: doesn't assume which claim is first; just mutates
   `cur["claims"][0]["status"]` to "verified" (different from current degenerate).
   Relies on first claim being one whose status is mutable.
3. **Case G tests both anchors AND legend**: catches if either feature is broken
   separately (e.g. someone refactors claim row ID generation but forgets legend).

## §D 测试

```bash
$ python3 PersoanlQuery/_smoke_audit_regression.py
=== paper_claims_audit regression smoke test (iter #105, #121) ===
Audit script: /home/wlia0047/ar57/wenyu/PersoanlQuery/paper_claims_audit.py
Frozen baseline: 2026-07-21 (audit summary 0/6/4/1/1/5/0)

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'
  PASS
Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965
  PASS
Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message
  PASS
Case D: --strict --json-only (full audit) expects exit 1
  PASS
Case E: --diff <current JSON> expects exit 0 + 'NO FLIP' (iter #121)
  PASS  exit_code=0, 'NO FLIP: 17 claims match baseline'
Case F: --diff fake-flip-baseline expects exit 1 + flip table (iter #121)
  PASS  exit_code=1, flip table contains 'status_flip'
Case G: dashboard HTML has anchors + status legend (iter #121)
  PASS  17 anchors present + status legend block present

All 7 cases passed. Audit CLI frozen baseline verified.
```

Full audit CI 测试 PASS (regenerate + 7 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/_smoke_audit_regression.py` (+3 cases E/F/G, +docstring)
- 验证: 7/7 cases pass; pre-commit hook 现在 freeze 7 invariants (4 + 3) ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 regression test evolution 的关系

regression test coverage timeline:
- **iter #105** — 4 cases (CLI + claim statuses)
- **iter #121** — 7 cases (+ --diff self / --diff flip / dashboard anchors + legend) ✓ 本轮

每次新 affordance 都应该 regression test 冻结,否则 silent flip 不能被 catch。

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample