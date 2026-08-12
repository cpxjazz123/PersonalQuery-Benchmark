# Iteration #114 — Audit CLI `--diff` mode (baseline vs current)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` (add `--diff BASELINE` flag)
**prior**: iter #105 regression test (auto) + iter #107 pre-commit CI (auto); this iter adds ad-hoc spot-check

## §A 审稿意见

iter #105 加 regression smoke test 自动 freeze baseline,iter #107 把 test
绑到 pre-commit hook 自动 catch silent flip。但 reviewer / developer 想
**ad-hoc** 检查 "我的本地 audit 是不是和上次 commit 的 baseline 一致?"
只能跑 `_smoke_audit_regression.py` 看 4 个 frozen cases,不能列出所有
17 个 claim 的 per-claim flip。

需求: `--diff BASELINE_PATH` mode:
1. 跑当前 audit (写 current JSON)
2. 对比 baseline JSON
3. 列出所有 per-claim flip (status change / vc status flip / vc extracted
   change / new claim / removed claim)
4. exit 0 if no flip, exit 1 if any flip

Use cases:
- Reviewer: "我刚跑 audit, 跟 paper 提交时的 baseline 一致吗?"
- Developer: "我改了一个 script, 看看 audit 哪些 claim 状态变了"
- CI server: 可以跑 `--diff main_baseline.json` 比 main vs PR

## §B 本轮 (iter #114) 改动

### §B.1 paper_claims_audit.py

**CLI arg** (新加):
```python
parser.add_argument("--diff", default=None, help="(iter #114) Path to baseline JSON to diff against; exit 1 if any per-claim flip detected.")
```

**Helper function** (新加, ~50 lines):
```python
def _diff_against_baseline(current_audit: list[dict], baseline_path: Path) -> int:
    """Compare current audit JSON's per-claim state against baseline JSON.
    Returns 0 = no flip, 1 = silent flip detected."""
```

Detects 5 flip kinds:
- `new_claim` — current claim not in baseline
- `removed_claim` — baseline claim missing from current
- `status_flip` — claim status changed (e.g. verified → degenerate)
- `new_vc` — new value_check subclaim
- `vc_status_flip` — value_check status changed (e.g. value_match → value_mismatch)
- `vc_extracted_change` — value_check extracted value changed (silent numeric drift)

**main() integration**:
```python
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_doc, f, indent=2, ensure_ascii=False)
# iter #114: --diff mode. Run after writing current JSON; compare against baseline.
if args.diff:
    diff_exit = _diff_against_baseline(audit, Path(args.diff))
    return diff_exit
```

## §C 关键改动点

1. **Self-contained diff**: 不需要外部工具 (git diff / jq),audit script 自己
   load 两个 JSON + 比对,生成 readable text report。
2. **Multiple flip kinds**: 不仅 status flip,还有 vc-level flip,细粒度 — reviewer
   知道 "baseline 0.6235 vs current 0.6241" 也是 silent numeric drift (虽然
   还在 tolerance 内,但 extracted value 变了)。
3. **Actionable error msg**: 检测到 flip 后告诉 user 如何 accept new baseline
   (与 iter #107 失败 message 一致)。
4. **Exit code semantic**: 0 = clean, 1 = silent flip。 与 `--strict` 一致
   (1 = failure),CI hook 可以同时用 `--strict --diff` 双重检查。

## §D 测试

```bash
# Test 1: self-diff (no flip expected)
$ python3 PersoanlQuery/paper_claims_audit.py --diff result/personal_query/iterations/paper_claims_audit.json --json-only --output /tmp/test.json
NO FLIP: 17 claims match baseline.
EXIT=0

# Test 2: fake baseline with status flip (expect flip detection)
$ python3 -c "
import json
b = json.load(open('result/personal_query/iterations/paper_claims_audit.json'))
b['claims'][0]['status'] = 'verified'  # fake flip
json.dump(b, open('/tmp/fake_baseline.json', 'w'))
"
$ python3 PersoanlQuery/paper_claims_audit.py --diff /tmp/fake_baseline.json --json-only --output /tmp/test.json
SILENT FLIP DETECTED: 1 change(s) vs baseline /tmp/fake_baseline.json:

  [status_flip           ] RQ1_Table1_Hit10             verified → degenerate

To accept new baseline as expected (e.g. after iter #87/#88 Stage re-run):
  1. Investigate: bash PersoanlQuery/_run_audit_ci.sh
  ...
EXIT=1
```

Full audit CI 测试 PASS (regression test 不依赖 --diff mode, 4 cases 仍然 pass)。

## §E 用法

```bash
# Ad-hoc: 检查本地 audit 是否与已 commit baseline 一致
python3 PersoanlQuery/paper_claims_audit.py --diff result/personal_query/iterations/paper_claims_audit.json

# CI mode: 每次 PR 检查 audit 是否与 main 一致
python3 PersoanlQuery/paper_claims_audit.py --diff $MAIN_BASELINE_JSON --json-only

# Strict + Diff 双重检查
python3 PersoanlQuery/paper_claims_audit.py --diff baseline.json --strict --json-only
```

## §F 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+argparse arg + ~50 line helper function + main() branch)
- 验证: self-diff exit 0 ✓; fake-flip-diff exit 1 + flip table ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §G 与 audit infra timeline 的关系

- **#104** — argparse CLI (foundation)
- **#105** — regression smoke test (auto-freeze baseline)
- **#107** — pre-commit CI hook (auto-run regression)
- **#114** — `--diff` mode (ad-hoc spot-check) ✓ 本轮

现在 reviewer / developer 有 3 档选择:
1. **Quick spot-check**: `python3 paper_claims_audit.py --diff baseline.json`
2. **Auto on commit**: `bash install_audit_precommit.sh` → pre-commit hook 跑 regression
3. **Ad-hoc inspect**: `python3 paper_claims_audit.py --claim-id X --verbose`

## §H backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified (will use --diff to detect flips)
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified (will use --diff)
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample