# Iteration #107 — Pre-commit audit CI hook

**日期**: 2026-07-21
**scope**: `PersoanlQuery/_run_audit_ci.sh` + `install_audit_precommit.sh` + `_AUDIT_CI_README.md` (NEW)
**prior**: iter #104 CLI + iter #105 regression test; this iter wires them to git pre-commit hook

## §A 审稿意见

iter #105 加 regression smoke test 但需要手动跑 (`python3 _smoke_audit_regression.py`)。
reviewer / developer 不一定记得跑, audit baseline silent flip 不会被抓到直到
reviewer / CI 发现。

iter #107 把 audit CI 自动化:
1. `PersoanlQuery/_run_audit_ci.sh` — wrapper bash script (regenerate + regression)
2. `PersoanlQuery/install_audit_precommit.sh` — 安装 helper (symlink hook)
3. `PersoanlQuery/_AUDIT_CI_README.md` — 用法 + uninstall + bypass 文档

设计选择: simple shell + symlink, 不用 pre-commit framework (避免 pip install overhead,
任何 git repo 都能用)。

## §B 本轮 (iter #107) 改动

### §B.1 PersoanlQuery/_run_audit_ci.sh (NEW)

```bash
#!/usr/bin/env bash
set -euo pipefail
# [1/2] Regenerate audit JSON
# [2/2] Run regression test
# Exit 0 = pass, 1 = fail
# Mode: --quick skips regenerate
```

### §B.2 PersoanlQuery/install_audit_precommit.sh (NEW)

```bash
#!/usr/bin/env bash
# One-time setup: bash PersoanlQuery/install_audit_precommit.sh
# Symlinks .git/hooks/pre-commit -> PersoanlQuery/_run_audit_ci.sh
# Backs up existing pre-commit to .git/hooks/pre-commit.bak
```

### §B.3 PersoanlQuery/_AUDIT_CI_README.md (NEW)

文档:
- What it does
- What it catches (6 silent flip types)
- One-time setup
- Manual run
- Bypass once: `git commit --no-verify`
- Uninstall: `rm .git/hooks/pre-commit`
- When baseline legitimately changes (iter #87 / #88)
- Exit codes (0/1)

## §C 关键改动点

1. **No framework dependency**: 直接 bash + symlink, 不需要 pip install pre-commit
2. **Symlink, not copy**: `_run_audit_ci.sh` 修改自动 pick up, 不用重 install
3. **Backs up existing pre-commit**: 如果 repo 已有 pre-commit hook (从其他 source),
   .bak 保留避免覆盖
4. **Two modes**: full (regenerate + regression) vs --quick (regression only),
   让 CI server 可以选择 (full 在 push 前, --quick 在 commit 前)
5. **Detailed failure message**: audit CI 失败时告诉 user 如何 diagnose + accept
   new baseline (更新 _smoke_audit_regression.py)

## §D 测试结果

```bash
$ bash PersoanlQuery/_run_audit_ci.sh
=== paper_claims_audit CI (iter #107) ===
Repo root: /home/wlia0047/wenyu
Mode: full (regenerate + regression)

[1/2] Regenerating audit JSON...
  PASS  audit JSON regenerated

[2/2] Running regression smoke test...
=== paper_claims_audit regression smoke test (iter #105) ===
Audit script: /home/wlia0047/wenyu/PersoanlQuery/paper_claims_audit.py
Frozen baseline: 2026-07-21 (audit summary 0/6/4/1/1/5/0)

Case A: --claim-id Pipeline_Regeneration_10x10 expects 'verified'
  PASS  status=verified, reason='1/1 output globs match (no numerical claim to validate)'

Case B: --claim-id RQ3_Fleiss_Kappa_0.72 expects 'discrepant' with abs_delta≈0.0965
  PASS  status=discrepant, extracted=0.6235, abs_delta=0.0965

Case C: --claim-id DOES_NOT_EXIST expects exit 2 + error message
  PASS  exit_code=2, stderr="ERROR: claim id 'DOES_NOT_EXIST' not found. Use --list to se"

Case D: --strict --json-only (full audit) expects exit 1
  PASS  exit_code=1, stderr='STRICT MODE: 11 non-verified claim(s) detected (exit 1)'

All 4 cases passed. Audit CLI frozen baseline verified.

=== AUDIT CI PASSED ===
```

## §E 安装 + 用法

### 安装 (one-time)

```bash
bash PersoanlQuery/install_audit_precommit.sh
```

Creates: `.git/hooks/pre-commit` -> `PersoanlQuery/_run_audit_ci.sh`

### 手动跑

```bash
bash PersoanlQuery/_run_audit_ci.sh           # full (~2s)
bash PersoanlQuery/_run_audit_ci.sh --quick   # regression only (~1s)
```

### Bypass (occasional)

```bash
git commit --no-verify
```

### Uninstall

```bash
rm .git/hooks/pre-commit
```

## §F 文件 & 命令

- 模块: `PersoanlQuery/_run_audit_ci.sh` (NEW, ~50 lines)
- 模块: `PersoanlQuery/install_audit_precommit.sh` (NEW, ~20 lines)
- 模块: `PersoanlQuery/_AUDIT_CI_README.md` (NEW, ~70 lines)
- 命令: `bash PersoanlQuery/install_audit_precommit.sh` (安装)
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (手动跑)
- 验证: `bash PersoanlQuery/_run_audit_ci.sh` → "AUDIT CI PASSED" ✓
- 跑时间: full ≈2s, --quick ≈1s

## §G 与 loop.md §8 的关系

iter #107 完成 audit CI 自动化:
- 每次 `git commit` 自动 run paper_claims_audit + regression test
- Silent flip 被自动 catch
- 不需要 pre-commit framework 依赖

后续 candidate:
- **iter #108** — paper_claims_audit HTML dashboard (visualization)
- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified (需 update baseline + regression test)
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified (需 update baseline + regression test)