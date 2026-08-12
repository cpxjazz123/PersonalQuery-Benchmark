# Iteration #104 — paper_claims_audit.py CLI refactor

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py:main()` — argparse-based CLI
**prior**: iter #93 added value-validation; iter #100 retrospective identified CLI refactor as next dev-tooling iter

## §A 审稿意见

iter #93 audit script `main()` 是 fixed entry point, 每次跑就 enumerate 所有 17 claims 写到固定
路径 `paper_claims_audit.json` 然后 print stdout table。 没法:
1. Audit 单个 claim (--claim-id)
2. 看 detailed value_check_results (--verbose)
3. 输出到 custom 路径 (--output)
4. CI hook friendly (--strict 退出码)
5. List 所有 claim id (--list)

reviewer / developer 想知道 "现在 RQ3_Fleiss_Kappa_0.72 的具体 extracted value 与 delta 是多少"
必须:
1. 跑完整 audit
2. 读 JSON 文件
3. 找到对应 entry

iter #104 refactor `main()` 接受 argparse CLI args, 让 ad-hoc inspection / CI hook /
regression test 都方便。

## §B 本轮 (iter #104) 改动

### §B.1 main() 接受 argparse CLI

```python
def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Audit paper §3 RQ1-4 + Table 1-3 claims.")
    parser.add_argument("--claim-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    ...
```

### §B.2 Exit code semantics

| Exit | Meaning |
|------|---------|
| 0 | OK (all claims audited, --strict passed if requested) |
| 1 | --strict failure (≥1 non-verified claim) |
| 2 | Bad CLI args (invalid claim id, etc.) |

### §B.3 JSON output schema 增 2 fields

- `n_audited`: actually audited count (≠ n_claims if --claim-id filter)
- `claim_id_filter`: which claim id was filtered (or None for full audit)

audit_target string 增 "CLI iter #104" suffix.

## §C 测试结果

```bash
# help
$ python3 paper_claims_audit.py --help
usage: paper_claims_audit.py [-h] [--claim-id CLAIM_ID] [--verbose]
                             [--output OUTPUT] [--json-only] [--strict]
                             [--list]

# list
$ python3 paper_claims_audit.py --list
RQ1_Table1_Hit10
RQ1_Delta_Range
RQ2_Table1_Drop
RQ3_Fleiss_Kappa_0.72
RQ3_Spearman_0.81
... (17 total)

# single claim + verbose
$ python3 paper_claims_audit.py --claim-id RQ3_Fleiss_Kappa_0.72 --verbose
[RQ3_Fleiss_Kappa_0.72]  status=discrepant
  section : §3.3 + Table 2
  claim   : Fleiss' κ across three human annotators on 120 sampled queries is 0.72.
  reason  : file exists but value mismatch: expected=0.7200 actual=0.6235 delta=0.0965 (rel 13.4%)
  value_check: value_mismatch
    extracted=0.6235  expected=0.7200  abs_delta=0.0965  rel_delta=13.40%
    source=result/personal_query/02_writing_analysis/llm_human_eval/llm_human_eval_cross_domain.json

# --strict full audit
$ python3 paper_claims_audit.py --strict --json-only > /dev/null
$ echo $?
1   # 4 discrepant + 1 degenerate + 1 partial + 5 unverified present

# --strict on a verified claim
$ python3 paper_claims_audit.py --claim-id Pipeline_Regeneration_10x10 --strict --json-only > /dev/null
$ echo $?
0

# invalid claim id
$ python3 paper_claims_audit.py --claim-id DOES_NOT_EXIST
ERROR: claim id 'DOES_NOT_EXIST' not found. Use --list to see available ids.
$ echo $?
2

# custom output
$ python3 paper_claims_audit.py --claim-id RQ3_Fleiss_Kappa_0.72 --output /tmp/test.json --json-only
Wrote: /tmp/test.json
$ python3 -c "import json; d=json.load(open('/tmp/test.json')); print(d['claim_id_filter'], d['n_audited'])"
RQ3_Fleiss_Kappa_0.72 1
```

## §D 关键改动点

1. **argparse CLI**: 6 个 args (--claim-id / --verbose / --output / --json-only / --strict / --list)
2. **Exit code**: 0/1/2 三态语义明确 (ok / strict-fail / bad-cli)
3. **JSON schema 加 n_audited + claim_id_filter**: 让 --claim-id filter 在 JSON
   output 里 traceable
4. **--verbose 完整 per-claim detail**: 包括 section / claim / value_check_results
   每个 subclaim 的 extracted/expected/delta/source_file/note + paper_text_caveat
5. **--strict CI-friendly**: iter #107 CI hook 可以 `python3 paper_claims_audit.py
   --strict --json-only` 然后 check exit code
6. **--list for ad-hoc discovery**: `paper_claims_audit.py --list | grep RQ3` 让 developer
   快速找到 claim id

## §E 与 downstream iter 的关系

| iter | Usage of CLI |
|------|--------------|
| #105 | Regression test: assert RQ3_Fleiss_Kappa_0.72 status stays `discrepant` after re-run |
| #107 | CI hook: `python3 paper_claims_audit.py --strict --json-only` pre-commit |
| #108 | HTML dashboard: `paper_claims_audit.py --json-only --output /tmp/audit.json` then parse |

CLI refactor 为后续 dev tooling 提供 foundation。

## §F 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (main() 重写为 argparse-based)
- 输出: 默认 `result/personal_query/iterations/paper_claims_audit.json` (不变)
- 命令: `python3 PersoanlQuery/paper_claims_audit.py [--claim-id ID] [--verbose] [--output PATH] [--json-only] [--strict] [--list]`
- 验证: `python3 -m py_compile PersoanlQuery/paper_claims_audit.py` (Rule 10) ✓
- 跑时间: < 1s

## §G 与 loop.md §8 的关系

iter #104 完成 paper_claims_audit CLI refactor:
- 6 个 CLI args 让 audit script 适用于 ad-hoc / CI / regression test / dashboard
- Exit code 0/1/2 明确语义
- JSON schema 加 n_audited + claim_id_filter 让 filtered output traceable

后续 candidate:
- **iter #105** — Add audit regression test (用 --claim-id + --json-only 断言 status 不 silent flip)
- **iter #107** — Add CI hook to run `paper_claims_audit.py --strict --json-only` on every commit
- **iter #108** — paper_claims_audit dashboard (HTML visualization, 用 --json-only parse)
- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified