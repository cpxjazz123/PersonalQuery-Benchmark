# Iteration #99 — Paper Abstract + §1 Introduction: audit mention

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:7` (abstract) + `:39` (§1 → §2 transition)
**prior**: iter #93-#98 added Table 1/2/3 footnotes + §5 Limitations + bidirectional cross-references; reviewer navigation now complete

## §A 审稿意见

iter #93-#98 让 reviewer 读完 paper 任何地方都能 navigate 到 release-vs-paper
disclosure (Table footnotes ↔ §5 Limitations 双向链接)。 但 reviewer 一进 paper
(读 abstract + §1 introduction) 看不到任何 audit/release-vs-paper 提示, 直到
读 §3 Table 1 才突然发现 "actually release pipeline gives different numbers"。

Reviewer 视角问题:
1. Abstract 说 "Experimental results show..." 没 disclose 这 results 是 originally-computed
   还是 release-pipeline
2. §1 introduction 列 4 个 contributions, 但没 disclose 5/17 paper claims audit unverified
3. §5 Limitations 是 reviewer 通常最后读, 不一定回看到

iter #99 在 abstract + §1 增加 brief audit mention, 让 reviewer 一进 paper 就
知道:
- Paper claims 已 reviewer-audit (17 claims, full coverage)
- Audit status: 6 verified / 4 discrepant / 1 degenerate / 5 unverified / 1 partial
- Inline Table footnotes + §5 Limitations 详细说明具体 gaps

## §B 本轮 (iter #99) 改动

### §B.1 Abstract 末尾添加 audit mention (line 7)

在 "Code and dataset are available at: https://..." 前加 1-2 句:

```
For reproducibility, every quantitative claim in this paper has been
audited against the open-source release pipeline (see Table 1/2/3 footnotes
and §5 Limitations for release-vs-paper discrepancies); the full
per-claim audit with extracted values and relative deltas is available at
`result/personal_query/iterations/paper_claims_audit.json`.
```

### §B.2 §1 → §2 transition 添加 audit mention (between line 39 and line 40)

在 4 contributions list 之后, §2 开始之前加 1 段:

```
**Reproducibility audit.** All 17 quantitative claims in this paper (RQ1-4
in §3 + 6 §2.2 infrastructure claims) have been audited against the
open-source release pipeline with both file-existence and value-validation
checks (iter #93). Audit summary: 6 claims verified with extracted values
matching paper numbers, 4 discrepant (Table 2 human-eval / LLM full-set
metrics with relative deltas 13.4–78.4%), 1 degenerate (Table 1 bootstrap CI
degenerate due to Stage 6 query-pool size), 5 unverified (Table 3 +
Stage 12 outputs missing), and 1 partial (env-var feature with no output
artifact). Inline Table 1/2/3 footnotes detail each discrepancy; §5
Limitations lists re-execution costs. See
`result/personal_query/iterations/paper_claims_audit.json` for full audit.
```

## §C 关键改动点

1. **Abstract audit pointer**: 1-2 句 abstract 末尾说 "every claim audited",
   reviewer 读完 abstract 知道 paper 有完整 audit infrastructure
2. **§1 audit summary block**: 在 contributions list 之后加完整 audit status
   summary (6/4/1/5/1), reviewer 知道 paper 哪些部分 reproduce 哪些部分有 gap
3. **JSON reference**: abstract + §1 都指向 `paper_claims_audit.json`, reviewer
   可自主 verify
4. **Cross-link §5 + Table footnotes**: 让 reviewer 从 §1 也能 navigate 到其他
   disclosure sections

## §D 完整 navigation 现在覆盖

| Paper section | Audit disclosure | iter |
|---------------|------------------|------|
| Abstract | Brief mention + JSON ref | iter #99 (本轮) |
| §1 Introduction | Full audit summary block | iter #99 (本轮) |
| §3 Table 1 | Inline footnote (degenerate + 2-query cell caveat) | iter #95 |
| §3 Table 2 | Inline footnote (4 discrepant metrics) | iter #94 |
| §3 Table 3 | Inline footnote (5 unverified claims) | iter #96 |
| §3.2 §3.3 §3.4 | (parent sections reference footnotes) | (implicit) |
| §5 Limitations | 3 backward cross-refs to inline footnotes | iter #94 + #98 |

reviewer 现在 paper top-down navigation 完整: Abstract → §1 → §3 Tables → §5 Limitations, 任何
section 都能 navigate 到 audit disclosure。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:7` (abstract 末尾) + `:39` (§1 → §2 之间)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "Reproducibility audit\|audit summary" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #99 完成 paper top-down navigation 完整:
- Abstract + §1 introduction 现在 disclose audit state
- Reviewer 进 paper 立即知道 release-vs-paper gap 全貌 (6/4/1/5/1 status)
- 后续 Table footnotes + §5 Limitations 提供具体 detail

后续 candidate:
- **iter #100** — 反思: paper 全 audit infrastructure 已完整 (abstract + §1 + Table 1/2/3 footnote + §5 Limitations 双向 cross-ref), iter 进度回顾 + 下一阶段 pivot
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 重跑 (1.5-3 h) — 修复 Table 1 bootstrap CI degenerate