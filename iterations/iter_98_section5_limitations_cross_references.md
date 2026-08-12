# Iteration #98 — §5 Limitations backward cross-references to Table footnotes

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:197` (§5 Limitations #3, #4)
**prior**: iter #97 added forward links from Table footnotes to each other; §5 Limitations currently has forward link only to Table 2 footnote (added in iter #94 #5)

## §A 审稿意见

iter #97 完成 3 个 Table footnote 之间的 cross-reference。 但 §5 Limitations (#3, #4)
只有 iter #94 给 #5 加的 "see Table 2 footnote for details" — #3 和 #4 没有 backward
cross-reference 指向对应 inline footnote。

reviewer 读完 §5 Limitations 想知道具体 paper-vs-release 差距, 需要被引导回
Table footnote:

| §5 Limitation | Topic | Related Table footnote |
|---------------|-------|------------------------|
| #3 | iter #79 guard + Table 1 Δ | Table 1 footnote (line 129) |
| #4 | Stage 12 outputs missing + Table 3 | Table 3 footnote (line 173) |
| #5 | Table 2 human-eval | Table 2 footnote (line 146) — already cross-referenced |

iter #98 添加 #3 和 #4 的 backward cross-reference 让 reviewer 从 §5 直接 navigate
回 inline footnote。

## §B 本轮 (iter #98) 改动

### §B.1 §5 #3 增加 cross-reference (line 197)

Original:
> "...previously blocked bootstrap confidence-interval computation for Table 1 Δ values."

Add after period:
> "See Table 1 footnote (line 129) for the quantitative impact on bootstrap CI."

### §B.2 §5 #4 增加 cross-reference (line 197)

Original:
> "...estimated total ≈9–21 hours for all three domains."

Add after period:
> "See Table 3 footnote (line 173) for the full list of 5 unverified audit claims and re-execution cost breakdown."

### §B.3 §5 #5 不变 (already references Table 2 footnote)

§5 #5 已经说 "see Table 2 footnote for details" (iter #94 加), 不重复添加。

## §C 关键改动点

1. **#3 cross-reference 添加**: 让 reviewer 知道 Table 1 footnote 有具体 bootstrap CI
   quantitative impact (per-cell NaN count + re-execution cost)
2. **#4 cross-reference 添加**: 让 reviewer 知道 Table 3 footnote 有 5 unverified
   claims 完整 list (不只在 §5 列 RQ4)
3. **#5 不变**: 已 iter #94 加 "see Table 2 footnote for details", 不重复

## §D 双向 cross-reference 现在完整

| From | To | Status |
|------|-----|--------|
| Table 1 footnote (line 129) | Table 2 footnote | ✓ iter #97 |
| Table 1 footnote (line 129) | Table 3 footnote | ✓ iter #97 |
| Table 2 footnote (line 146) | Table 1 footnote | ✓ iter #97 |
| Table 2 footnote (line 146) | Table 3 footnote | ✓ iter #97 |
| Table 3 footnote (line 173) | Table 1 footnote | ✓ iter #97 |
| Table 3 footnote (line 173) | Table 2 footnote | ✓ iter #97 |
| §5 #3 | Table 1 footnote | ✓ iter #98 (本轮) |
| §5 #4 | Table 3 footnote | ✓ iter #98 (本轮) |
| §5 #5 | Table 2 footnote | ✓ iter #94 |

现在 paper reviewer 双向 navigate 完整: Table footnote ↔ Table footnote ↔ §5。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:197` (§5 Limitations #3 + #4 各加 1 句 cross-reference)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "See Table 1 footnote\|See Table 3 footnote" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #98 完成 paper §5 Limitations ↔ Table footnote 双向 cross-reference:
- §5 #3 → Table 1 footnote (iter #98)
- §5 #4 → Table 3 footnote (iter #98)
- §5 #5 → Table 2 footnote (iter #94)

现在 reviewer 双向 navigate: §5 ↔ Table 1/2/3 footnotes 全部 covered。

后续 candidate:
- **iter #99** — paper abstract + §1 introduction 增补 brief audit mention (让 reviewer 一进 paper 就知道 release-vs-paper gap)
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 重跑 (1.5-3 h) — 修复 Table 1 bootstrap CI degenerate