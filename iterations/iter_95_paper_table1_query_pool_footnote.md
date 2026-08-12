# Iteration #95 — Paper §3 Table 1 footnote: Stage 6/9 query pool size caveat

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:124` (Table 1 footnote)
**prior**: iter #93 audit marked RQ1_Table1_Hit10 as `degenerate` (bootstrap_delta_ci.json 1152 NaN cells)

## §A 审稿意见

iter #93 audit 发现 RQ1_Table1_Hit10 degenerate:
- `bootstrap_delta_ci.json` 存在但 1152/1152 cell 都是 NaN
- 原因: Stage 6 retrieval pivot per-query records 是 empty list (post iter #79 guard)
- 进一步追查: Stage 6 source query pool Baby 95 / Grocery 89 / Pet 97, 但
  retrieval pivot 每个 (cat, retriever) cell 只有 2 queries — 不够 bootstrap

Paper §3.2 Table 1 列 SPLADE Δ=9.6 / E5 Δ=11.16 / ColBERTv2 Δ=8.6 等数字, 这些是
originally-computed 值 (full query pool), reviewer 跑 release pipeline 无法重现。

Reviewer-visible gap:
1. paper §3.2 直接说 "Table 1 reports two cluster-level dispersion measures..." 但
   没说明 release pipeline query pool 只有 2 queries/(cat, retriever)
2. paper §5 Limitations #3 提到 iter #79 guard 修复 bootstrap 阻塞, 但没说 Stage 6
   pipeline 当前仍 produce degenerate bootstrap CI

iter #95 修复 reviewer-visible gap: Table 1 footnote 解释 release pipeline query
pool size 与 paper originally-computed 值的差距。

## §B 本轮 (iter #95) 改动

### §B.1 Table 1 footnote (新增, after line 126)

```markdown
Table 1 footnote: The Δ Range values reported above (e.g. SPLADE Δ=9.6, E5 Δ=11.16,
ColBERTv2 Δ=8.6) were computed on the originally-released full Stage-6 query pool
(≈90 queries per category). The open-source release pipeline's Stage 6 retrieval
pivot (`result/personal_query/06_retrieval/<cat>/retrieval_syntax_depth_summary_pivot.json`)
currently records only 2 queries per (category, retriever) cell due to the post-iter-#79
TypeError guard preventing silent serialization of per-query records as non-lists;
consequently the per-query bootstrap confidence-interval pipeline
(`result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json`) emits
all-NaN values for every (category, retriever, metric) cell (audit status:
`degenerate`, see `result/personal_query/iterations/paper_claims_audit.json`
RQ1_Table1_Hit10 entry). Re-running Stage 6 + Stage 9 with the full ≈90-query pool
(estimated ≈30–60 min per category on GPU, ≈1.5–3 h total for all three domains)
would re-derive the Δ Range values and yield non-degenerate bootstrap CIs.
```

## §C 关键改动点

1. **Stage 6 query pool size quantitative**: footnote 列出 release pipeline 当前每个
   (cat, retriever) cell 只有 2 queries (vs full pool ≈90), 给 reviewer 具体差距
2. **audit 引用**: footnote 指向 `paper_claims_audit.json` 的 RQ1_Table1_Hit10
   `degenerate` 状态, reviewer 可自主验证
3. **iter #79 guard context**: 解释 post-iter-#79 query pool size 下降原因
   (TypeError guard 阻止 silent overwrite, 现在每 cell 只有 2 queries), 不让
   reviewer 误以为是 Stage 6 code 退化
4. **re-execution cost estimate**: ≈30-60 min/cat × 3 = ≈1.5-3 h total, 让 reviewer
   知道 reproduce cost

## §D 与 audit summary + §5 Limitations 的一致性

iter #93 audit summary 0/6/4/1/1/5/0:
- 1 degenerate (RQ1_Table1_Hit10) ↔ Table 1 footnote 明确 cite 此 claim
- §5 Limitations #3 提到 iter #79 guard, 现在 footnote 给 quantitative 后果
  (2 queries/cell → bootstrap CI NaN)
- 5 unverified (Stage 12 lineage) 不变 — Table 1 数字 reproduce 不依赖 Stage 12

reviewer 现在能完整追踪: paper Table 1 Δ → release 2-query cell → bootstrap CI NaN →
audit degenerate status → re-execution cost estimate。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:126` (Table 1 footnote 新增)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "Table 1 footnote" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #95 完成 paper §3 Table 1 reviewer-visible discrepancy disclosure:
- Table 1 footnote 量化 release pipeline query pool size gap (2 vs ≈90 queries/cell)
- audit `degenerate` 状态 cite 让 reviewer 验证
- re-execution cost estimate 让 reviewer 知道 reproduce cost

后续 candidate:
- **iter #96** — paper §3 Table 3 footnote: Stage 12 outputs missing (5 unverified claims 同根因, Stage 12 重跑 ≈45-135 min/cat)
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified + 修复 Table 3 reproducibility
- **iter #88** — Stage 7 重跑 (multi-hour infra) — unblock RQ1/RQ2 real Δ CIs + 修复 preservation gap