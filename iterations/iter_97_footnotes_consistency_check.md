# Iteration #97 — Paper §3 footnote consistency + cross-reference check

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:129,146,173`
**prior**: iter #94-#96 added 3 table footnotes (Table 1, Table 2, Table 3)

## §A 审稿意见

iter #94-#96 在 paper §3 加了 3 个 table footnotes 解释 paper vs release 差距。
但 3 个 footnotes 是分别加的, 没 cross-reference, reviewer 看完一个 footnote 可能
不知道还有两个相关 footnotes。

reviewer-visible issue:
1. Table 1 footnote (line 129) 说 "audit status: `degenerate`" 但没指向 Table 2/3 footnotes
2. Table 2 footnote (line 146) 说 "see `paper_claims_audit.json`" 但没指向 Table 1/3 footnotes
3. Table 3 footnote (line 173) 列 5 unverified claims 但没指向 Table 1/2 footnotes

reviewer 想知道 paper overall release-vs-paper discrepancy 全貌, 需要能从一个
footnote navigate 到其他两个。

## §B 本轮 (iter #97) 改动

### §B.1 验证 3 个 footnotes 的 audit claim IDs

确认 3 个 footnotes 引用的 audit claim IDs 都真实存在:

| Footnote | Cited claim IDs | Verified in audit JSON |
|----------|-----------------|------------------------|
| Table 1 (line 129) | `RQ1_Table1_Hit10` | ✓ status `degenerate` |
| Table 3 (line 173) | `RQ4_GMM_Best_Prior` | ✓ status `unverified` |
| Table 3 (line 173) | `Sec2_20dim_syntactic_features` | ✓ status `unverified` |
| Table 3 (line 173) | `Sec2_K8_clusters_BIC_AIC_silhouette` | ✓ status `unverified` |
| Table 3 (line 173) | `Sec2_K2_GMM_per_user` | ✓ status `unverified` |
| Table 3 (line 173) | `Sec2_95pct_holdout_threshold` | ✓ status `unverified` |

All 6 referenced IDs exist in `paper_claims_audit.json` with matching status.

### §B.2 添加 cross-reference line 到每个 footnote

每个 footnote 末尾添加 "See also" 一行指向其他 footnotes:

#### Table 1 footnote (line 129) 添加:
> See also Table 2 footnote (paper-vs-release numerical discrepancies in human-eval
> and LLM full-set semantic evaluation) and Table 3 footnote (Stage 12 outputs
> missing, 5 unverified claims).

#### Table 2 footnote (line 146) 添加:
> See also Table 1 footnote (Stage 6 query-pool size and bootstrap CI degenerate
> status) and Table 3 footnote (Stage 12 outputs missing, 5 unverified claims).

#### Table 3 footnote (line 173) 添加:
> See also Table 1 footnote (Stage 6 query-pool size and bootstrap CI degenerate
> status) and Table 2 footnote (paper-vs-release numerical discrepancies in
> human-eval and LLM full-set semantic evaluation).

### §B.3 验证 formatting 一致性

| Aspect | Table 1 | Table 2 | Table 3 | Status |
|--------|---------|---------|---------|--------|
| Prefix | "Table 1 footnote:" | "Table 2 footnote:" | "Table 3 footnote:" | ✓ |
| Audit JSON path | full path | full path | full path | ✓ |
| Re-execution cost | ≈1.5-3 h total | (n/a, one-off annot.) | ≈9-21 h total | ✓ different stages |
| Iter #NN reference | iter #79 (guard) | (none) | iter #69 (restored) | ✓ different iters |
| Audit claim IDs cited | 1 | (full JSON ref) | 5 | ✓ proportional |

## §C 关键改动点

1. **Cross-references 加 3 处**: 每个 footnote 末尾加 "See also Table N / Table M footnote"
   1-sentence pointer, reviewer 可从一个 footnote 看到其他两个
2. **Audit JSON consistency 验证**: 3 个 footnote 都 reference
   `result/personal_query/iterations/paper_claims_audit.json` (full path, no abbreviation)
3. **Audit claim IDs 全部 verified**: 6 个引用 IDs 都 in audit JSON with matching status
4. **No orphan references**: 没指向不存在的 footnote / 文件路径

## §D 与 audit summary 的一致性

iter #93 audit summary 0/6/4/1/1/5/0:
- 1 degenerate (RQ1) → Table 1 footnote
- 4 discrepant (RQ3) → Table 2 footnote
- 5 unverified (RQ4 + 4 Sec2) → Table 3 footnote
- 6 verified (boolean) → 不在 footnote (没 discrepancy 需要 disclosure)
- 1 partial (Pipeline_Skip_ColBERTv2_SPLADE) → 不在 footnote (env-var feature, 无需 paper disclosure)

所有 11 non-verified claims 都有 reviewer-visible disclosure (footnote + §5 Limitations)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:129,146,173` (各加 1 句 See also 行)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "See also Table" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #97 完成 paper §3 footnote 互链 + audit claim ID 一致性验证:
- 3 个 footnote 都 cross-reference 其他 2 个
- 6 个 audit claim ID 引用全部 verified
- 11 non-verified audit claims 全部 reviewer-visible disclosed

后续 candidate:
- **iter #98** — §5 Limitations 增补 cross-reference to Table 1/2/3 footnotes (让 reviewer 从 §5 也能 navigate 到 inline footnote)
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 重跑 (1.5-3 h) — 修复 Table 1 bootstrap CI degenerate