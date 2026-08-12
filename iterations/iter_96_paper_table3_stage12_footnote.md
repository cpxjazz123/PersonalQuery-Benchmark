# Iteration #96 — Paper §3 Table 3 footnote: Stage 12 outputs missing (5 unverified claims)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:171` (Table 3 footnote)
**prior**: iter #93 audit marked 5 claims as `unverified` due to Stage 12 outputs missing

## §A 审稿意见

iter #93 audit summary 0/6/4/1/1/5/0 中 5 个 unverified 全部统一根因:
- **RQ4_GMM_Best_Prior** — Stage 12 outputs missing → Table 3 数字 originally-computed
- **Sec2_20dim_syntactic_features** — Stage 12 outputs missing → features emit 但未 persist
- **Sec2_K8_clusters_BIC_AIC_silhouette** — Stage 12 outputs missing → clusters compute 但未 persist
- **Sec2_K2_GMM_per_user** — Stage 12 outputs missing → user profiles train 但未 persist
- **Sec2_95pct_holdout_threshold** — Stage 12 outputs missing → thresholds calibrate 但未 persist

Paper §3.4 Table 3 (lines 167-171) 列 Gaussian Mixture log 𝑝=-1.09 等数字, 没任何
table-level disclosure of release-vs-paper discrepancy. §5 Limitations #4 (line 193)
有说明 Stage 12 outputs missing, 但 reviewer 看完 Table 3 直接往下读 §4, 不一定
会回头看 §5 Limitations。

iter #95/iter #94 给 Table 1 + Table 2 都加了 inline footnote, iter #96 给 Table 3
也加同样的 inline footnote 让 reviewer 直接在 Table 3 看到 disclosure。

## §B 本轮 (iter #96) 改动

### §B.1 Table 3 footnote (新增, after line 171)

```markdown
Table 3 footnote: The 3-domain log 𝑝, intercept, and median-user log 𝑝 values
reported above (GMM log 𝑝=-1.09, t=-1.24, Laplace=-1.22, Logistic=-1.41; GMM intercept=-0.39,
t=-0.98, Laplace=-1.07, Logistic=-1.30; GMM 𝑞50=-1.06, t=-1.22, Laplace=-1.21,
Logistic=-1.40) were computed on the originally-released Stage 12 PRF clause-features
output (`result/personal_query/12_complexity_analysis_clause_features/<cat>/<tag>/user_profiles.jsonl`
and `sentences.jsonl`). The open-source release does not persist these per-category
files; `compute_prior_bic_aic.py` therefore falls back to a parameter-count stub per
category with a printed WARNING (see `PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py`,
iter #69 restored). Five audit claims are marked `unverified` due to this lineage gap:
RQ4_GMM_Best_Prior (Table 3 numbers), Sec2_20dim_syntactic_features, Sec2_K8_clusters_BIC_AIC_silhouette,
Sec2_K2_GMM_per_user, Sec2_95pct_holdout_threshold (all §2.2 infrastructure claims).
Re-running Stage 12 (≈45–135 min per category on GPU, ≈9–21 h total for all three
domains) would re-derive Table 3 row by row and flip these 5 claims to `verified`.
See `result/personal_query/iterations/paper_claims_audit.json` for full audit status.
```

## §C 关键改动点

1. **Table 3 specific numbers disclosed**: footnote 列出所有 12 个 cell 的 paper
   originally-computed 数字 (GMM/t/Laplace/Logistic × 3 metric), 让 reviewer 能
   直接 cite 具体数字
2. **5 unverified claims 列出**: RQ4 + 4 个 Sec2 §2.2 claims, 让 reviewer 知道
   这些是 unverified 而非 verified
3. **§2.2 + §3.4 cross-reference**: footnote 同时 cite Table 3 (RQ4) 和 §2.2
   (Sec2_K8/BIC/AIC/silhouette 等), 把 paper 多个 section 关联起来
4. **re-execution cost explicit**: ≈45-135 min/cat × 3 = ≈9-21 h total, reviewer
   知道 reproduce cost

## §D 与 audit summary + §5 Limitations 的一致性

iter #93 audit summary 0/6/4/1/1/5/0:
- 5 unverified (RQ4 + 4 Sec2) ↔ Table 3 footnote 明确 cite 这 5 个 claim
- 1 degenerate (RQ1) ↔ Table 1 footnote (iter #95) 已 cite
- 4 discrepant (RQ3) ↔ Table 2 footnote (iter #94) 已 cite
- 现在 paper 3 个 quantitative table 全部有 reviewer-visible disclosure

§5 Limitations #4 已 mention Stage 12 outputs missing (line 193); Table 3 footnote
提供 table-level inline disclosure, reviewer 不需要回头看 §5。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:171` (Table 3 footnote 新增)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "Table 3 footnote" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #96 完成 paper §3 Table 3 reviewer-visible disclosure:
- Table 3 footnote 列出所有 12 个 cell paper numbers
- 5 unverified claims 列出 (RQ4 + Sec2_K8/BIC/K2_GMM/95pct_holdout)
- re-execution cost ≈9-21 h total

现在 paper 3 个 quantitative table 都有 inline disclosure (Table 1/2/3 全部 covered)。

后续 candidate:
- **iter #97** — cross-table consistency check: 验证 Table 1/2/3 footnote 都 consistent + cross-reference 互链
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified + 修复 Table 3 reproducibility
- **iter #88** — Stage 6/9 重跑 (1.5-3 h) — 修复 Table 1 bootstrap CI degenerate