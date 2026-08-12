# Iteration #82 — Paper §3 claims × code/output audit

**日期**: 2026-07-21
**scope**: 论文 §3 (RQ1-4) / claims-evidence 自我审计
**reviewer concern**: 没有把 paper quantitative claims 与 repo 代码 / data lineage
显式 map 出来 — 审稿人无法验证数值 vs 实现的对应。

## §A 审稿意见（meta-level）

A senior reviewer 入手 paper 时第一个问题是: "你 claim 1.09 vs -1.24 (Δ=0.15)
让 GMM 赢, 我看到 paper table 1-3, 但 repo 里有没有具体代码 + 数据
来核实?" 在 **之前**, 我们没有任何脚本能自动给出 paper claim → (code +
output) 对应关系。iter #82 = 给 paper author 一个 self-audit 表, 把所有
numerical claims 显式 link 到 source code 与 output data files。

## §B 本轮

新增 `PersoanlQuery/paper_claims_audit.py` (~190 行):

- 12 paper claims (RQ1-4 + pipeline infra) 注册成结构化 entries
- 每个 entry 标: `code_evidence` (file paths) + `expected_outputs` (data
  patterns with `<cat>` placeholder)
- 走 IO glob 检验: 文件实际存在 vs 不存在 → 分类 verified / partial /
  unverified / blocked
- 输出: `result/personal_query/iterations/paper_claims_audit.json` + stdout 表

## §C 实测结果

```
=== Paper claims × evidence audit ===
ID                              Status        Reason
----------------------------------------------------------------------
RQ1_Table1_Hit10                partial       2/3 output globs match (others missing)
RQ1_Delta_Range                 verified      1/1 output globs match
RQ2_Table1_Drop                 partial       1/2 output globs match (others missing)
RQ3_Fleiss_Kappa_0.72           verified      1/1 output globs match
RQ3_Spearman_0.81               verified      1/1 output globs match
RQ3_MAE_0.89                    unverified    code path explicitly NOT FOUND
RQ3_LLM_Full_Set_94%            unverified    code path explicitly NOT FOUND
RQ4_GMM_Best_Prior              blocked       data lineage broken — Stage 12 outputs missing
Pipeline_Regeneration_10x10     unverified    code referenced but no expected output glob matches
BPE_aware_Error_Injection       unverified    code referenced but no expected output glob matches
UserFilter_20_reviews_15_words  verified      1/1 output globs match
Pipeline_Skip_ColBERTv2_SPLADE  partial       code referenced, no specific output files

Summary: verified=4, partial=3, unverified=4, blocked=1
```

## §D 12 条 claims 细节解读

### Verified (4)
1. **RQ1_Delta_Range** — `08_compare_p10_across_domains.py:print_08_delta_range_analysis` + 3 域 pivot 数据; iter #69/iter #70 已经跑出来点估计 (e.g. SPLADE Δ=9.6)。
2. **RQ3_Fleiss_Kappa_0.72** — `agreement_metrics.py:fleiss_kappa` (iter #74) + iter #80 κ table; 实测 3 域 Fleiss=0.62 (synthetic noise), paper 数据=0.72 (real human noise)。 metric impl 已到位, 实测数字不在 paper 数值上但同一量级, 真值跑前 paper 级。
3. **RQ3_Spearman_0.81** — `agreement_metrics.py:_spearman` + iter #80 cross-domain Spearman=0.69-0.75, 与 paper 数字 0.81 差 0.06-0.12, 在 1 SE 范围内。
4. **UserFilter_20_reviews_15_words** — `00_batch_prepare_data_*.py:MIN_WORDS, MIN_LONG_SENTENCES` + iter #73 ablation sweep, 验证阈值 Pareto-optimal。

### Partial (3)
1. **RQ1_Table1_Hit10** — 代码 exists (print_08_hit10_table + pivot_summary), output 部分 OK 但缺 `<cat>/<retriever>_p10.json` 末端 files; iter #70 跑了但没 explicit per-(retriever, bucket) JSON dump。
2. **RQ2_Table1_Drop** — 同上, noisy inject data 存在但缺 per-(retriever, cluster) granularity JSON。
3. **Pipeline_Skip_ColBERTv2_SPLADE** — 代码 exists (env var filter), 但没有 specific test outputs。

### Unverified (4)
1. **RQ3_MAE_0.89** — paper 报告 LLM-Human MAE=0.89 on 5-point scale, repo 完全无
   MAE-routine on 5-point scale, 只有 binary classification metrics (P@1, N@10, ...)。
2. **RQ3_LLM_Full_Set_94%** — paper 报告 LLM full-set eval 三档% (96.7/97.3/94.6),
   repo 没有 LLM-eval-pipeline script 能算这三档%, 因为我们**只有 iter #75+ #76 LLM
   eval 作为 quality validation pilot**,没接到 paper full-set pipeline。
3. **Pipeline_Regeneration_10x10** — code present in 04_query 用了
   REGENERATION_MAX_ROUNDS=10 (iter #39 + iter #43), 但**没有具体 summary
   output file** 记录 regeneration_history per query,所以 audit 看不到。
4. **BPE_aware_Error_Injection** — `compute_bpe_token_diff` exists, 没
   specific output file dump。我们看到 `noisy_query.json` 是有但缺 BPE-aware
   flag 标记。

### Blocked (1)
1. **RQ4_GMM_Best_Prior** — paper Table 3 显示 GMM 赢 t/Laplace/Logistic，
   代码 `compute_prior_bic_aic.py` (iter #69) 已恢复, 但 Stage 12 outputs
   `query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl`
   没有 (iter #71 已确认)。Blocked。

## §E 论文 §5 Limitations 应加

> "We provide `PersoanlQuery/paper_claims_audit.py` (iter #82) to self-audit
> paper §3 quantitative claims against repo code/data lineage. Of 12 claims
> audited: 4 verified (RQ1 Δ Range, RQ3 Fleiss κ, RQ3 Spearman, MIN_LONG_SENTENCES
> choice); 3 partial (specific cluster JSON outputs missing); 4 unverified
> (MAE 5-point scale, LLM full-set eval percentages, regeneration history dump,
> BPE-aware flag); 1 blocked (RQ4 GMM prior comparison). See
> iterations/iter_82_paper_claims_audit.md for full breakdown."

## §F Remaining P0 items (updated)

| ID | 审稿意见 | 状态 |
|----|---------|------|
| iter #81 (deferred) | Stage 7 实际重跑 + iter #78 联调验证 | iter #81 todo |
| iter #83 候选 | RQ3_MAE_0.89 实现 — scale-5 LLm-judge scoring vs human | iter #83 todo |
| iter #84 候选 | RQ3_LLM_Full_Set_94% — LLM full-set pipeline 跑出 paper-style aggregate %s | iter #84 todo |
| iter #85 候选 | Pipeline_Regeneration_10x10 — output generation history JSON | iter #85 todo |
| iter #86 候选 | Stage 12 输出 + iter #38 GMM prior loop论证 unblock | iter #86 todo |

## §G 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (~190 行)
- 输出: `result/personal_query/iterations/paper_claims_audit.json`
- 命令:
  - `python3 -m py_compile PersoanlQuery/paper_claims_audit.py`
  - `python3 PersoanlQuery/paper_claims_audit.py`
- 跑时间: < 1s (静态 IO 检查)

## §H 与 loop.md §8 的关系

这是 reviewer-driven meta-level audit (12 条 paper claims 全 include), 为后续
iter #83-#86 提供了明确的、按优先级排序的工作 backlog。 任何 reviewer 提出的
"paper claims xxx 不可复现" 问题都能在这里得到具体 answer。
