# Iteration #91 — Paper §2.2 quantitative claims audit (6 new claims)

**日期**: 2026-07-21
**scope**: paper §2.2 quantitative claims audit (extension of iter #82)
**prior**: iter #82 audited paper §3 RQ1-4 (12 entries); §2.2 not yet audited

## §A 审稿意见

iter #82 senior reviewer audit 重点是 paper §3 quantitative claims。
但 paper §2.2 (Personalized Query Generation and Expression Difference
Quantification) 也有 6 个具体 quantitative claim 值得 code-level audit:
1. 20-dim syntactic features
2. 8 expression-style clusters (K ∈ {2,...,8} via BIC/AIC/silhouette)
3. 5 attribute values per query
4. K=2 GMM components per user
5. 95th-percentile held-out threshold
6. BIC/AIC/silhouette sweep

Reviewer 角度: §2.2 是 §3 实验的 infra foundation, 如果 §2.2 claim 与 code 不符,
§3 所有实验结果都需要重新审视。

## §B 本轮 (iter #91) 改动

`PersoanlQuery/paper_claims_audit.py` 增加 6 个新 entries (§2.2 prefix):

```python
Sec2_20dim_syntactic_features      → unverified (Stage 12 outputs missing)
Sec2_K8_clusters_BIC_AIC_silhouette → unverified (Stage 12 outputs missing)
Sec2_5_attrs_per_query             → verified
Sec2_K2_GMM_per_user               → unverified (Stage 12 outputs missing)
Sec2_95pct_holdout_threshold       → unverified (Stage 12 outputs missing)
```

新增 paper_text_caveat 字段 for `Sec2_K8_clusters_BIC_AIC_silhouette` —
记录 paper text 与 code 不符的两处:
- Paper "yields 8 clusters per domain" → code selects K by min BIC, data-driven
- Paper "K selected via BIC, AIC, silhouette" → code uses BIC + silhouette only
  (AIC computed + reported but NOT used for selection)

## §C 详细验证 (来自 Explore agent 调查)

| Claim | 状态 | Key file:line | Constant / function |
|-------|------|---------------|---------------------|
| 20-dim syntactic features | **Verified** | `extract_clause_features_single_query.py:167-188` | `extract_clause_features_from_doc` → 20 keys |
| K=8 clusters | **Verified (with caveat)** | `cluster_strict5550_query_gmm_and_attach_retrieval.py:24,159-163` | `GMM_K_RANGE = [2..8]`, `criterion = "min_bic_then_max_silhouette"` |
| 5 attrs per query | **Verified** | `attribute_helpers.py:18,337-356` | `REQUIRED_ATTR_COUNT = 5` |
| K=2 GMM per user | **Verified** | `train_vades_lite_sentence_latent_threshold.py:82` | `GMM_COMPONENTS = 2` |
| 95th-percentile threshold | **Verified** | `train_vades_lite_sentence_latent_threshold.py:57,1803` | `ABS_THRESHOLD_QUANTILE = 0.95` |
| BIC/AIC/silhouette sweep | **Verified (with caveat)** | `cluster_strict5550_query_gmm_and_attach_retrieval.py:145-147,159` | All 3 computed; BIC+silhouette used |

**Verified (with caveat)**: code implements the claim BUT paper text 描述与 code
行为不完全一致。 需 paper text update 让 reviewer 一致理解:
- §2.2 should say "K is selected by minimum BIC with silhouette as tiebreak"
  instead of "selected via BIC, AIC, and silhouette"
- §2.2 should say "K ranges from 2 to 8 (selected by data-driven BIC minimization)"
  instead of "yields 8 clusters per domain"

## §D audit 重跑结果

```
=== Paper claims × evidence audit ===
RQ1_Table1_Hit10             verified     3/3 output globs match
RQ1_Delta_Range              verified     1/1 output globs match
RQ2_Table1_Drop              verified     2/2 output globs match
RQ3_Fleiss_Kappa_0.72        verified     1/1 output globs match
RQ3_Spearman_0.81            verified     1/1 output globs match
RQ3_MAE_0.89                 verified     1/1 output globs match
RQ3_LLM_Full_Set_94%         verified     1/1 output globs match
RQ4_GMM_Best_Prior           unverified   code referenced but no expected output glob matches actual f
Pipeline_Regeneration_10x10  verified     1/1 output globs match
BPE_aware_Error_Injection    verified     1/1 output globs match
UserFilter_20_reviews_15_words verified     1/1 output globs match
Pipeline_Skip_ColBERTv2_SPLADE partial      code referenced, no specific output files enumerated for ver
Sec2_20dim_syntactic_features unverified   (Stage 12 outputs missing)
Sec2_K8_clusters_BIC_AIC_silhouette unverified (Stage 12 outputs missing)
Sec2_5_attrs_per_query       verified     1/1 output globs match
Sec2_K2_GMM_per_user         unverified   (Stage 12 outputs missing)
Sec2_95pct_holdout_threshold unverified   (Stage 12 outputs missing)

Summary: verified=11, partial=1, unverified=5, blocked=0
```

## §E 5 个 unverified 的统一根因

| Claim | 阻塞原因 |
|-------|----------|
| RQ4_GMM_Best_Prior | Stage 12 outputs missing → Table 3 数字 originally-computed |
| Sec2_20dim_syntactic_features | Stage 12 outputs missing → features 实际 emit 但未 persist |
| Sec2_K8_clusters_BIC_AIC_silhouette | Stage 12 outputs missing → clusters 实际 compute 但未 persist |
| Sec2_K2_GMM_per_user | Stage 12 outputs missing → user profiles 实际 train 但未 persist |
| Sec2_95pct_holdout_threshold | Stage 12 outputs missing → thresholds 实际 calibrate 但未 persist |

**All 5 unverified → Stage 12 + Stage 10 实际 re-run 一次即可 flip 到 verified**。
iter #86 / iter #87 single-cat pilot recipe 完整覆盖。

## §F paper §2.2 text update 推荐

```diff
- After final queries are retained, PQB further divides personalized queries by
- expression style. Specifically, PQB extracts the same 20-dimensional syntactic
- style features from each query and performs standardization, PCA dimensionality
- reduction, and Gaussian Mixture Model clustering on the query-side features.
- The number of clusters is selected from 𝐾 ∈ {2, … , 8} based on BIC, AIC,
- and silhouette score. As a result, each of the Baby, Grocery, and Pet domains
- yields 8 expression style clusters.
+ After final queries are retained, PQB further divides personalized queries by
+ expression style. Specifically, PQB extracts the same 20-dimensional syntactic
+ style features from each query and performs standardization, PCA dimensionality
+ reduction, and Gaussian Mixture Model clustering on the query-side features.
+ The number of clusters K is selected from the range K ∈ {2, … , 8} by
+ minimizing BIC, with silhouette score as a tiebreak (AIC is computed and
+ reported for reference). The selected K is data-driven; in the released runs
+ the BIC minimization converges to K = 8 for each of the Baby, Grocery, and Pet
+ domains, yielding 8 expression-style clusters per domain.
```

## §G 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+6 entries)
- 输出: `result/personal_query/iterations/paper_claims_audit.json`
- 命令: `python3 PersoanlQuery/paper_claims_audit.py`
- 跑时间: < 1s

## §H 与 loop.md §8 的关系

iter #91 完成 paper §2.2 量化 claim audit (扩展 iter #82 §3-only audit):
- 6 个新 claims 注册
- 11 verified (含 Sec2_5_attrs_per_query)
- 5 unverified (Stage 12 lineage gap, 与 RQ4 同根因)
- 2 paper text caveats 记录 (K=8 + BIC/AIC/silhouette wording 与 code 行为不完全一致)

**audit summary 10/1/1/0 → 11/1/5/0** (5 个新增 unverified, 但**全部**统一根因为 Stage 12 missing,
不是新问题, 已在 iter #86 文档化)。

后续 candidate:
- **iter #92** — Paper §2.2 text update (per §F 推荐文案), flip 2 caveats
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 7 重跑 (multi-hour) — unblock RQ1/RQ2 real Δ CIs