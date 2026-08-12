# Iteration 257 — §3.4 Table 3 审查 + §5 Limitations 充分性检查 + Audit 同步

**日期**: 2026-07-22
**角色**: NLP/IR 专业审稿人
**scope**: §3.4 Table 3 claim 审查；§5 Limitations 现有内容充分性评估；audit.json Sec2_K2/Sec2_95pct 更新到 main checkout

---

## §A §3.4 Table 3 审查

### Paper §3.4 缺失

当前 paper markdown（139 行）**不包含 §3.4 节和 Table 3**。Paper 只有：
- §3.1: Experimental Setup (RQ1-4 questions + Table 1)
- §3.2: Retrieval Performance across Personalized Syntactic Structure Clusters

Table 1 footnote 提及 "Table 3 footnote (Stage 12 outputs missing, 8 unverified/partial claims)"，但 Table 3 内容本身未写入 paper。

### RQ4 相关 Claims 状态

| claim_id | Section | Status | Blocking |
|---|---|---|---|
| `RQ4_GMM_Best_Prior` | §3.4 + Table 3 | `partial` | GMM mechanism verified (K=8 via BIC), log p=-1.09 pending Stage 12 |
| `Sec2_K8_clusters_BIC_AIC_silhouette` | §2.2 | `partial` | K-selection mechanism verified (min BIC + max silhouette), actual K=8 pending Stage 12 |
| `Sec2_K2_GMM_per_user` | §2.2 | `partial` (本轮更新) | GMM_components=2 confirmed in code, output pending Stage 12 |
| `Sec2_95pct_holdout_threshold` | §2.2 | `partial` (本轮更新) | 95th percentile filtering mechanism confirmed in code, output pending Stage 12 |
| `Review_Query_Hypothesis_Correlation` | §2.2 (implicit) | `unverified` | framework (Kendall τ + bootstrap CI + perm test) completed, real-data pilot pending |
| `UserFilter_20_reviews_15_words` | §2.1 | `unverified` | code uses MIN_LONG_SENTENCES=10 not ≥20 reviews, paper-code drift |
| `Sec2_20dim_syntactic_features` | §2.2 | `unverified` | Stage 05/06 data lineage missing |

### Paper Table 3 的 claim 列表

根据 paper line 40 审稿摘要，Table 3 相关 claim（Stage 12 输出缺失导致 unverified/partial）：
1. `Sec2_20dim_syntactic_features` — unverified
2. `Sec2_K8_clusters_BIC_AIC_silhouette` — partial
3. `Sec2_K2_GMM_per_user` — partial (本轮)
4. `Sec2_95pct_holdout_threshold` — partial (本轮)
5. `Review_Query_Hypothesis_Correlation` — unverified

---

## §B §5 Limitations 充分性检查

### 当前 §5 Limitations 内容

Paper 仅有 line 139 footnote（Δ Range formula 澄清）和 line 140 footnote（STAR taxonomy），但**没有独立的 §5 Limitations 节**。

Paper line 40 的 "§5 Limitations lists re-execution costs" 描述不准确。

### 实际 Limitations 内容

基于 paper footnotes，当前已记录的 limitations：
1. **Δ Range 数据无法复现**：Stage 06/09 数据缺失，需要 rerun（≈1.5-3h GPU）
2. **Stage 12 GMM 输出缺失**：5 个 §2.2 claim 无法验证
3. **域泛化性**：3 个 Amazon domain 的结论泛化性未验证
4. **假设未验证**：review≠query 假设虽有 literature 支撑但 syntactic depth 专项未验证

### §5 缺失

Paper **没有独立的 §5 Limitations 节**。line 40 的描述错误（声称有 §5 Limitations 但实际没有）。这是 paper 结构问题，需要补充。

---

## §C Audit 同步问题

### 问题：Worktree 修改未同步到 main checkout

iter #254/#255 的 worktree 修改了 worktree 内的 audit.json（Sec2_K2 → partial, Sec2_95pct → partial），但 worktree 未与 main 合并。

本轮在 main checkout 直接更新：
- `Sec2_K2_GMM_per_user`: unverified → partial
- `Sec2_95pct_holdout_threshold`: unverified → partial

**当前 audit.json 状态（iter #257）**：
- 2 verified + 1 verified_value_match + 4 discrepant + 3 degenerate + 5 partial + 3 unverified = 18 claims

---

## §D §3.4 审稿意见

### 问题 1: Paper §3.3、§3.4、§4、§5 全部缺失

**严重程度**: Major

Paper line 40 审稿摘要提到 Table 2/3 footnotes 和 §5 Limitations，但 paper markdown 中 **§3.3、§3.4、§4、§5 全部不存在**。Paper 仅有 139 行，结构停在 §3.2。

**实际 paper 结构**：
```
§1 Introduction (line 17)
§2 Benchmark Construction (line 42)
  §2.1 Task Definition (line 44)
  §2.2 Personalized Query Generation (line 50)
§3 Retrieval Benchmark Experiments (line 114)
  §3.1 Experimental Setup (line 116)
  §3.2 Retrieval Performance (line 135)
[§3.3 MISSING]
[§3.4 MISSING]
[§4 Related Work MISSING]
[§5 Conclusion/Limitations MISSING]
```

**缺失内容**：
- §3.3: LLM full-set evaluation and human sample validation consistency
- §3.4: Multivariate Gaussian Mixture prior evaluation (RQ4)
- §4: Related Work（loop.md iter #48 声称已扩展但实际没有）
- §5: Conclusion + Limitations（loop.md iter #48 声称已扩展但实际没有）

**证据**：Paper 全文搜索 `## 4\|## 5\|§3.3\|§3.4` 无结果，`tail paper.md` 显示最后内容为 §3.2 footnotes。

**对应的 claim 无法审查**：
- RQ3 (LLM vs human eval): §3.3 完全不存在
- RQ4 (GMM prior): §3.4 完全不存在，Table 3 不存在
- Related Work: §4 完全不存在
- Limitations: §5 完全不存在

---

## §E 下一步

**iter #258**: 
1. 确认 paper §3.4 + Table 3 是否计划写入（还是仅为 reference 而非实际内容）
2. 如果需要写入，审查 Table 3 内容的 claim 依据
3. 补充 §5 Limitations 节（如果需要）

**不修复**（Rule 7）：
- paper §3.4 + Table 3 完全缺失是 paper 写作问题，超出代码审计范围
- 需要作者确认 Table 3 是否应该存在
