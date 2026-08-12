# Iteration #41 — 审稿人视角：§3.2-3.4 Δ值统计显著性 + LLM-human一致性批判

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: §3.2 Table 1 Δ值统计显著性缺失；§3.3 LLM-human一致性Spearman=0.81无代码实现；§3.4 GMM prior比较无显著性测试

---

## §A 审稿意见（尖锐批评）

---

### 问题 1: Major — Table 1 Δ 值完全没有统计显著性支撑，代码有 infrastructure 但未应用于 Δ

**论文位置**: §3.1-3.2, Table 1
**原文引用**:
> "SPLADE has the largest correct-query range (Δ =9.6) and a moderate error-effect range (Δ =6.51)"
> "Δ denotes the cross-cluster fluctuation averaged over the three domains."

**具体批评**:
Table 1 报告的 Δ 值（correct-query range 和 error-effect range）是**纯点估计**，没有任何：
1. 置信区间
2. 统计显著性测试
3. 多重比较校正

**但代码里其实有统计检验 infrastructure！**

`08_compare_p10_across_domains.py` 已经有：
- `bootstrap_mean_ci(values, n_boot=2000, confidence=0.95)` — line 656
- `stats.ttest_1samp(diffs_array, popmean=0.0)` — line 707
- `significance_star(p_value)` — line 76

**然而这些检验只用于 `print_09_paired_significance`（noisy vs correct 配对差），完全没有用于 Table 1 的 Δ Range 值。**

Table 1 的 Δ 值计算方式（以 SPLADE Δ=9.6 为例）：
- Correct-query range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) per domain，跨 3 个 domain 取均值
- Error-effect range = max(per-cluster error change) - min(per-cluster error change)，跨 3 个 domain 取均值

**这些 Δ 值本质上是聚合统计量（aggregate of aggregates），无法做 significance test 因为：**
1. Per-cluster Hit@10 是均值，不是个体观测值
2. 只有 3 个 domain，无法估计方差
3. Δ 本身是排序统计量（range = max - min），不是均值

**审稿人会问**:
> "Table 1 reports Δ values as point estimates without any statistical uncertainty quantification. Stage 08 has bootstrap CI and t-test infrastructure but only uses it for noisy-correct paired diffs, not for Δ Range. How should we interpret the claim 'SPLADE Δ=9.6 is the largest' without knowing whether this difference is statistically significant across domains?"

---

### 问题 2: Major — §3.4 GMM "best fits" 比较无统计显著性，循环论证升级版

**论文位置**: §3.4, Table 3
**原文引用**:
> "the Gaussian Mixture Model achieves the highest value on all three aggregated metrics"
> "these results show that the Multivariate Gaussian Mixture consistently outperforms the other priors"

**具体批评**:
Table 3 报告了 4 个 prior（Gaussian Mixture / t-distribution / Laplace / Logistic）在 3 个指标上的比较，GMM 全胜。但**没有任何统计显著性测试**来证明 GMM "显著优于" 其他 prior。

更重要的是，这个问题在 iter #38 已经被标记为"循环论证"，但 iter #41 要指出一个更具体的问题：

**代码里有没有对这个比较做统计检验？**

`train_vades_lite_sentence_latent_threshold.py` 中训练 VAE 时比较了不同的 prior distribution，但：
- 没有报告任何 model comparison statistics（BIC/AIC 差值的显著性）
- 没有做任何似然比检验（Likelihood Ratio Test）
- 论文的 "best fits" 结论是定性描述，不是统计推断

**更严重的是：** Table 3 的指标是 log p（对数似然）和 q50（median log p），这些都是在**同一训练数据**上计算的。GMM 有 2 个 mixture components，比其他单-component 模型有更多自由参数，所以 likelihood 更高是必然的。这不是"更正确"，是**过拟合的迹象**。

**审稿人会问**:
> "Table 3 shows GMM has the highest log p, but was any statistical test performed to determine whether this advantage is significant? With GMM having 2 mixture components vs 1 component for other priors, is the higher likelihood simply due to more parameters? What is the BIC or AIC comparison with proper penalty for model complexity?"

---

### 问题 3: Major — §3.3 LLM-Human Validation 完全没有在代码中实现

**论文位置**: §3.3, Table 2
**原文引用**:
> "Fleiss' Kappa across the three annotators is 0.72, reaching substantial agreement"
> "Spearman correlation between the LLM and human evaluations is 0.81"
> "MAE of the scoring results is 0.89"

**具体批评**:
Table 2 报告了 LLM-human agreement 的三个关键指标：Kappa=0.72, Spearman=0.81, MAE=0.89。但**代码中完全没有实现这个评估**。

搜索整个 codebase：
```
grep -rn "kappa\|fleiss\|spearman.*human\|human.*spearman" PersoanlQuery/ --include="*.py"
```
返回：只在 `evaluate_review_query_alignment.py` 中有 `spearmanr`，但那是用于 **review 风格 vs query 风格的 alignment**，不是 LLM-human agreement。

这意味着：
1. **这个实验是在代码 pipeline 之外单独运行的**（可能是独立的手动实验或 Python notebook）
2. 代码中没有可复现的 LLM-human evaluation 实现
3. 论文无法通过运行代码来验证这些数字

**Stage 13 的 `persona_utils.py` 有 relevance scoring prompt，但那评估的是 product-query relevance，不是 query quality 的 LLM evaluation。**

**审稿人会问**:
> "Table 2 reports Fleiss' Kappa=0.72 and Spearman=0.81 for LLM-human agreement, but no corresponding code exists in the repository. Where does this evaluation happen? Is there a separate script or notebook that is not in the main codebase? How can reviewers verify these numbers?"

---

### 问题 4: Minor — Stage 08 有 bootstrap infrastructure 但 Δ 值用不上

**论文位置**: §3.2 + `08_compare_p10_across_domains.py`
**具体批评**:

Stage 08 的 `bootstrap_mean_ci` 和 `ttest_1samp` 只能用于**个体观测值的差**（如 noisy-correct paired diff per query），无法用于 Δ Range，因为：

1. **Δ Range 是聚合统计量的函数**：Δ = max(C1..C8) - min(C1..C8)，不是均值的差
2. **per-cluster Hit@10 已经是均值**：每个 cluster 的 Hit@10 是该 cluster 内所有 query 的均值，不是原始观测值
3. **bootstrap 在聚合均值上无意义**：对均值再做 bootstrap 无法得到有意义的置信区间

**解决方案**：如果要对 Δ Range 做统计推断，需要：
- 原始 per-query Hit@10 数据（每个 query 一个 Hit@10 值）
- 按 cluster 分组
- 对每个 cluster 的 query-level Hit@10 做 bootstrap CI

但论文的 Table 1 只报告了聚合后的均值，完全没有 per-query 原始数据。

---

## §B 对应代码缺陷

| 审稿问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1: Δ 值无统计显著性 | `08_compare_all_domain/08_compare_p10_across_domains.py:382-418` | `print_08_hit10_table()` 只打印点估计，无 CI/significance |
| 问题1: bootstrap infrastructure 存在但未用于 Δ | `08_compare_p10_across_domains.py:656-713` | `bootstrap_mean_ci` 和 `ttest_1samp` 只用于 noisy-correct diff，不用于 Δ Range |
| 问题2: GMM 比较无统计检验 | `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py` | 无 BIC/AIC/LRT 比较；"best fits" 只是点估计描述 |
| 问题3: LLM-human eval 无代码 | 整个 codebase | 无实现 human annotation + LLM agreement 评估的代码 |
| 问题4: Δ 是聚合统计量 | `08_compare_p10_across_domains.py` | per-cluster Hit@10 已是均值，bootstrap 无意义 |

---

## §C 本轮代码优化

**无代码改动**（本轮为审稿分析轮，发现了深层次的统计和方法论问题）

---

## §D 验证

N/A（审稿分析轮次）

---

## §E Git Commit

N/A（审稿分析轮次）

---

## §F 最需要 rebuttal 的审稿意见

**Top 3 致命审稿意见**（本轮新增）：

1. **[Major] Table 1 Δ 值无统计显著性**
   - 必须在论文中补充 bootstrap CI 或明确说明 n=3 无法做统计推断
   - 或者重新设计实验：在更多 domain 上评估（>5 个 domain）

2. **[Major] §3.4 GMM prior 比较需要 model complexity 校正**
   - 补充 BIC/AIC 报告，对似然比做统计检验
   - 明确说明 GMM 的优势是否在控制了参数数量后仍然显著

3. **[Major] §3.3 LLM-human validation 代码缺失**
   - 提供可复现的代码或说明这是外部实验
   - 如果是外部实验，在论文中明确说明数据来源和实验设置

---

## §G 下轮建议

| 优先级 | 行动 |
|--------|------|
| P0 | 确认 §3.3 的 LLM-human evaluation 是否在代码中有独立实现 |
| P0 | 在 Stage 10 或 Stage 08 中补充 GMM vs other priors 的 BIC/AIC 报告 |
| P1 | 研究能否用原始 per-query Hit@10 数据（而非 per-cluster 均值）做 bootstrap CI |
| P2 | 在 Stage 08 添加 Δ Range 的敏感性分析：per-domain Δ 值是否稳定？ |
