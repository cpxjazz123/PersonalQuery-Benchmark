# Iteration 175 — GMM BIC/AIC Parameter Count Bug

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §3.4 GMM prior BIC/AIC parameter count vs train_vades_lite implementation

## §A 审稿意见（尖锐批评）

### 问题: GMM BIC parameter count wrong → circular reasoning

**严重程度**: Major

**具体批评**:

`PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py:117-122` 的参数计数与代码实际实现的 nn.Module 形状不符:

| 分布 | 代码声明 k | 实际 nn.Module 形状 | 正确 k |
|------|------------|---------------------|--------|
| GMM (K=2) | `2*K*d + (K-1)` = 33 | mu[K,D] + logvar[K,D] + mix_logits[K] | `2*K*d + K` = **34** |
| Laplace | `2*d + 1` = 17 | mu[D] + log_b[D] | `2*d` = **16** |
| Logistic | `2*d + 1` = 17 | mu[D] + log_s[D] | `2*d` = **16** |
| t-distribution | `2*d + 1` = 17 | mu[D] + log_scale[D] + df_raw[1] | `2*d + 1` = **17** ✓ |

具体证据 (train_vades_lite_sentence_latent_threshold.py):
- L554-556 `UserDistributionTableGMM`: `user_mu[U,K,D]` + `user_logvar[U,K,D]` + `mix_logits[U,K]` → 2*K*d + K (softmax logits are K free, not K-1)
- L600-606 `UserDistributionTableLaplace`: `user_mu[U,D]` + `user_log_b[U,D]` → 2*d (no extra scale)
- L620-626 `UserDistributionTableLogistic`: `user_mu[U,D]` + `user_log_s[U,D]` → 2*d (no extra scale)
- L568-583 `UserDistributionTableStudentT`: `user_mu[U,D]` + `user_log_scale[U,D]` + `user_df_raw[U]` → 2*d + 1 ✓

**论文位置**: PersonalQuery-Benchmark_evaluating_retrieval.md §3.4 + Table 3

**论文原文**:
> "PQB further divides personalized queries by expression style. ... performs ... Gaussian Mixture Model clustering ... number of clusters K is selected from the range K ∈ {2, …, 8} by minimizing the Bayesian Information Criterion (BIC)"

**对结论的影响**:

BIC = k * log(n) - 2 * log(L)

修复前 (n=1000):
- GMM penalty: 33 * log(1000) = 227.96
- Laplace penalty: 17 * log(1000) = 117.43
- GMM 占优势 = (33-17) * 6.908 = **110.5 优势**

修复后 (n=1000):
- GMM penalty: 34 * log(1000) = 234.86
- Laplace penalty: 16 * log(1000) = 110.52
- GMM 占优势 = (34-16) * 6.908 = **124.4 优势**

GMM 需要的 log-likelihood 优势从 55.3 增加到 62.2 (Δ +6.9)。这是个真正的修正，但量级不大 — GMM 的优势更明显了，所以"circular reasoning"的方向不是"高估 GMM"，反而是"低估 GMM 参数"。

更严重的是：**Laplace/Logistic 的参数被高估 1**（多了 1 个 free param），让它们的 BIC 看起来更差，相当于偏袒 GMM/t-distribution。

**结论**: 这是参数计数公式与 nn.Module 形状不匹配的代码缺陷，iter #41 设计的 BIC/AIC framework 没有核对实际模型形状。

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| §3.4 Table 3 BIC 验证 | compute_prior_bic_aic.py:117-122 | GMM/Laplace/Logistic 参数计数与 train_vades_lite nn.Module 形状不匹配 |

## §C 本轮代码优化

- **修复**: `compute_prior_bic_aic.py` 参数计数公式:
  - GMM: `2*K*d + (K-1)` → `2*K*d + K` (mix_logits 经 softmax 后 K 个自由参数, 不是 K-1)
  - Laplace: `2*d + 1` → `2*d` (无 df)
  - Logistic: `2*d + 1` → `2*d` (无 df)
  - t-distribution: `2*d + 1` (保持不变, df 经 sigmoid 映射)
- **更新 docstring**: 列出对应 nn.Module 类名, 避免未来 mismatch
- **注释**: 添加每条公式的 nn.Module shape 引用

## §D 验证

- `python3 -m py_compile PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py` 通过
- 数值 sanity check (d=8, K=2, n=1000):
  - GMM k: 33 → 34 (BIC penalty +6.91)
  - Laplace k: 17 → 16 (BIC penalty -6.91)
  - Logistic k: 17 → 16 (BIC penalty -6.91)
  - t-dist k: 17 (不变)
  - GMM 占优势从 110.5 → 124.4 (Δ +13.8)

## §E Git Commit

- iter #175: fix GMM/Laplace/Logistic BIC parameter count in compute_prior_bic_aic.py
