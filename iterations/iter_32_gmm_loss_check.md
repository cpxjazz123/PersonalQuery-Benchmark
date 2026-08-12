# Iteration #32 — 论文第2次通读：§2.2 GMM/VAE 训练目标（Eq.1-8）+ Stage 10 对照检查

**日期**: 2026-07-20
**scope**: 论文第2次通读：§2.2 GMM/VAE 训练目标 + Stage 10 loss 实现完整性检查
**关联 stage**: 10_complexity_analysis

## §A 论文发现

### §2.2 VAE/GMM 训练目标（Eq.1-8）
论文定义了以下 loss terms：
1. **Feature reconstruction loss**（Eq.4）：reconstruct syntactic features from encoder mean μ_i
2. **Sentence-user alignment loss**（Eq.6）：KL(q_φ(z_i|x_i) || p_ψ(z_i|u)) — KL between sentence latent and user mixture
3. **User-match loss**（cross-entropy, Eq.7）：classify sentence x_i to owning user u
4. **Regularization**：sentence-level KL to N(0,I) + user-level KL to N(0,I)
5. **Per-user mixture weights** w_{u,k}, μ_{u,k}, σ^2_{u,k}
6. **95th percentile filtering**（Section 2.2 最后）：candidate query 的 surrogate log-likelihood 必须不超过用户 held-out review distribution 的 95th percentile

### §2.2 Clustering
- K ∈ {2,…,8} selected by **BIC + AIC + silhouette score**
- 最终每 domain 8 clusters

## §B 代码问题

### §B.1 Stage 10 loss 实现完整性 ✅

对照论文 Eq.1-8 检查 `train_vades_lite_sentence_latent_threshold.py`：

| 论文 term | 代码实现 | 状态 |
|----------|---------|------|
| Feature reconstruction (Eq.4) | `style_recon_loss = F.mse_loss(reconstruction, batch_features)` | ✅ |
| Sentence-user alignment (Eq.6) | `kl_matrix_tensor = gmm_log_likelihood(...)` → cross_entropy | ✅ |
| User-match cross-entropy (Eq.7) | `user_match_loss = F.cross_entropy(kl_matrix_tensor, batch_user_indices)` | ✅ |
| Sentence-level KL to N(0,I) | `sent_kl_loss = standard_normal_kl(...)` | ✅ |
| User-level KL to N(0,I) | `user_prior_kl_loss` | ✅ |
| Per-user mixture (w, μ, σ²) | GMM component params via `user_table` | ✅ |
| Per-user 2-component mixture | `GMM_COMPONENTS=2` (default) | ✅ |
| BIC+AIC+silhouette for K | `cluster_strict5550.py` BIC+silhouette（缺 AIC） | ⚠️ 部分 |
| **95th percentile filtering** | **NOT FOUND** | ❌ 缺失 |

### §B.2 缺失：95th Percentile Style Filtering

论文 Section 2.2 描述：
> "the negative surrogate log-likelihood of the candidate query under the target user's mixture must not exceed the 95th percentile of the user's held-out review distribution"

这意味着在生成 candidate queries 后，需要用 GMM 计算每个 candidate 在用户 distribution 下的 log-likelihood，拒绝超过 95th percentile 的 sample。

**可能实现位置**：Stage 05（inject noisy query 生成阶段）或 Stage 10（query filtering）

## §C 本轮已实施改动

无代码改动（本轮为论文分析）

## §D 验证

N/A

## §E Git Commit

N/A（分析轮次）

## §F 下轮建议

- 在 Stage 05 或 Stage 10 找到 95th percentile filtering 的实现位置，确认是否已实现
- 检查 Stage 05 `apply_lambdamart_userbased_noisy.py` 或相关 generate 脚本
- 检查 Stage 10 `cluster_strict5550.py` 是否只做了 clustering 而非 filtering
