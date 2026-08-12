# Iteration #32 — 论文 §2.2 GMM/VAE 训练目标通读

**日期**: 2026-07-20
**scope**: 论文第2次通读：§2.2 Personalized Query Generation and Expression Difference Quantification
**关联 stage**: 10_complexity_analysis

## §A 论文发现

### §2.2 核心方法：GMM/VAE 用户表达风格建模

**输入**：用户历史 Amazon 评价句子（需 ≥15 words，需 ≥20 reviews/user）
**输出**：用户级别多元高斯混合参数 {w_u,k, μ_u,k, σ²_u,k} (k=1,2)

#### 句法风格特征（20维）：
- Syntactic depth（句法深度）
- Clause structure（从句结构）
- Dependency distance（依存距离）
- Modifier density（修饰语密度）
- Coordination structure（并列结构）

#### 模型结构：
- **Encoder**：将标准化句法特征 x_i ∈ ℝ²⁰ 映射到潜在空间的对角多元高斯分布 q_φ(z|x_i) = N(μ_i, diag(σ²_i))
- **Decoder**：从 encoder mean μ_i 直接重建原始句法特征
- **User-level 2-component GMM**：每个用户维护两个对角高斯组件 p_ψ(z_u,k) = N(μ_u,k, σ²_u,k)，权重 w_u,1 + w_u,2 = 1

#### 4类 Loss 项（Eq.1-8）：

| Loss 项 | 论文代号 | 代码实现 | 状态 |
|---------|---------|---------|------|
| Feature reconstruction | Eq.4 | `style_recon_loss = F.mse_loss(reconstruction, batch_features)` | ✅ |
| Sentence-user alignment | Eq.6 KL surrogate (Jensen log-sum-exp) | `user_match_loss = F.cross_entropy(kl_matrix, batch_user_indices)` | ⚠️ 待核实 |
| User-match cross-entropy | Eq.7 | 同上 `user_match_loss` | ✅ |
| Regularization (sentence-level) | Eq.8 sentence term | `sent_kl_loss = standard_normal_kl(sent_mu, sent_logvar).mean()` | ✅ |
| Regularization (user-level) | Eq.8 user term (per-component closed-form) | `user_prior_kl_loss = (mix_probs * per_component_kl).sum(dim=-1).mean()` | ✅ |

#### 关键约束：
- 20维句法特征需从 spaCy dependency parser 提取
- 用户需保留 ≥20 历史 reviews，每条 ≥15 words
- 2-component mixture per user（短句attribute listing vs. clause extension）

## §B 代码问题

- [10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:~1230] `user_match_loss` 使用 `F.cross_entropy(kl_matrix_tensor, batch_user_indices)` — 是否实现了论文 Eq.6 的 Jensen-style log-sum-exp surrogate？需核对 kl_matrix 的构建逻辑
- [10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py:~1238] `latent_align_loss = F.mse_loss(sent_mu, batch_features)` — 代码有额外的 latent align loss，论文 Eq.4 未明确定义此项，可能为代码自有增强

## §C 本轮已实施改动

无代码改动，仅论文研读。

## §D 验证

无。

## §E Git Commit

无（新发现待下轮核实）。

## §F 下轮建议

- 核对 Stage 10 `user_match_loss` 的 kl_matrix 构建逻辑是否匹配论文 Eq.6 Jensen-style surrogate
- 检查 `latent_align_loss` 是否有论文依据（Eq.4 以外）
- Stage 05 syntactic depth extraction 代码检查（spaCy 20维特征提取）
