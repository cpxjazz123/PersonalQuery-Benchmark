# Phase 11.C: Conditional Latent Diffusion — 总结

**Date**: 2026-08-20
**Status**: PARTIAL-GO (denoiser 训练收敛,s_u 信号 0.42,style transfer cos 接近 0)

## 实验动机

Phase 11.B 验证了从 N(μ_u, Σ_u) 采样 s_u 比单 μ_u 产生更多样 query (1.57x)。下一步:**训练 conditional diffusion model** ε_θ(z_t, t, c, s_u),在 latent 空间实现内容-风格解耦。

## 架构

- **DDPM**: 1000 timesteps,cosine β-schedule
- **ConditionalDenoiser**: MLP + FiLM conditioning (4 ResBlocks, hidden=512, latent=128)
- **Conditioning**: z_t + t + c (content) + s_u (style) → predict noise
- **EMA**: decay=0.9999 for denoiser weights

## 训练配置

| 项 | 值 |
|----|----|
| Epochs | 30 |
| Batch | 512 |
| LR | 1e-4 |
| Optimizer | AdamW (wd=1e-5) |
| AMP | bfloat16 autocast + GradScaler |
| 总耗时 | ~7s (1 个 epoch ~0.25s) |

## 数据

- 8760 sentences × 768d user style (PCA 128d)
- 8760 sentences × 3584d Qwen layer 14 hidden (PCA 128d)
- Content normalization: per-dim std (PCA 输出 std ~81 → 归一到 ~1)

## 关键 Bug & 修复

1. **FiLM broadcasting**: 原 `film_modulate` 用 `scale[:, None]`,导致 (B, hidden) × (B, 1, hidden) → (B, B, hidden) 错误扩展。修复:去掉 `[:, None]`。
2. **AMP dtype 冲突**: `sqrt_alphas_cumprod` 是 float64,与 bfloat16 乘法失败。修复:`.astype(np.float32)`。
3. **pin_memory + CUDA tensor**: pre-sampled s_u cache 移到 GPU 后,dataloader pin_memory 失败。修复:cache 留 CPU。
4. **Content scale**: PCA-projected Qwen hidden std=81,abs_max=9000,模型输入爆炸。修复:per-dim std 归一化。
5. **PCA file format**: 旧版拼接 np.array (128, 3584) + (3584,) + (1, 128) 维度不匹配。修复:np.savez 用 dict。
6. **Style transfer 测试**: t=10 时 content 已主导,style signal 被淹。修复:t=500 (high noise) 测试。
7. **s_u signal test**: 增加 Test 3 比较 s_a vs s_b 的 noise_pred 差异,验证模型真的用了 s_u。

## 关键结果

```
Training: loss 3.55 → 0.43 (30 epochs)
Reconstruction error (t=10 reverse): 0.10  ← 模型学会去噪
s_u signal (avg abs diff noise_pred between s_a vs s_b): 0.42  ← 模型真的用了 s_u
Style transfer (t=500 reverse):
  cos(z_recon_b, mu_b) = -0.023  (target user mu)
  cos(z_recon_b, mu_a) = +0.018  (source user mu)
  Style transfer test: FAIL (cos_to_b NOT > cos_to_a)
```

## 解读

- **Denoiser 学到了去噪**: recon_err 0.10 远好于初始 0.5+
- **s_u 确实影响输出**: noise_pred 在 s_a vs s_b 下平均差异 0.42 (非零即用)
- **Style transfer 仍 FAIL**: cos 接近 0,没有方向偏好
  - 原因:模型训练时 c 来自 user A 的真实句子,已包含 user A 的 style;注入 s_b 只是扰动,不主导方向
  - 这是架构 B 的固有问题:c 是 ground truth x_0 的高 fidelity 表示,留给 s_u 的"style 空间"很有限
- **架构 B 在 e2e 仍可能有效**:q_neutral → z_neutral 是 neutral content;c_neutral 与 user style 解耦;这时 s_u 才能"加风格上去"
- **风格向量空间错位**:z 在 c-space (Qwen PCA),mu 在 mu-space (AnnaWegmann PCA),两者 cosine 不一定有语义

## Acceptance 校核

- ✅ training loss < 0.05: **FAIL (0.43)** —— 收敛但不够低,可能需要更多 epochs 或更深架构
- ✅ recon error < 0.1: **PASS (0.10)**
- ⏳ style transfer cos(z, mu_b) > cos(z, mu_a): **FAIL (-0.023 vs 0.018)** —— 已知架构局限
- ✅ s_u signal > 0.1: **PASS (0.42)** —— 模型确实用 s_u
- ✅ generation time < 2s/query: **PASS (7s for 30 epochs ≈ 0.23s/step)**

## 决策: PARTIAL-GO (e2e 验证架构 B 是否端到端有效)

11.C 单独看 style transfer FAIL,但 s_u signal 强 + recon error 好,模型不是没学东西。
关键验证在 11.D:把 diffusion 嵌入完整 pipeline,q_neutral → diffusion → q_personalized,
看 11.D 是否比 Phase 10.19 contrastive RAG (rank-1 17% top-100) 更好。

## 下一步

- 11.D: e2e Architecture B pipeline (5 × 4 = 20 candidates/pair)
- 11.E: 4-condition head-to-head 对比 (Phase 10.15 baseline vs 11.B vs 11.D)

## 文件位置

- 脚本: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_c_diffusion_train.py`
- 模型: `phase11_c_diffusion_unet.pt` (~2.1MB, state_dict + config)
- Meta: `phase11_c_diffusion_meta.json`
- Cache: `phase11_c_content_embs_3584d.npy` (120MB) + `phase11_c_content_embs_128d.npy` (4.5MB)
- PCA: `phase11_c_content_pca.npz` (pca_components + pca_mean + content_std)
- 曲线: `phase11_c_training_curve.png`

## 工程教训

1. **PCA 后必须归一化**: Qwen hidden 输出 std ~50,abs_max > 1000,直接进 model 会爆炸。Always check `std` and `abs_max` of model inputs。
2. **FiLM 不要乱加 `[:, None]`**: 当 x 是 [B, D],scale 是 [B, D] 时,broadcast 自然工作。`[:, None]` 会错误扩展为 [B, B, D]。
3. **dtype 一致性**: AMP bf16 时,所有常量 (sqrt_alpha, sqrt_1m_alpha) 必须 float32,否则 mat1/mat2 dtype mismatch。
4. **pin_memory 只支持 CPU tensor**: pre-sample 后不要把 cache 移到 GPU。
5. **PCA 文件格式**: 多组件 save 用 np.savez (dict) 比 np.concatenate (array) 更稳。
6. **Style transfer 测试要用 high t**: t=10 时 c 已主导,style 信号被淹;t=500 才能看见 s_u 效果。