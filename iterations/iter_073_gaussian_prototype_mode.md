# Iter 073: Gaussian 用户 Gaussian 分布 → Cluster Prototype 模式

## Context

之前 Gaussian 训练 (`diagonal_logistic` / `diagonal_disentangled`) 每用户独立拟合 N(mu_u, Sigma_u),
缺点是 30 用户 × 768d 的 (mu_u, Sigma_u) 是稀疏、不可迁移、难聚类的。

新加 **`diagonal_prototype`** 模式:
1. **预聚类**: 对 raw user features 做 KMeans(K=8) → user_cluster_ids
2. **Cluster prototype**: 每个 cluster 一个 style anchor(用户风格锚点)
3. **Support/Query 切分**: 每用户 5 support + 5 query(0.5 frac)
4. **用户间对比损失**: InfoNCE 让同 cluster 用户的 (mu_u, logvar_u) 互相靠近,跨 cluster 推远
5. **v4/v5/v6 三变体**:
   - v4: teacher projection(sg(p_u) → 通过 projector 锚定 mu_u)
   - v5: raw anchor 直接用 raw user prototype 当 anchor
   - v6: raw anchor + 双层结构
6. **Stage1/Stage2 两阶段训练**:
   - Stage 1 (20 epochs): 只 train teacher + InfoNCE,关 recon/KL
   - Stage 2 (剩余): teacher loss 权重从 10.0 → 1.0,加回 recon + KL + GMM user_prior

## 改动清单 (gaussian/gaussian_vades.py, +1078/-136)

### 新增 15 个 env-var 控制常量
```python
USER_MATCH_WEIGHT     # =1.0 (cosine match)
STYLE_RECON_WEIGHT    # =0.8
SENT_KL_WEIGHT        # =0.05
USER_PRIOR_KL_WEIGHT  # =0.02
PROTOTYPE_NUM_CLUSTERS # =8
LAMBDA_USER           # =1.0 (L_user 权重)
LAMBDA_CONTRAST       # =0.5 (InfoNCE)
LAMBDA_GMM            # =0.05 (user_prior KL)
SUPPORT_FRAC          # =0.5 (5 support + 5 query)
CONTRAST_TEMPERATURE  # =0.1
PROTOTYPE_USER_CHUNK  # =32
PROTOTYPE_VARIANCE_INIT_BIAS  # logvar 初值
PROTOTYPE_VARIANT     # =v4 (v4=teacher_proj, v5=raw, v6=双层)
LAMBDA_TEACHER        # =10.0 (Stage1)
LAMBDA_TEACHER_STAGE2 # =1.0 (Stage2)
STAGE1_EPOCHS         # =20
STAGE1_LAMBDA_RECON   # =0.0 (Stage1 关重建)
STAGE1_LAMBDA_KL      # =0.0 (Stage1 关 sent KL)
SENTENCES_PER_USER    # =4
```

### 新增函数
- `_build_user_to_indices` / `_sample_support_query_indices`: support/query 切分
- `_compute_losses_for_prototype` (v3 baseline)
- `_compute_losses_for_prototype_v4` / `v5` / `v6`: 三个变体
- `precompute_raw_user_protos` / `precompute_prototype_user_mu`: prototype 缓存
- 新增 `covariance_mode` 选项 `"diagonal_prototype"`

### 训练流程
1. 入口: `python gaussian/gaussian_vades.py train`
2. main_train 检测 `COVARIANCE_MODE="diagonal_prototype"` → 走 prototype 分支
3. 预聚类 + cache support set prototype (per user)
4. 每个 batch: 抽 SENTENCES_PER_USER=4 句 × user, 算 (mu_u, logvar_u) from user table
5. Stage 1: teacher + InfoNCE only, 20 epochs
6. Stage 2: 降 teacher 权重 + 加回 recon/KL/GMM

## Verdict

**GO** (架构层,未跑实验):prototype 模式解决了
1. 用户间 (mu, Sigma) 不可比 → cluster anchor 提供共享参照
2. 训练稀疏 → support/query split + InfoNCE 引入跨用户信号
3. 两阶段让 teacher 先稳,再 fine-tune joint

**待验证**:
- v4 vs v5 vs v6 三变体哪个 lift 最好
- Stage1 epochs=20 是否够,要不要扩到 40
- InfoNCE temperature=0.1 是否合适(可能 0.05 更尖)
- cluster K=8 是否最优(可 sweep K=4/8/16/32)

## 文件清单

| Path | 用途 |
|------|------|
| `gaussian/gaussian_vades.py` | 主入口,新增 `diagonal_prototype` 模式 + 15 env vars + 9 函数 |

## 下一步

1. 跑 prototype mode smoke test (5 user × 5 epoch): `VADES_COV_MODE=diagonal_prototype python gaussian/gaussian_vades.py smoke`
2. 对比 baseline `diagonal_logistic` vs prototype mode 的 user hiddens 重构质量
3. sweep K=4/8/16/32 + temperature=0.05/0.1/0.2
4. 把 prototype 模式下的 user mu 接入 query_gen 做 Phase 17 实验