#!/usr/bin/env python3
"""VADES prototype v4: 用户分组采样 + teacher loss (raw 47d) + 两阶段训练.

关键修复 (vs v3):
  1. teacher loss L_teacher = MSE(projector(mu_q), r_u) — 用 raw features (AUC=0.65) 作老师,
     直接绕开 style_centers 对 encoder 的同质化压力.
  2. 两阶段训练: Stage 1 (前 20 epochs) 关闭 recon + KL, 只用 teacher + contrastive;
     Stage 2 (后 10 epochs) 加入全 VADES losses.
  3. 用户分组采样 U=32 users × K=4 sentences — 每用户固定 2 support + 2 query.

评估目标 (用户的 7 条):
  1. 每 batch 同用户样本数 >= 2  ✓
  2. 每 batch 正样本数 > 0     ✓ (每 query 都有 own-user p_u)
  3. L_contrastive 正样本 sim > 负样本 sim
  4. encoder holdout AUC > 0.65 或 gap CI 完全 > 0
  5. permutation p < 0.05
  6. target_user dist < wrong_user dist
  7. 生成 query 目标用户 style 分数 > 随机用户/错误用户
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

# === v4 配置 ===
os.environ["VADES_OUTPUT_TAG"] = "vades_prototype_3000u_v4"
os.environ["VADES_COVARIANCE_MODE"] = "diagonal_prototype"
os.environ["VADES_PROTOTYPE_K"] = "8"
os.environ["VADES_SUPPORT_FRAC"] = "0.5"  # 每用户 4 句 → 2 support + 2 query
os.environ["VADES_SENTENCES_PER_USER"] = "4"  # v4: 每用户限制 K=4 句
os.environ["VADES_PROTOTYPE_USER_CHUNK"] = "32"  # v4: U=32 users/batch
os.environ["VADES_LAMBDA_TEACHER"] = "10.0"  # Stage 1 teacher 重
os.environ["VADES_LAMBDA_TEACHER_STAGE2"] = "1.0"  # Stage 2 teacher 降
os.environ["VADES_LAMBDA_USER"] = "5.0"
os.environ["VADES_LAMBDA_CONTRAST"] = "1.0"
os.environ["VADES_LAMBDA_GMM"] = "0.01"
os.environ["VADES_CONTRAST_TEMP"] = "0.1"
os.environ["VADES_PROTOTYPE_VAR_BIAS"] = "0.0"
os.environ["VADES_STAGE1_EPOCHS"] = "20"  # 前 20 epochs 只 teacher + contrast
os.environ["VADES_STAGE1_LAMBDA_RECON"] = "0.0"
os.environ["VADES_STAGE1_LAMBDA_KL"] = "0.0"
os.environ["VADES_EPOCHS"] = "30"
os.environ["VADES_MAX_USERS"] = "3000"
os.environ["VADES_SKIP_POST_CLUSTERING"] = "1"
os.environ["VADES_EARLY_STOP"] = "10"
os.environ["VADES_REVIEW_SOURCE"] = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"
os.environ["VADES_SKIP_DEDUP"] = "1"

import gaussian_vades as gv

print(f"OUTPUT_TAG: {gv.OUTPUT_TAG}")
print(f"COVARIANCE_MODE: {gv.COVARIANCE_MODE}")
print(f"PROTOTYPE_NUM_CLUSTERS: {gv.PROTOTYPE_NUM_CLUSTERS}")
print(f"USER_CHUNK: {gv.PROTOTYPE_USER_CHUNK}")
print(f"SENTENCES_PER_USER: {gv.SENTENCES_PER_USER}")
print(f"LAMBDA_TEACHER: {gv.LAMBDA_TEACHER} / stage2: {gv.LAMBDA_TEACHER_STAGE2}")
print(f"LAMBDA_USER: {gv.LAMBDA_USER}, LAMBDA_CONTRAST: {gv.LAMBDA_CONTRAST}")
print(f"STAGE1_EPOCHS: {gv.STAGE1_EPOCHS}, total EPOCHS: 30")

gv.main_train()
