#!/usr/bin/env python3
"""VADES prototype v6: aggregation-first — teacher/contrastive 在用户级 z_u 上.

v5 失败根因 (用户诊断):
  - L_teacher=0.006 在 mu_q (单句) 上, 但 mu_q 单句噪声大
  - raw_user_proto 是多句均值, encoder mu_q 是单句 — 训练目标/评估对象在不同空间
  - 负 gap -0.46 说明 user_mu 在错误方向

v6 修复:
  1. support (5 句/user) → 聚合 → z_u (user-level stable prototype)
  2. L_teacher 在 z_u 上 vs raw_user_proto (用户级对齐, 消除单句噪声)
  3. L_contrastive 在 batch 内 z_u 上 (用户级互证)
  4. 每 query sentence → L_user (接近 z_u)
  5. user_table(z_u, cluster) → mu_u (GMM 派生)
  6. 评估: 用户级 z_u 聚合后的 AUC 必须 ≥ 0.65 (raw baseline)
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

# === v6 配置 ===
os.environ["VADES_OUTPUT_TAG"] = "vades_prototype_3000u_v6"
os.environ["VADES_PROTOTYPE_VARIANT"] = "v6"
os.environ["VADES_COVARIANCE_MODE"] = "diagonal_prototype"
os.environ["VADES_PROTOTYPE_K"] = "8"
os.environ["VADES_SUPPORT_FRAC"] = "0.5"  # 10 句/user → 5 support + 5 query
os.environ["VADES_SENTENCES_PER_USER"] = "10"  # 10 句聚合 (vs v5 的 4 句)
os.environ["VADES_PROTOTYPE_USER_CHUNK"] = "32"
os.environ["VADES_LAMBDA_TEACHER"] = "10.0"
os.environ["VADES_LAMBDA_TEACHER_STAGE2"] = "1.0"
os.environ["VADES_LAMBDA_USER"] = "1.0"  # 调低 (L_teacher + L_contrastive 主导)
os.environ["VADES_LAMBDA_CONTRAST"] = "5.0"  # 调高 (用户级 contrast 是关键)
os.environ["VADES_LAMBDA_GMM"] = "0.01"
os.environ["VADES_CONTRAST_TEMP"] = "0.1"
os.environ["VADES_STAGE1_EPOCHS"] = "20"
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
print(f"VARIANT: {gv.PROTOTYPE_VARIANT}")
print(f"USER_CHUNK: {gv.PROTOTYPE_USER_CHUNK}, SENTENCES_PER_USER: {gv.SENTENCES_PER_USER}")
print(f"LAMBDA_TEACHER: {gv.LAMBDA_TEACHER} / stage2: {gv.LAMBDA_TEACHER_STAGE2}")
print(f"LAMBDA_USER: {gv.LAMBDA_USER}, LAMBDA_CONTRAST: {gv.LAMBDA_CONTRAST}")
print(f"STAGE1_EPOCHS: {gv.STAGE1_EPOCHS}, total EPOCHS: 30")

gv.main_train()
