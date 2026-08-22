#!/usr/bin/env python3
"""VADES prototype v5: 直接用 raw_user_proto 作 InfoNCE 正样本 + 教师信号.

v4 失败诊断 (epoch_details):
  - L_teacher 2.02 → 0.40 (teacher_proj 学到了 user-specific 输出)
  - L_user 0.0007 → 0.01 (mu_q ≈ p_u, 但 p_u 是 learned collapse)
  - L_contrastive 卡在 1.79 (batch=32 随机 baseline=ln(32)=3.47, 有进展但不再下降)
  - encoder probe = 0.02% (随机, 确认 encoder 没学 user 信号)

根因: teacher_proj 2-layer MLP + ReLU 有 ~840 参数, 足够吸收 teacher loss 梯度
      → encoder 没收到 user-discriminative 信号
      → learned p_u collapse 到同一点 (所有用户共享)
      → L_user ≈ 0 是虚假对齐, 不是真学到了

v5 修复:
  1. 移除 teacher_proj, 改用直接 cosine L_teacher = 1 - cos_sim(mu_q, normalize(raw_target))
  2. InfoNCE 正样本改为 raw_user_proto (数据驱动, 用户特异, 不会 collapse)
  3. 保留 learned p_u 仅供 user_table forward (动态 GMM)
  4. 两阶段训练: Stage 1 (前 20 ep) 重 teacher + contrast; Stage 2 (后 10 ep) 加 recon + KL
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

# === v5 配置 ===
os.environ["VADES_OUTPUT_TAG"] = "vades_prototype_3000u_v5"
os.environ["VADES_PROTOTYPE_VARIANT"] = "v5"  # 关键: 启用 v5 (raw 直接 anchor)
os.environ["VADES_COVARIANCE_MODE"] = "diagonal_prototype"
os.environ["VADES_PROTOTYPE_K"] = "8"
os.environ["VADES_SUPPORT_FRAC"] = "0.5"
os.environ["VADES_SENTENCES_PER_USER"] = "4"
os.environ["VADES_PROTOTYPE_USER_CHUNK"] = "32"
os.environ["VADES_LAMBDA_TEACHER"] = "10.0"
os.environ["VADES_LAMBDA_TEACHER_STAGE2"] = "1.0"
os.environ["VADES_LAMBDA_USER"] = "5.0"
os.environ["VADES_LAMBDA_CONTRAST"] = "1.0"
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
