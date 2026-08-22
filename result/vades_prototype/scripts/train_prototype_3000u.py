#!/usr/bin/env python3
"""Train VADES prototype mode on 3000u data.

Uses output_tag = vades_prototype_3000u, reads same review source as 3000u disentangle.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

# === 配置 (必须在 import gaussian_vades 前) ===
os.environ["VADES_OUTPUT_TAG"] = "vades_prototype_3000u_v3"
os.environ["VADES_COVARIANCE_MODE"] = "diagonal_prototype"
os.environ["VADES_PROTOTYPE_K"] = "8"
os.environ["VADES_SUPPORT_FRAC"] = "0.5"
os.environ["VADES_LAMBDA_USER"] = "5.0"   # 加强 L_user (10x disentangle USER_MATCH_WEIGHT=1)
os.environ["VADES_LAMBDA_CONTRAST"] = "1.0"
os.environ["VADES_LAMBDA_GMM"] = "0.01"  # 几乎不约束 (避免坍缩)
os.environ["VADES_CONTRAST_TEMP"] = "0.1"
os.environ["VADES_PROTOTYPE_USER_CHUNK"] = "128"  # 4x chunk (32→128) 减少 75% Python 迭代
os.environ["VADES_PROTOTYPE_VAR_BIAS"] = "0.0"
os.environ["VADES_EPOCHS"] = "30"  # 已观察到 loss 10.4→5.4 快速下降, 30 epochs 应足够
os.environ["VADES_MAX_USERS"] = "3000"
os.environ["VADES_SKIP_POST_CLUSTERING"] = "1"
os.environ["VADES_EARLY_STOP"] = "10"
os.environ["VADES_REVIEW_SOURCE"] = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"
os.environ["VADES_SKIP_DEDUP"] = "1"  # 已有 3000u dedup profile, 跳过重复

import gaussian_vades as gv

print(f"OUTPUT_TAG: {gv.OUTPUT_TAG}")
print(f"COVARIANCE_MODE: {gv.COVARIANCE_MODE}")
print(f"PROTOTYPE_NUM_CLUSTERS: {gv.PROTOTYPE_NUM_CLUSTERS}")
print(f"SUPPORT_FRAC: {gv.SUPPORT_FRAC}")
print(f"LAMBDA_USER: {gv.LAMBDA_USER}")
print(f"LAMBDA_CONTRAST: {gv.LAMBDA_CONTRAST}")

gv.main_train()
