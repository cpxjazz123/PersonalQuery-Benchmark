#!/usr/bin/env python3
"""Phase 10.9.1: 构造 Profile-Preserving Counterfactual 训练对.

核心设计 (用户指定):
  对同一 (q_A, q_B) 配对,构造 2 个 DPO sample:
    Sample 1: profile_u → chosen=q_A, rejected=q_B   (推动 M_A > 0)
    Sample 2: profile_v → chosen=q_B, rejected=q_A   (推动 M_B > 0)

  这样 model 必须读取 profile 才能同时满足两个偏好;
  只学"qA 总体比 qB 好"无法完成, 必须学"profile决定偏好"。

数据来源:
  - q_A 来自 user_u 的 chosen_query (preference_pairs_60.jsonl, maha_target_u 最低)
  - q_B 来自 user_v 的 chosen_query (preference_pairs_60.jsonl, maha_target_v 最低)
  - profile_u = [mu_u, log_sigma_u]
  - profile_v = [mu_v, log_sigma_v]

  q_A / q_B 内容可不同 (不同商品/属性), 关键是 DPO 必须依赖 profile。

配对策略:
  - 30 dev user × 2 random partner = 60 反事实对 = 120 DPO sample
  - 每个 sample 包含 (chosen + rejected) 2 文本
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "preference_pairs_60.jsonl"
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"
OUT_PAIRS = OUT_DIR / "phase10_9_counterfactual_pairs.jsonl"
OUT_META = OUT_DIR / "phase10_9_counterfactual_pairs_meta.json"

# === 硬编码 ===
SEED = 42
N_PARTNERS_PER_USER = 2  # 每个 user 配 2 个 partner → 60*2 = 120 counterfactual pairs
MIN_MARGIN = 50.0        # 选 margin >= 50 的 pair (确保 chosen/rejected 明显区分)

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"


def log(m):
    print(f"[phase10-9.1] {m}", flush=True)


def main():
    log("=" * 70)
    log("Phase 10.9.1: 构造 Profile-Preserving Counterfactual Pairs")
    log("=" * 70)
    random.seed(SEED)

    # === 1. 加载 dev 用户 ===
    dev_ids = sorted(json.loads(DEV_USER_FILE.read_text())["user_ids"])
    log(f"[1] dev users = {len(dev_ids)}")

    # === 2. 加载 user profiles (拿 mu/log_sigma) ===
    log("[2] Loading user profiles ...")
    user_id_to_idx = {}
    user_mu_arr = []
    user_logvar_arr = []
    with USER_PROFILE_FILE.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            user_id_to_idx[row["user_id"]] = idx
            user_mu_arr.append(row["user_mu"])
            user_logvar_arr.append(row["user_logvar"])
    user_mu = {row["user_id"]: row["user_mu"] for row in [
        json.loads(line) for line in USER_PROFILE_FILE.read_text().splitlines()
    ]}
    user_logvar = {row["user_id"]: row["user_logvar"] for row in [
        json.loads(line) for line in USER_PROFILE_FILE.read_text().splitlines()
    ]}
    # 只取交集 (确保 dev 用户都在 profiles 里)
    dev_in_profile = [u for u in dev_ids if u in user_id_to_idx]
    log(f"  dev ∩ profile = {len(dev_in_profile)}")
    assert len(dev_in_profile) == len(dev_ids), "有 dev user 不在 profile 里"

    # === 3. 加载 preference_pairs, 提取 chosen_query + margin ===
    log("[3] Loading preference pairs ...")
    records = []
    with PAIRS_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            margin = d.get("maha_rejected", 0) - d.get("maha_chosen", 0)
            if margin < MIN_MARGIN:
                continue
            if d["user_id"] not in dev_in_profile:
                continue
            records.append(d)
    log(f"  filtered to {len(records)} records (margin >= {MIN_MARGIN})")

    # 按 user 索引, 选 margin 最大的一条 (最显著的 chosen)
    best_per_user: Dict[str, dict] = {}
    for r in records:
        uid = r["user_id"]
        margin = r["maha_rejected"] - r["maha_chosen"]
        if uid not in best_per_user or margin > (
            best_per_user[uid]["maha_rejected"] - best_per_user[uid]["maha_chosen"]
        ):
            best_per_user[uid] = r
    log(f"  best_per_user: {len(best_per_user)}/{len(dev_in_profile)}")

    # === 4. 给每个 user 配 N_PARTNERS partners ===
    log(f"[4] Pairing each user with {N_PARTNERS_PER_USER} partners ...")
    counterfactual_pairs = []
    rng = random.Random(SEED)
    # 只配 best_per_user 里有的用户
    eligible_users = list(best_per_user.keys())
    log(f"  eligible users (have best_per_user) = {len(eligible_users)}")
    for u in eligible_users:
        u_record = best_per_user[u]
        candidates = [v for v in eligible_users if v != u]
        partners = rng.sample(candidates, N_PARTNERS_PER_USER)
        for v in partners:
            v_record = best_per_user[v]
            # 构造 1 个反事实对 = 2 个 DPO sample
            # Sample A: profile_u → chosen=q_u_chosen, rejected=q_v_chosen
            # Sample B: profile_v → chosen=q_v_chosen, rejected=q_u_chosen
            pair = {
                "user_u": u,
                "user_v": v,
                "asin_u": u_record["asin"],
                "asin_v": v_record["asin"],
                "attrs_u": u_record["attrs"],
                "attrs_v": v_record["attrs"],
                "prompt_u": u_record["prompt_target_style"],
                "prompt_v": v_record["prompt_target_style"],
                "chosen_q_u": u_record["chosen_query"],  # user_u 的 chosen (maha_target_u 最低)
                "chosen_q_v": v_record["chosen_query"],  # user_v 的 chosen (maha_target_v 最低)
                "maha_chosen_u": u_record["maha_chosen"],
                "maha_chosen_v": v_record["maha_chosen"],
                "margin_u": u_record["maha_rejected"] - u_record["maha_chosen"],
                "margin_v": v_record["maha_rejected"] - v_record["maha_chosen"],
                # z_u = [mu_u, log_sigma_u] (40d)
                "z_u": u_record["mu_u"] + u_record["log_sigma_u"],
                "z_v": v_record["mu_u"] + v_record["log_sigma_u"],
            }
            counterfactual_pairs.append(pair)
    log(f"  counterfactual pairs = {len(counterfactual_pairs)}")

    # === 5. Sanity check ===
    log("[5] Sanity check ...")
    log(f"  pair[0].user_u = {counterfactual_pairs[0]['user_u'][:12]}")
    log(f"  pair[0].user_v = {counterfactual_pairs[0]['user_v'][:12]}")
    log(f"  pair[0].chosen_q_u[:50] = {counterfactual_pairs[0]['chosen_q_u'][:50]}...")
    log(f"  pair[0].chosen_q_v[:50] = {counterfactual_pairs[0]['chosen_q_v'][:50]}...")
    log(f"  pair[0].z_u[:5] = {counterfactual_pairs[0]['z_u'][:5]}")

    # === 6. 写文件 ===
    log("[6] Writing ...")
    with OUT_PAIRS.open("w") as f:
        for p in counterfactual_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"  wrote {len(counterfactual_pairs)} pairs → {OUT_PAIRS}")

    # === 7. Meta ===
    meta = {
        "n_counterfactual_pairs": len(counterfactual_pairs),
        "n_partners_per_user": N_PARTNERS_PER_USER,
        "n_dev_users": len(dev_in_profile),
        "min_margin_filter": MIN_MARGIN,
        "seed": SEED,
        "note": "Each pair = 2 DPO samples. Sample A: profile_u → chosen=q_u; Sample B: profile_v → chosen=q_v. Model must read profile.",
        "expected_dpo_loss_signs": {
            "sample_A": "log P(q_u|u) - log P(q_v|u) > 0  (chosen_u > rejected_v under profile_u)",
            "sample_B": "log P(q_v|v) - log P(q_u|v) > 0  (chosen_v > rejected_u under profile_v)",
        },
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  wrote meta → {OUT_META}")
    log("=" * 70)


if __name__ == "__main__":
    main()