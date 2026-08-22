#!/usr/bin/env python3
"""Phase 10.2 数据准备: 30 用户 × 10 商品 (有完整 5 attrs) 的测试对 + 3 组 prompt 模板.

设计:
  - N_users = 30 (从 v6 raw profiles 中随机抽, 避开 pilot 阶段的 3 用户)
  - N_products = 10 (从 amazon meta 找有完整 5 attrs 的 asin)
  - 每个 (user, asin) 对, 3 组 prompt:
      A) No-style: 只含商品 5 attrs
      B) Random-style: A + 随机用户的 raw 20d mu + 20d log_sigma 数值描述
      C) Target-style: A + 目标用户的 raw 20d mu + 20d log_sigma 数值描述
  - 每组 N=5 候选 (用 Qwen 直接生成, 不 rerank)

输出:
  - phase10_pairs.jsonl: 每行 {user_id, asin, attrs, target_style_desc, random_style_desc, prompts: {no_style, random_style, target_style}}
  - phase10_pairs_meta.json: 汇总

走 llm_client.py::QwenLocalClient::call (本地 Qwen2-7B), 不引入 vLLM (候选量小)。
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
N_USERS = 30
N_PRODUCTS = 10
N_CANDIDATES_PER_GROUP = 5

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)


def load_user_profiles() -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    import numpy as np
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    return user_id_to_idx, user_mu, user_logvar


def find_users_with_clean_products() -> Dict[str, List[Tuple[str, dict]]]:
    """从 stage1 reviews 找 (user_id, asin) 对, 然后过滤有 5 attrs 的."""
    import numpy as np
    meta_path = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"

    # 1. 收集所有 user 用过的 asin (从 stage1 reviews)
    stage1_file = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews_3000u.json"
    user_to_asins: Dict[str, set] = {}
    with stage1_file.open() as f:
        data = json.load(f)
    # data 是 list of {user_id, asin, reviews, ...}
    for item in data:
        uid = item["user_id"]
        asin = item.get("asin", "")
        if asin:
            user_to_asins.setdefault(uid, set()).add(asin)
    print(f"  stage1 users: {len(user_to_asins)}")

    # 2. 从 meta 找有完整 5 attrs 的 asin
    print(f"  扫描 meta 找有完整 5 attrs 的 asin ...")
    attrs_by_asin: Dict[str, dict] = {}
    n_scanned = 0
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            n_scanned += 1
            r = json.loads(line)
            pa = r.get("parent_asin", "")
            details = r.get("details", {})
            if not isinstance(details, dict):
                continue
            attrs = {}
            ok = True
            for k in ATTR_FIELDS:
                v = details.get(k, "")
                if not v:
                    ok = False
                    break
                attrs[k] = str(v).strip()[:64]
            if ok:
                attrs_by_asin[pa] = attrs
            if len(attrs_by_asin) >= 50000:
                break
    print(f"  meta scanned: {n_scanned}, attrs cached: {len(attrs_by_asin)}")

    # 3. 找 user-asin 对
    user_pairs: Dict[str, List[Tuple[str, dict]]] = {}
    for uid, asins in user_to_asins.items():
        for a in asins:
            if a in attrs_by_asin:
                user_pairs.setdefault(uid, []).append((a, attrs_by_asin[a]))

    valid_users = [u for u, ps in user_pairs.items() if len(ps) >= 1]
    print(f"  valid users (有 clean product): {len(valid_users)}")
    return user_pairs, valid_users, attrs_by_asin


def format_attrs(attrs: dict) -> str:
    """格式化 attrs 为 prompt 中的 'Brand=X, Item Weight=Y, ...'."""
    return "\n".join(f"{k}: {v}" for k, v in attrs.items())


def format_style_desc(mu: np.ndarray, log_sigma: np.ndarray) -> str:
    """把 user style vector 数值描述成自然语言 (供 LLM 理解)."""
    mu_str = ", ".join(f"{v:.2f}" for v in mu)
    log_sigma_str = ", ".join(f"{v:.2f}" for v in log_sigma)
    return (
        f"Your query should match this user's stylistic profile (20-d syntactic mean: "
        f"[{mu_str}], log std: [{log_sigma_str}]). "
        f"Emulate this user's typical syntactic complexity and clause structure."
    )


def build_prompts(attrs: dict, target_style_desc: str, random_style_desc: str) -> dict:
    """构造 3 组 prompt."""
    attr_text = format_attrs(attrs)
    no_style = (
        f"Product attributes:\n{attr_text}\n\n"
        f"Write a short shopping query that includes every attribute."
    )
    random_style = (
        f"Product attributes:\n{attr_text}\n\n"
        f"{random_style_desc}\n\n"
        f"Write a short shopping query that includes every attribute."
    )
    target_style = (
        f"Product attributes:\n{attr_text}\n\n"
        f"{target_style_desc}\n\n"
        f"Write a short shopping query that includes every attribute."
    )
    return {"A_no_style": no_style, "B_random_style": random_style, "C_target_style": target_style}


def main():
    print("=" * 70)
    print(f"Phase 10.2: 准备 {N_USERS} 用户 × {N_PRODUCTS} 商品 × 3 组 prompt")
    print("=" * 70)

    # === 1. Load user profiles + find users with clean products ===
    user_id_to_idx, user_mu, user_logvar = load_user_profiles()
    print(f"  total users in profiles: {len(user_id_to_idx)}")

    user_pairs, valid_users, attrs_by_asin = find_users_with_clean_products()
    print(f"  users with at least 1 clean product: {len(valid_users)}")

    # === 2. 选取 N_USERS 个测试用户 ===
    import random
    rng = random.Random(42)
    # 按可用 clean product 数排序, 选最多的 N_USERS
    valid_users_sorted = sorted(valid_users, key=lambda u: -len(user_pairs[u]))
    selected_users = valid_users_sorted[:N_USERS]
    print(f"  selected {len(selected_users)} users")

    # === 3. 为每个用户选 1 个 clean product (简单起见, Phase 10 不做多 product) ===
    test_pairs = []
    rng_other = random.Random(123)
    for uid in selected_users:
        pairs = user_pairs[uid]
        chosen_asin, chosen_attrs = pairs[0]  # 取第一个
        target_idx = user_id_to_idx[uid]

        # 随机用户 (跟 target 不同)
        other_idxs = [i for i in range(len(user_id_to_idx)) if i != target_idx]
        random_idx = rng_other.choice(other_idxs)
        random_user_id = list(user_id_to_idx.keys())[random_idx]

        target_style_desc = format_style_desc(user_mu[target_idx], user_logvar[target_idx])
        random_style_desc = format_style_desc(user_mu[random_idx], user_logvar[random_idx])
        prompts = build_prompts(chosen_attrs, target_style_desc, random_style_desc)

        test_pairs.append({
            "user_id": uid,
            "asin": chosen_asin,
            "attrs": chosen_attrs,
            "random_user_id": random_user_id,
            "target_style_desc": target_style_desc,
            "random_style_desc": random_style_desc,
            "prompts": prompts,
        })

    out_path = OUT_DIR / "phase10_pairs.jsonl"
    with out_path.open("w") as f:
        for r in test_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  已写入 {out_path}")

    # === 4. 汇总 ===
    summary = {
        "n_users": len(test_pairs),
        "n_products_per_user": 1,  # Phase 10.2 简化: 每个 user 1 个 product
        "n_candidates_per_group": N_CANDIDATES_PER_GROUP,
        "total_candidates_per_group": len(test_pairs) * N_CANDIDATES_PER_GROUP,
        "groups": ["A_no_style", "B_random_style", "C_target_style"],
    }
    summary_path = OUT_DIR / "phase10_pairs_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  已写入 {summary_path}")
    print("=" * 70)
    print("Phase 10.2 数据准备完成 (未生成候选, 需要 phase10_generate.py 跑 LLM)")
    print("=" * 70)


if __name__ == "__main__":
    main()