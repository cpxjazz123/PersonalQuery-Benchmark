#!/usr/bin/env python3
"""Phase 10.9.5s: Build 1000 (user, asin) test pairs for n=1000 scale-free eval.

Design:
  - N_pairs = 1000 (从 2918 v6 users 中选, 跳过原 dev 60 + counterfactual 94)
  - 每对: (user_id, asin, attrs, random_user_id, target/random style desc, prompts)
  - 复用 build_phase10_pairs_60 的 attribute/style/prompt 构造逻辑

输出:
  - phase10_pairs_1000.jsonl
  - phase10_pairs_1000_meta.json
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
CORE_FIELDS = ["Brand", "Color", "Material"]
N_PAIRS = 1000
SEED = 42

# Skip these (already in dev/eval)
EXISTING_DEV_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"
EXISTING_PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"


def log(m):
    print(f"[phase10-9.5s-build1000] {m}", flush=True)


def load_user_profiles() -> Tuple[Dict[str, int], "list", "list"]:
    import numpy as np
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    return user_id_to_idx, user_mu, user_logvar


def find_users_with_clean_products() -> Tuple[Dict[str, List[Tuple[str, dict]]], List[str], Dict[str, dict]]:
    """Stage1 reviews → asin list per user; meta scan → attrs."""
    import numpy as np
    meta_path = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"

    stage1_file = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews_3000u.json"
    user_to_asins: Dict[str, set] = {}
    with stage1_file.open() as f:
        data = json.load(f)
    for item in data:
        uid = item["user_id"]
        asin = item.get("asin", "")
        if asin:
            user_to_asins.setdefault(uid, set()).add(asin)
    log(f"  stage1 users: {len(user_to_asins)}")

    log("  扫描 meta 找有完整 5 attrs 的 asin ...")
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
    log(f"  meta scanned: {n_scanned}, attrs cached: {len(attrs_by_asin)}")

    user_pairs: Dict[str, List[Tuple[str, dict]]] = {}
    for uid, asins in user_to_asins.items():
        for a in asins:
            if a in attrs_by_asin:
                user_pairs.setdefault(uid, []).append((a, attrs_by_asin[a]))
    valid_users = [u for u, ps in user_pairs.items() if len(ps) >= 1]
    log(f"  valid users (有 ≥1 clean product): {len(valid_users)}")
    return user_pairs, valid_users, attrs_by_asin


def format_style_desc(mu, log_sigma) -> str:
    mu_str = ", ".join(f"{v:.2f}" for v in mu)
    log_sigma_str = ", ".join(f"{v:.2f}" for v in log_sigma)
    return (
        f"Your query should match this user's stylistic profile (20-d syntactic mean: "
        f"[{mu_str}], log std: [{log_sigma_str}]). "
        f"Emulate this user's typical syntactic complexity and clause structure."
    )


def build_prompts(attrs: dict, target_style_desc: str, random_style_desc: str) -> dict:
    attr_text = "\n".join(f"{k}: {v}" for k, v in attrs.items())
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
    log("=" * 70)
    log(f"Phase 10.9.5s: Build {N_PAIRS} (user, asin) pairs (scale-up n=60→1000)")
    log("=" * 70)

    user_id_to_idx, user_mu, user_logvar = load_user_profiles()
    log(f"  total users in profiles: {len(user_id_to_idx)}")

    user_pairs, valid_users, attrs_by_asin = find_users_with_clean_products()

    # 跳过已有 dev/eval 用户
    skip_users = set()
    if EXISTING_DEV_FILE.exists():
        skip_users |= set(json.loads(EXISTING_DEV_FILE.read_text())["user_ids"])
        log(f"  skip dev 60 users: {len(skip_users)}")
    if EXISTING_PAIRS_FILE.exists():
        with EXISTING_PAIRS_FILE.open() as f:
            for line in f:
                skip_users.add(json.loads(line)["user_id"])
        log(f"  skip + pairs 60 users: total {len(skip_users)}")
    # cf pairs
    cf_file = OUT_DIR / "phase10_9_counterfactual_pairs.jsonl"
    if cf_file.exists():
        with cf_file.open() as f:
            for line in f:
                r = json.loads(line)
                skip_users.add(r.get("user_u", ""))
                skip_users.add(r.get("user_v", ""))
        log(f"  skip + cf users: total {len(skip_users)}")

    valid_users_filtered = [u for u in valid_users if u not in skip_users]
    log(f"  valid users (after skip): {len(valid_users_filtered)}")

    rng = random.Random(SEED)
    # 按可用 product 数排序, 选 product 数最多的 N_PAIRS 用户
    valid_users_sorted = sorted(valid_users_filtered, key=lambda u: -len(user_pairs[u]))
    if len(valid_users_sorted) < N_PAIRS:
        log(f"  WARN: only {len(valid_users_sorted)} valid users, less than N_PAIRS={N_PAIRS}")

    selected_users = valid_users_sorted[:N_PAIRS]
    log(f"  selected {len(selected_users)} users for n=1000")

    test_pairs = []
    rng_other = random.Random(123)
    all_uids = list(user_id_to_idx.keys())
    for uid in selected_users:
        ps = user_pairs[uid]
        # 随机选 1 个 product (不同 seed)
        chosen_asin, chosen_attrs = rng.choice(ps)
        target_idx = user_id_to_idx[uid]

        # 随机用户 (不同 target; 也跳过自己)
        for _ in range(20):
            random_uid = rng.choice(all_uids)
            if random_uid != uid:
                break
        else:
            random_uid = all_uids[(target_idx + 1) % len(all_uids)]
        random_idx = user_id_to_idx[random_uid]

        target_style_desc = format_style_desc(user_mu[target_idx], user_logvar[target_idx])
        random_style_desc = format_style_desc(user_mu[random_idx], user_logvar[random_idx])
        prompts = build_prompts(chosen_attrs, target_style_desc, random_style_desc)

        # 验证 Brand/Color/Material 至少 2 个有值
        core_ok = sum(1 for k in CORE_FIELDS if chosen_attrs.get(k)) >= 2
        if not core_ok:
            continue

        test_pairs.append({
            "user_id": uid,
            "asin": chosen_asin,
            "attrs": chosen_attrs,
            "random_user_id": random_uid,
            "target_style_desc": target_style_desc,
            "random_style_desc": random_style_desc,
            "prompts": prompts,
        })

    out_path = OUT_DIR / "phase10_pairs_1000.jsonl"
    with out_path.open("w") as f:
        for r in test_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  已写入 {out_path} ({len(test_pairs)} pairs)")

    summary = {
        "n_pairs": len(test_pairs),
        "n_candidates_per_group": 5,
        "total_candidates_per_cond": len(test_pairs) * 5,
        "conditions": ["D_target_style", "C_random_style"],
        "skip_users": len(skip_users),
        "core_fields": CORE_FIELDS,
        "all_fields": ATTR_FIELDS,
        "seed": SEED,
        "note": "n=1000 scale-up from n=60 dev. CI half-width should shrink ~4x (16x more samples).",
    }
    summary_path = OUT_DIR / "phase10_pairs_1000_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"  已写入 {summary_path}")

    log("=" * 70)
    log(f"Phase 10.9.5s: {len(test_pairs)} pairs ready. 下一步: 跑 phase10_9_4i2 on this.")
    log("=" * 70)


if __name__ == "__main__":
    main()