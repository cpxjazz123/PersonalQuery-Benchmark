#!/usr/bin/env python3
"""Phase 10.3: 从 phase10_candidates_post_60.jsonl 构造 chosen/rejected 偏好训练数据.

流程:
  1. 加载 candidates (3 组: A_no_style, B_random_style, C_target_style)
  2. 对每个 (user, asin):
     - 强制过滤: attr_pass=True (5 attrs 全在 query 中)
     - 选 chosen: C_target_style 中 maha_target 最小的
     - 选 rejected: B_random_style 中 maha_target 最大的
       (或 fallback: 任何 C/B/A 中 maha_target 最大的)
  3. 输出 preference_pairs_60.jsonl: 每行
     {user_id, asin, attrs, mu_u, log_sigma_u, prompt, chosen_query, rejected_query, maha_chosen, maha_rejected}
  4. 输出 preference_meta_60.json: 汇总
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
IN_CANDIDATES = OUT_DIR / "phase10_candidates_post_60.jsonl"
IN_PAIRS = OUT_DIR / "phase10_pairs_60.jsonl"
OUT_PREF = OUT_DIR / "preference_pairs_60.jsonl"
OUT_META = OUT_DIR / "preference_meta_60.json"

sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402


def load_user_profiles() -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)

    # 训练集标准化参数
    feat_rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            feat_rows.append([float(r["features"][k]) for k in list(r["features"].keys())])
    feat_array = np.array(feat_rows, dtype=np.float32)
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9
    return user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std


def maha_one(query_norm: np.ndarray, user_mu_n: np.ndarray, user_logvar: np.ndarray) -> float:
    diff = query_norm - user_mu_n
    return float(((diff ** 2) * np.exp(-user_logvar)).sum())


def main():
    log = lambda m: print(f"[build-pref] {m}", flush=True)
    log("=" * 70)
    log(f"Phase 10.3: 构造 chosen/rejected 偏好训练数据")
    log("=" * 70)

    # === 1. 加载 user profiles + 标准化参数 ===
    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std = load_user_profiles()
    user_mu_norm = (user_mu - feat_mean) / feat_std
    log(f"  users: {len(user_id_to_idx)}")

    # === 2. 加载 pairs (prompt 模板) ===
    pairs = []
    with IN_PAIRS.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    pair_by_uid_asin = {(p["user_id"], p["asin"]): p for p in pairs}
    log(f"  pairs: {len(pairs)}")

    # === 3. 加载 candidates, 按 (user, asin, group) 分组 ===
    log(f"读取 {IN_CANDIDATES}")
    grouped: Dict[Tuple[str, str], Dict[str, list]] = {}
    n_total = 0
    n_attr_pass = 0
    if IN_CANDIDATES.exists():
        with IN_CANDIDATES.open() as f:
            for line in f:
                r = json.loads(line)
                key = (r["user_id"], r["asin"])
                if key not in grouped:
                    grouped[key] = {"A_no_style": [], "B_random_style": [], "C_target_style": []}
                grouped[key].setdefault(r["group"], []).append(r)
                n_total += 1
                if r.get("attr_pass"):
                    n_attr_pass += 1
        log(f"  total candidates: {n_total}, attr_pass: {n_attr_pass} ({100*n_attr_pass/max(n_total,1):.1f}%)")
    else:
        log(f"  [WARN] {IN_CANDIDATES} 不存在, 请先运行 phase10_generate.py")
        return

    # === 4. feature names (标准化) ===
    with SENTENCE_FILE.open() as f:
        first = json.loads(f.readline())
    feature_names = list(first["features"].keys())

    # === 5. 构造 chosen/rejected ===
    pref_pairs = []
    n_pairs_with_data = 0
    n_skipped_no_target = 0
    n_skipped_no_random = 0
    for key, groups in grouped.items():
        user_id, asin = key
        target_idx = user_id_to_idx.get(user_id, -1)
        if target_idx < 0:
            continue
        pair_meta = pair_by_uid_asin.get(key, {})

        # 过滤 attr_pass=True
        target_cands = [c for c in groups["C_target_style"] if c.get("attr_pass")]
        random_cands = [c for c in groups["B_random_style"] if c.get("attr_pass")]
        if not target_cands:
            n_skipped_no_target += 1
            continue
        if not random_cands:
            n_skipped_no_random += 1
            continue
        n_pairs_with_data += 1

        # 计算每个候选的 maha_target
        for cs in target_cands + random_cands:
            try:
                feat_dict = extract_clause_features(cs["candidate_query"])
                feat_raw = np.array([float(feat_dict[n]) for n in feature_names], dtype=np.float32)
                feat_n = (feat_raw - feat_mean) / feat_std
                cs["maha_target"] = maha_one(feat_n, user_mu_norm[target_idx], user_logvar[target_idx])
            except Exception:
                cs["maha_target"] = float("inf")

        # chosen = target_cands 中 maha 最小
        chosen = min(target_cands, key=lambda c: c["maha_target"])
        # rejected = random_cands 中 maha 最大
        rejected = max(random_cands, key=lambda c: c["maha_target"])

        pref_pairs.append({
            "user_id": user_id,
            "asin": asin,
            "attrs": pair_meta.get("attrs", {}),
            "prompt_no_style": pair_meta.get("prompts", {}).get("A_no_style", ""),
            "prompt_target_style": pair_meta.get("prompts", {}).get("C_target_style", ""),
            "mu_u": user_mu[target_idx].tolist(),
            "log_sigma_u": user_logvar[target_idx].tolist(),
            "chosen_query": chosen["candidate_query"],
            "rejected_query": rejected["candidate_query"],
            "maha_chosen": chosen["maha_target"],
            "maha_rejected": rejected["maha_target"],
            "group_chosen": "C_target_style",
            "group_rejected": "B_random_style",
        })

    # === 6. 输出 ===
    OUT_PREF.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PREF.open("w") as f:
        for r in pref_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"已写入 {OUT_PREF} ({len(pref_pairs)} 对)")

    meta = {
        "n_total_candidates": n_total,
        "n_attr_pass": n_attr_pass,
        "attr_pass_rate": n_attr_pass / max(n_total, 1),
        "n_pairs_with_data": n_pairs_with_data,
        "n_skipped_no_target": n_skipped_no_target,
        "n_skipped_no_random": n_skipped_no_random,
        "n_preference_pairs": len(pref_pairs),
        "groups": ["A_no_style", "B_random_style", "C_target_style"],
    }
    if pref_pairs:
        maha_chosen = [p["maha_chosen"] for p in pref_pairs]
        maha_rejected = [p["maha_rejected"] for p in pref_pairs]
        meta["maha_chosen_mean"] = float(np.mean(maha_chosen))
        meta["maha_rejected_mean"] = float(np.mean(maha_rejected))
        meta["maha_diff_mean"] = float(np.mean(np.array(maha_rejected) - np.array(maha_chosen)))
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写入 {OUT_META}")

    log("=" * 70)
    log(f"汇总: preference_pairs={len(pref_pairs)} (skipped_no_target={n_skipped_no_target}, no_random={n_skipped_no_random})")
    log("=" * 70)


if __name__ == "__main__":
    main()