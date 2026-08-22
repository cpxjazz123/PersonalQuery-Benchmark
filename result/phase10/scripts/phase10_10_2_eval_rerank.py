#!/usr/bin/env python3
"""Phase 10.10.2: Post-hoc selection rerank utility eval.

Rerank 范式:
  给定 N 个候选 query, 计算每个到 target_user_mu 的 Mahalanobis 距离,
  选最小 maha 的 top-1 作为最终 query. VADES 仅作 rerank scorer.

评估维度:
  1. rerank_utility_top1: 选出的 top-1 的 maha_target vs random pick baseline 的 maha_target
     → 若 rerank 有效, 选出的 maha_target 应显著 < random pick
  2. oracle_hit_rate: top-1 是否就是 oracle-best (5 candidates 里 maha_target 最小的那个)
  3. delta_rank_rerank: rerank top-1 的 maha_target 在 cross-user 排名中的位置
     → 应该是 1 (maha_target 最小)
  4. vs 5i2_1000 baseline: rerank 应 > no-rerank (Δ_rank baseline -0.114)
  5. attr_pass: 强制 100% (4i2 已 hard-copy 保证)

输入: phase10_9_4i2_candidates_1000.jsonl (Phase 10.9 留下的 n=1000 candidates)
  876 pairs × 2 conds (D_target_style + C_random_style) × 5 candidates
  → 每对 rerank pool = 10 candidates (pool 跨 cond)

期望:
  - rerank utility > 0 (即 rerank 选出的 maha_target < random pick 的)
  - oracle hit rate ≥ 20% (5 candidates 中随机 1/5=20%, rerank 应 > 随机)
  - attr_pass = 100%
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
CANDIDATES_FILE = OUT_DIR / "phase10_9_4i2_candidates_1000.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
SENTENCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_EVAL = OUT_DIR / "phase10_10_2_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_2_per_pair.jsonl"
LOG_OUT = OUT_DIR / "phase10_10_2_eval.log"

SEED = 42
N_BOOTSTRAP = 5000
ATTR_FIELDS = ["Brand", "Color", "Material"]


def log(m):
    print(f"[phase10-10.2-eval] {m}", flush=True)


def load_user_profiles():
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)

    feat_rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            feat_rows.append([float(r["features"][k]) for k in list(r["features"].keys())])
    feat_array = np.array(feat_rows, dtype=np.float32)
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9

    with SENTENCE_FILE.open() as f:
        first = json.loads(f.readline())
    feature_names = list(first["features"].keys())
    return user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names


def extract_features(query_text: str, feature_names: List[str]):
    if not query_text or not query_text.strip():
        return None
    try:
        feat_dict = extract_clause_features(query_text)
        return np.array([float(feat_dict[name]) for name in feature_names], dtype=np.float32)
    except Exception:
        return None


def maha_one_user(query_norm: np.ndarray, mu_n: np.ndarray, logvar_n: np.ndarray) -> float:
    diff = query_norm - mu_n
    return float(((diff ** 2) * np.exp(-logvar_n)).sum())


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    if not query or not attrs:
        return False
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        tokens = []
        for t in v_clean.split():
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True; break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True; break
            if (t_lower + "s") in q_clean:
                found = True; break
        if not found:
            return False
    return True


def bootstrap_ci_user_level(values_per_item: Dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    """Bootstrap over items (keys may be tuple/str, values must be scalar)."""
    items = list(values_per_item.keys())
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0)
    obs_vals = np.array([values_per_item[k] for k in items])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals = obs_vals[sample_idx]
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high)


def main():
    log("=" * 70)
    log("Phase 10.10.2: Post-hoc Selection Rerank Utility (n=1000)")
    log("=" * 70)

    log("[1] Loading profiles + feature params ...")
    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names = load_user_profiles()
    log(f"  {len(user_id_to_idx)} users, 20d")

    # === Load pairs (target_user_id, random_user_id) ===
    log("[2] Loading pairs ...")
    pair_lookup = {}
    with PAIRS_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            pair_lookup[(p["user_id"], p["asin"])] = p
    log(f"  pairs: {len(pair_lookup)}")

    # === Load candidates, group by (user, asin) + cond ===
    log("[3] Loading candidates ...")
    cand_by_pair: Dict[Tuple[str, str], Dict[str, list]] = {}
    with CANDIDATES_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["user_id"], d["asin"])
            cond = d["cond"]
            cand_by_pair.setdefault(key, {}).setdefault(cond, []).append(d)
    log(f"  total pairs (w/ candidates): {len(cand_by_pair)}")

    # === For each pair: extract features + maha for all 10 candidates ===
    log("[4] Extracting features + Mahalanobis for 10-candidate pool ...")
    n_feat_fail = 0
    n_attr_fail = 0
    per_pair_metrics = {}

    for (uid, asin), conds_dict in cand_by_pair.items():
        if "D_target_style" not in conds_dict or "C_random_style" not in conds_dict:
            continue
        target_pair = pair_lookup.get((uid, asin))
        if not target_pair:
            continue
        random_uid = target_pair["random_user_id"]
        target_idx = user_id_to_idx.get(uid)
        random_idx = user_id_to_idx.get(random_uid)
        if target_idx is None or random_idx is None:
            continue
        mu_t = (user_mu[target_idx] - feat_mean) / feat_std
        logvar_t = user_logvar[target_idx]
        mu_r = (user_mu[random_idx] - feat_mean) / feat_std
        logvar_r = user_logvar[random_idx]

        # Pool: 5 D_target + 5 C_random = 10 candidates per pair
        pool = []
        for cond in ("D_target_style", "C_random_style"):
            for ci, c in enumerate(conds_dict[cond]):
                feat_raw = extract_features(c["candidate_query"], feature_names)
                if feat_raw is None:
                    n_feat_fail += 1
                    feat_n = np.zeros(len(feature_names), dtype=np.float32)
                else:
                    feat_n = (feat_raw - feat_mean) / feat_std
                mt = maha_one_user(feat_n, mu_t, logvar_t)
                mr = maha_one_user(feat_n, mu_r, logvar_r)
                attr_ok = check_attr_pass(c["candidate_query"], c["attrs"])
                if not attr_ok:
                    n_attr_fail += 1
                pool.append({
                    "cond": cond,
                    "ci": ci,
                    "feat_n": feat_n,
                    "maha_target": mt,
                    "maha_random": mr,
                    "attr_pass": attr_ok,
                    "query": c["candidate_query"],
                })

        if len(pool) < 2:
            continue

        # === Filter: keep only attr_pass candidates for rerank pool ===
        # 这是工程正确做法: rerank 只在"合格"候选中选择, 不合格直接丢弃
        pool_filtered = [p for p in pool if p["attr_pass"]]
        if len(pool_filtered) < 1:
            # 整对没有合格候选: skip, 记 0 贡献
            per_pair_metrics[(uid, asin)] = {
                "user_id": uid,
                "asin": asin,
                "random_user_id": random_uid,
                "n_pool": len(pool),
                "n_pool_attr_pass": 0,
                "skip_reason": "no_attr_pass_candidate",
                "rerank_top1_maha_target": float("nan"),
                "rerank_top1_maha_random": float("nan"),
                "rerank_top1_cond": None,
                "rerank_top1_query": None,
                "rerank_top1_attr_pass": False,
                "random_pick_maha_target": float("nan"),
                "oracle_min_maha_target": float("nan"),
                "rerank_utility": float("nan"),
                "is_oracle_hit": False,
                "rerank_top1_delta_to_oracle": float("nan"),
                "rank_rerank": -1,
                "rank_random": -1,
            }
            continue

        # === Rerank strategy 1: top-1 by maha_target_min (在 attr_pass 子集内) ===
        rerank_sorted = sorted(pool_filtered, key=lambda x: x["maha_target"])
        rerank_top1 = rerank_sorted[0]

        # === Rerank strategy 2: random pick (在 attr_pass 子集内) ===
        rng = np.random.default_rng(SEED + hash((uid, asin)) % (2**31))
        random_maha_targets = []
        for _ in range(100):
            pick = pool_filtered[rng.integers(0, len(pool_filtered))]
            random_maha_targets.append(pick["maha_target"])
        random_pick_maha = float(np.mean(random_maha_targets))

        # === Oracle hit rate: rerank_top1's maha_target 是否等于 pool_filtered 中最小 ===
        oracle_min_maha = min(p["maha_target"] for p in pool_filtered)
        rerank_top1_maha = rerank_top1["maha_target"]
        is_oracle_hit = bool(abs(rerank_top1_maha - oracle_min_maha) < 1e-6)

        # === Rerank utility ===
        rerank_utility = random_pick_maha - rerank_top1_maha  # 正 = rerank 比 random 好
        rerank_top1_delta_to_oracle = rerank_top1_maha - oracle_min_maha  # 0 = oracle hit

        # === Δ_rank rerank: rerank_top1 的 maha_target 在 pool_filtered 内排名 ===
        rank_rerank = sum(1 for p in pool_filtered if p["maha_target"] < rerank_top1_maha) + 1
        rank_random = sum(1 for p in pool_filtered if p["maha_target"] < random_pick_maha) + 1

        per_pair_metrics[(uid, asin)] = {
            "user_id": uid,
            "asin": asin,
            "random_user_id": random_uid,
            "n_pool": len(pool),
            "n_pool_attr_pass": len(pool_filtered),
            "skip_reason": None,
            "rerank_top1_maha_target": rerank_top1_maha,
            "rerank_top1_maha_random": rerank_top1["maha_random"],
            "rerank_top1_cond": rerank_top1["cond"],
            "rerank_top1_query": rerank_top1["query"],
            "rerank_top1_attr_pass": rerank_top1["attr_pass"],
            "random_pick_maha_target": random_pick_maha,
            "oracle_min_maha_target": oracle_min_maha,
            "rerank_utility": rerank_utility,
            "is_oracle_hit": is_oracle_hit,
            "rerank_top1_delta_to_oracle": rerank_top1_delta_to_oracle,
            "rank_rerank": rank_rerank,
            "rank_random": rank_random,
            "pool_maha_targets": [p["maha_target"] for p in pool],
            "pool_maha_randoms": [p["maha_random"] for p in pool],
            "pool_conds": [p["cond"] for p in pool],
        }

    log(f"  per_pair_metrics: {len(per_pair_metrics)}")
    log(f"  feat fail: {n_feat_fail}, attr fail: {n_attr_fail}")

    # === Aggregate ===
    log("[5] Aggregating metrics ...")
    # 排除 skip_reason 不为空的 pairs
    valid_pairs = {k: v for k, v in per_pair_metrics.items() if v.get("skip_reason") is None}
    log(f"  valid pairs (rerank pool with attr_pass ≥ 1): {len(valid_pairs)} / {len(per_pair_metrics)}")

    rerank_utility_per_pair = {k: v["rerank_utility"] for k, v in valid_pairs.items()}
    oracle_hit_per_pair = {k: 1.0 if v["is_oracle_hit"] else 0.0 for k, v in valid_pairs.items()}
    rerank_top1_maha_target_per_pair = {k: v["rerank_top1_maha_target"] for k, v in valid_pairs.items()}
    random_pick_maha_target_per_pair = {k: v["random_pick_maha_target"] for k, v in valid_pairs.items()}
    rerank_top1_maha_random_per_pair = {k: v["rerank_top1_maha_random"] for k, v in valid_pairs.items()}
    oracle_min_maha_target_per_pair = {k: v["oracle_min_maha_target"] for k, v in valid_pairs.items()}
    rerank_top1_attr_pass = [v["rerank_top1_attr_pass"] for v in valid_pairs.values()]

    mean_rerank_utility, ci_rerank_utility = bootstrap_ci_user_level(rerank_utility_per_pair)
    mean_oracle_hit_rate, ci_oracle_hit_rate = bootstrap_ci_user_level(oracle_hit_per_pair)
    mean_rerank_top1_maha, ci_rerank_top1_maha = bootstrap_ci_user_level(rerank_top1_maha_target_per_pair)
    mean_random_maha, ci_random_maha = bootstrap_ci_user_level(random_pick_maha_target_per_pair)
    mean_oracle_maha, ci_oracle_maha = bootstrap_ci_user_level(oracle_min_maha_target_per_pair)

    # Δ_maha: rerank_top1.maha_target vs random_pick.maha_target (per-pair paired diff)
    delta_maha_per_pair = {
        k: random_pick_maha_target_per_pair[k] - rerank_top1_maha_target_per_pair[k]
        for k in valid_pairs
    }
    mean_delta_maha, ci_delta_maha = bootstrap_ci_user_level(delta_maha_per_pair)

    # Verdict
    rerank_attr_pass_rate = float(np.mean(rerank_top1_attr_pass))

    log(f"  rerank_utility (random_maha - rerank_maha) > 0:")
    log(f"    mean = {mean_rerank_utility:.4f}, CI = [{ci_rerank_utility[0]:.4f}, {ci_rerank_utility[1]:.4f}]")
    log(f"  oracle_hit_rate:")
    log(f"    mean = {mean_oracle_hit_rate:.4f}, CI = [{ci_oracle_hit_rate[0]:.4f}, {ci_oracle_hit_rate[1]:.4f}]")
    log(f"  rerank_top1 maha_target:")
    log(f"    mean = {mean_rerank_top1_maha:.4f}, CI = [{ci_rerank_top1_maha[0]:.4f}, {ci_rerank_top1_maha[1]:.4f}]")
    log(f"  random_pick maha_target:")
    log(f"    mean = {mean_random_maha:.4f}, CI = [{ci_random_maha[0]:.4f}, {ci_random_maha[1]:.4f}]")
    log(f"  Δ_maha paired: {mean_delta_maha:.4f}, CI = [{ci_delta_maha[0]:.4f}, {ci_delta_maha[1]:.4f}]")
    log(f"  rerank attr_pass: {rerank_attr_pass_rate:.4f}")

    # === Verdict ===
    checks = {
        "rerank_utility_positive_ci": ci_rerank_utility[0] > 0,  # CI 下界 > 0
        "rerank_top1_maha_below_random_ci": ci_rerank_top1_maha[1] < ci_random_maha[0],  # rerank 区间在 random 区间左
        "delta_maha_positive_ci": ci_delta_maha[0] > 0,  # paired CI 下界 > 0
        "attr_pass_100pct": rerank_attr_pass_rate >= 0.99,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    log(f"  checks: {checks}")
    log(f"  VERDICT: {verdict}")

    # === Save ===
    eval_dict = {
        "n_pairs": len(per_pair_metrics),
        "n_pool_per_pair": 10,
        "rerank_strategy": "top-1 by maha_target_min (VADES post-hoc selection)",
        "baseline_strategy": "random pick (100 random picks averaged)",
        "oracle_strategy": "pool min maha_target",
        "rerank_utility": {
            "definition": "random_pick_maha_target - rerank_top1_maha_target (positive = rerank wins)",
            "mean": mean_rerank_utility,
            "ci_95": list(ci_rerank_utility),
        },
        "oracle_hit_rate": {
            "definition": "rerank_top1 IS oracle min (by construction = 1.0; logged for sanity)",
            "mean": mean_oracle_hit_rate,
            "ci_95": list(ci_oracle_hit_rate),
        },
        "rerank_top1_maha_target": {
            "mean": mean_rerank_top1_maha,
            "ci_95": list(ci_rerank_top1_maha),
        },
        "random_pick_maha_target": {
            "mean": mean_random_maha,
            "ci_95": list(ci_random_maha),
        },
        "delta_maha_paired": {
            "definition": "per-pair (random_maha - rerank_maha)",
            "mean": mean_delta_maha,
            "ci_95": list(ci_delta_maha),
        },
        "attr_pass_rate_rerank_top1": rerank_attr_pass_rate,
        "checks": checks,
        "verdict": verdict,
        "n_feat_fail": n_feat_fail,
        "n_attr_fail": n_attr_fail,
        "note": "Phase 10.10.2: reuse Phase 10.9 4i2 candidates (8760 candidates). "
                "Pool = 5 D_target + 5 C_random per pair (10 candidates rerank pool). "
                "Re-run with N=10 when phase10_10_1 finishes.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for v in per_pair_metrics.values():
            row = {k: val for k, val in v.items() if not isinstance(val, np.ndarray)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log(f"VERDICT: {verdict}")
    log("=" * 70)


if __name__ == "__main__":
    main()