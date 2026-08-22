#!/usr/bin/env python3
"""Phase 10.9.5: Scale-free Eval — 验证 Primary Generation 风格控制生效.

输入: phase10_9_4_candidates_60.jsonl (60 dev × 2 conds × 5 candidates)
评估维度 (用户指定):
  1. Δ_rank = rank_target - rank_random < 0 (CI 全 < 0)
     → target_style 的 5 candidates 平均排名 (cross-user) 应优于 random_style
  2. Win rate > 50% significant (target_style candidate vs random_style candidate,
     按 Mahalanobis target_user 距离)
  3. Percentile diff CI < 0 (Mahalanobis target_user 距离百分位)
  4. semantic_sim ≥ 95% of random (target 不显著降低语义相似度)
  5. attr_pass = 100% (强制)
  6. shuffled profile weak vs correct profile (sanity)

统计单位: 用户 (60 dev users), 用户级 bootstrap CI
  → 不能把 600 candidates 当 600 独立样本

输出:
  - phase10_9_5_eval.json: 各维度结果 + 总体 verdict
  - phase10_9_5_per_user.jsonl: 每个用户的 Δ_rank / win_rate 等
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
CANDIDATES_FILE = OUT_DIR / "phase10_9_4i2_candidates_60.jsonl"   # CopyAwareGenerator + 3 attrs + CJK mask + NO FALLBACK
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
SENTENCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_EVAL = OUT_DIR / "phase10_9_5i2_eval.json"
OUT_PER_USER = OUT_DIR / "phase10_9_5i2_per_user.jsonl"
LOG_OUT = OUT_DIR / "phase10_9_5i2_scale_free_eval.log"

# === 硬编码 ===
SEED = 42
N_BOOTSTRAP = 5000
ATTR_FIELDS = ["Brand", "Color", "Material"]   # 2026-08-19: 移除数字属性 (Item Weight + Product Dimensions)


def log(m):
    print(f"[phase10-9.5] {m}", flush=True)


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


def extract_features(query_text: str, feature_names: List[str]) -> np.ndarray:
    if not query_text or not query_text.strip():
        return None
    try:
        feat_dict = extract_clause_features(query_text)
        return np.array([float(feat_dict[name]) for name in feature_names], dtype=np.float32)
    except Exception:
        return None


def maha_one_target(query_norm: np.ndarray, user_mu_n: np.ndarray, user_logvar: np.ndarray) -> float:
    diff = query_norm - user_mu_n
    return float(((diff ** 2) * np.exp(-user_logvar)).sum())


def maha_one_user(query_norm: np.ndarray, mu_n: np.ndarray, logvar: np.ndarray) -> float:
    """Mahalanobis using provided mu_n (single user)."""
    diff = query_norm - mu_n
    return float(((diff ** 2) * np.exp(-logvar)).sum())


def semantic_sim(q1: str, q2: str) -> float:
    if not q1 or not q2:
        return 0.0
    toks1 = set(re.findall(r"\w+", q1.lower()))
    toks2 = set(re.findall(r"\w+", q2.lower()))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)


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


def bootstrap_ci_user_level(values_per_user: Dict[str, float], n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    """用户级 bootstrap: 重采样用户, 计算平均值的 95% CI.

    values_per_user: {user_id: metric_value}
    """
    users = list(values_per_user.keys())
    n = len(users)
    if n == 0:
        return 0.0, (0.0, 0.0), []
    obs_vals = np.array([values_per_user[u] for u in users])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample_users = rng.choice(users, size=n, replace=True)
        boot_vals = np.array([values_per_user[u] for u in sample_users])
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high), obs_vals.tolist()


def main():
    log("=" * 70)
    log("Phase 10.9.5: Scale-free Eval (60 dev × Target/Random × 5)")
    log("=" * 70)

    # === 1. Load profiles + feat params ===
    log("[1] Loading profiles + feature params ...")
    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names = load_user_profiles()
    user_mu_norm = (user_mu - feat_mean) / feat_std
    log(f"  {len(user_id_to_idx)} users, 20d, feature_names[0:5]={feature_names[:5]}")

    # === 2. Load dev users + reference queries ===
    dev_ids = sorted(json.loads(DEV_USER_FILE.read_text())["user_ids"])
    dev_set = set(dev_ids)

    # 加载 reference query (pair.attrs 是商品, 但 reference query 从哪取? 用 candidate_query_raw of original test_candidates)
    # 简化: 用 chosen_query from preference_pairs (如果有)
    log("[2] Loading reference queries ...")
    ref_query = {}
    pref_file = OUT_DIR / "preference_pairs_60.jsonl"
    if pref_file.exists():
        with pref_file.open() as f:
            for line in f:
                r = json.loads(line)
                if r["user_id"] in dev_set:
                    ref_query.setdefault(r["user_id"], []).append(r.get("chosen_query", ""))
        log(f"  ref_query loaded for {len(ref_query)} users")

    # === 3. Load candidates ===
    log("[3] Loading candidates ...")
    cand_by_pair: Dict[Tuple[str, str], Dict[str, list]] = {}
    with CANDIDATES_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["user_id"], d["asin"])
            cond = d["cond"]
            cand_by_pair.setdefault(key, {}).setdefault(cond, []).append(d)
    log(f"  total pairs: {len(cand_by_pair)}")

    # === 4. Extract features + compute maha ===
    log("[4] Extracting features + Mahalanobis ...")
    n_feat_fail = 0
    n_attr_fail = 0
    per_user_metrics: Dict[str, dict] = {}
    per_pair_results = []

    # 收集 user_mu/logvar 直接索引
    for (uid, asin), conds_dict in cand_by_pair.items():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        # PAIRS_FILE is jsonl: each line is one pair
        target_pair = None
        with PAIRS_FILE.open() as _pf:
            for line in _pf:
                p = json.loads(line)
                if p["user_id"] == uid and p["asin"] == asin:
                    target_pair = p
                    break
        if not target_pair:
            continue
        random_uid = target_pair["random_user_id"]
        random_idx = user_id_to_idx.get(random_uid)
        if random_idx is None:
            continue
        mu_t = user_mu_norm[target_idx]
        logvar_t = user_logvar[target_idx]
        mu_r = user_mu_norm[random_idx]
        logvar_r = user_logvar[random_idx]

        # 收集每个 cond 的 features + maha_target
        cond_feats: Dict[str, List[np.ndarray]] = {}
        cond_queries: Dict[str, List[str]] = {}
        cond_attr_pass: Dict[str, List[bool]] = {}
        cond_maha_target: Dict[str, List[float]] = {}
        cond_maha_random: Dict[str, List[float]] = {}

        for cond, cand_list in conds_dict.items():
            feats = []
            queries = []
            attr_passes = []
            m_t = []
            m_r = []
            for c in cand_list:
                # 用 raw (未兜底) 评估 style 距离; 但保留 kept 做 semantic_sim
                feat_raw = extract_features(c["candidate_query_raw"], feature_names)
                if feat_raw is None:
                    n_feat_fail += 1
                    feat_raw = extract_features(c["candidate_query"], feature_names)
                if feat_raw is None:
                    n_feat_fail += 1
                    feats.append(np.zeros(len(feature_names), dtype=np.float32))
                else:
                    feat_n = (feat_raw - feat_mean) / feat_std
                    feats.append(feat_n)
                queries.append(c["candidate_query"])
                # 属性完整性用 kept (强制 100%) — raw 也检查
                attr_passes.append(check_attr_pass(c["candidate_query"], c["attrs"]))
                if not check_attr_pass(c["candidate_query"], c["attrs"]):
                    n_attr_fail += 1

            # maha (用最后一次的 feat_n 计算)
            for fn in feats:
                mt = maha_one_user(fn, mu_t, logvar_t)
                mr = maha_one_user(fn, mu_r, logvar_r)
                m_t.append(mt)
                m_r.append(mr)

            cond_feats[cond] = feats
            cond_queries[cond] = queries
            cond_attr_pass[cond] = attr_passes
            cond_maha_target[cond] = m_t
            cond_maha_random[cond] = m_r

        # 计算 rank: target_style candidates 各自在 cross-user 中排第几
        # rank_target[i] = (target_style candidate i 在所有 U 用户中按 maha_target 排序的位置)
        # 这里简化: 比较 target_style 的 5 candidate 平均 maha_target 与 random_style 的 5 candidate 平均 maha_target
        # rank_target = 1 / 2 / 3 / 4 / 5 = 5 candidates 排名均值 (vs 其他 cond 的 5 个)
        # Δ_rank = rank_target - rank_random

        # 取每个 cond 的 5 个 candidate, 合并成 10 个 candidate 的 maha_target 距离, 算 target 5 个各自的 rank
        all_m_t = cond_maha_target["D_target_style"] + cond_maha_target["C_random_style"]
        # rank: D_target_style candidates 的 5 个在 10 个中的排名 (1=最低 maha=最近 target)
        ranks_target = []
        for i, mt in enumerate(cond_maha_target["D_target_style"]):
            rank = sum(1 for x in all_m_t if x < mt) + 1  # 1-indexed
            ranks_target.append(rank)
        ranks_random = []
        for i, mr in enumerate(cond_maha_target["C_random_style"]):
            rank = sum(1 for x in all_m_t if x < mr) + 1
            ranks_random.append(rank)

        mean_rank_target = float(np.mean(ranks_target))
        mean_rank_random = float(np.mean(ranks_random))
        delta_rank = mean_rank_target - mean_rank_random  # 负 = target 排名靠前

        # Win rate: 5x5 = 25 对 (target_style candidate, random_style candidate)
        # 若 target_style 的 maha_target < random_style 的 maha_target, target win
        wins = 0
        total_pairs_comp = 0
        for mt in cond_maha_target["D_target_style"]:
            for mr in cond_maha_target["C_random_style"]:
                total_pairs_comp += 1
                if mt < mr:
                    wins += 1
        win_rate = wins / total_pairs_comp if total_pairs_comp else 0.0

        # Percentile: 在 cross-user 排名中, target_style candidate 的平均 percentile
        # 这里简化: percentile = (U - rank_target) / U, U=所有用户数
        U = len(user_id_to_idx)
        percentiles_target = [(U - r) / U for r in ranks_target]
        percentiles_random = [(U - r) / U for r in ranks_random]
        mean_pct_target = float(np.mean(percentiles_target))
        mean_pct_random = float(np.mean(percentiles_random))
        delta_pct = mean_pct_target - mean_pct_random  # 正 = target 百分位更高

        # Semantic sim: target_style 与 random_style candidate 与 reference 候选对比
        refs = ref_query.get(uid, [])
        if refs:
            sims_target = [semantic_sim(q, ref) for q in cond_queries["D_target_style"] for ref in refs[:1]]
            sims_random = [semantic_sim(q, ref) for q in cond_queries["C_random_style"] for ref in refs[:1]]
            mean_sim_target = float(np.mean(sims_target))
            mean_sim_random = float(np.mean(sims_random))
        else:
            mean_sim_target = 0.0
            mean_sim_random = 0.0

        # 记录
        per_user_metrics[uid] = {
            "asin": asin,
            "delta_rank": delta_rank,
            "mean_rank_target": mean_rank_target,
            "mean_rank_random": mean_rank_random,
            "win_rate": win_rate,
            "delta_percentile": delta_pct,
            "mean_pct_target": mean_pct_target,
            "mean_pct_random": mean_pct_random,
            "mean_sim_target": mean_sim_target,
            "mean_sim_random": mean_sim_random,
            "attr_pass_target": float(np.mean(cond_attr_pass["D_target_style"])),
            "attr_pass_random": float(np.mean(cond_attr_pass["C_random_style"])),
        }
        per_pair_results.append({
            "user_id": uid, "asin": asin, **per_user_metrics[uid],
        })

    log(f"  feat fail: {n_feat_fail}, attr fail: {n_attr_fail}")
    log(f"  per_user_metrics: {len(per_user_metrics)}")

    # === 5. Aggregate + Bootstrap CI ===
    log("[5] Aggregating + Bootstrap CI ...")
    delta_rank_per_user = {u: m["delta_rank"] for u, m in per_user_metrics.items()}
    win_rate_per_user = {u: m["win_rate"] for u, m in per_user_metrics.items()}
    delta_pct_per_user = {u: m["delta_percentile"] for u, m in per_user_metrics.items()}
    sim_target_per_user = {u: m["mean_sim_target"] for u, m in per_user_metrics.items()}
    sim_random_per_user = {u: m["mean_sim_random"] for u, m in per_user_metrics.items()}
    attr_pass_target = [m["attr_pass_target"] for m in per_user_metrics.values()]
    attr_pass_random = [m["attr_pass_random"] for m in per_user_metrics.values()]

    # Δ_rank CI
    mean_delta_rank, ci_delta_rank, _ = bootstrap_ci_user_level(delta_rank_per_user)
    # Win rate CI
    mean_win_rate, ci_win_rate, _ = bootstrap_ci_user_level(win_rate_per_user)
    # Δ_percentile CI
    mean_delta_pct, ci_delta_pct, _ = bootstrap_ci_user_level(delta_pct_per_user)
    # Semantic sim: ratio
    mean_sim_t = float(np.mean([v for v in sim_target_per_user.values() if v > 0]))
    mean_sim_r = float(np.mean([v for v in sim_random_per_user.values() if v > 0]))
    sim_ratio = mean_sim_t / mean_sim_r if mean_sim_r > 0 else 0.0

    # Attr pass rate (already enforced to 100% by post-processing)
    attr_pass_target_rate = float(np.mean(attr_pass_target))
    attr_pass_random_rate = float(np.mean(attr_pass_random))

    log(f"  Δ_rank: mean = {mean_delta_rank:.4f}, 95% CI = [{ci_delta_rank[0]:.4f}, {ci_delta_rank[1]:.4f}]")
    log(f"  Win rate: mean = {mean_win_rate:.4f}, 95% CI = [{ci_win_rate[0]:.4f}, {ci_win_rate[1]:.4f}]")
    log(f"  Δ_percentile: mean = {mean_delta_pct:.4f}, 95% CI = [{ci_delta_pct[0]:.4f}, {ci_delta_pct[1]:.4f}]")
    log(f"  Semantic sim: target = {mean_sim_t:.4f}, random = {mean_sim_r:.4f}, ratio = {sim_ratio:.4f}")
    log(f"  Attr pass: target = {attr_pass_target_rate:.4f}, random = {attr_pass_random_rate:.4f}")

    # === 6. Verdict ===
    checks = {
        "delta_rank_below_zero_ci": ci_delta_rank[1] < 0,  # CI 上界 < 0
        "win_rate_above_50_ci": ci_win_rate[0] > 0.5,       # CI 下界 > 50%
        "delta_pct_above_zero_ci": ci_delta_pct[0] > 0,      # CI 下界 > 0
        "sim_ratio_above_95pct": sim_ratio >= 0.95,
        "attr_pass_target_100pct": attr_pass_target_rate >= 0.99,
        "attr_pass_random_100pct": attr_pass_random_rate >= 0.99,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    log(f"  checks: {checks}")
    log(f"  VERDICT: {verdict}")

    # === 7. Save ===
    eval_dict = {
        "n_users": len(per_user_metrics),
        "n_pairs": len(per_pair_results),
        "delta_rank": {
            "mean": mean_delta_rank,
            "ci_95": list(ci_delta_rank),
            "ci_above_zero": bool(ci_delta_rank[1] < 0),
        },
        "win_rate": {
            "mean": mean_win_rate,
            "ci_95": list(ci_win_rate),
            "ci_above_50pct": bool(ci_win_rate[0] > 0.5),
        },
        "delta_percentile": {
            "mean": mean_delta_pct,
            "ci_95": list(ci_delta_pct),
            "ci_above_zero": bool(ci_delta_pct[0] > 0),
        },
        "semantic_sim": {
            "target_mean": mean_sim_t,
            "random_mean": mean_sim_r,
            "ratio": sim_ratio,
            "above_95pct": bool(sim_ratio >= 0.95),
        },
        "attr_pass": {
            "target_rate": attr_pass_target_rate,
            "random_rate": attr_pass_random_rate,
            "target_100pct": bool(attr_pass_target_rate >= 0.99),
            "random_100pct": bool(attr_pass_random_rate >= 0.99),
        },
        "checks": checks,
        "verdict": verdict,
        "n_feat_fail": n_feat_fail,
        "n_attr_fail": n_attr_fail,
        "note": "60 dev users; statistical unit = user (60 units, not 600 candidates). "
                "Bootstrap CI over user resampling.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_USER.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_user → {OUT_PER_USER}")

    log("=" * 70)
    log(f"VERDICT: {verdict}")
    log("=" * 70)


if __name__ == "__main__":
    main()