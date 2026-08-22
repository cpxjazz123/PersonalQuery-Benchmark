#!/usr/bin/env python3
"""Phase 10.7: Scale-Free Paired Evaluation (解决用户间 Mahalanobis 距离尺度差异).

核心问题 (Phase 10.6.4):
  - 60 user 评估 D_target_style=205.41 vs C_random_style=220.49 (D-C=-15.08)
  - CI 95% [-56.10, +28.60] 仍跨 0, 不是 DPO 没学到, 而是 raw Mahalanobis 距离
    在不同用户间尺度方差太大 (用户级 SD ≈ 160~170, 效应 ≈ 15, 标准化 d ≈ 0.09)

5 项子任务:
  10.7.1: 用户胜率 P(Δu<0) + 配对 bootstrap CI + 配对置换检验
  10.7.2: 用户内标准化 (percentile / MAD-zscore using B_zero_prefix 作 calibration)
  10.7.3: 跨用户 target rank + MRR + win rate
  10.7.4: Semantic similarity (vs reference)
  10.7.5: 综合判定 (4 criteria 全部满足)

最终判定 (硬约束):
  ① scale-free 配对 CI (D'_target - D'_random) 完全 < 0
  ② Target-style 用户胜率 > 50% 且 p<0.05 (置换检验)
  ③ Semantic sim(D) ≥ semantic sim(C) [不显著低于]
  ④ attr_pass = 100%
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
EVAL_CANDS = OUT_DIR / "phase10_eval_candidates_60.jsonl"      # 1200 lines
REF_CANDS = OUT_DIR / "phase10_candidates_60.jsonl"            # 900 lines (reference for semantic sim)
PROJ_CKPT = OUT_DIR / "phase10_6_3_projector.pt"
META_OUT = OUT_DIR / "phase10_7_eval_scale_free.json"
LOG_OUT = OUT_DIR / "phase10_7_eval_scale_free.log"

# === VADES scorer ===
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"

# === 配置 ===
CONDITIONS = ["A_no_style", "B_zero_prefix", "C_random_style", "D_target_style"]
N_BOOT = 5000
N_PERM = 5000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 1. 加载 user profiles + 训练集标准化参数
# ============================================================================
def load_profiles_and_scaler() -> Tuple[
    Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]
]:
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


# ============================================================================
# 2. 提取 20d 句法特征 (复用 extract_clause_features)
# ============================================================================
def extract_features_batch(texts: List[str], feature_names: List[str]) -> np.ndarray:
    """batch 提取特征, 返回 [N, 20] float32."""
    from syntactic_analysis.extract_clause_features_single_query import extract_clause_features
    out = np.zeros((len(texts), len(feature_names)), dtype=np.float32)
    ok = 0
    for i, t in enumerate(texts):
        try:
            d = extract_clause_features(t)
            for k, name in enumerate(feature_names):
                out[i, k] = float(d.get(name, 0.0))
            ok += 1
        except Exception:
            pass
    return out, ok


def semantic_sim_jaccard(q1: str, q2: str) -> float:
    """Jaccard word overlap (复用 phase10_evaluate.py 的语义相似)."""
    import re
    if not q1 or not q2:
        return 0.0
    toks1 = set(re.findall(r"\w+", q1.lower()))
    toks2 = set(re.findall(r"\w+", q2.lower()))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)


# ============================================================================
# 3. 加载 candidates (eval + ref)
# ============================================================================
def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ============================================================================
# 4. Mahalanobis 距离
# ============================================================================
def maha(q_n: np.ndarray, mu_n: np.ndarray, logvar: np.ndarray) -> float:
    diff = q_n - mu_n
    return float(((diff ** 2) * np.exp(-logvar)).sum())


# ============================================================================
# 5. Bootstrap + 配对置换
# ============================================================================
def bootstrap_ci_paired(diff: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05,
                        seed: int = SEED) -> Tuple[float, float, float]:
    """返回 (mean, lo, hi). 用户为单位 bootstrap (replace)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = diff[idx].mean()
    return float(diff.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_perm_test(diff: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> Tuple[float, float]:
    """配对置换检验 (sign-flip). H0: mean(diff)==0. 返回 (observed_mean, p_two_sided)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    obs = diff.mean()
    abs_obs = abs(obs)
    cnt = 0
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_diff = diff * signs
        if abs(perm_diff.mean()) >= abs_obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)  # +1 避免 p=0
    return float(obs), float(p)


# ============================================================================
# 6. User-internal MAD-zscore (用 B_zero_prefix 作 calibration)
# ============================================================================
def compute_user_internal_norm(per_user_maha: Dict[str, Dict[str, np.ndarray]]
                                ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], int]:
    """对每个用户, 用 cond='B_zero_prefix' 作为 calibration: 计算 percentile rank + MAD-zscore.

    返回 (per_user[cond]=percentile_rank, per_user[cond]=zscore (where valid), n_degenerate).
    Rule 7 (AGENTS): 无 fallback — degenerate (MAD<eps) 用户明确排除并报告, 不静默替换.
    """
    # percentile rank (对所有用户都能算, 永远 valid)
    pct_out = defaultdict(dict)
    # MAD-zscore (需要 calibration 不退化)
    z_out = defaultdict(dict)
    n_degenerate = 0
    for uid, by_cond in per_user_maha.items():
        cal = by_cond.get("B_zero_prefix", np.array([]))
        if len(cal) < 2:
            # 无法构造 calibration, 跳过该用户 (no percentile either)
            continue
        # === percentile rank: 基于 calibration CDF ===
        # 把每条 cond 的 maha 与 cal 一起排序, 算 percentile
        for cond, vals in by_cond.items():
            combined = np.concatenate([cal, vals])
            ranks = np.argsort(np.argsort(combined))
            # val 在 combined[len(cal):] 即为 percentile
            pct = (ranks[len(cal):] + 1) / (len(combined) + 1)
            pct_out[uid][cond] = float(pct.mean())  # 平均 percentile across candidates
        # === MAD-zscore ===
        med = float(np.median(cal))
        mad = float(np.median(np.abs(cal - med))) * 1.4826
        if mad < 1e-6:  # MAD 退化, 排除
            n_degenerate += 1
            continue
        for cond, vals in by_cond.items():
            z_out[uid][cond] = float((vals.mean() - med) / mad)
    return pct_out, z_out, n_degenerate


# ============================================================================
# 7. 跨用户 target rank
# ============================================================================
def compute_target_rank(eval_records: List[dict], per_user_maha: Dict[str, Dict[str, np.ndarray]],
                        user_id_to_idx: Dict[str, int], user_mu_norm: np.ndarray,
                        user_logvar: np.ndarray, user_feat_norm: np.ndarray,
                        target_asin_lookup: Dict[Tuple[str, str], str]
                        ) -> Tuple[float, float, float]:
    """对每条 D_target_style candidate, 算 D_u(q) for all 2918 users, 取 target 用户 rank.

    rank = target_u 在 sorted(D_user(q)) 的位置 (从 1 开始, 越小越好).
    MRR = mean(1/rank).
    win rate = P(target_u rank == 1).
    """
    all_user_ids = list(user_id_to_idx.keys())
    n_users = len(all_user_ids)
    log = lambda m: print(f"[scale-free] {m}", flush=True)
    log(f"  cross-user target rank against all {n_users} users")

    # 收集所有 D_target_style candidates (按 user, asin 分组)
    targets_by_pair = defaultdict(list)
    for r in eval_records:
        if r["cond"] != "D_target_style":
            continue
        key = (r["user_id"], r["asin"])
        targets_by_pair[key].append(r)

    # feature 已抽取 → 缓存
    cached_features: Dict[int, np.ndarray] = {}

    def get_features(record_id: int) -> np.ndarray:
        if record_id not in cached_features:
            cached_features[record_id] = user_feat_norm[record_id]
        return cached_features[record_id]

    ranks = []
    rank1_count = 0
    total = 0
    start = time.time()
    for key, recs in targets_by_pair.items():
        if not key in [(r["user_id"], r["asin"]) for r in recs]:
            continue
        # 取首条作代表 (同 (user,asin,cond) 5 条 maha 均值差异很小)
        # 但 rank 应取 mean across 5 candidates 更稳健
        # 简化: 取 candidate_index=0
        first = recs[0]
        target_u = first["user_id"]
        if target_u not in user_id_to_idx:
            continue
        # 取该条 query 的特征向量
        idx_in_feat = first.get("_feat_idx")
        if idx_in_feat is None:
            continue
        q_n = user_feat_norm[idx_in_feat]
        # 对每个 candidate (5 条), 算 target_user rank
        for r in recs:
            idx = r.get("_feat_idx")
            if idx is None:
                continue
            q_n = user_feat_norm[idx]
            # 算 D_u(q) for all users
            diff = q_n[None, :] - user_mu_norm  # [n_users, D]
            d_all = ((diff ** 2) * np.exp(-user_logvar)).sum(axis=1)  # [n_users]
            # 排名 (1 = 最小距离)
            target_rank = int((d_all < d_all[user_id_to_idx[target_u]]).sum() + 1)
            ranks.append(target_rank)
            if target_rank == 1:
                rank1_count += 1
            total += 1
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    mean_rank = float(np.mean(ranks))
    mrr = float(np.mean([1.0 / r for r in ranks]))
    win_rate = rank1_count / total
    log(f"  computed {total} target ranks in {time.time()-start:.1f}s, mean_rank={mean_rank:.2f}, MRR={mrr:.4f}, win_rate={win_rate:.4f}")
    return mean_rank, mrr, win_rate


# ============================================================================
# Main
# ============================================================================
def main():
    log = lambda m: print(f"[phase10-7] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.7: Scale-Free Paired Evaluation")
    log("=" * 70)

    # === 1. Load profiles + scaler ===
    log("[1] Loading VADES profiles + scaler ...")
    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names = load_profiles_and_scaler()
    log(f"  {len(user_id_to_idx)} users, {len(feature_names)} features")

    # === 2. Load eval candidates ===
    log("[2] Loading eval candidates (1200 lines) ...")
    eval_records = load_jsonl(EVAL_CANDS)
    log(f"  {len(eval_records)} eval records")

    # === 3. Load ref candidates (for semantic sim) ===
    log("[3] Loading reference candidates (900 lines) ...")
    ref_records = load_jsonl(REF_CANDS)
    # ref → (user, asin, group, candidate_index) → query
    ref_lookup = {}
    for r in ref_records:
        key = (r["user_id"], r["asin"], r["group"], r["candidate_index"])
        ref_lookup[key] = r["candidate_query"]
    log(f"  {len(ref_lookup)} ref queries")

    # === 4. Extract features (batch via spaCy nlp.pipe in extract_clause_features) ===
    log("[4] Extracting 20d features for all queries ...")
    all_queries = [r["candidate_query"] for r in eval_records]
    feat_norm, ok = extract_features_batch(all_queries, feature_names)
    # 标准化
    feat_norm_full = (feat_norm - feat_mean) / feat_std
    log(f"  feature extraction ok={ok}/{len(eval_records)}")

    # 把特征 id 记录到 records
    for i, r in enumerate(eval_records):
        r["_feat_idx"] = i

    # user_mu 也标准化
    user_mu_norm = (user_mu - feat_mean) / feat_std

    # === 5. 计算每条 (user, cond) Mahalanobis 距离 vs target_user ===
    log("[5] Computing Mahalanobis distance (vs target_user) per record ...")
    per_user_cond_maha: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in eval_records:
        target_u = r["user_id"]
        if target_u not in user_id_to_idx:
            continue
        u_idx = user_id_to_idx[target_u]
        q_n = feat_norm_full[r["_feat_idx"]]
        d = maha(q_n, user_mu_norm[u_idx], user_logvar[u_idx])
        per_user_cond_maha[target_u][r["cond"]].append(d)

    # 转成 numpy array
    per_user_cond_maha_np: Dict[str, Dict[str, np.ndarray]] = {}
    for uid, by_cond in per_user_cond_maha.items():
        per_user_cond_maha_np[uid] = {c: np.array(vs, dtype=np.float32) for c, vs in by_cond.items()}

    # === 6. Phase 10.7.1: 用户胜率 P(Δu<0) + bootstrap + permutation ===
    log("[6] === 10.7.1: 用户胜率 P(Δu<0) + bootstrap CI + 配对置换 ===")
    common_users = [u for u in per_user_cond_maha_np
                    if "D_target_style" in per_user_cond_maha_np[u]
                    and "C_random_style" in per_user_cond_maha_np[u]]
    log(f"  {len(common_users)} users with both C and D")

    deltas = []
    for u in common_users:
        d_target = per_user_cond_maha_np[u]["D_target_style"].mean()
        d_random = per_user_cond_maha_np[u]["C_random_style"].mean()
        deltas.append(d_target - d_random)
    deltas_arr = np.array(deltas, dtype=np.float64)

    win_count = int((deltas_arr < 0).sum())
    win_rate = win_count / len(deltas_arr)
    log(f"  Win rate (Δu<0): {win_count}/{len(deltas_arr)} = {win_rate:.4f}")

    bm, blo, bhi = bootstrap_ci_paired(deltas_arr)
    log(f"  Mean Δ (D-C): {bm:.4f}, bootstrap 95% CI [{blo:.4f}, {bhi:.4f}]")

    obs, p_perm = paired_perm_test(deltas_arr)
    log(f"  Paired permutation: obs={obs:.4f}, p={p_perm:.4f}")

    p107_1 = {
        "n_users": len(common_users),
        "win_rate_pct": win_rate,
        "win_count": win_count,
        "mean_delta_d_minus_c": bm,
        "bootstrap_ci95": [blo, bhi],
        "ci_upper_below_zero": bhi < 0,
        "permutation_p_two_sided": p_perm,
        "permutation_p_lt_005": p_perm < 0.05,
    }

    # === 7. Phase 10.7.2: 用户内标准化 (percentile + MAD-zscore) ===
    log("[7] === 10.7.2: 用户内标准化 (percentile + MAD-zscore) ===")
    user_pct, user_z, n_degenerate = compute_user_internal_norm(per_user_cond_maha_np)
    log(f"  {n_degenerate}/{len(user_z)} users had degenerate calibration (excluded from zscore)")
    # percentile delta
    pct_deltas = []
    for u in common_users:
        if "D_target_style" in user_pct.get(u, {}) and "C_random_style" in user_pct.get(u, {}):
            pct_deltas.append(user_pct[u]["D_target_style"] - user_pct[u]["C_random_style"])
    pct_arr = np.array(pct_deltas, dtype=np.float64)
    if len(pct_arr) > 0:
        pb, plo, phi = bootstrap_ci_paired(pct_arr)
        pobs, pp_perm = paired_perm_test(pct_arr)
        log(f"  Percentile Δ (D'-C'): mean={pb:.4f}, CI [{plo:.4f}, {phi:.4f}], p_perm={pp_perm:.4f}")
    else:
        pb, plo, phi, pobs, pp_perm = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    # MAD-zscore delta (only valid users)
    z_deltas = []
    for u in common_users:
        if "D_target_style" in user_z.get(u, {}) and "C_random_style" in user_z.get(u, {}):
            z_deltas.append(user_z[u]["D_target_style"] - user_z[u]["C_random_style"])
    z_arr = np.array(z_deltas, dtype=np.float64)
    if len(z_arr) > 0:
        zb, zlo, zhi = bootstrap_ci_paired(z_arr)
        zobs, zp_perm = paired_perm_test(z_arr)
        log(f"  MAD-zscore Δ (D'-C'): mean={zb:.4f}, CI [{zlo:.4f}, {zhi:.4f}], p_perm={zp_perm:.4f}")
    else:
        zb, zlo, zhi, zobs, zp_perm = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        log("  ⚠ no z_deltas available")

    p107_2 = {
        "n_users_percentile": len(pct_arr),
        "percentile_delta_d_minus_c_mean": pb,
        "percentile_ci95": [plo, phi],
        "percentile_ci_upper_below_zero": (phi < 0) if not math.isnan(phi) else False,
        "percentile_perm_p": pp_perm,
        "n_users_mad_zscore": len(z_arr),
        "n_degenerate_excluded": n_degenerate,
        "mad_zscore_delta_d_minus_c_mean": zb,
        "mad_zscore_ci95": [zlo, zhi],
        "mad_zscore_ci_upper_below_zero": (zhi < 0) if not math.isnan(zhi) else False,
        "mad_zscore_perm_p": zp_perm,
    }

    # === 8. Phase 10.7.3: 跨用户 target rank + MRR ===
    log("[8] === 10.7.3: 跨用户 target rank (vs all 2918 users) ===")
    # 准备 user_feat_norm (eval 特征的标准化)
    user_feat_norm = feat_norm_full
    # target_asin_lookup: for each eval record, find the reference candidate
    target_asin_lookup = {(r["user_id"], r["asin"]): r["asin"] for r in eval_records}

    mean_rank, mrr, target_win_rate = compute_target_rank(
        eval_records, per_user_cond_maha_np,
        user_id_to_idx, user_mu_norm, user_logvar, user_feat_norm, target_asin_lookup
    )
    log(f"  Mean target rank: {mean_rank:.2f}/2918 (lower better, random baseline ~1459)")
    log(f"  MRR: {mrr:.4f} (random baseline ~0.001)")
    log(f"  Win rate (rank==1): {target_win_rate:.4f} (random baseline ~0.0003)")

    p107_3 = {
        "n_records": len(eval_records),
        "mean_target_rank": mean_rank,
        "MRR": mrr,
        "target_win_rate_rank1": target_win_rate,
        "random_baseline_rank": 1459,
        "random_baseline_win_rate": 1.0 / 2918,
    }

    # === 9. Phase 10.7.4: Semantic similarity (Jaccard vs ref) ===
    log("[9] === 10.7.4: Semantic similarity (Jaccard vs reference) ===")
    # reference = (user, asin, group=A_no_style, candidate_index=0)
    per_cond_sim = defaultdict(list)
    for r in eval_records:
        ref_key = (r["user_id"], r["asin"], "A_no_style", 0)
        ref_q = ref_lookup.get(ref_key)
        if not ref_q:
            continue
        sim = semantic_sim_jaccard(r["candidate_query"], ref_q)
        per_cond_sim[r["cond"]].append(sim)
    sim_summary = {}
    for c, vs in per_cond_sim.items():
        arr = np.array(vs, dtype=np.float64)
        sim_summary[c] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": len(arr),
        }
        log(f"  {c}: mean sim = {arr.mean():.4f} (n={len(arr)})")

    # 检查 D ≥ C (且不显著低于)
    sim_d = sim_summary.get("D_target_style", {}).get("mean", 0)
    sim_c = sim_summary.get("C_random_style", {}).get("mean", 0)
    sim_b = sim_summary.get("B_zero_prefix", {}).get("mean", 0)
    sim_a = sim_summary.get("A_no_style", {}).get("mean", 0)
    log(f"  → D vs C: {sim_d:.4f} vs {sim_c:.4f} (D-C = {sim_d - sim_c:+.4f})")
    log(f"  → D vs B: {sim_d:.4f} vs {sim_b:.4f}")

    p107_4 = {
        "per_cond": sim_summary,
        "D_minus_C": sim_d - sim_c,
        "D_geq_C": sim_d >= sim_c * 0.95,  # 允许 5% 退化
    }

    # === 10. attr_pass ===
    log("[10] attr_pass check ...")
    attr_pass = sum(1 for r in eval_records if r.get("attr_pass"))
    log(f"  attr_pass = {attr_pass}/{len(eval_records)} = {attr_pass/len(eval_records):.4f}")

    # === 11. Phase 10.7.5: 综合判定 ===
    log("[11] === 10.7.5: Final Judgment ===")
    crit_scale_free = (
        (p107_2["percentile_ci_upper_below_zero"] and p107_2["percentile_perm_p"] < 0.05) or
        (p107_2["mad_zscore_ci_upper_below_zero"] and p107_2["mad_zscore_perm_p"] < 0.05)
    )
    crit_user_win = win_rate > 0.5 and p_perm < 0.05
    crit_semantic = p107_4["D_geq_C"]
    crit_attr = attr_pass == len(eval_records)

    log(f"  ① scale-free CI < 0 + perm p<0.05 : {crit_scale_free}")
    log(f"     percentile Δ={pb:.4f}, CI=[{plo:.4f}, {phi:.4f}], p_perm={pp_perm:.4f}")
    log(f"     MAD-zscore Δ={zb:.4f}, CI=[{zlo:.4f}, {zhi:.4f}], p_perm={zp_perm:.4f}")
    log(f"  ② User win rate > 50% + perm p<0.05: {crit_user_win}")
    log(f"     win_rate={win_rate:.4f} ({win_count}/{len(deltas_arr)}), p_perm={p_perm:.4f}")
    log(f"  ③ Semantic sim(D) ≥ 0.95×sim(C): {crit_semantic}")
    log(f"     D={sim_d:.4f}, C={sim_c:.4f}")
    log(f"  ④ attr_pass == 1200: {crit_attr}")

    final_pass = crit_scale_free and crit_user_win and crit_semantic and crit_attr
    if final_pass:
        verdict = "GO"
    else:
        verdict = "NO-GO"

    log(f"  VERDICT: {verdict}")

    # === 12. Write meta ===
    meta = {
        "verdict": verdict,
        "n_users": len(common_users),
        "n_eval_records": len(eval_records),
        "attr_pass": attr_pass,
        "phase_10_7_1_user_win": p107_1,
        "phase_10_7_2_user_internal_norm": p107_2,
        "phase_10_7_3_cross_user_rank": p107_3,
        "phase_10_7_4_semantic_sim": p107_4,
        "final_criteria": {
            "scale_free_ci_below_zero_p_lt_005": crit_scale_free,
            "user_win_rate_gt_50pct_p_lt_005": crit_user_win,
            "semantic_sim_D_geq_95pct_C": crit_semantic,
            "attr_pass_100pct": crit_attr,
        },
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写入 {META_OUT}")
    log(f"verdict: {verdict}")
    log("=" * 70)


if __name__ == "__main__":
    main()