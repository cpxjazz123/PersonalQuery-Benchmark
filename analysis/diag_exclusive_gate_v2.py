#!/usr/bin/env python3
"""Stage 8 exclusive-gate 实测诊断 v2。

不再需要 spaCy 重编码 candidate。直接读
  /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz
的 z_val (876922 句子, 32d) — 这是 user 的真实 query-like 分布,
作为天然 candidate 池。对每个 ASIN 的 cohort,在 z_val 上重算
target_inside / competitor_inside / pass_unique,按 cohort_size 分桶
统计 competitor_inside.sum(axis=1) 分布。

回答问题:"cohort_size↑ ⇒ exclusive gate 通过率↓" 在真实数据上
是什么形状,K=0/1/2 ablation 能否显著恢复 selection。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
EMB_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz")
OUT_PATH = REPO_ROOT / "result/05_gaussian_audit/exclusive_gate_overlap.json"

# 抽样规模
SAMPLE_N_ASINS_PER_BUCKET = 80
COHORT_BUCKETS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== diag v2: Stage 8 exclusive gate cross-overlap ===")

    # --- Load Stage 04 ---
    with open(STAGE04_PATH) as f:
        s4 = json.load(f)
    users_all_raw = s4["users"]
    cohort_gates = s4["cohort_gates"]

    # 转 numpy 结构 (跟 Stage 8 _gauss_from_stats 一致)
    users_all: Dict[str, dict] = {}
    for uid, st in users_all_raw.items():
        mu = np.asarray(st["mu"], dtype=np.float32)
        inv = np.asarray(st["sigma_inv"], dtype=np.float32)
        if mu.shape != (32,) or inv.shape != (32, 32):
            continue
        users_all[uid] = {
            "mu": mu,
            "inv_sigma": inv,
            "gate_T": float(st["d2_q95"]),
            "n": int(st["n"]),
            "n_val": int(st["n_val"]),
        }
    log(f"  Stage 04 users: {len(users_all)}")

    # --- Load frozen val embeddings (天然 candidate 池) ---
    emb = np.load(EMB_PATH, allow_pickle=False)
    z_val = emb["z_val"].astype(np.float32)              # (876922, 32)
    val_idx = emb["val_idx"]                              # (876922,) -> uid_list index
    uid_list = emb["uid_list"]
    log(f"  z_val: {z_val.shape}  unique users in val: {len(np.unique(val_idx))}")

    # --- 按 cohort_size 分桶 ---
    bucket_def = [
        ("cs=2", 2, 2), ("cs=3", 3, 3), ("cs=4", 4, 4), ("cs=5", 5, 5),
        ("cs=6-7", 6, 7), ("cs=8-9", 8, 9), ("cs=10-14", 10, 14),
        ("cs=15-19", 15, 19), ("cs=20-29", 20, 29),
        ("cs=30-49", 30, 49), ("cs=50-99", 50, 99), ("cs>=100", 100, 10**9),
    ]
    sized = {name: [] for name, _, _ in bucket_def}
    for asin, cohort in cohort_gates.items():
        if not isinstance(cohort, dict):
            continue
        fitted = [u for u in cohort.keys() if u in users_all]
        cs = len(fitted)
        for name, lo, hi in bucket_def:
            if lo <= cs <= hi:
                sized[name].append((asin, fitted))
                break
    log("  cohort_size distribution (fitted uids per ASIN, Stage 4 fitted cohort):")
    for name, _, _ in bucket_def:
        log(f"    {name}: {len(sized[name])} ASINs")

    rng = np.random.default_rng(42)
    sample: Dict[str, List[Tuple[str, List[str]]]] = {}
    for name, _, _ in bucket_def:
        pool = sized[name]
        n = min(SAMPLE_N_ASINS_PER_BUCKET, len(pool))
        if n == 0:
            continue
        idx = rng.choice(len(pool), size=n, replace=False)
        sample[name] = [pool[i] for i in idx]
    n_sample_asins = sum(len(v) for v in sample.values())
    log(f"  sampled: {n_sample_asins} ASINs across buckets")

    # --- 主诊断:对每 ASIN × 全体 z_val 算 d2_matrix,然后 gate ---
    bucket_stats: Dict[str, dict] = {}
    for name, asin_list in sample.items():
        log(f"  === bucket {name} ({len(asin_list)} ASINs) ===")
        # 累计 candidate-level 统计
        per_cand_competitor_hits: List[int] = []
        per_cand_target_inside: List[int] = []
        per_cand_pass_unique_K0: List[int] = []   # ~any (K=0)
        per_cand_pass_unique_K1: List[int] = []   # sum <= 1
        per_cand_pass_unique_K2: List[int] = []   # sum <= 2
        per_cand_pass_unique_K5: List[int] = []   # sum <= 5
        per_target_pass_gate: List[int] = []
        per_target_n_cands: List[int] = []
        per_target_pass_unique_K0: List[int] = []
        per_target_pass_unique_K1: List[int] = []
        per_target_pass_unique_K2: List[int] = []
        per_asin_unique_users_K0: Dict[str, int] = {}
        per_asin_unique_users_K1: Dict[str, int] = {}
        per_asin_unique_users_K2: Dict[str, int] = {}
        n_total_cands = 0
        n_total_targets = 0
        t0 = time.time()
        # 抽样 z_val 一部分以加速:每 ASIN 用 2000 个 candidate
        N_CAN_PER_ASIN = 2000
        rng2 = np.random.default_rng(123)
        for asin, fitted_uids in asin_list:
            # 用全体 z_val 的随机子集当 candidate
            cand_idx = rng2.choice(z_val.shape[0], size=min(N_CAN_PER_ASIN, z_val.shape[0]),
                                   replace=False)
            Z_q = z_val[cand_idx]                          # (n_can, 32)
            mu = np.stack([users_all[u]["mu"] for u in fitted_uids], axis=0)        # (cs, 32)
            inv = np.stack([users_all[u]["inv_sigma"] for u in fitted_uids], axis=0)  # (cs, 32, 32)
            gate_Ts = np.asarray([users_all[u]["gate_T"] for u in fitted_uids], dtype=np.float64)
            diff = Z_q[:, None, :] - mu[None, :, :]      # (n_can, cs, 32)
            left = np.einsum("cud,ude->cue", diff, inv)
            d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)  # (n_can, cs)

            # 对每个 target uid 算 unique mask (K=0/1/2)
            for ti, uid in enumerate(fitted_uids):
                target_d2 = d2_matrix[:, ti]
                gate_T = float(gate_Ts[ti])
                target_inside = target_d2 <= gate_T
                competitor_inside = d2_matrix <= gate_Ts[None, :]
                competitor_inside[:, ti] = False
                n_hits = competitor_inside.sum(axis=1).astype(np.int32)
                # K=0: ~any
                k0 = target_inside & (n_hits == 0)
                # K=1: sum <= 1
                k1 = target_inside & (n_hits <= 1)
                # K=2: sum <= 2
                k2 = target_inside & (n_hits <= 2)
                # K=5: sum <= 5
                k5 = target_inside & (n_hits <= 5)
                per_cand_competitor_hits.extend(int(x) for x in n_hits)
                per_cand_target_inside.extend(int(x) for x in target_inside)
                per_cand_pass_unique_K0.extend(int(x) for x in k0)
                per_cand_pass_unique_K1.extend(int(x) for x in k1)
                per_cand_pass_unique_K2.extend(int(x) for x in k2)
                per_cand_pass_unique_K5.extend(int(x) for x in k5)
                per_target_pass_gate.append(int(target_inside.sum()))
                per_target_n_cands.append(len(Z_q))
                per_target_pass_unique_K0.append(int(k0.sum()))
                per_target_pass_unique_K1.append(int(k1.sum()))
                per_target_pass_unique_K2.append(int(k2.sum()))
                n_total_cands += len(Z_q)
                n_total_targets += 1
                per_asin_unique_users_K0.setdefault(asin, 0)
                per_asin_unique_users_K1.setdefault(asin, 0)
                per_asin_unique_users_K2.setdefault(asin, 0)
                if k0.any():
                    per_asin_unique_users_K0[asin] += 1
                if k1.any():
                    per_asin_unique_users_K1[asin] += 1
                if k2.any():
                    per_asin_unique_users_K2[asin] += 1
            n_total_cands += 0  # 上面累计过了
        log(f"    computed in {time.time()-t0:.1f}s "
            f"({n_total_targets} target-users, {n_total_cands} candidates)")

        hits = np.asarray(per_cand_competitor_hits, dtype=np.int32)
        n = len(hits)
        max_k = int(hits.max()) if n else 0
        k_dist = {int(k): int((hits == k).sum()) for k in range(max_k + 1)}
        cum = {}
        running = 0
        for k in range(max_k + 1):
            running += k_dist.get(k, 0)
            cum[k] = running

        n_pass_gate = sum(per_target_pass_gate)
        n_pass_K0 = sum(per_cand_pass_unique_K0)
        n_pass_K1 = sum(per_cand_pass_unique_K1)
        n_pass_K2 = sum(per_cand_pass_unique_K2)
        n_pass_K5 = sum(per_cand_pass_unique_K5)
        n_target_unique_K0 = sum(1 for v in per_asin_unique_users_K0.values() if v >= 1)
        n_target_unique_K1 = sum(1 for v in per_asin_unique_users_K1.values() if v >= 1)
        n_target_unique_K2 = sum(1 for v in per_asin_unique_users_K2.values() if v >= 1)
        n_target_ge2_K0 = sum(1 for v in per_asin_unique_users_K0.values() if v >= 2)
        n_target_ge2_K1 = sum(1 for v in per_asin_unique_users_K1.values() if v >= 2)
        n_target_ge2_K2 = sum(1 for v in per_asin_unique_users_K2.values() if v >= 2)

        bucket_stats[name] = {
            "n_asins_sampled": len(asin_list),
            "n_target_users": n_total_targets,
            "n_candidates": n_total_cands,
            "n_pass_gate": n_pass_gate,
            "pass_gate_rate": n_pass_gate / max(1, n_total_cands),
            "n_pass_unique_K0": n_pass_K0,
            "n_pass_unique_K1": n_pass_K1,
            "n_pass_unique_K2": n_pass_K2,
            "n_pass_unique_K5": n_pass_K5,
            "pass_unique_rate_K0": n_pass_K0 / max(1, n_total_cands),
            "pass_unique_rate_K1": n_pass_K1 / max(1, n_total_cands),
            "pass_unique_rate_K2": n_pass_K2 / max(1, n_total_cands),
            "pass_unique_rate_K5": n_pass_K5 / max(1, n_total_cands),
            "n_target_with_any_unique_K0": n_target_unique_K0,
            "n_target_with_any_unique_K1": n_target_unique_K1,
            "n_target_with_any_unique_K2": n_target_unique_K2,
            "n_target_ge2_unique_users_K0": n_target_ge2_K0,
            "n_target_ge2_unique_users_K1": n_target_ge2_K1,
            "n_target_ge2_unique_users_K2": n_target_ge2_K2,
            "competitor_hit_dist": k_dist,
            "competitor_hit_cum": cum,
            "p_competitor_hit_eq_0": k_dist.get(0, 0) / max(1, n),
            "p_competitor_hit_le_1": cum.get(1, 0) / max(1, n),
            "p_competitor_hit_le_2": cum.get(2, 0) / max(1, n),
            "p_competitor_hit_le_5": cum.get(5, 0) / max(1, n),
            "p_competitor_hit_le_10": cum.get(10, 0) / max(1, n),
        }
        log(f"    pass_gate_rate={bucket_stats[name]['pass_gate_rate']:.4f}")
        log(f"    pass_unique_rate: K0={bucket_stats[name]['pass_unique_rate_K0']:.6f}  "
            f"K1={bucket_stats[name]['pass_unique_rate_K1']:.6f}  "
            f"K2={bucket_stats[name]['pass_unique_rate_K2']:.6f}  "
            f"K5={bucket_stats[name]['pass_unique_rate_K5']:.6f}")
        log(f"    P(competitor_hits=0)={bucket_stats[name]['p_competitor_hit_eq_0']:.4f}  "
            f"<=1={bucket_stats[name]['p_competitor_hit_le_1']:.4f}  "
            f"<=2={bucket_stats[name]['p_competitor_hit_le_2']:.4f}")
        log(f"    ≥2 unique users: K0={n_target_ge2_K0}/{len(asin_list)}  "
            f"K1={n_target_ge2_K1}/{len(asin_list)}  K2={n_target_ge2_K2}/{len(asin_list)}")

    out = {
        "config": {
            "candidate_source": "z_val (876922 held-out user sents, 32d)",
            "n_candidate_per_asin": 2000,
            "sample_n_per_bucket": SAMPLE_N_ASINS_PER_BUCKET,
            "buckets": [n for n, _, _ in bucket_def],
            "stage04_source": str(STAGE04_PATH),
            "embeddings_source": str(EMB_PATH),
            "gate_quantile": "d2_q95 per user (Stage 04 raw full-Σ)",
        },
        "buckets": bucket_stats,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  wrote → {OUT_PATH}")


if __name__ == "__main__":
    main()
