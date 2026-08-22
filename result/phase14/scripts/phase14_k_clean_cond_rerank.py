#!/usr/bin/env python3
"""Phase 14.K: A22_a0.5 K=8 clean candidate comprehensive rerank.

Hypothesis: A14_a1.0 K=30 had severe generator degeneracy (18.3% loops/CJK mix/garbage).
A22_a0.5 K=8 has 0% degeneracy. Phase 14.F SOTA confirmed A22_a0.5 layer 26 + best-of-K Maha = 90% top-100.

Goal: re-run dist2dist and style offset on A22_a0.5 (clean) to verify if those paradigms
     work when generator is healthy (vs all NO-GO with A14_a1.0 K=30 generator degeneracy).

Pipeline:
  1. Load 720 cand residuals (Phase 14.F cache, 30 pair × K=8 × 3 conds)
  2. Filter to A22_a0.5 (240) + D_off (240) + A14_a1.0 (240) [all clean candidates]
  3. Run methods on each cond:
     - best_of_k_maha (Phase 14.F SOTA style)
     - maha_pooled (Phase 14.E dist2dist)
     - bhattacharyya
     - w2
     - symmetric_kl
  5. Compare across conds + best-of-K
  6. Also run style_offset (A22 - D_off) vs user (orig - LLM_neutral) → 4 metrics
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

JSONL_IN = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
CAND_RESID = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"
HIDDEN_NPZ_USER = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
HIDDEN_NPZ_USER_NEUTRAL = OUT_DIR / "phase14_f_paired_user_neutral_hiddens.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
N_L = LAYERS_5.index(RERANK_LAYER)
LW_SHRINK = 0.1
MIN_VAR = 1e-4
N_PAIRS = 30
K_PER_PAIR = 8

CONDS = ["D_off", "A22_a0.5", "A14_a1.0"]

EVAL_OUT = OUT_DIR / "phase14_k_clean_cond_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_k_clean_cond_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_k_clean_cond_meta.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 14.K: A22_a0.5 K=8 clean candidate comprehensive rerank")
    log("=" * 70)

    log("[1] Load K=8 candidates (720 = 30 pair × 8 K × 3 conds) ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in CONDS:
                all_results.append(r)
    cand_qs = [r["q_final_post"] for r in all_results]
    log(f"  rows: {len(all_results)}")

    log("[2] Load cand residuals (Qwen layer 26 residual) ...")
    cand_residuals_all = np.load(CAND_RESID).astype(np.float32)  # [720, 5, 3584]
    cand_residuals = cand_residuals_all[:, N_L, :]  # [720, 3584]
    log(f"  cand_residuals shape: {cand_residuals.shape}")

    log("[3] Load user hiddens (original) at layer 26 ...")
    h_d = np.load(HIDDEN_NPZ_USER, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    user_orig_layer_h = user_hiddens_arr[:, :, N_L, :]  # [198, 30, 3584]
    log(f"  user_orig shape: {user_orig_layer_h.shape}")

    log("[4] Load user paired neutral hiddens ...")
    n_d = np.load(HIDDEN_NPZ_USER_NEUTRAL, allow_pickle=True)
    user_neutral_ids = list(n_d["user_ids"])
    user_neutral_hiddens = n_d["hiddens"].astype(np.float32)[:, N_L, :]  # [5940, 3584]
    log(f"  user_neutral shape: {user_neutral_hiddens.shape}")

    log("[5] Compute per-user mu/var (original) at layer 26 ...")
    user_mu_per_cond = {}  # cond -> [n_users, 3584]
    user_var_per_cond = {}  # cond -> [n_users, 3584]
    for cond in ["original", "neutral"]:
        if cond == "original":
            h_arr = user_orig_layer_h
            ids = cached_user_ids
        else:
            # Build per-user neutral from paired cache
            sent_idx_arr = n_d["sent_idx"]
            h_dict = defaultdict(list)
            for i, uid in enumerate(user_neutral_ids):
                h_dict[uid].append(user_neutral_hiddens[i])
            ids = cached_user_ids
            h_arr = np.stack([np.stack(h_dict[u]) for u in ids])  # [n_users, 30, 3584]
        user_mu_per_cond[cond] = h_arr.mean(axis=1)
        user_var_per_cond[cond] = h_arr.var(axis=1, ddof=0) + LW_SHRINK
        user_var_per_cond[cond] = np.maximum(user_var_per_cond[cond], MIN_VAR)
    log(f"  user original_mu norm: {np.linalg.norm(user_mu_per_cond['original'], axis=1).mean():.2f}")
    log(f"  user neutral_mu norm: {np.linalg.norm(user_mu_per_cond['neutral'], axis=1).mean():.2f}")

    log("[6] Group cand residuals by cond ...")
    # all_results order = 3 conds × 30 pairs × 8 K = 720
    by_cond_pair = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(all_results):
        by_cond_pair[r["condition"]][(r["user_id"], r["asin"])].append(i)
    for cond in CONDS:
        n_pairs = len(by_cond_pair[cond])
        log(f"  {cond}: {n_pairs} pairs, total {sum(len(v) for v in by_cond_pair[cond].values())} cands")

    log("[7] Compute pooled var per cond ...")
    pooled_var_per_cond = {}
    for cond in ["original", "neutral"]:
        pooled_var_per_cond[cond] = user_var_per_cond[cond].mean(axis=0)

    log("[8] Run all methods ...")
    metric_names = ["maha_pooled", "bhattacharyya", "w2", "symmetric_kl"]
    eval_results = {}
    per_pair_results = []

    # ===== Method 1: best-of-K Maha per cond (Phase 14.F SOTA style) =====  #
    for cond in CONDS:
        rank_list = []
        rank1_count = 0
        for (uid, asin), idx_list in by_cond_pair[cond].items():
            uid_idx = cached_user_ids.index(uid) if uid in cached_user_ids else -1
            if uid_idx == -1:
                continue
            target_user = cached_user_ids[uid_idx]
            target_mu = user_mu_per_cond["original"][uid_idx]
            target_var = user_var_per_cond["original"][uid_idx]
            # K candidates residuals
            cand_K_resid = cand_residuals[idx_list]  # [K, 3584]
            # Compute pooled Maha per candidate
            diff = cand_K_resid[:, None, :] - user_mu_per_cond["original"][None, :, :]
            maha_K = (diff ** 2 / pooled_var_per_cond["original"][None, None, :]).sum(axis=-1)  # [K, n_users]
            target_idx = uid_idx
            best_rank = min((maha_K[c] < maha_K[c, target_idx]).sum() for c in range(len(idx_list)))
            rank_list.append(best_rank)
            if best_rank == 0:
                rank1_count += 1
            per_pair_results.append({
                "method": f"best_of_k_maha_{cond}",
                "user_id": uid,
                "asin": asin,
                "rank": int(best_rank),
            })
        ranks = np.array(rank_list)
        eval_results[f"best_of_k_maha_{cond}"] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_count),
            "rank1_pct": float(rank1_count / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    # ===== Method 2: dist2dist (4 metrics) per cond (Phase 14.E/H/I style) =====  #
    for cond in CONDS:
        rank_lists = {m: [] for m in metric_names}
        rank1_counts = {m: 0 for m in metric_names}
        for (uid, asin), idx_list in by_cond_pair[cond].items():
            uid_idx = cached_user_ids.index(uid) if uid in cached_user_ids else -1
            if uid_idx == -1:
                continue
            # cand distribution: K=8 residuals
            cand_K_resid = cand_residuals[idx_list]  # [K, 3584]
            q_mu = cand_K_resid.mean(axis=0)
            q_var = cand_K_resid.var(axis=0, ddof=0) + LW_SHRINK
            q_var = np.maximum(q_var, MIN_VAR)
            target_idx = uid_idx
            user_var_arr = user_var_per_cond["original"]
            user_mu_arr = user_mu_per_cond["original"]
            pooled_var = pooled_var_per_cond["original"]
            mu_diff = q_mu[None, :] - user_mu_arr
            maha_per_user = (mu_diff ** 2 / pooled_var[None, :]).sum(axis=-1)
            avg_var = (q_var[None, :] + user_var_arr) / 2
            geo_mean = np.sqrt(q_var[None, :] * user_var_arr + 1e-30)
            bc_per_dim = 0.25 * mu_diff ** 2 / (q_var[None, :] + user_var_arr) + 0.5 * np.log(avg_var / geo_mean)
            bc_per_user = bc_per_dim.sum(axis=-1)
            sigma_diff = np.sqrt(q_var[None, :]) - np.sqrt(user_var_arr)
            w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
            kl_pq = 0.5 * (np.log(user_var_arr / q_var[None, :] + 1e-30) +
                           q_var[None, :] / user_var_arr +
                           mu_diff ** 2 / user_var_arr - 1)
            kl_qp = 0.5 * (np.log(q_var[None, :] / user_var_arr + 1e-30) +
                           user_var_arr / q_var[None, :] +
                           mu_diff ** 2 / q_var[None, :] - 1)
            kl_per_user = (kl_pq + kl_qp).sum(axis=-1)
            ranks = {
                "maha_pooled": int((maha_per_user < maha_per_user[target_idx]).sum()),
                "bhattacharyya": int((bc_per_user < bc_per_user[target_idx]).sum()),
                "w2": int((w2_per_user < w2_per_user[target_idx]).sum()),
                "symmetric_kl": int((kl_per_user < kl_per_user[target_idx]).sum()),
            }
            for m, r in ranks.items():
                rank_lists[m].append(r)
                if r == 0:
                    rank1_counts[m] += 1
            per_pair_results.append({
                "method": f"dist2dist_{cond}",
                "user_id": uid,
                "asin": asin,
                **ranks,
            })
        eval_results[f"dist2dist_{cond}"] = {}
        for m in metric_names:
            ranks = np.array(rank_lists[m])
            eval_results[f"dist2dist_{cond}"][m] = {
                "n_pairs": len(ranks),
                "rank1": int(rank1_counts[m]),
                "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
                "top10": int((ranks < 10).sum()),
                "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "top100": int((ranks < 100).sum()),
                "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
            }

    # ===== Method 3: style offset (A22 - D_off) vs user (orig - LLM_neutral) =====  #
    rank_lists = {m: [] for m in metric_names}
    rank1_counts = {m: 0 for m in metric_names}
    # Compute per-pair A22 - D_off offset
    pair_offsets = {}
    for (uid, asin) in by_cond_pair["A22_a0.5"].keys():
        if (uid, asin) not in by_cond_pair["D_off"]:
            continue
        a22_idx = by_cond_pair["A22_a0.5"][(uid, asin)]
        d_off_idx = by_cond_pair["D_off"][(uid, asin)]
        if len(a22_idx) != K_PER_PAIR or len(d_off_idx) != K_PER_PAIR:
            continue
        a22_resid = cand_residuals[a22_idx]  # [K, 3584]
        d_off_resid = cand_residuals[d_off_idx]
        offset = a22_resid - d_off_resid  # [K, 3584]
        pair_offsets[(uid, asin)] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
        }

    # Compare with user offset = original - neutral
    user_offset_mu = user_mu_per_cond["original"] - user_mu_per_cond["neutral"]
    user_offset_var = (user_var_per_cond["original"] + user_var_per_cond["neutral"]) / 2 + LW_SHRINK
    user_offset_var = np.maximum(user_offset_var, MIN_VAR)
    pooled_var_offset = user_offset_var.mean(axis=0)
    log(f"  user_offset_mu norm: {np.linalg.norm(user_offset_mu, axis=1).mean():.2f}")

    for (uid, asin), po in pair_offsets.items():
        uid_idx = cached_user_ids.index(uid) if uid in cached_user_ids else -1
        if uid_idx == -1:
            continue
        target_idx = uid_idx
        q_mu = po["mu"]
        q_var = po["sigma_sq"] + LW_SHRINK
        q_var = np.maximum(q_var, MIN_VAR)
        mu_diff = q_mu[None, :] - user_offset_mu
        maha_per_user = (mu_diff ** 2 / pooled_var_offset[None, :]).sum(axis=-1)
        avg_var = (q_var[None, :] + user_offset_var) / 2
        geo_mean = np.sqrt(q_var[None, :] * user_offset_var + 1e-30)
        bc_per_dim = 0.25 * mu_diff ** 2 / (q_var[None, :] + user_offset_var) + 0.5 * np.log(avg_var / geo_mean)
        bc_per_user = bc_per_dim.sum(axis=-1)
        sigma_diff = np.sqrt(q_var[None, :]) - np.sqrt(user_offset_var)
        w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
        kl_pq = 0.5 * (np.log(user_offset_var / q_var[None, :] + 1e-30) +
                       q_var[None, :] / user_offset_var +
                       mu_diff ** 2 / user_offset_var - 1)
        kl_qp = 0.5 * (np.log(q_var[None, :] / user_offset_var + 1e-30) +
                       user_offset_var / q_var[None, :] +
                       mu_diff ** 2 / q_var[None, :] - 1)
        kl_per_user = (kl_pq + kl_qp).sum(axis=-1)
        ranks = {
            "maha_pooled": int((maha_per_user < maha_per_user[target_idx]).sum()),
            "bhattacharyya": int((bc_per_user < bc_per_user[target_idx]).sum()),
            "w2": int((w2_per_user < w2_per_user[target_idx]).sum()),
            "symmetric_kl": int((kl_per_user < kl_per_user[target_idx]).sum()),
        }
        for m, r in ranks.items():
            rank_lists[m].append(r)
            if r == 0:
                rank1_counts[m] += 1
        per_pair_results.append({
            "method": "style_offset_a22_doff",
            "user_id": uid,
            "asin": asin,
            **ranks,
        })

    eval_results["style_offset_a22_doff"] = {}
    for m in metric_names:
        ranks = np.array(rank_lists[m])
        eval_results["style_offset_a22_doff"][m] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_counts[m]),
            "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    log(f"  eval summary:")
    print(json.dumps(eval_results, indent=2))
    EVAL_OUT.write_text(json.dumps(eval_results, indent=2))
    with PER_PAIR_OUT.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.K",
        "method": "A22_a0.5 K=8 clean candidate comprehensive rerank (vs A14_a1.0 K=30 degenerate)",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "conds": CONDS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "metrics": metric_names,
        "n_cands_total": len(all_results),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.K CLEAN COND COMPREHENSIVE RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()