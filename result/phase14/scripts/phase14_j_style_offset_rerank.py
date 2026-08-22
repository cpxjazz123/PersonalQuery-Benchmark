#!/usr/bin/env python3
"""Phase 14.J: Style-offset rerank.

User side (already have, Phase 14.F-paired):
  30 原评论 hidden - 30 neutral 改写 hidden = user_style_offset (mean + var per dim)
Query side (Phase 14.H K=30):
  K=30 A14_a1.0 query hidden - K=30 D_off query hidden = q_style_offset (mean + var)

Compare distributions of (user_style_offset) vs (q_style_offset) using:
- Maha_pooled
- Bhattacharyya
- W2
- Symmetric KL

Plus baselines:
- Absolute A14 dist vs user original dist
- Absolute D_off dist vs user original dist
- A14 mean - D_off mean vs user original - neutral mean (single vector Mahalanobis)
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

# Inputs
JSONL_IN = OUT_DIR / "phase14_h_k30_generations.jsonl"
HIDDEN_NPZ_USER = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
HIDDEN_NPZ_USER_NEUTRAL = OUT_DIR / "phase14_f_paired_user_neutral_hiddens.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

N_PAIRS = 30
K_PER_PAIR = 30
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
LW_SHRINK = 0.1
MIN_VAR = 1e-4
CONDS = ["D_off", "A14_a1.0"]

EVAL_OUT = OUT_DIR / "phase14_j_style_offset_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_j_style_offset_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_j_style_offset_meta.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("[1] Load K=30 candidates ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            all_results.append(json.loads(line))
    log(f"  rows: {len(all_results)}")

    log("[2] Load user hiddens (original) ...")
    h_d = np.load(HIDDEN_NPZ_USER, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    log(f"  shape: {user_hiddens_arr.shape}")

    log("[3] Load user neutral hiddens ...")
    n_d = np.load(HIDDEN_NPZ_USER_NEUTRAL, allow_pickle=True)
    user_neutral_ids = list(n_d["user_ids"])
    user_neutral_hiddens = n_d["hiddens"].astype(np.float32)  # [5940, 5, 3584]
    log(f"  shape: {user_neutral_hiddens.shape}")

    log("[4] Load global neutral ref ...")
    g_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = g_d["vecs"].astype(np.float32)
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in LAYERS_5}

    log("[5] Compute per-user style offset (original - neutral) ...")
    user_offset_stats = {}  # uid -> {mu, sigma_sq}
    n_l = LAYERS_5.index(RERANK_LAYER)
    for ui, uid in enumerate(cached_user_ids):
        # Find this user's neutral hiddens
        neutral_indices = [i for i, u in enumerate(user_neutral_ids) if u == uid]
        if len(neutral_indices) != 30:
            log(f"  WARN: uid {uid[:10]} neutral={len(neutral_indices)} (skip)")
            continue
        # Original hiddens
        orig_layer_h = user_hiddens_arr[ui, :, n_l, :]  # [30, 3584]
        # Neutral hiddens (find indices in user_neutral_hiddens that match this uid)
        # user_neutral_ids is parallel to user_neutral_hiddens: each row has a user_id
        # We need 30 neutral sentences per user, in same order as original
        # Phase 14.F-paired stores (uid, sent_idx) where sent_idx 0..29
        sent_idx_arr = n_d["sent_idx"]
        this_user_neutral = []
        for si in range(30):
            for ni, uid_n in enumerate(user_neutral_ids):
                if uid_n == uid and int(sent_idx_arr[ni]) == si:
                    this_user_neutral.append(user_neutral_hiddens[ni, n_l, :])
                    break
        if len(this_user_neutral) != 30:
            log(f"  WARN: uid {uid[:10]} neutral_hiddens={len(this_user_neutral)} (skip)")
            continue
        this_user_neutral = np.stack(this_user_neutral)  # [30, 3584]
        # Compute residuals
        orig_resid = orig_layer_h - neutral_per_layer[RERANK_LAYER][None, :]
        neutral_resid = this_user_neutral - neutral_per_layer[RERANK_LAYER][None, :]
        # Style offset = original - neutral (per sentence)
        offset = orig_resid - neutral_resid  # [30, 3584]
        user_offset_stats[uid] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
            "orig_mu": orig_resid.mean(axis=0),
            "neutral_mu": neutral_resid.mean(axis=0),
        }
    log(f"  user_offset_stats: {len(user_offset_stats)}")

    log("[6] Encode K=30 candidates at layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    cand_qs = [r["q_final_post"] for r in all_results]
    hiddens = client.get_hidden_states(cand_qs, [RERANK_LAYER])
    cand_hiddens = hiddens[RERANK_LAYER]
    cand_residuals_full = cand_hiddens - neutral_per_layer[RERANK_LAYER][None, :]
    log(f"  cand_residuals shape: {cand_residuals_full.shape}")

    log("[7] Group candidates by (cond, pair) ...")
    by_cond_pair = defaultdict(lambda: defaultdict(list))
    for r_idx, r in enumerate(all_results):
        by_cond_pair[r["condition"]][(r["user_id"], r["asin"])].append(r_idx)
    log(f"  conds: {list(by_cond_pair.keys())}")

    log("[8] Compute query style offset per pair ...")
    # For each pair: K A14 - K D_off → q_style_offset
    pair_offsets = {}  # (uid, asin) -> {mu, sigma_sq}
    for (uid, asin), idx_list in by_cond_pair["D_off"].items():
        d_off_indices = idx_list
        a14_indices = by_cond_pair["A14_a1.0"].get((uid, asin), [])
        if len(d_off_indices) != K_PER_PAIR or len(a14_indices) != K_PER_PAIR:
            continue
        d_off_resid = cand_residuals_full[d_off_indices]
        a14_resid = cand_residuals_full[a14_indices]
        # Style offset = A14 - D_off (per candidate pair)
        offset = a14_resid - d_off_resid  # [30, 3584]
        pair_offsets[(uid, asin)] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
            "a14_mu": a14_resid.mean(axis=0),
            "d_off_mu": d_off_resid.mean(axis=0),
            "a14_resid": a14_resid,
            "d_off_resid": d_off_resid,
        }
    log(f"  pair_offsets: {len(pair_offsets)}")

    log("[9] Compute pooled var for style offset ...")
    all_user_var = np.stack([user_offset_stats[u]["sigma_sq"] for u in cached_user_ids if u in user_offset_stats])
    all_user_var = all_user_var + LW_SHRINK
    all_user_var = np.maximum(all_user_var, MIN_VAR)
    pooled_var_offset = all_user_var.mean(axis=0)
    log(f"  pooled_var_offset norm: {np.linalg.norm(pooled_var_offset):.2f}")

    log("[10] Rerank per metric ...")
    user_uid_list = sorted([u for u in cached_user_ids if u in user_offset_stats])
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}
    user_mu_arr = np.stack([user_offset_stats[u]["mu"] for u in user_uid_list])
    user_var_arr = np.stack([user_offset_stats[u]["sigma_sq"] for u in user_uid_list]) + LW_SHRINK
    user_var_arr = np.maximum(user_var_arr, MIN_VAR)
    log(f"  user arrays: mu={user_mu_arr.shape}, var={user_var_arr.shape}")

    eval_results = {}
    per_pair_results = []

    # ===== Method 1: Style offset (paired) =====  #
    metric_names = ["maha_pooled", "bhattacharyya", "w2", "symmetric_kl"]
    rank_lists = {m: [] for m in metric_names}
    rank1_counts = {m: 0 for m in metric_names}
    for (uid, asin), q_off in pair_offsets.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        q_mu = q_off["mu"]
        q_var = q_off["sigma_sq"] + LW_SHRINK
        q_var = np.maximum(q_var, MIN_VAR)
        # Per-user distances
        mu_diff = q_mu[None, :] - user_mu_arr  # [n_users, H]
        # Maha
        maha_per_user = (mu_diff ** 2 / pooled_var_offset[None, :]).sum(axis=-1)
        # BC
        avg_var = (q_var[None, :] + user_var_arr) / 2
        geo_mean = np.sqrt(q_var[None, :] * user_var_arr + 1e-30)
        bc_per_dim = 0.25 * mu_diff ** 2 / (q_var[None, :] + user_var_arr) + 0.5 * np.log(avg_var / geo_mean)
        bc_per_user = bc_per_dim.sum(axis=-1)
        # W2
        sigma_diff = np.sqrt(q_var[None, :]) - np.sqrt(user_var_arr)
        w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
        # KL
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
            "method": "style_offset",
            "user_id": uid,
            "asin": asin,
            **ranks,
        })

    eval_results["style_offset"] = {}
    for m in metric_names:
        ranks = np.array(rank_lists[m])
        eval_results["style_offset"][m] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_counts[m]),
            "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    # ===== Method 2: Best-of-K A14 single-vector (Phase 14.F SOTA style) =====  #
    rank_lists = {m: [] for m in metric_names}
    rank1_counts = {m: 0 for m in metric_names}
    # For baseline: A14 mean per pair vs user original mean per user (Phase 14.H style)
    user_orig_mu = np.stack([user_offset_stats[u]["orig_mu"] for u in user_uid_list])
    user_orig_var = np.stack([user_offset_stats[u]["sigma_sq"] for u in user_uid_list]) + LW_SHRINK
    user_orig_var = np.maximum(user_orig_var, MIN_VAR)
    pooled_var_orig = user_orig_var.mean(axis=0)

    for (uid, asin), q_off in pair_offsets.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        q_mu = q_off["a14_mu"]  # A14 mean
        q_var = q_off["a14_resid"].var(axis=0, ddof=0) + LW_SHRINK
        q_var = np.maximum(q_var, MIN_VAR)
        mu_diff = q_mu[None, :] - user_orig_mu
        maha_per_user = (mu_diff ** 2 / pooled_var_orig[None, :]).sum(axis=-1)
        avg_var = (q_var[None, :] + user_orig_var) / 2
        geo_mean = np.sqrt(q_var[None, :] * user_orig_var + 1e-30)
        bc_per_dim = 0.25 * mu_diff ** 2 / (q_var[None, :] + user_orig_var) + 0.5 * np.log(avg_var / geo_mean)
        bc_per_user = bc_per_dim.sum(axis=-1)
        sigma_diff = np.sqrt(q_var[None, :]) - np.sqrt(user_orig_var)
        w2_per_user = (mu_diff ** 2).sum(axis=-1) + (sigma_diff ** 2).sum(axis=-1)
        kl_pq = 0.5 * (np.log(user_orig_var / q_var[None, :] + 1e-30) +
                       q_var[None, :] / user_orig_var +
                       mu_diff ** 2 / user_orig_var - 1)
        kl_qp = 0.5 * (np.log(q_var[None, :] / user_orig_var + 1e-30) +
                       user_orig_var / q_var[None, :] +
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
            "method": "absolute_a14",
            "user_id": uid,
            "asin": asin,
            **ranks,
        })

    eval_results["absolute_a14"] = {}
    for m in metric_names:
        ranks = np.array(rank_lists[m])
        eval_results["absolute_a14"][m] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_counts[m]),
            "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    # ===== Method 3: Best-of-K per-candidate A14 (Phase 14.F style) =====  #
    rank_list = []
    rank1_count = 0
    for (uid, asin), q_off in pair_offsets.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        # K A14 candidates × n_users Maha
        a14_resid_K = q_off["a14_resid"]  # [K, H]
        diff = a14_resid_K[:, None, :] - user_orig_mu[None, :, :]  # [K, n_users, H]
        maha_K = (diff ** 2 / pooled_var_orig[None, None, :]).sum(axis=-1)  # [K, n_users]
        # Best-of-K = min rank across K
        best_rank = min((maha_K[c] < maha_K[c, target_idx]).sum() for c in range(K_PER_PAIR))
        rank_list.append(best_rank)
        if best_rank == 0:
            rank1_count += 1
        per_pair_results.append({
            "method": "best_of_k_maha_a14",
            "user_id": uid,
            "asin": asin,
            "maha_pooled": int(best_rank),
        })
    ranks = np.array(rank_list)
    eval_results["best_of_k_maha_a14"] = {
        "n_pairs": len(ranks),
        "rank1": int(rank1_count),
        "rank1_pct": float(rank1_count / len(ranks)) if len(ranks) > 0 else 0.0,
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
        "phase": "14.J",
        "method": "style_offset = user(orig - neutral) vs query(A14 - D_off)",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "conds": CONDS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "metrics": metric_names,
        "n_total": len(all_results),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.J STYLE OFFSET RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()