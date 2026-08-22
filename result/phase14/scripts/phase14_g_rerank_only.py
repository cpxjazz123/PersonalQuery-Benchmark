#!/usr/bin/env python3
"""Phase 14.G rerank-only: load existing JSONL + encode + rerank + length + eval."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

JSONL_IN = OUT_DIR / "phase14_g_attrs_sweep.jsonl"
HIDDEN_NPZ = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
CONDS = ["D_off", "A14_a1.0"]
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
LW_SHRINK = 0.1
MIN_VAR = 1e-4

STATS_OUT = OUT_DIR / "phase14_g_attrs_sweep_length_stats.json"
EVAL_OUT = OUT_DIR / "phase14_g_attrs_sweep_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_g_attrs_sweep_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_g_attrs_sweep_meta.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("[1] Load JSONL rows ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            all_results.append(json.loads(line))
    log(f"  rows: {len(all_results)}")

    log("[2] Length stats ...")
    stats = {}
    for n in N_VALUES:
        stats[n] = {}
        for cond in CONDS:
            queries = [r["q_final_post"] for r in all_results
                       if r["n_attrs"] == n and r["condition"] == cond]
            char_lens = [len(q) for q in queries]
            word_lens = [len(q.split()) for q in queries]
            stats[n][cond] = {
                "n": len(queries),
                "char_mean": float(np.mean(char_lens)),
                "char_median": float(np.median(char_lens)),
                "char_std": float(np.std(char_lens)),
                "char_min": int(np.min(char_lens)),
                "char_max": int(np.max(char_lens)),
                "word_mean": float(np.mean(word_lens)),
                "word_median": float(np.median(word_lens)),
                "word_std": float(np.std(word_lens)),
            }
    STATS_OUT.write_text(json.dumps(stats, indent=2))
    log(f"  saved → {STATS_OUT}")

    log("[3] Load user Gaussians for layer 26 ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in LAYERS_5}
    user_gauss = {}
    for ui, uid in enumerate(cached_user_ids):
        user_gauss[uid] = {}
        for li, layer in enumerate(LAYERS_5):
            layer_h = user_hiddens_arr[ui, :, li, :]
            residual = layer_h - neutral_per_layer[layer][None, :]
            user_gauss[uid][layer] = {"mu": residual.mean(axis=0), "sigma_diag": residual.std(axis=0, ddof=0)}
    log(f"  users: {len(user_gauss)}")

    log("[4] Encode candidates at layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    cand_qs = [r["q_final_post"] for r in all_results]
    hiddens = client.get_hidden_states(cand_qs, [RERANK_LAYER])
    cand_hiddens = hiddens[RERANK_LAYER]
    log(f"  cand_hiddens shape: {cand_hiddens.shape}")

    log("[5] Compute residuals ...")
    neutral_layer = neutral_per_layer[RERANK_LAYER]
    cand_residuals = cand_hiddens - neutral_layer[None, :]
    log(f"  residual norm: {np.linalg.norm(cand_residuals, axis=-1).mean():.2f}")

    log("[6] Build per-user mu/var for layer 26 ...")
    user_uid_list = list(user_gauss.keys())
    user_mu_26 = np.stack([user_gauss[uid][RERANK_LAYER]["mu"] for uid in user_uid_list])
    user_var_26 = np.stack([user_gauss[uid][RERANK_LAYER]["sigma_diag"] ** 2 for uid in user_uid_list])
    user_var_26 = user_var_26 + LW_SHRINK
    user_var_26 = np.maximum(user_var_26, MIN_VAR)
    pooled_var_26 = user_var_26.mean(axis=0)
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}
    log(f"  user_mu_26 shape: {user_mu_26.shape}, pooled_var norm: {np.linalg.norm(pooled_var_26):.2f}")

    log("[7] Rerank per (N, cond) ...")
    by_n_cond = defaultdict(list)
    for r_idx, r in enumerate(all_results):
        by_n_cond[(r["n_attrs"], r["condition"])].append((r_idx, r["user_id"]))

    per_pair_results = []
    eval_summary = {}
    for n in N_VALUES:
        eval_summary[n] = {}
        for cond in CONDS:
            entries = by_n_cond[(n, cond)]
            by_pair = defaultdict(list)
            for c_idx, uid in entries:
                asin = all_results[c_idx]["asin"]
                by_pair[(uid, asin)].append(c_idx)
            ranks = []
            rank1 = 0
            for (uid, asin), c_idx_list in by_pair.items():
                if uid not in uid_to_idx:
                    continue
                target_idx = uid_to_idx[uid]
                local_resid = cand_residuals[c_idx_list]
                diff = local_resid[:, None, :] - user_mu_26[None, :, :]
                maha = (diff ** 2 / pooled_var_26[None, None, :]).sum(axis=-1)
                best_rank = min((maha[c] < maha[c, target_idx]).sum() for c in range(len(c_idx_list)))
                ranks.append(best_rank)
                if best_rank == 0:
                    rank1 += 1
                per_pair_results.append({
                    "n_attrs": n,
                    "condition": cond,
                    "user_id": uid,
                    "asin": asin,
                    "rank": int(best_rank),
                    "mean_rank_per_cand": float(maha[:, target_idx].mean()),
                })
            ranks = np.array(ranks)
            eval_summary[n][cond] = {
                "n": len(ranks),
                "rank1": int(rank1),
                "rank1_pct": float(rank1 / len(ranks)) if len(ranks) > 0 else 0.0,
                "top10": int((ranks < 10).sum()),
                "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "top100": int((ranks < 100).sum()),
                "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
            }
    log(f"  eval summary:")
    print(json.dumps(eval_summary, indent=2))
    EVAL_OUT.write_text(json.dumps(eval_summary, indent=2))

    with PER_PAIR_OUT.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.G",
        "n_values": N_VALUES,
        "conds": CONDS,
        "k_per_pair": 8,
        "n_pairs": 30,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "n_total": len(all_results),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.G RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
