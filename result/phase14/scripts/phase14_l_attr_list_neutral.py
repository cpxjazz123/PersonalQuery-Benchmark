#!/usr/bin/env python3
"""Phase 14.L: Attribute list as TRUE neutral query.

Idea: Instead of using LLM-rewrite or D_off as "neutral" query, generate a TRUE neutral
format: comma-separated attribute list with no sentence structure.
  Example: "BabyBond, 4.93 pounds, 16.2\"D x 13.24\"W x 13.61\"H, Green, Silicone"

Hypothesis: This format has no syntactic structure, so the diff (A22_hidden - attr_list_hidden)
should have a norm comparable to user_offset norm (~71).

Pipeline:
  1. Load 30 pairs from phase14_b_pairs (attrs available)
  2. For each pair, K=8 attribute list candidates via LLM (very strict prompt)
  3. Encode attribute lists at layer 26
  4. Compute q_offset = A22_hidden - attr_list_hidden (per-pair mean+var)
  5. Compare with user_offset = user_orig - user_paired_neutral (Phase 14.F-paired cache)
  6. Diagnostic: norm balance q vs user
  7. Rerank 4 dist2dist metrics + best-of-K Maha baseline

If norm balanced (q_norm ≈ user_norm ~70), style offset might work for the first time.
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
PAIRS_IN = OUT_DIR / "phase14_b_pairs.jsonl"
A22_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
CAND_RESID = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"
HIDDEN_NPZ_USER = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
HIDDEN_NPZ_USER_NEUTRAL = OUT_DIR / "phase14_f_paired_user_neutral_hiddens.npz"

# Outputs
OUT_ATTR_LIST_CACHE = OUT_DIR / "phase14_l_attr_list_cache.jsonl"
OUT_ATTR_LIST_HIDDENS = OUT_DIR / "phase14_l_attr_list_hiddens.npy"

EVAL_OUT = OUT_DIR / "phase14_l_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_l_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_l_meta.json"

COND = "A22_a0.5"
N_PAIRS = 30
K_PER_PAIR = 8
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
N_L = LAYERS_5.index(RERANK_LAYER)
LW_SHRINK = 0.1
MIN_VAR = 1e-4

# Strict prompt: only attribute values, comma-separated, no sentences
ATTR_LIST_PROMPT = """Output ONLY a comma-separated list of the following attribute values, in the exact same order. No sentences, no verbs, no articles, no connectors. Just the values, comma-separated.

Example: "Value1, Value2, Value3, Value4, Value5"

Attributes:
{attrs}

Output:"""
ATTR_LIST_TEMP = 0.3
ATTR_LIST_BATCH = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 14.L: Attribute list as TRUE neutral query")
    log("=" * 70)

    log("[1] Load 30 pairs ...")
    pairs = []
    with PAIRS_IN.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    log("[2] Build attr_list prompts (30 × K=8 = 240 prompts) ...")
    prompts = []
    pair_idx_per_prompt = []
    for pi, p in enumerate(pairs[:N_PAIRS]):
        attrs_text = "\n".join(f"{k}: {v}" for k, v in p["attrs"].items())
        prompt = ATTR_LIST_PROMPT.format(attrs=attrs_text)
        for k in range(K_PER_PAIR):
            prompts.append(prompt)
            pair_idx_per_prompt.append(pi)
    log(f"  prompts: {len(prompts)}")

    log("[3] Generate attribute list candidates via LLM (batched) ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    import torch
    tokenizer = client._hidden_backend.tokenizer
    model = client._hidden_backend.model

    # Batched generation
    outputs = []
    t0 = time.time()
    for st in range(0, len(prompts), ATTR_LIST_BATCH):
        chunk = prompts[st:st + ATTR_LIST_BATCH]
        msgs_list = [[{"role": "user", "content": p}] for p in chunk]
        formatted = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs_list
        ]
        enc = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        do_sample = ATTR_LIST_TEMP > 0
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=80,
                do_sample=do_sample, temperature=ATTR_LIST_TEMP if do_sample else 1.0,
                top_p=0.95 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for d in decoded:
            outputs.append(d.strip().split("\n")[0])  # take first line only
        if st % (ATTR_LIST_BATCH * 5) == 0 or st + ATTR_LIST_BATCH >= len(prompts):
            log(f"    generated {min(st + ATTR_LIST_BATCH, len(prompts))}/{len(prompts)} ({time.time()-t0:.1f}s)")

    log(f"  total generated: {len(outputs)} ({time.time()-t0:.1f}s)")

    # Save raw cache
    with OUT_ATTR_LIST_CACHE.open("w") as f:
        for i, (p_idx, out) in enumerate(zip(pair_idx_per_prompt, outputs)):
            f.write(json.dumps({
                "pair_idx": p_idx,
                "user_id": pairs[p_idx]["user_id"],
                "asin": pairs[p_idx]["asin"],
                "attrs": pairs[p_idx]["attrs"],
                "attr_list": out,
            }, ensure_ascii=False) + "\n")
    log(f"  cached → {OUT_ATTR_LIST_CACHE}")

    log("[4] Sample 5 generated attr_lists ...")
    import random
    random.seed(42)
    sample_idx = random.sample(range(len(outputs)), 5)
    for idx in sample_idx:
        log(f"  [{len(outputs[idx]):3d}c] {outputs[idx][:200]}")

    log("[5] Encode attr_lists at layer 26 ...")
    h_dict = client.get_hidden_states(outputs, [RERANK_LAYER], batch_size=16, max_length=256)
    attr_list_h = h_dict[RERANK_LAYER]  # [240, 3584]
    log(f"  attr_list_h shape: {attr_list_h.shape}, norm mean: {np.linalg.norm(attr_list_h, axis=1).mean():.2f}")
    np.save(OUT_ATTR_LIST_HIDDENS, attr_list_h.astype(np.float32))

    log("[6] Load A22_a0.5 hiddens from cand_residuals cache ...")
    cand_residuals_all = np.load(CAND_RESID).astype(np.float32)  # [720, 5, 3584]
    cand_residuals = cand_residuals_all[:, N_L, :]

    a22_results = []
    with A22_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == COND:
                a22_results.append(r)
    log(f"  A22 rows: {len(a22_results)}")

    # Build mapping by (uid, asin)
    a22_q_to_pair_idx = {}
    for i, r in enumerate(a22_results):
        key = (r["user_id"], r["asin"])
        a22_q_to_pair_idx.setdefault(key, []).append(i)

    log("[7] Build q_offset per pair (A22 - attr_list) ...")
    # For each pair, K A22 cands + K attr_lists
    pair_offsets = {}
    for i, p in enumerate(pairs[:N_PAIRS]):
        key = (p["user_id"], p["asin"])
        if key not in a22_q_to_pair_idx:
            log(f"  WARN: pair {key} no A22 cands")
            continue
        a22_indices = a22_q_to_pair_idx[key]
        if len(a22_indices) != K_PER_PAIR:
            log(f"  WARN: pair {key} A22 count {len(a22_indices)} != {K}")
            continue
        attr_indices = [i * K_PER_PAIR + k for k in range(K_PER_PAIR)]
        a22_h = cand_residuals[a22_indices]  # [K, 3584]
        # attr_list_h is mean-pool of full Qwen embedding (not residual)
        # Need to add global neutral ref to match cand_residuals space
        from llm_client import create_qwen_local_client
        # Load global neutral
        n_d = np.load(OUT_DIR / "phase13_a_neutral_hiddens.npz", allow_pickle=True)
        neutral_vecs = n_d["vecs"].astype(np.float32)
        global_neutral_layer = neutral_vecs[:, RERANK_LAYER, :].mean(axis=0)

        # Compute attr_list residual = attr_list_h - global_neutral
        attr_resid = attr_list_h[attr_indices] - global_neutral_layer[None, :]
        offset = a22_h - attr_resid  # [K, 3584]
        pair_offsets[key] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
            "a22_residuals": a22_h - a22_h.mean(axis=0),
            "a22_mean": a22_h.mean(axis=0),
            "offset_samples": offset,
        }

    log(f"  pair_offsets: {len(pair_offsets)}")

    log("[8] Diagnostic: sample offset norms ...")
    for i, (key, po) in enumerate(pair_offsets.items()):
        if i >= 3:
            break
        sample_offset = po["offset_samples"]
        log(f"  pair {key[0][:8]}: offset norm mean={np.linalg.norm(sample_offset, axis=1).mean():.2f}")

    log("[9] Load user hiddens ...")
    h_d = np.load(HIDDEN_NPZ_USER, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    log(f"  user orig shape: {user_hiddens_arr.shape}")

    n_d = np.load(HIDDEN_NPZ_USER_NEUTRAL, allow_pickle=True)
    user_neutral_ids = list(n_d["user_ids"])
    user_neutral_hiddens = n_d["hiddens"].astype(np.float32)
    log(f"  user neutral shape: {user_neutral_hiddens.shape}")

    log("[10] Compute per-user style offset ...")
    user_offset_stats = {}
    for ui, uid in enumerate(cached_user_ids):
        sent_idx_arr = n_d["sent_idx"]
        this_user_neutral = []
        for si in range(30):
            for ni, uid_n in enumerate(user_neutral_ids):
                if uid_n == uid and int(sent_idx_arr[ni]) == si:
                    this_user_neutral.append(user_neutral_hiddens[ni, N_L, :])
                    break
        if len(this_user_neutral) != 30:
            continue
        this_user_neutral = np.stack(this_user_neutral)
        orig_layer_h = user_hiddens_arr[ui, :, N_L, :]
        offset = orig_layer_h - this_user_neutral
        user_offset_stats[uid] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
        }

    log("[11] Norm diagnostic (q vs user) ...")
    user_offset_norms = []
    for uid in cached_user_ids:
        if uid in user_offset_stats:
            user_offset_norms.append(np.linalg.norm(user_offset_stats[uid]["mu"]))
    q_offset_norms = []
    for key, po in pair_offsets.items():
        q_offset_norms.append(np.linalg.norm(po["mu"]))
    log(f"  user_offset norm: mean={np.mean(user_offset_norms):.2f}")
    log(f"  q_offset norm:   mean={np.mean(q_offset_norms):.2f}")
    log(f"  RATIO (q/user): mean={np.mean(q_offset_norms)/np.mean(user_offset_norms):.2f}x")

    log("[12] Compute pooled var ...")
    user_uid_list = sorted([u for u in cached_user_ids if u in user_offset_stats])
    user_mu_arr = np.stack([user_offset_stats[u]["mu"] for u in user_uid_list])
    user_var_arr = np.stack([user_offset_stats[u]["sigma_sq"] for u in user_uid_list]) + LW_SHRINK
    user_var_arr = np.maximum(user_var_arr, MIN_VAR)
    pooled_var_offset = user_var_arr.mean(axis=0)
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}

    log("[13] Rerank style_offset_attr_list + best_of_k_maha ...")
    metric_names = ["maha_pooled", "bhattacharyya", "w2", "symmetric_kl"]
    eval_results = {}
    per_pair_results = []

    # Method A: style_offset_attr_list
    rank_lists = {m: [] for m in metric_names}
    rank1_counts = {m: 0 for m in metric_names}
    for (uid, asin), po in pair_offsets.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        q_mu = po["mu"]
        q_var = po["sigma_sq"] + LW_SHRINK
        q_var = np.maximum(q_var, MIN_VAR)
        mu_diff = q_mu[None, :] - user_mu_arr
        maha_per_user = (mu_diff ** 2 / pooled_var_offset[None, :]).sum(axis=-1)
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
            "method": "style_offset_attr_list",
            "user_id": uid,
            "asin": asin,
            **ranks,
        })

    eval_results["style_offset_attr_list"] = {}
    for m in metric_names:
        ranks = np.array(rank_lists[m])
        eval_results["style_offset_attr_list"][m] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_counts[m]),
            "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    # Method B: best-of-K A22 single Maha (Phase 14.F baseline)
    rank_list = []
    rank1_count = 0
    user_orig_mu = np.stack([user_hiddens_arr[cached_user_ids.index(u), :, N_L, :].mean(axis=0)
                              for u in user_uid_list])
    user_orig_var = np.stack([user_hiddens_arr[cached_user_ids.index(u), :, N_L, :].var(axis=0, ddof=0)
                              for u in user_uid_list]) + LW_SHRINK
    user_orig_var = np.maximum(user_orig_var, MIN_VAR)
    pooled_var_orig = user_orig_var.mean(axis=0)

    for (uid, asin), po in pair_offsets.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        a22_resid_K = po["a22_residuals"]
        diff = a22_resid_K[:, None, :] - user_orig_mu[None, :, :]
        maha_K = (diff ** 2 / pooled_var_orig[None, None, :]).sum(axis=-1)
        best_rank = min((maha_K[c] < maha_K[c, target_idx]).sum() for c in range(K_PER_PAIR))
        rank_list.append(best_rank)
        if best_rank == 0:
            rank1_count += 1
        per_pair_results.append({
            "method": "best_of_k_maha_a22",
            "user_id": uid,
            "asin": asin,
            "maha_pooled": int(best_rank),
        })
    ranks = np.array(rank_list)
    eval_results["best_of_k_maha_a22"] = {
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
        "phase": "14.L",
        "method": "style_offset_attr_list (A22 - attribute_list as true neutral)",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "metrics": metric_names,
        "user_offset_norm_mean": float(np.mean(user_offset_norms)),
        "q_offset_norm_mean": float(np.mean(q_offset_norms)),
        "norm_ratio_q_over_user": float(np.mean(q_offset_norms) / np.mean(user_offset_norms)),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.L ATTR LIST TRUE NEUTRAL STYLE OFFSET COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()