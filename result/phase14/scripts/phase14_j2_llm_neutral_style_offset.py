#!/usr/bin/env python3
"""Phase 14.J2: Style-offset rerank using LLM neutral on both sides.

User said: query 的中性版也使用 LLM 改写 (instead of D_off).

Pipeline:
  1. Load K=30 candidates (Phase 14.H, 1800 rows = 30 pairs × 2 conds × K=30)
  2. Extract unique A14 queries (~870) and unique D_off queries (~869)
  3. LLM rewrite each → neutral (cache JSONL)
  4. Encode both styled and neutral at layer 26 (mean-pool)
  5. q_style_offset = a14_hidden - a14_neutral_hidden (per pair, mean+var over K=30)
  6. user_style_offset = user_orig_hidden - user_paired_neutral_hidden (Phase 14.F-paired cache)
  7. Rerank with 4 metrics (maha_pooled, bhattacharyya, w2, symmetric_kl)
  8. Compare with best-of-K Maha A14 baseline (Phase 14.F style)

Expected outcome:
  - Both offsets use LLM neutral (consistent magnitude)
  - q_offset norm should be similar to user_offset norm
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

# Outputs
OUT_A14_NEUTRAL_CACHE = OUT_DIR / "phase14_j2_a14_neutral_cache.jsonl"
OUT_A14_NEUTRAL_HIDDENS = OUT_DIR / "phase14_j2_a14_neutral_hiddens.npy"
OUT_A14_RESIDUALS = OUT_DIR / "phase14_j2_a14_residuals.npy"
OUT_A14_NEUTRAL_TEXT = OUT_DIR / "phase14_j2_a14_neutral_sents.jsonl"

EVAL_OUT = OUT_DIR / "phase14_j2_style_offset_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_j2_style_offset_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_j2_style_offset_meta.json"

N_PAIRS = 30
K_PER_PAIR = 30
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26
N_L = LAYERS_5.index(RERANK_LAYER)
LW_SHRINK = 0.1
MIN_VAR = 1e-4

# Rewrite LLM settings (matching phase14_f_paired)
REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following product search query in a plain, neutral, "
    "matter-of-fact style. Keep the exact same meaning and all facts/attributes, but remove personal tone, "
    "slang, exclamations, and emotional words. Output ONLY the rewritten query, nothing else."
)
REWRITE_MAX_NEW = 100
REWRITE_TEMP = 0.3
REWRITE_BATCH = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def batch_rewrite_transformers(client, sentences, batch_size=REWRITE_BATCH,
                              max_new=REWRITE_MAX_NEW, temp=REWRITE_TEMP):
    """Batched transformers generate rewrite. Returns list of neutral rewrites."""
    import torch
    model = client._hidden_backend.model
    tokenizer = client._hidden_backend.tokenizer

    msgs_list = []
    for s in sentences:
        msgs = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Query: {s}\nNeutral rewrite:"},
        ]
        msgs_list.append(msgs)
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs_list
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    do_sample = temp > 0
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=do_sample, temperature=temp if do_sample else 1.0,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    new_tokens = gen[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def main():
    log("=" * 70)
    log("Phase 14.J2: Style-offset rerank using LLM neutral on both sides")
    log("=" * 70)

    log("[1] Load K=30 candidates ...")
    all_results = []
    with JSONL_IN.open() as f:
        for line in f:
            all_results.append(json.loads(line))
    log(f"  rows: {len(all_results)}")

    log("[2] Extract unique A14 queries ...")
    a14_q_to_rows = defaultdict(list)
    for i, r in enumerate(all_results):
        if r["condition"] == "A14_a1.0":
            a14_q_to_rows[r["q_final_post"]].append(i)
    unique_a14 = list(a14_q_to_rows.keys())
    log(f"  unique A14: {len(unique_a14)} (from {sum(len(v) for v in a14_q_to_rows.values())} rows)")

    log("[3] LLM rewrite unique A14 → neutral (cache JSONL) ...")
    # Build cache: q → rewrite
    rewrites = {}
    if OUT_A14_NEUTRAL_CACHE.exists():
        with OUT_A14_NEUTRAL_CACHE.open() as f:
            for line in f:
                r = json.loads(line)
                rewrites[r["q_styled"]] = r["rewrite"]
        log(f"  loaded {len(rewrites)} cached rewrites")
    todo = [q for q in unique_a14 if q not in rewrites]
    log(f"  todo rewrites: {len(todo)}")

    if len(todo) > 0:
        from llm_client import create_qwen_local_client
        client = create_qwen_local_client(with_vllm=False)
        t0 = time.time()
        with OUT_A14_NEUTRAL_CACHE.open("a") as f:
            for st in range(0, len(todo), REWRITE_BATCH):
                chunk = todo[st:st + REWRITE_BATCH]
                try:
                    chunk_rewrites = batch_rewrite_transformers(client, chunk)
                except Exception as e:
                    log(f"  ERROR at chunk {st}: {e}")
                    raise
                for q, n in zip(chunk, chunk_rewrites):
                    rewrites[q] = n
                    f.write(json.dumps({"q_styled": q, "rewrite": n}, ensure_ascii=False) + "\n")
                f.flush()
                if st % (REWRITE_BATCH * 5) == 0 or st + REWRITE_BATCH >= len(todo):
                    log(f"    rewrites {min(st + REWRITE_BATCH, len(todo))}/{len(todo)} ({time.time()-t0:.1f}s)")
        log(f"  total rewrites: {len(rewrites)} ({time.time()-t0:.1f}s)")

    log("[4] Encode A14 + A14_neutral at layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    # Build paired list (A14 q, neutral q) — order matters
    pairs = [(q, rewrites[q]) for q in unique_a14]
    a14_styled = [p[0] for p in pairs]
    a14_neutral = [p[1] for p in pairs]

    # Encode both at layer 26
    h_styled = client.get_hidden_states(a14_styled, [RERANK_LAYER], batch_size=32, max_length=256)
    h_neutral = client.get_hidden_states(a14_neutral, [RERANK_LAYER], batch_size=32, max_length=256)
    a14_h = h_styled[RERANK_LAYER]  # [N_unique, 3584]
    a14_neutral_h = h_neutral[RERANK_LAYER]
    log(f"  a14_h: {a14_h.shape}, a14_neutral_h: {a14_neutral_h.shape}")

    # Save
    np.save(OUT_A14_NEUTRAL_HIDDENS, a14_neutral_h.astype(np.float32))
    a14_residuals = a14_h - a14_neutral_h  # [N_unique, 3584]
    np.save(OUT_A14_RESIDUALS, a14_residuals.astype(np.float32))

    with OUT_A14_NEUTRAL_TEXT.open("w") as f:
        for q, n in pairs:
            f.write(json.dumps({"q_styled": q, "q_neutral": n}, ensure_ascii=False) + "\n")
    log(f"  saved hiddens + residuals")

    log("[5] Build q_offset per (uid, asin) from K=30 A14 + their neutral hiddens ...")
    # Group by pair: each row has (uid, asin) and q_final_post → look up a14_h[idx] and a14_neutral_h[idx]
    a14_q_to_idx = {q: i for i, q in enumerate(unique_a14)}
    pair_offsets = {}
    pair_resid_full = {}  # for best-of-K
    for r in all_results:
        if r["condition"] != "A14_a1.0":
            continue
        pair = (r["user_id"], r["asin"])
        idx = a14_q_to_idx[r["q_final_post"]]
        pair_offsets.setdefault(pair, {"a14_h": [], "neutral_h": []})
        pair_offsets[pair]["a14_h"].append(a14_h[idx])
        pair_offsets[pair]["neutral_h"].append(a14_neutral_h[idx])

    pair_stats = {}  # (uid, asin) -> {mu, sigma_sq, a14_residuals}
    for pair, v in pair_offsets.items():
        if len(v["a14_h"]) != K_PER_PAIR:
            continue
        a14_arr = np.stack(v["a14_h"])  # [K, 3584]
        neutral_arr = np.stack(v["neutral_h"])
        offset = a14_arr - neutral_arr  # [K, 3584]
        pair_stats[pair] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
            "a14_residuals": a14_arr - a14_arr.mean(axis=0),  # for absolute baseline
            "a14_mean": a14_arr.mean(axis=0),
            "offset_samples": offset,
        }
    log(f"  pair_stats: {len(pair_stats)}")

    log("[6] Compute offset norm magnitude for diagnostic ...")
    sample_pair = list(pair_stats.keys())[0]
    sample_offset = pair_stats[sample_pair]["offset_samples"]
    log(f"  sample pair {sample_pair[0][:8]} offset norm: mean={np.linalg.norm(sample_offset, axis=1).mean():.2f}")

    log("[7] Load user hiddens (original + paired neutral) ...")
    h_d = np.load(HIDDEN_NPZ_USER, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    log(f"  user orig shape: {user_hiddens_arr.shape}")

    n_d = np.load(HIDDEN_NPZ_USER_NEUTRAL, allow_pickle=True)
    user_neutral_ids = list(n_d["user_ids"])
    user_neutral_hiddens = n_d["hiddens"].astype(np.float32)
    log(f"  user neutral shape: {user_neutral_hiddens.shape}")

    log("[8] Compute per-user style offset (LLM neutral on both sides) ...")
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
        offset = orig_layer_h - this_user_neutral  # [30, 3584]
        user_offset_stats[uid] = {
            "mu": offset.mean(axis=0),
            "sigma_sq": offset.var(axis=0, ddof=0),
        }

    log("[9] Diagnostic: compare offset norms (LLM neutral on both sides) ...")
    user_offset_norms = []
    for uid in cached_user_ids:
        if uid in user_offset_stats:
            user_offset_norms.append(np.linalg.norm(user_offset_stats[uid]["mu"]))
    q_offset_norms = []
    for pair, v in pair_stats.items():
        q_offset_norms.append(np.linalg.norm(v["mu"]))
    log(f"  user_offset norm: mean={np.mean(user_offset_norms):.2f}, median={np.median(user_offset_norms):.2f}")
    log(f"  q_offset norm:   mean={np.mean(q_offset_norms):.2f}, median={np.median(q_offset_norms):.2f}")

    log("[10] Compute pooled var for style offset ...")
    user_uid_list = sorted([u for u in cached_user_ids if u in user_offset_stats])
    user_mu_arr = np.stack([user_offset_stats[u]["mu"] for u in user_uid_list])
    user_var_arr = np.stack([user_offset_stats[u]["sigma_sq"] for u in user_uid_list]) + LW_SHRINK
    user_var_arr = np.maximum(user_var_arr, MIN_VAR)
    pooled_var_offset = user_var_arr.mean(axis=0)
    log(f"  pooled_var_offset norm: {np.linalg.norm(pooled_var_offset):.2f}")
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}

    log("[11] Rerank ...")
    metric_names = ["maha_pooled", "bhattacharyya", "w2", "symmetric_kl"]
    eval_results = {}
    per_pair_results = []

    # Method A: style_offset LLM neutral
    rank_lists = {m: [] for m in metric_names}
    rank1_counts = {m: 0 for m in metric_names}
    for (uid, asin), ps in pair_stats.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        q_mu = ps["mu"]
        q_var = ps["sigma_sq"] + LW_SHRINK
        q_var = np.maximum(q_var, MIN_VAR)
        mu_diff = q_mu[None, :] - user_mu_arr  # [n_users, H]
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
            "method": "style_offset_llm",
            "user_id": uid,
            "asin": asin,
            **ranks,
        })

    eval_results["style_offset_llm"] = {}
    for m in metric_names:
        ranks = np.array(rank_lists[m])
        eval_results["style_offset_llm"][m] = {
            "n_pairs": len(ranks),
            "rank1": int(rank1_counts[m]),
            "rank1_pct": float(rank1_counts[m] / len(ranks)) if len(ranks) > 0 else 0.0,
            "top10": int((ranks < 10).sum()),
            "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "top100": int((ranks < 100).sum()),
            "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
            "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
        }

    # Method B: best-of-K A14 single-vector Maha (Phase 14.F style)
    rank_list = []
    rank1_count = 0
    user_orig_mu = np.stack([user_hiddens_arr[cached_user_ids.index(u), :, N_L, :].mean(axis=0)
                              for u in user_uid_list])
    user_orig_var = np.stack([user_hiddens_arr[cached_user_ids.index(u), :, N_L, :].var(axis=0, ddof=0)
                              for u in user_uid_list]) + LW_SHRINK
    user_orig_var = np.maximum(user_orig_var, MIN_VAR)
    pooled_var_orig = user_orig_var.mean(axis=0)

    for (uid, asin), ps in pair_stats.items():
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        # A14 mean → per-user single Maha, then best-of-K
        a14_resid_K = ps["a14_residuals"]  # [K, 3584]
        diff = a14_resid_K[:, None, :] - user_orig_mu[None, :, :]  # [K, n_users, H]
        maha_K = (diff ** 2 / pooled_var_orig[None, None, :]).sum(axis=-1)  # [K, n_users]
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
        "phase": "14.J2",
        "method": "style_offset = user(orig - LLM_neutral) vs query(A14 - LLM_neutral)",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "unique_a14": len(unique_a14),
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "metrics": metric_names,
        "user_offset_norm_mean": float(np.mean(user_offset_norms)),
        "q_offset_norm_mean": float(np.mean(q_offset_norms)),
    }, indent=2))
    log(f"  saved → {EVAL_OUT}, {PER_PAIR_OUT}, {META_OUT}")
    log("=" * 70)
    log("PHASE 14.J2 STYLE OFFSET (LLM NEUTRAL) RERANK COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()