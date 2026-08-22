#!/usr/bin/env python3
"""Phase 14.G: n_attrs sweep (3..10) — generate queries + length + rerank.

Pipeline:
1. Load 30 pairs + attrs pool (with raw metadata attrs)
2. Load Qwen + StyleVector per-user per-layer Gaussian (reuse phase14_b cache)
3. Per (pair, N_attrs, cond): generate K=8 candidates with D_off and A14_a1.0
   - 3-5 attrs: random sample from real 5
   - 6-10 attrs: real 5 + (N-5) random from raw metadata
4. Length stats
5. Rerank (Qwen mean-pool residual, layer 26, pooled Maha)
6. Aggregate per (N, cond): rank-1, top-10, top-100, mean rank
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# Inputs
ATTRS_POOL = OUT_DIR / "phase14_g_attrs_pool.json"
PAIRS_FILE = OUT_DIR / "phase14_b_pairs.jsonl"
HIDDEN_NPZ = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

# Sweep
N_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]
CONDS = [
    ("D_off", None, 0.0),
    ("A14_a1.0", 14, 1.0),
]
K_PER_PAIR = 8
N_PAIRS = 30
LAYERS_5 = [8, 14, 18, 22, 26]
RERANK_LAYER = 26  # Phase 14.F SOTA layer
LW_SHRINK = 0.1
MIN_VAR = 1e-4
N_BOOTSTRAP = 2000
SEED = 42

# Generation
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_PROMPT_LEN = 256

# Outputs
JSONL_OUT = OUT_DIR / "phase14_g_attrs_sweep.jsonl"
STATS_OUT = OUT_DIR / "phase14_g_attrs_sweep_length_stats.json"
EVAL_OUT = OUT_DIR / "phase14_g_attrs_sweep_eval.json"
PER_PAIR_OUT = OUT_DIR / "phase14_g_attrs_sweep_per_pair.jsonl"
META_OUT = OUT_DIR / "phase14_g_attrs_sweep_meta.json"

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_missing_attrs(q: str, attrs_str: str) -> str:
    """Append missing attrs (hard-copy fallback)."""
    q_low = q.lower()
    missing = []
    for kv in attrs_str.split(","):
        kv = kv.strip()
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        v = v.strip()
        if v and v.lower() not in q_low:
            missing.append(f"{k}: {v}")
    if missing:
        q = q.rstrip(".") + ", " + ", ".join(missing) + "."
    return q


def main():
    # ============== [1] Load pairs + attrs pool ==============
    log("[1] Loading pairs + attrs pool ...")
    # Phase 14.B has 30 pairs all with cached users
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  pairs loaded: {len(pairs)}")
    pool = json.loads(ATTRS_POOL.read_text())
    log(f"  pool asin: {len(pool)}")

    # ============== [2] Load user Gaussians (per layer) ==============
    log("[2] Loading user Gaussians per layer ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    layers_cache = list(h_d["layers"])
    assert layers_cache == LAYERS_5, f"layer mismatch: {layers_cache} vs {LAYERS_5}"
    log(f"  users: {len(cached_user_ids)}, hiddens shape: {user_hiddens_arr.shape}, layers: {layers_cache}")
    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)  # [N, 28, 3584]
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in LAYERS_5}
    # Fit per-user per-layer Gaussian (use all 30 sentences)
    user_gauss = {}
    for ui, uid in enumerate(cached_user_ids):
        user_gauss[uid] = {}
        for li, layer in enumerate(LAYERS_5):
            layer_h = user_hiddens_arr[ui, :, li, :]  # [30, 3584]
            residual = layer_h - neutral_per_layer[layer][None, :]
            user_gauss[uid][layer] = {"mu": residual.mean(axis=0), "sigma_diag": residual.std(axis=0, ddof=0)}
    log(f"  users: {len(user_gauss)}")
    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  pairs after filter: {len(pairs)}")

    # ============== [3] Load Qwen ==============
    log("[3] Loading Qwen client (transformers, batched) ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    tok = client._hidden_backend.tokenizer
    model = client._hidden_backend.model
    device = model.device

    # ============== [4] Generate per (pair, N, cond) ==============
    import torch  # imported here so hook lambda can reference it
    log("[4] Generating queries ...")
    n_conds = len(CONDS)
    rows_per_call = n_conds * K_PER_PAIR * len(N_VALUES)
    # batched: 2 conds × 8 N × K=8 = 128 rows per call (may be too many — keep 1 N at a time)

    # Process one N at a time to keep batch manageable
    log(f"  conds: {[c[0] for c in CONDS]}, K={K_PER_PAIR}, N={N_VALUES}")
    all_results = []
    t_start = time.time()

    for n_idx, n in enumerate(N_VALUES):
        log(f"  === N={n} ===")
        # Per cond: hook layer for A14, none for D_off
        cond_layer = [c[1] for c in CONDS]
        cond_alpha = [c[2] for c in CONDS]
        cond_is_off = [c[0] == "D_off" for c in CONDS]
        layers_used = sorted({layer for layer in cond_layer if layer is not None})

        # Build per-cond steer (filled per pair)
        steer_per_cond = [None] * n_conds

        def make_layer_hook(layer_idx):
            def _hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                for cond_idx in range(n_conds):
                    if cond_layer[cond_idx] != layer_idx:
                        continue
                    v = steer_per_cond[cond_idx]
                    if v is None:
                        continue
                    v_dev = v.to(h.device).to(h.dtype)
                    for k in range(K_PER_PAIR):
                        b = cond_idx * K_PER_PAIR + k
                        h[b, -1] = h[b, -1] + v_dev
                if is_tuple:
                    return (h,) + output[1:]
                return h
            return _hook

        handles = []
        for layer_idx in layers_used:
            h = model.model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))
            handles.append(h)

        try:
            for i, pair in enumerate(pairs):
                uid = pair["user_id"]
                asin = pair["asin"]
                attrs = pool[asin]["by_n"][str(n)]
                attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
                prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
                prompts = [prompt] * rows_per_call

                # Compute steer per cond
                for cond_idx, (cond_name, layer_idx, alpha) in enumerate(CONDS):
                    if cond_is_off[cond_idx]:
                        steer_per_cond[cond_idx] = None
                    else:
                        g = user_gauss[uid][layer_idx]
                        mu = torch.tensor(g["mu"], dtype=torch.float32, device=device)
                        steer_per_cond[cond_idx] = alpha * mu

                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_PROMPT_LEN).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        pad_token_id=tok.pad_token_id or tok.eos_token,
                    )
                input_len = enc["input_ids"].shape[1]
                row = 0
                for cond_idx, (cond_name, layer_idx, alpha) in enumerate(CONDS):
                    for k in range(K_PER_PAIR):
                        gen_ids = out[row][input_len:]
                        gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                        all_results.append({
                            "user_id": uid,
                            "asin": asin,
                            "n_attrs": n,
                            "attrs": dict(attrs),
                            "attrs_str": attrs_str,
                            "condition": cond_name,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "cand_local_idx": k,
                            "q_styled": gen_text,
                            "q_final_post": append_missing_attrs(gen_text, attrs_str),
                        })
                        row += 1
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / (n_idx + 1) / (i + 1) * (len(N_VALUES) - n_idx - 1) * len(pairs) + elapsed / (n_idx + 1) * (len(pairs) - i - 1)
                    log(f"    N={n} pair {i + 1}/{len(pairs)} ({elapsed:.1f}s, ETA {eta:.1f}s)")
        finally:
            for h in handles:
                h.remove()
        # Save partial after each N
        with JSONL_OUT.open("w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"  saved partial → {JSONL_OUT} ({len(all_results)} rows)")

    log(f"  total generation: {len(all_results)} rows in {time.time()-t_start:.1f}s")

    # ============== [5] Length statistics ==============
    log("[5] Computing length statistics ...")
    stats = {}
    for n in N_VALUES:
        stats[n] = {}
        for cond_name, _, _ in CONDS:
            queries = [r["q_final_post"] for r in all_results
                       if r["n_attrs"] == n and r["condition"] == cond_name]
            char_lens = [len(q) for q in queries]
            word_lens = [len(q.split()) for q in queries]
            stats[n][cond_name] = {
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

    # ============== [6] Encode candidates (Qwen hidden layer 26) ==============
    log(f"[6] Encoding candidates at layer {RERANK_LAYER} ...")
    cand_qs = [r["q_final_post"] for r in all_results]
    from llm_client import create_qwen_local_client
    # reuse client
    hiddens = client.get_hidden_states(cand_qs, [RERANK_LAYER])
    # hiddens is dict[layer, ndarray [N, H]]
    cand_hiddens = hiddens[RERANK_LAYER]  # [N, H]
    log(f"  cand_hiddens shape: {cand_hiddens.shape}")

    # ============== [7] Load global neutral ref ==============
    log("[7] Loading global neutral ref ...")
    neutral_layer = neutral_per_layer[RERANK_LAYER]
    cand_residuals = cand_hiddens - neutral_layer[None, :]
    log(f"  global neutral ref shape: {neutral_layer.shape}, residual norm: {np.linalg.norm(cand_residuals, axis=-1).mean():.2f}")

    # ============== [8] Per-user Gaussian rerank (use results from [2]) ==============
    log(f"[8] Building per-user mu/var for layer {RERANK_LAYER} ...")
    H = 3584
    user_uid_list = list(user_gauss.keys())
    user_mu_26 = np.stack([user_gauss[uid][RERANK_LAYER]["mu"] for uid in user_uid_list])  # [n_users, H]
    user_var_26 = np.stack([user_gauss[uid][RERANK_LAYER]["sigma_diag"] ** 2 for uid in user_uid_list])  # [n_users, H]
    user_var_26 = user_var_26 + LW_SHRINK
    user_var_26 = np.maximum(user_var_26, MIN_VAR)
    pooled_var_26 = user_var_26.mean(axis=0)  # [H]
    uid_to_idx = {uid: i for i, uid in enumerate(user_uid_list)}
    log(f"  user_mu_26 shape: {user_mu_26.shape}, pooled_var norm: {np.linalg.norm(pooled_var_26):.2f}")

    # ============== [9] Rerank per (N, cond) ==============
    log("[9] Rerank per (N, cond) ...")
    # Group by (N, cond)
    by_n_cond = defaultdict(list)
    for r_idx, r in enumerate(all_results):
        by_n_cond[(r["n_attrs"], r["condition"])].append((r_idx, r["user_id"]))
    n_users = len(user_gauss)

    per_pair_results = []
    eval_summary = {}
    for n in N_VALUES:
        eval_summary[n] = {}
        for cond_name, _, _ in CONDS:
            entries = by_n_cond[(n, cond_name)]
            # Group by (uid, asin)
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
                # Compute Maha distance
                local_resid = cand_residuals[c_idx_list]  # [K, H]
                # diff: [K, n_users, H]
                diff = local_resid[:, None, :] - user_mu_26[None, :, :]
                # Maha: sum_d (diff^2 / pooled_var)
                maha = (diff ** 2 / pooled_var_26[None, None, :]).sum(axis=-1)  # [K, n_users]
                # For each candidate, rank of target
                best_rank = min((maha[c] < maha[c, target_idx]).sum() for c in range(len(c_idx_list)))
                ranks.append(best_rank)
                if best_rank == 0:
                    rank1 += 1
                per_pair_results.append({
                    "n_attrs": n,
                    "condition": cond_name,
                    "user_id": uid,
                    "asin": asin,
                    "rank": int(best_rank),
                    "mean_rank_per_cand": float(maha[:, target_idx].mean()),
                })
            ranks = np.array(ranks)
            eval_summary[n][cond_name] = {
                "n": len(ranks),
                "rank1": int(rank1),
                "rank1_pct": float(rank1 / len(ranks)) if len(ranks) > 0 else 0.0,
                "top10": int((ranks < 10).sum()),
                "top10_pct": float((ranks < 10).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "top100": int((ranks < 100).sum()),
                "top100_pct": float((ranks < 100).sum() / len(ranks)) if len(ranks) > 0 else 0.0,
                "mean_rank": float(ranks.mean()) if len(ranks) > 0 else 0.0,
            }
    log(f"  eval summary: {json.dumps(eval_summary, indent=2)}")
    EVAL_OUT.write_text(json.dumps(eval_summary, indent=2))
    log(f"  saved → {EVAL_OUT}")

    with PER_PAIR_OUT.open("w") as f:
        for r in per_pair_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  saved → {PER_PAIR_OUT}")

    META_OUT.write_text(json.dumps({
        "phase": "14.G",
        "n_values": N_VALUES,
        "conds": [c[0] for c in CONDS],
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "rerank_layer": RERANK_LAYER,
        "lw_shrink": LW_SHRINK,
        "min_var": MIN_VAR,
        "n_total": len(all_results),
    }, indent=2))
    log(f"  saved → {META_OUT}")
    log("=" * 70)
    log("PHASE 14.G COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    import torch
    main()
