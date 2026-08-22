#!/usr/bin/env python3
"""Phase 14.H: K=30 generation for distribution-to-distribution rerank.

Generate 30 pair × 2 conds (D_off, A14_a1.0) × K=30 = 1800 queries.
Use Phase 14.B's 30 pair (all with cached user, same attrs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

# Inputs (reuse Phase 14.B pairs + attrs pool)
PAIRS_FILE = OUT_DIR / "phase14_b_pairs.jsonl"
ATTRS_POOL = OUT_DIR / "phase14_g_attrs_pool.json"
HIDDEN_NPZ = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"

# Sweep
N_PAIRS = 30
K_PER_PAIR = 30
LAYERS_5 = [8, 14, 18, 22, 26]
CONDS = [
    ("D_off", None, 0.0),
    ("A14_a1.0", 14, 1.0),
]
SEED = 42

# Generation
BATCH_SIZE = 32  # 2 conds × K=30 = 60 rows per pair → keep batch manageable
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_PROMPT_LEN = 256

# Outputs
JSONL_OUT = OUT_DIR / "phase14_h_k30_generations.jsonl"
META_OUT = OUT_DIR / "phase14_h_k30_meta.json"

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
    """Hard-copy fallback: append missing attrs to ensure completeness."""
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
    log("[1] Loading pairs + attrs pool ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")
    pool = json.loads(ATTRS_POOL.read_text())

    log("[2] Loading user Gaussians (per layer) ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    user_hiddens_arr = h_d["hiddens"].astype(np.float32)
    layers_cache = list(h_d["layers"])
    assert layers_cache == LAYERS_5, f"layer mismatch: {layers_cache} vs {LAYERS_5}"
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
    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  pairs after filter: {len(pairs)}")

    log("[3] Loading Qwen client ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    tok = client._hidden_backend.tokenizer
    model = client._hidden_backend.model
    device = model.device

    log("[4] Generating queries (K=30) ...")
    n_conds = len(CONDS)
    rows_per_call = n_conds * K_PER_PAIR  # 60 rows per pair

    cond_layer = [c[1] for c in CONDS]
    cond_alpha = [c[2] for c in CONDS]
    cond_is_off = [c[0] == "D_off" for c in CONDS]
    layers_used = sorted({layer for layer in cond_layer if layer is not None})

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

    all_results = []
    t_start = time.time()

    try:
        for i, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            # use the SAME attrs as Phase 14.B (5 attrs - real_5)
            attrs = pool[asin]["real_5"]
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
            if (i + 1) % 5 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                log(f"    pair {i + 1}/{len(pairs)} ({elapsed:.1f}s, ETA {eta:.1f}s)")
                # Save partial
                with JSONL_OUT.open("w") as f:
                    for r in all_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for h in handles:
            h.remove()

    log(f"  total: {len(all_results)} rows in {time.time()-t_start:.1f}s")
    with JSONL_OUT.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  saved → {JSONL_OUT}")

    META_OUT.write_text(json.dumps({
        "phase": "14.H",
        "k_per_pair": K_PER_PAIR,
        "n_pairs": N_PAIRS,
        "conds": [c[0] for c in CONDS],
        "n_total": len(all_results),
        "n_attrs": 5,
    }, indent=2))


if __name__ == "__main__":
    main()