#!/usr/bin/env python3
"""Phase 14.B: Layer × α grid sweep — verify if layer 14 is optimal.

Sweep layers ∈ {8, 14, 18, 22, 26} × α ∈ {0.5, 1.0} + D_off = 11 conds.
Batched multi-cond: 11 conds × K=8 = 88 rows per generate call.
Per-layer hooks (8/14/18/22/26) inject only rows belonging to that layer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# Inputs (reuse from phase16_n30)
HIDDEN_NPZ = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
OUT_GAUSS_N30 = OUT_DIR / "phase16_n30_user_gaussians_qwen.npz"
HIDDEN_CACHE_LAYER = OUT_DIR / "phase13_b_v2_user_hiddens_layer_cache.npz"  # NEW: all layers cached

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
N_FIT = 30
K = 8
N_PAIRS = 30
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 80
SEED = 42

# Sweep config: (cond_name, layer, alpha_or_None_for_off)
SWEEP = [
    ("D_off",     None, 0.0),
    ("A8_a0.5",    8,   0.5),
    ("A8_a1.0",    8,   1.0),
    ("A14_a0.5",  14,   0.5),
    ("A14_a1.0",  14,   1.0),
    ("A18_a0.5",  18,   0.5),
    ("A18_a1.0",  18,   1.0),
    ("A22_a0.5",  22,   0.5),
    ("A22_a1.0",  22,   1.0),
    ("A26_a0.5",  26,   0.5),
    ("A26_a1.0",  26,   1.0),
]

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)

# Outputs
JSONL_OUT = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
META_OUT = OUT_DIR / "phase14_b_meta.json"
USER_EMBS_LAYERS_OUT = OUT_DIR / "phase14_b_user_gaussians_qwen_all_layers.npz"  # user Gaussians for ALL layers


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_missing_attrs(q: str, attrs_str: str) -> str:
    q_low = q.lower()
    missing = []
    for kv in attrs_str.split(","):
        kv = kv.strip()
        if ":" not in kv:
            continue
        _, v = kv.split(":", 1)
        v = v.strip()
        if v and v.lower() not in q_low:
            missing.append(v)
    if missing:
        q = q.rstrip(".") + ", " + ", ".join(missing) + "."
    return q


def fit_user_gaussians_all_layers(cached_hiddens, cached_counts, cached_user_ids,
                                   neutral_vecs_per_layer, n_fit=N_FIT):
    """Fit (μ, σ_diag) Gaussian per user per layer, residual = h - neutral[layer]."""
    valid_set = {uid for uid, c in zip(cached_user_ids, cached_counts) if c >= n_fit}
    user_gauss = {}  # uid → {layer: {"mu", "sigma_diag"}}
    for uid, h, c in zip(cached_user_ids, cached_hiddens, cached_counts):
        if c < n_fit or uid not in valid_set:
            continue
        h_use = h[:n_fit].astype(np.float32)  # [N, L, H]
        if h_use.ndim == 2:
            # legacy mean-pool format [N, H]
            continue
        user_gauss[uid] = {}
        for layer in range(h_use.shape[1]):
            layer_h = h_use[:, layer, :]
            neutral = neutral_vecs_per_layer[layer]
            residual = layer_h - neutral[None, :]
            mu = residual.mean(axis=0)
            sigma = residual.std(axis=0, ddof=0)
            user_gauss[uid][layer] = {"mu": mu, "sigma_diag": sigma}
    return user_gauss


def main() -> None:
    log("=" * 70)
    log("Phase 14.B: Layer × α grid sweep (11 conds batched)")
    log("=" * 70)
    np.random.seed(SEED)

    log("[1] Loading pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    log("[2] Loading cached hiddens + neutral ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    cached_hiddens = list(h_d["hiddens"])
    cached_counts = list(h_d["counts"])
    log(f"  users: {len(cached_user_ids)}, max count: {max(cached_counts)}")

    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)  # [N, 28, 3584]
    neutral_per_layer = {layer: neutral_vecs[:, layer, :].mean(axis=0) for layer in range(28)}
    log(f"  neutral vec per layer computed (28 layers)")

    log("[3] Fitting user Gaussian per user per layer ...")
    user_gauss = fit_user_gaussians_all_layers(
        cached_hiddens, cached_counts, cached_user_ids, neutral_per_layer, N_FIT
    )
    log(f"  users with Gaussian ({len(user_gauss)})")

    # Cache
    sample_user = next(iter(user_gauss))
    log(f"  sample user {sample_user[:10]} has layers: {list(user_gauss[sample_user].keys())[:5]}...")

    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  filtered pairs: {len(pairs)}")

    log("[4] Loading Qwen ...")
    os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tok = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}, n_layers: {len(model.model.layers)}")

    log("[5] Registering per-layer hooks ...")
    # Register hooks on all layers used in SWEEP
    layers_used = sorted({spec[1] for spec in SWEEP if spec[1] is not None})
    log(f"  layers to hook: {layers_used}")

    n_conds = len(SWEEP)
    rows_per_call = n_conds * K  # 88

    # Build a fast lookup: for each cond_idx, which layer + alpha
    cond_layer = [spec[1] for spec in SWEEP]
    cond_alpha = [spec[2] for spec in SWEEP]
    cond_is_off = [spec[0] == "D_off" for spec in SWEEP]

    # steer_per_cond: list of tensors or None
    steer_per_cond: list = [None] * n_conds  # filled per pair in loop

    def make_layer_hook(layer_idx):
        def _hook(module, args, output):
            import torch
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            for cond_idx in range(n_conds):
                if cond_layer[cond_idx] != layer_idx:
                    continue
                v = steer_per_cond[cond_idx]
                if v is None:
                    continue
                v_dev = v.to(h.device).to(h.dtype)
                for k in range(K):
                    b = cond_idx * K + k
                    h[b, -1] = h[b, -1] + v_dev
            if is_tuple:
                return (h,) + output[1:]
            return h
        return _hook

    handles = []
    for layer_idx in layers_used:
        h = model.model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))
        handles.append(h)
    log(f"  hooks registered on layers {layers_used}")

    log("[6] Batched sweep ...")
    all_results = []
    t_start = time.time()
    try:
        for i, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
            prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
            prompts = [prompt] * rows_per_call

            import torch
            # Compute steer per cond
            for cond_idx, spec in enumerate(SWEEP):
                cond_name, layer_idx, alpha = spec
                if cond_is_off[cond_idx]:
                    steer_per_cond[cond_idx] = None
                else:
                    g = user_gauss[uid][layer_idx]
                    mu = torch.tensor(g["mu"], dtype=torch.float32, device=device)
                    steer_per_cond[cond_idx] = alpha * mu

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
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
            for cond_idx, (cond_name, layer_idx, alpha) in enumerate(SWEEP):
                for k in range(K):
                    gen_ids = out[row][input_len:]
                    gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    all_results.append({
                        "user_id": uid,
                        "asin": asin,
                        "attrs": attrs,
                        "condition": cond_name,
                        "layer": layer_idx,
                        "alpha": alpha,
                        "cand_local_idx": k,
                        "q_styled": gen_text,
                        "q_final_post": append_missing_attrs(gen_text, attrs_str),
                        "attrs_str": attrs_str,
                    })
                    row += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                log(f"    pair {i + 1}/{len(pairs)} ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")
    finally:
        for h in handles:
            h.remove()

    log(f"[7] Total records: {len(all_results)} (expected {len(pairs) * rows_per_call})")
    log(f"  sweep total: {time.time() - t_start:.1f}s")

    log(f"[8] Writing jsonl: {JSONL_OUT}")
    with JSONL_OUT.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.B",
        "n_configs": n_conds,
        "config_names": [c[0] for c in SWEEP],
        "layers": [c[1] for c in SWEEP],
        "alphas": [c[2] for c in SWEEP],
        "n_pairs": len(pairs),
        "K": K,
        "N_fit": N_FIT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_total_records": len(all_results),
        "n_user_gaussians": len(user_gauss),
        "rows_per_call": rows_per_call,
    }, indent=2, ensure_ascii=False))
    log(f"  meta: {META_OUT}")

    log("=" * 70)
    log("PHASE 14.B SWEEP DONE — run eval next")
    log("=" * 70)


if __name__ == "__main__":
    main()