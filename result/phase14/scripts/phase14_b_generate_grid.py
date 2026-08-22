#!/usr/bin/env python3
"""Phase 14.B: Layer × α grid sweep, generate candidates with Gaussian StyleVector injection.

User pivot from Phase 13.D layer=14 (NO-GO in 318d, GO in 768d with best-K margin
lift +0.098 but rank-1 still 0%): ACL StyleVector paper recommends layer ≥ 15
(middle-to-later layers). Phase 14.B sweeps layer × α grid at 768d style rerank.

Grid:
  LAYER_GRID = [17, 19, 21, 23, 25]  (5 mid-late layers, 60-89% of 28)
  ALPHA_GRID = [0.5, 1.0, 1.5]         (3 strengths)
  → 15 configs

Per config: 30 pairs × K=8 candidates, ρ=0.5 Gaussian sampled s_u^(k)
Output: phase14_b_<layer>_<alpha>.jsonl  (240 records each)

Reuses:
  - phase13_a_per_sentence_residuals_qwen.npz (all 28 layers available)
  - phase13_b_user_gaussians_qwen.npz (mu, sigma_diag, valid per layer)
  - phase10_pairs_1000.jsonl (30 pair subset)
  - phase10_19_b_contrastive_rag._append_missing_attrs (hard-copy)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_GAUSSIANS = OUT_DIR / "phase13_b_user_gaussians_qwen.npz"
IN_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_DIR_GEN = OUT_DIR / "phase14_b_gen"
OUT_DIR_GEN.mkdir(parents=True, exist_ok=True)

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Hardcoded grid config ===
N_PAIRS = 30
K_CANDIDATES = 8
LAYER_GRID = [17, 19, 21, 23, 25]
ALPHA_GRID = [0.5, 1.0, 1.5]
RHO = 0.5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_hook(steer_vecs, alpha):
    """Forward hook: h' = h + alpha*v at last token. Single-tensor output (Qwen2)."""
    def _hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        B = h.size(0)
        pos = h.size(1) - 1
        for b in range(B):
            v = steer_vecs[b]
            if v is None:
                continue
            v_dev = v.to(h.device).to(h.dtype)
            h[b, pos] = h[b, pos] + alpha * v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def main():
    log("=" * 70)
    log("Phase 14.B: Layer × α grid sweep, Gaussian StyleVector injection")
    log("=" * 70)
    log(f"  LAYER_GRID = {LAYER_GRID}")
    log(f"  ALPHA_GRID = {ALPHA_GRID}")
    log(f"  Total configs: {len(LAYER_GRID) * len(ALPHA_GRID)}")
    log(f"  Per config: {N_PAIRS} pairs × K={K_CANDIDATES} = {N_PAIRS * K_CANDIDATES}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # === [1] Load pairs and Gaussian cache ===
    log("[1] Loading pairs + Gaussian cache ...")
    pairs: list[dict] = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    npz = np.load(IN_GAUSSIANS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    mu = npz["mu"]               # (298, 28, 3584)
    std_diag = npz["std_diag"]   # (298, 28, 3584)
    valid = npz["valid"]
    valid_uids = [u for u, v in zip(user_ids_cache, valid) if v]
    valid_idxs = [i for i, v in enumerate(valid) if v]
    uid_to_gidx = {u: i for i, u in enumerate(user_ids_cache)}
    valid_set = set(valid_uids)
    log(f"  pairs total: {len(pairs)}, valid users in cache: {len(valid_uids)}")

    pairs = [p for p in pairs if p["user_id"] in valid_set][:N_PAIRS]
    log(f"  filtered pairs: {len(pairs)}")

    # === [2] Load Qwen2-7B ===
    log("[2] Loading Qwen2-7B ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault("QWEN_MODEL_PATH", QWEN_MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    log(f"  Qwen2-7B loaded, n_layers={n_layers}")

    # Hard-copy post-process
    sys.path.insert(0, str(OUT_DIR))
    from phase10_19_b_contrastive_rag import _append_missing_attrs

    # === [3] Generate grid ===
    configs = [(l, a) for l in LAYER_GRID for a in ALPHA_GRID]
    log(f"[3] Generating {len(configs)} configs × {len(pairs)} pairs × K={K_CANDIDATES} ...")
    t0_total = time.time()

    # Pre-tokenize all pair prompts
    pair_encs = []
    pair_attrs = []
    for p in pairs:
        uid = p["user_id"]
        attrs = p["attrs"]
        asin = p["asin"]
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
        prompt_text = (
            "You are a shopping assistant. Write a concise search query for the "
            "product below.\n\n"
            f"Product: {attr_str}\n\nQuery:"
        )
        enc = tok(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")
        pair_encs.append(enc)
        pair_attrs.append(attrs)

    for cfg_idx, (layer, alpha) in enumerate(configs):
        log(f"\n  --- cfg {cfg_idx + 1}/{len(configs)}: layer={layer}, alpha={alpha} ---")
        cfg_records = []
        t0 = time.time()

        for pi, p in enumerate(pairs):
            uid = p["user_id"]
            attrs = pair_attrs[pi]
            asin = p["asin"]
            u_gidx = uid_to_gidx[uid]
            enc = pair_encs[pi]
            prompt_len = enc.shape[1]

            # Build K=8 steer vectors (sampled from Gaussian at this layer)
            mu_u = torch.from_numpy(mu[u_gidx, layer]).to(torch.float32).to("cuda:0")
            std_u = torch.from_numpy(std_diag[u_gidx, layer]).to(torch.float32).to("cuda:0")
            steer_vecs = []
            for k in range(K_CANDIDATES):
                eps = torch.randn_like(mu_u)
                s = mu_u + RHO * std_u * eps
                steer_vecs.append(s)

            # Register hook
            hook_handle = model.model.layers[layer].register_forward_hook(
                make_hook(steer_vecs, alpha))

            try:
                with torch.no_grad():
                    enc_rep = enc.repeat(K_CANDIDATES, 1)
                    gen = model.generate(
                        enc_rep,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                        num_return_sequences=1, use_cache=True,
                    )
            finally:
                hook_handle.remove()

            # Decode K candidates
            for k in range(K_CANDIDATES):
                row = gen[k, prompt_len:]
                if tok.eos_token_id is not None and tok.eos_token_id in row:
                    row = row[:row.tolist().index(tok.eos_token_id)]
                q = tok.decode(row, skip_special_tokens=True).strip()
                q_final = _append_missing_attrs(q, attrs)
                cfg_records.append({
                    "pair_idx": pi,
                    "user_id": uid,
                    "asin": asin,
                    "attrs": attrs,
                    "layer": layer,
                    "alpha": alpha,
                    "cand_idx": k,
                    "q_styled": q,
                    "q_final_post": q_final,
                })

        cfg_elapsed = time.time() - t0

        # Save this config
        out_file = OUT_DIR_GEN / f"phase14_b_L{layer}_A{alpha:.1f}.jsonl"
        with out_file.open("w") as f:
            for r in cfg_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"    saved {len(cfg_records)} records → {out_file.name} in {cfg_elapsed:.1f}s")

    total_elapsed = time.time() - t0_total
    log(f"\n[4] Total elapsed: {total_elapsed:.1f}s")

    # === [4] Save meta ===
    meta = {
        "phase": "14.B generate",
        "model": Path(QWEN_MODEL_PATH).name,
        "n_pairs": len(pairs),
        "k_candidates": K_CANDIDATES,
        "layer_grid": LAYER_GRID,
        "alpha_grid": ALPHA_GRID,
        "rho": RHO,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_configs": len(configs),
        "records_per_config": N_PAIRS * K_CANDIDATES,
        "elapsed_s": round(total_elapsed, 1),
    }
    (OUT_DIR_GEN / "phase14_b_gen_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"  meta → phase14_b_gen_meta.json")

    log("=" * 70)
    log("PHASE 14.B GENERATION COMPLETE — ready for 14.B rerank")
    log("=" * 70)


if __name__ == "__main__":
    main()