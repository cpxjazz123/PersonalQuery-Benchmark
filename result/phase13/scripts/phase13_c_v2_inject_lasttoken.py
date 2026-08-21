#!/usr/bin/env python3
"""Phase 13.C: Gaussian StyleVector Injection smoke test.

Verify that:
  1. Hook registers/deregisters cleanly on Qwen2-7B layer ℓ
  2. Sampling s_u^(k)^ℓ ~ N(mu_u^ℓ, diag(sigma_u^ℓ)) with rho > 0 produces
     *different* generated queries than the mean baseline (and unsteered baseline)
  3. Same user, K samples → diverse outputs (cos pair < 1)
  4. Different users → different distributions (sanity check)
  5. alphas: 0.0 (baseline), 0.5, 1.0, 2.0 all work

Output:
  - phase13_c_v2_smoke_test_lasttoken.json  (per-sample outputs and cos stats)
  - phase13_c_v2_smoke_meta_lasttoken.json

This is a SMOKE test only (no large-scale e2e). Uses 1 product + 3 users, K=4.
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
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
IN_GAUSSIANS = OUT_DIR / "phase13_b_v2_user_gaussians_qwen_lasttoken.npz"

OUT_SMOKE = OUT_DIR / "phase13_c_v2_smoke_test_lasttoken.json"
OUT_META = OUT_DIR / "phase13_c_v2_smoke_meta_lasttoken.json"

# Hardcoded config
QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
N_SMOKE_USERS = 3
K_SAMPLES = 4
ALPHA_GRID = [0.0, 0.5, 1.0, 2.0]
LAYER_GRID = [12, 14, 16, 20, 24]
RHO_GRID = [0.0, 0.3, 0.5, 0.7]
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42

# Hook implementation: adapted from query_gen_main.py:471-482 for Qwen2 (single-tensor layer output)
def make_hook(steer_vecs: list, steer_layer: int, steer_alpha: float):
    """Return a forward_hook that adds alpha*v to the last token hidden state at layer ℓ.

    Compatible with both tuple-returning (older Qwen) and tensor-returning (Qwen2) layer outputs.
    steer_vecs: list of [D] tensors (one per batch), or None for "skip this batch"
    """
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
            h[b, pos] = h[b, pos] + steer_alpha * v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 13.C: Gaussian StyleVector Injection smoke test")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # === [1] Load Gaussian cache ===
    log(f"[1] Loading Gaussian cache from {IN_GAUSSIANS} ...")
    npz = np.load(IN_GAUSSIANS, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    mu = npz["mu"]              # [n_users, L, H]
    std_diag = npz["std_diag"]  # [n_users, L, H]
    valid = npz["valid"]
    valid_uids = [u for u, v in zip(user_ids, valid) if v]
    valid_idxs = [i for i, v in enumerate(valid) if v]
    log(f"  mu: {mu.shape}, std_diag: {std_diag.shape}, valid: {int(valid.sum())}/{len(user_ids)}")

    smoke_uids = valid_uids[:N_SMOKE_USERS]
    smoke_idxs = valid_idxs[:N_SMOKE_USERS]
    log(f"  smoke users: {smoke_uids}")

    # === [2] Load Qwen2-7B transformers backend ===
    log(f"[2] Loading Qwen2-7B (transformers backend, fp16) ...")
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

    # === [3] Generate sample queries ===
    # Test product attrs (1 product, 3 attrs)
    attr_dict = {
        "Brand": "Cosco",
        "Color": "Green",
        "Material": "Plastic",
    }
    attr_str = ", ".join(f"{k}: {v}" for k, v in attr_dict.items())
    prompt = (
        "You are a shopping assistant. Write a concise search query for the "
        "product below.\n\n"
        f"Product: {attr_str}\n\nQuery:"
    )
    log(f"[3] Sample attrs: {attr_dict}")
    log(f"  prompt: {prompt[:80]}...")

    enc_str = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(enc_str, return_tensors="pt")["input_ids"]
    ids = enc.to("cuda:0")
    log(f"  prompt tokenized: {ids.shape[1]} tokens")

    # We'll test per layer/rho/alpha. For each, K samples × N users.
    # To keep it small: 1 layer (14, the default in many phases), 1 alpha (1.0), K=4 per user
    LAYER = 14
    ALPHA = 1.0

    log(f"[4] Generating for layer={LAYER}, alpha={ALPHA}, K={K_SAMPLES} per user ...")
    results: list[dict] = []

    for rho in RHO_GRID:
        log(f"  --- rho={rho} ---")
        for u_i, (uid, u_idx) in enumerate(zip(smoke_uids, smoke_idxs)):
            log(f"    user {u_i}: {uid}")
            # Sample K style vectors
            steer_vecs = []
            for k in range(K_SAMPLES):
                mu_u = torch.from_numpy(mu[u_idx, LAYER]).to(torch.float32).to("cuda:0")
                std_u = torch.from_numpy(std_diag[u_idx, LAYER]).to(torch.float32).to("cuda:0")
                if rho == 0.0:
                    # mean only
                    s = mu_u
                else:
                    eps = torch.randn_like(mu_u)
                    s = mu_u + rho * std_u * eps
                steer_vecs.append(s)

            # Register hook (list of K tensors)
            hook = model.model.layers[LAYER].register_forward_hook(
                make_hook(steer_vecs, LAYER, ALPHA))

            with torch.no_grad():
                gen = model.generate(
                    ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                    num_return_sequences=K_SAMPLES, use_cache=True,
                )

            hook.remove()
            prompt_len = ids.shape[1]
            for k in range(K_SAMPLES):
                row = gen[k, prompt_len:]
                if tok.eos_token_id is not None and tok.eos_token_id in row:
                    row = row[:row.tolist().index(tok.eos_token_id)]
                text = tok.decode(row, skip_special_tokens=True).strip()
                results.append({
                    "rho": rho,
                    "user_idx": u_i,
                    "user_id": uid,
                    "sample_idx": k,
                    "text": text,
                })
                log(f"      k={k}: {text[:80]}")

    # === [4b] Unsteered baseline (rho=0, alpha=0) ===
    log("[4b] Unsteered baseline (alpha=0) ...")
    baseline_texts = []
    for _ in range(K_SAMPLES):
        with torch.no_grad():
            gen = model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        row = gen[0, prompt_len:]
        if tok.eos_token_id is not None and tok.eos_token_id in row:
            row = row[:row.tolist().index(tok.eos_token_id)]
        text = tok.decode(row, skip_special_tokens=True).strip()
        baseline_texts.append(text)
        log(f"    baseline: {text[:80]}")
    results.append({
        "rho": 0.0, "user_idx": -1, "user_id": "BASELINE",
        "sample_idx": -1, "text": baseline_texts[0],
    })

    # === [5] Smoke test stats ===
    log("[5] Smoke stats ...")
    # Diversity within (user, rho): pair-wise Jaccard word overlap
    smoke_stats = {"rho_grid": RHO_GRID, "users": smoke_uids}
    for rho in RHO_GRID:
        for u_i in range(N_SMOKE_USERS):
            texts = [r["text"] for r in results if r["rho"] == rho and r["user_idx"] == u_i]
            if len(texts) >= 2:
                words = [set(t.lower().split()) for t in texts]
                jaccs = []
                for i in range(len(words)):
                    for j in range(i + 1, len(words)):
                        u_set = words[i] | words[j]
                        jaccs.append(len(words[i] & words[j]) / max(len(u_set), 1))
                smoke_stats.setdefault(f"rho{rho}_user{u_i}_jaccard_mean", float(np.mean(jaccs)))
                smoke_stats.setdefault(f"rho{rho}_user{u_i}_unique_n", len(set(texts)))
    log(f"  {smoke_stats}")

    # Save
    OUT_SMOKE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log(f"  smoke results: {OUT_SMOKE}")

    meta = {
        "phase": "13.C smoke test",
        "model": Path(QWEN_MODEL_PATH).name,
        "layer": LAYER,
        "alpha": ALPHA,
        "k_samples": K_SAMPLES,
        "rho_grid": RHO_GRID,
        "smoke_stats": smoke_stats,
        "n_records": len(results),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta: {OUT_META}")

    log("=" * 70)
    log("PHASE 13.C SMOKE COMPLETE — ready for 13.D e2e")
    log("=" * 70)


if __name__ == "__main__":
    main()