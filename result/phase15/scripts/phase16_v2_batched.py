#!/usr/bin/env python3
"""Phase 16 v2 — batched multi-cond (5 remaining conds × K=8 in single generate call).

Appends to phase16_n30_gaussian_sweep.jsonl (already has 4 conds done).
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

HIDDEN_NPZ = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
OUT_GAUSS_N30 = OUT_DIR / "phase16_n30_user_gaussians_qwen.npz"
JSONL_OUT = OUT_DIR / "phase16_n30_gaussian_sweep.jsonl"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
LAYER = 14
N_FIT = 30
K = 8
N_PAIRS = 30
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 80
SEED = 42

# 仅 5 剩余 conds (D_off/A_a1.0/Ball_r0.3/0.5 已 done)
SWEEP = [
    ("Ball_r1.0",  "ball",   {"rho": 1.0}),
    ("Shrink_r0.5","shrink", {"rho": 0.5, "c": 0.3}),
    ("Shrink_r1.0","shrink", {"rho": 1.0, "c": 0.3}),
    ("Maha_a0.5",  "maha",   {"alpha": 0.5}),
    ("Maha_a1.0",  "maha",   {"alpha": 1.0}),
]

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)


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


def compute_steer(mu, sigma, mode, params):
    import torch
    if mode == "ball":
        rho = params["rho"]
        eps = torch.randn_like(mu)
        eps = eps / (eps.norm() + 1e-8)
        return mu + rho * sigma * eps
    if mode == "shrink":
        rho, c = params["rho"], params["c"]
        sigma_safe = torch.minimum(sigma, c * mu.abs())
        eps = torch.randn_like(mu)
        eps = eps / (eps.norm() + 1e-8)
        s = mu + rho * sigma_safe * eps
        max_norm = 2.0 * mu.norm()
        if s.norm() > max_norm:
            s = s * max_norm / s.norm()
        return s
    if mode == "maha":
        alpha = params["alpha"]
        sigma_norm = sigma / (sigma.mean() + 1e-8)
        confidence = 1.0 / (1.0 + sigma_norm)
        return alpha * mu * confidence
    raise ValueError(f"Unknown mode: {mode}")


def make_hook(steer_vecs: list):
    """steer_vecs: list of N_CONDS*K tensors [H]. Per-row injection at last token."""
    def _hook(module, args, output):
        import torch
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for b in range(h.size(0)):
            v = steer_vecs[b]
            if v is None:
                continue
            v_dev = v.to(h.device).to(h.dtype)
            h[b, -1] = h[b, -1] + v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def main() -> None:
    log("=" * 70)
    log("Phase 16 v2 batched multi-cond (5 remaining conds × K=8 = 40 samples/call)")
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

    # Re-fit Gaussian (cheap, CPU)
    log("[2] Loading cached Gaussian N=30 ...")
    g = np.load(OUT_GAUSS_N30, allow_pickle=True)
    user_gauss = {}
    for uid, mu, sd in zip(g["user_ids"], g["mu"], g["sigma_diag"]):
        user_gauss[str(uid)] = {"mu": mu, "sigma_diag": sd}
    log(f"  users with Gaussian: {len(user_gauss)}")

    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  filtered pairs: {len(pairs)}")

    log("[3] Loading Qwen ...")
    os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tok = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}")

    # 注册 hook (单次, all conds share same hook)
    log("[4] Registering hook on layer 14 ...")
    steer_vecs: list = []  # N_CONDS*K elements, per batch row
    handle = model.model.layers[LAYER].register_forward_hook(make_hook(steer_vecs))

    log("[5] Batched multi-cond sweep ...")
    all_results = []
    n_conds = len(SWEEP)
    rows_per_call = n_conds * K  # 40
    log(f"  conds: {[c[0] for c in SWEEP]}, K={K}, rows_per_call={rows_per_call}")

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
            mu = torch.tensor(user_gauss[uid]["mu"], dtype=torch.float32, device=device)
            sigma = torch.tensor(user_gauss[uid]["sigma_diag"], dtype=torch.float32, device=device)

            # 算 steer: 5 conds × K=8 samples (不同 seed for randomness)
            steer_vecs.clear()
            for cond_name, mode, params in SWEEP:
                for _ in range(K):
                    steer_vecs.append(compute_steer(mu, sigma, mode, params))

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            row = 0
            for cond_name, mode, params in SWEEP:
                for k in range(K):
                    gen_ids = out[row][input_len:]
                    gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    all_results.append({
                        "user_id": uid,
                        "asin": asin,
                        "attrs": attrs,
                        "condition": cond_name,
                        "mode": mode,
                        "params": params,
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
        handle.remove()

    log(f"[6] Total records: {len(all_results)} (expected {len(pairs) * n_conds * K})")
    log(f"  sweep total: {time.time() - t_start:.1f}s")

    # Append to jsonl
    log(f"[7] Appending to {JSONL_OUT}")
    with JSONL_OUT.open("a") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  done. final jsonl size: {JSONL_OUT.stat().st_size:,} bytes")

    log("=" * 70)
    log("PHASE 16 v2 BATCHED DONE — run eval next")
    log("=" * 70)


if __name__ == "__main__":
    main()