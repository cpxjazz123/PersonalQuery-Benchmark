#!/usr/bin/env python3
"""Phase 16: 真正用 Gaussian steer 的 sweep (ball / shrink / Mahalanobis).

Step 1: 用 cached hiddens (phase13_b_v2_user_hiddens_n100.npz) + neutral hidden
        重拟合 N=30 Gaussian (μ, σ_diag) per user
Step 2: 加载 Qwen, sweep 6 conds × 30 pairs × K=8 = 1440 records
        D_off, A_a1.0, Ball-rho0.3, Ball-rho0.5, Shrink-rho0.5, Maha-a1.0
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

# Inputs
HIDDEN_NPZ = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
NEUTRAL_NPZ = OUT_DIR / "phase13_a_neutral_hiddens.npz"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
LAYER = 14
N_FIT = 30  # sweet spot from Phase 13.B-v2
K = 8
N_PAIRS = 30
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 80
SEED = 42

# Outputs
OUT_GAUSS_N30 = OUT_DIR / "phase16_n30_user_gaussians_qwen.npz"
JSONL_OUT = OUT_DIR / "phase16_n30_gaussian_sweep.jsonl"
META_OUT = OUT_DIR / "phase16_n30_meta.json"

# Steer 范式 sweep
SWEEP = [
    ("D_off",      "off",         {}),  # no hook, baseline
    ("A_a1.0",     "mu_only",     {"alpha": 1.0}),  # 主路线
    ("Ball_r0.3",  "ball",        {"rho": 0.3}),  # 方案1
    ("Ball_r0.5",  "ball",        {"rho": 0.5}),
    ("Ball_r1.0",  "ball",        {"rho": 1.0}),
    ("Shrink_r0.5","shrink",      {"rho": 0.5, "c": 0.3}),  # 方案2
    ("Shrink_r1.0","shrink",      {"rho": 1.0, "c": 0.3}),
    ("Maha_a0.5",  "maha",        {"alpha": 0.5}),  # 方案3 (no noise)
    ("Maha_a1.0",  "maha",        {"alpha": 1.0}),
]

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)

PROMPT_NEEDS_HARD_COPY = True


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


def compute_steer(mu: torch.Tensor, sigma: torch.Tensor, mode: str, params: dict) -> torch.Tensor:
    """Compute steer vector per mode.

    Returns [H] tensor.
    """
    import torch
    if mode == "off":
        return mu  # 不调用 hook,这里占位
    if mode == "mu_only":
        return params["alpha"] * mu
    if mode == "ball":
        # Ball-constrained: ε ~ N(0, I), ||ε|| ≤ 1
        rho = params["rho"]
        eps = torch.randn_like(mu)
        eps = eps / (eps.norm() + 1e-8)  # ||eps|| = 1
        # σ_safe = min(σ, c × |μ|), c=1.0 (no shrink in v1)
        return mu + rho * sigma * eps
    if mode == "shrink":
        # Shrink σ to ≤ c × |μ|, then ball sample
        rho = params["rho"]
        c = params["c"]
        sigma_safe = torch.minimum(sigma, c * mu.abs())
        eps = torch.randn_like(mu)
        eps = eps / (eps.norm() + 1e-8)
        s = mu + rho * sigma_safe * eps
        # Final norm clip at 2 × ||μ||
        max_norm = 2.0 * mu.norm()
        if s.norm() > max_norm:
            s = s * max_norm / s.norm()
        return s
    if mode == "maha":
        # Mahalanobis weight: high σ dim = low confidence = less steer
        alpha = params["alpha"]
        sigma_norm = sigma / (sigma.mean() + 1e-8)
        confidence = 1.0 / (1.0 + sigma_norm)
        return alpha * mu * confidence
    raise ValueError(f"Unknown mode: {mode}")


def make_hook(mode: str, steer_vecs: list, params: dict):
    """Hook 装在 layer 14 forward, target last token."""
    def _hook(module, args, output):
        import torch
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for b in range(h.size(0)):
            v = steer_vecs[b]
            if v is None:
                continue
            v_dev = v.to(h.device).to(h.dtype)
            pos = h.size(1) - 1
            h[b, pos] = h[b, pos] + v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def get_module(model, layer: int):
    return model.model.layers[layer]


def generate_for_config(client, model, tok, device, pairs, user_gauss, gauss_meta,
                        cond_name: str, mode: str, params: dict, K: int) -> list[dict]:
    import torch
    log(f"  [{cond_name}] generating {len(pairs)} pairs × K={K} (mode={mode}, params={params}) ...")
    is_off = (mode == "off")
    handle = None
    if not is_off:
        layer_module = get_module(model, LAYER)
        steer_vecs: list = []
        handle = layer_module.register_forward_hook(make_hook(mode, steer_vecs, params))

    results = []
    t0 = time.time()
    try:
        for i, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
            prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
            prompts = [prompt] * K

            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)

            if not is_off:
                if uid not in user_gauss:
                    log(f"    WARN: user {uid[:10]} not in Gaussian, skip")
                    continue
                mu = torch.tensor(user_gauss[uid]["mu"], dtype=torch.float32, device=device)
                sigma = torch.tensor(user_gauss[uid]["sigma_diag"], dtype=torch.float32, device=device)
                # 算 K 个 steer (每个 sample 一个, ball/shrink 模式会有 randomness)
                steer_list = []
                for _ in range(K):
                    s = compute_steer(mu, sigma, mode, params)
                    steer_list.append(s)
                steer_vecs.clear()
                steer_vecs.extend(steer_list)

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
            for k in range(K):
                gen_ids = out[k][input_len:]
                gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                results.append({
                    "user_id": uid,
                    "asin": asin,
                    "attrs": attrs,
                    "condition": cond_name,
                    "mode": mode,
                    "params": params,
                    "cand_local_idx": k,
                    "q_styled": gen_text,
                    "attrs_str": attrs_str,
                })
            if (i + 1) % 10 == 0:
                log(f"    {cond_name} {i + 1}/{len(pairs)} ({time.time() - t0:.1f}s)")
    finally:
        if handle is not None:
            handle.remove()
    log(f"  → {len(results)} records, sweep total {time.time() - t0:.1f}s")
    return results


def main() -> None:
    log("=" * 70)
    log("Phase 16: N=30 Gaussian steer sweep (ball/shrink/Maha)")
    log("=" * 70)

    np.random.seed(SEED)

    # === [1] Load pairs ===
    log("[1] Loading pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    # === [2] Fit Gaussian (N=30) using cached hiddens + neutral ===
    log("[2] Fitting Gaussian N=30 from cached hiddens ...")
    h_d = np.load(HIDDEN_NPZ, allow_pickle=True)
    cached_user_ids = list(h_d["user_ids"])
    cached_hiddens = list(h_d["hiddens"])
    cached_counts = list(h_d["counts"])

    n_d = np.load(NEUTRAL_NPZ, allow_pickle=True)
    neutral_vecs = n_d["vecs"].astype(np.float32)  # [N, 28, 3584]
    neutral_vec = neutral_vecs[:, 14, :].mean(axis=0)  # [3584]

    # 仅用 N ≥ 30 sentences 的 users
    valid_set = {uid for uid, c in zip(cached_user_ids, cached_counts) if c >= N_FIT}
    user_gauss: dict[str, dict] = {}
    skipped = 0
    for uid, h, c in zip(cached_user_ids, cached_hiddens, cached_counts):
        if c < N_FIT or uid not in valid_set:
            skipped += 1
            continue
        h_use = h[:N_FIT].astype(np.float32)  # [N, H]
        residual = h_use - neutral_vec[None, :]
        mu = residual.mean(axis=0)
        sigma = residual.std(axis=0, ddof=0)
        user_gauss[uid] = {"mu": mu, "sigma_diag": sigma}
    log(f"  users with ≥{N_FIT} sents: {len(user_gauss)}, skipped: {skipped}")

    # Filter pairs to valid Gaussian users
    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  filtered pairs: {len(pairs)}")

    # Cache N=30 Gaussians
    np.savez(OUT_GAUSS_N30,
             user_ids=np.array(list(user_gauss.keys())),
             mu=np.stack([user_gauss[u]["mu"] for u in user_gauss]),
             sigma_diag=np.stack([user_gauss[u]["sigma_diag"] for u in user_gauss]))
    log(f"  saved N=30 Gaussians: {OUT_GAUSS_N30}")

    # === [3] Load Qwen ===
    log("[3] Loading Qwen ...")
    os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tok = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}")

    # === [4] Sweep ===
    log("[4] Sweep ...")
    all_results = []
    for cond_name, mode, params in SWEEP:
        log(f"--- Config: {cond_name} (mode={mode}, params={params}) ---")
        results = generate_for_config(
            client, model, tok, device, pairs, user_gauss, None,
            cond_name, mode, params, K,
        )
        # Hard-copy post-processing
        for r in results:
            r["q_final_post"] = append_missing_attrs(r["q_styled"], r["attrs_str"])
        all_results.extend(results)
        # Incremental write
        with JSONL_OUT.open("a" if JSONL_OUT.exists() else "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"[5] Total records: {len(all_results)}")

    # === [5] Meta ===
    META_OUT.write_text(json.dumps({
        "phase": "16",
        "n_configs": len(SWEEP),
        "config_names": [c[0] for c in SWEEP],
        "n_pairs": len(pairs),
        "K": K,
        "N_fit": N_FIT,
        "layer": LAYER,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hard_copy_post": PROMPT_NEEDS_HARD_COPY,
        "n_total_records": len(all_results),
        "n_user_gaussians": len(user_gauss),
    }, indent=2, ensure_ascii=False))
    log(f"  meta: {META_OUT}")

    log("=" * 70)
    log("PHASE 16 SWEEP DONE — run eval next")
    log("=" * 70)


if __name__ == "__main__":
    main()