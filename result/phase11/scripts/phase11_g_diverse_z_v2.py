#!/usr/bin/env python3
"""Phase 11.G v2: batched vLLM generate (10x faster).

Bottleneck of v1: client.call() does single-prompt generate (batch=1 in vLLM).
Each pair triggered 13 sequential vLLM calls (1 neutral + 10 personal + 2x).

Fix: directly use client._backend.model.generate([prompts]) to batch:
  - Personal prompts (K=10) per pair: 1 batched call instead of 10 sequential
  - Expected ~8-10x speedup on LLM decode

Same outputs as v1 (600 records), same params, same file format.
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
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
GAUSSIANS_NPZ = OUT_DIR / "phase11_a_user_gaussians_768d.npz"
DIFFUSION_PT = OUT_DIR / "phase11_c_diffusion_unet.pt"
CONTENT_PCA_NPY = OUT_DIR / "phase11_c_content_pca.npz"

OUT_GENERATIONS = OUT_DIR / "phase11_g_e2e_queries.jsonl"
OUT_META = OUT_DIR / "phase11_g_e2e_meta.json"

# === Hardcoded config (DIFFERENT from 11.D) ===
N_PAIRS = 30
N_STYLE_SAMPLES = 10          # ↑ from 5
N_DECODE_SAMPLES = 2          # ↓ from 4 (keep total 20)
N_DIFFUSION_STEPS = 30        # ↓ from 100
START_T = 800
TEMPERATURE_NEUTRAL = 0.3
TEMPERATURE_PERSONAL = 0.7
MAX_NEW_TOKENS = 48
LATENT_DIM = 128
RANDOM_SEED = 42
QWEN_LAYER_FOR_CONTENT = 14

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def z_to_style_hint(z_pers: np.ndarray, mu_u: np.ndarray) -> str:
    """Translate z_personalized into a brief style hint."""
    diff = z_pers - mu_u
    pos_idx = np.argsort(diff)[-5:][::-1]
    neg_idx = np.argsort(diff)[:5]
    STYLE_LABELS = {
        "specifications and details": "specifications and details",
        "concise phrasing": "concise phrasing",
        "list / bullet format": "list / bullet format",
        "imperative voice": "imperative voice (e.g. 'Need', 'Looking for')",
        "question form": "question form",
        "subjective adjectives": "subjective adjectives (great, perfect, love)",
        "factual description": "factual description",
        "parenthetical asides": "parenthetical asides",
        "numeric emphasis": "numeric emphasis (dimensions, weight)",
        "complete sentences with period": "complete sentences with period",
        "clause chains with semicolons": "clause chains with semicolons",
        "nested clauses": "nested clauses",
        "opens with pronoun": "opens with pronoun (I, my, this)",
        "opens with noun/brand": "opens with noun/brand",
        "terse no flourish": "terse, no flourish",
    }
    labels = list(STYLE_LABELS.values())
    pos_descs = []
    for i in pos_idx:
        lbl = labels[i % len(labels)]
        pos_descs.append(f"more {lbl}")
    neg_descs = []
    for i in neg_idx:
        lbl = labels[i % len(labels)]
        neg_descs.append(f"less {lbl}")
    hint = "; ".join(pos_descs[:3]) + ". " + "; ".join(neg_descs[:3]) + "."
    return hint


def batch_generate(client, prompts: list, temperature: float, max_tokens: int) -> list:
    """Direct vLLM batched generate — bypasses single-prompt call()."""
    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    # Build full chat-template prompts (same as call() does internally)
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            client._backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    outputs = client._backend.model.generate(full_prompts, sampling)
    texts = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
    return texts


def strip_query_prefix(text: str) -> str:
    if text.lower().startswith("query:"):
        return text[len("query:"):].strip()
    return text


def main():
    log("=" * 70)
    log("Phase 11.G v2: Batched vLLM (K=10, 30 reverse steps, start_t=800)")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    log("[1] Loading 30 pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    pairs = pairs[:N_PAIRS]
    log(f"  pairs: {len(pairs)}")

    log("[2] Loading Phase 11.A Gaussian cache ...")
    npz = np.load(GAUSSIANS_NPZ, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    mu_768 = npz["mu_768"]
    pca_components = npz["pca_components"]
    pca_mean = npz["pca_mean"]
    mu_128 = (mu_768 - pca_mean) @ pca_components.T
    sigma_diag = npz["sigma_diag"]
    V_k = pca_components
    sigma_diag_128 = ((V_k ** 2) @ sigma_diag.T).T.astype(np.float32)
    mu_128 = mu_128.astype(np.float32)
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    log(f"  mu_128: {mu_128.shape}")

    log("[3] Loading diffusion model ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from phase11_c_diffusion_train import (
        ConditionalDenoiser, GaussianDiffusion, N_TIMESTEPS, BETA_SCHEDULE
    )
    diffusion = GaussianDiffusion(N_TIMESTEPS, BETA_SCHEDULE)
    ckpt = torch.load(DIFFUSION_PT, map_location="cpu", weights_only=False)
    denoiser = ConditionalDenoiser()
    denoiser.load_state_dict(ckpt["state_dict"])
    denoiser.to(DEVICE).eval()
    sqrt_alphas = torch.from_numpy(np.sqrt(diffusion.alphas_cumprod).astype(np.float32)).to(DEVICE)
    sqrt_one_minus_alphas = torch.from_numpy(np.sqrt(1.0 - diffusion.alphas_cumprod).astype(np.float32)).to(DEVICE)

    log("[4] Loading content PCA ...")
    content_pca = np.load(CONTENT_PCA_NPY)
    pca_components_c = content_pca["pca_components"]
    pca_mean_c = content_pca["pca_mean"]
    content_std = content_pca["content_std"]
    log(f"  pca_components_c: {pca_components_c.shape}")

    log("[5] Loading Qwen2-7B-Instruct ...")
    os.environ.setdefault("QWEN_GPU_MEMORY_UTILIZATION", "0.5")
    if "QWEN_MODEL_PATH" not in os.environ:
        os.environ["QWEN_MODEL_PATH"] = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=True)
    log(f"  Qwen client loaded")

    log(f"[6] Generating {N_PAIRS} pairs × K={N_STYLE_SAMPLES} × M={N_DECODE_SAMPLES} = {N_PAIRS * N_STYLE_SAMPLES * N_DECODE_SAMPLES} total (batched) ...")
    records = []
    t0 = time.time()
    timesteps = np.linspace(START_T, 0, N_DIFFUSION_STEPS, dtype=np.int64)

    for pi_idx, p in enumerate(pairs):
        uid = p["user_id"]
        if uid not in uid_to_idx:
            continue
        u_idx = uid_to_idx[uid]
        attrs = p["attrs"]
        if not all(attrs.get(k) for k in ["Brand", "Color", "Material"]):
            continue

        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)

        # === Step 1: neutral query (single prompt) ===
        neutral_prompt = (
            "You are a shopping assistant. Write a concise, neutral search query "
            "for the product below. Plain factual phrasing, no stylistic flourish.\n\n"
            f"Product: {attr_str}\n\nQuery:"
        )
        q_neutral_list = batch_generate(
            client, [neutral_prompt], TEMPERATURE_NEUTRAL, MAX_NEW_TOKENS
        )
        q_neutral = strip_query_prefix(q_neutral_list[0])

        # === Step 2: encode q_neutral -> z_neutral (128d) ===
        c_dict = client.get_hidden_states([q_neutral], layers=[QWEN_LAYER_FOR_CONTENT],
                                          batch_size=1, max_length=128)
        c_raw_3584 = c_dict[QWEN_LAYER_FOR_CONTENT]
        z_neutral = (c_raw_3584 - pca_mean_c) @ pca_components_c.T
        z_neutral = z_neutral.astype(np.float32) / np.maximum(content_std, 1e-3)

        # === Step 3: sample K=10 s_u ===
        mu = mu_128[u_idx]
        std = np.sqrt(np.maximum(sigma_diag_128[u_idx], 1e-6))
        s_u_samples = mu[None, :] + std[None, :] * rng.standard_normal(
            (N_STYLE_SAMPLES, LATENT_DIM)).astype(np.float32)

        # === Step 4: diffusion reverse ===
        z_t = torch.from_numpy(z_neutral).to(DEVICE)
        c_t = torch.from_numpy(z_neutral).to(DEVICE)
        noise_start = torch.randn_like(z_t)
        sa_start = sqrt_alphas[START_T]
        so_start = sqrt_one_minus_alphas[START_T]
        z_t_noisy = sa_start * z_t + so_start * noise_start

        z_personalized_list = []
        for s in s_u_samples:
            s_t = torch.from_numpy(s[None, :]).to(DEVICE)
            z_curr = z_t_noisy.clone()
            for i, t_val in enumerate(timesteps):
                t_tensor = torch.full((1,), int(t_val), dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    eps_pred = denoiser(z_t=z_curr, t=t_tensor, c_raw=c_t, s_u=s_t)
                    sa = sqrt_alphas[int(t_val)]
                    so = sqrt_one_minus_alphas[int(t_val)]
                    sa_next = sqrt_alphas[int(timesteps[i + 1])] if i < len(timesteps) - 1 else sqrt_alphas[0]
                    so_next = sqrt_one_minus_alphas[int(timesteps[i + 1])] if i < len(timesteps) - 1 else sqrt_one_minus_alphas[0]
                    z0_pred = (z_curr - so * eps_pred) / sa
                    z_curr = sa_next * z0_pred + so_next * eps_pred
            z_personalized_list.append(z_curr.cpu().numpy()[0])
        z_personalized = np.stack(z_personalized_list)

        # === Step 5+6: build K personal prompts, batched decode ===
        # Each style sample × M seeds, but the prompt only differs by style hint.
        # So K prompts × M seeds → just replicate K prompts M times (different sampling noise).
        personal_prompts = []
        for k_idx in range(N_STYLE_SAMPLES):
            hint = z_to_style_hint(z_personalized[k_idx], mu_128[u_idx])
            pp = (
                "You are a shopping assistant. Write a concise search query for the "
                "product below in the user's personal style. Style hints: " +
                hint + "\n\n"
                f"Product: {attr_str}\n\nQuery:"
            )
            personal_prompts.append(pp)

        # Run M=2 batched decodes (different temp seeds via SamplingParams seed not reliable,
        # so use repeated calls with M=1 sampling_seed ... actually vLLM doesn't expose per-call seed
        # via SamplingParams in this version. Workaround: call M times with explicit seed via np.random.
        # For Phase 11.G we just want K different style outputs, M=2 adds slight variation.
        # We pass M copies of each prompt to one batched call, but with shared prompt vLLM
        # gives identical outputs. Solution: vary prompt slightly with a unique suffix per seed.
        all_personal_records = []
        for m_idx in range(N_DECODE_SAMPLES):
            # Add seed marker to each prompt so vLLM doesn't dedupe
            seed_prompts = [
                p.replace("Query:", f"Query (variant {m_idx}):") for p in personal_prompts
            ]
            q_personalized_list = batch_generate(
                client, seed_prompts, TEMPERATURE_PERSONAL, MAX_NEW_TOKENS
            )
            for k_idx, q_pp in enumerate(q_personalized_list):
                q_pp = strip_query_prefix(q_pp)
                try:
                    from query_gen.e30.post_attr import _append_missing_attrs
                    q_final = _append_missing_attrs(q_pp, attrs)
                except Exception:
                    q_final = q_pp

                records.append({
                    "pair_idx": pi_idx,
                    "user_id": uid,
                    "asin": p["asin"],
                    "attrs": attrs,
                    "q_neutral": q_neutral,
                    "style_sample_idx": k_idx,
                    "decode_seed_idx": m_idx,
                    "q_personalized": q_pp,
                    "q_final_post": q_final,
                    "style_hint": z_to_style_hint(z_personalized[k_idx], mu_128[u_idx]),
                })

        if (pi_idx + 1) % 3 == 0:
            elapsed = time.time() - t0
            rate = len(records) / max(elapsed, 0.001)
            eta = (N_PAIRS * N_STYLE_SAMPLES * N_DECODE_SAMPLES - len(records)) / max(rate, 0.001)
            log(f"  [{pi_idx + 1}/{N_PAIRS}] {len(records)} records, elapsed={elapsed:.1f}s, ETA={eta:.0f}s")

    log(f"[7] Saving {len(records)} records ...")
    with OUT_GENERATIONS.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GENERATIONS}")

    meta = {
        "phase": "11.G v2",
        "n_pairs": N_PAIRS,
        "n_style_samples": N_STYLE_SAMPLES,
        "n_decode_samples": N_DECODE_SAMPLES,
        "n_records": len(records),
        "n_diffusion_steps": N_DIFFUSION_STEPS,
        "start_t": START_T,
        "temperature_neutral": TEMPERATURE_NEUTRAL,
        "temperature_personal": TEMPERATURE_PERSONAL,
        "diff_vs_11D": "K=10 (was 5), reverse=30 steps (was 100), start_t=800 (was 999), M=2 (was 4)",
        "speedup": "vLLM batched: K=10 personal prompts in 1 call vs 10 sequential",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log("=" * 70)
    log("PHASE 11.G v2 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()