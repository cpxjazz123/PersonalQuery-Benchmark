#!/usr/bin/env python3
"""Phase 15.E: α-only sweep on v1 mean-pool StyleVector (NO ρ).

Reuse Phase 15.D framework but:
  - Remove ρ entirely (steer = α × μ only, no std scaling)
  - Sweep α ∈ {0.3, 0.5, 1.0, 1.5} to find best injection strength
  - Same 30 pairs × K=8 = 240 per config, 5 conds total = 1200 records

Goal: confirm best α that maximizes 768d style BoK margin lift vs D_off
       WITHOUT the ρ × std_diag scaling that destroyed lift in Phase 15.D.
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

# === Hardcoded config ===
IN_GAUSSIANS = OUT_DIR / "phase13_b_user_gaussians_qwen.npz"  # mean-pool v1
QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"  # Phase 13.D 取前 30 保证可比

JSONL_OUT = OUT_DIR / "phase15_e_alpha_sweep.jsonl"
SAMPLES_OUT = OUT_DIR / "phase15_e_samples.json"
META_OUT = OUT_DIR / "phase15_e_meta.json"
LOG_PATH = OUT_DIR / "phase15_e.log"

LAYER = 14
K = 8
N_PAIRS = 30
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 80
SEED = 42

# α sweep (NO ρ, just inject μ × α)
SWEEP = [
    ("D_off", 0.0),
    ("A_a0.3", 0.3),
    ("A_a0.5", 0.5),
    ("A_a1.0", 1.0),
    ("A_a1.5", 1.5),
]

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)

PROMPT_NEEDS_HARD_COPY = True  # 沿用 Phase 13.D 的硬属性覆盖 post-processing


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_hook(steer_vecs: list, steer_layer: int, steer_alpha: float):
    def _hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for b in range(h.size(0)):
            v = steer_vecs[b]
            if v is None:
                continue
            v_dev = v.to(h.device).to(h.dtype)
            pos = h.size(1) - 1
            h[b, pos] = h[b, pos] + steer_alpha * v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def get_module(model, layer: int):
    return model.model.layers[layer]


def generate_for_config(client, model, tok, device, pairs: list, user_gauss: dict,
                        cond_name: str, alpha: float, K: int) -> list[dict]:
    import torch
    log(f"  [{cond_name}] generating {len(pairs)} pairs × K={K} (α={alpha}) ...")
    is_baseline = (alpha == 0.0)
    handle = None
    if not is_baseline:
        layer_module = get_module(model, LAYER)
        steer_vecs: list = []
        handle = layer_module.register_forward_hook(make_hook(steer_vecs, LAYER, alpha))

    results: list[dict] = []
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

            if not is_baseline:
                if uid not in user_gauss:
                    log(f"    WARN: user {uid[:10]} not in Gaussian cache, skip")
                    continue
                mu = torch.tensor(user_gauss[uid]["mu"][LAYER], dtype=torch.float32,
                                  device=device)  # [H]
                # NO ρ — just inject μ × α
                steer_vecs.clear()
                steer_vecs.extend([mu.clone() for _ in range(K)])

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
                    "alpha": alpha,
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


def append_missing_attrs(q: str, attrs_str: str) -> str:
    """Hard-copy attributes that may have been dropped during generation."""
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


def main() -> None:
    log("=" * 70)
    log("Phase 15.E: α-only sweep on v1 mean-pool StyleVector (no ρ)")
    log("=" * 70)

    np.random.seed(SEED)

    # === [1] Load pairs (filter to Gaussian cache, 与 Phase 13.D 同协议) ===
    log("[1] Loading pairs and filtering to Gaussian cache ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    # Load Gaussian user_ids (will cache below)
    gdata_preview = np.load(IN_GAUSSIANS, allow_pickle=True)
    valid = gdata_preview["valid"]
    user_ids_preview = list(gdata_preview["user_ids"])
    valid_uids = [u for u, v in zip(user_ids_preview, valid) if v]
    valid_set = set(valid_uids)
    log(f"  pairs total: {len(pairs)}, valid users in Gaussian cache: {len(valid_set)}")
    pairs = [p for p in pairs if p["user_id"] in valid_set][:N_PAIRS]
    log(f"  filtered pairs (target user in valid set): {len(pairs)}")
    log(f"  first pair attrs: {pairs[0]['attrs']}")

    # === [2] Load user Gaussians ===
    log("[2] Loading v1 mean-pool user_gaussians (298 users, layer 14) ...")
    gdata = np.load(IN_GAUSSIANS, allow_pickle=True)
    user_ids_gauss = list(gdata["user_ids"])
    mu_arr = gdata["mu"]            # [298, 28, 3584]
    sigma_arr = gdata["sigma_diag"]  # not used in this sweep (NO ρ)
    uid_to_gidx = {u: i for i, u in enumerate(user_ids_gauss)}
    user_gauss: dict[str, dict] = {}
    for u in user_ids_gauss:
        i = uid_to_gidx[u]
        user_gauss[u] = {"mu": mu_arr[i], "sigma_diag": sigma_arr[i]}
    log(f"  cached {len(user_gauss)} user Gaussians")

    # === [3] Load Qwen (transformers backend, with_vllm=False) ===
    log("[3] Loading Qwen (transformers backend) ...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tok = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}")

    # === [4] Sweep ===
    all_results: list[dict] = []
    log("[4] Sweep over α configs (no ρ) ...")
    for cond_name, alpha in SWEEP:
        log(f"--- Config: {cond_name} (α={alpha}) ---")
        results = generate_for_config(
            client, model, tok, device, pairs, user_gauss, cond_name, alpha, K,
        )
        # Hard-copy post-processing (Phase 13.D 协议)
        if PROMPT_NEEDS_HARD_COPY and alpha > 0:
            for r in results:
                r["q_final_post"] = append_missing_attrs(r["q_styled"], r["attrs_str"])
        else:
            for r in results:
                r["q_final_post"] = r["q_styled"]
        all_results.extend(results)
        # Write incremental
        with JSONL_OUT.open("a" if JSONL_OUT.exists() else "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"[5] Total records: {len(all_results)}")

    # === [5] Save meta ===
    META_OUT.write_text(json.dumps({
        "phase": "15.E",
        "n_configs": len(SWEEP),
        "n_pairs": len(pairs),
        "K": K,
        "layer": LAYER,
        "alpha_sweep": [a for _, a in SWEEP],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hard_copy_post": PROMPT_NEEDS_HARD_COPY,
        "n_total_records": len(all_results),
        "n_user_gaussians": len(user_gauss),
    }, indent=2, ensure_ascii=False))
    log(f"  meta: {META_OUT}")

    log("=" * 70)
    log("PHASE 15.E SWEEP COMPLETE — run phase15_d_eval_only.py next (or new eval)")
    log("=" * 70)


if __name__ == "__main__":
    main()