#!/usr/bin/env python3
"""Phase 13.D: E2E generation, 4 conditions × 30 pairs × K=8 candidates.

Reuses QwenLocalClient with hooks + num_return_sequences for batched K-sample gen.
Conditions:
  A sampled-s_u   : s_u^(k) ~ N(mu_u^14, diag(sigma_u^14)) rho=0.5, K independent
  B mean-s_u      : s_u = mu_u^14 (single mean baseline, ACL 2025 StyleVector protocol)
  C shuffled-s_u  : s_u = mu_other^14 (wrong user, sanity)
  D injection-off : no intervention (Phase 10.15 baseline)

Output:
  - phase13_d_v2_e2e_queries_lasttoken.jsonl (30 pairs × 8 cands × 4 conditions = 960 records)
  - phase13_d_v2_meta_lasttoken.json

Reuses:
  - phase13_b_v2_user_gaussians_qwen_lasttoken.npz (mu, std_diag, valid)
  - phase10_pairs_1000.jsonl (30 pair subset)
  - phase10_19_b_contrastive_rag._append_missing_attrs (hard-copy post-process)
"""
from __future__ import annotations

import json
import os
import random
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
IN_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_QUERIES = OUT_DIR / "phase13_d_v2_e2e_queries_lasttoken.jsonl"
OUT_META = OUT_DIR / "phase13_d_v2_meta_lasttoken.json"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Hardcoded config ===
N_PAIRS = 30
K_CANDIDATES = 8
LAYER = 14
ALPHA = 1.0
RHO = 0.5
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_hook(steer_vecs: list, steer_layer: int, steer_alpha: float):
    """Forward hook: h' = h + alpha*v at last token, layer ℓ. Single-tensor output (Qwen2)."""
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


def main():
    log("=" * 70)
    log(f"Phase 13.D: 4 conditions × {N_PAIRS} pairs × K={K_CANDIDATES} candidates")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    # === [1] Load pairs and Gaussian cache ===
    log("[1] Loading pairs and Gaussian cache ...")
    pairs: list[dict] = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    npz = np.load(IN_GAUSSIANS, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    mu = npz["mu"]
    std_diag = npz["std_diag"]
    valid = npz["valid"]
    valid_uids = [u for u, v in zip(user_ids, valid) if v]
    valid_idxs = [i for i, v in enumerate(valid) if v]
    uid_to_gidx = {u: i for i, u in enumerate(user_ids)}
    valid_set = set(valid_uids)
    log(f"  pairs total: {len(pairs)}, valid users in Gaussian cache: {len(valid_uids)}")

    # Filter to first 30 pairs where target user is in valid set
    pairs = [p for p in pairs if p["user_id"] in valid_set][:N_PAIRS]
    log(f"  filtered pairs (target user in valid set): {len(pairs)}")

    # === [2] Load Qwen2-7B (transformers backend) ===
    log(f"[2] Loading Qwen2-7B ...")
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

    # Pre-compute prompt tokens for each pair (with right padding handled via repeated call)
    # For simplicity: process each pair individually (30 pairs × 4 conditions × 8 cands = 960 records)

    # Hard-copy post-process
    sys.path.insert(0, str(OUT_DIR))
    from phase10_19_b_contrastive_rag import _append_missing_attrs

    # === [3] Generation loop ===
    log(f"[3] Generating {len(pairs)} pairs × {len(CONDITIONS)} conditions × K={K_CANDIDATES} ...")
    all_records: list[dict] = []
    t0 = time.time()

    for pi, p in enumerate(pairs):
        uid = p["user_id"]
        attrs = p["attrs"]
        asin = p["asin"]
        u_gidx = uid_to_gidx[uid]

        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
        prompt_text = (
            "You are a shopping assistant. Write a concise search query for the "
            "product below.\n\n"
            f"Product: {attr_str}\n\nQuery:"
        )
        enc = tok(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")
        prompt_len = enc.shape[1]

        for cond in CONDITIONS:
            # === Build K=8 steer vectors per condition ===
            steer_vecs = None
            if cond == "A_sampled":
                steer_vecs = []
                mu_u = torch.from_numpy(mu[u_gidx, LAYER]).to(torch.float32).to("cuda:0")
                std_u = torch.from_numpy(std_diag[u_gidx, LAYER]).to(torch.float32).to("cuda:0")
                for k in range(K_CANDIDATES):
                    eps = torch.randn_like(mu_u)
                    s = mu_u + RHO * std_u * eps
                    steer_vecs.append(s)
            elif cond == "B_mean":
                mu_u = torch.from_numpy(mu[u_gidx, LAYER]).to(torch.float32).to("cuda:0")
                steer_vecs = [mu_u.clone() for _ in range(K_CANDIDATES)]
            elif cond == "C_shuffled":
                # Pick a different valid user
                other_gidx = rng.choice([i for i in valid_idxs if i != u_gidx])
                mu_o = torch.from_numpy(mu[other_gidx, LAYER]).to(torch.float32).to("cuda:0")
                steer_vecs = [mu_o.clone() for _ in range(K_CANDIDATES)]
            # D_off: steer_vecs stays None

            # Register hook
            hook_handle = None
            if steer_vecs is not None:
                hook_handle = model.model.layers[LAYER].register_forward_hook(
                    make_hook(steer_vecs, LAYER, ALPHA))

            # Generate K candidates via num_return_sequences
            # Note: for batched gen with different steer per sequence, we'd need batch_size>1
            # but transformers generate doesn't natively support per-sample hooks easily.
            # Workaround: generate K=8 sequentially (still batched within one forward via repetition)
            # Actually, simplest: replicate prompt K=8 times (one big batch), K=8 different v's
            try:
                if cond == "D_off":
                    # K=8 candidates via num_return_sequences (same prompt)
                    with torch.no_grad():
                        gen = model.generate(
                            enc.repeat(K_CANDIDATES, 1),
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                            num_return_sequences=1, use_cache=True,
                        )
                else:
                    # Batched K with K different steer vectors
                    # Repeat prompt K times (each row identical), K hooks apply per row
                    enc_rep = enc.repeat(K_CANDIDATES, 1)
                    with torch.no_grad():
                        gen = model.generate(
                            enc_rep,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                            num_return_sequences=1, use_cache=True,
                        )
            except Exception as e:
                log(f"  ERROR pair {pi} cond {cond}: {e}")
                if hook_handle is not None:
                    hook_handle.remove()
                raise
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            # Decode K candidates
            for k in range(K_CANDIDATES):
                row = gen[k, prompt_len:]
                if tok.eos_token_id is not None and tok.eos_token_id in row:
                    row = row[:row.tolist().index(tok.eos_token_id)]
                q = tok.decode(row, skip_special_tokens=True).strip()
                q_final = _append_missing_attrs(q, attrs)
                all_records.append({
                    "pair_idx": pi,
                    "user_id": uid,
                    "asin": asin,
                    "attrs": attrs,
                    "condition": cond,
                    "cand_idx": k,
                    "q_styled": q,
                    "q_final_post": q_final,
                })

        elapsed = time.time() - t0
        eta = (len(pairs) - pi - 1) * (elapsed / (pi + 1))
        log(f"  [{pi + 1}/{len(pairs)}] {len(all_records)} records, "
            f"elapsed={elapsed:.1f}s, ETA={eta:.0f}s")

    # === [4] Save ===
    log(f"[4] Saving {len(all_records)} records to {OUT_QUERIES} ...")
    with OUT_QUERIES.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_QUERIES}")

    # Meta
    cond_counts = {c: sum(1 for r in all_records if r["condition"] == c) for c in CONDITIONS}
    meta = {
        "phase": "13.D",
        "model": Path(QWEN_MODEL_PATH).name,
        "n_pairs": len(pairs),
        "n_conditions": len(CONDITIONS),
        "k_candidates": K_CANDIDATES,
        "n_records": len(all_records),
        "layer": LAYER,
        "alpha": ALPHA,
        "rho": RHO,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "conditions": CONDITIONS,
        "records_per_condition": cond_counts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta: {OUT_META}")

    log("=" * 70)
    log("PHASE 13.D COMPLETE — ready for 13.E evaluation")
    log("=" * 70)


if __name__ == "__main__":
    main()