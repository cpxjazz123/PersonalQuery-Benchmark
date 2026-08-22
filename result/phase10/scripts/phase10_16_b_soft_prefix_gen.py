#!/usr/bin/env python3
"""Phase 10.16.B: 4-control batch generation with soft prefix injection.

Strategy:
  - Use e22_t3 trained projector + 318d phase10 user vectors
  - Generate 4 controls per (user, asin):
    * A real-z:       projector(z_user)         (correct user condition)
    * B shuffled-z:   projector(z_other_user)   (wrong user, same prompt)
    * C true-zero-z:  zeros_like(prefix)         (no signal; explicit zero)
    * D injection-off: no prefix at all          (baseline)
  - For each control, K seeds = 4 candidates (seeds 0..3) -> 4*876*4 = 14016 total

Why: Phase 10.15 rank-1 coverage = 1.26% (96 candidates, no prefix injection).
     Soft prefix injection may push coverage up by conditioning LLM on user
     style. Strictest test: shuffled-z should NOT show rank-1 (sanity control).

Implementation:
  - Reuses e22_t3_injector_best.pt (K=16, model_dim=3584, alpha=0.277)
  - Native transformers forward with inputs_embeds (no vllm; vllm does not
    support per-prompt soft prefix embeddings).
  - Copy-aware content guard REMOVED (control D = no prefix at all; we want
    to test pure prefix effect, not copy_head effect).
  - 1 candidate per control per pair per seed (4 controls * 4 seeds = 16/pair).

Output: phase10_16_b_generations_4control.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

from projector import SoftPrefixProjector  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
USER_VECS_NPY = OUT_DIR / "phase10_user_vectors_318d.npy"
USER_VECS_IDS = OUT_DIR / "phase10_user_vectors_318d_user_ids.json"

INJECTOR_CKPT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_injector_best.pt"
MODEL_PATH = "/fs04/scratch2/hj82/yubow/phd2026/open_models/Qwen/Qwen2.5-Coder-7B-Instruct"  # hidden=3584, matches e22_t3 projector

OUT_GENERATIONS = OUT_DIR / "phase10_16_b_generations_4control.jsonl"

# === Hard-coded config (CLAUDE.md Rule 3) ===
N_CONTROLS = 4  # real-z, shuffled-z, true-zero-z, injection-off
N_SEEDS_PER_CONTROL = 4  # 4 candidates per (user, asin, control)
SEED_BASE = 7777
BATCH = 32  # batches (Qwen 7B bf16 ~14GB; 64GB free → batch 32 fits)
MAX_NEW = 48  # 48 tokens sufficient for shopping queries
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.92
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)

CONTROLS = ["real-z", "shuffled-z", "true-zero-z", "injection-off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def attr_prompt(attrs: dict) -> str:
    lines = ["Product attributes:"]
    for k, v in attrs.items():
        if v:
            lines.append(f"{k}: {v}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def chat_prompt(tokenizer, attrs: dict) -> list[int]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": attr_prompt(attrs)},
    ]
    out = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "input_ids"):
        return list(out.input_ids)
    return list(out)


@torch.inference_mode()
def generate_batch(
    model, tok, proj, prefix_embeds_list, prompts_token_lists,
    max_new: int, temperature: float, top_k: int, top_p: float,
    num_tokens: int, model_dim: int, dtype, device: str,
) -> list[str]:
    """Batched KV-cache decoding with optional per-sample prefix.

    prefix_embeds_list[b] = [K, H] prefix embeddings or None (= injection-off).
    """
    bsz = len(prompts_token_lists)
    pad_id = tok.pad_token_id or tok.eos_token_id
    encs = [list(p) for p in prompts_token_lists]
    max_p = max(len(e) for e in encs)
    ids = torch.tensor(
        [e + [pad_id] * (max_p - len(e)) for e in encs],
        dtype=torch.long, device=device)
    masks = torch.tensor(
        [[1] * len(e) + [0] * (max_p - len(e)) for e in encs],
        dtype=torch.long, device=device)
    emb = model.get_input_embeddings()
    text_emb = emb(ids).to(dtype)

    has_prefix = any(p is not None for p in prefix_embeds_list)
    if has_prefix:
        prefix = torch.stack([
            p if p is not None else torch.zeros(num_tokens, model_dim, dtype=DTYPE, device=DEVICE)
            for p in prefix_embeds_list
        ])
        full = torch.cat([prefix, text_emb], dim=1)
        attn = torch.cat([
            torch.ones(bsz, num_tokens, dtype=torch.long, device=DEVICE), masks
        ], dim=1)
        pos = torch.cat([
            torch.arange(num_tokens, device=DEVICE).unsqueeze(0).expand(bsz, -1),
            torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1) + num_tokens
        ], dim=1)
    else:
        full = text_emb
        attn = masks
        pos = torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1)

    past = None
    generated: list[list[int]] = [[] for _ in range(bsz)]
    done = [False] * bsz
    next_emb = full
    next_ids = None
    for step in range(max_new):
        first = past is None
        out = model(
            inputs_embeds=next_emb if first else None,
            input_ids=None if first else next_ids,
            attention_mask=attn,
            position_ids=pos if first else None,
            use_cache=True, past_key_values=past, output_hidden_states=False)
        past = out.past_key_values
        if first:
            last_valid = attn.sum(dim=1) - 1
            logits = out.logits[torch.arange(bsz, device=DEVICE), last_valid]
        else:
            logits = out.logits[:, -1]

        # temperature + top-k + top-p
        logits = logits.float() / max(temperature, 1e-4)
        if top_k > 0:
            v = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1].unsqueeze(1)
            logits = torch.where(logits < v, torch.full_like(logits, float("-inf")), logits)
        if top_p < 1.0:
            sorted_l, idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.nn.functional.softmax(sorted_l, dim=-1), dim=-1)
            mask = cum - torch.nn.functional.softmax(sorted_l, dim=-1) > top_p
            sorted_l[mask] = float("-inf")
            logits = torch.gather(sorted_l, 1, idx.argsort(dim=-1))
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        sample_ids = torch.multinomial(probs, 1).squeeze(1)
        next_ids = sample_ids.clone()
        for b in range(bsz):
            if done[b]:
                next_ids[b] = tok.eos_token_id
                continue
            tid = int(next_ids[b])
            generated[b].append(tid)
            if tid == tok.eos_token_id:
                done[b] = True
        if all(done):
            break
        next_emb = emb(next_ids.unsqueeze(1)).to(dtype)
        next_ids = next_ids.unsqueeze(1)
        attn = torch.cat([
            attn, torch.ones(bsz, 1, dtype=attn.dtype, device=DEVICE)
        ], dim=1)
    return [tok.decode(g, skip_special_tokens=True).strip() for g in generated]


def main():
    log("=" * 70)
    log("Phase 10.16.B: Soft prefix injection — 4-control batch generation")
    log("=" * 70)

    # === Load pairs ===
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    # Filter pairs with main 3 attrs
    valid_pairs = []
    for p in pairs:
        attrs_5 = p.get("attrs_5") or p.get("attrs") or {}
        if all(attrs_5.get(k) for k in ["Brand", "Color", "Material"]):
            valid_pairs.append(p)
    log(f"  valid pairs (≥3 main attrs): {len(valid_pairs)}")

    # === Load user vectors ===
    user_vecs = np.load(USER_VECS_NPY)
    user_vec_ids = json.loads(USER_VECS_IDS.read_text())
    user_to_vec = {u: user_vecs[i] for i, u in enumerate(user_vec_ids)}
    log(f"  user vectors: {user_vecs.shape}, ids={len(user_vec_ids)}")

    # === Load model ===
    log(f"[1] Loading model {Path(MODEL_PATH).name} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    log(f"  model loaded, hidden_dim={H}")

    # === Load projector ===
    log(f"[2] Loading injector from {INJECTOR_CKPT.name} ...")
    ckpt = torch.load(INJECTOR_CKPT, map_location=DEVICE)
    proj_state = ckpt["proj"]
    user_dim = proj_state["mlp.0.weight"].shape[1]  # 318
    num_tokens = proj_state["mlp.2.weight"].shape[0] // H  # 16
    log(f"  user_dim={user_dim}, num_tokens={num_tokens}, model_dim={H}")
    proj = SoftPrefixProjector(
        user_dim=user_dim, hidden_dim=128,
        num_tokens=num_tokens, model_dim=H, dtype=DTYPE, gate_init=1e-3
    ).to(DEVICE)
    proj.load_state_dict(proj_state)
    proj.eval()

    # === Pre-compute all prefix embeddings ===
    log(f"[3] Pre-computing prefix embeddings ...")
    zero_prefix = torch.zeros(num_tokens, H, dtype=DTYPE, device=DEVICE)
    rng = np.random.default_rng(SEED_BASE)

    # For shuffled-z: pick a random OTHER user (deterministic per pair)
    user_ids_in_pairs = sorted(set(p["user_id"] for p in valid_pairs))

    # Build per-pair (z_user, z_other)
    pair_info = []
    for pi, p in enumerate(valid_pairs):
        uid = p["user_id"]
        if uid not in user_to_vec:
            continue
        z_user = user_to_vec[uid]
        # pick a random other user (NOT same uid)
        candidates_other = [u for u in user_ids_in_pairs if u != uid]
        if not candidates_other:
            continue
        other_uid = candidates_other[int(rng.integers(0, len(candidates_other)))]
        z_other = user_to_vec[other_uid]
        pair_info.append({
            "pair_idx": pi,
            "user_id": uid,
            "asin": p["asin"],
            "attrs": p.get("attrs_5") or p.get("attrs") or {},
            "z_user": z_user,
            "z_other_uid": other_uid,
            "z_other": z_other,
        })
    log(f"  pair_info: {len(pair_info)}")

    # Compute prefix tensors (real-z, shuffled-z) and group by control
    with torch.inference_mode():
        z_user_t = torch.tensor(
            np.stack([pi["z_user"] for pi in pair_info]),
            dtype=torch.float32, device=DEVICE).to(DTYPE)
        z_other_t = torch.tensor(
            np.stack([pi["z_other"] for pi in pair_info]),
            dtype=torch.float32, device=DEVICE).to(DTYPE)
        pref_real = proj(z_user_t)         # [n, K, H]
        pref_shuf = proj(z_other_t)        # [n, K, H]
    log(f"  pref_real: {pref_real.shape}, pref_shuf: {pref_shuf.shape}")

    # === Generate per control × seed ===
    log(f"[4] Generating: {len(pair_info)} pairs × {N_CONTROLS} controls × {N_SEEDS_PER_CONTROL} seeds = {len(pair_info) * N_CONTROLS * N_SEEDS_PER_CONTROL} total ...")

    # Build all jobs: list of (pair_idx, control, seed, prefix_tensor_or_None)
    jobs = []
    for pi_idx, pi in enumerate(pair_info):
        for ctrl in CONTROLS:
            for seed in range(N_SEEDS_PER_CONTROL):
                if ctrl == "real-z":
                    pe = pref_real[pi_idx]
                elif ctrl == "shuffled-z":
                    pe = pref_shuf[pi_idx]
                elif ctrl == "true-zero-z":
                    pe = zero_prefix
                else:  # injection-off
                    pe = None
                jobs.append((pi_idx, ctrl, seed, pe))
    log(f"  jobs: {len(jobs)}")

    torch.manual_seed(SEED_BASE)
    records = []
    t0 = time.time()
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        # Build prompts for chunk
        prompts = [chat_prompt(tokenizer, pair_info[pi_idx]["attrs"])
                   for pi_idx, _, _, _ in chunk]
        prefix_embeds = [pe for _, _, _, pe in chunk]
        texts = generate_batch(
            model, tokenizer, proj, prefix_embeds, prompts,
            max_new=MAX_NEW, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
            num_tokens=num_tokens, model_dim=H, dtype=DTYPE, device=DEVICE,
        )
        for k, ((pi_idx, ctrl, seed, _), txt) in enumerate(zip(chunk, texts)):
            pi = pair_info[pi_idx]
            records.append({
                "user_id": pi["user_id"],
                "asin": pi["asin"],
                "attrs": pi["attrs"],
                "control": ctrl,
                "seed": seed,
                "candidate_query": txt,
                "shuffled_user": pi["z_other_uid"],
            })
        if start % (BATCH * 16) == 0:
            elapsed = time.time() - t0
            rate = (start + len(chunk)) / max(elapsed, 0.001)
            eta = (len(jobs) - start) / max(rate, 0.001)
            log(f"  [{start + len(chunk)}/{len(jobs)}] elapsed {elapsed:.1f}s, rate={rate:.2f}/s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    log(f"  generated {len(records)} in {elapsed:.1f}s, rate={len(records)/elapsed:.2f}/s")

    # === Save ===
    log(f"[5] Saving generations to {OUT_GENERATIONS} ...")
    with OUT_GENERATIONS.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GENERATIONS}")

    # === Summary ===
    summary = {
        "version": "phase10_16_b_v1",
        "n_pairs": len(pair_info),
        "n_controls": N_CONTROLS,
        "n_seeds_per_control": N_SEEDS_PER_CONTROL,
        "n_total_generations": len(records),
        "controls": CONTROLS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW,
        "num_tokens_prefix": num_tokens,
        "model": Path(MODEL_PATH).name,
        "injector_checkpoint": str(INJECTOR_CKPT),
        "elapsed_sec": elapsed,
        "rate_per_sec": len(records) / elapsed,
    }
    summary_path = OUT_DIR / "phase10_16_b_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"  → {summary_path}")

    log("=" * 70)
    log("PHASE 10.16.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()