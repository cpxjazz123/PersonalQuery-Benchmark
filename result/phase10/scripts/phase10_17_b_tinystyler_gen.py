#!/usr/bin/env python3
"""Phase 10.17.B: TinyStyler范式 4-control batch generation.

Reuses E30.34 (gen_styled_tinystyler_noforce.py) verified stack:
  - TinyStyler (T5-v1_1-large + 1-layer Linear proj 768->1024)
  - HuggingFace tinystyler/tinystyler weights
  - 768d AnnaWegmann style embedding (Phase 10.17.A cache)
  - Prefix: K=8, alpha=2.0, NO LayerNorm

4 controls per (user, asin):
  A real-z       : style = AnnaWegmann(u_target) * alpha → proj (correct user)
  B shuffled-z   : style = AnnaWegmann(u_other)  * alpha → proj (wrong user, sanity)
  C true-zero-z  : style = zeros(768) * alpha → proj  (no signal)
  D injection-off: no prefix (baseline)

Per control: K_SEEDS = 4 candidates (seeds 0..3)
Total: 876 pairs × 4 controls × 4 seeds = 14016 candidates
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
sys.path.insert(0, str(REPO_ROOT / "TinyStyler" / "tinystyler"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
USER_EMBS_NPZ = OUT_DIR / "phase10_user_embs_768d.npz"
OUT_GENERATIONS = OUT_DIR / "phase10_17_b_generations_4control.jsonl"

# === TinyStyler model paths ===
T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

# === Hard-coded config (CLAUDE.md Rule 3) ===
N_CONTROLS = 4
N_SEEDS_PER_CONTROL = 4
SEED_BASE = 7777
BATCH = 16  # T5-large fp16 ~3GB; 64GB free → batch 16 fits; E30.34 used 2 but we partition into 2 fwd (prefix + no-prefix), so effective throughput ~CHUNK*4 pairs/batch
MAX_NEW_TOKENS = 48  # 48 tokens sufficient for shopping queries
TEMPERATURE = 1.0
TOP_P = 0.95
STYLE_K_PREFIX = 8   # E30.34 verified
STYLE_ALPHA = 2.0    # E30.34 verified

CONTROLS = ["real-z", "shuffled-z", "true-zero-z", "injection-off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _generate_k_prefix_batched(model, prompts_token_lists, styles_list,
                                  k_prefix, alpha, max_new_tokens,
                                  temperature, top_p, num_seeds_per_sample=1,
                                  pad_token_id=0, zero_emb_768=None):
    """Batched generation where each sample has its own style prefix.

    prompts_token_lists: list[list[int]] — each sample's tokenized prompt.
    styles_list: list[Optional[Tensor[768]]] — each sample's style embedding (None = no prefix).

    Strategy:
      - We build per-sample input_embeds + attention_mask.
      - If styles_list has any non-None entry, prepend a [bsz, k_prefix, H] prefix.
      - For samples with style=None (injection-off), we DO NOT add prefix at all.
        → Instead, we partition: prefix_samples (have style) vs no_prefix_samples (no style).
        → For no_prefix_samples, we run a separate generate call without prefix.
      - All prefix-bearing samples get `style_proj = model.proj(style_in * alpha)`,
        with style_in being zeros for "true-zero-z" control.
    """
    bsz = len(prompts_token_lists)
    H = model.proj.weight.shape[0]  # 1024 (T5 d_model)
    D_in = model.proj.weight.shape[1]  # 768 (style embed dim)
    device = next(model.parameters()).device
    dtype = model.proj.weight.dtype

    # Partition into prefix-bearing vs no-prefix
    pi_idx_with_prefix = [i for i, s in enumerate(styles_list) if s is not None]
    pi_idx_no_prefix = [i for i, s in enumerate(styles_list) if s is None]

    all_outputs: list = [None] * bsz  # filled with [num_seeds, max_new] per sample

    # --- 1) Prefix-bearing samples ---
    if pi_idx_with_prefix:
        bsz_p = len(pi_idx_with_prefix)
        prompts_p = [prompts_token_lists[i] for i in pi_idx_with_prefix]
        styles_p = [styles_list[i] for i in pi_idx_with_prefix]

        max_p = max(len(p) for p in prompts_p)
        pad_id = pad_token_id
        input_ids = torch.full((bsz_p, max_p), pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(prompts_p):
            input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()

        # Build [bsz_p, D_in] style tensor (zero for None-style samples)
        style_in = torch.zeros(bsz_p, D_in, dtype=dtype, device=device)
        for i, s in enumerate(styles_p):
            if s is not None:
                style_in[i] = s.to(dtype=dtype, device=device)
        # Per-sample proj (vectorized)
        style_proj = model.proj(style_in * alpha)  # [bsz_p, H]
        style_prefix = style_proj.unsqueeze(1).expand(bsz_p, k_prefix, H).contiguous()
        prefix_mask = torch.ones((bsz_p, k_prefix), dtype=torch.long, device=device)
        full_attn = torch.cat([prefix_mask, attention_mask], dim=1)
        input_embeds = model.model.shared(input_ids)
        full_embeds = torch.cat([style_prefix, input_embeds], dim=1)
        out_ids = model.model.generate(
            inputs_embeds=full_embeds, attention_mask=full_attn,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            num_return_sequences=num_seeds_per_sample,
        )
        # Group back: each input sample produced `num_seeds_per_sample` outputs
        for j, pi_idx in enumerate(pi_idx_with_prefix):
            all_outputs[pi_idx] = out_ids[j * num_seeds_per_sample: (j + 1) * num_seeds_per_sample]

    # --- 2) No-prefix samples (injection-off) ---
    if pi_idx_no_prefix:
        bsz_n = len(pi_idx_no_prefix)
        prompts_n = [prompts_token_lists[i] for i in pi_idx_no_prefix]
        max_p = max(len(p) for p in prompts_n)
        pad_id = pad_token_id
        input_ids = torch.full((bsz_n, max_p), pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(prompts_n):
            input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()
        out_ids = model.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            num_return_sequences=num_seeds_per_sample,
        )
        for j, pi_idx in enumerate(pi_idx_no_prefix):
            all_outputs[pi_idx] = out_ids[j * num_seeds_per_sample: (j + 1) * num_seeds_per_sample]

    # Return flat list of [num_seeds, max_new] tensors in original order
    return all_outputs


def main():
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

    log("=" * 70)
    log("Phase 10.17.B: TinyStyler 4-control batch generation")
    log("=" * 70)

    # === Load pairs ===
    log("[1] Loading pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    # === Load AnnaWegmann 768d user embeddings ===
    log("[2] Loading 768d AnnaWegmann embeddings ...")
    npz = np.load(USER_EMBS_NPZ, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    embs = npz["embs"]  # (876, 768)
    uid_to_emb = {u: embs[i].astype(np.float32) for i, u in enumerate(user_ids)}
    log(f"  embeddings: {embs.shape}, ids: {len(user_ids)}")

    # === Load TinyStyler ===
    log(f"[3] Loading TinyStyler model from {Path(TINYSTYLER_WEIGHTS).name} ...")
    from tinystyler import TinyStyler
    model = TinyStyler(base_model=T5_BASE, use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu')
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = model.state_dict()
    cur.update(saved)
    model.load_state_dict(cur)
    model.to('cuda:0').half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  loaded TinyStyler")

    # === Load T5 tokenizer ===
    from transformers import T5Tokenizer
    log(f"[4] Loading T5 tokenizer ...")
    tokenizer = T5Tokenizer.from_pretrained(T5_SNAP, legacy=True)
    log(f"  loaded tokenizer")

    # === Build per-pair (user, other_user, attrs, prompt) ===
    log("[5] Building per-pair info ...")
    rng = np.random.default_rng(SEED_BASE)
    all_uid = sorted(uid_to_emb.keys())
    # zero_emb dtype must match model.proj (half)
    zero_emb = torch.zeros(768, dtype=torch.half, device='cuda:0')
    pair_info = []
    for pi, p in enumerate(pairs):
        uid = p["user_id"]
        if uid not in uid_to_emb:
            continue
        candidates_other = [u for u in all_uid if u != uid]
        if not candidates_other:
            continue
        other_uid = candidates_other[int(rng.integers(0, len(candidates_other)))]
        attrs = p["attrs"]
        # 5 main attrs (E30.34 uses 4 attrs; use all 5 for higher coverage)
        if not all(attrs.get(k) for k in ["Brand", "Color", "Material"]):
            continue
        pair_info.append({
            "pair_idx": pi,
            "user_id": uid,
            "asin": p["asin"],
            "attrs": attrs,
            "z_user_emb": uid_to_emb[uid],
            "z_other_uid": other_uid,
            "z_other_emb": uid_to_emb[other_uid],
            "prompt_A_no_style": p["prompts"]["A_no_style"],
        })
    log(f"  pair_info: {len(pair_info)}")

    # === Build all jobs ===
    log(f"[6] Building jobs: {len(pair_info)} pairs × {N_CONTROLS} controls × {N_SEEDS_PER_CONTROL} seeds = {len(pair_info) * N_CONTROLS * N_SEEDS_PER_CONTROL} jobs ...")
    jobs = []
    for pi_idx, pi in enumerate(pair_info):
        for ctrl in CONTROLS:
            for seed in range(N_SEEDS_PER_CONTROL):
                if ctrl == "real-z":
                    style = pi["z_user_emb"]
                elif ctrl == "shuffled-z":
                    style = pi["z_other_emb"]
                elif ctrl == "true-zero-z":
                    style = None  # use zero_emb below
                else:  # injection-off
                    style = None
                jobs.append((pi_idx, ctrl, seed, style))
    log(f"  jobs: {len(jobs)}")

    # === Generate batched (one forward per chunk of BATCH samples) ===
    # Each sample is one (pair, control) combo with num_seeds=N_SEEDS_PER_CONTROL candidates.
    # We chunk samples into batches of BATCH; within each batch, run _generate_k_prefix_batched.
    torch.manual_seed(SEED_BASE)
    records = []
    t0 = time.time()

    # Pre-tokenize all prompts (one per pair)
    log(f"  pre-tokenizing {len(pair_info)} prompts ...")
    pi_prompts_tokens = []
    for pi in pair_info:
        toks = tokenizer(pi["prompt_A_no_style"], padding=False, truncation=True,
                          max_length=256)['input_ids']
        pi_prompts_tokens.append(toks)
    log(f"  tokenized, sample lens: min={min(len(t) for t in pi_prompts_tokens)}, "
        f"max={max(len(t) for t in pi_prompts_tokens)}")

    # Build sample list: each (pi_idx, ctrl) -> BATCH * N_SEEDS candidates
    # Group by pi_idx: per pair, all 4 controls in 1 batch
    n_done = 0
    CHUNK = max(1, BATCH // N_SEEDS_PER_CONTROL)  # how many (pair) per chunk
    log(f"  CHUNK = {CHUNK} pairs/batch = {CHUNK * N_SEEDS_PER_CONTROL} candidates/batch")

    pi_index_list = list(range(len(pair_info)))
    for chunk_start in range(0, len(pi_index_list), CHUNK):
        chunk_pis = pi_index_list[chunk_start:chunk_start + CHUNK]
        # Build per-sample prompts and styles
        prompts_tok = []  # list of list of tokens (one per sample, no N_SEEDS replicates — handled inside)
        styles = []       # one per sample
        sample_meta = []  # (pi_idx, ctrl) for each sample
        for pi_idx in chunk_pis:
            pi = pair_info[pi_idx]
            for ctrl in CONTROLS:
                # prompt: shared across seeds; we add once per (pi_idx, ctrl)
                prompts_tok.append(pi_prompts_tokens[pi_idx])
                # style: pick one style for this (pair, control), shared across seeds
                if ctrl == "injection-off":
                    styles.append(None)
                elif ctrl == "true-zero-z":
                    styles.append(zero_emb.clone())
                elif ctrl == "real-z":
                    styles.append(torch.from_numpy(pi["z_user_emb"]).to('cuda:0'))
                else:  # shuffled-z
                    styles.append(torch.from_numpy(pi["z_other_emb"]).to('cuda:0'))
                sample_meta.append((pi_idx, ctrl))
        # Now generate (returns list of [num_seeds, max_new] tensors per sample)
        with torch.no_grad():
            per_sample_out_ids = _generate_k_prefix_batched(
                model,
                prompts_tok, styles,
                k_prefix=STYLE_K_PREFIX,
                alpha=STYLE_ALPHA,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_seeds_per_sample=N_SEEDS_PER_CONTROL,
                pad_token_id=tokenizer.pad_token_id or 0,
                zero_emb_768=zero_emb,
            )
        # Decode each sample's N_SEEDS outputs
        for s_idx, (pi_idx, ctrl) in enumerate(sample_meta):
            pi = pair_info[pi_idx]
            for seed in range(N_SEEDS_PER_CONTROL):
                txt = tokenizer.decode(per_sample_out_ids[s_idx][seed], skip_special_tokens=True).strip()
                records.append({
                    "user_id": pi["user_id"],
                    "asin": pi["asin"],
                    "attrs": pi["attrs"],
                    "control": ctrl,
                    "seed": seed,
                    "candidate_query": txt,
                    "shuffled_user": pi["z_other_uid"],
                })
        n_done = len(records)
        if chunk_start % (CHUNK * 20) == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 0.001)
            eta = (len(jobs) - n_done) / max(rate, 0.001)
            log(f"  [{n_done}/{len(jobs)}] elapsed {elapsed:.1f}s, rate={rate:.2f}/s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    log(f"  generated {len(records)} in {elapsed:.1f}s, rate={len(records)/elapsed:.2f}/s")

    # === Save ===
    log(f"[7] Saving generations to {OUT_GENERATIONS} ...")
    with OUT_GENERATIONS.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GENERATIONS}")

    # === Summary ===
    summary = {
        "version": "phase10_17_b_v1",
        "model": "TinyStyler (T5-v1_1-large + 768->1024 Linear proj)",
        "weights_source": TINYSTYLER_WEIGHTS,
        "style_embed_source": "AnnaWegmann/Style-Embedding (phase10_17_a)",
        "n_pairs": len(pair_info),
        "n_controls": N_CONTROLS,
        "n_seeds_per_control": N_SEEDS_PER_CONTROL,
        "n_total_generations": len(records),
        "controls": CONTROLS,
        "k_prefix": STYLE_K_PREFIX,
        "alpha": STYLE_ALPHA,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_sec": elapsed,
        "rate_per_sec": len(records) / elapsed,
    }
    SUMMARY_OUT = OUT_DIR / "phase10_17_b_meta.json"
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"  → {SUMMARY_OUT}")

    log("=" * 70)
    log("PHASE 10.17.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()