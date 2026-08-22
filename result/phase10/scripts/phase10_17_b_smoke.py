#!/usr/bin/env python3
"""Mini smoke for Phase 10.17.B batched rewrite: 5 pairs, 4 controls, 1 seed.
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

T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

N_PAIRS = 5
N_SEEDS = 1
CONTROLS = ["real-z", "shuffled-z", "true-zero-z", "injection-off"]
BATCH = 4  # pairs per batch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _generate_k_prefix_batched(model, prompts_token_lists, styles_list,
                                  k_prefix, alpha, max_new_tokens,
                                  temperature, top_p, num_seeds_per_sample=1,
                                  pad_token_id=0, zero_emb_768=None):
    bsz = len(prompts_token_lists)
    H = model.proj.weight.shape[0]
    D_in = model.proj.weight.shape[1]  # 768 (style embed dim)
    device = next(model.parameters()).device
    dtype = model.proj.weight.dtype
    max_p = max(len(p) for p in prompts_token_lists)
    pad_id = pad_token_id
    input_ids = torch.full((bsz, max_p), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompts_token_lists):
        input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_id).long()
    has_prefix = any(s is not None for s in styles_list)
    if has_prefix:
        # Build [bsz, D_in] style tensor (zero_emb for None)
        style_in = torch.zeros(bsz, D_in, dtype=dtype, device=device)
        if zero_emb_768 is None:
            zero_emb_768 = torch.zeros(D_in, dtype=dtype, device=device)
        for i, s in enumerate(styles_list):
            if s is not None:
                style_in[i] = s.to(dtype=dtype, device=device)
        # Per-sample proj
        style_proj = model.proj(style_in * alpha)  # [bsz, H]
        style_prefix = style_proj.unsqueeze(1).expand(bsz, k_prefix, H).contiguous()
        prefix_mask = torch.ones((bsz, k_prefix), dtype=torch.long, device=device)
        full_attn = torch.cat([prefix_mask, attention_mask], dim=1)
    else:
        style_prefix = None
        full_attn = attention_mask
    input_embeds = model.model.shared(input_ids)
    if has_prefix:
        full_embeds = torch.cat([style_prefix, input_embeds], dim=1)
    else:
        full_embeds = input_embeds
    out_ids = model.model.generate(
        inputs_embeds=full_embeds, attention_mask=full_attn,
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        num_return_sequences=num_seeds_per_sample,
    )
    return out_ids


def main():
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

    log(f"Mini smoke: {N_PAIRS} pairs × 4 controls × {N_SEEDS} seed = {N_PAIRS*4*N_SEEDS} cands")

    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    pairs = pairs[:N_PAIRS]

    npz = np.load(USER_EMBS_NPZ, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    embs = npz["embs"]
    uid_to_emb = {u: embs[i].astype(np.float32) for i, u in enumerate(user_ids)}

    from tinystyler import TinyStyler
    log("loading TinyStyler...")
    t0 = time.time()
    model = TinyStyler(base_model=T5_BASE, use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu')
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = model.state_dict()
    cur.update(saved)
    model.load_state_dict(cur)
    model.to('cuda:0').half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  loaded in {time.time()-t0:.1f}s")

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_SNAP, legacy=True)
    rng = np.random.default_rng(42)
    all_uid = sorted(uid_to_emb.keys())
    zero_emb = torch.zeros(768, dtype=torch.half, device='cuda:0')

    # Pre-tokenize
    pi_tokens = [tokenizer(p["prompts"]["A_no_style"], padding=False, truncation=True,
                           max_length=256)['input_ids'] for p in pairs]

    CHUNK = max(1, BATCH // N_SEEDS)  # pairs per chunk
    log(f"  CHUNK = {CHUNK} pairs/batch = {CHUNK * 4 * N_SEEDS} cands/batch")

    records = []
    t0 = time.time()
    for chunk_start in range(0, len(pairs), CHUNK):
        chunk_pis = list(range(chunk_start, min(chunk_start + CHUNK, len(pairs))))
        prompts_tok, styles, sample_meta = [], [], []
        for pi_idx in chunk_pis:
            p = pairs[pi_idx]
            uid = p["user_id"]
            if uid not in uid_to_emb:
                continue
            other_uid = rng.choice([u for u in all_uid if u != uid])
            for ctrl in CONTROLS:
                for _ in range(N_SEEDS):
                    prompts_tok.append(pi_tokens[pi_idx])
                if ctrl == "injection-off":
                    styles.append(None)
                elif ctrl == "true-zero-z":
                    styles.append(zero_emb.clone())
                elif ctrl == "real-z":
                    styles.append(torch.from_numpy(uid_to_emb[uid]).to('cuda:0'))
                else:
                    styles.append(torch.from_numpy(uid_to_emb[other_uid]).to('cuda:0'))
                sample_meta.append((pi_idx, ctrl, uid, other_uid))
        with torch.no_grad():
            out_ids = _generate_k_prefix_batched(
                model, prompts_tok, styles,
                k_prefix=8, alpha=2.0, max_new_tokens=48,
                temperature=1.0, top_p=0.95,
                num_seeds_per_sample=N_SEEDS,
                pad_token_id=tokenizer.pad_token_id or 0,
                zero_emb_768=zero_emb,
            )
        idx = 0
        for pi_idx, ctrl, uid, other_uid in sample_meta:
            p = pairs[pi_idx]
            for seed in range(N_SEEDS):
                txt = tokenizer.decode(out_ids[idx], skip_special_tokens=True).strip()
                records.append({"user_id": uid, "asin": p["asin"], "attrs": p["attrs"],
                                "control": ctrl, "seed": seed,
                                "candidate_query": txt, "shuffled_user": other_uid})
                idx += 1
        log(f"  chunk {chunk_start}: {len(records)} records in {time.time()-t0:.1f}s, "
            f"sample: {records[-1]['candidate_query'][:80]}")

    log(f"\nTOTAL: {len(records)} records, rate={len(records)/(time.time()-t0):.2f}/s")
    log("SMOKE OK")


if __name__ == "__main__":
    main()