#!/usr/bin/env python3
"""Smoke test for Phase 10.16.B: 5 pairs × 4 controls × 1 seed = 20 generations.

Verify:
  1. Qwen 7B loads OK
  2. Projector loads OK
  3. Per-pair prefix embeddings shape correct
  4. generate_batch produces text (no crash)
  5. Per-control outputs are different from each other
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path = [str(REPO_ROOT), str(REPO_ROOT / "query" / "soft_prefix")] + sys.path

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
USER_VECS_NPY = OUT_DIR / "phase10_user_vectors_318d.npy"
USER_VECS_IDS = OUT_DIR / "phase10_user_vectors_318d_user_ids.json"
INJECTOR_CKPT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_injector_best.pt"
MODEL_PATH = "/fs04/scratch2/hj82/yubow/phd2026/open_models/Qwen/Qwen2.5-Coder-7B-Instruct"

N_PAIRS = 5
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

from projector import SoftPrefixProjector

def attr_prompt(attrs):
    lines = ["Product attributes:"]
    for k, v in attrs.items():
        if v:
            lines.append(f"{k}: {v}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)

SYSTEM_PROMPT = ("You are a shopping query writer. Write one short natural "
                 "shopping query that mentions every listed attribute of the product.")

def chat_prompt(tokenizer, attrs):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": attr_prompt(attrs)},
    ]
    out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "input_ids"):
        return out.input_ids
    return list(out)


@torch.no_grad()
def gen_one(model, tok, proj, prefix_or_none, prompt_ids, max_new=40, H=3584, num_tokens=16):
    emb = model.get_input_embeddings()
    text_emb = emb(prompt_ids.unsqueeze(0)).to(DTYPE)
    if prefix_or_none is not None:
        full = torch.cat([prefix_or_none, text_emb], dim=1)
        attn = torch.ones(1, full.size(1), dtype=torch.long, device=DEVICE)
        pos = torch.arange(full.size(1), device=DEVICE).unsqueeze(0)
    else:
        full = text_emb
        attn = torch.ones(1, full.size(1), dtype=torch.long, device=DEVICE)
        pos = torch.arange(full.size(1), device=DEVICE).unsqueeze(0)
    past = None
    next_emb = full
    next_ids = None
    generated = []
    for step in range(max_new):
        first = past is None
        out = model(
            inputs_embeds=next_emb if first else None,
            input_ids=None if first else next_ids,
            attention_mask=attn,
            position_ids=pos if first else None,
            use_cache=True, past_key_values=past)
        past = out.past_key_values
        if first:
            last_valid = attn.sum(dim=1) - 1
            logits = out.logits[0, last_valid]
        else:
            logits = out.logits[0, -1]
        logits = logits.float() / 0.7
        topk_v, _ = torch.topk(logits, 40)
        v = topk_v.min()  # 40th-largest value
        logits = torch.where(logits < v, torch.full_like(logits, float("-inf")), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        tid = int(torch.multinomial(probs, 1).item())
        generated.append(tid)
        if tid == tok.eos_token_id:
            break
        next_ids = torch.tensor([[tid]], device=DEVICE)
        attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=DEVICE)], dim=1)
    return tok.decode(generated, skip_special_tokens=True).strip()


def main():
    print(f"[smoke] loading...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    valid = [p for p in pairs if all((p.get("attrs_5") or p.get("attrs") or {}).get(k) for k in ["Brand", "Color", "Material"])]
    valid = valid[:N_PAIRS]

    user_vecs = np.load(USER_VECS_NPY)
    user_vec_ids = json.loads(USER_VECS_IDS.read_text())
    user_to_vec = {u: user_vecs[i] for i, u in enumerate(user_vec_ids)}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    ckpt = torch.load(INJECTOR_CKPT, map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=318, hidden_dim=128, num_tokens=16, model_dim=3584, dtype=DTYPE, gate_init=1e-3).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()

    print(f"[smoke] generating {N_PAIRS} × 4 controls = {N_PAIRS * 4} generations")
    rng = np.random.default_rng(42)
    other_users = [u for u in user_vec_ids if u not in [p["user_id"] for p in valid]]
    t0 = time.time()
    for pi, p in enumerate(valid):
        uid = p["user_id"]
        attrs = p.get("attrs_5") or p.get("attrs") or {}
        z_user = user_to_vec.get(uid)
        if z_user is None:
            print(f"  pair {pi}: user {uid} not in user_vecs")
            continue
        z_user_t = torch.tensor(z_user, dtype=torch.float32, device=DEVICE).to(DTYPE).unsqueeze(0)
        other_uid = other_users[rng.integers(0, len(other_users))]
        z_other_t = torch.tensor(user_to_vec[other_uid], dtype=torch.float32, device=DEVICE).to(DTYPE).unsqueeze(0)
        with torch.no_grad():
            pref_real = proj(z_user_t)  # [1, 16, 3584]
            pref_shuf = proj(z_other_t)
            pref_zero = torch.zeros_like(pref_real)
        prompt_ids = torch.tensor(chat_prompt(tok, attrs), device=DEVICE)
        out_real = gen_one(model, tok, proj, pref_real, prompt_ids)
        out_shuf = gen_one(model, tok, proj, pref_shuf, prompt_ids)
        out_zero = gen_one(model, tok, proj, pref_zero, prompt_ids)
        out_off = gen_one(model, tok, proj, None, prompt_ids)
        print(f"\n=== pair {pi} (user={uid[:20]}...) asin={p['asin'][:20]} ===")
        print(f"  real-z   : {out_real[:120]}")
        print(f"  shuffled : {out_shuf[:120]}")
        print(f"  zero-z   : {out_zero[:120]}")
        print(f"  off      : {out_off[:120]}")
    print(f"\n[smoke] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()