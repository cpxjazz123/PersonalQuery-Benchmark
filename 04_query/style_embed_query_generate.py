#!/usr/bin/env python3
"""Style-Embedding Injection Query Generator — Inference.

Loads trained LoRA adapter + style projection. For each user (with their
precomputed user style vector), generates a query aligned with their writing
style via greedy decoding.

Inputs:
- LoRA adapter dir (contains adapter_config.json + adapter_model.safetensors)
- style_projection.pt (Linear projection weights)
- user_vecs.npz (user_id -> 1024-dim style vector)

Outputs:
- output_dir/<category>_style_embed_queries.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
USER_DIM = 1024


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class StyleInjectedGenerator(nn.Module):
    """Qwen2-7B + LoRA + style prefix injection."""

    def __init__(self, base_model, user_dim: int = USER_DIM):
        super().__init__()
        self.base = base_model
        target_dtype = next(self.base.parameters()).dtype
        # Style projection placeholder; will be loaded from state_dict
        self.style_proj = nn.Linear(user_dim, base_model.config.hidden_size, bias=False).to(target_dtype)

    def forward(self, input_ids, user_vecs, attention_mask=None):
        embed_fn = self.base.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)
        target_dtype = next(self.base.parameters()).dtype
        inputs_embeds = inputs_embeds.to(target_dtype)
        style_prefix = self.style_proj(user_vecs.to(target_dtype)).unsqueeze(1)
        inputs_embeds = torch.cat([style_prefix, inputs_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        else:
            # Prepend 1 for style prefix
            prefix_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)


def load_model(model_dir: Path):
    log("loading Qwen2-7B base...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "lora_adapter", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    log("loading LoRA adapter...")
    base = PeftModel.from_pretrained(base, str(model_dir / "lora_adapter"))
    base.eval()
    model = StyleInjectedGenerator(base).to("cuda:0")
    log("loading style projection...")
    state = torch.load(model_dir / "style_projection.pt", map_location="cuda:0")
    model.style_proj.load_state_dict(state)
    model.eval()
    log("loading user vecs...")
    user_vecs_npz = np.load(model_dir / "user_vecs.npz")
    user_vecs = {uid: user_vecs_npz[uid] for uid in user_vecs_npz.files}
    return model, tokenizer, user_vecs


def generate_query(model, tokenizer, user_vec: np.ndarray, asin: str, max_new_tokens: int = 32) -> str:
    prompt = f"Generate a shopping query for ASIN {asin}."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    user_vecs_t = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        # Manually call base to use inputs_embeds
        embed_fn = model.base.get_input_embeddings()
        target_dtype = next(model.base.parameters()).dtype
        inputs_embeds = embed_fn(input_ids).to(target_dtype)
        style_prefix = model.style_proj(user_vecs_t.to(target_dtype)).unsqueeze(1)
        full_embeds = torch.cat([style_prefix, inputs_embeds], dim=1)
        attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device="cuda:0")
        out = model.base.generate(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


def generate_for_records(
    model, tokenizer, user_vecs: Dict[str, np.ndarray], records: List[Dict]
) -> List[Dict]:
    out = []
    for i, r in enumerate(records):
        uid = r["user_id"]
        if uid not in user_vecs:
            out.append({**r, "generated_query": None})
            continue
        try:
            q = generate_query(model, tokenizer, user_vecs[uid], r["asin"])
        except Exception as e:
            log(f"  [{i}] {uid} {r['asin']}: error {e}")
            q = None
        out.append({**r, "generated_query": q})
        if (i + 1) % 10 == 0:
            log(f"  generated {i + 1}/{len(records)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="dir with lora_adapter/, style_projection.pt, user_vecs.npz")
    ap.add_argument("--category", required=True)
    ap.add_argument("--out_dir", default="/home/wlia0047/hj82_scratch2/wenyu/RAG/style_embed_queries")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0=all records")
    ap.add_argument("--use_existing_records", action="store_true", help="reuse 04_query records as input asin list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    model, tokenizer, user_vecs = load_model(model_dir)
    log(f"loaded {len(user_vecs)} user vecs")

    # Build records: list of (user_id, asin)
    if args.use_existing_records:
        records_p = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/04_query") / args.category / "query_by_syntax_depth_no_depth_check_10.json"
        with open(records_p) as f:
            data = json.load(f)
        records = [{"user_id": r["user_id"], "asin": r["asin"]} for r in data]
    else:
        # Build from user_vecs keys only (no asin)
        records = [{"user_id": uid, "asin": ""} for uid in user_vecs.keys()]
    if args.limit:
        records = records[: args.limit]
    log(f"generating queries for {len(records)} records...")

    results = generate_for_records(model, tokenizer, user_vecs, records)
    out_p = out_dir / f"{args.category}_style_embed_queries.json"
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"wrote {out_p}")
    log("=== done ===")


if __name__ == "__main__":
    main()