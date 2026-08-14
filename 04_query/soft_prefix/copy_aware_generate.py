#!/usr/bin/env python3
"""E14-P — Copy-aware free-form query generation: inference.

Greedy decoding over the MIXED distribution:
    P(token) = p_copy * P_copy + (1 - p_copy) * P_generate
where P_copy is the pointer distribution over the product-attribute spans and
P_generate is the base LM distribution. The user style condition (VADES
user_mu soft prefix) enters ONLY P_generate (P7): swapping the user condition
must leave attribute strings untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from user_style_vectors import (  # noqa: E402
    UserVectorProvider,
    load_vades_profiles,
)
from projector import SoftPrefixProjector  # noqa: E402
from copy_aware import (  # noqa: E402
    CopyAwareHead,
    build_attr_prompt_lines,
    attr_token_spans,
    mixed_logits,
)
from copy_aware_train import build_messages  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class CopyAwareGenerator(nn.Module):
    def __init__(self, base_model, projector, copy_head, device: str):
        super().__init__()
        self.base = base_model
        self.projector = projector
        self.copy_head = copy_head
        self.device = device
        self.num_tokens = projector.num_tokens
        self.base.eval()
        self.projector.eval()
        self.copy_head.eval()

    @torch.no_grad()
    def generate(self, prompt_str: str, user_vec: Optional[np.ndarray], attrs: Dict[str, str],
                 tokenizer, max_new_tokens: int, eos_token_id: int) -> str:
        target_dtype = next(self.base.parameters()).dtype
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(self.device)
        spans = attr_token_spans(prompt_str, attrs, tokenizer)
        src_attr_mask = torch.zeros(prompt_ids.size(1), dtype=torch.bool, device=self.device)
        for sp in spans.values():
            for (a, b) in sp:
                src_attr_mask[a:b + 1] = True

        emb = self.base.get_input_embeddings()
        text_emb = emb(prompt_ids).to(target_dtype)
        if self.num_tokens > 0 and user_vec is not None:
            z = torch.as_tensor(user_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            prefix = self.projector(z.to(target_dtype))
            full_emb = torch.cat([prefix, text_emb], dim=1)
            attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=self.device)
            prefix_pos = torch.arange(self.num_tokens, device=self.device).unsqueeze(0)
            text_pos = torch.arange(prompt_ids.size(1), device=self.device).unsqueeze(0) + self.num_tokens
            position_ids = torch.cat([prefix_pos, text_pos], dim=1)
            src_hidden_offset = self.num_tokens
        else:
            full_emb = text_emb
            attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=self.device)
            position_ids = torch.arange(prompt_ids.size(1), device=self.device).unsqueeze(0)
            src_hidden_offset = 0

        # first forward: get source hidden (prompt part) and first logits
        out = self.base(
            inputs_embeds=full_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=None,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        src_hidden = hidden[:, src_hidden_offset: src_hidden_offset + prompt_ids.size(1), :]
        gen_logits = out.logits[:, -1]  # [1, V] (no padding here)
        gen_hidden = hidden[:, -1:, :]
        p_copy, copy_logits = self.copy_head(gen_hidden, src_hidden, src_attr_mask.unsqueeze(0), prompt_ids)
        mixed = mixed_logits(gen_logits.unsqueeze(1), p_copy, copy_logits)[0, 0]

        generated = []
        next_id = int(torch.argmax(mixed))
        generated.append(next_id)
        past = out.past_key_values
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)], dim=1)
        next_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens - 1):
            if next_id == eos_token_id:
                break
            out = self.base(
                input_ids=next_ids,
                attention_mask=attention_mask,
                position_ids=None,
                use_cache=True,
                past_key_values=past,
                output_hidden_states=True,
            )
            past = out.past_key_values
            gen_logits = out.logits[:, -1]
            gen_hidden = out.hidden_states[-1][:, -1:, :]
            p_copy, copy_logits = self.copy_head(gen_hidden, src_hidden, src_attr_mask.unsqueeze(0), prompt_ids)
            mixed = mixed_logits(gen_logits.unsqueeze(1), p_copy, copy_logits)[0, 0]
            next_id = int(torch.argmax(mixed))
            generated.append(next_id)
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vector_mode", default="vades")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    cfg = json.load(open(ckpt / "config.json"))
    tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model_path"], torch_dtype=torch.bfloat16,
                                                device_map="cuda:0", trust_remote_code=True)
    lora_dir = ckpt / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(lora_dir))
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    projector = SoftPrefixProjector(cfg["user_dim"], 128, cfg["num_tokens"], cfg["model_dim"],
                                    dtype=torch.bfloat16).to("cuda:0")
    projector.load_state_dict(torch.load(ckpt / "projector.pt", map_location="cuda:0"))
    vocab_size = base.get_output_embeddings().weight.size(0)
    copy_head = CopyAwareHead(cfg["model_dim"], vocab_size, dtype=torch.bfloat16).to("cuda:0")
    copy_head.load_state_dict(torch.load(ckpt / "copy_head.pt", map_location="cuda:0"))
    model = CopyAwareGenerator(base, projector, copy_head, device="cuda:0")
    log(f"loaded checkpoint: K={cfg['num_tokens']} mode={args.vector_mode}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    profiles = load_vades_profiles(args.category)
    provider = UserVectorProvider(args.vector_mode, profiles, [r["user_id"] for r in records], seed=42)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, rec in enumerate(records):
        uid, asin = rec["user_id"], rec["asin"]
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        if not attrs:
            for cand in rec.get("syntax_depth_queries", []):
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if not attrs:
            results.append({"user_id": uid, "asin": asin, "generated_query": None, "error": "no attrs"})
            continue
        prompt_str = tokenizer.apply_chat_template(build_messages(attrs), tokenize=False, add_generation_prompt=True)
        vec, has_vector = provider.get(uid)
        q = model.generate(prompt_str, vec, attrs, tokenizer, args.max_new_tokens, tokenizer.eos_token_id)
        results.append({"user_id": uid, "asin": asin, "generated_query": q, "vector_mode": args.vector_mode,
                        "has_vector": bool(has_vector)})
        if (i + 1) % 10 == 0:
            log(f"  generated {i + 1}/{len(records)}")

    out_p = out_dir / f"{args.category}_copyaware_vec={args.vector_mode}.json"
    json.dump({"config": {**cfg, "vector_mode": args.vector_mode}, "results": results}, open(out_p, "w"), indent=2)
    log(f"wrote {out_p}")
    log("=== done ===")


if __name__ == "__main__":
    main()
