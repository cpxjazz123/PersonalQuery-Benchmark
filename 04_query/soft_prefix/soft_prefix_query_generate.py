#!/usr/bin/env python3
"""E11 — VADES soft-prefix conditioning: generation.

Loads a checkpoint produced by soft_prefix_query_train.py (projector.pt +
tokenizer/ + config.json + split/vector manifests) and generates one query per
(user_id, asin) record using the SAME preprocessing as training:
- identical prompt (five attributes) and Qwen chat template;
- identical UserVectorProvider (mode, seed, permutation) so shuffled/zero
  controls are exactly reproducible;
- identical soft-prefix injection (K tokens, position ids, attention mask).

Outputs: <out_dir>/<category>_mode=<mode>_K=<K>_generated.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
from content_validation import build_attr_prompt  # noqa: E402
from soft_prefix_query_train import build_messages  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class SoftPrefixGenerator(nn.Module):
    """Frozen Qwen + projector; single-sequence generation (no padding)."""

    def __init__(self, base_model, projector, device: str):
        super().__init__()
        self.base = base_model
        self.projector = projector
        self.device = device
        self.num_tokens = projector.num_tokens
        self.base.eval()
        self.projector.eval()

    def build_inputs(self, prompt_ids, user_vec: Optional[np.ndarray]):
        text_embeds = self.base.get_input_embeddings()(prompt_ids)
        target_dtype = next(self.base.parameters()).dtype
        text_embeds = text_embeds.to(target_dtype)
        if self.num_tokens > 0 and user_vec is not None:
            z = torch.as_tensor(user_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            prefix = self.projector(z.to(target_dtype))  # [1, K, d_model]
            full = torch.cat([prefix, text_embeds], dim=1)
            attention_mask = torch.ones(full.shape[:2], dtype=torch.long, device=self.device)
            prefix_pos = torch.arange(self.num_tokens, device=self.device).unsqueeze(0)
            text_pos = torch.arange(
                prompt_ids.size(1), device=self.device
            ).unsqueeze(0) + self.num_tokens
            position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        else:
            full = text_embeds
            attention_mask = torch.ones(full.shape[:2], dtype=torch.long, device=self.device)
            position_ids = torch.arange(
                prompt_ids.size(1), device=self.device
            ).unsqueeze(0)
        return full, attention_mask, position_ids

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        user_vec: Optional[np.ndarray],
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int,
        repetition_penalty: float = 1.0,
    ) -> str:
        full, attention_mask, position_ids = self.build_inputs(prompt_ids, user_vec)
        out = self.base.generate(
            inputs_embeds=full,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return out


def load_checkpoint(ckpt_dir: Path, device: str):
    config_p = ckpt_dir / "config.json"
    with open(config_p) as f:
        config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    lora_dir = ckpt_dir / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(lora_dir))
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    projector = SoftPrefixProjector(
        user_dim=config["user_dim"],
        hidden_dim=config["hidden_dim"],
        num_tokens=config["num_tokens"],
        model_dim=config["model_dim"],
        dtype=torch.bfloat16,
    ).to(device)
    state = torch.load(ckpt_dir / "projector.pt", map_location=device)
    projector.load_state_dict(state)
    model = SoftPrefixGenerator(base, projector, device=device).to(device)
    return model, tokenizer, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0=all records")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config = load_checkpoint(ckpt_dir, args.device)
    mode = config["mode"]
    log(f"loaded checkpoint: mode={mode} K={config['num_tokens']} seed={config['seed']}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    log(f"records={len(records)} (04_query 10-candidate source)")

    with open(ckpt_dir / "vector_manifest.json") as f:
        vector_manifest = json.load(f)
    profiles = load_vades_profiles(args.category)
    provider = UserVectorProvider(
        mode=vector_manifest["mode"],
        profiles=profiles,
        user_ids=[r["user_id"] for r in records],
        seed=vector_manifest["seed"],
    )
    log(f"provider: {provider.manifest()}")

    results = []
    for i, rec in enumerate(records):
        uid, asin = rec["user_id"], rec["asin"]
        attrs = rec.get("syntax_depth_query", {}).get("attrs_used")
        if not attrs:
            for cand in rec.get("syntax_depth_queries", []):
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if not attrs:
            results.append({**rec, "generated_query": None, "error": "no attrs"})
            continue
        prompt_str = tokenizer.apply_chat_template(
            build_messages(attrs), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(args.device)
        vec, has_vector = provider.get(uid)
        out = model.generate(
            prompt_ids,
            vec,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        results.append(
            {
                "user_id": uid,
                "asin": asin,
                "generated_query": text,
                "mode": mode,
                "num_tokens": config["num_tokens"],
                "has_vector": bool(has_vector),
            }
        )
        if (i + 1) % 10 == 0:
            log(f"  generated {i + 1}/{len(records)}")

    out_p = out_dir / f"{args.category}_mode={mode}_K={config['num_tokens']}_generated.json"
    with open(out_p, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    log(f"wrote {out_p}")
    log("=== done ===")


if __name__ == "__main__":
    main()
