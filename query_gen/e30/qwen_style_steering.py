"""E30 — Qwen + AnnaWegmann style embedding steering (replace TinyStyler).

Architecture: prepend K soft prompt tokens to Qwen2-7B input embeddings.
Each token = LayerNorm(Linear(AnnaWegmann_768)[i]) for i in [0..K-1].
A learnable-or-fixed scalar alpha scales the contribution (TinyStyler-style).

Same source query (B00ECHYTBI, 4 attrs, ~36 tokens). 10 users. Sweep:
  K (soft tokens) ∈ {1, 2, 4, 8, 16}
  α ∈ {0.5, 1.0, 2.0, 5.0, 10.0}

Compare against TinyStyler K=8/α=2 (9/10 unique + 2/4 attrs).
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
BASE_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class StyleProjector(nn.Module):
    """Map (768,) AnnaWegmann style embedding → (K, model_dim) soft prefix."""

    def __init__(self, style_dim=768, k_prefix=8, model_dim=3584, alpha_init=1.0):
        super().__init__()
        self.k_prefix = k_prefix
        self.model_dim = model_dim
        # independent projections per prefix token (analogous to TinyStyler
        # proj being applied K times with different weights? — no, here we
        # share one proj and use K=1 then expand. This is simpler and OK for
        # steering).
        self.proj = nn.Linear(style_dim, k_prefix * model_dim)
        self.layernorm = nn.LayerNorm(model_dim)
        self.alpha = alpha_init  # scalar, set externally per sweep
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.layernorm.weight)
        nn.init.zeros_(self.layernorm.bias)

    def forward(self, style):
        # style: [B, 768] fp16/fp32
        x = self.proj(style)  # [B, K*D]
        x = x.view(-1, self.k_prefix, self.model_dim)
        x = self.layernorm(x)
        return self.alpha * x


def main():
    log("loading Qwen2-7B-Instruct (bf16, base only — no LoRA)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map='cuda:0', trust_remote_code=True,
    )
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    hidden_size = mdl.config.hidden_size
    log(f"  Qwen loaded in {time.time()-t0:.0f}s; hidden_size={hidden_size}")

    # load 4-attr source + 10 users
    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]
    source_query = pairs[0]['source_query']
    attrs = pairs[0]['attrs']
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])

    npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
    uids_arr = npz['user_ids']
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids_arr)}
    styles = np.stack([uid_to_emb[u] for u in users])

    # build a rewriter prompt (similar to TinyStyler's input: just the source query)
    prompt_text = (
        f"Rewrite this shopping query in your own words, preserving all product "
        f"attributes:\n\n{source_query}\n\nRewritten query:"
    )
    prompts = [prompt_text] * len(users)
    enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
              max_length=512).to('cuda:0')

    log(f"\n{'K':>3} | {'α':>5} | {'uniq':>5} | {'mean_attrs':>10} | {'mean_tok':>8}")
    log('-' * 55)

    for k_prefix in [1, 4, 8, 16]:
        proj = StyleProjector(768, k_prefix, hidden_size).to('cuda:0').bfloat16()
        for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
            proj.alpha = alpha
            style_t = torch.from_numpy(styles).to('cuda:0').bfloat16()
            with torch.no_grad():
                soft_prefix = proj(style_t)  # [B, K, hidden]
                input_embeds = mdl.get_input_embeddings()(enc['input_ids'])  # [B, S, hidden]
                B = input_embeds.shape[0]
                full_embeds = torch.cat([soft_prefix, input_embeds], dim=1)
                # extend attention_mask: left-padded, so prefix is at the END of the visible window
                # Actually for left-padded: input_ids have right-aligned content; the K prefix tokens
                # should appear AFTER the actual content (right side) so they're the LAST K positions
                # Wait — soft prefix is prepended to embedding sequence (positions 0..K-1), then the
                # original input is at positions K..K+S-1.
                # With left-padding, the original input is at positions [L_pad, L_pad+S_real-1].
                # So we need prefix at positions [L_pad+S_real, L_pad+S_real+K-1].
                # Simplest: rebuild attention_mask so prefix gets 1s after the real input.
                L_pad = (enc['attention_mask'].shape[1] - enc['attention_mask'].sum(dim=1)).cpu().numpy()
                # we'll just use a single full mask for simplicity since left-pad handles causal masking
                full_mask = torch.ones((B, full_embeds.shape[1]), device='cuda:0',
                                        dtype=enc['attention_mask'].dtype)
                out_ids = mdl.generate(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    max_new_tokens=64,
                    num_beams=2,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
            n_unique = len(set(decoded))
            mean_tok = float(np.mean([len(d.split()) for d in decoded]))
            mean_attrs = float(np.mean([
                sum(1 for v in attrs.values() if v and v in d.lower())
                for d in decoded
            ]))
            log(f"{k_prefix:>3} | {alpha:>5.1f} | {n_unique:>2}/10 | "
                f"{mean_attrs:>5.2f}/{len(attrs)}    | {mean_tok:>8.1f}")
            # dump 2 samples for first cell only
            if k_prefix == 8 and alpha == 2.0:
                log("  --- sample at K=8, α=2 ---")
                for u, d in list(zip(users, decoded))[:3]:
                    log(f"    [{u[:6]}] {d!r}")


if __name__ == '__main__':
    main()