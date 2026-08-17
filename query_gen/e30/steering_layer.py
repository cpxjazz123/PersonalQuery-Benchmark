"""E30 — middle-layer LLM steering on TinyStyler/T5.

Hooks the T5 encoder middle layer (block 12 of 24) and adds a scaled style
direction vector to its hidden states. Then generates styled output per user.

This goes beyond TinyStyler's prefix-only style injection by adding the style
vector directly to intermediate representations.
"""
import json
import os
import sys
import time
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'
sys.path.insert(0, '/fs04/ar57/wenyu/PersoanlQuery/TinyStyler/tinystyler')

import numpy as np
import torch
from transformers import AutoTokenizer
from tinystyler import TinyStyler

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

MAX_NEW_TOKENS = 64
NUM_BEAMS = 2
STEERING_LAYER = 12  # 0-indexed; T5-large has 24 blocks (encoder + decoder each 24)
                     # We steer the encoder at block 12 (middle)
ALPHAS = [0.0, 5.0, 20.0, 50.0]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SteeringHook:
    """Add a per-sample direction vector to encoder hidden states at layer k."""

    def __init__(self, model, layer_idx, direction_t):
        # direction_t: [B, D] — added to encoder hidden states at layer_idx
        self.layer_idx = layer_idx
        self.direction_t = direction_t  # tensor on cuda:0 (fp16)
        self.handle = None
        # Get encoder block at layer_idx
        # T5 has encoder.block[i] and decoder.block[i]
        self.encoder_block = model.encoder.block[layer_idx]

    def __call__(self, module, args, kwargs):
        # The encoder block forward returns a tuple (hidden_states, ...).
        # We hook by modifying hidden_states BEFORE block forward.
        # PyTorch forward_pre_hook is the cleanest way.
        pass

    def attach(self):
        # forward_pre_hook on the encoder block: receive hidden_states input
        # T5 forward signature: forward(hidden_states, attention_mask=None, ...)
        def pre_hook(module, args, kwargs):
            # args[0] is hidden_states [B, S, D]
            h = args[0]
            B = h.shape[0]
            # broadcast direction to sequence length and add
            d = self.direction_t[:B].unsqueeze(1)  # [B, 1, D]
            h_new = h + d
            # rebuild args
            return (h_new,) + args[1:], kwargs

        self.handle = self.encoder_block.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def detach(self):
        if self.handle:
            self.handle.remove()


def main():
    log("loading TinyStyler (fp16)...")
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
    log(f"loaded in {time.time()-t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(T5_SNAP, legacy=True)

    npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
    uids_arr = npz['user_ids']
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids_arr)}

    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]
    source_query = pairs[0]['source_query']
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])
    styles = np.stack([uid_to_emb[u] for u in users])

    log(f"running middle-layer steering sweep: 10 users, layer={STEERING_LAYER}, alphas={ALPHAS}")
    enc = tokenizer([source_query] * len(users), return_tensors='pt', padding=True,
                     truncation=True, max_length=128).to('cuda:0')

    # The T5 encoder is wrapped by self.model.encoder; TinyStyler.model.encoder.block[k]
    # When we steer the encoder at layer k, the prefix style embedding is NOT included
    # at that point (it goes through the full encoder). To make direction vectors add-able
    # to encoder hidden states [B, S, D], we need D = d_model = 1024. Our user embs are 768.
    # Solution: project user embs through TinyStyler.proj to get them into 1024-dim,
    # then broadcast-add at the encoder layer.
    d_model = model.model.config.d_model  # 1024
    log(f"T5 d_model = {d_model}")

    # We need access to model.proj (the style projection 768→1024). It's defined in TinyStyler.
    proj = model.proj  # nn.Linear(768, 1024)

    for alpha in ALPHAS:
        # project user styles to 1024-dim
        style_t = torch.from_numpy(styles).to('cuda:0').half()  # [B, 768]
        with torch.no_grad():
            direction_1024 = proj(style_t)  # [B, 1024]
        if alpha == 0:
            direction_1024 = torch.zeros_like(direction_1024)
        else:
            direction_1024 = direction_1024 * alpha

        # attach hook
        hook = SteeringHook(model.model, STEERING_LAYER, direction_1024)
        hook.attach()
        try:
            # We also still feed the original user style as TinyStyler prefix (alpha=1)
            style_prefix = style_t  # unchanged
            t1 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    style=style_prefix,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            elapsed = time.time() - t1
            c = Counter(decoded)
            n_unique = len(c)
            log(f"\n=== α_steering={alpha} ({elapsed:.0f}s, {n_unique}/10 unique) ===")
            for txt, n in c.most_common():
                user_tags = [u[:6] for u, d in zip(users, decoded) if d == txt]
                log(f"  [{n}x] {txt!r}")
                log(f"      users: {','.join(user_tags)}")
        finally:
            hook.detach()


if __name__ == '__main__':
    main()