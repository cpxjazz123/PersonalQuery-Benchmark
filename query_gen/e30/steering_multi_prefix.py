"""E30 — multi-token style prefix injection on TinyStyler/T5.

Hypothesis: T5's encoder layer-norm washes out middle-layer activation steering
(_steering_layer.py showed 4/10 unique across α∈{0,5,20,50}). The natural
alternative is to *prepend MULTIPLE style prefix tokens* (K-prefix) so the
signal survives all layers (analogous to long prefix-tuning vectors).

We use a single shared proj matrix repeated K times — this is equivalent to
giving the model a "thicker" style token prefix without the T5 having to
retrain on K>1 prefix lengths.

Sweep:
  K (style prefix length) ∈ {1, 2, 4, 8, 16}
  α ∈ {1.0, 2.0, 5.0}
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_k_prefix_generate(model, k_prefix, alpha):
    """Return a function input_ids/attention_mask/style -> out_ids that
    prepends K copies of the (alpha-scaled) proj(style) as prefix tokens.
    """
    base_generate = model.model.generate  # underlying T5ForConditionalGeneration.generate
    # T5 input embeddings live in model.model.shared (a shared nn.Embedding)
    proj = model.proj  # nn.Linear 768 -> 1024 (fp16 after .half())

    def gen(input_ids, attention_mask, style, **kwargs):
        B = input_ids.shape[0]
        # Use the shared embedding directly (T5 uses model.model.shared)
        input_embeds = model.model.shared(input_ids)  # [B, S, D]
        # shared proj, K copies
        s = proj(style * alpha)  # [B, D]
        style_prefix = s.unsqueeze(1).expand(B, k_prefix, s.shape[-1]).contiguous()
        input_embeds = torch.cat([style_prefix, input_embeds], dim=1)
        prefix_mask = torch.ones((B, k_prefix),
                                  device=attention_mask.device,
                                  dtype=attention_mask.dtype)
        am = torch.cat([prefix_mask, attention_mask], dim=1)
        return base_generate(
            inputs_embeds=input_embeds, attention_mask=am, **kwargs
        )
    return gen


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

    enc = tokenizer([source_query] * len(users), return_tensors='pt', padding=True,
                     truncation=True, max_length=128).to('cuda:0')

    log(f"sweep: K_prefix ∈ {{1, 2, 4, 8, 16}} × α ∈ {{1.0, 2.0, 5.0}}")

    for k_prefix in [1, 2, 4, 8, 16]:
        for alpha in [1.0, 2.0, 5.0]:
            gen = make_k_prefix_generate(model, k_prefix, alpha)
            style_t = torch.from_numpy(styles).to('cuda:0').half()
            t1 = time.time()
            with torch.no_grad():
                out_ids = gen(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    style=style_t,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            elapsed = time.time() - t1
            c = Counter(decoded)
            n_unique = len(c)
            log(f"\n=== K_prefix={k_prefix}, α={alpha} ({elapsed:.0f}s, {n_unique}/10 unique) ===")
            for txt, n in c.most_common(3):
                user_tags = [u[:6] for u, d in zip(users, decoded) if d == txt]
                log(f"  [{n}x] {txt[:120]!r}...")
                log(f"      users: {','.join(user_tags)}")


if __name__ == '__main__':
    main()