"""E30 LLM steering experiment — scale TinyStyler style prefix and (optionally)
inject scaled style direction into T5 middle-layer hidden states.

Reads e30_styled_queries.jsonl (10 user records), regenerates styled outputs
with:
  α ∈ {1.0 (baseline), 2.0, 5.0, 10.0}
and reports per-α unique-output count + sample outputs.
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

ALPHAS = [1.0, 2.0, 5.0, 10.0, 20.0]


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)


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

    # read the 10 styled records (use their source queries)
    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]

    # unique source query (all 10 records share the same source)
    source_query = pairs[0]['source_query']
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])
    styles = np.stack([uid_to_emb[u] for u in users])

    log(f"running steering sweep: source_query length={len(source_query)} chars, {len(users)} users")
    enc = tokenizer([source_query] * len(users), return_tensors='pt', padding=True,
                     truncation=True, max_length=128).to('cuda:0')

    for alpha in ALPHAS:
        style_t = torch.from_numpy(styles).to('cuda:0').half() * alpha
        t1 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
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
        log(f"\n=== α={alpha} ({elapsed:.0f}s, {n_unique}/10 unique) ===")
        # group by output
        for txt, n in c.most_common():
            user_tags = [u[:6] for u, d in zip(users, decoded) if d == txt]
            log(f"  [{n}x] {txt!r}")
            log(f"      users: {','.join(user_tags)}")


if __name__ == '__main__':
    main()