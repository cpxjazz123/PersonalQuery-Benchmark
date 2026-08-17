"""E30 — test whether output length interacts with style steering strength.

Hypothesis: at fixed K=8 + α=2.0, style influence per token should be roughly
constant.  But total style discrimination should grow with length (more tokens
= more chances for user-style vocabulary/syntax to surface).

We sweep MAX_NEW_TOKENS ∈ {16, 24, 32, 48, 64, 96, 128} at K=8/α=2 and measure:
  1. n_unique (lexical diversity)
  2. cos(styled_emb, user_emb) — style fidelity vs user reference
  3. cos(styled_emb, source_emb) — content preservation
  4. n_attrs_in_styled (preservation rate)
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
from sentence_transformers import SentenceTransformer

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'
ANNAWEGMANN = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots'

K_PREFIX = 8
ALPHA = 2.0
NUM_BEAMS = 2
LENGTHS = [16, 24, 32, 48, 64, 96, 128]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_k_prefix(model, input_ids, attention_mask, style, k_prefix, alpha,
                  max_new_tokens, num_beams):
    input_embeds = model.model.shared(input_ids)
    s = model.proj(style * alpha)
    style_prefix = s.unsqueeze(1).expand(
        input_ids.shape[0], k_prefix, s.shape[-1]
    ).contiguous()
    input_embeds = torch.cat([style_prefix, input_embeds], dim=1)
    B = input_ids.shape[0]
    prefix_mask = torch.ones((B, k_prefix),
                              device=attention_mask.device,
                              dtype=attention_mask.dtype)
    am = torch.cat([prefix_mask, attention_mask], dim=1)
    return model.model.generate(
        inputs_embeds=input_embeds, attention_mask=am,
        max_new_tokens=max_new_tokens, num_beams=num_beams,
        do_sample=False, early_stopping=True,
    )


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
    attrs = pairs[0]['attrs']

    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])
    styles = np.stack([uid_to_emb[u] for u in users])

    enc = tokenizer([source_query] * len(users), return_tensors='pt', padding=True,
                     truncation=True, max_length=128).to('cuda:0')

    # Load AnnaWegmann for style fidelity measurement
    log("loading AnnaWegmann for style fidelity measurement...")
    import glob
    aw_snap = sorted(glob.glob(f'{ANNAWEGMANN}/*/'))[-1]
    log(f"  using {aw_snap}")
    aw_model = SentenceTransformer(aw_snap).to('cuda:0').eval()
    aw_dim = aw_model.get_sentence_embedding_dimension()
    log(f"  AnnaWegmann dim={aw_dim}")

    user_embs_t = torch.from_numpy(styles).to('cuda:0').float()  # 768
    if user_embs_t.shape[1] != aw_dim:
        log(f"  WARNING: user_embs dim={user_embs_t.shape[1]} != aw_dim={aw_dim}; "
            f"need compatible style embedding model")

    log(f"\nlength sweep at K={K_PREFIX}, α={ALPHA}")
    log(f"{'len':>4} | {'uniq':>4} | {'cos_user':>9} | {'cos_src':>8} | {'mean_tok':>8} | {'n_attrs':>7}")
    log('-' * 65)

    for max_new in LENGTHS:
        style_t = torch.from_numpy(styles).to('cuda:0').half()
        t1 = time.time()
        with torch.no_grad():
            out_ids = gen_k_prefix(
                model, enc['input_ids'], enc['attention_mask'],
                style=style_t, k_prefix=K_PREFIX, alpha=ALPHA,
                max_new_tokens=max_new, num_beams=NUM_BEAMS,
            )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        elapsed = time.time() - t1

        # 1. n_unique
        n_unique = len(set(decoded))

        # 2-3. encode all styled + source with AnnaWegmann
        with torch.no_grad():
            styled_embs = aw_model.encode(decoded, convert_to_tensor=True,
                                            normalize_embeddings=True,
                                            show_progress_bar=False)
            src_emb = aw_model.encode([source_query], convert_to_tensor=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False)[0]
            user_refs = aw_model.encode(
                [f"placeholder reference for {u}" for u in users],
                convert_to_tensor=True, normalize_embeddings=True,
                show_progress_bar=False,
            )

        # cos to user reference per row — but user_refs is just placeholder, so
        # compute cos to user's mean style embedding from e30_style_embs.npz
        user_refs_norm = user_embs_t / (user_embs_t.norm(dim=1, keepdim=True) + 1e-12)
        styled_embs_norm = styled_embs.float() / (styled_embs.float().norm(dim=1, keepdim=True) + 1e-12)
        cos_user_per = (user_refs_norm * styled_embs_norm).sum(dim=1).cpu().numpy()
        cos_user_mean = float(cos_user_per.mean())

        # cos to source per row
        cos_src_per = (styled_embs @ src_emb).cpu().numpy()
        cos_src = float(cos_src_per.mean())

        # mean tokens
        mean_tok = float(np.mean([len(t.split()) for t in decoded]))

        # n_attrs
        n_attrs = sum(1 for v in attrs.values() if v)
        n_attrs_in_styled = []
        for d in decoded:
            # match as substring (lowercased, attr value normalized)
            d_lc = d.lower()
            n = 0
            for v in attrs.values():
                if v and v.lower() in d_lc:
                    n += 1
            n_attrs_in_styled.append(n)
        mean_attrs = float(np.mean(n_attrs_in_styled))

        log(f"{max_new:>4} | {n_unique:>4}/10 | {cos_user_mean:>+9.4f} | "
            f"{cos_src:>+8.4f} | {mean_tok:>8.1f} | {mean_attrs:>5.2f}/{n_attrs} "
            f"({elapsed:.0f}s)")

        # save sample for inspection at key lengths
        if max_new == 32 or max_new == 64 or max_new == 128:
            log(f"\n  --- sample (max_new={max_new}) ---")
            for u, d in list(zip(users, decoded))[:5]:
                log(f"    [{u[:6]}] {d!r}")
            log('')


if __name__ == '__main__':
    main()