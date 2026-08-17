"""E30 Step 4 — generate styled queries via TinyStyler.

For each (product, user) pair, generate:
  - styled_query   = TinyStyler(source_query, user_style_emb)
  - baseline_query = TinyStyler(source_query, zero_style_emb)

Source queries come from Step 3 (e30_source_queries.jsonl).  User style
embeddings come from Step 2 (e30_style_embs.npz).  We use beam search
(width=4) for fluency.

NOTE on LLM client rule (AGENTS.md Rule 8/9):
  TinyStyler is a T5 model, not a Qwen client.  The codebase has its own
  self-contained package at TinyStyler/tinystyler/ that we integrate against
  directly, mirroring the pattern in tinystyler_authorship.py.  This is the
  documented integration path; we do NOT wrap TinyStyler into llm_client.py.

Writes /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_styled_queries.jsonl
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "TinyStyler" / "tinystyler"))

from tinystyler import TinyStyler  # noqa: E402

T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
T5_LOCAL = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large'  # HF cache root
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

PICKED_JSON = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products.json')
SOURCE_JSONL = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_source_queries.jsonl')
EMB_NPZ = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_style_embs.npz')
OUT_JSONL = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_styled_queries.jsonl')
OUT_LOG = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_gen_styled.log')

MAX_NEW_TOKENS = 64
BATCH_SIZE = 4  # 10 users × 2 (styled+baseline) = 20 sequences; chunk to 4 to fit memory
NUM_BEAMS = 2  # 2 beams to keep activations manageable inside 34GB cgroup
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1  # 1-product smoke run
SEED = 42
# Style steering (E30.6b tuning): K=8 prefix tokens × α=2.0 scaling → 9/10 unique
# (vs baseline K=1/α=1 → 4/10 unique).  See _steering_multi_prefix.py.
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0
# TinyStyler was trained on paraphrase-style inputs — no special prefix is
# used in tinystyler_authorship.py (passes raw text straight through).  We
# follow the same convention.
TINYSTYLER_PREFIX = ''


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


def _generate_k_prefix(model, input_ids, attention_mask, style,
                        k_prefix, alpha, **kwargs):
    """Prepend K copies of the (alpha-scaled) proj(style) embedding as prefix tokens.

    E30.6b tuning showed K=8 + α=2.0 gives 9/10 unique outputs vs K=1/α=1 baseline's 4/10.
    T5's layer-norm washes out middle-layer activation steering, so multi-token prefix
    is the most effective way to amplify the style signal.
    """
    input_embeds = model.model.shared(input_ids)  # [B, S, D]
    s = model.proj(style * alpha)  # [B, D]
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
        inputs_embeds=input_embeds, attention_mask=am, **kwargs
    )


def main():
    # Pin HF cache to /fs04 so the offline tokenizer/model loaders find T5
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

    log("loading TinyStyler (fp16)...")
    t0 = time.time()
    model = TinyStyler(base_model=T5_BASE, use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu')
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = model.state_dict()
    cur.update(saved)
    model.load_state_dict(cur)
    model.to('cuda:0').half().eval()  # fp16 to fit in 34GB cgroup limit
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded TinyStyler in {time.time()-t0:.0f}s")

    log("loading T5 tokenizer from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(T5_SNAP, legacy=True)
    log(f"tokenizer ok; vocab={tokenizer.vocab_size}")

    log("loading inputs...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products']
    products = products[:N_PRODUCTS]
    log(f"  {len(products)} products (N_PRODUCTS={N_PRODUCTS})")

    src_map = {}
    with open(SOURCE_JSONL) as f:
        for line in f:
            r = json.loads(line)
            src_map[r['asin']] = r
    log(f"  {len(src_map)} source queries")

    npz = np.load(EMB_NPZ, allow_pickle=True)
    uids = list(npz['user_ids'])
    embs = npz['embs']  # (n_users, 768)
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids)}
    log(f"  {len(uids):,} user style embeddings")

    rng = np.random.default_rng(SEED)

    log(f"generating styled + baseline queries (n_users/product={N_USERS_PER_PRODUCT}, beam={NUM_BEAMS}, batch={BATCH_SIZE})...")
    t0 = time.time()
    n_done = 0
    n_total = len(products) * N_USERS_PER_PRODUCT
    with open(OUT_JSONL, 'w') as fout:
        for pi, p in enumerate(products):
            asin = p['asin']
            src = src_map.get(asin)
            if src is None:
                continue
            source_query = src['source_query']
            attrs = p['attrs']
            users = list(p['candidate_users'])
            if len(users) > N_USERS_PER_PRODUCT:
                users = list(rng.choice(users, size=N_USERS_PER_PRODUCT, replace=False))
            else:
                users = users[:N_USERS_PER_PRODUCT]

            # Build batch inputs (styled + baseline for each chosen user)
            styled_inputs = []
            user_ids = []
            for u in users:
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                styled_inputs.append((u, source_query, emb))
                user_ids.append(u)

            if not styled_inputs:
                continue

            # Tokenize all styled inputs
            styled_texts = [TINYSTYLER_PREFIX + src_q for _, src_q, _ in styled_inputs]
            styled_styles = np.stack([e for _, _, e in styled_inputs])  # (B, 768)

            # Tokenize all baseline inputs (same source_query, zero style)
            baseline_texts = styled_texts[:]  # identical
            baseline_styles = np.zeros_like(styled_styles)

            # Build (style_label, text, style_vec) tuples; treat styled/baseline
            # uniformly in chunks.  We label each input with its destination
            # category ('styled' or 'baseline') and the user-index it belongs to.
            entries = []
            for i, (u, src_q, emb) in enumerate(styled_inputs):
                entries.append(('styled', i, TINYSTYLER_PREFIX + src_q, emb))
            for i, (u, src_q, emb) in enumerate(styled_inputs):
                entries.append(('baseline', i, TINYSTYLER_PREFIX + src_q,
                                np.zeros_like(emb)))

            styled_dec = [None] * len(styled_inputs)
            baseline_dec = [None] * len(styled_inputs)
            for s in range(0, len(entries), BATCH_SIZE):
                chunk = entries[s:s + BATCH_SIZE]
                chunk_texts = [c[2] for c in chunk]
                chunk_styles = np.stack([c[3] for c in chunk])
                enc = tokenizer(
                    chunk_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to('cuda:0')
                style_t = torch.from_numpy(chunk_styles).to('cuda:0').half()
                with torch.no_grad():
                    # E30.6b: use K-prefix repeated style embedding with α-scaling
                    out_ids = _generate_k_prefix(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        early_stopping=True,
                    )
                chunk_dec = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                for (cat, idx, _, _), txt in zip(chunk, chunk_dec):
                    if cat == 'styled':
                        styled_dec[idx] = txt
                    else:
                        baseline_dec[idx] = txt

            # Verify all decoded
            for i in range(len(styled_inputs)):
                if styled_dec[i] is None or baseline_dec[i] is None:
                    raise RuntimeError(f"missing decode at index {i}")

            for (u, src_q, _), styled_q, baseline_q in zip(
                styled_inputs, styled_dec, baseline_dec
            ):
                n_attr_src = sum(1 for v in attrs.values() if v and v in src_q)
                n_attr_styled = sum(1 for v in attrs.values() if v and v in styled_q)
                n_attr_baseline = sum(1 for v in attrs.values() if v and v in baseline_q)
                rec = {
                    'asin': asin,
                    'user_id': str(u),
                    'attrs': attrs,
                    'source_query': src_q,
                    'styled_query': styled_q,
                    'baseline_query': baseline_q,
                    'n_attrs': len([v for v in attrs.values() if v]),
                    'n_attrs_in_source': n_attr_src,
                    'n_attrs_in_styled': n_attr_styled,
                    'n_attrs_in_baseline': n_attr_baseline,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1

            if (pi + 1) % 10 == 0 or pi == 0:
                log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} pairs to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
