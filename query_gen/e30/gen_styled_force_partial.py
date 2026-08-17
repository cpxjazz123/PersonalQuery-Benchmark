"""E30.20 — Partial prefix-forcing: only the noisiest attributes.

E30.15 (force all 4 attrs) → 100% preserved but 1/10 unique.
E30.16 (force all 4 + I-primer) → 100% preserved, 4/10 unique.
E30.17 (best-of-K sampling) → 50% preserved, 10/10 unique.
E30.18 (attr-as-input) → 50% preserved (echo behaviour).
E30.19 (combo input + 8-beam) → 50% preserved.

The model preserves A1+A2 (brand + model name) naturally because they're
short and high-frequency tokens; it loses A3+A4 (long noisy marketing
copy / spec descriptions) consistently.

Strategy: only force A3 and A4 as decoder prefix (skip A1, A2).  This
gives the decoder freedom to generate A1+A2 in user-style phrasing while
the long noisy attrs are guaranteed present in the output.

This is true direct generation: the decoder emits all tokens; we only
constrain which ones it must emit at specific positions.

Inputs: same as e30/gen_styled.py (K=8/α=2 style steering unchanged).
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
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

PAPER = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper')
PICKED_JSON = PAPER / 'e30_picked_products.json'
SOURCE_JSONL = PAPER / 'e30_source_queries.jsonl'
EMB_NPZ = PAPER / 'e30_style_embs.npz'
OUT_JSONL = PAPER / 'e30_styled_queries_partial.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_partial.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 4
NUM_BEAMS = 4
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0

# Only force the attributes that the model fails to preserve naturally.
# A1 = brand (short), A2 = model (short) — model preserves these.
# A3 = marketing copy (long noisy), A4 = spec description (long noisy) —
# model drops these.  Force only A3+A4.
FORCE_KEYS = ['A3', 'A4']


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


def _attr_token_ids(tokenizer, value: str) -> list[int]:
    if not value:
        return []
    return tokenizer(value, add_special_tokens=False)['input_ids']


def _is_attr_in_query(attr_value: str, query: str) -> bool:
    if not attr_value:
        return True
    ql = query.lower()
    if attr_value.lower() in ql:
        return True
    head = attr_value[:30].strip().rstrip('":;,')
    if head and head.lower() in ql:
        return True
    return False


def _count_attrs_preserved(attrs: dict, query: str) -> int:
    return sum(1 for v in attrs.values() if _is_attr_in_query(v, query))


def _build_forced_prefix(tokenizer, attrs: dict, force_keys: list[str]) -> list[int]:
    """Tokenize selected attributes, join with ", ".  Strip trailing period."""
    forced_lists = []
    for k in force_keys:
        v = attrs.get(k)
        if v:
            ids = _attr_token_ids(tokenizer, v)
            if ids:
                forced_lists.append(ids)
    if not forced_lists:
        return []
    comma_id = tokenizer.convert_tokens_to_ids(',')
    space_id = tokenizer.convert_tokens_to_ids(' ')
    period_id = tokenizer.convert_tokens_to_ids('.')
    forced = []
    for i, lst in enumerate(forced_lists):
        if i > 0:
            forced.append(comma_id)
            forced.append(space_id)
        forced.extend(lst)
    while forced and forced[-1] == period_id:
        forced.pop()
    # Add " I" as continuation primer so the decoder must continue
    i_id = tokenizer('I', add_special_tokens=False)['input_ids']
    forced.append(space_id)
    forced.extend(i_id)
    return forced


def _generate_k_prefix_decoder(model, input_ids, attention_mask, style,
                                k_prefix, alpha, decoder_input_ids,
                                **kwargs):
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
        inputs_embeds=input_embeds,
        attention_mask=am,
        decoder_input_ids=decoder_input_ids,
        **kwargs,
    )


def main():
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
    model.to('cuda:0').half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded TinyStyler in {time.time()-t0:.0f}s")

    log("loading T5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(T5_SNAP, legacy=True)
    PAD = tokenizer.pad_token_id

    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products'][:N_PRODUCTS]

    src_map = {}
    with open(SOURCE_JSONL) as f:
        for line in f:
            r = json.loads(line)
            src_map[r['asin']] = r

    npz = np.load(EMB_NPZ, allow_pickle=True)
    uids = list(npz['user_ids'])
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids)}

    rng = np.random.default_rng(SEED)

    n_done = 0
    n_total = len(products) * N_USERS_PER_PRODUCT
    t0 = time.time()
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

            forced_prefix = _build_forced_prefix(tokenizer, attrs, FORCE_KEYS)
            log(f"  forced prefix tokens: {len(forced_prefix)} (keys {FORCE_KEYS})")

            styled_inputs = []
            for u in users:
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                styled_inputs.append((u, source_query, emb))
            if not styled_inputs:
                continue

            styled_texts = ['' + source_query for _, src_q, _ in styled_inputs]
            styled_styles = np.stack([e for _, _, e in styled_inputs])
            decoder_ids = torch.tensor(
                [PAD] + forced_prefix, dtype=torch.long, device='cuda:0'
            ).unsqueeze(0).expand(len(styled_inputs), -1).contiguous()

            styled_dec = [None] * len(styled_inputs)

            for s in range(0, len(styled_inputs), BATCH_SIZE):
                chunk_texts = styled_texts[s:s + BATCH_SIZE]
                chunk_styles = styled_styles[s:s + BATCH_SIZE]
                chunk_decoder_ids = decoder_ids[s:s + BATCH_SIZE]
                enc = tokenizer(chunk_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to('cuda:0')
                style_t = torch.from_numpy(chunk_styles).to('cuda:0').half()
                with torch.no_grad():
                    out_ids = _generate_k_prefix_decoder(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        decoder_input_ids=chunk_decoder_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        early_stopping=True,
                    )
                chunk_dec = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                for i, txt in enumerate(chunk_dec):
                    styled_dec[s + i] = txt

            for i in range(len(styled_inputs)):
                if styled_dec[i] is None:
                    raise RuntimeError(f"missing decode at index {i}")

            for (u, src_q, _), styled_q in zip(styled_inputs, styled_dec):
                n_attr_src = sum(1 for v in attrs.values() if v and v.lower() in src_q.lower())
                n_attr_styled = _count_attrs_preserved(attrs, styled_q)
                rec = {
                    'asin': asin,
                    'user_id': str(u),
                    'attrs': attrs,
                    'source_query': src_q,
                    'styled_query': styled_q,
                    'n_attrs': len([v for v in attrs.values() if v]),
                    'n_attrs_in_source': n_attr_src,
                    'n_attrs_in_styled': n_attr_styled,
                    'forced_keys': FORCE_KEYS,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()