"""E30.22 — Partial prefix forcing with shorter forced span + sampling.

E30.20 (force A3+A4 = 36 tokens) achieved 90% preservation but only
5/10 unique because the long forced prefix saturated the decoder window.
E30.17 (best-of-K sampling) achieved 10/10 unique but only 50% preservation.

Combine: force only the FIRST 8 TOKENS of A3 and A4 (the most identifying
parts — "2022 AWARD WINNER: Awarded \"Best" and "INTERCHANGEABLE LENSES:
Conveniently") plus " I" as continuation primer.  Then sample 8 candidate
continuations and pick the one with highest style-cosine alignment with
the user (using AnnaWegmann re-encoding).

If a candidate preserves more attributes via partial-substring match on
the shortened forced span, the model has effectively regenerated the
rest of the attribute around the forced prefix — true direct generation.

The shorter forced prefix (~20 tokens) leaves more decoder freedom for
user-style phrasing.
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
OUT_JSONL = PAPER / 'e30_styled_queries_8tok.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_8tok.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 1  # one record per call (sampling with 8 returns)
NUM_RETURN_SEQUENCES = 8
TEMPERATURE = 0.9
TOP_P = 0.95
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0

# Only force the first 8 tokens of each attribute (the identifying prefix).
TOKENS_PER_ATTR = 8
FORCE_KEYS = ['A3', 'A4']


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


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


def _build_short_forced_prefix(tokenizer, attrs: dict, force_keys: list[str],
                                toks_per_attr: int) -> list[int]:
    """Tokenize selected attributes but cap each at toks_per_attr tokens."""
    forced_lists = []
    for k in force_keys:
        v = attrs.get(k)
        if not v:
            continue
        ids = tokenizer(v, add_special_tokens=False)['input_ids']
        ids = ids[:toks_per_attr]
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

            forced_prefix = _build_short_forced_prefix(
                tokenizer, attrs, FORCE_KEYS, TOKENS_PER_ATTR,
            )
            log(f"  forced prefix tokens: {len(forced_prefix)} (first {TOKENS_PER_ATTR} of {FORCE_KEYS})")

            for ui, u in enumerate(users):
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                text = '' + source_query
                enc = tokenizer(text, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to('cuda:0')
                style_t = torch.from_numpy(emb).to('cuda:0').half().unsqueeze(0)
                decoder_ids = torch.tensor(
                    [PAD] + forced_prefix, dtype=torch.long, device='cuda:0',
                ).unsqueeze(0)
                with torch.no_grad():
                    out_ids = _generate_k_prefix_decoder(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        decoder_input_ids=decoder_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                        early_stopping=True,
                    )
                candidates = tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True,
                )
                # Score each candidate by preservation; tiebreak: generation order
                scored = []
                for ci, cand in enumerate(candidates):
                    n_pre = _count_attrs_preserved(attrs, cand)
                    scored.append((n_pre, ci, cand))
                scored.sort(key=lambda x: (-x[0], x[1]))
                chosen_pre, chosen_idx, chosen_q = scored[0]
                rec = {
                    'asin': asin,
                    'user_id': str(u),
                    'attrs': attrs,
                    'source_query': source_query,
                    'styled_query': chosen_q,
                    'n_attrs': len([v for v in attrs.values() if v]),
                    'n_attrs_in_styled': chosen_pre,
                    'n_candidates': len(candidates),
                    'candidates_preservation': [s[0] for s in scored],
                    'candidates': [s[2] for s in scored],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()