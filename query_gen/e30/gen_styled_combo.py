"""E30.19 — Combined: structured input with attribute markers + multi-beam.

E30.18 confirmed: feeding T5 a bulleted attribute list makes it echo the
input (no style transfer).  We need source_query as input (for style
signal) AND attrs visible in the input (for cross-attention preservation).

Approach: structured input that interleaves source_query phrasing with
attribute markers, then beam-search with high beam count for diversity.

Input template (per record):
    <source_query>
    Specs:
    A1: <val1>  A2: <val2>  A3: <val3>  A4: <val4>

The model sees a natural-language query (so it can produce a natural
query), plus attribute markers (so cross-attention can copy).  We use
num_beams=8 to explore more of the output space than the baseline 4.
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
OUT_JSONL = PAPER / 'e30_styled_queries_combo.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_combo.log'

MAX_NEW_TOKENS = 96
BATCH_SIZE = 2  # num_beams=8 is heavy
NUM_BEAMS = 8
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0


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


def _build_combo_input(source_query: str, attrs: dict) -> str:
    """Source query first (for style/structure), then attribute markers
    (for cross-attention copy).  T5 was trained on natural-language inputs
    so source_query must lead; the attribute markers act as a content
    anchor the decoder can copy from."""
    lines = [source_query, 'Specs:']
    for k in sorted(attrs, key=lambda x: int(x[1:])):
        v = attrs[k]
        if v:
            lines.append(f"{k}: {v}")
    return '\n'.join(lines)


def _generate_k_prefix(model, input_ids, attention_mask, style,
                        k_prefix, alpha, **kwargs):
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
        inputs_embeds=input_embeds, attention_mask=am, **kwargs,
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

            combo_input = _build_combo_input(source_query, attrs)
            log(f"  combo input ({len(combo_input)} chars, first 100): {combo_input[:100]!r}")

            styled_inputs = []
            for u in users:
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                styled_inputs.append((u, source_query, emb))
            if not styled_inputs:
                continue

            styled_texts = [combo_input] * len(styled_inputs)
            styled_styles = np.stack([e for _, _, e in styled_inputs])
            baseline_styles = np.zeros_like(styled_styles)

            styled_dec = [None] * len(styled_inputs)
            baseline_dec = [None] * len(styled_inputs)

            for s in range(0, len(styled_inputs), BATCH_SIZE):
                chunk_texts = styled_texts[s:s + BATCH_SIZE]
                chunk_styles = styled_styles[s:s + BATCH_SIZE]
                enc = tokenizer(chunk_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=256).to('cuda:0')
                style_t = torch.from_numpy(chunk_styles).to('cuda:0').half()
                with torch.no_grad():
                    out_ids = _generate_k_prefix(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        num_return_sequences=NUM_BEAMS,
                        do_sample=False,
                        early_stopping=True,
                    )
                # Decode all candidates then pick the one with most attrs preserved
                all_cands = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                # out_ids is (BATCH*NUM_BEAMS, T); reshape to (BATCH, NUM_BEAMS, T)
                nb = NUM_BEAMS
                bs = (out_ids.shape[0] // nb)
                # Take the best candidate per row
                chosen = []
                for i in range(bs):
                    row_cands = all_cands[i*nb:(i+1)*nb]
                    scored = sorted(
                        ((_count_attrs_preserved(attrs, c), idx, c)
                         for idx, c in enumerate(row_cands)),
                        key=lambda x: (-x[0], x[1]),
                    )
                    chosen.append(scored[0][2])
                for i, txt in enumerate(chosen):
                    styled_dec[s + i] = txt

                chunk_styles_b = baseline_styles[s:s + BATCH_SIZE]
                style_t_b = torch.from_numpy(chunk_styles_b).to('cuda:0').half()
                with torch.no_grad():
                    out_ids_b = _generate_k_prefix(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t_b,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        num_return_sequences=1,
                        do_sample=False,
                        early_stopping=True,
                    )
                chunk_dec_b = tokenizer.batch_decode(out_ids_b, skip_special_tokens=True)
                for i, txt in enumerate(chunk_dec_b):
                    baseline_dec[s + i] = txt

            for i in range(len(styled_inputs)):
                if styled_dec[i] is None or baseline_dec[i] is None:
                    raise RuntimeError(f"missing decode at index {i}")

            for (u, src_q, _), styled_q, baseline_q in zip(
                styled_inputs, styled_dec, baseline_dec
            ):
                n_attr_src = sum(1 for v in attrs.values() if v and v.lower() in src_q.lower())
                n_attr_styled = _count_attrs_preserved(attrs, styled_q)
                n_attr_baseline = _count_attrs_preserved(attrs, baseline_q)
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
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()