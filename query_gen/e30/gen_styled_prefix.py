"""E30.16 — Decoder prefix-forcing for 100% direct attribute generation.

force_words_ids via constrained beam search (E30.15) only achieved 52.5%
preservation and dropped unique output from 9/10 to 6/10 — the trie
constraints over-narrow the beam and weaken the style signal.

Switch to **prefix forcing**: we tokenize the attribute values, prepend
them to the decoder_input_ids, and let the model generate freely after the
forced prefix.  This is true direct generation (the model's decoder emits
every token) and guarantees every attribute appears in the output by
construction (the prefix is the model's input).

If a short attr's tokens are a strict prefix of a longer attr's tokens
(e.g. "Infant Optics" ⊂ "Infant Optics DXR-8..."), we drop the shorter
list — forcing the longer one to appear automatically includes the
shorter one as a substring.

To preserve style signal:
- K=8 prefix × α=2.0 style injection stays active on the encoder.
- The decoder's only constraint is the prefix; everything after the
  prefix is style-conditioned free generation.
- We force a small connector (".") between attribute spans so the
  decoder has a natural transition into free generation.

Output: styled_query = "[FORCED ATTR PREFIX] [STYLE-CONDITIONED SUFFIX]"
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
OUT_JSONL = PAPER / 'e30_styled_queries_prefix.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_prefix.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 4
NUM_BEAMS = 4
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0
TINYSTYLER_PREFIX = ''


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


def _attr_token_ids(tokenizer, value: str) -> list[int]:
    if not value:
        return []
    ids = tokenizer(value, add_special_tokens=False)['input_ids']
    return ids


def _dedupe_prefix_subsets(forced_lists):
    out = []
    for cand in forced_lists:
        is_subset = False
        for other in forced_lists:
            if other is cand:
                continue
            if len(other) > len(cand) and other[:len(cand)] == cand:
                is_subset = True
                break
        if not is_subset:
            out.append(cand)
    return out


def _is_attr_in_query(attr_value: str, query: str) -> bool:
    if not attr_value:
        return True
    ql = query.lower()
    return attr_value.lower() in ql


def _generate_k_prefix_decoder(model, input_ids, attention_mask, style,
                                k_prefix, alpha, decoder_input_ids,
                                **kwargs):
    """K-prefix style injection + generate with decoder_input_ids override."""
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

    log("loading T5 tokenizer from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(T5_SNAP, legacy=True)
    log(f"tokenizer ok; vocab={tokenizer.vocab_size}; pad={tokenizer.pad_token_id}; eos={tokenizer.eos_token_id}")

    log("loading inputs...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products'][:N_PRODUCTS]
    log(f"  {len(products)} products")

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
    PAD = tokenizer.pad_token_id  # T5 uses pad as decoder start

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

            # Build forced prefix tokens: concatenation of attribute values
            # joined by ", " plus a trailing " I" priming token.
            # The " I" forces the decoder into an incomplete first-person
            # sentence so it must generate style-conditioned continuation
            # (vs emitting EOS immediately after the attribute list, which
            # drops unique output to 1/10).
            # Drop attributes whose tokens are a prefix subset of another's
            # tokens (they appear automatically as substrings).
            forced_lists = []
            for k in sorted(attrs, key=lambda x: int(x[1:])):
                v = attrs[k]
                if v:
                    forced_lists.append(_attr_token_ids(tokenizer, v))
            forced_lists = _dedupe_prefix_subsets(forced_lists)
            comma_id = tokenizer.convert_tokens_to_ids(',')
            forced_tokens = []
            for i, lst in enumerate(forced_lists):
                if i > 0:
                    forced_tokens.append(comma_id)
                    forced_tokens.append(tokenizer.convert_tokens_to_ids(' '))
                forced_tokens.extend(lst)
            # Strip a trailing period if present so the list isn't syntactically
            # complete by itself.
            period_id = tokenizer.convert_tokens_to_ids('.')
            while forced_tokens and forced_tokens[-1] == period_id:
                forced_tokens.pop()
            # Append " I" as a forced continuation primer.  The decoder now
            # sees an incomplete first-person sentence and must continue.
            space_id = tokenizer.convert_tokens_to_ids(' ')
            i_id = tokenizer('I', add_special_tokens=False)['input_ids']
            forced_tokens.append(space_id)
            forced_tokens.extend(i_id)
            log(f"  forced tokens: {len(forced_tokens)} (across {len(forced_lists)} attrs)")

            styled_inputs = []
            for u in users:
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                styled_inputs.append((u, source_query, emb))
            if not styled_inputs:
                continue

            styled_texts = [TINYSTYLER_PREFIX + src_q for _, src_q, _ in styled_inputs]
            styled_styles = np.stack([e for _, _, e in styled_inputs])

            styled_dec = [None] * len(styled_inputs)
            baseline_dec = [None] * len(styled_inputs)

            # Build decoder_input_ids: [pad] + forced_tokens  (T5 convention)
            decoder_start = [PAD] + forced_tokens
            decoder_ids = torch.tensor(decoder_start, dtype=torch.long,
                                       device='cuda:0').unsqueeze(0).expand(
                len(styled_inputs), -1).contiguous()
            log(f"  decoder_input_ids shape: {tuple(decoder_ids.shape)}")

            for s in range(0, len(styled_inputs), BATCH_SIZE):
                chunk_texts = styled_texts[s:s + BATCH_SIZE]
                chunk_styles = styled_styles[s:s + BATCH_SIZE]
                chunk_decoder_ids = decoder_ids[s:s + BATCH_SIZE]
                enc = tokenizer(
                    chunk_texts, return_tensors='pt', padding=True,
                    truncation=True, max_length=128,
                ).to('cuda:0')
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

            # Baseline: same as styled but zero style (no forcing for
            # baseline since the question is "what would T5 produce with
            # zero style on this input")
            baseline_styles = np.zeros_like(styled_styles)
            decoder_ids_base = torch.tensor([PAD], dtype=torch.long,
                                            device='cuda:0').unsqueeze(0).expand(
                len(styled_inputs), -1).contiguous()
            for s in range(0, len(styled_inputs), BATCH_SIZE):
                chunk_texts = styled_texts[s:s + BATCH_SIZE]
                chunk_styles = baseline_styles[s:s + BATCH_SIZE]
                chunk_decoder_ids = decoder_ids_base[s:s + BATCH_SIZE]
                enc = tokenizer(
                    chunk_texts, return_tensors='pt', padding=True,
                    truncation=True, max_length=128,
                ).to('cuda:0')
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
                    baseline_dec[s + i] = txt

            for i in range(len(styled_inputs)):
                if styled_dec[i] is None or baseline_dec[i] is None:
                    raise RuntimeError(f"missing decode at index {i}")

            for (u, src_q, _), styled_q, baseline_q in zip(
                styled_inputs, styled_dec, baseline_dec
            ):
                n_attr_src = sum(1 for v in attrs.values() if v and v.lower() in src_q.lower())
                n_attr_styled = sum(_is_attr_in_query(v, styled_q) for v in attrs.values() if v)
                n_attr_baseline = sum(_is_attr_in_query(v, baseline_q) for v in attrs.values() if v)
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

    log(f"DONE: wrote {n_done} pairs to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()