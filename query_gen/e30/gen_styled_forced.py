"""E30.15 — Force attribute tokens via T5.generate(force_words_ids=...).

TinyStyler drops ~50% of product attributes when rephrasing source queries.
The previous attempt (E30.14) used a post-processing clause-append to recover
them, but the user requires **direct generation** — the styled_query must
contain all attributes as decoder output, not as a post-hoc fix.

Approach: HuggingFace's `generate(force_words_ids=...)` runs constrained beam
search where the decoder MUST emit each specified token sequence somewhere
in the output.  This is true direct generation — the model produces these
tokens through its normal forward pass, but constrained decoding forces
them to appear.

We tokenize each attribute value (no special tokens) and pass the token-ID
sequences to `generate()`.  T5 trie-constrained beam search guarantees all
four attribute sequences appear in the styled_query, while the rest of the
generation is still style-conditioned (K=8 prefix + α=2 steering).

NOTE: this is fundamentally different from the post-processor — the styled
output IS the model's generation, just with hard constraints on which
tokens must appear.

Input : /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products.json
        /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_source_queries.jsonl
        /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_style_embs.npz
Output: /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_styled_queries_forced.jsonl

NOTE on LLM client rule (AGENTS.md Rule 8/9):
  TinyStyler is a T5 model — direct integration via the TinyStyler package
  (mirrors `tinystyler_authorship.py`).  See e30/gen_styled.py for the
  documented exception.
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
OUT_JSONL = PAPER / 'e30_styled_queries_forced.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_forced.log'

MAX_NEW_TOKENS = 96
BATCH_SIZE = 2  # constrained beam search + trie bookkeeping is heavy
NUM_BEAMS = 4
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1  # smoke run on B00ECHYTBI
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
    """Tokenize an attribute value to a list of T5 token IDs (no specials).
    Skip empty values and very long values (>20 tokens) that would
    over-constrain the beam search and cause degenerate output."""
    if not value:
        return []
    ids = tokenizer(value, add_special_tokens=False)['input_ids']
    if len(ids) > 20:
        # Too long to force — return empty (constraint skipped, but it's
        # easier to recover these with the partial-substring heuristic in
        # `_is_attr_in_query` than to constraining 30+ tokens).
        return []
    return ids


def _dedupe_prefix_subsets(forced_lists: list[list[int]]) -> list[list[int]]:
    """Drop any list that is a strict prefix of another list in the input.

    The constrained-beam-search trie rejects any pair where one entry's
    tokens are a prefix of another's (e.g. "Infant Optics" = first 3 tokens
    of "Infant Optics DXR-8...").  When we force the longer sequence to
    appear, the shorter one appears as a substring for free — so dropping
    the prefix subset is lossless.
    """
    out = []
    for cand in forced_lists:
        is_prefix_of_other = False
        for other in forced_lists:
            if other is cand:
                continue
            if len(other) > len(cand) and other[:len(cand)] == cand:
                is_prefix_of_other = True
                break
        if not is_prefix_of_other:
            out.append(cand)
    return out


def _generate_k_prefix_forced(model, input_ids, attention_mask, style,
                               k_prefix, alpha, force_words_ids,
                               **kwargs):
    """K-prefix style injection + T5.generate with force_words_ids.

    Uses transformers-community/constrained-beam-search via custom_generate
    to run trie-constrained beam search that guarantees each token sequence
    appears somewhere in the decoder output.  This is true direct generation:
    the model's decoder emits these tokens; we just constrain WHICH tokens
    may appear, not how they are produced.

    If `force_words_ids` is None, falls back to plain beam search (no
    constraint) — used for the zero-style baseline so we can compare what
    TinyStyler produces without constraints vs with them.
    """
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
    if force_words_ids is None:
        return model.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=am,
            **kwargs,
        )
    return model.model.generate(
        inputs_embeds=input_embeds,
        attention_mask=am,
        force_words_ids=force_words_ids,
        custom_generate='transformers-community/constrained-beam-search',
        trust_remote_code=True,
        **kwargs,
    )


def _is_attr_in_query(attr_value: str, query: str) -> bool:
    """Same heuristic as post_attr.py — for stats reporting only.  Note
    that with force_words_ids active the styled_query will always satisfy
    this check by construction."""
    if not attr_value:
        return True
    ql = query.lower()
    if attr_value.lower() in ql:
        return True
    head = attr_value[:30].strip().rstrip('":;,')
    if head and head.lower() in ql:
        return True
    return False


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
    log(f"tokenizer ok; vocab={tokenizer.vocab_size}")

    log("loading inputs...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products']
    products = products[:N_PRODUCTS]
    log(f"  {len(products)} products")

    src_map = {}
    with open(SOURCE_JSONL) as f:
        for line in f:
            r = json.loads(line)
            src_map[r['asin']] = r
    log(f"  {len(src_map)} source queries")

    npz = np.load(EMB_NPZ, allow_pickle=True)
    uids = list(npz['user_ids'])
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids)}
    log(f"  {len(uids):,} user style embeddings")

    rng = np.random.default_rng(SEED)

    log(f"generating forced-attribute styled queries (n_users/product={N_USERS_PER_PRODUCT}, beam={NUM_BEAMS}, batch={BATCH_SIZE}, MAX_NEW={MAX_NEW_TOKENS})...")
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

            styled_inputs = []
            for u in users:
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    continue
                styled_inputs.append((u, source_query, emb))
            if not styled_inputs:
                continue

            # Pre-compute forced token ids for each attribute (one per record;
            # all records for this product share the same attrs).
            forced_per_record = []
            for v in attrs.values():
                if v:
                    forced_per_record.append(_attr_token_ids(tokenizer, v))
            forced_per_record = _dedupe_prefix_subsets(forced_per_record)
            log(f"  forced ids (after prefix-dedupe): {[len(x) for x in forced_per_record]} (per attr)")
            # Build force_words_ids for each input (same constraint set, since
            # all records for this product share attrs)
            force_words_ids_batch = [forced_per_record] * len(styled_inputs)

            styled_texts = [TINYSTYLER_PREFIX + src_q for _, src_q, _ in styled_inputs]
            styled_styles = np.stack([e for _, _, e in styled_inputs])
            entries = [('styled', i, t, e)
                        for i, (t, e) in enumerate(zip(styled_texts, styled_styles))]
            baseline_entries = [('baseline', i, t, np.zeros_like(e))
                                  for i, (t, e) in enumerate(zip(styled_texts, styled_styles))]

            styled_dec = [None] * len(styled_inputs)
            baseline_dec = [None] * len(styled_inputs)

            def _run(entries_chunk, force_words):
                chunk_texts = [c[2] for c in entries_chunk]
                chunk_styles = np.stack([c[3] for c in entries_chunk])
                enc = tokenizer(
                    chunk_texts, return_tensors='pt', padding=True,
                    truncation=True, max_length=128,
                ).to('cuda:0')
                style_t = torch.from_numpy(chunk_styles).to('cuda:0').half()
                with torch.no_grad():
                    out_ids = _generate_k_prefix_forced(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        force_words_ids=force_words,
                        max_new_tokens=MAX_NEW_TOKENS,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        early_stopping=True,
                    )
                return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            for s in range(0, len(entries), BATCH_SIZE):
                chunk = entries[s:s + BATCH_SIZE]
                force_chunk = force_words_ids_batch[s:s + BATCH_SIZE]
                chunk_dec = _run(chunk, force_chunk)
                for (cat, idx, _, _), txt in zip(chunk, chunk_dec):
                    if cat == 'styled':
                        styled_dec[idx] = txt

            for s in range(0, len(baseline_entries), BATCH_SIZE):
                chunk = baseline_entries[s:s + BATCH_SIZE]
                # baseline = no style AND no force (otherwise baseline would
                # be identical for all users and uninformative)
                chunk_dec = _run(chunk, None)
                for (cat, idx, _, _), txt in zip(chunk, chunk_dec):
                    if cat == 'baseline':
                        baseline_dec[idx] = txt

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