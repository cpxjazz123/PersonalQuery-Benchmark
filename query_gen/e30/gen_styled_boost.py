"""E30.21 — LogitsProcessor that boosts attribute token probabilities.

E30.15-20 all forced attribute tokens into the output and either broke
uniqueness or failed to preserve.  LogitsProcessor nudges attribute tokens
without forcing them — model still emits tokens via normal softmax, but
attribute tokens get a multiplicative bonus on their logits.

This is true direct generation with style-conditioned decoding:
- For each attribute value, tokenize it to get the token IDs.
- Add a `LogitsProcessor` that adds a constant bonus to those token IDs
  at every step (modulated by how many attribute tokens have already
  been generated so far).
- Run regular beam search — model produces natural-looking text with
  attribute tokens over-represented.

The bonus is constant (e.g. +5.0 logits) and applied to ALL attribute
tokens every step.  This pushes the model toward including attrs in its
output without breaking fluency.
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
import torch.nn.functional as F
from transformers import AutoTokenizer, LogitsProcessor

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
OUT_JSONL = PAPER / 'e30_styled_queries_boost.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_boost.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 4
NUM_BEAMS = 4
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0
# Use decaying bonus: starts at 2.0, decays to 0 over 25 attribute-token emissions.
ATTR_LOGIT_BONUS = 2.0
ATTR_T_DECAY = 25
LOGITS_PROCESSOR = 'decaying'  # 'constant' or 'decaying'


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


class AttrBoostLogitsProcessor(LogitsProcessor):
    """Add a constant bonus to attribute token logits at every step.

    Note: this is process-level, not token-level — once an attribute
    token is generated, we keep boosting all attribute tokens so the
    model can continue producing them.  This biases toward attribute
    inclusion without forcing exact sequences.
    """
    def __init__(self, attr_token_ids: list[int], bonus: float):
        self.attr_ids = torch.tensor(sorted(set(attr_token_ids)),
                                     dtype=torch.long)
        self.bonus = bonus

    def __call__(self, input_ids, scores):
        device = scores.device
        ids = self.attr_ids.to(device)
        scores.index_add_(
            1, ids,
            torch.full((scores.shape[0], len(ids)), self.bonus, device=device),
        )
        return scores


class DecayingAttrBoostLogitsProcessor(LogitsProcessor):
    """Bonus that decays as more attribute tokens get generated.  This
    prevents degenerate infinite loops (which E30.21 produced at +5
    bonus) while still biasing the model toward attribute inclusion.

    bonus_eff(t) = bonus * max(0, 1 - t/T_decay)

    where t = number of attribute tokens generated so far.  After
    T_decay tokens, the bonus is zero and generation proceeds normally.
    """
    def __init__(self, attr_token_ids: list[int], bonus: float,
                 t_decay: int = 30):
        self.attr_ids = torch.tensor(sorted(set(attr_token_ids)),
                                     dtype=torch.long)
        self.bonus = bonus
        self.t_decay = t_decay

    def __call__(self, input_ids, scores):
        device = scores.device
        ids = self.attr_ids.to(device)
        # Count how many attribute tokens have been emitted so far
        attr_mask = (input_ids.unsqueeze(-1) == ids.view(1, 1, -1)).any(-1)
        n_attr_emitted = attr_mask.sum(dim=-1).float()  # (B,)
        decay = torch.clamp(1.0 - n_attr_emitted / self.t_decay, min=0.0)
        bonus_add = (self.bonus * decay).unsqueeze(-1).expand(-1, len(ids))
        scores.index_add_(1, ids, bonus_add.to(device))
        return scores


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

            # Tokenize all attribute values to get the boost set
            attr_ids = []
            for v in attrs.values():
                attr_ids.extend(_attr_token_ids(tokenizer, v))
            log(f"  attr token IDs to boost: {len(set(attr_ids))} unique")

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

            styled_dec = [None] * len(styled_inputs)

            for s in range(0, len(styled_inputs), BATCH_SIZE):
                chunk_texts = styled_texts[s:s + BATCH_SIZE]
                chunk_styles = styled_styles[s:s + BATCH_SIZE]
                enc = tokenizer(chunk_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to('cuda:0')
                style_t = torch.from_numpy(chunk_styles).to('cuda:0').half()
                if LOGITS_PROCESSOR == 'constant':
                    boost_proc = AttrBoostLogitsProcessor(attr_ids, ATTR_LOGIT_BONUS)
                else:
                    boost_proc = DecayingAttrBoostLogitsProcessor(
                        attr_ids, ATTR_LOGIT_BONUS, ATTR_T_DECAY,
                    )
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
                        do_sample=False,
                        early_stopping=True,
                        logits_processor=[boost_proc],
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
                    'logit_bonus': ATTR_LOGIT_BONUS,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()