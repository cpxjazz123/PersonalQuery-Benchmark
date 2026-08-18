"""E30.30 — E30.29 + greedy unique assignment.

E30.29 max-marginal rerank → 8-9/10 unique chosen. The remaining
collision is u4, u5, u9 all picking the same truncated template
"I'm looking for the Infant Optics DXR-8 Video Baby Monitor." because
their style embeddings are similar (low max style cos).

Strategy: greedy assignment with "already-assigned" penalty.
After user i picks a candidate, mark that candidate as taken.
For user i+1, penalize candidates that are already taken (force
distinct picks).

Process users in order of largest style embedding spread first
(most distinctive users get first pick), then less distinctive users
get the remaining candidates.
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
OUT_JSONL = PAPER / 'e30_styled_queries_v2_unique_assignment.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_v2_unique_assignment.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 2
NUM_RETURN_SEQUENCES = 48
TEMPERATURE = 1.0
TOP_P = 0.95
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0
MARGINAL_PENALTY = 0.5
ASSIGNED_PENALTY = 1.0  # heavy penalty for already-assigned candidates

FORCE_KEYS = ['A3', 'A4']
ANNAWEGMANN_PATH = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots'


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


def _build_forced_prefix(tokenizer, attrs: dict, force_keys: list[str]) -> list[int]:
    forced_lists = []
    for k in force_keys:
        v = attrs.get(k)
        if v:
            ids = tokenizer(v, add_special_tokens=False)['input_ids']
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
    # Keep the trailing period of A4 (so substring match against "A4." works)
    # Then append space + "I'm" so continuation reads naturally
    i_id = tokenizer("I'm", add_special_tokens=False)['input_ids']
    # If last token is period, append space; else append space + period + space
    # Simpler: just append " I'm" — model continues with " looking for..."
    if forced and forced[-1] != period_id:
        forced.append(space_id)
    else:
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

    log("loading AnnaWegmann style encoder for rerank...")
    import glob
    aw_snapshot = sorted(glob.glob(os.path.join(ANNAWEGMANN_PATH, '*')))[-1]
    from sentence_transformers import SentenceTransformer
    aw_model = SentenceTransformer(aw_snapshot, device='cuda:0')
    log(f"loaded AnnaWegmann from {aw_snapshot}")

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

            # Pass 1: collect candidates from all users
            user_data = {}
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
                    )
                candidates = tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True,
                )
                cand_pres = [_count_attrs_preserved(attrs, c) for c in candidates]
                full_idx = [i for i, p_ in enumerate(cand_pres) if p_ == len(attrs)]
                if not full_idx:
                    # DEBUG: show what candidates looked like
                    log(f"      DEBUG user {str(u)[:8]}: all 48 candidates:")
                    for ci, (q, p) in enumerate(zip(candidates, cand_pres)):
                        log(f"        [{ci:2d}] p={p}/4 | {q}")
                    raise RuntimeError(
                        f"No 4/4-preserving candidate for user {str(u)[:8]} "
                        f"(preservation range: {min(cand_pres)}-{max(cand_pres)}/4). "
                        f"Generation failed to produce complete attributes."
                    )
                cand_embs_all = aw_model.encode(
                    candidates, convert_to_numpy=True,
                    normalize_embeddings=True, batch_size=8,
                    show_progress_bar=False,
                )
                user_data[str(u)] = {
                    'candidates': candidates,
                    'cand_pres': cand_pres,
                    'cand_embs_all': cand_embs_all,
                    'full_idx': full_idx,
                    'filtered_cands': [candidates[i] for i in full_idx],
                    'filtered_embs': cand_embs_all[full_idx],
                    'emb': emb,
                    'chosen': None,
                }
                if (ui + 1) % 2 == 0:
                    log(f"    user {ui+1}/{len(users)}: {len(full_idx)}/{len(candidates)} 4/4-preserving")

            # Pass 2: greedy unique assignment with max-marginal rerank
            # Process users in order of "distinctiveness" (mean distance to other users)
            user_ids = list(user_data.keys())
            user_embs = np.stack([user_data[u]['emb'] for u in user_ids])
            user_embs_norm = user_embs / (np.linalg.norm(user_embs, axis=1, keepdims=True) + 1e-8)
            sim_matrix = user_embs_norm @ user_embs_norm.T
            np.fill_diagonal(sim_matrix, 0)
            distinctiveness = (1 - sim_matrix).sum(axis=1)  # higher = more distinctive
            # Sort from most distinctive to least
            user_order = [user_ids[i] for i in np.argsort(-distinctiveness)]
            log(f"  user order (most distinctive first): {[u[:8] for u in user_order]}")

            taken_strs = set()  # track assigned string candidates
            for u_str in user_order:
                ud = user_data[u_str]
                cand_embs = ud['filtered_embs']
                user_norm = ud['emb'] / (np.linalg.norm(ud['emb']) + 1e-8)
                own_cos = cand_embs @ user_norm
                # Max other cos
                other_cos = []
                for other_u in user_ids:
                    if other_u == u_str:
                        continue
                    other_norm = user_data[other_u]['emb'] / (
                        np.linalg.norm(user_data[other_u]['emb']) + 1e-8
                    )
                    other_cos.append(cand_embs @ other_norm)
                if other_cos:
                    other_cos = np.stack(other_cos, axis=0)
                    max_other_cos = other_cos.max(axis=0)
                else:
                    max_other_cos = np.zeros(len(cand_embs))
                # Already-assigned penalty: for each candidate, check if its string
                # is in taken_strs
                assigned_pen = np.array([
                    ASSIGNED_PENALTY if c in taken_strs else 0.0
                    for c in ud['filtered_cands']
                ])
                scores = own_cos - MARGINAL_PENALTY * max_other_cos - assigned_pen
                rel_idx = int(np.argmax(scores))
                best_idx = ud['full_idx'][rel_idx]
                chosen_q = ud['filtered_cands'][rel_idx]
                chosen_pre = ud['cand_pres'][best_idx]
                chosen_cos = float(own_cos[rel_idx])
                chosen_max_other = float(max_other_cos[rel_idx])
                # Mark as taken
                taken_strs.add(chosen_q)
                rec = {
                    'asin': asin,
                    'user_id': u_str,
                    'attrs': attrs,
                    'source_query': source_query,
                    'styled_query': chosen_q,
                    'n_attrs': len([v for v in attrs.values() if v]),
                    'n_attrs_in_styled': chosen_pre,
                    'n_candidates': len(ud['candidates']),
                    'n_unique_candidates': len(set(ud['candidates'])),
                    'n_44_candidates': len(ud['full_idx']),
                    'best_idx': best_idx,
                    'chosen_style_cos': chosen_cos,
                    'chosen_max_other_cos': chosen_max_other,
                    'candidates_preservation': ud['cand_pres'],
                    'candidates_style_cos': own_cos.tolist(),
                    'all_candidates': ud['candidates'],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
