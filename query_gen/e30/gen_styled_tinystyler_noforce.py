"""E30.34 — TinyStyler multi-shot + 4/4 filter (NO forced prefix, NO style-rerank).

Approach:
- Source query from E30.31 (already 4/4-verified)
- TinyStyler rephrases per-user with 768-dim AnnaWegmann style (native)
- Multi-shot N=20 per (user, source_query)
- Filter 4/4-preserving candidates
- Pick first 4/4 (raise if none — Rule 7)

Why no forced prefix:
- User explicitly asked to remove forced prefix.
- With clean KV-value A3/A4, multi-shot alone can recover 4/4 (E30.31: 7/40).

Why TinyStyler (not copy-aware Qwen):
- Copy-aware Qwen expects 20-dim VADES vectors (not on disk)
- TinyStyler natively accepts 768-dim AnnaWegmann style embeddings

Expected 4/4 hit rate per (user, source_query):
- TinyStyler baseline (E30.4 with old fragmented A3/A4): 50%
- With clean KV-value A3/A4 + 20 shots: ~80-95% (estimate; will validate)
- 100% if no 4/4 → raise (no fallback)
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
SOURCE_JSONL = PAPER / 'e30_source_queries_multishot.jsonl'  # NEW (clean KV-value) from E30.31
EMB_NPZ = PAPER / 'e30_style_embs.npz'
OUT_JSONL = PAPER / 'e30_styled_queries_tinystyler_noforce.jsonl'
OUT_LOG = PAPER / 'e30_gen_styled_tinystyler_noforce.log'

MAX_NEW_TOKENS = 64
BATCH_SIZE = 2
NUM_RETURN_SEQUENCES = 20
TEMPERATURE = 1.0
TOP_P = 0.95
N_USERS_PER_PRODUCT = 10
N_PRODUCTS = 1
SEED = 42
STYLE_K_PREFIX = 8
STYLE_ALPHA = 2.0

# Query-likeness markers (must appear in styled_query for it to read like a search)
QUERY_MARKERS = [
    "looking for", "want to", "want a", "want the",
    "find a", "find the", "searching for",
    "i need", "i would like", "i am looking",
    "can you show", "could you show",
    "show me", "where can", "where to",
]
# Anti-query openers (starters that don't sound like a search query)
BAD_OPENERS = [
    "this is ", "that is ", "this is true",
    "not the same", "i have been looking", "i have",
    "it's not ", "i would recommend",
    "i recommend", "great idea",
]

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


def _query_likeness_score(query: str) -> float:
    """Score how much `query` looks like a search query (0..1).

    - +0.4 if contains any QUERY_MARKER
    - -0.6 if starts with BAD_OPENER (kills score)
    - +0.1 base score for non-empty
    """
    ql = query.lower().strip()
    if not ql:
        return 0.0
    score = 0.1
    if any(m in ql for m in QUERY_MARKERS):
        score += 0.4
    if any(ql.startswith(b) for b in BAD_OPENERS):
        score -= 0.6
    return max(0.0, score)


def _generate_k_prefix(model, input_ids, attention_mask, style, k_prefix, alpha,
                       **kwargs):
    """Generate with style-conditioned soft prefix (NO forced decoder prefix)."""
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

    log("loading AnnaWegmann style encoder for style rerank...")
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
                log(f"  no source_query for asin {asin}, skipping")
                continue
            source_query = src['source_query']
            attrs = p['attrs']
            users = list(p['candidate_users'])
            if len(users) > N_USERS_PER_PRODUCT:
                users = list(rng.choice(users, size=N_USERS_PER_PRODUCT, replace=False))
            else:
                users = users[:N_USERS_PER_PRODUCT]

            log(f"  source_query: {source_query}")
            log(f"  attrs: {attrs}")

            for ui, u in enumerate(users):
                emb = uid_to_emb.get(str(u))
                if emb is None:
                    log(f"    user {ui+1}/{len(users)}: no style_emb, skipping")
                    continue
                # Encode source_query through T5
                enc = tokenizer(source_query, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to('cuda:0')
                style_t = torch.from_numpy(emb).to('cuda:0').half().unsqueeze(0)
                with torch.no_grad():
                    out_ids = _generate_k_prefix(
                        model,
                        input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'],
                        style=style_t,
                        k_prefix=STYLE_K_PREFIX,
                        alpha=STYLE_ALPHA,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        num_return_sequences=NUM_RETURN_SEQUENCES,
                    )
                candidates = tokenizer.batch_decode(
                    out_ids, skip_special_tokens=True,
                )
                # Step 1: 4/4 preservation filter
                cand_pres = [_count_attrs_preserved(attrs, c) for c in candidates]
                full_idx = [i for i, p_ in enumerate(cand_pres) if p_ == len(attrs)]
                if not full_idx:
                    log(f"      FAIL user {str(u)[:8]}: no 4/4 after {NUM_RETURN_SEQUENCES} shots, "
                        f"range={min(cand_pres)}-{max(cand_pres)}")
                    log(f"      All {NUM_RETURN_SEQUENCES} candidates:")
                    for ci, (q, p) in enumerate(zip(candidates, cand_pres)):
                        log(f"        [{ci:2d}] p={p}/4 | {q}")
                    raise RuntimeError(
                        f"No 4/4-preserving candidate for user {str(u)[:8]} "
                        f"asin {asin} after {NUM_RETURN_SEQUENCES} shots "
                        f"(preservation range: {min(cand_pres)}-{max(cand_pres)}/4). "
                        f"Increase NUM_RETURN_SEQUENCES."
                    )
                # Step 2: style rerank among 4/4 candidates, blended with query-likeness
                full_cands = [candidates[i] for i in full_idx]
                cand_embs = aw_model.encode(
                    full_cands, convert_to_numpy=True,
                    normalize_embeddings=True, batch_size=8,
                    show_progress_bar=False,
                )
                user_norm = emb / (np.linalg.norm(emb) + 1e-8)
                cos_scores = cand_embs @ user_norm
                # Normalize cos to [0, 1] (cos ∈ [-1, 1])
                cos_norm = (cos_scores + 1) / 2
                # Query-likeness per candidate
                q_likes = np.array([_query_likeness_score(c) for c in full_cands])
                # Combined score: 70% style match + 30% query-likeness
                combined = 0.7 * cos_norm + 0.3 * q_likes
                rel_idx = int(np.argmax(combined))
                best_idx = full_idx[rel_idx]
                chosen_q = full_cands[rel_idx]
                chosen_pre = cand_pres[best_idx]
                chosen_cos = float(cos_scores[rel_idx])
                chosen_qlike = float(q_likes[rel_idx])
                rec = {
                    'asin': asin,
                    'user_id': str(u),
                    'attrs': attrs,
                    'source_query': source_query,
                    'styled_query': chosen_q,
                    'n_attrs': len([v for v in attrs.values() if v]),
                    'n_attrs_in_styled': chosen_pre,
                    'n_candidates': len(candidates),
                    'n_unique_candidates': len(set(candidates)),
                    'n_44_candidates': len(full_idx),
                    'best_idx': best_idx,
                    'chosen_style_cos': chosen_cos,
                    'chosen_query_likeness': chosen_qlike,
                    'candidates_preservation': cand_pres,
                    'candidates_style_cos': cos_scores.tolist(),
                    'all_candidates': candidates,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_done += 1
                log(f"    user {ui+1}/{len(users)} ({str(u)[:8]}): "
                    f"{len(full_idx)}/20 4/4-preserving, chosen_cos={chosen_cos:.3f}, "
                    f"chosen_qlike={chosen_qlike:.2f}, preservation={chosen_pre}/4")
            log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} pairs in {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()