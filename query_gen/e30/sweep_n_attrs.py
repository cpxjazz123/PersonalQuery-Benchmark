"""E30 — sweep n_attrs ∈ {5,6,7,8,9,10} at fixed K=8/α=2 TinyStyler steering.

Single product (B00ECHYTBI), 10 candidate users from e30_styled_queries.jsonl.

For each n in {5,6,7,8,9,10}:
  1. Build prompt with A1..A_n attrs
  2. Run copy-aware Qwen → source_query_n
  3. Run TinyStyler K=8/α=2 with MAX_NEW_TOKENS=64 for all 10 users → styled_queries_n
  4. Measure: source len, styled len, n_unique, n_attrs_in_styled, cos_user
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

REPO_ROOT = Path('/fs04/ar57/wenyu/PersoanlQuery')
sys.path.insert(0, str(REPO_ROOT / 'query'))
sys.path.insert(0, str(REPO_ROOT / 'query' / 'soft_prefix'))
sys.path.insert(0, str(REPO_ROOT / 'TinyStyler' / 'tinystyler'))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinystyler import TinyStyler
from sentence_transformers import SentenceTransformer

from copy_aware import CopyAwareHead, find_attr_value_spans
from projector import SoftPrefixProjector
from copy_aware_generate import CopyAwareGenerator, build_messages

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
CKPT_DIR = Path('/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p/checkpoint')
BASE_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'
ANNAWEGMANN_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots/d7d0f5ca829316a8f5695e49dfce80b86db5e76c'

K_PREFIX = 8
ALPHA = 2.0
MAX_NEW_TOKENS = 64
NUM_BEAMS = 2
N_VALUES = [5, 6, 7, 8, 9, 10]
TARGET_ASIN = 'B00ECHYTBI'

OUT_JSON = Path(f'{PAPER}/e30_n_attrs_sweep.json')


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


def count_attrs_in(text, attrs_dict):
    """Count attribute values present in text (case-insensitive substring)."""
    text_lc = text.lower()
    n = 0
    for v in attrs_dict.values():
        if v and v.lower() in text_lc:
            n += 1
    return n


def main():
    log("=== E30 n_attrs sweep (single product, K=8/α=2 fixed) ===")

    # ---- Load Qwen (copy-aware) ----
    log("loading copy-aware Qwen...")
    t0 = time.time()
    cfg = json.load(open(CKPT_DIR / 'config.json'))
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR / 'tokenizer', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg['base_model_path'], torch_dtype=torch.bfloat16,
        device_map='cuda:0', trust_remote_code=True,
    )
    lora_dir = CKPT_DIR / 'lora_adapter'
    if lora_dir.exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(lora_dir))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    projector = SoftPrefixProjector(cfg['user_dim'], 128, cfg['num_tokens'],
                                    cfg['model_dim'], dtype=torch.bfloat16,
                                    gate_init=cfg.get('gate_init', 1e-3)).to('cuda:0')
    projector.load_state_dict(torch.load(CKPT_DIR / 'projector.pt', map_location='cuda:0'))
    vocab_size = base.get_output_embeddings().weight.size(0)
    copy_head = CopyAwareHead(cfg['model_dim'], vocab_size, dtype=torch.bfloat16).to('cuda:0')
    copy_head.load_state_dict(torch.load(CKPT_DIR / 'copy_head.pt', map_location='cuda:0'))
    qwen_model = CopyAwareGenerator(base, projector, copy_head, device='cuda:0')
    log(f"  Qwen loaded in {time.time()-t0:.0f}s")

    # ---- Load TinyStyler ----
    log("loading TinyStyler (fp16)...")
    t0 = time.time()
    ts = TinyStyler(base_model='google/t5-v1_1-large', use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu')
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = ts.state_dict(); cur.update(saved); ts.load_state_dict(cur)
    ts.to('cuda:0').half().eval()
    for p in ts.parameters():
        p.requires_grad_(False)
    t5_tok = AutoTokenizer.from_pretrained(T5_SNAP, legacy=True)
    log(f"  TinyStyler loaded in {time.time()-t0:.0f}s")

    # ---- Load user embs + users ----
    npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
    uids_arr = npz['user_ids']
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids_arr)}

    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])
    styles = np.stack([uid_to_emb[u] for u in users])

    # ---- Load AnnaWegmann for style fidelity ----
    log("loading AnnaWegmann...")
    aw = SentenceTransformer(ANNAWEGMANN_SNAP).to('cuda:0').eval()
    log(f"  AnnaWegmann loaded; dim={aw.get_embedding_dimension()}")

    # ---- Load 10-attr product spec ----
    with open(f'{PAPER}/e30_picked_products_nattrs.json') as f:
        picked = json.load(f)
    target = next(p for p in picked['products'] if p['asin'] == TARGET_ASIN)
    attrs_10 = target['attrs_n']
    log(f"\ntarget product: {TARGET_ASIN}")
    log(f"  A1: {attrs_10['A1'][:60]!r}")
    log(f"  A2: {attrs_10['A2'][:60]!r}")
    log(f"  A3: {attrs_10['A3'][:60]!r}")
    log(f"  ... ({len(attrs_10)} attrs total)")

    # ---- Sweep ----
    ZERO_VEC = np.zeros(cfg['user_dim'], dtype=np.float32)
    results = []

    log(f"\n{'n':>3} | {'src_tok':>7} | {'sty_tok':>7} | {'uniq':>5} | "
        f"{'attrs_src':>9} | {'attrs_sty':>9} | {'cos_user':>8} | {'cos_src':>8}")
    log('-' * 80)

    for n in N_VALUES:
        # Build attrs dict for this n
        attrs_n = {f'A{i}': attrs_10[f'A{i}'] for i in range(1, n + 1)}
        # Generate source query
        prompt = tokenizer.apply_chat_template(
            build_messages(attrs_n), tokenize=False, add_generation_prompt=True,
        )
        with torch.no_grad():
            outs = qwen_model.generate_batch(
                [prompt], [attrs_n], [ZERO_VEC], tokenizer,
                MAX_NEW_TOKENS, tokenizer.eos_token_id,
            )
        source_query = outs[0]

        # Generate styled queries via TinyStyler K=8/α=2 for all 10 users
        enc = t5_tok([source_query] * len(users), return_tensors='pt', padding=True,
                      truncation=True, max_length=128).to('cuda:0')
        style_t = torch.from_numpy(styles).to('cuda:0').half()
        with torch.no_grad():
            out_ids = gen_k_prefix(ts, enc['input_ids'], enc['attention_mask'],
                                    style=style_t, k_prefix=K_PREFIX, alpha=ALPHA,
                                    max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
        styled = t5_tok.batch_decode(out_ids, skip_special_tokens=True)

        # Encode for cos_user / cos_src
        with torch.no_grad():
            styled_embs = aw.encode(styled, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False)
            src_emb = aw.encode([source_query], convert_to_tensor=True,
                                  normalize_embeddings=True, show_progress_bar=False)[0]
        user_refs = torch.from_numpy(styles).to('cuda:0').float()
        user_refs = user_refs / (user_refs.norm(dim=1, keepdim=True) + 1e-12)
        cos_user_per = (user_refs * styled_embs.float()).sum(dim=1).cpu().numpy()
        cos_src_per = (styled_embs @ src_emb).cpu().numpy()

        # Metrics
        n_unique = len(set(styled))
        src_tok = len(source_query.split())
        sty_tok = float(np.mean([len(s.split()) for s in styled]))
        attrs_src = count_attrs_in(source_query, attrs_n)
        attrs_sty = float(np.mean([count_attrs_in(s, attrs_n) for s in styled]))
        cos_user_mean = float(cos_user_per.mean())
        cos_src_mean = float(cos_src_per.mean())

        log(f"{n:>3} | {src_tok:>7} | {sty_tok:>7.1f} | {n_unique:>2}/10 | "
            f"{attrs_src:>5}/{n}    | {attrs_sty:>5.2f}/{n}   | "
            f"{cos_user_mean:>+8.4f} | {cos_src_mean:>+8.4f}")

        results.append({
            'n_attrs': n,
            'attrs_n': attrs_n,
            'source_query': source_query,
            'source_tok': src_tok,
            'styled_queries': styled,
            'styled_tok_mean': sty_tok,
            'n_unique': n_unique,
            'n_attrs_in_source': attrs_src,
            'n_attrs_in_styled_mean': attrs_sty,
            'cos_user_mean': cos_user_mean,
            'cos_src_mean': cos_src_mean,
        })

    # write results
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\nwrote {OUT_JSON}")

    # sample dump
    log("\n--- sample styled queries at n=10 ---")
    for r in results:
        if r['n_attrs'] == 10:
            for i, s in enumerate(r['styled_queries'][:5]):
                log(f"  [{users[i][:6]}] {s!r}")
            break


if __name__ == '__main__':
    main()