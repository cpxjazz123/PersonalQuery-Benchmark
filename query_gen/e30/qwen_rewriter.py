"""E30 — Qwen as style-controlled query rewriter (replaces TinyStyler).

Strategy: zero/few-shot prompting with Qwen2-7B-Instruct.
For each user, we provide 3 example sentences from their writing, then ask
Qwen to rewrite the source query in the user's style while preserving all
attributes.

We test 0-shot vs 3-shot to see if user examples help style transfer.
"""
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
BASE_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'

USER_SENTS_PKL = f'{PAPER}/user_sents_cache2.pkl'

X_WORDS = 8  # min words filter (consistent with E30 X=8 cell)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_prompt(source_query, attrs, user_examples=None):
    """Build Qwen chat-template prompt for rewriting source_query in user style.

    user_examples: list of 3 strings (user's actual review sentences).
    """
    attr_list = '\n'.join(f'  - {k}: {v}' for k, v in attrs.items() if v)

    system = (
        "You rewrite shopping queries. Preserve EVERY product attribute verbatim. "
        "Match the writing style of the user examples. Output ONLY the rewritten "
        "query, no quotes, no explanation."
    )

    user_msg_parts = []
    user_msg_parts.append(f"Attributes (preserve verbatim):\n{attr_list}")
    user_msg_parts.append(f"\nOriginal query:\n{source_query}")
    if user_examples:
        examples_text = '\n'.join(f'- {e}' for e in user_examples)
        user_msg_parts.append(
            f"\nUser writing examples (match their style):\n{examples_text}"
        )
    user_msg_parts.append("\nRewritten query:")

    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': '\n'.join(user_msg_parts)},
    ]


def main():
    log("loading Qwen2-7B-Instruct...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.padding_side = 'left'  # CRITICAL for batched decoder generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map='cuda:0', trust_remote_code=True,
    )
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    log(f"  Qwen loaded in {time.time()-t0:.0f}s")

    # load 4-attr source query + 10 users from e30_styled_queries.jsonl
    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]
    source_query = pairs[0]['source_query']
    attrs = pairs[0]['attrs']
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])

    # load user sentences cache
    log("loading user_sents_cache2...")
    with open(USER_SENTS_PKL, 'rb') as f:
        usc = pickle.load(f)

    # collect 3 examples per user (longest first)
    user_examples = {}
    for u in users:
        if u not in usc:
            user_examples[u] = []
            continue
        sents = [(s, wc) for s, wc, _ in usc[u] if wc >= X_WORDS]
        # sort by word count desc, take top 3
        sents.sort(key=lambda x: x[1], reverse=True)
        user_examples[u] = [s for s, _ in sents[:3]]

    # generate for each user (0-shot then 3-shot)
    log(f"\n=== rewriting source_query for {len(users)} users ===\n")
    log(f"Source query: {source_query!r}\n")
    log(f"Attrs: {attrs}\n")

    # mode: '0shot' or '3shot'
    for mode in ['0shot', '3shot']:
        log(f"\n--- mode: {mode} ---")
        prompts = []
        for u in users:
            msgs = build_prompt(
                source_query, attrs,
                user_examples[u] if mode == '3shot' else None,
            )
            prompt_str = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt_str)

        # batch generate
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                   max_length=2048).to('cuda:0')

        t1 = time.time()
        with torch.no_grad():
            out_ids = mdl.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_new_tokens=64,
                num_beams=2,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_token_ids = out_ids[:, enc['input_ids'].shape[1]:]
        decoded = tok.batch_decode(new_token_ids, skip_special_tokens=True)
        elapsed = time.time() - t1
        c = Counter(decoded)
        n_unique = len(c)
        log(f"  {n_unique}/10 unique, {elapsed:.0f}s")
        for i, (u, d) in enumerate(zip(users, decoded)):
            d_clean = d.strip().split('\n')[0]
            n_attrs = sum(1 for v in attrs.values() if v and v in d_clean)
            log(f"  [{u[:6]}] ({n_attrs}/{len(attrs)} attrs) {d_clean[:150]!r}")


if __name__ == '__main__':
    main()