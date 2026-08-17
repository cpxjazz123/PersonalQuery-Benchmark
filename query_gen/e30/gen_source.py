"""E30 Step 3 — generate source queries via copy-aware Qwen.

For each picked product, build attribute-driven prompt and run
`CopyAwareGenerator.generate_batch`. user_vec=None so the source query is
NOT conditioned on any user — it's the "content-neutral" baseline.

Writes /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_source_queries.jsonl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query"))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

from copy_aware import (  # noqa: E402
    CopyAwareHead,
    build_attr_prompt_lines,
    find_attr_value_spans,
)
from projector import SoftPrefixProjector  # noqa: E402
from copy_aware_generate import (  # noqa: E402
    CopyAwareGenerator,
    build_messages,
)

CKPT_DIR = Path('/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p/checkpoint')
BASE_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'
# E30.8: use 6-attr picked products to grow source query length
PICKED_JSON = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products_6attrs.json')
OUT_JSONL = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_source_queries_6attrs.jsonl')
OUT_LOG = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_gen_source_6attrs.log')

MAX_NEW_TOKENS = 128  # 64 was clipping A3/A4 in many products — 128 gives copy head room
BATCH_SIZE = 8
ATTRS_FIELD = 'attrs_6'  # 'attrs' for 4-attr, 'attrs_6' for 6-attr
N_PRODUCTS = 1  # 1-product smoke run
ZERO_VEC_DIM = 20  # e14p checkpoint user_dim=20; the upstream hardcoded
                   # np.zeros(34) doesn't match this checkpoint — supply a
                   # matching zero vector so the projector doesn't crash.


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


def main():
    log(f"loading checkpoint from {CKPT_DIR}")
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
    model = CopyAwareGenerator(base, projector, copy_head, device='cuda:0')
    log(f"loaded checkpoint K={cfg['num_tokens']}")

    log("loading picked products...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products']
    log(f"{len(products)} products")

    # Build prompts (apply chat template)
    prompts = []
    attrs_list = []
    for p in products:
        prompt = tokenizer.apply_chat_template(
            build_messages(p['attrs']), tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
        attrs_list.append(p['attrs'])

    # Build user_vecs=ZERO_VEC_DIM zeros per product (not None — the projector
    # expects a (B, 20) tensor when num_tokens > 0)
    log(f"generating source queries for {len(prompts)} products (batch={BATCH_SIZE})...")
    t0 = time.time()
    all_q = []
    zero_vec = np.zeros(ZERO_VEC_DIM, dtype=np.float32)
    for s in range(0, len(prompts), BATCH_SIZE):
        chunk_p = prompts[s:s + BATCH_SIZE]
        chunk_a = attrs_list[s:s + BATCH_SIZE]
        chunk_v = [zero_vec] * len(chunk_p)
        outs = model.generate_batch(
            chunk_p, chunk_a, chunk_v, tokenizer,
            MAX_NEW_TOKENS, tokenizer.eos_token_id,
        )
        all_q.extend(outs)
        log(f"  {len(all_q)}/{len(prompts)} done in {time.time()-t0:.0f}s")
    log(f"all generated in {time.time()-t0:.0f}s")

    # Write JSONL with attribute-preservation check
    with open(OUT_JSONL, 'w') as f:
        n_full = 0
        for p, q in zip(products, all_q):
            attrs = p['attrs']
            preserved = []
            missing = []
            for k, v in attrs.items():
                if v and v in q:
                    preserved.append(k)
                elif v:
                    missing.append(k)
            if not missing:
                n_full += 1
            rec = {
                'asin': p['asin'],
                'attrs': attrs,
                'source_query': q,
                'attr_preserved': preserved,
                'attr_missing': missing,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    log(f"wrote {OUT_JSONL}: {n_full}/{len(products)} preserved all attrs")


if __name__ == '__main__':
    main()