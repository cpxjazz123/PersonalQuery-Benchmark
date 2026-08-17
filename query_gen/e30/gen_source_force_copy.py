"""E30.13 — Force copy head p_copy=1.0 for full attribute preservation.

Hypothesis: CopyAwareHead's p_copy gate is zero-init, so at inference the
learned gate outputs ~0 (gate[-1].weight and bias are both zero), meaning
P_generate dominates and copy branch is essentially never used.

Fix: at inference, override p_copy to 1.0 → P = P_copy only. This forces the
model to attend to source attribute positions and copy verbatim.

Compare against baseline (gen_source.py) on 4-attr / 6-attr / 8-attr sets.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path('/fs04/ar57/wenyu/PersoanlQuery')
sys.path.insert(0, str(REPO_ROOT / 'query'))
sys.path.insert(0, str(REPO_ROOT / 'query' / 'soft_prefix'))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from copy_aware import CopyAwareHead
from projector import SoftPrefixProjector
from copy_aware_generate import CopyAwareGenerator, build_messages

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
CKPT_DIR = Path('/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p/checkpoint')
BASE_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'

PICKED_JSON = Path(f'{PAPER}/e30_picked_products_nattrs.json')
OUT_JSONL = Path(f'{PAPER}/e30_source_queries_force_copy.jsonl')
OUT_LOG = Path(f'{PAPER}/e30_gen_source_force_copy.log')

MAX_NEW_TOKENS = 128
BATCH_SIZE = 4
N_VALUES = [4, 6, 8]
FORCE_P_COPY = 1.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
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

    # MONKEY-PATCH: override p_copy in copy_head forward to FORCE_P_COPY
    orig_forward = copy_head.forward

    def forward_forced(hidden, src_hidden, src_attr_mask, src_ids):
        # call original but replace p_copy with constant
        p_copy_learned, copy_logits = orig_forward(hidden, src_hidden,
                                                     src_attr_mask, src_ids)
        B, T, _ = p_copy_learned.shape
        p_copy_forced = torch.full((B, T, 1), FORCE_P_COPY,
                                    device=p_copy_learned.device,
                                    dtype=p_copy_learned.dtype)
        return p_copy_forced, copy_logits

    copy_head.forward = forward_forced
    log(f"loaded checkpoint K={cfg['num_tokens']} (force p_copy={FORCE_P_COPY})")

    log("loading picked products...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products']
    log(f"  {len(products)} products")
    target = next(p for p in products if p['asin'] == 'B00ECHYTBI')
    attrs_10 = target['attrs_n']
    log(f"  target: {target['asin']}")

    # Build prompts for n in N_VALUES
    ZERO_VEC = np.zeros(cfg['user_dim'], dtype=np.float32)
    results = []
    for n in N_VALUES:
        attrs_n = {f'A{i}': attrs_10[f'A{i}'] for i in range(1, n + 1)}
        prompt = tokenizer.apply_chat_template(
            build_messages(attrs_n), tokenize=False, add_generation_prompt=True,
        )
        with torch.no_grad():
            outs = model.generate_batch(
                [prompt], [attrs_n], [ZERO_VEC], tokenizer,
                MAX_NEW_TOKENS, tokenizer.eos_token_id,
            )
        q = outs[0]
        # count preserved attrs (lowercase substring)
        n_pres = sum(1 for v in attrs_n.values() if v and v.lower() in q.lower())
        results.append((n, q, n_pres))
        log(f"\n  n_attrs={n} ({n_pres}/{n} preserved):")
        log(f"  Q: {q!r}")

    # write
    with open(OUT_JSONL, 'w') as f:
        for n, q, n_pres in results:
            f.write(json.dumps({
                'asin': target['asin'],
                'n_attrs': n,
                'source_query': q,
                'n_preserved': n_pres,
                'force_p_copy': FORCE_P_COPY,
            }, ensure_ascii=False) + '\n')
    log(f"\nwrote {OUT_JSONL}")


if __name__ == '__main__':
    main()