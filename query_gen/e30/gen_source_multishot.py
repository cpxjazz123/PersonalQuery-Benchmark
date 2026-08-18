"""E30.31 — Multi-shot copy-aware Qwen + 4/4 filter (no forced prefix).

E30.30 used TinyStyler + forced prefix to guarantee 4/4 attribute preservation.
The user wants to bypass forced prefix and use only copy-aware Qwen with
multi-shot sampling.

Strategy:
- For each product, generate N=10 candidates from copy-aware Qwen
  (no style condition, no forced prefix)
- Filter to 4/4-preserving candidates
- Pick the first 4/4 one (raise if none)

Expected hit rate:
- Single-shot: 32% (E30.30 source baseline)
- 10-shot: 1 - (1 - 0.32)^10 ≈ 97.5%
- 20-shot: 1 - (1 - 0.32)^20 ≈ 99.9%

This is direct generation (no string-rewrite). Persona's copy head biases
each candidate to copy attributes, but doesn't guarantee — so we sample
multiple times and pick the first 4/4.
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
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query"))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

from projector import SoftPrefixProjector  # noqa: E402
from copy_aware import CopyAwareHead, build_attr_prompt_lines  # noqa: E402
from copy_aware_generate import CopyAwareGenerator  # noqa: E402
from copy_aware_train import build_messages  # noqa: E402

CKPT_DIR = '/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p/checkpoint'

PAPER = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper')
PICKED_JSON = PAPER / 'e30_picked_products.json'
OUT_JSONL = PAPER / 'e30_source_queries_multishot.jsonl'
OUT_LOG = PAPER / 'e30_gen_source_multishot.log'

MAX_NEW_TOKENS = 1024
N_CANDIDATES_PER_PRODUCT = 40  # 40-shot → ~99.99% 4/4 hit rate
TEMPERATURE = 1.0
TOP_K = 50
N_PRODUCTS = 1  # 0 = all 100, 1 = just the first one for quick test
# Note: B00ECHYTBI's new A3/A4 are categorical ('Plastic', 'Monitor') — model
# should easily reproduce them.
SEED = 42


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


def main():
    log(f"loading copy-aware Qwen checkpoint from {CKPT_DIR}...")
    t0 = time.time()
    cfg = json.load(open(Path(CKPT_DIR) / "config.json"))
    tokenizer = AutoTokenizer.from_pretrained(Path(CKPT_DIR) / "tokenizer", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_path"], torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )
    lora_dir = Path(CKPT_DIR) / "lora_adapter"
    if lora_dir.exists():
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, str(lora_dir))
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    projector = SoftPrefixProjector(
        cfg["user_dim"], 128, cfg["num_tokens"], cfg["model_dim"],
        dtype=torch.bfloat16, gate_init=cfg.get("gate_init", 1e-3),
    ).to("cuda:0")
    projector.load_state_dict(torch.load(Path(CKPT_DIR) / "projector.pt", map_location="cuda:0"))
    vocab_size = base.get_output_embeddings().weight.size(0)
    copy_head = CopyAwareHead(cfg["model_dim"], vocab_size, dtype=torch.bfloat16).to("cuda:0")
    copy_head.load_state_dict(torch.load(Path(CKPT_DIR) / "copy_head.pt", map_location="cuda:0"))
    model = CopyAwareGenerator(base, projector, copy_head, device="cuda:0")
    log(f"loaded model in {time.time()-t0:.0f}s")

    with open(PICKED_JSON) as f:
        picked = json.load(f)
    products = picked['products']
    if N_PRODUCTS > 0:
        products = products[:N_PRODUCTS]
    log(f"processing {len(products)} products, {N_CANDIDATES_PER_PRODUCT} candidates each")

    eos_token_id = tokenizer.eos_token_id
    rng = np.random.default_rng(SEED)
    n_done = 0
    n_total = len(products)
    n_picked_at_first = 0
    n_picked_after_n = 0
    n_failed = 0

    t0 = time.time()
    with open(OUT_JSONL, 'w') as fout:
        for pi, p in enumerate(products):
            asin = p['asin']
            attrs = p['attrs']
            # Build prompt (no user style, no forced prefix)
            messages = build_messages(attrs)
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            # Generate N candidates using single-shot generation (no batching
            # across different attrs because the length differs)
            candidates = []
            for c_i in range(N_CANDIDATES_PER_PRODUCT):
                torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
                out = model.generate(
                    prompt_str=prompt_str,
                    user_vec=None,  # no style conditioning
                    attrs=attrs,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=eos_token_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                )
                candidates.append(out)
            # Filter 4/4
            cand_pres = [_count_attrs_preserved(attrs, c) for c in candidates]
            full_idx = [i for i, p_ in enumerate(cand_pres) if p_ == len(attrs)]
            if not full_idx:
                n_failed += 1
                raise RuntimeError(
                    f"No 4/4-preserving candidate for asin {asin} after "
                    f"{N_CANDIDATES_PER_PRODUCT} shots "
                    f"(preservation range: {min(cand_pres)}-{max(cand_pres)}/4). "
                    f"Increase N_CANDIDATES_PER_PRODUCT."
                )
            # Pick first 4/4 (deterministic for reproducibility)
            chosen_idx = full_idx[0]
            chosen_q = candidates[chosen_idx]
            n_picked_at_first += 1 if chosen_idx == 0 else 0
            if chosen_idx > 0:
                n_picked_after_n += 1
            rec = {
                'asin': asin,
                'attrs': attrs,
                'source_query': chosen_q,
                'n_attrs': len([v for v in attrs.values() if v]),
                'n_attrs_in_styled': cand_pres[chosen_idx],
                'n_candidates': len(candidates),
                'chosen_idx': chosen_idx,
                'candidates_preservation': cand_pres,
                'all_candidates': candidates,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_done += 1
            if (pi + 1) % 5 == 0:
                log(f"  product {pi+1}/{len(products)}: {n_done}/{n_total} done, "
                    f"{n_picked_at_first} first-shot, {n_picked_after_n} after-resample, "
                    f"{n_failed} failed, total {time.time()-t0:.0f}s")

    log(f"DONE: wrote {n_done} records to {OUT_JSONL}")
    log(f"  First-shot 4/4: {n_picked_at_first}/{n_done} ({n_picked_at_first/n_done*100:.1f}%)")
    log(f"  After-resample 4/4: {n_picked_after_n}/{n_done} ({n_picked_after_n/n_done*100:.1f}%)")
    log(f"  Failed: {n_failed}/{n_done}")
    log(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
