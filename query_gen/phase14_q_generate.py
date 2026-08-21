#!/usr/bin/env python3
"""Phase 14.Q-2: Generate K=8 queries per (pair × cond) via vLLM batched.

Conditions:
  - Q_Dynamic: per-user text conditions only
  - Q_DynamicExemplar: per-user text conditions + 2 exemplars

Inputs: phase14_q_generation_prompts.jsonl (60 prompts)
Output: phase14_q_generated.jsonl (60 pairs × 8 = 480 queries)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PROMPTS = OUT_DIR / "phase14_q_generation_prompts.jsonl"
OUT_GEN = OUT_DIR / "phase14_q_generated.jsonl"
OUT_META = OUT_DIR / "phase14_q_gen_meta.json"

K_PER_PAIR = 8
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q-2: Generate K=8 queries per (pair × cond)")
    log("=" * 70)

    # === [1] Load prompts ===
    log("[1] Loading prompts ...")
    prompts = []
    with IN_PROMPTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompts.append(r)
    log(f"  prompts: {len(prompts)}")

    # === [2] vLLM batched generate ===
    log("[2] vLLM batched generate K=8 per prompt ...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client

    client = create_qwen_local_client(with_vllm=True)
    log(f"  client loaded (model: {client.model_name})")

    # Build all prompts with variant suffix (vLLM dedup same prompts)
    all_prompts_text = []
    all_prompt_meta = []  # (prompt_idx, sample_idx)
    for pi, p in enumerate(prompts):
        for sk in range(K_PER_PAIR):
            # Add unique variant suffix to avoid vLLM dedup
            full_prompt = (
                f"{p['prompt']}\n\n(variant {sk + 1}/{K_PER_PAIR})"
            )
            all_prompts_text.append(full_prompt)
            all_prompt_meta.append((pi, sk))

    log(f"  total generations: {len(all_prompts_text)} (60 prompts × 8 = 480)")

    # Apply chat template
    chat_prompts = []
    for p_text in all_prompts_text:
        chat_prompts.append(
            client._backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": p_text}],
                tokenize=False, add_generation_prompt=True,
            )
        )

    # vLLM generate via SamplingParams
    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )

    t0 = time.time()
    log(f"  generating {len(chat_prompts)} prompts (max_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}) ...")

    outputs = client._backend.model.generate(chat_prompts, sampling)
    texts = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
    elapsed = time.time() - t0
    log(f"  generation done in {elapsed:.1f}s ({len(chat_prompts)/elapsed:.1f} prompts/s)")

    # === [3] Save outputs ===
    log("[3] Saving outputs ...")
    out_records = []
    for (pi, sk), text in zip(all_prompt_meta, texts):
        p = prompts[pi]
        rec = {
            "user_id": p["user_id"],
            "asin": p["asin"],
            "condition": p["condition"],
            "layer": None,  # No StyleVector injection
            "alpha": 0.0,
            "cand_local_idx": sk,
            "q_styled": text,
            "q_final_post": text,  # post-processing later
            "attrs_str": p["prompt"].split("Product attributes:\n")[1].split("\n\n")[0] if "Product attributes:" in p["prompt"] else "",
        }
        out_records.append(rec)

    with OUT_GEN.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  generated → {OUT_GEN} ({len(out_records)} records)")

    # Quick stats
    n_empty = sum(1 for r in out_records if not r["q_styled"])
    log(f"  empty queries: {n_empty}")

    # Length stats
    import numpy as np
    lens = [len(r["q_styled"]) for r in out_records if r["q_styled"]]
    log(f"  char length: mean={np.mean(lens):.1f}, median={np.median(lens):.1f}, max={max(lens)}, min={min(lens)}")

    meta = {
        "phase": "14.Q-2",
        "method": "vLLM K=8 batched generate per (pair × cond)",
        "n_prompts": len(prompts),
        "k_per_pair": K_PER_PAIR,
        "total_generated": len(out_records),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "elapsed_s": elapsed,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-2 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()