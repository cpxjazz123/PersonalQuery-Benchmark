#!/usr/bin/env python3
"""Phase 14.Q-7 gen: vLLM batched generation."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PROMPTS = OUT_DIR / "phase14_q7_generation_prompts.jsonl"
OUT_GEN = OUT_DIR / "phase14_q7_generated.jsonl"

K_PER_PAIR = 8
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q-7 gen: vLLM batched generation")
    log("=" * 70)

    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client

    log("[1] Loading prompts ...")
    prompts = []
    with IN_PROMPTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    log(f"  prompts: {len(prompts)}")

    log("[2] vLLM batched generate ...")
    client = create_qwen_local_client(with_vllm=True)
    log(f"  client loaded (model: {client.model_name})")

    all_prompts_text = []
    all_prompt_meta = []
    for pi, p in enumerate(prompts):
        for sk in range(K_PER_PAIR):
            full_prompt = f"{p['prompt']}\n\n(variant {sk + 1}/{K_PER_PAIR})"
            all_prompts_text.append(full_prompt)
            all_prompt_meta.append((pi, sk))
    log(f"  total: {len(all_prompts_text)} ({len(prompts)} prompts × {K_PER_PAIR})")

    chat_prompts = [
        client._backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": p_text}],
            tokenize=False, add_generation_prompt=True,
        )
        for p_text in all_prompts_text
    ]

    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )

    t0 = time.time()
    log(f"  generating ...")
    outputs = client._backend.model.generate(chat_prompts, sampling)
    texts = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
    elapsed = time.time() - t0
    log(f"  done in {elapsed:.1f}s ({len(chat_prompts)/elapsed:.1f} prompts/s)")

    log("[3] Saving outputs ...")
    out_records = []
    for (pi, sk), text in zip(all_prompt_meta, texts):
        p = prompts[pi]
        out_records.append({
            "user_id": p["user_id"],
            "asin": p["asin"],
            "condition": p["condition"],
            "layer": None,
            "alpha": 0.0,
            "cand_local_idx": sk,
            "q_styled": text,
            "q_final_post": text,
        })

    with OUT_GEN.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GEN} ({len(out_records)} records)")
    log("=" * 70)
    log("PHASE 14.Q-7 GEN COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
