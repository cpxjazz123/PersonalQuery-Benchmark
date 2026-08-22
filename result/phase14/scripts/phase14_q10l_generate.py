#!/usr/bin/env python3
"""Phase 14.Q10-L: Generate K=8 D_off candidates for 2041 user-asin pairs.

Validation at scale (Baby Products 2023 users with >=30 reviews).

Pipeline:
- 2041 pairs × K=8 = 16,328 queries
- D_off condition (no style injection, baseline)
- vLLM batched generation

Output:
- phase14_q10l_generated.jsonl: 16,328 records
"""
from __future__ import annotations
import json
import time
from pathlib import Path

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"
OUT_GEN = OUT_DIR / "phase14_q10l_generated.jsonl"

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)

K = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 64


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_missing_attrs(q: str, attrs_str: str) -> str:
    q_low = q.lower()
    missing = []
    for kv in attrs_str.split(","):
        kv = kv.strip()
        if ":" not in kv:
            continue
        _, v = kv.split(":", 1)
        v = v.strip()
        if v and v.lower() not in q_low:
            missing.append(v)
    if missing:
        q = q.rstrip(".") + ", " + ", ".join(missing) + "."
    return q


def main() -> None:
    log("=" * 70)
    log(f"Phase 14.Q10-L: Generate {K}× D_off candidates for 2041 pairs (vLLM batched)")
    log("=" * 70)

    log("[1] Loading pairs ...")
    pairs = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    log("[2] Loading vLLM client ...")
    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=True)
    log(f"  client loaded (vLLM backend)")

    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P if TEMPERATURE > 0 else 1.0,
    )

    log("[3] Building prompts ...")
    prompts_with_meta = []
    for pair in pairs:
        attrs_str = ", ".join(f"{k}: {v}" for k, v in pair["attrs"].items() if v)
        prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
        # K variants per pair, with (variant N/K) suffix to avoid vLLM dedup
        for k in range(K):
            suffix = f" (variant {k + 1}/{K})"
            prompts_with_meta.append({
                "user_id": pair["user_id"],
                "asin": pair["asin"],
                "attrs": pair["attrs"],
                "attrs_str": attrs_str,
                "k": k,
                "prompt": prompt + suffix,
            })
    log(f"  total prompts: {len(prompts_with_meta)} (expected {len(pairs)}×{K} = {len(pairs)*K})")

    log("[4] Apply chat template ...")
    full_prompts = [
        client._backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_with_meta
    ]

    log("[5] Batched vLLM generation ...")
    t0 = time.time()
    BATCH = 256
    outputs = []
    for i in range(0, len(full_prompts), BATCH):
        batch_prompts = full_prompts[i: i + BATCH]
        out = client._backend.model.generate(batch_prompts, sampling, use_tqdm=False)
        outputs.extend(out)
        if (i // BATCH) % 5 == 0:
            elapsed = time.time() - t0
            done = min(i + BATCH, len(full_prompts))
            rate = done / elapsed
            eta = (len(full_prompts) - done) / rate
            log(f"  gen {done}/{len(full_prompts)} ({rate:.1f} q/s, ETA {eta:.0f}s)")
    log(f"  done in {time.time() - t0:.1f}s")

    log("[6] Write jsonl ...")
    with OUT_GEN.open("w") as f:
        for meta, out in zip(prompts_with_meta, outputs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            q_post = append_missing_attrs(text, meta["attrs_str"])
            rec = {
                "user_id": meta["user_id"],
                "asin": meta["asin"],
                "attrs": meta["attrs"],
                "attrs_str": meta["attrs_str"],
                "condition": "D_off",
                "k": meta["k"],
                "q_styled": text,
                "q_final_post": q_post,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GEN}")

    log("=" * 70)
    log("PHASE 14.Q10-L GENERATION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()