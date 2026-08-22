#!/usr/bin/env python3
"""Phase 16 test: Decode-time attribute constraint via guided_regex.

GOAL:
- No exemplars (just plain D_off prompt)
- No hard-copy post-processing
- ONLY at decoding stage: enforce all attribute values appear in output

APPROACH:
vLLM guided_regex per-prompt pattern:
  .*attr1_value.*attr2_value.*attr3_value.*attr4_value.*attr5_value.*

This forces the LLM to generate a query containing all attribute values
in any order with arbitrary text between them.

TEST: Generate 100 queries on first 100 pairs, measure:
1. Attr completeness rate (target: 100%)
2. Style quality sample (subjective)
3. Speed comparison vs unconstrained
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"

OUT_GEN = OUT_DIR / "phase16_test_100q.jsonl"
N_TEST = 100
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 1024

PROMPT_TEMPLATE_NO_EXEMPLAR = (
    "Write a short shopping search query for a product with these attributes:\n"
    "{attrs_str}\n\n"
    "Search query:"
)

SKIP_NUMERIC = True  # Skip numeric values (dimensions, weights) in regex constraint


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_attr_regex(attrs_str, skip_numeric=True):
    """Build guided_regex pattern requiring all (non-numeric) attribute values in output.

    Case-insensitive prefix (?i) handles "1 Pound" vs "1 Pounds" plural variants.

    skip_numeric=True: skip values that are mostly numeric (dimensions, weights).
    Numeric values are hard to enforce exactly via regex (1 pound vs 1 pounds)
    and LLM often paraphrases them, so we don't constrain them.
    """
    # Extract values from "Key: Value, Key: Value, ..."
    values = []
    for kv in attrs_str.split(","):
        kv = kv.strip()
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        v = v.strip()
        k = k.strip()
        if not v:
            continue
        if skip_numeric and _is_numeric_value(v):
            continue
        values.append(v)
    if not values:
        # No attrs to constrain, accept anything
        return "(?i).*"
    # Build (?i).*v1.*v2.*v3.*
    pattern_parts = []
    for v in values:
        # Escape and then strip escape from `\ ` to ` ` since xgrammar doesn't understand
        escaped = re.escape(v).replace("\\ ", " ").replace("\\.", ".")
        pattern_parts.append(".*" + escaped + ".*")
    return "(?i)" + "".join(pattern_parts)


def _is_numeric_value(s):
    """Heuristic: is this attr value mostly numeric (dimensions, weights, etc)?"""
    # Count digits vs letters
    n_digits = sum(1 for c in s if c.isdigit())
    n_letters = sum(1 for c in s if c.isalpha())
    # Numeric if digits dominate or starts with digit
    if s[0].isdigit():
        return True
    # Or > 50% digits
    if n_digits > n_letters:
        return True
    return False


def main() -> None:
    log("=" * 70)
    log(f"Phase 16 TEST: decode-time attr constraint on {N_TEST} queries")
    log("=" * 70)

    log("[1] Loading first 100 pairs ...")
    pairs = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    pairs = pairs[:N_TEST]

    log("[2] Building prompts + regex constraints ...")
    prompts = []
    regex_patterns = []
    for p in pairs:
        attrs_str = ", ".join(f"{k}: {v}" for k, v in p["attrs"].items() if v)
        prompt = PROMPT_TEMPLATE_NO_EXEMPLAR.format(attrs_str=attrs_str)
        prompts.append(prompt)
        regex_patterns.append(build_attr_regex(attrs_str, skip_numeric=SKIP_NUMERIC))

    log(f"  prompts: {len(prompts)}")
    log(f"  sample prompt[0]:\n    {prompts[0]}")
    log(f"  sample regex[0]: {regex_patterns[0]}")

    log("[3] Loading Qwen + vLLM (low gpu_memory_utilization due to shared GPU) ...")
    import os
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.4"
    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=True)

    log("[4] Generation with structured_outputs regex ...")
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    full_prompts = []
    for p in prompts:
        full = client._backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        full_prompts.append(full)

    outputs = []
    t0 = time.time()
    for i in range(len(full_prompts)):
        structured = StructuredOutputsParams(regex=regex_patterns[i])
        sampling = SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            structured_outputs=structured,
        )
        out = client._backend.model.generate(
            [full_prompts[i]], sampling, use_tqdm=False
        )
        outputs.append(out[0])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(full_prompts) - i - 1) / rate
            log(f"  gen {i + 1}/{len(full_prompts)} ({rate:.2f} q/s, ETA {eta:.0f}s)")

    log(f"  done in {time.time() - t0:.1f}s")

    log("[5] Verify attr completeness ...")
    n_complete = 0
    n_modified_by_regex = 0
    results = []
    for i, (pair, out) in enumerate(zip(pairs, outputs)):
        text = out.outputs[0].text.strip() if out.outputs else ""
        # Check missing attrs (just for reporting, NO post-processing applied)
        attrs_str = ", ".join(f"{k}: {v}" for k, v in pair["attrs"].items() if v)
        q_low = text.lower()
        missing = []
        for kv in attrs_str.split(","):
            kv = kv.strip()
            if ":" not in kv:
                continue
            _, v = kv.split(":", 1)
            v = v.strip()
            if v and v.lower() not in q_low:
                missing.append(v)
        is_complete = len(missing) == 0
        if is_complete:
            n_complete += 1
        # vLLM guided_regex doesn't reject, it samples valid tokens only
        results.append({
            "user_id": pair["user_id"],
            "asin": pair["asin"],
            "attrs": pair["attrs"],
            "attrs_str": attrs_str,
            "regex_pattern": regex_patterns[i],
            "q_raw": text,
            "n_missing": len(missing),
            "missing_attrs": missing,
            "is_complete": is_complete,
        })

    log(f"  complete: {n_complete}/{len(results)} ({n_complete/len(results)*100:.1f}%)")

    log("[6] Write output ...")
    with OUT_GEN.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GEN}")

    log("[7] Sample outputs (first 10) ...")
    for i, r in enumerate(results[:10]):
        log(f"  [{i+1}] q: {r['q_raw'][:120]}")

    log("=" * 70)
    log("PHASE 16 TEST COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()