#!/usr/bin/env python3
"""VLLM batch test: 100 prompts in one call (验证直接 batch 路径)."""
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

import phase10_12_a_syntactic_diversity as mod

PAIRS_FILE = mod.PAIRS_FILE

pairs = []
with PAIRS_FILE.open() as f:
    for line in f:
        pairs.append(json.loads(line))

valid_pairs = []
for p in pairs[:50]:  # 50 pairs
    attrs = p.get("attrs_5", {}) or p.get("attrs", {})
    if all(k in attrs and attrs[k] for k in ["Brand", "Color", "Material"]):
        valid_pairs.append(p)

print(f"valid_pairs: {len(valid_pairs)}")
total = len(valid_pairs) * len(mod.SYNTAX_CONDITIONS) * mod.N_SEEDS
print(f"total: {total}")

from llm_client import create_qwen_local_client
client = create_qwen_local_client()

# Build prompts
all_prompts = []
for pair_idx, p in enumerate(valid_pairs):
    attrs = p.get("attrs_5") or p.get("attrs", {})
    for cond_idx, cond in enumerate(mod.SYNTAX_CONDITIONS):
        for seed_idx in range(mod.N_SEEDS):
            prompt = mod.build_prompt(attrs, attrs, cond)
            all_prompts.append((pair_idx, cond_idx, seed_idx, prompt))

prompts_only = [p[3] for p in all_prompts]

from vllm import SamplingParams
sampling = SamplingParams(
    max_tokens=120,
    temperature=0.9,
    top_p=0.95,
    stop=["\n\n", "Product attributes:", "<|im_end|>"],
)

backend = client._backend
print(f"\nCalling model.generate({total} prompts) ...")
t0 = time.time()
outputs = backend.model.generate(prompts_only, sampling)
elapsed = time.time() - t0
print(f"Generated {len(outputs)} in {elapsed:.1f}s, rate={len(outputs)/elapsed:.1f}/s")

# Quality check: how many mention all 3 attrs?
n_attr_pass = 0
n_with_append = 0
for i, out in enumerate(outputs):
    pair_idx = all_prompts[i][0]
    pair = valid_pairs[pair_idx]
    attrs = pair.get("attrs_5") or pair.get("attrs", {})
    txt = out.outputs[0].text.strip() if out.outputs else ""
    txt_lower = txt.lower()
    has_brand = str(attrs.get("Brand", "")).lower() in txt_lower
    has_color = str(attrs.get("Color", "")).lower() in txt_lower
    has_material = str(attrs.get("Material", "")).lower() in txt_lower
    if has_brand and has_color and has_material:
        n_attr_pass += 1
    else:
        n_with_append += 1

print(f"\nQuality check (hard-copy before):")
print(f"  attr_pass (all 3 attrs in raw output): {n_attr_pass}/{len(outputs)} = {100*n_attr_pass/len(outputs):.1f}%")
print(f"  need hard-copy append: {n_with_append}/{len(outputs)}")

# 估算 full scale
print(f"\nFull scale 21024 prompts at {len(outputs)/elapsed:.1f}/s would take {21024/(len(outputs)/elapsed):.0f}s = {21024/(len(outputs)/elapsed)/60:.1f}min")

print()
print("=== Random 10 outputs ===")
import random
random.seed(0)
for i in random.sample(range(len(outputs)), min(10, len(outputs))):
    txt = outputs[i].outputs[0].text.strip() if outputs[i].outputs else "(empty)"
    pair_idx = all_prompts[i][0]
    cond_idx = all_prompts[i][1]
    print(f"  [{i}] cond={cond_idx}: {txt[:180]}")