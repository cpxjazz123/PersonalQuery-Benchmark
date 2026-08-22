#!/usr/bin/env python3
"""Concurrency smoke test: 200 candidates with 16 workers."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
for p in pairs[:50]:  # 50 pairs only
    attrs = p.get("attrs_5", {}) or p.get("attrs", {})
    if all(k in attrs and attrs[k] for k in ["Brand", "Color", "Material"]):
        valid_pairs.append(p)

print(f"valid_pairs: {len(valid_pairs)}")
print(f"SYNTAX_CONDITIONS: {len(mod.SYNTAX_CONDITIONS)}")
total = len(valid_pairs) * len(mod.SYNTAX_CONDITIONS) * mod.N_SEEDS
print(f"total: {total}")

from llm_client import create_qwen_local_client
client = create_qwen_local_client()

# Build all prompts
all_prompts = []
for pair_idx, p in enumerate(valid_pairs):
    attrs = p.get("attrs_5", {})
    for cond_idx, cond in enumerate(mod.SYNTAX_CONDITIONS):
        for seed_idx in range(mod.N_SEEDS):
            prompt = mod.build_prompt(attrs, attrs, cond)
            all_prompts.append((pair_idx, cond_idx, seed_idx, prompt))

# Generate concurrent
N_WORKERS = 16
print(f"\nGenerating {total} candidates with {N_WORKERS} workers ...")
t0 = time.time()
results = [""] * total

def gen_one(idx_pair):
    idx, (pair_idx, cond_idx, seed_idx, prompt) = idx_pair
    try:
        return idx, client.call(prompt, max_tokens=120, temperature=0.9).strip()
    except Exception as e:
        return idx, ""

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(gen_one, (i, p)) for i, p in enumerate(all_prompts)]
    done = 0
    for fut in as_completed(futures):
        idx, result = fut.result()
        results[idx] = result
        done += 1

elapsed = time.time() - t0
rate = total / elapsed
print(f"Generated {total} in {elapsed:.1f}s, rate={rate:.1f}/s")
print(f"  -> full {21024/rate:.0f}s = {21024/rate/60:.1f}min for full scale")
print()
print("=== Sample outputs (first 5) ===")
for i in range(5):
    print(f"  [{i}] {results[i][:200]}")