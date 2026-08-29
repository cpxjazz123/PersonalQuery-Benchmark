"""N ablation 14B FULL — 1619 ASINs × K=25 × N=5,7,10 with Qwen2.5-14B-Instruct.

跟 _n_ablation_7b_full.py 完全一致,只把 MODEL_7B 改成 14B 路径。
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "gen_query"))

from syntax_subspace_utils import (
    VLLM_URL, K_POOL, TEMP, MAX_TOKENS, log,
)
from syntax_subspace_pool_regen import (
    make_prompt, batch_generate_vllm,
    count_attrs_covered, has_invalid_punct, has_first_person, n_tokens_simple,
)
from _n_ablation_pool_v2 import get_top_n_attrs

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_14B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2.5-14B-Instruct"
N_LIST = [5, 7, 10]
K_VARIANTS = 25

import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_14B
sspr.MODEL_NAME = MODEL_14B


def main():
    log("=== N ABLATION 14B FULL: 1619 ASINs × K=25 × N=5,7,10 ===")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))
    asin_data = json.load(open(
        "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"))["asins"]
    log(f"  total asins: {len(asin_data)}")

    for N in N_LIST:
        pool_file = OUT_DIR / f"pool_N{N}.json"
        if pool_file.exists():
            log(f"  N={N}: cache hit")
            continue

        log(f"\n=== N={N} ===")
        asin_attrs = {}
        for e in asin_data:
            a = e["asin"]
            pa = pattrs.get(a)
            if pa is None: continue
            attrs = get_top_n_attrs(pa, N)
            if len(attrs) == N:
                asin_attrs[a] = attrs
        log(f"  ASINs with {N} attrs: {len(asin_attrs)}")

        all_prompts = []
        asin_idx = []
        for a, attrs in asin_attrs.items():
            for k in range(K_VARIANTS):
                prompt = make_prompt(attrs, N, k=k)
                all_prompts.append(prompt)
                asin_idx.append((a, k))
        log(f"  total prompts: {len(all_prompts)}")

        log(f"  generating (14B)...")
        t0 = time.time()
        outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
        log(f"  generated in {time.time()-t0:.1f}s")

        pools = collections.defaultdict(list)
        n_total_strict = 0
        for (a, k), out in zip(asin_idx, outputs):
            text = out.strip() if out else ""
            if not text: continue
            attrs = asin_attrs[a]
            n_cov = count_attrs_covered(text, attrs)
            invalid = has_invalid_punct(text)
            first_p = has_first_person(text)
            is_strict = (n_cov == N) and (not invalid) and first_p
            if is_strict:
                n_total_strict += 1
            pools[a].append({
                "k": k, "query": text, "strict": is_strict,
                "attrs_covered": n_cov, "invalid": invalid,
                "n_tok": n_tokens_simple(text),
            })

        log(f"  strict: {n_total_strict}/{len(outputs)} ({n_total_strict/len(outputs)*100:.1f}%)")

        json.dump({
            "config": {"N": N, "K": K_VARIANTS, "model": "Qwen2.5-14B-Instruct",
                       "n_asins_used": len(asin_attrs)},
            "pools": dict(pools),
        }, open(pool_file, "w"), ensure_ascii=False, indent=2)
        log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()