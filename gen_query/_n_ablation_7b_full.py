"""N ablation 7B FULL — 1619 ASINs × K=25 × N=5,7,10.

扩到全 1619 ASINs,保证每个 N 都有 ≥1000 ASINs 的 strict pool。
只跑 7B 模型(N=5/7/10),1.5B 用之前 v2 数据即可。
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
    count_attrs_covered, has_invalid_punct, n_tokens_simple,
)
from _n_ablation_pool_v2 import get_top_n_attrs

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_7b_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_7B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
N_LIST = [5, 7, 10]
K_VARIANTS = 25  # 1293 ASINs × 25 = 32k prompts (N=5)

import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_7B
sspr.MODEL_NAME = MODEL_7B


def main():
    log("=== N ABLATION 7B FULL: 1619 ASINs × K=25 × N=5,7,10 ===")
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

        log(f"  generating (7B)...")
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
            is_strict = (n_cov == N) and (not invalid)
            if is_strict:
                n_total_strict += 1
            pools[a].append({
                "k": k, "query": text, "strict": is_strict,
                "attrs_covered": n_cov, "invalid": invalid,
                "n_tok": n_tokens_simple(text),
            })

        log(f"  strict: {n_total_strict}/{len(outputs)} ({n_total_strict/len(outputs)*100:.1f}%)")

        json.dump({
            "config": {"N": N, "K": K_VARIANTS, "model": "Qwen2-7B-Instruct",
                       "n_asins_used": len(asin_attrs)},
            "pools": dict(pools),
        }, open(pool_file, "w"), ensure_ascii=False, indent=2)
        log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()