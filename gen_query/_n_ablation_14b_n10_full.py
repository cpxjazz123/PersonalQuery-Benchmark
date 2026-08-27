"""N=10 ablation 14B FULL POOL — 从 product_attributes.json 全集取 2000 ASINs × K=20。

之前 _n_ablation_14b_full.py 用 stage8_5_asins.json 子集 1619 ASINs,
其中 N=10 只有 570 个,但 product_attributes.json 全集 N=10 有 68,719 个。
本脚本直接用全集,保证 N=10 也有 ≥1000 ASINs。
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

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_14B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2.5-14B-Instruct"
N = 10
N_ASINS = 2000  # 从 68719 ASINs 里随机抽 2000,保证 ≥1000 严格率下也够
K_VARIANTS = 20  # 2000 × 20 = 40000 prompts
SEED = 42

import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_14B
sspr.MODEL_NAME = MODEL_14B


def main():
    log(f"=== N={N} ABLATION 14B FULL POOL: {N_ASINS} ASINs × K={K_VARIANTS} ===")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))
    log(f"  total attrs ASINs: {len(pattrs)}")

    # 全集:找出所有 N=10 ASINs
    n10_asins = []
    for asin, attrs in pattrs.items():
        top10 = get_top_n_attrs(attrs, N)
        if len(top10) == N:
            n10_asins.append(asin)
    log(f"  full-pool N={N} ASINs: {len(n10_asins)}")

    rng = random.Random(SEED)
    chosen = rng.sample(n10_asins, min(N_ASINS, len(n10_asins)))
    log(f"  sampled: {len(chosen)} ASINs")

    pool_file = OUT_DIR / f"pool_N{N}.json"
    if pool_file.exists():
        log(f"  cache hit: {pool_file}")
        return

    asin_attrs = {}
    for a in chosen:
        attrs = get_top_n_attrs(pattrs[a], N)
        if len(attrs) == N:
            asin_attrs[a] = attrs
    log(f"  ASINs with {N} attrs after sampling: {len(asin_attrs)}")

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
                   "n_asins_used": len(asin_attrs), "SEED": SEED,
                   "source": "product_attributes.json full pool"},
        "pools": dict(pools),
    }, open(pool_file, "w"), ensure_ascii=False, indent=2)
    log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()