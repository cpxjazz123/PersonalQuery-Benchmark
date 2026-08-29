"""N=10 14B 增量 K=60→K=80 补足。

读现有 pool_N10.json,继续生 K=60..79 的 variants,merge 到现有 pool。
K=60 时 875 ASINs ≥10 strict,K=80 预期 950-1050 ASINs。
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "gen_query"))

from syntax_subspace_utils import (
    TEMP, MAX_TOKENS, log,
)
from syntax_subspace_pool_regen import (
    make_prompt, batch_generate_vllm,
    count_attrs_covered, has_invalid_punct, has_first_person, n_tokens_simple,
)
from _n_ablation_pool_v2 import get_top_n_attrs

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full")
MODEL_14B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2.5-14B-Instruct"
N = 10
K_START = 60  # 当前 K=0..59 已生成,补 K=60..79
K_END = 80

import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_14B
sspr.MODEL_NAME = MODEL_14B


def main():
    log(f"=== N={N} INCREMENT K={K_START}..{K_END-1} ===")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))

    pool_file = OUT_DIR / f"pool_N{N}.json"
    pool = json.load(open(pool_file))
    pools = pool["pools"]
    log(f"  existing pool: {len(pools)} ASINs")

    # 找已有 K 集合,跳过已存在的
    existing_ks = collections.defaultdict(set)
    for a, qs in pools.items():
        for q in qs:
            existing_ks[a].add(q["k"])

    asin_attrs = {}
    all_prompts = []
    asin_idx = []
    for a, qs in pools.items():
        attrs = get_top_n_attrs(pattrs.get(a, {}), N)
        if len(attrs) != N:
            continue
        asin_attrs[a] = attrs
        for k in range(K_START, K_END):
            if k in existing_ks[a]:
                continue  # 跳过已存在
            prompt = make_prompt(attrs, N, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))

    log(f"  ASINs to extend: {len(asin_attrs)}")
    log(f"  new prompts: {len(all_prompts)}")

    if len(all_prompts) == 0:
        log("  nothing to add, exiting")
        return

    log(f"  generating (14B)...")
    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  generated in {time.time()-t0:.1f}s")

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

    log(f"  new strict: {n_total_strict}/{len(outputs)} ({n_total_strict/len(outputs)*100:.1f}%)")

    # 统计新分布
    strict_counts = [sum(1 for q in qs if q["strict"]) for qs in pools.values()]
    import statistics
    log(f"  total pool: {len(pools)} ASINs, mean strict={sum(strict_counts)/len(strict_counts):.1f}, median={statistics.median(strict_counts):.0f}")
    for thresh in [1, 2, 5, 8, 10, 12, 15, 20, 30, 40]:
        n = sum(1 for c in strict_counts if c >= thresh)
        log(f"    ASINs with ≥{thresh} strict: {n}")

    json.dump({
        "config": {**pool["config"], "K_max": K_END, "incremented": True},
        "pools": dict(pools),
    }, open(pool_file, "w"), ensure_ascii=False, indent=2)
    log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()