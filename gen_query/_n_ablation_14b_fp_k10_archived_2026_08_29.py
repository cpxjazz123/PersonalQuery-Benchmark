"""N ablation 14B FIRST-PERSON, K=10 — 覆盖现有 3rd-person pools。

用户指令 2026-08-27: 立即重新生成 N=5/7/10 first-person,K=10。
- N=5/7: 1619 ASINs (stage8_5_asins 子集) × K=10
- N=10: 2000 ASINs (product_attributes.json 全集随机抽样) × K=10

输出覆盖现有 pool_N5/7/10.json (3rd-person pool 已备份到
n_ablation_14b_first_person/pool_N*_3rdperson.json)。
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

SUB_POOL_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_full")
N10_POOL_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_14b_n10_full")
MODEL_14B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2.5-14B-Instruct"

N_ASINS_FULL = 2000  # N=10 抽样数
K_VARIANTS = 10
SEED = 42

import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_14B
sspr.MODEL_NAME = MODEL_14B


def generate_for_n(N: int, asin_attrs: dict, out_dir: Path, source: str):
    """Generate K=10 queries per ASIN, write to out_dir/pool_N{N}.json."""
    pool_file = out_dir / f"pool_N{N}.json"
    log(f"\n=== N={N} ({source}): {len(asin_attrs)} ASINs × K={K_VARIANTS} ===")

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
    n_total_first_person = 0
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
        if first_p:
            n_total_first_person += 1
        pools[a].append({
            "k": k, "query": text, "strict": is_strict,
            "attrs_covered": n_cov, "invalid": invalid,
            "first_person": first_p,
            "n_tok": n_tokens_simple(text),
        })

    log(f"  first-person: {n_total_first_person}/{len(outputs)} ({n_total_first_person/len(outputs)*100:.1f}%)")
    log(f"  strict (attrs + valid + 1st person): {n_total_strict}/{len(outputs)} ({n_total_strict/len(outputs)*100:.1f}%)")

    # ASIN-level distribution
    strict_counts = [sum(1 for q in qs if q["strict"]) for qs in pools.values()]
    log(f"  ASINs with ≥10 strict: {sum(1 for c in strict_counts if c >= 10)}")
    log(f"  ASINs with ≥5 strict: {sum(1 for c in strict_counts if c >= 5)}")
    log(f"  ASINs with ≥1 strict: {sum(1 for c in strict_counts if c >= 1)}")

    json.dump({
        "config": {"N": N, "K": K_VARIANTS, "model": "Qwen2.5-14B-Instruct",
                   "n_asins_used": len(asin_attrs), "source": source,
                   "perspective": "first-person"},
        "pools": dict(pools),
    }, open(pool_file, "w"), ensure_ascii=False, indent=2)
    log(f"  wrote → {pool_file}")


def main():
    log(f"=== N ABLATION 14B FIRST-PERSON, K={K_VARIANTS} ===")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))
    asin_data = json.load(open(
        "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"))["asins"]
    log(f"  total stage8_5 asins: {len(asin_data)}")

    # === N=5, 7: sub-set (stage8_5_asins) ===
    for N in [5, 7]:
        asin_attrs = {}
        for e in asin_data:
            a = e["asin"]
            pa = pattrs.get(a)
            if pa is None: continue
            attrs = get_top_n_attrs(pa, N)
            if len(attrs) == N:
                asin_attrs[a] = attrs
        log(f"  N={N}: stage8_5 sub-set with {N} attrs: {len(asin_attrs)}")
        generate_for_n(N, asin_attrs, SUB_POOL_DIR, source="stage8_5 sub-set")

    # === N=10: full-pool (product_attributes.json) ===
    n10_asins = []
    for asin, attrs in pattrs.items():
        top10 = get_top_n_attrs(attrs, 10)
        if len(top10) == 10:
            n10_asins.append(asin)
    log(f"  N=10: full-pool ASINs: {len(n10_asins)}")

    rng = random.Random(SEED)
    chosen = rng.sample(n10_asins, min(N_ASINS_FULL, len(n10_asins)))
    asin_attrs = {a: get_top_n_attrs(pattrs[a], 10) for a in chosen}
    asin_attrs = {a: attrs for a, attrs in asin_attrs.items() if len(attrs) == 10}
    log(f"  N=10: sampled {len(asin_attrs)} ASINs")
    generate_for_n(10, asin_attrs, N10_POOL_DIR, source="product_attributes.json full pool")


if __name__ == "__main__":
    main()