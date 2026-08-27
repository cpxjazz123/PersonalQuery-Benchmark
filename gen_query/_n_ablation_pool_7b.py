"""N ablation 7B — 用 Qwen2-7B-Instruct 重跑 N=5,7,10。

跟 v2 (1.5B) 完全对齐:
- 同样 200 ASINs, 同样 K=20 variants
- 同样 product_attributes.json + get_top_n_attrs
- 同样 N_LIST subset (5, 7, 10) 节省时间

输出: /home/wlia0047/hj82_scratch2/wenyu/n_ablation_7b/pool_N{N}.json
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
    VLLM_URL, MODEL_NAME, K_POOL, TEMP, MAX_TOKENS, log,
)
from syntax_subspace_pool_regen import (
    make_prompt, batch_generate_vllm,
    count_attrs_covered, has_invalid_punct, has_first_person, n_tokens_simple,
)
from _n_ablation_pool_v2 import get_top_n_attrs

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_7b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 7B 配置(跟 utils.py 不同,直接 inline)
MODEL_7B = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
N_LIST = [5, 7, 10]
N_ASINS = 200
K_VARIANTS = 20

# patch MODEL_NAME for 7B in BOTH utils and pool_regen (import-time binding)
import syntax_subspace_utils as ssu
import syntax_subspace_pool_regen as sspr
ssu.MODEL_NAME = MODEL_7B
sspr.MODEL_NAME = MODEL_7B


def main():
    log("=== N ABLATION 7B: Qwen2-7B-Instruct × N=5,7,10 ===")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))
    log(f"  attrs: {len(pattrs)} ASINs")

    asin_data = json.load(open(
        "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"))["asins"]

    valid_asins = []
    for e in asin_data:
        a = e["asin"]
        pa = pattrs.get(a)
        if pa is None: continue
        top10 = get_top_n_attrs(pa, 10)
        if len(top10) == 10:
            valid_asins.append(a)
    rng = random.Random(42)
    chosen = rng.sample(valid_asins, min(N_ASINS, len(valid_asins)))
    log(f"  using {len(chosen)} ASINs")

    for N in N_LIST:
        pool_file = OUT_DIR / f"pool_N{N}.json"
        if pool_file.exists():
            log(f"  N={N}: cache hit")
            continue

        log(f"\n=== N={N}: pool regen (7B) ===")
        asin_attrs = {}
        for a in chosen:
            attrs = get_top_n_attrs(pattrs[a], N)
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

        log(f"  generating (7B may be slower)...")
        t0 = time.time()
        # 7B 用 bs=256 减少 HTTP overhead
        outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
        log(f"  generated in {time.time()-t0:.1f}s")

        pools = collections.defaultdict(list)
        n_total_strict = 0
        n_lens = []
        for (a, k), out in zip(asin_idx, outputs):
            text = out.strip() if out else ""
            if not text:
                continue
            attrs = asin_attrs[a]
            n_cov = count_attrs_covered(text, attrs)
            invalid = has_invalid_punct(text)
            first_p = has_first_person(text)
            is_strict = (n_cov == N) and (not invalid) and first_p
            if is_strict:
                n_total_strict += 1
                n_lens.append(n_tokens_simple(text))
            pools[a].append({
                "k": k,
                "query": text,
                "strict": is_strict,
                "attrs_covered": n_cov,
                "invalid": invalid,
                "n_tok": n_tokens_simple(text),
            })

        log(f"  strict: {n_total_strict}/{len(outputs)} "
            f"({n_total_strict/max(1,len(outputs))*100:.1f}%)")
        if n_lens:
            log(f"  strict n_tok: mean={sum(n_lens)/len(n_lens):.1f} "
                f"median={sorted(n_lens)[len(n_lens)//2]} max={max(n_lens)}")

        json.dump({
            "config": {"N": N, "K": K_VARIANTS, "TEMP": TEMP, "MAX_TOKENS": MAX_TOKENS,
                       "model": "Qwen2-7B-Instruct",
                       "n_asins_used": len(asin_attrs)},
            "pools": dict(pools),
        }, open(pool_file, "w"), ensure_ascii=False, indent=2)
        log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()