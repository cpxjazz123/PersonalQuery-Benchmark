"""N ablation — Stage 1 pool regen + flip rate per N.

Baby Products 类目 attrs 上限 6 (select_top_attrs 验证):
  N=5: 338/1619 ASINs (20.9%)
  N=6: 72/1619 ASINs (4.4%)
  N=7+: 不可行

对比 N=4 (基线, 现有 pool.json) / N=5 / N=6 三档 flip rate。
K=25 variants per ASIN,跳 Stage 4 selection 直接 random K=8 (ablation 简化)。
"""
from __future__ import annotations

import collections
import gzip
import json
import random
import re
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
    count_attrs_covered, has_invalid_punct, n_tokens_simple,
)
from attribute_extraction.build_dataset import select_top_attrs

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LIST = [5, 6]
K_VARIANTS = 25
K_SELECT = 8  # 取 K=8 per ASIN (ablation 用 random, 跳过 Mahalanobis)
SEED = 2024
rng = random.Random(SEED)

log("loading meta...")
meta = {}
for line in gzip.open(REPO / "data/meta_Baby_Products_2023.jsonl.gz", "rt"):
    r = json.loads(line)
    a = r.get("parent_asin")
    if a and a not in meta:
        meta[a] = r
log(f"  meta: {len(meta)} ASINs")

log("loading stage8_5_asins...")
asin_data_full = json.load(open(
    "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"))["asins"]
log(f"  total asins: {len(asin_data_full)}")


def run_pool_for_n(N: int):
    pool_file = OUT_DIR / f"pool_N{N}.json"
    if pool_file.exists():
        log(f"  N={N}: cache hit {pool_file}")
        return json.load(open(pool_file))

    log(f"\n=== N={N}: pool regen ===")
    asin_attrs = {}
    for e in asin_data_full:
        a = e["asin"]
        m = meta.get(a)
        if m is None:
            continue
        attrs = select_top_attrs(m, max_n=N)
        if len(attrs) < N:
            continue
        asin_attrs[a] = attrs
    log(f"  ASINs with ≥{N} attrs: {len(asin_attrs)}")

    all_prompts = []
    asin_idx = []
    for a, attrs in asin_attrs.items():
        for k in range(K_VARIANTS):
            prompt = make_prompt(attrs, N, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))
    log(f"  total prompts: {len(all_prompts)}")

    log(f"  generating...")
    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  generated in {time.time()-t0:.1f}s")

    pools = collections.defaultdict(list)
    n_total_strict = 0
    for (a, k), out in zip(asin_idx, outputs):
        text = out.strip() if out else ""
        if not text:
            continue
        attrs = asin_attrs[a]
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        is_strict = (n_cov == N) and (not invalid)
        if is_strict:
            n_total_strict += 1
        pools[a].append({
            "k": k,
            "query": text,
            "strict": is_strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "n_tok": n_tokens_simple(text),
        })

    log(f"  total queries: {len(outputs)}, strict: {n_total_strict} "
        f"({n_total_strict/max(1,len(outputs))*100:.1f}%)")

    json.dump({
        "config": {"N": N, "K": K_VARIANTS, "TEMP": TEMP, "MAX_TOKENS": MAX_TOKENS,
                   "n_asins_total": len(asin_data_full), "n_asins_used": len(asin_attrs)},
        "pools": dict(pools),
    }, open(pool_file, "w"), ensure_ascii=False, indent=2)
    log(f"  wrote → {pool_file}")
    return json.load(open(pool_file))


# N=4 baseline: 复用现有 pool.json (1619 ASINs × 50 variants, 但严格按 asin 的 attrs_used)
def load_n4_baseline():
    """复用现有 pool.json 1619 ASINs × 50 queries, 重新按 N=4 attrs_used 评估 strict."""
    log("\n=== N=4: baseline from existing pool.json ===")
    pool = json.load(open(REPO / "result/gen_query/pool.json"))
    pools_raw = pool["pools"]
    n_strict_total = 0
    out_pools = {}
    for asin, qs in pools_raw.items():
        # 取 N=4 attrs from stage8_5_asins.json
        e = next((x for x in asin_data_full if x["asin"] == asin), None)
        if e is None:
            continue
        attrs = e["attrs_used"]
        new_qs = []
        for q in qs:
            text = q["query"]
            n_cov = count_attrs_covered(text, attrs)
            invalid = has_invalid_punct(text)
            is_strict = (n_cov == 4) and (not invalid)
            if is_strict:
                n_strict_total += 1
            new_qs.append({
                "k": q["k"],
                "query": text,
                "strict": is_strict,
                "attrs_covered": n_cov,
                "invalid": invalid,
                "n_tok": n_tokens_simple(text),
            })
        out_pools[asin] = new_qs
    log(f"  total queries: {len(pools_raw)*50}, strict: {n_strict_total}")
    return {"config": {"N": 4}, "pools": out_pools}


def main():
    log("=== N ABLATION: N=4 baseline + N=5 + N=6 ===")
    # 先存 N=4 baseline (复用现有数据)
    n4_data = load_n4_baseline()
    n4_file = OUT_DIR / "pool_N4.json"
    json.dump(n4_data, open(n4_file, "w"), ensure_ascii=False, indent=2)
    log(f"  wrote → {n4_file}")

    # 跑 N=5, N=6
    for N in N_LIST:
        run_pool_for_n(N)

    log("\n=== Stage 1 done. Stage 5 (retrieval + flip) separately. ===")


if __name__ == "__main__":
    main()