"""N ablation v2 — 用 product_attributes.json (extract_attrs 已结构化)。

v1 用 select_top_attrs(max_n=N) 从 raw meta 抽 → N=5+ 退化 (empty list 污染)。
v2 用 result/product_attributes.json 的真实 attrs 分布:
  平均 13.6 attrs/ASIN, 89.5% ASINs 有 ≥7 attrs。

跑 N=5,6,7,8,9,10 (Baby Products 类目上限是 10+)。
每个 ASIN 对所有 N 都有效 → 同 ASIN 子集对比,无 ASIN selection bias。
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
    count_attrs_covered, has_invalid_punct, n_tokens_simple,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/n_ablation_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LIST = [5, 6, 7, 8, 9, 10]
N_ASINS = 200  # 抽样 200 ASINs (每个 ASIN 对所有 N 都用)
K_VARIANTS = 20  # 200 ASINs × 6 N × 20 = 24k prompts,~5min/N


def get_top_n_attrs(attrs: dict, n: int) -> dict:
    """从 product_attributes 选 top-n 真实非空 attrs。

    排除: empty string/list/dict, numeric values (避免 Stage 1 之前 EXCLUDE_NUMERIC_ATTRS 的策略)。
    优先: brand / category / color / material / style / size 这种语义字段。
    """
    # 排除项
    SKIP_KEYS = {
        "Average Rating", "Price", "Rating Number", "Item model number",
        "Batteries required", "Is Discontinued By Manufacturer",
        "Date First Available", "Item Weight", "Maximum weight recommendation",
        "Minimum weight recommendation", "Package Dimensions", "Product Dimensions",
        "Number Of Items", "ASIN", "UPC",
    }
    NUMERIC_KW = {"price", "average rating", "rating number", "item weight",
                  "item model number", "date first available",
                  "package dimensions", "product dimensions",
                  "minimum weight recommendation", "maximum weight recommendation"}

    def is_real(k, v):
        if v is None: return False
        s = str(v).strip()
        if s in ('', '[]', '{}', 'None'): return False
        if any(nk in k.lower() for nk in NUMERIC_KW): return False
        if k in SKIP_KEYS: return False
        return True

    real = [(k, v) for k, v in attrs.items() if is_real(k, v)]
    # 优先级: 视觉/语义先,其他后
    PREF = ["Brand", "Color", "Material", "Material Type", "Style",
            "Fabric Type", "Frame Material", "Pattern", "Theme", "Size",
            "Age Range (Description)", "Target gender", "Special Feature",
            "Main Category", "Manufacturer", "Harness type", "Form Factor",
            "Item Weight", "Shape"]
    def keyfn(item):
        k = item[0]
        try:
            return (PREF.index(k), k)
        except ValueError:
            return (len(PREF), k)
    real.sort(key=keyfn)
    return dict(real[:n])


def main():
    log("=== N ABLATION v2: product_attributes.json × N=5..10 ===")
    log("loading product_attributes.json...")
    pattrs = json.load(open(REPO / "result/product_attributes.json"))
    log(f"  {len(pattrs)} ASINs")

    # 选 200 ASINs:要求所有 N=5..10 都有足够 attrs
    asin_data = json.load(open(
        "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"))["asins"]
    valid_asins = []
    for e in asin_data:
        a = e["asin"]
        pa = pattrs.get(a)
        if pa is None: continue
        # 要求 top-10 attrs 都存在
        top10 = get_top_n_attrs(pa, 10)
        if len(top10) == 10:
            valid_asins.append(a)
    log(f"  ASINs with top-10 real attrs: {len(valid_asins)}")
    rng = random.Random(42)
    chosen = rng.sample(valid_asins, min(N_ASINS, len(valid_asins)))
    log(f"  using {len(chosen)} ASINs")

    # 对每个 N 跑 LLM 生成 + strict filter
    for N in N_LIST:
        pool_file = OUT_DIR / f"pool_N{N}.json"
        if pool_file.exists():
            log(f"  N={N}: cache hit")
            continue

        log(f"\n=== N={N}: pool regen ===")
        # 取 top-N attrs per ASIN
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

        log(f"  generating...")
        t0 = time.time()
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
            is_strict = (n_cov == N) and (not invalid)
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
                       "n_asins_used": len(asin_attrs)},
            "pools": dict(pools),
        }, open(pool_file, "w"), ensure_ascii=False, indent=2)
        log(f"  wrote → {pool_file}")


if __name__ == "__main__":
    main()