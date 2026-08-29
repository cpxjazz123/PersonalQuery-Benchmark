#!/usr/bin/env python3
"""Stage 1 — 扩展 pool 到全 4789 ASINs。

用户指令 2026-08-29: 取消"只评估有 K=200 pool 的 ASIN"限制,
为剩余 4055 个 ASINs 生成 K=50 pool (用 K=50 节省 vLLM 调用,
200 → 50, 811K → 202K calls), 合并到现有 pool_K200_F3pca48.json,
新建全量 pool_K200_F3pca48_full.json (734 老 ASINs K=200 + 4055 新 ASINs K=50)。

I/O:
  输入: stage8_5_asins.json (4789 ASINs)
        pool_K200_F3pca48.json (736 ASINs × K=200, 部分 ASINs K<200)
        result/product_attributes.json
  输出: pool_K200_F3pca48_full.json (4789 ASINs, 老 K=200 + 新 K=50)

运行: python gen_query/syntax_subspace_pool_expand_4789.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RESULT = REPO_ROOT / "result"

# === Inputs ===
ASINS_IN = SCRATCH / "stage8_5_asins.json"
PRODUCT_ATTRS = RESULT / "product_attributes.json"
EXISTING_POOL = SCRATCH / "pool_K200_F3pca48.json"

# === Output ===
FULL_POOL = SCRATCH / "pool_K200_F3pca48_full.json"

# === Stage 1 配置 ===
K_NEW = 50  # 新 ASINs 用 K=50
N_INPUT = 5  # 与 Stage 1 一致
TEMP = 0.5
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60

VLLM_URL = None  # use syntax_subspace_utils.VLLM_URL (完整 endpoint "/v1/completions")
MODEL_NAME = None  # use syntax_subspace_utils.MODEL_NAME


def log(msg: str) -> None:
    print(f"[pool_expand] {msg}", flush=True)


# 复用 stage_pool_regen 的 filter functions
sys.path.insert(0, str(REPO_ROOT / "gen_query"))
from syntax_subspace_pool_regen import (  # noqa: E402
    count_attrs_covered, has_invalid_punct, has_first_person,
    has_emoji, has_self_talk, is_query_too_long, n_tokens_simple,
    make_prompt, batch_generate_vllm,
)
from _n_ablation_pool_v2 import get_top_n_attrs  # noqa: E402


def main() -> None:
    log("=== Stage 1 EXPAND — pool to all 4789 ASINs ===")

    # 1. Load cohort + existing pool
    asin_data = json.load(open(ASINS_IN))["asins"]
    cohort_asins = {a["asin"] for a in asin_data}
    log(f"  cohort ASINs: {len(cohort_asins)}")

    existing = json.load(open(EXISTING_POOL))
    existing_pools: dict[str, list] = existing["pools"]
    log(f"  existing pool: {len(existing_pools)} ASINs × "
        f"{existing['n_total_strict']} strict total")

    new_asins = sorted(cohort_asins - set(existing_pools.keys()))
    log(f"  new ASINs to generate: {len(new_asins)}")

    # 2. Build prompts for new ASINs only
    pattrs = json.load(open(PRODUCT_ATTRS))
    asin_attrs = {}
    n_no_attrs = n_lt_n = 0
    for a in new_asins:
        pa = pattrs.get(a)
        if pa is None:
            n_no_attrs += 1
            continue
        top_n = get_top_n_attrs(pa, N_INPUT)
        if len(top_n) < N_INPUT:
            n_lt_n += 1
            continue
        asin_attrs[a] = top_n
    log(f"  new ASINs with ≥{N_INPUT} non-numeric attrs: {len(asin_attrs)} "
        f"  (filtered: no_attrs={n_no_attrs}, lt_{N_INPUT}={n_lt_n})")

    all_prompts = []
    asin_idx = []
    for a, attrs in asin_attrs.items():
        for k in range(K_NEW):
            prompt = make_prompt(attrs, N_INPUT, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))

    log(f"  generating {len(all_prompts)} queries via vLLM (K={K_NEW} × "
        f"{len(asin_attrs)} ASINs)...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    # 3. Apply strict filter (5-layer)
    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0
    n_cov_full = n_invalid = n_first_p = n_emoji = n_self_talk = n_too_long = 0

    for (a, k), out in zip(asin_idx, all_outputs):
        attrs = asin_attrs[a]
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        is_strict = (
            (n_cov == N_INPUT) and (not invalid) and first_p
            and (not emoji) and (not self_talk) and (not too_long)
        )
        if n_cov == N_INPUT: n_cov_full += 1
        if not invalid: n_invalid += 1
        if first_p: n_first_p += 1
        if not emoji: n_emoji += 1
        if not self_talk: n_self_talk += 1
        if not too_long: n_too_long += 1
        if is_strict:
            n_total_strict += 1
            strict_counts[a] = strict_counts.get(a, 0) + 1
        pools[a].append({
            "k": k,
            "query": text,
            "strict": is_strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "first_person": first_p,
            "emoji": emoji,
            "self_talk": self_talk,
            "too_long": too_long,
            "n_tok": n_tokens_simple(text),
        })

    log(f"  new pool: {len(pools)} ASINs × {n_total_strict} strict")

    # 4. Merge: old (K=200) + new (K=50)
    merged = dict(existing_pools)
    for a, cands in pools.items():
        merged[a] = cands
    n_old_strict = sum(1 for cs in existing_pools.values() for c in cs if c.get("strict"))
    log(f"  merged: {len(merged)} ASINs "
        f"(old: {len(existing_pools)} × K~200 + "
        f"new: {len(pools)} × K={K_NEW})")
    log(f"  total strict (old + new): {n_old_strict} + {n_total_strict} = "
        f"{n_old_strict + n_total_strict}")

    # 5. Save
    out = {
        "config": {
            "description": (f"Full pool: old 734 ASINs × K~200 + "
                           f"new {len(pools)} ASINs × K={K_NEW}; "
                           f"Stage 1 vLLM K={K_NEW}, N_INPUT={N_INPUT}, "
                           f"TEMP={TEMP}, MAX_TOKENS={MAX_TOKENS}, "
                           f"5-layer strict filter"),
            "K_NEW": K_NEW,
            "N_INPUT": N_INPUT,
            "TEMP": TEMP,
            "MAX_TOKENS": MAX_TOKENS,
            "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
        },
        "n_asins": len(merged),
        "n_old_asins": len(existing_pools),
        "n_new_asins": len(pools),
        "n_total_strict": n_old_strict + n_total_strict,
        "pools": merged,
    }
    FULL_POOL.parent.mkdir(parents=True, exist_ok=True)
    with open(FULL_POOL, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote → {FULL_POOL}")


if __name__ == "__main__":
    main()