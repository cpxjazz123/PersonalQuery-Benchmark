#!/usr/bin/env python3
"""Corrected Baseline — 50 ASINs PILOT (Apple-to-apple with prototype pilot).

用户指令 2026-08-30: prototype pilot 修了 build_user_content 的 attrs bug 之后,
cov_full(5/5) 从 0% → 48.2%. 担心旧 baseline pool 的 attrs 处理路径可能不同,
导致 prototype vs baseline 的对比不公平。

本脚本: 用与 prototype pilot 完全相同的 prompt 模板 (attrs 注入、5-layer
filter、first-person、length cap), 只去掉 prototype guidance 行, K=200 / ASIN,
但只跑 50 ASINs pilot (同 prototype pilot 的种子 + 抽样, 保证 ASIN list 一致),
与 prototype pilot 做 Apple-to-apple 比较。

**输入**: product_attributes.json, stage8_5_asins_strict34_intersect2174_ksweep_K200.json
**输出**: prototype_pool_K16_baseline_corrected.json

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# Import helpers from prototype pilot (which now has the FIXED build_user_content).
_PROTO = REPO_ROOT / "gen_query" / "syntax_subspace_prototype_pool_regen.py"
_spec = importlib.util.spec_from_file_location("proto_regen", _PROTO)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["proto_regen"] = _mod
_spec.loader.exec_module(_mod)

batch_generate_vllm = _mod.batch_generate_vllm
get_top_n_attrs = _mod.get_top_n_attrs
log = _mod.log
build_user_content = _mod.build_user_content

PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
ASINS_IN = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"

# Same pilot params as prototype pilot (apple-to-apple)
PILOT_N_ASIN = 50
RANDOM_SEED = 42
K_POOL_PER_ASIN = 200
N_INPUT = 5
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60

OUT_POOL = SCRATCH / "pool_K200_corrected_attrs_pilot.json"

# Prompt template (same as prototype pilot BUT with prototype guidance removed)
GEN_SYSTEM_TMPL_CORRECTED = (
    "You are a real customer searching for a product. "
    "Write a single natural search query as if you are a shopper looking for "
    "this exact product. The query MUST mention ALL of these attributes:\n"
    "{ATTRIBUTES}\n\n"
    "Rules:\n"
    "- First-person voice: use 'I', 'my', 'I'm looking for', 'I want', etc.\n"
    "- One natural sentence (NOT a bulleted list).\n"
    "- Do NOT start with 'Here:', 'Brand:', 'Attribute:' or any prefix.\n"
    "- Do NOT use emoji.\n"
    "- Do NOT talk to the AI (no 'can someone help', no 'thanks!').\n"
    "- Maximum 60 tokens.\n"
    "Write only the query, no commentary."
)


def make_corrected_prompt(attrs: dict, k: int) -> str:
    """Same attrs-corrected prompt as prototype pilot but NO prototype guidance."""
    # Use prototype regen's (now fixed) build_user_content
    return GEN_SYSTEM_TMPL_CORRECTED.format(
        ATTRIBUTES=build_user_content(attrs)
    )


def main():
    log("=== Corrected Baseline PILOT — 50 ASINs, K=200/ASIN, no prototype ===")

    pattrs = json.load(open(PATTRS_PATH))
    asin_data = json.load(open(ASINS_IN))["asins"]

    # Same seeded sample as prototype pilot
    rng = random.Random(RANDOM_SEED)
    asin_data_sorted = sorted(asin_data, key=lambda x: x["asin"])
    rng.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    log(f"  ASINs in cohort: {len(asin_data)}, pilot: {len(asin_data_pilot)} (seed={RANDOM_SEED})")

    asin_attrs: dict[str, dict] = {}
    n_no_attrs = n_lt_n = 0
    for entry in asin_data_pilot:
        a = entry["asin"]
        pa = pattrs.get(a)
        if pa is None:
            n_no_attrs += 1
            continue
        top_n = get_top_n_attrs(pa, N_INPUT)
        if len(top_n) < N_INPUT:
            n_lt_n += 1
            continue
        asin_attrs[a] = top_n
    log(f"  ASINs with ≥{N_INPUT} non-numeric attrs: {len(asin_attrs)}")

    # Build K=200 prompts per ASIN
    all_prompts = []
    asin_k = []
    for a, attrs_dict in asin_attrs.items():
        for k in range(K_POOL_PER_ASIN):
            all_prompts.append(make_corrected_prompt(attrs_dict, k))
            asin_k.append((a, k))
    n_prompts = len(all_prompts)
    log(f"  total prompts: {n_prompts} = {len(asin_attrs)} ASINs × {K_POOL_PER_ASIN}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    # Apply 5-layer strict filter (same helpers as prototype pilot)
    count_attrs_covered = _mod.count_attrs_covered
    has_invalid_punct = _mod.has_invalid_punct
    has_first_person = _mod.has_first_person
    has_emoji = _mod.has_emoji
    has_self_talk = _mod.has_self_talk
    is_query_too_long = _mod.is_query_too_long
    n_tokens_simple = _mod.n_tokens_simple

    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full": 0, "invalid": 0, "first_p": 0, "emoji": 0, "self_talk": 0, "too_long": 0}

    for (a, k), out in zip(asin_k, outputs):
        attrs_dict = asin_attrs[a]
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs_dict)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        strict = (n_cov == N_INPUT) and (not invalid) and first_p and (not emoji) and (not self_talk) and (not too_long)
        if n_cov == N_INPUT:
            filter_counts["cov_full"] += 1
        if invalid:
            filter_counts["invalid"] += 1
        if first_p:
            filter_counts["first_p"] += 1
        if emoji:
            filter_counts["emoji"] += 1
        if self_talk:
            filter_counts["self_talk"] += 1
        if too_long:
            filter_counts["too_long"] += 1
        if strict:
            n_total_strict += 1
        pools[a].append({
            "k": k, "query": text, "strict": strict,
            "attrs_covered": n_cov, "invalid": invalid,
            "first_person": first_p, "emoji": emoji,
            "self_talk": self_talk, "too_long": too_long,
            "n_tok": n_tokens_simple(text),
        })

    log(f"  filter breakdown: {filter_counts}")
    log(f"  total strict: {n_total_strict}")
    log(f"  mean queries per ASIN: {sum(len(v) for v in pools.values())/max(1,len(pools)):.1f}")

    out_doc = {
        "config": {
            "description": "CORRECTED baseline pilot: same attrs-corrected prompt as prototype pilot, NO prototype guidance, K=200/ASIN, T=0.7, MAX_TOKENS=120",
            "K_POOL_PER_ASIN": K_POOL_PER_ASIN,
            "PILOT_N_ASIN": PILOT_N_ASIN,
            "RANDOM_SEED": RANDOM_SEED,
            "N_INPUT": N_INPUT,
            "TEMP": TEMP,
            "MAX_TOKENS": MAX_TOKENS,
            "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
        },
        "n_asins": len(asin_attrs),
        "n_total_strict": n_total_strict,
        "filter_counts": filter_counts,
        "pools": pools,
    }
    with open(OUT_POOL, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {OUT_POOL}")


if __name__ == "__main__":
    main()