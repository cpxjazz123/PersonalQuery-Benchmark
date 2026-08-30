#!/usr/bin/env python3
"""Syntax-Prototype Guided Pool Regen — PILOT (50 ASINs).

用户指令 2026-08-30: 大规模 139K prompts 太慢. 先用 50 ASINs pilot 看效果,
确认 prototype guidance 在小规模上 F3%/min_d_self 是否真的改善再决定是否全量。

逻辑与 syntax_subspace_prototype_pool_regen.py 完全一致, 但:
  - 只取前 50 ASINs (随机 seed=42, 保证 reproducible)
  - 输出文件名加 _pilot 后缀, 不覆盖主 pool

**输入**:
  - syntax_prototypes_K16.json
  - product_attributes.json
  - stage8_5_asins_strict34_intersect2174_ksweep_K200.json (ASIN list)

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/prototype_pool_K16_pilot.json
  - 自动 fork Phase 3 audit (gaussian/build_strict34_prototype_audit_pilot.py)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

# Reuse the full-scale module's helpers (5-layer filter, prompt building,
# vLLM client) so pilot logic stays consistent with full-scale run.
_FULL = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/gen_query/syntax_subspace_prototype_pool_regen.py")
_spec = importlib.util.spec_from_file_location("proto_full", _FULL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["proto_full"] = _mod
_spec.loader.exec_module(_mod)

# Pull helpers from the full-scale module
batch_generate_vllm = _mod.batch_generate_vllm
make_prototype_prompt = _mod.make_prototype_prompt
get_top_n_attrs = _mod.get_top_n_attrs
log = _mod.log
PROTOTYPES_PATHS = _mod.PROTOTYPES_PATHS
ASINS_IN = _mod.ASINS_IN
PATTRS_PATH = _mod.PATTRS_PATH
N_INPUT = _mod.N_INPUT
K_PER_PROTO = _mod.K_PER_PROTO
SCRATCH = _mod.SCRATCH

# Pilot-specific
PILOT_N_ASIN = 50
RANDOM_SEED = 42
PILOT_OUT = SCRATCH / "prototype_pool_K16_pilot.json"
SUMMARY_OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/prototype_pool_pilot_summary.json")


def main():
    log("=== Phase 2 PILOT — 50 ASINs × 16 prototypes × 4 ===")

    with open(PROTOTYPES_PATHS[16], "r", encoding="utf-8") as f:
        proto_data = json.load(f)
    prototypes = proto_data["prototypes"]
    log(f"  loaded {len(prototypes)} prototypes")

    with open(ASINS_IN, "r", encoding="utf-8") as f:
        asin_data = json.load(f)["asins"]

    pattrs = json.load(open(PATTRS_PATH))

    rng = random.Random(RANDOM_SEED)
    asin_data_sorted = sorted(asin_data, key=lambda x: x["asin"])  # canonical order
    rng.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    log(f"  total ASINs in cohort: {len(asin_data)}, pilot: {len(asin_data_pilot)} (seed={RANDOM_SEED})")

    n_no_attrs = n_lt_n = 0
    asin_attrs: dict[str, dict] = {}
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

    all_prompts = []
    asin_proto_k = []
    for a, attrs_dict in asin_attrs.items():
        for proto in prototypes:
            for k in range(K_PER_PROTO):
                prompt = make_prototype_prompt(attrs_dict, proto, k)
                all_prompts.append(prompt)
                asin_proto_k.append((a, proto["cluster_id"], k))
    log(f"  total prompts: {len(all_prompts)} = {len(asin_attrs)} ASINs × {len(prototypes)} × {K_PER_PROTO}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full": 0, "invalid": 0, "first_p": 0, "emoji": 0, "self_talk": 0, "too_long": 0}

    count_attrs_covered = _mod.count_attrs_covered
    has_invalid_punct = _mod.has_invalid_punct
    has_first_person = _mod.has_first_person
    has_emoji = _mod.has_emoji
    has_self_talk = _mod.has_self_talk
    is_query_too_long = _mod.is_query_too_long
    n_tokens_simple = _mod.n_tokens_simple

    for (a, proto_id, k), out in zip(asin_proto_k, outputs):
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
            "n_tok": n_tokens_simple(text), "prototype_id": proto_id,
        })

    log(f"  filter breakdown: {filter_counts}")
    log(f"  total strict: {n_total_strict}")
    log(f"  mean queries per ASIN: {sum(len(v) for v in pools.values())/max(1,len(pools)):.1f}")

    out_doc = {
        "config": {
            "description": "PILOT 50 ASINs — syntax-prototype guided pool K_syntax=16, K_per_proto=4",
            "K_syntax": 16, "K_per_proto": K_PER_PROTO, "N_INPUT": N_INPUT,
            "PILOT_N_ASIN": PILOT_N_ASIN, "RANDOM_SEED": RANDOM_SEED,
            "K_POOL_eff": 16 * K_PER_PROTO,
        },
        "n_asins": len(asin_attrs),
        "n_total_strict": n_total_strict,
        "filter_counts": filter_counts,
        "pools": pools,
    }
    with open(PILOT_OUT, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {PILOT_OUT}")

    summary = {
        "description": "Phase 2 PILOT 50 ASINs — sanity check before 2174 ASIN full run",
        "filter_counts": filter_counts,
        "n_total_strict": n_total_strict,
        "next_step": "Run gaussian/build_strict34_prototype_audit_pilot.py on this pool; if F3 drops ≥10pp or min_d_self ≤9 → go full-scale.",
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")


if __name__ == "__main__":
    main()