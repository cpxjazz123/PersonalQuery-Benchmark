#!/usr/bin/env python3
"""User-specific Exemplar Anchor PILOT.

用户指令 2026-08-30 (Phase 4): K-sweep + global prototype 都未解决 F3% 高问题。
下一步必须测 user-specific style signal 是否能解决 distribution misalignment。

**严格 leakage 控制**:
- 每个 (user, asin) pair: 抽 1-2 句 user 历史 review(过 5-layer filter) 作 style seed
- 商品 attrs 仍来自目标 ASIN(不变), exemplar 不引入商品内容
- LLM prompt 明确指示: "模仿下面句子的句法风格, 不要提到里面任何商品或属性值"
- exemplar 文字不直接进入 pool, 只作 style seed

**Pilot 设计** (per pair):
- 50 ASINs × 20 users/ASIN = 1000 (user, asin) pairs
- 过滤有 ≥1 review 的 pair = 169 pairs (38%)
- 每 pair 抽 2 exemplar sentences, 生成 K_PER_PAIR=4 query
- 总 prompts = 169 × 4 = 676 prompts
- 5-layer strict filter applied

**对照**: corrected baseline pilot (F3=85.7%, min_d_self=11.12)
**判别**: F3 明显下降 + min_d_self < 9 = 真信号

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

# Import helpers from corrected baseline pilot (same attrs-corrected prompt).
_CORR = REPO_ROOT / "gen_query" / "syntax_subspace_corrected_baseline_pool_pilot.py"
_spec = importlib.util.spec_from_file_location("corr", _CORR)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["corr"] = _mod
_spec.loader.exec_module(_mod)
batch_generate_vllm = _mod.batch_generate_vllm
get_top_n_attrs = _mod.get_top_n_attrs
build_user_content = _mod.build_user_content
log = _mod.log

# Pull strict-filter helpers from prototype regen (which has them all).
_PROTO = REPO_ROOT / "gen_query" / "syntax_subspace_prototype_pool_regen.py"
_spec2 = importlib.util.spec_from_file_location("proto_regen", _PROTO)
_proto = importlib.util.module_from_spec(_spec2)
sys.modules["proto_regen"] = _proto
_spec2.loader.exec_module(_proto)
count_attrs_covered = _proto.count_attrs_covered
has_invalid_punct = _proto.has_invalid_punct
has_first_person = _proto.has_first_person
has_emoji = _proto.has_emoji
has_self_talk = _proto.has_self_talk
is_query_too_long = _proto.is_query_too_long
n_tokens_simple = _proto.n_tokens_simple

PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
COHORT_ASINS = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"
SENT_IN = SCRATCH / "sentences_for_rewrite_10k.jsonl"

PILOT_N_ASIN = 50
N_SAMPLE_USERS = 20       # per ASIN
RANDOM_SEED_ASIN = 42
RANDOM_SEED_USERS = 0     # separate seed for user sampling
K_PER_PAIR = 4            # 4 queries per (user, asin) pair
N_EXEMPLARS = 2           # 2 review sentences per pair as style seed
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5

OUT_POOL = SCRATCH / "pool_exemplar_anchor_pilot.json"

# Prompt: ASIN attrs (不变) + user exemplar (style only, NOT content)
GEN_SYSTEM_TMPL_EXEMPLAR = (
    "You are a real customer searching for a product. "
    "Write a single natural search query as if you are a shopper looking for "
    "this exact product. The query MUST mention ALL of these product attributes:\n"
    "{ATTRIBUTES}\n\n"
    "For syntactic style reference ONLY, here are 1-2 sentences written by "
    "a real customer with similar preferences (study the sentence structure, "
    "length, voice, and word order — DO NOT mention any of the products, "
    "brands, materials, or other attributes from these example sentences):\n"
    "{EXEMPLARS}\n\n"
    "Rules:\n"
    "- First-person voice: use 'I', 'my', 'I'm looking for', 'I want', etc.\n"
    "- One natural sentence (NOT a bulleted list).\n"
    "- Do NOT start with 'Here:', 'Brand:', 'Attribute:' or any prefix.\n"
    "- Do NOT use emoji.\n"
    "- Do NOT talk to the AI (no 'can someone help', no 'thanks!').\n"
    "- Do NOT mention any products, brands, materials, or attributes from "
    "the example sentences above. Style reference only.\n"
    "- Maximum 60 tokens.\n"
    "Write only the query, no commentary."
)


def make_exemplar_prompt(attrs: dict, exemplars: list[str]) -> str:
    """Build prompt: ASIN attrs (不变) + user exemplar (style only)."""
    ex_text = "\n".join(f"- {e}" for e in exemplars)
    return GEN_SYSTEM_TMPL_EXEMPLAR.format(
        ATTRIBUTES=build_user_content(attrs),
        EXEMPLARS=ex_text,
    )


def load_user_sentences() -> dict[str, list[str]]:
    """Map user_id -> list of sentences."""
    sent_by_user: dict[str, list[str]] = {}
    with open(SENT_IN, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("sentence_text", "").strip()
            wc = len(text.split())
            if wc < 5 or wc > 60:
                continue
            sent_by_user.setdefault(rec["user_id"], []).append(text)
    return sent_by_user


def main():
    log("=== User-specific Exemplar Anchor PILOT ===")
    pattrs = json.load(open(PATTRS_PATH))
    cohort = json.load(open(COHORT_ASINS))

    # Same pilot cohort as corrected baseline (seed=42 ASIN sample)
    rng_asin = random.Random(RANDOM_SEED_ASIN)
    asin_data_sorted = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng_asin.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    log(f"  pilot ASINs: {len(asin_data_pilot)} (seed={RANDOM_SEED_ASIN})")

    # For each ASIN, sample N_SAMPLE_USERS users from users_sampled (seed=0)
    rng_users = random.Random(RANDOM_SEED_USERS)
    pairs = []
    for entry in asin_data_pilot:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng_users.shuffle(users)
        for u in users:
            pairs.append((u, entry["asin"]))
    log(f"  total (user, asin) pairs: {len(pairs)}")

    # Load user sentences
    sent_by_user = load_user_sentences()
    log(f"  sentences map: {len(sent_by_user)} users, "
        f"pairs with sentences: {sum(1 for u, _ in pairs if u in sent_by_user)}")

    # Filter pairs to those with sentences
    pairs_with_sents = [(u, a) for u, a in pairs if u in sent_by_user and len(sent_by_user[u]) >= 1]
    log(f"  pairs with ≥1 sentence: {len(pairs_with_sents)}")

    # Build per-pair attrs + exemplar
    n_no_attrs = n_lt_n = 0
    pair_data = []  # list of {user, asin, attrs, exemplars}
    for u, a in pairs_with_sents:
        pa = pattrs.get(a)
        if pa is None:
            n_no_attrs += 1
            continue
        top_n = get_top_n_attrs(pa, N_INPUT)
        if len(top_n) < N_INPUT:
            n_lt_n += 1
            continue
        # Sample N_EXEMPLARS sentences
        sents = sent_by_user[u]
        rng = random.Random(hash((u, a)) & 0xffffffff)
        rng.shuffle(sents)
        exemplars = sents[:N_EXEMPLARS]
        pair_data.append({
            "user_id": u, "asin": a, "attrs": top_n, "exemplars": exemplars,
        })
    log(f"  valid pairs (with attrs + exemplars): {len(pair_data)} "
        f"(filtered: no_attrs={n_no_attrs}, lt_{N_INPUT}_attrs={n_lt_n})")

    # Build prompts: per pair × K_PER_PAIR
    all_prompts = []
    pair_k = []
    for pd in pair_data:
        prompt = make_exemplar_prompt(pd["attrs"], pd["exemplars"])
        for k in range(K_PER_PAIR):
            all_prompts.append(prompt)
            pair_k.append((pd["user_id"], pd["asin"], k))
    log(f"  total prompts: {len(all_prompts)} = {len(pair_data)} pairs × {K_PER_PAIR}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    # 5-layer strict filter (same as corrected baseline)

    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full": 0, "invalid": 0, "first_p": 0, "emoji": 0, "self_talk": 0, "too_long": 0}
    pair_strict_count = collections.Counter()

    for (u, a, k), out in zip(pair_k, outputs):
        text = out.strip() if out else ""
        # Find attrs for this pair
        attrs_dict = next(pd["attrs"] for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
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
            pair_strict_count[(u, a)] += 1
        pools[a].append({
            "user_id": u, "k": k, "query": text, "strict": strict,
            "attrs_covered": n_cov, "invalid": invalid,
            "first_person": first_p, "emoji": emoji,
            "self_talk": self_talk, "too_long": too_long,
            "n_tok": n_tokens_simple(text),
        })

    log(f"  filter breakdown: {filter_counts}")
    log(f"  total strict: {n_total_strict}")
    log(f"  pairs with ≥1 strict query: {len(pair_strict_count)}")
    log(f"  mean queries per ASIN: {sum(len(v) for v in pools.values())/max(1,len(pools)):.1f}")

    out_doc = {
        "config": {
            "description": "User-specific Exemplar Anchor PILOT — strict leakage control, ASIN attrs不变, exemplar仅style seed",
            "PILOT_N_ASIN": PILOT_N_ASIN,
            "N_SAMPLE_USERS": N_SAMPLE_USERS,
            "K_PER_PAIR": K_PER_PAIR,
            "N_EXEMPLARS": N_EXEMPLARS,
            "TEMP": TEMP,
            "MAX_TOKENS": MAX_TOKENS,
            "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
            "N_INPUT": N_INPUT,
        },
        "n_pairs_total": len(pair_data),
        "n_pairs_with_strict": len(pair_strict_count),
        "n_total_strict": n_total_strict,
        "filter_counts": filter_counts,
        "pools": pools,
    }
    with open(OUT_POOL, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {OUT_POOL}")


if __name__ == "__main__":
    main()