#!/usr/bin/env python3
"""Phase 4.B — N_EXEMPLARS Sweep {1, 3, 5, 8} (N=2 already in Phase 4).

用户指令 2026-08-30: Phase 4 (N=2) PARTIAL-GO (F3 74.9%, max_M 翻倍, 但
min_d_self 没降). 现在只改 N_EXEMPLARS, 其它一律不变, 隔离 signal quantity
vs quality.

**固定** (across all K):
- Same 50 ASINs (seed=42 cohort subset)
- Same 169 (user, asin) pairs (有 sentences 的)
- K_PER_PAIR=4
- Same ASIN attrs (不变)
- Same prompt template (除 N_EXEMPLARS 数量外)
- Same TEMP=0.7, MAX_TOKENS=120, MAX_QUERY_TOKENS=60
- Same 5-layer strict filter
- Same leakage-control rules

**采样策略** (与 Phase 4 不同, 避免连续句):
- 按 user sentence list 的均匀 index 抽样 (每隔 len/N 取 1 句),
  保证 5 个 exemplar 不会全来自同一条 review 的连续句子

**新增诊断量**:
- Δd = selected_margin (Stage 4 已记录 = d_nearest_other - d_self)
- 直接从 selection entry 提取, 无需重算

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

# Import helpers from prototype regen
_PROTO = REPO_ROOT / "gen_query" / "syntax_subspace_prototype_pool_regen.py"
_spec = importlib.util.spec_from_file_location("proto_regen", _PROTO)
_proto = importlib.util.module_from_spec(_spec)
sys.modules["proto_regen"] = _proto
_spec.loader.exec_module(_proto)
batch_generate_vllm = _proto.batch_generate_vllm
build_user_content = _proto.build_user_content
log = _proto.log
get_top_n_attrs = _proto.get_top_n_attrs
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

# === Phase 4 sweep config ===
PILOT_N_ASIN = 50
N_SAMPLE_USERS = 20
RANDOM_SEED_ASIN = 42
RANDOM_SEED_USERS = 0
K_PER_PAIR = 4
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5

# Sweep settings
N_EXEMPLARS_VALUES = [1, 3, 5, 8]

# Prompt template — exactly same as Phase 4 except N_EXEMPLARS variable
GEN_SYSTEM_TMPL_EXEMPLAR = (
    "You are a real customer searching for a product. "
    "Write a single natural search query as if you are a shopper looking for "
    "this exact product. The query MUST mention ALL of these product attributes:\n"
    "{ATTRIBUTES}\n\n"
    "For syntactic style reference ONLY, here are {N_EXEMPLARS} sentences written by "
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


def make_exemplar_prompt(attrs: dict, exemplars: list[str], n_exemplars: int) -> str:
    ex_text = "\n".join(f"- {e}" for e in exemplars)
    return GEN_SYSTEM_TMPL_EXEMPLAR.format(
        ATTRIBUTES=build_user_content(attrs),
        EXEMPLARS=ex_text,
        N_EXEMPLARS=n_exemplars,
    )


def sample_exemplars_spread(sents: list[str], n: int, rng: random.Random) -> list[str]:
    """Spread-sample N sentences from sents (no consecutive)."""
    if len(sents) <= n:
        return list(sents)
    # Pick N indices evenly spaced from [0, len-1]
    step = (len(sents) - 1) / max(1, n - 1) if n > 1 else 0
    idxs = [int(round(i * step)) for i in range(n)]
    return [sents[i] for i in idxs]


def load_user_sentences() -> dict[str, list[str]]:
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


def get_pairs_and_attrs() -> list[dict]:
    """Return list of {user_id, asin, attrs} for the 169 pairs."""
    pattrs = json.load(open(PATTRS_PATH))
    cohort = json.load(open(COHORT_ASINS))
    rng_asin = random.Random(RANDOM_SEED_ASIN)
    asin_data_sorted = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng_asin.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    rng_users = random.Random(RANDOM_SEED_USERS)
    pairs = []
    for entry in asin_data_pilot:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng_users_local = random.Random(RANDOM_SEED_USERS)
        rng_users_local.shuffle(users)
        for u in users:
            pa = pattrs.get(entry["asin"])
            if pa is None:
                continue
            top_n = get_top_n_attrs(pa, N_INPUT)
            if len(top_n) < N_INPUT:
                continue
            pairs.append({
                "user_id": u,
                "asin": entry["asin"],
                "attrs": top_n,
            })
    return pairs


def run_one_N(N_EXEMPLARS: int, pairs: list[dict], sent_by_user: dict) -> dict:
    log(f"\n=== N_EXEMPLARS = {N_EXEMPLARS} ===")

    # Sample exemplars per pair (spread-sample, deterministic seed)
    pair_data = []
    n_no_sents = 0
    for pd in pairs:
        sents = sent_by_user.get(pd["user_id"], [])
        if len(sents) < 1:
            n_no_sents += 1
            continue
        # Use deterministic seed per pair
        rng = random.Random(hash((pd["user_id"], pd["asin"], N_EXEMPLARS)) & 0xffffffff)
        # Sort sents by user stable order, then spread-sample
        sents_sorted = sorted(sents)  # stable
        rng.shuffle(sents_sorted)
        exemplars = sample_exemplars_spread(sents_sorted, N_EXEMPLARS, rng)
        pair_data.append({**pd, "exemplars": exemplars})
    log(f"  pairs with ≥1 sentence: {len(pair_data)} (filtered {n_no_sents})")

    # Build prompts
    all_prompts = []
    pair_k = []
    for pd in pair_data:
        prompt = make_exemplar_prompt(pd["attrs"], pd["exemplars"], N_EXEMPLARS)
        for k in range(K_PER_PAIR):
            all_prompts.append(prompt)
            pair_k.append((pd["user_id"], pd["asin"], k))
    n_prompts = len(all_prompts)
    log(f"  total prompts: {n_prompts} = {len(pair_data)} pairs × {K_PER_PAIR}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    # 5-layer strict filter
    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full": 0, "invalid": 0, "first_p": 0, "emoji": 0, "self_talk": 0, "too_long": 0}

    for (u, a, k), out in zip(pair_k, outputs):
        attrs_dict = next(pd["attrs"] for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
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
            "user_id": u, "k": k, "query": text, "strict": strict,
            "attrs_covered": n_cov, "invalid": invalid,
            "first_person": first_p, "emoji": emoji,
            "self_talk": self_talk, "too_long": too_long,
            "n_tok": n_tokens_simple(text),
            "n_exemplars": N_EXEMPLARS,
        })

    log(f"  filter breakdown: {filter_counts}")
    log(f"  total strict: {n_total_strict}")

    out_path = SCRATCH / f"pool_exemplar_anchor_N{N_EXEMPLARS}.json"
    out_doc = {
        "config": {
            "description": f"Exemplar anchor N_EXEMPLARS={N_EXEMPLARS}",
            "N_EXEMPLARS": N_EXEMPLARS,
            "K_PER_PAIR": K_PER_PAIR,
            "n_pairs": len(pair_data),
        },
        "filter_counts": filter_counts,
        "n_total_strict": n_total_strict,
        "pools": pools,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {out_path}")
    return {"N_EXEMPLARS": N_EXEMPLARS, "pool_path": str(out_path), "n_pairs": len(pair_data)}


def main():
    log("=== Phase 4.B — N_EXEMPLARS Sweep {1, 3, 5, 8} ===")
    log("  N=2 already in Phase 4 commit (a10987d)")
    pairs = get_pairs_and_attrs()
    log(f"  pilot pairs (with attrs): {len(pairs)}")
    sent_by_user = load_user_sentences()
    log(f"  sentences map: {len(sent_by_user)} users")

    results = {}
    for N in N_EXEMPLARS_VALUES:
        results[str(N)] = run_one_N(N, pairs, sent_by_user)

    summary = {
        "description": "Phase 4.B N_EXEMPLARS sweep (signal quantity)",
        "config": {
            "PILOT_N_ASIN": PILOT_N_ASIN,
            "N_SAMPLE_USERS": N_SAMPLE_USERS,
            "K_PER_PAIR": K_PER_PAIR,
            "TEMP": TEMP,
        },
        "N_EXEMPLARS_values": N_EXEMPLARS_VALUES,
        "results": results,
        "next_step": (
            "Phase 4.C: Run Stage 4 strict alignment on each N_EXEMPLARS pool, "
            "compute F3/min_d_self/max_M (=Delta_d)/PASSED%, then sweep-trend analysis."
        ),
    }
    summary_path = REPO_ROOT / "result/gen_query/exemplar_count_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {summary_path}")


if __name__ == "__main__":
    main()