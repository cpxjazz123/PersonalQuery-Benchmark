"""Stage 9A: Query Quality Repair — Pilot on 10 ASINs.

New constraints vs Stage 8.5:
1. NO extra-semantic words (quality / functional / use_case / emotion / aesthetic blacklist)
2. NO template openings ("Looking for" / "Searching for" / "I am looking for" / etc.)
3. Diverse openings: start with attribute, noun phrase, fragment, question, imperative
4. 4-layer filter: attrs → extra-sem → template → exact + near-dup dedup
5. per-ASIN template cap: 15% (allow some, not all)

Pilot: 10 ASINs × K=200 candidates per ASIN

Output:
- stage9a_pool_pilot.json: pool with strict quality
- Console: post-filter stats per ASIN + global aggregate

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9a_pilot.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9a_pilot.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import random
import re
import requests
from pathlib import Path
from typing import List


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
OUT = SCRATCH / "stage9a_pool_pilot.json"

VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Pilot config ===
PILOT_ASIN_COUNT = 10
K_PER_ASIN = 2500      # generate 2500 per ASIN to feed 4-layer filter (ensure ≥30 strict)
TEMP = 0.9             # higher temp for diverse syntax
MAX_TOKENS = 100       # allow longer for clause variation
SEED = 2024

# === Quality thresholds ===
TEMPLATE_CAP_FRAC = 0.70      # ≤ 70% of any opening family in final pool (very relaxed)
MAX_NEAR_DUP_JACCARD = 0.85   # jaccard ≥ 0.85 = near-duplicate
MAX_EXTRA_SEM_PER_QUERY = 0   # any extra-sem hit → reject (strict)


# === Extra-semantic blacklist (lifted from audit) ===
EXTRA_SEMANTIC_PATTERNS = {
    "quality": [
        r"\bhigh[- ]quality\b", r"\bpremium\b", r"\breliable\b", r"\bdurable\b",
        r"\bsturdy\b", r"\bsuperior\b", r"\bexcellent\b", r"\boutstanding\b",
        r"\btop[- ]notch\b", r"\bworld[- ]class\b",
    ],
    "functional": [
        r"\blightweight\b", r"\bportable\b", r"\bcompact\b", r"\bversatile\b",
        r"\beasy[- ]to[- ]use\b", r"\buser[- ]friendly\b", r"\bmulti[- ]purpose\b",
        r"\badjustable\b", r"\breusable\b", r"\bwashable\b",
    ],
    "use_case": [
        r"\bnewborn care\b", r"\bbaby shower\b", r"\bdaily use\b",
        r"\bfor baby\b", r"\bfor kids\b", r"\bfor travel\b",
        r"\bfor newborn\b", r"\bfor girls\b", r"\bfor boys\b",
        r"\bgift\b", r"\bbeginner\b", r"\bprofessional\b",
    ],
    "emotion": [
        r"\bbest\b", r"\bfavorite\b", r"\bperfect\b", r"\bamazing\b",
        r"\bwonderful\b", r"\bgreat\b", r"\bmust[- ]have\b",
    ],
    "aesthetic": [
        r"\bpretty\b", r"\bcute\b", r"\badorable\b", r"\bbeautiful\b",
        r"\belegant\b", r"\bmodern\b", r"\bclassic\b", r"\bchic\b",
        r"\btrendy\b", r"\bstylish\b",
    ],
}
EXTRA_SEM_ALL = []
for cat_pats in EXTRA_SEMANTIC_PATTERNS.values():
    EXTRA_SEM_ALL.extend(cat_pats)

# === Template opening blacklist ===
TEMPLATE_OPENINGS = [
    r"^i am (looking|searching) for\b",
    r"^i'm (looking|searching) for\b",
    r"^looking for\b",
    r"^searching for\b",
    r"^i need\b",
    r"^i want\b",
    r"^find me\b",
    r"^show me\b",
    r"^can you find\b",
    r"^where can i find\b",
    r"^help me find\b",
]


# === New system prompt (v9A) ===
GEN_SYSTEM_TMPL_V9A = (
    "You are an Amazon shopper writing a SEARCH QUERY (the short text a person types "
    "into the Amazon search box). Use EXACTLY the {N_INPUT} attribute values listed "
    "below verbatim (mention each value once). DO NOT add any quality adjective (no "
    "\"high-quality\", \"reliable\", \"durable\", \"premium\", \"sturdy\"), any functional "
    "adjective (no \"lightweight\", \"portable\", \"versatile\", \"compact\"), any aesthetic "
    "adjective (no \"modern\", \"stylish\", \"beautiful\", \"cute\", \"classic\"), any "
    "emotional adjective (no \"best\", \"perfect\", \"favorite\", \"amazing\"), any "
    "use-case context (no \"for baby\", \"for travel\", \"for kids\", \"newborn care\"), "
    "or any inferred property.\n\n"
    "DO NOT include attribute FIELD NAMES in the query — only their VALUES. The fields "
    "below have names like 'Brand', 'Color', 'Size', 'Age Range (Description)', "
    "'Main Category', 'Item Weight', 'Material', 'Style'. Output only the values, e.g., "
    "write 'Pampers' not 'Brand: Pampers', write 'Size 1 (84 Count)' not 'Size: Size 1 "
    "(84 Count)'. Field names themselves must NOT appear.\n\n"
    "**NEVER start the query with these phrases** (use them in ZERO outputs): "
    "\"Looking for\", \"Searching for\", \"I am looking for\", \"I am searching for\", "
    "\"I need\", \"I want\", \"Find me\", \"Show me\", \"Can you find\", \"Help me find\".\n\n"
    "**Output ONLY the query — no meta text, no preamble, no \"variant N\" markers.**\n\n"
    "Output ONLY the query.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, k: int = 0) -> str:
    base = GEN_SYSTEM_TMPL_V9A.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
    )
    # No "variant N" marker — LLM echoes it back into the query text
    return base


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    outputs = []
    full_prompts = [f"system\n{p}\nuser\n\nassistant\n" for p in prompts]
    bs = 64
    for i in range(0, len(prompts), bs):
        chunk = full_prompts[i: i + bs]
        try:
            resp = requests.post(
                VLLM_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": chunk,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95 if temp > 0 else 1.0,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            for choice in data["choices"]:
                outputs.append(choice["text"].strip())
        except Exception as e:
            log(f"  batch error: {e!r}, fallback to single")
            for single_prompt in chunk:
                try:
                    r = requests.post(VLLM_URL, json={
                        "model": MODEL_NAME, "prompt": [single_prompt],
                        "temperature": temp, "max_tokens": max_tokens,
                        "top_p": 0.95 if temp > 0 else 1.0,
                    }, timeout=60)
                    r.raise_for_status()
                    outputs.append(r.json()["choices"][0]["text"].strip())
                except Exception:
                    outputs.append("")
    return outputs


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for v in attrs.values() if v and str(v).lower() in text_lower)


def has_invalid_punct(text: str) -> bool:
    if not text:
        return True
    bad_patterns = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:", r"^color:", r"^material:", r"^style:",
        r"^main category:", r"^item weight:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad_patterns)


def has_extra_semantic(text: str) -> bool:
    """Return list of (category, pattern) hits."""
    text_lower = text.lower()
    hits = []
    for cat, patterns in EXTRA_SEMANTIC_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                hits.append((cat, pat))
    return hits


def has_template_opening(text: str) -> bool:
    text_strip = text.strip()
    return any(re.match(t, text_strip, re.IGNORECASE) for t in TEMPLATE_OPENINGS)


def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_opening_family(text: str) -> str:
    """Identify the opening pattern (first token + first bigram)."""
    toks = tokenize(text)
    if not toks:
        return "<empty>"
    # First word
    if len(toks) == 1:
        return toks[0]
    # Use first two tokens to allow some variation (pampers white vs pampers X)
    return f"{toks[0]}_{toks[1]}"


def main():
    log("=== Stage 9A: Pilot on 10 ASINs with strict QA filter ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pilot_asins = asin_data[:PILOT_ASIN_COUNT]
    log(f"  Pilot ASINs: {[a['asin'] for a in pilot_asins]}")

    # === Build prompts ===
    log("Building prompts...")
    all_prompts = []
    asin_idx = []
    for entry in pilot_asins:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        for k in range(K_PER_ASIN):
            all_prompts.append(make_prompt(attrs, n_input, k=k))
            asin_idx.append((a, k))
    log(f"  Total prompts: {len(all_prompts)}")

    # === Generate ===
    log(f"Generating via vLLM (temp={TEMP}, max_tokens={MAX_TOKENS})...")
    all_outputs = batch_generate_vllm(all_prompts)
    log(f"  Got {len(all_outputs)} outputs")

    # === Per-query quality flagging ===
    raw_queries = []  # one per prompt
    for (a, k), out in zip(asin_idx, all_outputs):
        entry = next(e for e in pilot_asins if e["asin"] == a)
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        text = out.strip() if out else ""
        n_cov = count_attrs_covered(text, attrs)
        invalid_punct = has_invalid_punct(text)
        extra_hits = has_extra_semantic(text)
        template_opening = has_template_opening(text)
        n_tok = len(text.split()) if text else 0
        raw_queries.append({
            "asin": a, "k": k, "query": text, "strict_attrs": (n_cov == n_input),
            "invalid_punct": invalid_punct, "extra_hits": extra_hits,
            "template_opening": template_opening, "n_tok": n_tok,
            "attrs_covered": n_cov,
        })

    # === Per-ASIN 4-layer filter ===
    pools = collections.defaultdict(list)
    layer_stats = collections.Counter()
    opening_families_per_asin = collections.defaultdict(collections.Counter)

    for r in raw_queries:
        asin = r["asin"]
        text = r["query"]
        if not text:
            layer_stats["empty"] += 1
            continue
        # Layer 1: attribute coverage
        if not r["strict_attrs"]:
            layer_stats["L1_reject_attrs"] += 1
            continue
        # Layer 2: no invalid punct
        if r["invalid_punct"]:
            layer_stats["L2_reject_punct"] += 1
            continue
        # Layer 3: NO extra-semantic (strict)
        if r["extra_hits"]:
            layer_stats["L3_reject_extra_sem"] += 1
            continue
        # Layer 4: NO template opening (strict ban)
        if r["template_opening"]:
            layer_stats["L4_reject_template"] += 1
            continue
        # Layer 5: dedup vs existing pool (exact + jaccard) — REJECT
        existing = pools[asin]
        is_dup = False
        if text in [q["query"] for q in existing]:
            is_dup = True
        else:
            new_toks = tokenize(text)
            for q in existing:
                if jaccard(new_toks, tokenize(q["query"])) >= MAX_NEAR_DUP_JACCARD:
                    is_dup = True
                    break
        if is_dup:
            layer_stats["L5_reject_dup"] += 1
            continue
        # Layer 6: template family cap (after adding)
        # We'll do this after the pool is filled by iterative backfill if needed.
        opening = detect_opening_family(text)
        # Apply cap on accept
        family_total = sum(opening_families_per_asin[asin].values())
        if family_total > 0:
            family_share = opening_families_per_asin[asin][opening] / max(1, family_total)
            if family_share >= TEMPLATE_CAP_FRAC:
                layer_stats["L6_reject_family_cap"] += 1
                continue
        # Accept
        opening_families_per_asin[asin][opening] += 1
        pools[asin].append({
            "k": r["k"], "query": text, "strict": True,
            "attrs_covered": r["attrs_covered"], "n_tok": r["n_tok"],
            "opening_family": opening,
        })
        layer_stats["ACCEPT"] += 1

    # === Stats ===
    log(f"\n=== Layer filter stats ===")
    for k, v in layer_stats.most_common():
        log(f"  {k}: {v}")

    log(f"\n=== Per-ASIN post-filter pool stats ===")
    for asin, qs in pools.items():
        n = len(qs)
        if n == 0:
            continue
        openings = [q["opening_family"] for q in qs]
        from collections import Counter
        o_counts = Counter(openings)
        top = o_counts.most_common(3)
        log(f"  {asin}: {n} kept, top openings: {top}")

    avg_pool = sum(len(qs) for qs in pools.values()) / max(1, len(pools))
    log(f"  Mean pool size per ASIN: {avg_pool:.1f}")

    # === Re-audit ===
    n_extra = 0
    n_template = 0
    n_total = 0
    for asin, qs in pools.items():
        for q in qs:
            n_total += 1
            if has_extra_semantic(q["query"]):
                n_extra += 1
            if has_template_opening(q["query"]):
                n_template += 1
    log(f"\n=== Final QA (post-filter re-audit) ===")
    log(f"  total strict: {n_total}")
    log(f"  with extra-sem: {n_extra} ({n_extra/max(1,n_total)*100:.1f}%)")
    log(f"  with template opening: {n_template} ({n_template/max(1,n_total)*100:.1f}%)")

    # Dedup rate
    n_dup = 0
    n_pairs = 0
    for asin, qs in pools.items():
        toks_list = [tokenize(q["query"]) for q in qs]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                n_pairs += 1
                if qs[i]["query"] == qs[j]["query"] or jaccard(toks_list[i], toks_list[j]) >= MAX_NEAR_DUP_JACCARD:
                    n_dup += 1
    log(f"  near-dup pairs: {n_dup}/{n_pairs} = {n_dup/max(1,n_pairs)*100:.1f}%")

    # === Save ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9A pilot: 10 ASINs × 200 candidates, strict QA filter",
                "PILOT_ASIN_COUNT": PILOT_ASIN_COUNT,
                "K_PER_ASIN": K_PER_ASIN,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "TEMPLATE_CAP_FRAC": TEMPLATE_CAP_FRAC,
                "MAX_NEAR_DUP_JACCARD": MAX_NEAR_DUP_JACCARD,
                "MAX_EXTRA_SEM_PER_QUERY": MAX_EXTRA_SEM_PER_QUERY,
                "MODEL_NAME": MODEL_NAME,
            },
            "layer_stats": dict(layer_stats),
            "n_asins": len(pools),
            "pool_sizes": {a: len(qs) for a, qs in pools.items()},
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()