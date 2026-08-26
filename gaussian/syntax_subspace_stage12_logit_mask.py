"""Stage 12 — Token-Level Logit Masking for Content/Structure Control.

Stage 11 used PROMPT-side blacklist ("Do not use 'best', 'perfect', ...")
which leaks ~35.5% of generations. Stage 12 upgrades to:

  HARD CONSTRAINT: vLLM `bad_words` → sets forbidden tokens' logits to -∞

Three-tier vocabulary (per user's framework):
  V_fixed_content  — must-include brand/color/age/category
  V_structural     — allowed function words (a/the/for/with/that/which/and/is...)
  V_forbidden      — bias words (best/perfect/ideal/durable/lightweight/premium...)
                     → logit = -∞ (impossible to emit)

Stage 11 prompt: 35.5% bias leak
Stage 12 target: 0% bias leak (hard guarantee via bad_words)

Per-class structural prompt (kept from Stage 11) provides additional
guidance; bad_words enforces the impossibility of forbidden tokens.

Inputs:
  - stage8_5_pool.json (for asin selection)
  - stage8_5_selection.json (asin, user_id) pairs

Outputs:
  - stage12_logit_masked_pool.json — 560 candidates (20 ASINs × 7 classes × K=4)
  - stage12_comparison.json — bias/content/class metrics vs Stage 11

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
        gaussian/syntax_subspace_stage12_logit_mask.py \\
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage12_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_IN = SCRATCH / "stage8_5_pool.json"
USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
STAGE8_5_SELECTION = SCRATCH / "stage8_5_selection.json"

OUTPUT_POOL = SCRATCH / "stage12_logit_masked_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage12_comparison.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL",
    "COORDINATION",
    "RELATIVE_CLAUSE",
    "MODIFIER_FRONTED",
    "PREDICATE",
    "QUESTION",
    "SIMPLE",
]

# Tier 3: forbidden semantic tokens — bias words that introduce
# unsupported semantic content. These will have logit = -∞ during decoding.
# Includes both full words AND common substrings that would otherwise
# slip through (e.g. "for baby" as a phrase).
FORBIDDEN_WORDS = [
    # adjectives
    "durable", "lightweight", "comfortable", "perfect", "modern",
    "stylish", "premium", "high-quality", "top-quality", "quality",
    "ideal", "great", "excellent", "best", "amazing",
    "soft", "smooth", "gentle", "cute", "adorable", "lovely",
    "safe", "safer", "safety", "reliable", "trustworthy",
    "beautiful", "elegant", "sleek", "unique", "innovative",
    "luxury", "luxurious", "deluxe", "classic", "traditional",
    # comparisons
    "better", "worse", "superior", "inferior",
    # superlatives / hyperbole
    "ultimate", "supreme", "extraordinary", "exceptional",
    # new product categories
    "organic", "natural", "eco-friendly", "sustainable",
    # marketing fluff
    "professional", "expert", "advanced", "premier",
    # common bias phrases (multi-token)
    "for baby", "for your baby", "for babies", "for the baby",
]

# Tier 2: structural / function vocabulary — these are allowed freely
# so the LLM can form grammatical sentences. We do NOT add these to
# `bad_words`; they remain at their normal logits.
STRUCTURAL_HINTS = (
    "a, the, for, with, in, of, that, which, and, or, but, "
    "is, are, designed, suitable, product, item, looking, need, "
    "want, find, search, buy, get, recommend, options, choices, "
    "available, similar, related, comparison"
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_structural(query: str, _=None) -> str:
    """Same regex classifier as Stage 11."""
    q = query.strip()
    low = q.lower()
    if q.endswith("?"):
        return "QUESTION"
    rel_match = re.search(r"\b(that|which|who)\b\s+\w+s?\s+\w+", low)
    if rel_match:
        return "RELATIVE_CLAUSE"
    if re.search(r"\b(and|but|or)\b\s+(a|an|the|I|it|they|[A-Z])", q):
        coord_match = re.search(r"\b(and|but|or)\s+(\w+)", low)
        if coord_match and coord_match.group(2) not in ("the", "a", "an"):
            return "COORDINATION"
    pp_match = re.search(r"\b(in|for|under|within)\s+[A-Z][\w\s&-]{2,50}\s*$", q)
    if pp_match:
        return "PREPOSITIONAL"
    if re.match(r"^(For|With|In|To|As)\s+\w+[\w\s]*,\s+", q):
        return "MODIFIER_FRONTED"
    if re.search(r"\bis\s+(a|an|the|designed|intended|suitable|perfect)\b", low):
        return "PREDICATE"
    if re.search(r"\bare\s+(a|an|the|designed|intended|suitable)\b", low):
        return "PREDICATE"
    return "SIMPLE"


SENTENCE_STARTERS = {"I", "I'm", "Looking", "Searching", "Find", "Need", "Want",
                     "Here", "There", "Get", "Buy"}


def extract_attrs(text: str):
    low = text.lower()
    brand = ""
    for m in re.finditer(r"\b([A-Z][a-zA-Z&\-]+(?:\s+[A-Z][a-zA-Z&\-]+)*)\b", text):
        cand = m.group(1)
        if cand not in SENTENCE_STARTERS and len(cand) >= 3:
            brand = cand
            break
    color_m = re.search(r"\b(white|black|red|blue|green|yellow|pink|grey|gray|silver|gold|brown|purple)\b", low)
    color = color_m.group(1) if color_m else ""
    age_m = re.search(r"\b(infant|toddler|baby|newborn|child|adult)s?\b", low)
    age = age_m.group(1) if age_m else ""
    cat_m = re.search(r"(?:in|within|under)\s+(?:the\s+)?([A-Z][\w\s&-]+?)(?:\s+main)?(?:\s+(?:category|section|department))?\s*\.?$", text)
    if cat_m:
        category = cat_m.group(1).strip()
    else:
        caps = re.findall(r"[A-Z][\w&-]+(?:\s+[A-Z][\w&-]+){1,4}", text)
        category = caps[-1] if caps else ""
    if len(category) < 5:
        category = ""
    return brand, color, age, category


def build_class_prompt(brand: str, color: str, age: str, category: str, cls: str) -> str:
    """Build structural-class-conditioned prompt with content lock."""
    base = (
        f"Generate ONE short search query (under 20 words) for a {brand} "
        f"{color} {age} product in {category} category. "
        f"Use only these words and normal function words: {STRUCTURAL_HINTS}. "
        f"Do not use any product-describing adjectives outside the listed hints."
    )
    rules = {
        "PREPOSITIONAL": f"End the query with 'in {category}'. Example: 'A {brand} product in {category}'.",
        "COORDINATION": f"Use 'and' to coordinate two noun or verb phrases. Example: '{brand} {color} and {age} products in {category}'.",
        "RELATIVE_CLAUSE": f"Include a relative clause with 'that' or 'which'. Example: 'A {brand} product that fits {age} needs in {category}'.",
        "MODIFIER_FRONTED": f"Start with a fronted modifier (For/With/In) and a comma. Example: 'For {age} use, {brand} {color} products in {category}'.",
        "PREDICATE": f"Use copula 'is' to link. Example: 'A {brand} product is in {category}'.",
        "QUESTION": f"Phrase as a question ending with '?'. Example: 'Where can I find {brand} {color} {age} products in {category}?'",
        "SIMPLE": f"No relative clause, no coordination, no fronted modifier. Example: '{brand} {color} {age} {category}'.",
    }
    return f"{base} {rules[cls]} Return only the query text, no numbering, no quotes."


def main():
    log("=== Stage 12 — Token-Level Logit Masking (bad_words hard constraint) ===")

    # Load LLM client (per CLAUDE.md Rule 8/9h)
    from llm_client import QwenLocalClient
    from vllm import SamplingParams
    log("  using QwenLocalClient + vLLM SamplingParams(bad_words=...)")

    # Load pool
    pool = json.load(open(POOL_IN))
    pools = pool["pools"]
    sorted_asins = sorted(pools.keys(), key=lambda a: -len(pools[a]))[:20]

    # Lazy-init client
    log("  initializing QwenLocalClient (loads vLLM in-process, ~30s)...")
    client = QwenLocalClient()
    tokenizer = client._backend.tokenizer
    model = client._backend.model
    log(f"  tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}")

    # Build SamplingParams with bad_words (hard forbidden tokens)
    sampling = SamplingParams(
        max_tokens=60, temperature=0.8,
        top_p=0.95 if 0.8 > 0 else 1.0,
        repetition_penalty=1.05,
        bad_words=FORBIDDEN_WORDS,  # ← HARD CONSTRAINT
    )
    log(f"  bad_words count: {len(FORBIDDEN_WORDS)}")
    log(f"  SamplingParams OK")

    # Generate per ASIN × class × variant
    content_locked_pool = {}
    bias_hits_total = 0
    coverage_total = 0
    n_total_outputs = 0

    for ai, asin in enumerate(sorted_asins):
        queries = pools[asin]
        attrs = None
        for q in queries:
            text = q.get("query", str(q))
            brand, color, age, category = extract_attrs(text)
            if brand and color and age and category:
                attrs = (brand, color, age, category)
                break
        if not attrs:
            for q in queries:
                text = q.get("query", str(q))
                brand, color, age, category = extract_attrs(text)
                attrs = (brand or "Product", color or "colored", age or "for users", category or "main category")
                break
        brand, color, age, category = attrs

        all_classes = list(STRUCTURAL_CLASSES)
        all_prompts = []
        for cls in all_classes:
            for k in range(4):  # K=4 variants per class
                p = build_class_prompt(brand, color, age, category, cls) + f" (variant {k+1})"
                all_prompts.append(p)

        # Batched generation with bad_words
        full_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in all_prompts
        ]

        t0 = time.time()
        vllm_outputs = model.generate(full_prompts, sampling)
        gen_time = time.time() - t0

        outputs = [o.outputs[0].text.strip() if o.outputs else "" for o in vllm_outputs]

        # Regroup + verify
        asin_locked = {}
        n_cands = 0
        for ci, cls in enumerate(all_classes):
            generated = []
            for k in range(4):
                idx = ci * 4 + k
                out = outputs[idx] if idx < len(outputs) else ""
                # Should be ZERO bias hits now (hard constraint), but verify
                bias_hit = any(b in out.lower() for b in FORBIDDEN_WORDS)
                content_ok = all(c.lower() in out.lower() for c in [brand, color, age])
                if bias_hit:
                    bias_hits_total += 1
                if content_ok:
                    coverage_total += 1
                if out:
                    n_cands += 1
                n_total_outputs += 1
                generated.append({
                    "text": out,
                    "content_ok": content_ok,
                    "bias_hit": bias_hit,
                    "predicted_class": classify_structural(out, None),
                })
            asin_locked[cls] = {"prompt": build_class_prompt(brand, color, age, category, cls), "generated": generated}
        content_locked_pool[asin] = {"attrs": attrs, "by_class": asin_locked}
        log(f"  [{ai+1}/{len(sorted_asins)}] {asin}: attrs={attrs}, {n_cands} cands, gen_time={gen_time:.1f}s")

    log(f"\n  Total bias hits: {bias_hits_total}/{n_total_outputs} ({bias_hits_total/max(n_total_outputs,1)*100:.2f}%)")
    log(f"  Total content coverage: {coverage_total}/{n_total_outputs} ({coverage_total/max(n_total_outputs,1)*100:.2f}%)")

    # Save pool
    with open(OUTPUT_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 12: logit-level hard-constraint generation via vLLM bad_words",
                "n_asins": len(content_locked_pool),
                "structural_classes": STRUCTURAL_CLASSES,
                "forbidden_words_count": len(FORBIDDEN_WORDS),
                "structural_hints": STRUCTURAL_HINTS,
            },
            "stats": {
                "bias_hits": bias_hits_total,
                "content_coverage": coverage_total,
                "total_outputs": n_total_outputs,
            },
            "pool": content_locked_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_POOL}")

    # === Per-class hit rate + bias breakdown ===
    log("\n=== Per-class hit rate (Stage 12) ===")
    class_hits = Counter()
    class_total = Counter()
    for asin, data in content_locked_pool.items():
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                if g.get("text"):
                    pred = g.get("predicted_class", cls)
                    class_total[cls] += 1
                    if pred == cls:
                        class_hits[cls] += 1
    log(f"  {'Class':<18} {'hits':>6} {'total':>6} {'rate':>6}")
    for cls in STRUCTURAL_CLASSES:
        h = class_hits[cls]
        t = class_total[cls]
        rate = h / t if t > 0 else 0
        log(f"  {cls:<18} {h:>6} {t:>6} {rate:>6.1%}")

    # === Comparison vs Stage 11 ===
    log("\n=== Comparison vs Stage 11 ===")
    s11 = json.load(open(SCRATCH / "stage11_content_locked_pool.json"))
    s11_pool = s11["pool"]
    s11_stats = s11["stats"]

    # Per-class hit rate for Stage 11
    s11_class_hits = Counter()
    s11_class_total = Counter()
    for asin, data in s11_pool.items():
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                if g.get("text"):
                    pred = g.get("predicted_class", cls)
                    s11_class_total[cls] += 1
                    if pred == cls:
                        s11_class_hits[cls] += 1

    log(f"  {'Class':<18} {'S11 rate':>10} {'S12 rate':>10} {'Δ':>8}")
    for cls in STRUCTURAL_CLASSES:
        s11_rate = s11_class_hits[cls] / max(s11_class_total[cls], 1)
        s12_rate = class_hits[cls] / max(class_total[cls], 1)
        log(f"  {cls:<18} {s11_rate:>10.1%} {s12_rate:>10.1%} {s12_rate - s11_rate:>+7.1%}")

    log(f"\n  {'Metric':<25} {'S11':>10} {'S12':>10} {'Δ':>10}")
    log(f"  {'-'*60}")
    s11_bias_rate = s11_stats["bias_hits"] / max(s11_stats.get("n_outputs", 560), 1)
    s12_bias_rate = bias_hits_total / max(n_total_outputs, 1)
    log(f"  {'bias_hit_rate':<25} {s11_bias_rate:>10.1%} {s12_bias_rate:>10.1%} {(s12_bias_rate - s11_bias_rate)*100:>+9.1f}pp")
    s11_cov_rate = s11_stats["content_coverage"] / max(s11_stats.get("n_outputs", 560), 1)
    s12_cov_rate = coverage_total / max(n_total_outputs, 1)
    log(f"  {'content_coverage_rate':<25} {s11_cov_rate:>10.1%} {s12_cov_rate:>10.1%} {(s12_cov_rate - s11_cov_rate)*100:>+9.1f}pp")

    # Save comparison
    comparison = {
        "config": {
            "description": "Stage 12 vs Stage 11: logit masking vs prompt blacklist",
            "forbidden_words_count": len(FORBIDDEN_WORDS),
        },
        "stage12": {
            "bias_hits": bias_hits_total,
            "content_coverage": coverage_total,
            "total_outputs": n_total_outputs,
            "bias_hit_rate": s12_bias_rate,
            "content_coverage_rate": s12_cov_rate,
            "per_class_hit_rate": {
                cls: class_hits[cls] / max(class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
        },
        "stage11": {
            "bias_hits": s11_stats["bias_hits"],
            "content_coverage": s11_stats["content_coverage"],
            "n_outputs": 560,
            "bias_hit_rate": s11_bias_rate,
            "content_coverage_rate": s11_cov_rate,
            "per_class_hit_rate": {
                cls: s11_class_hits[cls] / max(s11_class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
        },
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_COMPARISON}")

    log("\n=== Stage 12 complete ===")


if __name__ == "__main__":
    main()