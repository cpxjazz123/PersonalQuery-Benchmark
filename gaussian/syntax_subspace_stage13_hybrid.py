"""Stage 13 — Hybrid: Stage 11 class prompts + Stage 12 bad_words hard constraint.

Stage 11 lesson: Freeform prompts → 5.85 classes/asin diversity but 35.5% bias leak.
Stage 12 lesson: bad_words hard constraint → 1.8% bias leak but template collapse.

Stage 13 combines:
  ✓ Stage 11's per-class prompts (which got 5.85 classes diversity)
  ✓ Stage 12's bad_words forbidden list (which got 1.8% bias leak)
  ✓ NEW: explicit ban on "Looking for" template + similar search-intent openers
    (because that's where Stage 12 collapsed to)

This isolates whether the bad_words approach works WITHOUT prompt over-constraint.

Inputs:
  - stage8_5_pool.json
  - stage8_5_selection.json
Outputs:
  - stage13_hybrid_pool.json
  - stage13_comparison.json
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

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_IN = SCRATCH / "stage8_5_pool.json"

OUTPUT_POOL = SCRATCH / "stage13_hybrid_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage13_comparison.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL", "COORDINATION", "RELATIVE_CLAUSE",
    "MODIFIER_FRONTED", "PREDICATE", "QUESTION", "SIMPLE",
]

# Tier 3: forbidden semantic tokens (Stage 12 blacklist + NEW: template openers)
FORBIDDEN_WORDS = [
    # adjectives (Stage 12 list)
    "durable", "lightweight", "comfortable", "perfect", "modern",
    "stylish", "premium", "high-quality", "top-quality", "quality",
    "ideal", "great", "excellent", "best", "amazing",
    "soft", "smooth", "gentle", "cute", "adorable", "lovely",
    "safe", "safer", "safety", "reliable", "trustworthy",
    "beautiful", "elegant", "sleek", "unique", "innovative",
    "luxury", "luxurious", "deluxe", "classic", "traditional",
    "better", "worse", "superior", "inferior",
    "ultimate", "supreme", "extraordinary", "exceptional",
    "organic", "natural", "eco-friendly", "sustainable",
    "professional", "expert", "advanced", "premier",
    "for baby", "for your baby", "for babies", "for the baby",
    # NEW: template collapse prevention (Stage 13 lesson)
    "Looking for", "looking for", "Find the right", "Find the",
    "I need", "I want", "I'm looking", "I am looking",
    "Where can I find", "where can I find", "Where to find",
    "Need to find", "Want to find", "Searching for",
]

SENTENCE_STARTERS = {"I", "I'm", "Looking", "Searching", "Find", "Need", "Want",
                     "Here", "There", "Get", "Buy"}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_structural(query: str, _=None) -> str:
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
    """Stage 11's freeform prompts (no whitelist vocabulary)."""
    prompts = {
        "PREPOSITIONAL": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST end with a prepositional phrase like 'in {category}'. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium, high-quality, top-quality, quality, ideal, great, excellent, best, amazing.",
        "COORDINATION": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST use 'and' to coordinate two noun phrases or verb phrases. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        "RELATIVE_CLAUSE": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST contain a relative clause starting with 'that' or 'which'. Keep queries under 25 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        "MODIFIER_FRONTED": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST start with a fronted modifier (e.g., 'For {age}, ...' or 'In {category}, ...') followed by a comma. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        "PREDICATE": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST use copula 'is' (e.g., 'A {brand} product is ...'). Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        "QUESTION": f"Generate 4 search queries for {brand} {color} {age} product in {category} category phrased as questions ending with '?'. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        "SIMPLE": f"Generate 4 simple search queries for {brand} {color} {age} product in {category} category without using relative clauses, coordination, or fronted modifiers. Keep queries under 15 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
    }
    return prompts[cls]


def main():
    log("=== Stage 13 — Hybrid: Stage 11 prompts + Stage 12 bad_words ===")

    from llm_client import QwenLocalClient
    from vllm import SamplingParams

    pool = json.load(open(POOL_IN))
    pools = pool["pools"]
    sorted_asins = sorted(pools.keys(), key=lambda a: -len(pools[a]))[:20]

    log("  initializing QwenLocalClient...")
    client = QwenLocalClient()
    tokenizer = client._backend.tokenizer
    model = client._backend.model

    sampling = SamplingParams(
        max_tokens=80, temperature=0.8,
        top_p=0.95 if 0.8 > 0 else 1.0,
        repetition_penalty=1.05,
        bad_words=FORBIDDEN_WORDS,  # HARD CONSTRAINT (Stage 12 lesson)
    )
    log(f"  bad_words count: {len(FORBIDDEN_WORDS)} (includes Looking for/Find/I need anti-template)")
    log(f"  prompts: Stage 11 freeform (each asks for 4 variants × 7 classes × 20 ASINs = 560)")

    content_locked_pool = {}
    bias_hits_total = 0
    coverage_total = 0
    n_total_outputs = 0
    n_nonempty = 0

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

        # Stage 11 style: 4 variants per class × 7 classes = 28 prompts
        # but each prompt asks for 4 generations in one vLLM call
        # so per ASIN: 7 class prompts × 4 K=4 batches via variant suffix
        all_classes = list(STRUCTURAL_CLASSES)
        all_prompts = []
        for cls in all_classes:
            for k in range(4):  # 4 variants per class for diversity
                p = build_class_prompt(brand, color, age, category, cls) + f" (variant {k+1})"
                all_prompts.append(p)

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

        asin_locked = {}
        n_cands = 0
        for ci, cls in enumerate(all_classes):
            generated = []
            for k in range(4):
                idx = ci * 4 + k
                out = outputs[idx] if idx < len(outputs) else ""
                bias_hit = any(b in out for b in FORBIDDEN_WORDS)  # case-sensitive now (Looking vs looking)
                content_ok = all(c.lower() in out.lower() for c in [brand, color, age])
                if bias_hit:
                    bias_hits_total += 1
                if content_ok:
                    coverage_total += 1
                if out:
                    n_nonempty += 1
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
    log(f"  Non-empty: {n_nonempty}/{n_total_outputs} ({n_nonempty/max(n_total_outputs,1)*100:.2f}%)")

    # Per-class hit rate
    log("\n=== Per-class hit rate (Stage 13) ===")
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

    # Pool diversity
    n_classes_per_asin = []
    for asin, data in content_locked_pool.items():
        c = Counter()
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                if g.get("text"):
                    c[g.get("predicted_class", cls)] += 1
        n_classes_per_asin.append(len(c))
    log(f"\n  Pool classes/asin: mean={np.mean(n_classes_per_asin):.2f}, range={sorted(set(n_classes_per_asin))}")

    # Save pool
    with open(OUTPUT_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 13: Stage 11 freeform prompts + Stage 12 bad_words hard constraint",
                "n_asins": len(content_locked_pool),
                "structural_classes": STRUCTURAL_CLASSES,
                "forbidden_words_count": len(FORBIDDEN_WORDS),
            },
            "stats": {
                "bias_hits": bias_hits_total,
                "content_coverage": coverage_total,
                "total_outputs": n_total_outputs,
                "non_empty": n_nonempty,
            },
            "pool": content_locked_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_POOL}")

    # Comparison vs S11 + S12
    log("\n=== Comparison vs S11 + S12 ===")
    s11 = json.load(open(SCRATCH / "stage11_content_locked_pool.json"))
    s12 = json.load(open(SCRATCH / "stage12_logit_masked_pool.json"))

    def per_class_hit(pool_data):
        ch = Counter()
        ct = Counter()
        for asin, data in pool_data["pool"].items():
            for cls, info in data["by_class"].items():
                for g in info["generated"]:
                    if g.get("text"):
                        pred = g.get("predicted_class", cls)
                        ct[cls] += 1
                        if pred == cls:
                            ch[cls] += 1
        return ch, ct

    s11_ch, s11_ct = per_class_hit(s11)
    s12_ch, s12_ct = per_class_hit(s12)

    log(f"  {'Class':<18} {'S11 rate':>10} {'S12 rate':>10} {'S13 rate':>10} {'S13-S11':>10}")
    for cls in STRUCTURAL_CLASSES:
        s11_r = s11_ch[cls] / max(s11_ct[cls], 1)
        s12_r = s12_ch[cls] / max(s12_ct[cls], 1)
        s13_r = class_hits[cls] / max(class_total[cls], 1)
        log(f"  {cls:<18} {s11_r:>10.1%} {s12_r:>10.1%} {s13_r:>10.1%} {(s13_r - s11_r)*100:>+9.1f}pp")

    s11_b = s11["stats"]["bias_hits"] / 560
    s12_b = s12["stats"]["bias_hits"] / 560
    s13_b = bias_hits_total / max(n_total_outputs, 1)
    log(f"\n  {'Metric':<25} {'S11':>10} {'S12':>10} {'S13':>10}")
    log(f"  {'-'*60}")
    log(f"  {'bias_hit_rate':<25} {s11_b:>10.1%} {s12_b:>10.1%} {s13_b:>10.1%}")
    s11_c = s11["stats"]["content_coverage"] / 560
    s12_c = s12["stats"]["content_coverage"] / 560
    s13_c = coverage_total / max(n_total_outputs, 1)
    log(f"  {'content_coverage':<25} {s11_c:>10.1%} {s12_c:>10.1%} {s13_c:>10.1%}")

    # Pool diversity
    def pool_classes(pool_data):
        out = []
        for asin, data in pool_data["pool"].items():
            c = Counter()
            for cls, info in data["by_class"].items():
                for g in info["generated"]:
                    if g.get("text"):
                        c[g.get("predicted_class", cls)] += 1
            out.append(len(c))
        return np.mean(out)

    s11_div = pool_classes(s11)
    s12_div = pool_classes(s12)
    s13_div = np.mean(n_classes_per_asin)
    log(f"  {'pool_classes/asin':<25} {s11_div:>10.2f} {s12_div:>10.2f} {s13_div:>10.2f}")

    # Save comparison
    comparison = {
        "config": {
            "description": "Stage 13 hybrid: Stage 11 prompts + Stage 12 bad_words",
            "forbidden_words_count": len(FORBIDDEN_WORDS),
            "forbidden_words_added": [
                "Looking for", "looking for", "Find the right", "I need",
                "I want", "I'm looking", "Where can I find", "Searching for",
            ],
        },
        "stage13": {
            "bias_hits": bias_hits_total,
            "content_coverage": coverage_total,
            "total_outputs": n_total_outputs,
            "non_empty": n_nonempty,
            "bias_hit_rate": s13_b,
            "content_coverage_rate": s13_c,
            "pool_classes_per_asin": s13_div,
            "per_class_hit_rate": {
                cls: class_hits[cls] / max(class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
        },
        "stage11": {"bias_hit_rate": s11_b, "content_coverage_rate": s11_c, "pool_classes_per_asin": s11_div},
        "stage12": {"bias_hit_rate": s12_b, "content_coverage_rate": s12_c, "pool_classes_per_asin": s12_div},
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_COMPARISON}")

    log("\n=== Stage 13 complete ===")


if __name__ == "__main__":
    main()