"""Stage 14 — bad_words + 4/4 attribute coverage + post-gen semantic filter.

User's critique of Stage 13:
1. ✅ bad_words IS hard constraint (Stage 12/13)
2. ⚠️ bad_words is tokenization-sensitive (Best-selling bypasses 'best')
3. ⚠️ Stage 13 content_ok only checks brand+color+age, NOT category
4. ⚠️ Stage 13 is "blacklist constraint", not full content whitelist

Stage 14 implements the user's recommended architecture:
  bad_words (decoding-time hard mask)
  + 4/4 attribute hard coverage check (brand + color + age + category)
  + post-generation semantic filter (drop residual biased queries)

Goal: produce a clean pool where every query satisfies
  4/4 attributes preserved + 0 unsupported semantic content.

Inputs:
  - stage13_hybrid_pool.json (Stage 13 output)
Outputs:
  - stage14_clean_pool.json (filtered pool)
  - stage14_comparison.json (S13 → S14 metrics)
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

INPUT_POOL = SCRATCH / "stage13_hybrid_pool.json"
OUTPUT_POOL = SCRATCH / "stage14_clean_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage14_comparison.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL", "COORDINATION", "RELATIVE_CLAUSE",
    "MODIFIER_FRONTED", "PREDICATE", "QUESTION", "SIMPLE",
]

# Tier 3: expanded semantic blacklist for post-generation filter.
# Includes inflection / hyphenation / synonym variants that bypass bad_words.
FORBIDDEN_SEMANTIC = [
    # Direct adjectives (Stage 13 base)
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
    # NEW: inflection variants (Stage 14 expansion)
    "safest", "trusted", "trust", "reassuring",
    "softness", "softly",
    "gentlest", "gentleness",
    "stylishly", "stylish",
    "innovation",
    "perfectly", "perfection",
    "modernly", "modernity",
    "premium", "premium-grade", "premium-quality",
    # NEW: hyphenated variants
    "best-selling", "best-selling", "Best-selling", "BEST-selling",
    "high-rated", "high-end", "top-tier", "world-class",
    "industry-leading", "well-known", "long-lasting",
    "eco-conscious", "eco-safe", "baby-safe",
    # NEW: superlatives / hyperbole
    "amazingly", "incredibly", "extremely",
    # NEW: marketing fluff
    "breathable", "hypoallergenic", "non-toxic",
    "waterproof", "water-resistant", "leak-proof",
    # NEW: search-intent templates (Stage 13 anti-template)
    "Looking for", "looking for", "Find the right", "Find the",
    "I need", "I want", "I'm looking", "I am looking",
    "Where can I find", "where can I find", "Where to find",
    "Need to find", "Want to find", "Searching for",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_structural(query: str, _=None) -> str:
    """Same regex classifier as Stage 11/13."""
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


def has_bias_substring(text: str) -> tuple[bool, str]:
    """Check if text contains any forbidden semantic token (case-insensitive substring)."""
    low = text.lower()
    for b in FORBIDDEN_SEMANTIC:
        if b.lower() in low:
            return True, b
    return False, ""


def check_4_of_4_attrs(text: str, brand: str, color: str, age: str, category: str) -> tuple[bool, list]:
    """Check 4/4 attribute hard coverage. Returns (ok, missing_attrs)."""
    low = text.lower()
    missing = []
    if brand and brand.lower() not in low:
        missing.append("brand")
    if color and color.lower() not in low:
        missing.append("color")
    if age and age.lower() not in low:
        missing.append("age")
    if category and category.lower() not in low:
        missing.append("category")
    return (len(missing) == 0), missing


def main():
    log("=== Stage 14 — bad_words + 4/4 attrs + post-gen filter ===")

    # Load Stage 13 pool
    log(f"  loading Stage 13 pool: {INPUT_POOL}")
    s13 = json.load(open(INPUT_POOL))
    s13_pool = s13["pool"]
    log(f"  Stage 13: {len(s13_pool)} ASINs, "
        f"{sum(len(info['generated']) for d in s13_pool.values() for info in d['by_class'].values())} queries")

    # Apply 4/4 attribute + bias post-filter
    log("\n=== Step 1: Per-query evaluation (4/4 attrs + expanded bias list) ===")

    stats_per_query = []  # list of dicts
    n_total = 0
    n_attr3 = 0  # brand+color+age (Stage 13 standard)
    n_attr4 = 0  # brand+color+age+category (Stage 14 upgrade)
    n_bias = 0
    n_attr4_no_bias = 0
    pool_size_per_asin = {}

    for asin, data in s13_pool.items():
        brand, color, age, category = data["attrs"]
        asin_total = 0
        asin_attr4_clean = 0
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                text = g.get("text", "").strip()
                n_total += 1
                if not text:
                    stats_per_query.append({
                        "asin": asin, "cls": cls, "text": text,
                        "attr3_ok": False, "attr4_ok": False,
                        "bias_hit": False, "keep": False,
                    })
                    continue
                asin_total += 1
                # 3/4 check (Stage 13 style)
                attr3_ok = all(c.lower() in text.lower() for c in [brand, color, age])
                # 4/4 check (Stage 14 upgrade)
                attr4_ok, missing = check_4_of_4_attrs(text, brand, color, age, category)
                # Bias check (expanded list, case-insensitive)
                bias_hit, hit_word = has_bias_substring(text)
                if attr3_ok:
                    n_attr3 += 1
                if attr4_ok:
                    n_attr4 += 1
                if bias_hit:
                    n_bias += 1
                keep = attr4_ok and not bias_hit
                if keep:
                    n_attr4_no_bias += 1
                    asin_attr4_clean += 1
                stats_per_query.append({
                    "asin": asin, "cls": cls, "text": text,
                    "attr3_ok": attr3_ok, "attr4_ok": attr4_ok,
                    "missing_attrs": missing,
                    "bias_hit": bias_hit, "bias_word": hit_word,
                    "keep": keep,
                })
        pool_size_per_asin[asin] = {"total": asin_total, "clean": asin_attr4_clean}

    log(f"  total queries: {n_total}")
    log(f"  non-empty: {sum(1 for s in stats_per_query if s['text'])}")
    log(f"  3/4 attr coverage (Stage 13 metric): {n_attr3}/{n_total} ({n_attr3/n_total*100:.2f}%)")
    log(f"  4/4 attr coverage (Stage 14 metric): {n_attr4}/{n_total} ({n_attr4/n_total*100:.2f}%)")
    log(f"  bias hit (expanded list, case-insens): {n_bias}/{n_total} ({n_bias/n_total*100:.2f}%)")
    log(f"  KEEP (4/4 attrs + no bias): {n_attr4_no_bias}/{n_total} ({n_attr4_no_bias/n_total*100:.2f}%)")

    # Top missed attrs
    log("\n=== Top missing attributes (among 4/4 attr failures) ===")
    attr_miss_counter = Counter()
    for s in stats_per_query:
        if s.get("missing_attrs"):
            for m in s["missing_attrs"]:
                attr_miss_counter[m] += 1
    for attr, count in attr_miss_counter.most_common():
        log(f"  {attr}: {count}")

    # Top bias hits
    log("\n=== Top bias word hits ===")
    bias_counter = Counter()
    for s in stats_per_query:
        if s.get("bias_word"):
            bias_counter[s["bias_word"]] += 1
    for word, count in bias_counter.most_common(15):
        log(f"  {word!r}: {count}")

    # === Step 2: Build clean pool ===
    log("\n=== Step 2: Building clean pool (4/4 attrs + no bias) ===")
    clean_pool = {}
    for asin, data in s13_pool.items():
        clean_pool[asin] = {
            "attrs": data["attrs"],
            "by_class": {},
        }
        for cls, info in data["by_class"].items():
            kept = []
            for g, s in zip(info["generated"], stats_per_query):
                # Need to lookup the per-query stat by cls+text
                pass
            # Use a different approach: rebuild from stats_per_query
            cls_stats = [s for s in stats_per_query if s["asin"] == asin and s["cls"] == cls]
            kept = []
            for g, s in zip(info["generated"], cls_stats):
                if s["keep"]:
                    kept.append(g)
            clean_pool[asin]["by_class"][cls] = {
                "prompt": info["prompt"],
                "generated": kept,
            }

    # Per-class hit rate on clean pool
    log("\n=== Per-class hit rate on CLEAN pool ===")
    clean_class_hits = Counter()
    clean_class_total = Counter()
    for asin, data in clean_pool.items():
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                if g.get("text"):
                    pred = g.get("predicted_class", cls)
                    clean_class_total[cls] += 1
                    if pred == cls:
                        clean_class_hits[cls] += 1
    log(f"  {'Class':<18} {'hits':>6} {'total':>6} {'rate':>6}")
    for cls in STRUCTURAL_CLASSES:
        h = clean_class_hits[cls]
        t = clean_class_total[cls]
        rate = h / t if t > 0 else 0
        log(f"  {cls:<18} {h:>6} {t:>6} {rate:>6.1%}")

    # Pool diversity on clean pool
    n_classes_per_asin = []
    for asin, data in clean_pool.items():
        c = Counter()
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                if g.get("text"):
                    c[g.get("predicted_class", cls)] += 1
        n_classes_per_asin.append(len(c))
    log(f"\n  Pool classes/asin (clean): mean={np.mean(n_classes_per_asin):.2f}, "
        f"median={np.median(n_classes_per_asin):.1f}, "
        f"min={min(n_classes_per_asin)}, max={max(n_classes_per_asin)}")

    # Per-ASIN clean pool size
    log(f"\n  Clean pool size per ASIN: "
        f"mean={np.mean([v['clean'] for v in pool_size_per_asin.values()]):.1f}, "
        f"min={min(v['clean'] for v in pool_size_per_asin.values())}, "
        f"max={max(v['clean'] for v in pool_size_per_asin.values())}")
    log(f"  Total clean queries: {sum(v['clean'] for v in pool_size_per_asin.values())}")

    # Save clean pool
    with open(OUTPUT_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 14: bad_words (Stage 13) + 4/4 attrs + post-gen semantic filter",
                "forbidden_words_count": len(FORBIDDEN_SEMANTIC),
                "filter_criteria": "4/4 attributes AND no bias substring (case-insens)",
            },
            "stats": {
                "input_pool_size": n_total,
                "non_empty": sum(1 for s in stats_per_query if s["text"]),
                "attr3_coverage": n_attr3,
                "attr4_coverage": n_attr4,
                "bias_hits": n_bias,
                "kept_after_filter": n_attr4_no_bias,
                "filter_pass_rate": n_attr4_no_bias / max(n_total, 1),
            },
            "pool": clean_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {OUTPUT_POOL}")

    # === Comparison S13 → S14 ===
    log("\n=== Stage 13 → Stage 14 comparison ===")
    log(f"  {'Metric':<25} {'S13':>10} {'S14':>10} {'Δ':>10}")
    log(f"  {'-'*60}")
    log(f"  {'total_queries':<25} {n_total:>10} {n_attr4_no_bias:>10} "
        f"{(n_attr4_no_bias - n_total):>+10}")
    s13_bias_rate = s13["stats"]["bias_hits"] / max(s13["stats"]["total_outputs"], 1)
    s14_bias_rate = 0.0  # by construction (post-filter drops biased)
    log(f"  {'bias_hit_rate':<25} {s13_bias_rate:>10.1%} {s14_bias_rate:>10.1%} "
        f"{(s14_bias_rate - s13_bias_rate)*100:>+9.1f}pp")
    s13_attr3_rate = s13["stats"]["content_coverage"] / max(s13["stats"]["total_outputs"], 1)
    log(f"  {'3/4 attr coverage':<25} {s13_attr3_rate:>10.1%} {n_attr4/max(n_total,1):>10.1%}")
    s14_attr4_rate = n_attr4 / max(n_total, 1)
    log(f"  {'4/4 attr coverage':<25} {'(N/A)':>10} {s14_attr4_rate:>10.1%}")

    # Save comparison
    comparison = {
        "config": {
            "description": "Stage 14: post-gen filter on Stage 13 output",
            "forbidden_words_count": len(FORBIDDEN_SEMANTIC),
            "filter": "4/4 attrs (brand+color+age+category) AND no bias substring (case-insens)",
        },
        "stage14": {
            "input_total": n_total,
            "non_empty": sum(1 for s in stats_per_query if s["text"]),
            "attr3_ok_count": n_attr3,
            "attr4_ok_count": n_attr4,
            "bias_hits_count": n_bias,
            "kept_count": n_attr4_no_bias,
            "filter_pass_rate": n_attr4_no_bias / max(n_total, 1),
            "attr4_coverage_rate": s14_attr4_rate,
            "bias_hit_rate": s14_bias_rate,
            "pool_classes_per_asin_mean": float(np.mean(n_classes_per_asin)),
            "per_class_hit_rate": {
                cls: clean_class_hits[cls] / max(clean_class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
            "top_bias_hits": dict(bias_counter.most_common(15)),
            "top_missing_attrs": dict(attr_miss_counter.most_common()),
        },
        "stage13": {
            "total_outputs": s13["stats"]["total_outputs"],
            "bias_hits": s13["stats"]["bias_hits"],
            "content_coverage_3of4": s13["stats"]["content_coverage"],
            "bias_hit_rate": s13_bias_rate,
            "attr3_coverage_rate": s13_attr3_rate,
        },
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_COMPARISON}")

    log("\n=== Stage 14 complete ===")


if __name__ == "__main__":
    main()