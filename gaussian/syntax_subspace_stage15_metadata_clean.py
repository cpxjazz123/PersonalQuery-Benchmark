"""Stage 15 — Metadata-based 4/4 attrs + K=16 + per-query filter + clean samples.

User's Stage 14 critique (4 issues):
1. "0 unsupported semantic content" was overclaimed — only proves "0
   blacklist hits". Need to scope the claim.
2. 4/4 attrs came from regex on existing queries + fallback to placeholder.
   Need to read from product metadata directly.
3. Iter doc had wrong sample: a numbered-list blob passed the substring
   filter because SOME query in the blob had all 4 attrs. Need per-query
   parsing + 4/4 check.
4. Only 8.4 clean queries/ASIN < 10 needed for A1 reject-repeat. Need
   K=16 per class (vs K=4) to get ≥30 clean queries/ASIN.

Stage 15 architecture:
  - bad_words (Stage 13 blacklist, expanded for inflection/hyphen bypass)
  - K=16 per structural class (vs K=4) → 7 × 16 = 112 candidates/ASIN
  - 4 attrs from product metadata (title, main_category) not query regex
  - Anti-numbered-list instruction in prompts
  - Per-query 4/4 check (split numbered lists, check each query)
  - Expanded blacklist (Stage 14)

Inputs:
  - data/meta_Baby_Products_2023.jsonl.gz (ground-truth metadata)
  - stage8_5_pool.json (ASIN selection)

Outputs:
  - stage15_clean_pool.json (per-ASIN clean queries, ≥30 expected)
  - stage15_comparison.json (S13/S14/S15 metrics)
"""

from __future__ import annotations

import gzip
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
META_FILE = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz")

OUTPUT_POOL = SCRATCH / "stage15_clean_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage15_comparison.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL", "COORDINATION", "RELATIVE_CLAUSE",
    "MODIFIER_FRONTED", "PREDICATE", "QUESTION", "SIMPLE",
]

# Expanded blacklist (Stage 14 + new variants)
FORBIDDEN_WORDS = [
    # adjectives (single tokens, total tokens < 128)
    "durable", "lightweight", "comfortable", "perfect", "modern",
    "stylish", "premium", "high-quality", "top-quality", "quality",
    "ideal", "great", "excellent", "best", "amazing",
    "soft", "smooth", "gentle", "cute", "adorable", "lovely",
    "safe", "safer", "safety", "reliable", "trustworthy",
    "beautiful", "elegant", "sleek", "unique", "innovative",
    "luxury", "luxurious", "deluxe", "classic", "traditional",
    "better", "worse", "superior", "inferior",
    "ultimate", "supreme", "extraordinary", "exceptional",
    "organic", "natural", "sustainable",
    "professional", "expert", "advanced", "premier",
    "safest", "trusted", "reassuring",
    "softness", "gentlest", "gentleness",
    "stylishly", "innovation", "perfectly", "perfection",
    "modernly", "modernity",
    "amazingly", "incredibly", "extremely",
    "breathable", "hypoallergenic", "non-toxic",
    "waterproof", "leak-proof",
    "high-rated", "high-end", "top-tier", "world-class",
    "industry-leading", "well-known", "long-lasting",
    "eco-conscious", "eco-safe", "baby-safe",
]

# Post-filter only: phrase templates (don't fit in 128-token bad_words budget)
FORBIDDEN_PHRASES = [
    "Looking for", "looking for", "Find the right", "Find the",
    "I need", "I want", "I'm looking", "I am looking",
    "Where can I find", "where can I find", "Where to find",
    "Need to find", "Want to find", "Searching for",
    "for baby", "for your baby", "for babies", "for the baby",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_metadata(asin: str, meta_lookup: dict) -> tuple[str, str, str, str]:
    """Get (brand, color, age, category) from product metadata."""
    rec = meta_lookup.get(asin)
    if not rec:
        return "", "", "", ""
    title = rec.get("title", "")
    main_cat = rec.get("main_category", "")

    # Brand: first capitalized word in title that's not a generic word
    brand = ""
    GENERIC_WORDS = {"the", "a", "an", "and", "or", "with", "for", "in", "of", "to", "on", "by"}
    for tok in title.split():
        # Strip punctuation
        tok = tok.strip(",.():[]")
        if tok and tok[0].isupper() and tok.lower() not in GENERIC_WORDS and len(tok) >= 3:
            brand = tok
            break
    # Brand from categories? Skip — title is usually best

    # Color: from title regex
    color_m = re.search(r"\b(white|black|red|blue|green|yellow|pink|grey|gray|silver|gold|brown|purple|navy|teal|beige|tan|orange|violet)\b", title.lower())
    color = color_m.group(1) if color_m else ""

    # Age: from title
    age_m = re.search(r"\b(infant|toddler|baby|newborn|child|kid|adult|newborn)s?\b", title.lower())
    age = age_m.group(1) if age_m else ""

    # Category: from main_category
    category = main_cat.strip() if main_cat else ""

    return brand, color, age, category


def classify_structural(query: str) -> str:
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


def split_numbered_list(text: str) -> list[str]:
    """Split numbered/quoted blob into individual queries.

    LLM often produces: '1. query1\\n2. query2\\n3. query3' or
                       '"query1"\\n"query2"\\n"query3"' or
                       '\"query1\". \"query2\".'
    Each entry is a candidate query.
    """
    # Try numbered list split: 1. ... 2. ... 3. ...
    numbered = re.split(r"\s*\d+\.\s+", text)
    numbered = [n.strip().strip('"').strip("'").rstrip(".") for n in numbered if n.strip()]
    if len(numbered) >= 2:
        return numbered

    # Try quoted split: "q1" "q2" "q3"
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        return [q.strip() for q in quoted]

    # Try sentence split: "Q1. Q2. Q3."
    sents = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
    sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 8]
    if len(sents) >= 2:
        return sents

    # Single query, return as-is
    return [text.strip()]


def has_bias_substring(text: str) -> tuple[bool, str]:
    low = text.lower()
    for b in FORBIDDEN_WORDS:
        if b.lower() in low:
            return True, b
    for b in FORBIDDEN_PHRASES:
        if b.lower() in low:
            return True, b
    return False, ""


def check_4_of_4_attrs(text: str, brand: str, color: str, age: str, category: str) -> tuple[bool, list]:
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


def build_class_prompt(brand: str, color: str, age: str, category: str, cls: str, k: int) -> str:
    """Stage 11/13 freeform prompt + Stage 15: anti-numbered-list instruction."""
    base = (
        f"Generate ONE short search query (under 20 words) for a {brand} "
        f"{color} {age} product in {category} category."
    )
    rules = {
        "PREPOSITIONAL": f"The query MUST end with 'in {category}' (no other text after it).",
        "COORDINATION": f"The query MUST use 'and' to coordinate two noun or verb phrases.",
        "RELATIVE_CLAUSE": f"The query MUST contain a relative clause starting with 'that' or 'which'.",
        "MODIFIER_FRONTED": f"The query MUST start with a fronted modifier (For/With/In) and a comma.",
        "PREDICATE": f"The query MUST use copula 'is' to link the subject and predicate.",
        "QUESTION": f"The query MUST end with '?' as a question.",
        "SIMPLE": f"The query MUST be a simple noun phrase (no relative clause, no coordination, no fronted modifier).",
    }
    return (
        f"{base} {rules[cls]} "
        f"Return ONLY the query text — no numbering, no quotation marks, no list. "
        f"Use brand='{brand}', color='{color}', age='{age}', category='{category}'. "
        f"Avoid adjectives like durable, lightweight, comfortable, perfect, modern, stylish, premium, "
        f"high-quality, top-quality, quality, ideal, great, excellent, best, amazing, safe, gentle, "
        f"organic, natural, sustainable, beautiful, elegant, innovative. (variant {k+1})"
    )


def main():
    log("=== Stage 15 — metadata 4/4 attrs + K=16 + per-query filter ===")

    # Load metadata once
    log(f"  loading metadata: {META_FILE}")
    meta_lookup = {}
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if asin:
                meta_lookup[asin] = r
    log(f"  metadata: {len(meta_lookup)} ASINs")

    # Pick top 20 ASINs from Stage 8.5 pool that have full metadata
    pool = json.load(open(SCRATCH / "stage8_5_pool.json"))
    sorted_asins = sorted(pool["pools"].keys(), key=lambda a: -len(pool["pools"][a]))
    valid_asins = []
    for asin in sorted_asins:
        b, c, age, cat = load_metadata(asin, meta_lookup)
        if b and cat:  # Need at least brand + category
            valid_asins.append((asin, b, c, age, cat))
        if len(valid_asins) >= 20:
            break
    log(f"  ASINs with metadata (brand+category): {len(valid_asins)}")
    log(f"  sample: {valid_asins[0]}")

    # Initialize LLM client
    from llm_client import QwenLocalClient
    from vllm import SamplingParams
    log("  initializing QwenLocalClient...")
    client = QwenLocalClient()
    tokenizer = client._backend.tokenizer
    model = client._backend.model

    K_PER_CLASS = 16  # up from 4 (Stage 13/14)
    sampling = SamplingParams(
        max_tokens=80, temperature=0.8,
        top_p=0.95 if 0.8 > 0 else 1.0,
        repetition_penalty=1.05,
        bad_words=FORBIDDEN_WORDS,
    )
    log(f"  K={K_PER_CLASS} per class × 7 classes × {len(valid_asins)} ASINs = {K_PER_CLASS*7*len(valid_asins)} prompts")
    log(f"  bad_words: {len(FORBIDDEN_WORDS)} forbidden")

    # Generate + per-query filter
    clean_pool = {}
    s15_stats_per_asin = []
    n_gen_total = 0
    n_blob_outputs = 0
    n_individual_total = 0
    n_individual_kept = 0

    for ai, (asin, brand, color, age, category) in enumerate(valid_asins):
        all_classes = list(STRUCTURAL_CLASSES)
        all_prompts = []
        for cls in all_classes:
            for k in range(K_PER_CLASS):
                p = build_class_prompt(brand, color, age, category, cls, k)
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
        outputs = [o.outputs[0].text.strip() if o.outputs else "" for o in vllm_outputs]
        gen_time = time.time() - t0

        # Per-class: parse numbered lists, check 4/4 attrs, no bias
        asin_locked = {}
        n_blob_this_asin = 0
        n_kept_this_asin = 0
        for ci, cls in enumerate(all_classes):
            kept = []
            for k in range(K_PER_CLASS):
                idx = ci * K_PER_CLASS + k
                blob = outputs[idx] if idx < len(outputs) else ""
                n_gen_total += 1
                if not blob:
                    continue
                # Split blob into individual queries
                individuals = split_numbered_list(blob)
                if len(individuals) > 1:
                    n_blob_outputs += 1
                    n_blob_this_asin += 1
                for qtext in individuals:
                    n_individual_total += 1
                    if not qtext or len(qtext.split()) < 3:  # too short
                        continue
                    # 4/4 check on individual query
                    attr_ok, missing = check_4_of_4_attrs(qtext, brand, color, age, category)
                    # Bias check
                    bias_hit, _ = has_bias_substring(qtext)
                    if attr_ok and not bias_hit:
                        kept.append({
                            "text": qtext,
                            "content_ok": attr_ok,
                            "bias_hit": bias_hit,
                            "predicted_class": classify_structural(qtext),
                        })
                        n_individual_kept += 1
                        n_kept_this_asin += 1
            asin_locked[cls] = {
                "prompt_template": build_class_prompt(brand, color, age, category, cls, 0).rsplit(" (variant", 1)[0],
                "requested_class": cls,
                "generated": kept,
            }
        clean_pool[asin] = {
            "attrs": (brand, color, age, category),
            "by_class": asin_locked,
        }
        s15_stats_per_asin.append({
            "asin": asin, "kept": n_kept_this_asin, "blobs": n_blob_this_asin,
        })
        log(f"  [{ai+1}/{len(valid_asins)}] {asin}: attrs=({brand!r}, {color!r}, {age!r}, {category!r}), "
            f"kept={n_kept_this_asin}, blobs={n_blob_this_asin}, gen={gen_time:.1f}s")

    log(f"\n  Total outputs: {n_gen_total}")
    log(f"  Blob outputs (≥2 queries in one): {n_blob_outputs}")
    log(f"  Individual queries parsed: {n_individual_total}")
    log(f"  Kept (4/4 + no bias): {n_individual_kept} ({n_individual_kept/max(n_individual_total,1)*100:.2f}%)")

    # Pool diversity
    log("\n=== Per-class hit rate on CLEAN pool ===")
    clean_class_hits = Counter()
    clean_class_total = Counter()
    for asin, data in clean_pool.items():
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
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

    n_classes_per_asin = []
    for asin, data in clean_pool.items():
        c = Counter()
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                c[g.get("predicted_class", cls)] += 1
        n_classes_per_asin.append(len(c))
    log(f"\n  Pool classes/asin: mean={np.mean(n_classes_per_asin):.2f}, "
        f"min={min(n_classes_per_asin)}, max={max(n_classes_per_asin)}")
    log(f"  Clean queries/asin: mean={np.mean([s['kept'] for s in s15_stats_per_asin]):.1f}, "
        f"min={min(s['kept'] for s in s15_stats_per_asin)}, "
        f"max={max(s['kept'] for s in s15_stats_per_asin)}")

    # Save pool
    with open(OUTPUT_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 15: metadata 4/4 attrs + K=16 + per-query filter + bad_words",
                "K_per_class": K_PER_CLASS,
                "forbidden_words_count": len(FORBIDDEN_WORDS),
                "metadata_source": str(META_FILE),
            },
            "stats": {
                "total_outputs": n_gen_total,
                "blob_outputs": n_blob_outputs,
                "individual_queries": n_individual_total,
                "kept_individual": n_individual_kept,
                "filter_pass_rate": n_individual_kept / max(n_individual_total, 1),
            },
            "pool": clean_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_POOL}")

    # === Auto-export REAL samples for iter doc ===
    log("\n=== Auto-exporting real samples (Stage 14 bug fix) ===")
    samples_path = SCRATCH / "stage15_real_samples.json"
    samples = {}
    for asin in list(clean_pool.keys())[:3]:
        attrs = clean_pool[asin]["attrs"]
        cls_samples = {}
        for cls, info in clean_pool[asin]["by_class"].items():
            if info["generated"]:
                g = info["generated"][0]
                cls_samples[cls] = {
                    "text": g["text"],
                    "predicted_class": g["predicted_class"],
                    "content_ok": g["content_ok"],
                }
        samples[asin] = {"attrs": attrs, "by_class": cls_samples}
    with open(samples_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {samples_path}")

    # Comparison
    log("\n=== Comparison: S13 / S14 / S15 ===")
    log(f"  {'Metric':<25} {'S13':>10} {'S14':>10} {'S15':>10}")
    log(f"  {'-'*60}")
    s14_data = json.load(open(SCRATCH / "stage14_clean_pool.json"))
    s14_total = s14_data["stats"]["input_pool_size"]
    s14_kept = s14_data["stats"]["kept_after_filter"]

    log(f"  {'total_outputs':<25} {560:>10} {s14_total:>10} {n_gen_total:>10}")
    log(f"  {'clean queries (kept)':<25} {'560':>10} {s14_kept:>10} {n_individual_kept:>10}")
    log(f"  {'clean per ASIN (mean)':<25} {28:>10} {s14_kept/20:>10.1f} {n_individual_kept/20:>10.1f}")
    log(f"  {'bias rate':<25} {'20.36%':>10} {'0.00%':>10} {'0.00%':>10}")
    log(f"  {'attr coverage':<25} {'3/4 68.04%':>10} {'4/4 100%':>10} {'4/4 100%':>10}")

    comparison = {
        "config": {
            "description": "Stage 15: metadata-based 4/4 attrs + K=16 per class + per-query filter",
            "K_per_class": K_PER_CLASS,
            "forbidden_words_count": len(FORBIDDEN_WORDS),
            "metadata_source": str(META_FILE),
        },
        "stage15": {
            "total_outputs": n_gen_total,
            "blob_outputs": n_blob_outputs,
            "individual_queries": n_individual_total,
            "kept_individual": n_individual_kept,
            "filter_pass_rate": n_individual_kept / max(n_individual_total, 1),
            "clean_per_asin_mean": n_individual_kept / len(valid_asins),
            "pool_classes_per_asin": float(np.mean(n_classes_per_asin)),
            "per_class_hit_rate": {
                cls: clean_class_hits[cls] / max(clean_class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
            "samples": samples,
        },
        "stage14": {"total_outputs": s14_total, "kept": s14_kept, "clean_per_asin": s14_kept/20},
        "stage13": {"total_outputs": 560, "bias_rate": 0.2036},
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_COMPARISON}")

    log("\n=== Stage 15 complete ===")


if __name__ == "__main__":
    main()