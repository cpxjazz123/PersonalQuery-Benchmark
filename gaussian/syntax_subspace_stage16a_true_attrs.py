"""Stage 16A — True Attribute Source Audit.

User's Stage 15 critique (the "fake 4/4" problem):
- brand was heuristic ("first capitalized non-stopword in title" → "Diapers")
- color was regex on title (often empty)
- age was regex on title
- empty strings caused `check_4_of_4_attrs` to short-circuit (`if color and ...`)
- 14/20 ASINs missing color, 7/20 missing age → 77.2% of "clean" queries
  were fake 4/4 (passed because empty attrs were never checked)
- result: claimed "100% 4/4 coverage" but actual real 4/4 = 22.8%

Stage 16A design (per user proposal):
- Per-product 4-attribute sets, NOT uniform brand/color/age/category across all ASINs
- Candidate attribute types from Amazon metadata details field:
    Brand (67% freq), Color (53%), Item Weight (82%), Material (from features)
- For each ASIN: pick the first 4 non-empty attributes from {Brand, Color,
    Item Weight, Material} in priority order
- Only keep ASINs with exactly 4 non-empty attributes (true 4/4)
- Different ASINs may have different attribute type combinations

Hard constraint:
- All 4 attributes MUST be non-empty
- LLM must include all 4 attribute VALUES in the generated query
- Per-query 4/4 check (no short-circuit on empty)

Architecture:
  Stage 13/15 bad_words hard mask + Stage 14 expanded blacklist + Stage 15
  per-query filter + Stage 16A real 4-attribute enforcement

Inputs:
  - data/meta_Baby_Products_2023.jsonl.gz (ground-truth metadata)
  - stage8_5_pool.json (ASIN selection)

Outputs:
  - stage16a_clean_pool.json (per-ASIN queries with real 4/4 attrs)
  - stage16a_real_samples.json (auto-exported samples)
  - stage16a_comparison.json (S15 vs S16A metrics)
"""

from __future__ import annotations

import gzip
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
META_FILE = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz")

OUTPUT_POOL = SCRATCH / "stage16a_clean_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage16a_comparison.json"
OUTPUT_SAMPLES = SCRATCH / "stage16a_real_samples.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL", "COORDINATION", "RELATIVE_CLAUSE",
    "MODIFIER_FRONTED", "PREDICATE", "QUESTION", "SIMPLE",
]

# Single-token forbidden words (≤128 token budget for vLLM bad_words)
FORBIDDEN_WORDS = [
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

# Phrase templates (post-filter substring scan)
FORBIDDEN_PHRASES = [
    "Looking for", "looking for", "Find the right", "Find the",
    "I need", "I want", "I'm looking", "I am looking",
    "Where can I find", "where can I find", "Where to find",
    "Need to find", "Want to find", "Searching for",
    "for baby", "for your baby", "for babies", "for the baby",
]

# Common material nouns (for Material extraction from features)
MATERIAL_PATTERNS = [
    "Cotton", "Polyester", "Microfiber", "Plush", "Mesh", "Fleece",
    "Muslin", "Plastic", "Bamboo", "Nylon", "Silk", "Wool", "Linen",
    "Organic Cotton", "100% Cotton", "100% Polyester", "Spandex",
    "Velvet", "Satin", "Canvas", "Vinyl", "Rubber", "Silicone",
    "Acrylic", "Rayon", "Latex", "Stainless Steel", "Aluminum",
    "Wood", "BPA Free", "Food Grade", "Glass", "Ceramic",
]

# Garbage indicators (don't accept as attribute values)
GARBAGE_TOKENS = {"", "n/a", "na", "none", "unknown", "-", "—", "?", "n\\a"}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_garbage_value(val: str) -> bool:
    low = val.strip().lower()
    if low in GARBAGE_TOKENS:
        return True
    # Single character or single-digit values
    if len(val.strip()) <= 1:
        return True
    # Pure numbers without units (probably a model number)
    if re.fullmatch(r"\d+(\.\d+)?", val.strip()):
        return True
    return False


def get_4_attrs(asin: str, meta_lookup: dict) -> list[tuple[str, str]] | None:
    """Extract 4 real non-empty attributes from Amazon metadata.

    Returns: list of (attr_name, attr_value) tuples or None if <4 found.
    Attribute priority: Brand > Color > Item Weight > Material (first available).
    """
    rec = meta_lookup.get(asin)
    if not rec:
        return None
    details = rec.get("details", {})
    features = rec.get("features", [])

    attrs = []

    # 1. Brand (from details.Brand)
    brand = details.get("Brand", "").strip()
    if not is_garbage_value(brand):
        attrs.append(("Brand", brand))

    # 2. Color (from details.Color)
    color = details.get("Color", "").strip()
    if not is_garbage_value(color) and color.lower() not in {a[1].lower() for a in attrs}:
        attrs.append(("Color", color))

    # 3. Item Weight (from details.Item Weight)
    weight = details.get("Item Weight", "").strip()
    if not is_garbage_value(weight):
        attrs.append(("Item Weight", weight))

    # 4. Material (from features, first material-like short phrase)
    if len(attrs) < 4:
        for ftr in features:
            fl = ftr.strip().rstrip(".").strip()
            # Material is short, capitalized, no sentence punctuation
            if 1 <= len(fl.split()) <= 4 and len(fl) <= 40:
                # Must be in known material list OR look like a material
                fl_clean = fl.strip()
                if fl_clean in MATERIAL_PATTERNS:
                    if fl_clean.lower() not in {a[1].lower() for a in attrs}:
                        attrs.append(("Material", fl_clean))
                        break

    return attrs if len(attrs) == 4 else None


def has_bias_substring(text: str) -> tuple[bool, str]:
    low = text.lower()
    for b in FORBIDDEN_WORDS:
        if b.lower() in low:
            return True, b
    for b in FORBIDDEN_PHRASES:
        if b.lower() in low:
            return True, b
    return False, ""


def check_4_of_4_attrs(text: str, attrs: list[tuple[str, str]]) -> tuple[bool, list]:
    """Check all 4 attributes (REAL non-empty values) appear in text.

    Unlike Stage 15's check_4_of_4_attrs, this does NOT short-circuit on empty
    values: all 4 attrs are guaranteed non-empty by get_4_attrs.

    Brand matching uses first-2-words for 3+ word brands (natural abbreviation).
    """
    low = text.lower()
    missing = []
    for attr_name, attr_val in attrs:
        if attr_name == "Brand":
            # Brand: use first 2 words for matching if 3+ words
            words = attr_val.split()
            if len(words) >= 3:
                match_val = " ".join(words[:2])
            else:
                match_val = attr_val
        elif attr_name == "Item Weight":
            # Item Weight: use the number + "pound"/"ounce" word (ignore "ounces" plural)
            # Extract numeric part
            num_m = re.search(r"\d+\.?\d*", attr_val)
            unit_m = re.search(r"(pound|ounce|gram|kilogram|kg|lb|oz)\w*", attr_val.lower())
            if num_m and unit_m:
                match_val = f"{num_m.group()} {unit_m.group()}"
            elif num_m:
                match_val = num_m.group()
            else:
                match_val = attr_val
        else:
            match_val = attr_val
        if match_val.lower() not in low:
            missing.append(attr_name)
    return (len(missing) == 0), missing


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
    """Split numbered/quoted blob into individual queries."""
    numbered = re.split(r"\s*\d+\.\s+", text)
    numbered = [n.strip().strip('"').strip("'").rstrip(".") for n in numbered if n.strip()]
    if len(numbered) >= 2:
        return numbered
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        return [q.strip() for q in quoted]
    sents = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
    sents = [s.strip() for s in sents if s.strip() and len(s.strip()) > 8]
    if len(sents) >= 2:
        return sents
    return [text.strip()]


def build_class_prompt(attrs: list[tuple[str, str]], cls: str, k: int) -> str:
    """Build per-product prompt with real 4 attributes."""
    attr_descr = ". ".join(
        f"Use {name}='{val}'" for name, val in attrs
    )
    base = (
        f"Generate ONE short search query (under 25 words) for a product "
        f"with these attributes: {attr_descr}."
    )
    rules = {
        "PREPOSITIONAL": f"The query MUST end with a prepositional phrase (e.g., 'in <material>' or 'for <brand>').",
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
        f"Avoid adjectives like durable, lightweight, comfortable, perfect, modern, "
        f"stylish, premium, best, soft, gentle, organic. (variant {k+1})"
    )


def main():
    log("=== Stage 16A — True Attribute Source Audit ===")

    # Load metadata
    log(f"  loading metadata: {META_FILE}")
    meta_lookup = {}
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if asin:
                meta_lookup[asin] = r
    log(f"  metadata: {len(meta_lookup)} ASINs")

    # Pick ASINs with REAL 4 attributes — search Stage 8.5 pool first,
    # then fall back to full metadata until we have 30
    pool = json.load(open(SCRATCH / "stage8_5_pool.json"))
    candidate_asins = list(pool["pools"].keys())

    valid_asins = []
    attr_type_combos = Counter()
    for asin in candidate_asins:
        attrs = get_4_attrs(asin, meta_lookup)
        if attrs and len(attrs) == 4:
            valid_asins.append((asin, attrs))
            combo = "+".join(sorted([a[0] for a in attrs]))
            attr_type_combos[combo] += 1
    log(f"  Stage 8.5 ASINs with 4 real attrs: {len(valid_asins)}")

    # If <30, pull from full metadata
    if len(valid_asins) < 30:
        log(f"  pulling more from full metadata (have {len(valid_asins)}/30)...")
        for asin in meta_lookup:
            if asin in {a[0] for a in valid_asins}:
                continue
            attrs = get_4_attrs(asin, meta_lookup)
            if attrs and len(attrs) == 4:
                valid_asins.append((asin, attrs))
                combo = "+".join(sorted([a[0] for a in attrs]))
                attr_type_combos[combo] += 1
            if len(valid_asins) >= 30:
                break
    log(f"  ASINs with 4 real non-empty attrs: {len(valid_asins)}")
    log(f"  Attribute-type combinations:")
    for combo, cnt in attr_type_combos.most_common():
        log(f"    {combo}: {cnt}")
    log(f"  sample: {valid_asins[0]}")

    # Initialize LLM client
    from llm_client import QwenLocalClient
    from vllm import SamplingParams
    log("  initializing QwenLocalClient...")
    client = QwenLocalClient()
    tokenizer = client._backend.tokenizer
    model = client._backend.model

    K_PER_CLASS = 16
    sampling = SamplingParams(
        max_tokens=80, temperature=0.8,
        top_p=0.95 if 0.8 > 0 else 1.0,
        repetition_penalty=1.05,
        bad_words=FORBIDDEN_WORDS,
    )
    log(f"  K={K_PER_CLASS} per class × 7 classes × {len(valid_asins)} ASINs "
        f"= {K_PER_CLASS*7*len(valid_asins)} prompts")

    # Generate + per-query filter
    clean_pool = {}
    s16a_stats_per_asin = []
    n_gen_total = 0
    n_blob_outputs = 0
    n_individual_total = 0
    n_individual_kept = 0
    n_attr_miss_total = 0

    for ai, (asin, attrs) in enumerate(valid_asins):
        all_classes = list(STRUCTURAL_CLASSES)
        all_prompts = []
        for cls in all_classes:
            for k in range(K_PER_CLASS):
                p = build_class_prompt(attrs, cls, k)
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
                individuals = split_numbered_list(blob)
                if len(individuals) > 1:
                    n_blob_outputs += 1
                    n_blob_this_asin += 1
                for qtext in individuals:
                    n_individual_total += 1
                    if not qtext or len(qtext.split()) < 3:
                        continue
                    attr_ok, missing = check_4_of_4_attrs(qtext, attrs)
                    bias_hit, _ = has_bias_substring(qtext)
                    if attr_ok and not bias_hit:
                        kept.append({
                            "text": qtext,
                            "content_ok": attr_ok,
                            "bias_hit": bias_hit,
                            "predicted_class": classify_structural(qtext),
                            "attrs_used": [name for name, _ in attrs],
                        })
                        n_individual_kept += 1
                        n_kept_this_asin += 1
                    else:
                        n_attr_miss_total += 1
            asin_locked[cls] = {
                "prompt_template": build_class_prompt(attrs, cls, 0).rsplit(" (variant", 1)[0],
                "requested_class": cls,
                "generated": kept,
            }
        clean_pool[asin] = {
            "attrs": [(n, v) for n, v in attrs],
            "by_class": asin_locked,
        }
        s16a_stats_per_asin.append({
            "asin": asin, "kept": n_kept_this_asin, "blobs": n_blob_this_asin,
            "attrs": [(n, v) for n, v in attrs],
        })
        log(f"  [{ai+1}/{len(valid_asins)}] {asin}: attrs={[(n,v) for n,v in attrs]!r}, "
            f"kept={n_kept_this_asin}, blobs={n_blob_this_asin}, gen={gen_time:.1f}s")

    log(f"\n  Total outputs: {n_gen_total}")
    log(f"  Blob outputs (≥2 queries in one): {n_blob_outputs}")
    log(f"  Individual queries parsed: {n_individual_total}")
    log(f"  Kept (4/4 REAL + no bias): {n_individual_kept} "
        f"({n_individual_kept/max(n_individual_total,1)*100:.2f}%)")
    log(f"  Attr-missed (4/4 REAL check fail): {n_attr_miss_total}")

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
    log(f"  Clean queries/asin: mean={np.mean([s['kept'] for s in s16a_stats_per_asin]):.1f}, "
        f"min={min(s['kept'] for s in s16a_stats_per_asin)}, "
        f"max={max(s['kept'] for s in s16a_stats_per_asin)}")

    # Save pool
    with open(OUTPUT_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 16A: REAL 4-attribute audit from Amazon metadata (Brand, Color, Item Weight, Material); per-product attribute sets",
                "K_per_class": K_PER_CLASS,
                "forbidden_words_count": len(FORBIDDEN_WORDS) + len(FORBIDDEN_PHRASES),
                "metadata_source": str(META_FILE),
                "attr_priority": ["Brand", "Color", "Item Weight", "Material"],
            },
            "stats": {
                "total_outputs": n_gen_total,
                "blob_outputs": n_blob_outputs,
                "individual_queries": n_individual_total,
                "kept_individual": n_individual_kept,
                "filter_pass_rate": n_individual_kept / max(n_individual_total, 1),
                "asins_with_4_real_attrs": len(valid_asins),
                "attr_type_combos": dict(attr_type_combos),
                "clean_per_asin_mean": n_individual_kept / max(len(valid_asins), 1),
            },
            "pool": clean_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_POOL}")

    # === Auto-export REAL samples ===
    log("\n=== Auto-exporting real samples ===")
    samples = {}
    for asin in list(clean_pool.keys())[:3]:
        attrs_list = clean_pool[asin]["attrs"]
        cls_samples = {}
        for cls, info in clean_pool[asin]["by_class"].items():
            if info["generated"]:
                g = info["generated"][0]
                cls_samples[cls] = {
                    "text": g["text"],
                    "predicted_class": g["predicted_class"],
                    "content_ok": g["content_ok"],
                }
        samples[asin] = {"attrs": attrs_list, "by_class": cls_samples}
    with open(OUTPUT_SAMPLES, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_SAMPLES}")

    # Comparison S15 vs S16A
    log("\n=== Comparison: S15 (fake 4/4) vs S16A (REAL 4/4) ===")
    log(f"  {'Metric':<25} {'S15':>15} {'S16A':>15}")
    log(f"  {'-'*60}")
    s15_data = json.load(open(SCRATCH / "stage15_clean_pool.json"))
    s15_real4 = 0
    s15_total = 0
    for asin, data in s15_data["pool"].items():
        b, c, age, cat = data["attrs"]
        has_empty = not (b and c and age and cat)
        for cls, info in data["by_class"].items():
            for g in info["generated"]:
                s15_total += 1
                if has_empty:
                    pass  # auto fake
                else:
                    low = g["text"].lower()
                    if (b.lower() in low and c.lower() in low and age.lower() in low and cat.lower() in low):
                        s15_real4 += 1

    log(f"  {'clean queries total':<25} {s15_total:>15} {n_individual_kept:>15}")
    log(f"  {'REAL 4/4 (no empty)':<25} {s15_real4:>15} {n_individual_kept:>15}")
    log(f"  {'fake 4/4 (empty bypass)':<25} {s15_total-s15_real4:>15} {0:>15}")
    log(f"  {'REAL 4/4 rate':<25} {s15_real4/max(s15_total,1)*100:>14.1f}% {n_individual_kept/max(n_individual_total,1)*100:>14.1f}%")
    log(f"  {'ASINs':<25} {20:>15} {len(valid_asins):>15}")

    comparison = {
        "config": {
            "description": "Stage 16A: REAL 4-attribute audit from Amazon metadata",
            "K_per_class": K_PER_CLASS,
            "attr_priority": ["Brand", "Color", "Item Weight", "Material"],
            "metadata_source": str(META_FILE),
        },
        "stage16a": {
            "total_outputs": n_gen_total,
            "blob_outputs": n_blob_outputs,
            "individual_queries": n_individual_total,
            "kept_individual": n_individual_kept,
            "filter_pass_rate": n_individual_kept / max(n_individual_total, 1),
            "asins_with_4_real_attrs": len(valid_asins),
            "attr_type_combos": dict(attr_type_combos),
            "clean_per_asin_mean": n_individual_kept / len(valid_asins),
            "pool_classes_per_asin": float(np.mean(n_classes_per_asin)),
            "per_class_hit_rate": {
                cls: clean_class_hits[cls] / max(clean_class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
            "samples": samples,
        },
        "stage15_fake4of4_audit": {
            "total": s15_total,
            "real_4of4": s15_real4,
            "fake_4of4": s15_total - s15_real4,
            "fake_4of4_rate": (s15_total - s15_real4) / max(s15_total, 1),
        },
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_COMPARISON}")

    log("\n=== Stage 16A complete ===")


if __name__ == "__main__":
    main()