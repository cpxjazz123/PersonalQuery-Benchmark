"""Stage 8.5.X: Extra-Semantics Audit + Template Saturation Audit.

Scan existing stage8_5_pool.json to quantify:
1. Extra-semantic token frequency per ASIN:
   - quality: high-quality, premium, reliable, durable, sturdy, ...
   - functional: lightweight, portable, easy-to-use, ...
   - use_case: newborn care, baby shower, daily use, travel, ...
   - emotion: best, favorite, perfect, amazing, ...
2. Opening template saturation: per-ASIN fraction of queries starting with
   "Looking for" / "Searching for" / "I am looking for" / "I am searching for"
3. Exact + near-duplicate rate per ASIN (token Jaccard ≥ 0.85)

Output:
- stage8_5_audit.json: per-ASIN stats + global aggregate
- Console: top offending extra-semantic tokens, template percentages, dup rates

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_audit.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_audit.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_IN = SCRATCH / "stage8_5_pool.json"
ASINS_IN = SCRATCH / "stage8_5_asins.json"
OUT = SCRATCH / "stage8_5_audit.json"


# Extra-semantic blacklist (extra attributes / requirements not in canonical attrs)
EXTRA_SEMANTIC_PATTERNS = {
    "quality": [
        r"\bhigh[- ]quality\b",
        r"\bpremium\b",
        r"\breliable\b",
        r"\bdurable\b",
        r"\bsturdy\b",
        r"\bsuperior\b",
        r"\bexcellent\b",
        r"\boutstanding\b",
        r"\btop[- ]notch\b",
        r"\bworld[- ]class\b",
    ],
    "functional": [
        r"\blightweight\b",
        r"\bportable\b",
        r"\bcompact\b",
        r"\bversatile\b",
        r"\beasy[- ]to[- ]use\b",
        r"\buser[- ]friendly\b",
        r"\bmulti[- ]purpose\b",
        r"\badjustable\b",
        r"\breusable\b",
        r"\bwashable\b",
    ],
    "use_case": [
        r"\bnewborn care\b",
        r"\bbaby shower\b",
        r"\bdaily use\b",
        r"\btravel\b",
        r"\bgift\b",
        r"\bbeginner\b",
        r"\bprofessional\b",
        r"\bfor kids\b",
        r"\bfor baby\b",
    ],
    "emotion": [
        r"\bbest\b",
        r"\bfavorite\b",
        r"\bperfect\b",
        r"\bamazing\b",
        r"\bwonderful\b",
        r"\bgreat\b",
        r"\blove\b",
        r"\bmust[- ]have\b",
    ],
    "aesthetic": [
        r"\bpretty\b",
        r"\bcute\b",
        r"\badorable\b",
        r"\bbeautiful\b",
        r"\belegant\b",
        r"\bmodern\b",
        r"\bclassic\b",
        r"\bchic\b",
        r"\btrendy\b",
        r"\bstylish\b",
    ],
}

# Opening templates (will be matched case-insensitive at start of query)
OPENING_TEMPLATES = [
    r"^i am (looking|searching) for\b",
    r"^looking for\b",
    r"^searching for\b",
    r"^i need\b",
    r"^i want\b",
    r"^i'm (looking|searching) for\b",
    r"^find me\b",
    r"^show me\b",
    r"^can you find\b",
    r"^where can i find\b",
]


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    log("=== Stage 8.5.X: Extra-Semantics + Template Saturation Audit ===")

    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    asin_data = json.load(open(ASINS_IN))["asins"]
    asin_attrs = {a["asin"]: a["attrs_used"] for a in asin_data}

    log(f"  {len(pools)} ASINs, attrs loaded")

    # === Per-ASIN audit ===
    per_asin = {}
    pattern_hits_global = collections.Counter()
    pattern_categories_global = collections.Counter()
    extra_sem_per_asin = []  # fraction of strict queries containing any extra-semantic
    template_pct_per_asin = []  # fraction of strict queries with template opening

    for asin, qs in pools.items():
        strict_qs = [q for q in qs if q["strict"]]
        if not strict_qs:
            continue
        attrs = asin_attrs.get(asin, {})
        # Build attr value tokens (lowercase) — these are NOT extra-semantics
        attr_tokens = set()
        attr_phrases = []
        for k, v in attrs.items():
            if v:
                attr_phrases.append(str(v).lower())
                for t in tokenize(str(v)):
                    attr_tokens.add(t)

        # Per-ASIN extra-semantic hits
        asin_hits = collections.Counter()
        asin_categories = collections.Counter()
        n_with_extra = 0
        for q in strict_qs:
            text = q["query"]
            text_lower = text.lower()
            q_has_extra = False
            for cat, patterns in EXTRA_SEMANTIC_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, text_lower):
                        asin_hits[(cat, pat)] += 1
                        asin_categories[cat] += 1
                        pattern_hits_global[(cat, pat)] += 1
                        pattern_categories_global[cat] += 1
                        q_has_extra = True
                        break
            if q_has_extra:
                n_with_extra += 1
        extra_sem_frac = n_with_extra / len(strict_qs)

        # Per-ASIN template openings
        opening_counts = collections.Counter()
        n_template = 0
        for q in strict_qs:
            text = q["query"].strip()
            matched = False
            for tmpl in OPENING_TEMPLATES:
                if re.match(tmpl, text, re.IGNORECASE):
                    opening_counts[tmpl] += 1
                    matched = True
                    break
            if matched:
                n_template += 1
        template_frac = n_template / len(strict_qs)

        # Per-ASIN exact + near-dup
        n_exact_dup = 0
        n_near_dup = 0
        texts = [q["query"] for q in strict_qs]
        token_lists = [tokenize(t) for t in texts]
        seen_pairs = set()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] == texts[j]:
                    n_exact_dup += 1
                    seen_pairs.add((i, j))
                    continue
                jc = jaccard(token_lists[i], token_lists[j])
                if jc >= 0.85:
                    n_near_dup += 1
                    seen_pairs.add((i, j))
        n_unique = len(texts) - len(seen_pairs)
        dup_rate = len(seen_pairs) / max(1, len(texts) * (len(texts) - 1) / 2)

        per_asin[asin] = {
            "n_strict": len(strict_qs),
            "extra_sem_frac": extra_sem_frac,
            "n_with_extra": n_with_extra,
            "extra_sem_categories": dict(asin_categories),
            "opening_counts": dict(opening_counts),
            "template_frac": template_frac,
            "n_template": n_template,
            "n_unique_queries": n_unique,
            "n_exact_dup": n_exact_dup,
            "n_near_dup": n_near_dup,
            "dup_rate": dup_rate,
        }
        extra_sem_per_asin.append(extra_sem_frac)
        template_pct_per_asin.append(template_frac)

    # === Global aggregate ===
    log(f"\n=== Extra-Semantics Patterns (Top 20) ===")
    log(f"{'Category':<12} {'Pattern':<25} {'Hits':<6} {'ASINs'}")
    # Compute # ASINs per pattern
    pattern_asin_counts = collections.Counter()
    for asin, d in per_asin.items():
        # Re-scan
        pass  # already aggregated in pattern_hits_global

    log(f"  Total pattern hits: {sum(pattern_hits_global.values())}")
    log(f"  Total category hits: {sum(pattern_categories_global.values())}")
    log(f"  Categories by frequency:")
    for cat, n in pattern_categories_global.most_common():
        log(f"    {cat}: {n}")

    log(f"\n  Top patterns:")
    for (cat, pat), n in pattern_hits_global.most_common(20):
        log(f"    {cat:<12} {pat:<25} {n}")

    # Aggregate extra-semantic stats
    avg_extra = sum(extra_sem_per_asin) / len(extra_sem_per_asin) if extra_sem_per_asin else 0
    median_extra = sorted(extra_sem_per_asin)[len(extra_sem_per_asin) // 2] if extra_sem_per_asin else 0
    pct_asins_extra_above_30 = sum(1 for x in extra_sem_per_asin if x > 0.30) / len(extra_sem_per_asin) if extra_sem_per_asin else 0
    pct_asins_extra_above_50 = sum(1 for x in extra_sem_per_asin if x > 0.50) / len(extra_sem_per_asin) if extra_sem_per_asin else 0
    log(f"\n  Per-ASIN extra_sem_frac stats:")
    log(f"    mean: {avg_extra:.3f}")
    log(f"    median: {median_extra:.3f}")
    log(f"    > 30%: {pct_asins_extra_above_30*100:.1f}% of ASINs")
    log(f"    > 50%: {pct_asins_extra_above_50*100:.1f}% of ASINs")

    avg_template = sum(template_pct_per_asin) / len(template_pct_per_asin) if template_pct_per_asin else 0
    median_template = sorted(template_pct_per_asin)[len(template_pct_per_asin) // 2] if template_pct_per_asin else 0
    pct_asins_template_above_50 = sum(1 for x in template_pct_per_asin if x > 0.50) / len(template_pct_per_asin) if template_pct_per_asin else 0
    pct_asins_template_above_80 = sum(1 for x in template_pct_per_asin if x > 0.80) / len(template_pct_per_asin) if template_pct_per_asin else 0
    log(f"\n  Per-ASIN template_frac stats:")
    log(f"    mean: {avg_template:.3f}")
    log(f"    median: {median_template:.3f}")
    log(f"    > 50%: {pct_asins_template_above_50*100:.1f}% of ASINs")
    log(f"    > 80%: {pct_asins_template_above_80*100:.1f}% of ASINs")

    # Dup rates
    dup_rates = [d["dup_rate"] for d in per_asin.values()]
    avg_dup = sum(dup_rates) / len(dup_rates) if dup_rates else 0
    log(f"\n  Per-ASIN dup_rate (jaccard ≥ 0.85 or exact): mean {avg_dup:.3f}")

    # === Save ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5 audit: extra-semantics + template saturation + duplicate rate",
                "n_patterns": sum(len(v) for v in EXTRA_SEMANTIC_PATTERNS.values()),
                "n_templates": len(OPENING_TEMPLATES),
                "dup_threshold_jaccard": 0.85,
            },
            "global": {
                "pattern_hits": {f"{c}|{p}": n for (c, p), n in pattern_hits_global.items()},
                "category_hits": dict(pattern_categories_global),
                "avg_extra_sem_frac": avg_extra,
                "median_extra_sem_frac": median_extra,
                "pct_asins_extra_above_30": pct_asins_extra_above_30,
                "pct_asins_extra_above_50": pct_asins_extra_above_50,
                "avg_template_frac": avg_template,
                "median_template_frac": median_template,
                "pct_asins_template_above_50": pct_asins_template_above_50,
                "pct_asins_template_above_80": pct_asins_template_above_80,
                "avg_dup_rate": avg_dup,
            },
            "per_asin": per_asin,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()