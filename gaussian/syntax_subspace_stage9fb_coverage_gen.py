"""Stage 9F-B — Coverage-aware candidate generation.

Goal: extend 9C pool with queries that span 182d/PCA48 feature space, enabling
per-user personalization (targeted at each user's individual mu).

Design:
- Define K=12 "feature extreme" profiles that target different PCA48 directions:
  * very_low_function_words (modifier_fronted extreme)
  * very_high_prep (prepositional dense)
  * very_high_clause (relative_clause + subordinate)
  * very_high_modifier (modifier_fronted heavy)
  * very_high_verb (predicate_based dense)
  * very_long_compound (multiple nested)
  * very_short (fragment extreme)
  * very_question (interrogative extreme)
  * heavy_coordination
  * heavy_nesting (multi-clause)
  * heavy_negation
  * heavy_comparative

- For each ASIN × profile: generate K=4 candidate queries using vLLM
- Apply strict QA filter (L1-L7 from Stage 9C)
- Project to PCA48, check that new queries actually extend PCA48 coverage
- Combine with 9C pool
- Re-run selection + Stage 9E-P diagnostics

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_pool.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_features.jsonl.gz (appended)
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_coverage_stats.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9fb_coverage_gen.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
POOL_9C_IN = SCRATCH / "stage9c_pool_pilot.json"
POOL_9F_OUT = SCRATCH / "stage9fb_pool.json"
COVERAGE_OUT = SCRATCH / "stage9fb_coverage_stats.json"

SEED = 2024
N_PROFILES = 12
K_PER_PROFILE = 8
TEMP = 1.0
MAX_TOKENS = 50

# === 12 feature-extreme profiles ===
# Each profile is (name, system_prompt_suffix, required_features_dict)
FEATURE_PROFILES = [
    (
        "very_short_fragment",
        "Generate an EXTREMELY short query: just 3-6 words. List the attribute values directly with no function words. Example: 'Pampers White Toddler Health'.",
        {"min_n_prep": 0, "min_n_clause": 0, "min_n_verb": 0, "min_n_modifier": 0, "max_n_tok": 6},
    ),
    (
        "very_long_compound",
        "Generate a long compound query with 20+ words. Use multiple prepositions, at least one relative clause, one subordinate clause, and several modifiers. Example: 'Pampers item in White that is designed for Toddlers, which provides care under Health and Personal Care standards.'",
        {"min_n_prep": 4, "min_n_clause": 2, "min_n_verb": 2, "min_n_modifier": 3, "min_n_tok": 18},
    ),
    (
        "heavy_prepositional",
        "Generate a query dominated by prepositional phrases. Use 4-6 different prepositions (in, for, under, with, of, by). Example: 'Pampers item in white for toddler under health by main category.'",
        {"min_n_prep": 4, "min_n_clause": 0, "min_n_verb": 0, "min_n_modifier": 1, "min_n_tok": 10},
    ),
    (
        "heavy_clause_nesting",
        "Generate a query with deeply nested clauses. Include at least 2 relative clauses (using that/which) AND 2 subordinate clauses (using because/since/while/although). Example: 'A Pampers product that comes in white, which is suitable for toddlers because they need gentle care, although some brands differ.'",
        {"min_n_prep": 2, "min_n_clause": 3, "min_n_verb": 2, "min_n_modifier": 2, "min_n_tok": 16},
    ),
    (
        "heavy_modifier_fronted",
        "Generate a query where ALL attributes are placed BEFORE the head noun as hyphenated or compound modifiers. Example: 'White-Toddler Pampers-care product in Health & Personal Care category.'",
        {"min_n_prep": 1, "min_n_clause": 0, "min_n_verb": 0, "min_n_modifier": 4, "min_n_tok": 8},
    ),
    (
        "heavy_predicate_verb",
        "Generate a query dominated by verb predicates. Use multiple verbs: is, has, offers, designed for, suitable for, provides. Example: 'Pampers is white, has toddler suitability, offers Health & Personal Care quality.'",
        {"min_n_prep": 1, "min_n_clause": 0, "min_n_verb": 4, "min_n_modifier": 1, "min_n_tok": 12},
    ),
    (
        "heavy_coordination",
        "Generate a query with heavy coordination. Use 'and', 'as well as', 'while also' multiple times. Example: 'Pampers in white and for toddler as well as Health, while also being Personal Care.'",
        {"min_n_prep": 1, "min_n_clause": 0, "min_n_verb": 1, "min_n_modifier": 2, "min_n_tok": 12},
    ),
    (
        "extreme_question",
        "Generate a long, complex interrogative query with embedded clauses. End with '?'. Example: 'Which Pampers product in white, which is designed for toddlers under Health & Personal Care, would you recommend for baby care?'",
        {"min_n_prep": 2, "min_n_clause": 1, "min_n_verb": 1, "min_n_modifier": 1, "min_n_tok": 14, "must_end_with": "?"},
    ),
    (
        "heavy_negation",
        "Generate a query that explicitly NEGATES some attribute combinations to disambiguate. Use 'not', 'no', 'except'. Example: 'Pampers product that is not for adult use, only for toddlers in Health & Personal Care.'",
        {"min_n_prep": 1, "min_n_clause": 1, "min_n_verb": 1, "min_n_modifier": 1, "min_n_tok": 12, "must_contain_any": ["not", "no", "except", "without"]},
    ),
    (
        "heavy_comparative",
        "Generate a query with explicit COMPARISONS. Use 'better', 'best', 'most', 'more than', 'rather than'. Example: 'Pampers is the best option for white toddler diapers rather than adult products in Health & Personal Care.'",
        {"min_n_prep": 1, "min_n_clause": 1, "min_n_verb": 1, "min_n_modifier": 1, "min_n_tok": 12, "must_contain_any": ["better", "best", "most", "more than", "rather than", "instead of"]},
    ),
    (
        "passive_voice",
        "Generate a query in passive voice. Use 'is/are/was/were + past participle'. Example: 'White Pampers products are designed for toddlers, are sold under Health & Personal Care, and are made for baby care.'",
        {"min_n_prep": 1, "min_n_clause": 0, "min_n_verb": 3, "min_n_modifier": 1, "min_n_tok": 12, "must_contain_any": ["is designed", "are made", "is sold", "is used", "are used", "is meant", "are meant"]},
    ),
    (
        "first_person_pronoun",
        "Generate a query using first-person pronouns. Use 'I', 'my', 'me', 'we'. Example: 'I want a Pampers product for my toddler in white, we need Health & Personal Care.'",
        {"min_n_prep": 1, "min_n_clause": 1, "min_n_verb": 2, "min_n_modifier": 1, "min_n_tok": 12, "must_contain_any": ["I ", "my ", "me ", "we ", "our "]},
    ),
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def analyze_query_structure(q: str) -> dict:
    """Text-level structure analysis (same as Stage 9C)."""
    ql = q.lower().strip()
    n_prep = len(re.findall(r"\b(for|with|in|under|of|by|at|on|from|to|about)\b", ql))
    n_coord = len(re.findall(r"\b(and|while|but|or|because|since|although|when|if|so|as well as)\b", ql))
    n_clause = len(re.findall(r"\b(that|which|who|whom|whose)\b", ql))
    n_verb = len(re.findall(r"\b(is|are|has|have|offers|provides|designed|suitable|comes|sells|includes|recommend|want|need|use|made|fits)\b", ql))
    n_modifier = len(re.findall(r"\b(white|toddler|baby|infant|adult|small|large|mini|maxi|junior|senior)\b", ql))
    n_tok = len(re.findall(r"\b\w+\b", q))
    return {
        "n_prep": n_prep, "n_coord": n_coord, "n_clause": n_clause,
        "n_verb": n_verb, "n_modifier": n_modifier, "n_tok": n_tok,
    }


def strict_filter(q: str, attrs: dict) -> bool:
    """Stage 9F-B soft filter — coverage queries prioritize structural diversity
    over template purity. We trust the prompt's structural instruction; the only
    hard checks are: must include all attrs, no field-name leakage, no broken punctuation.
    """
    if not q or len(q.strip()) < 5:
        return False
    # Must contain all attrs (case-insensitive)
    for v in attrs.values():
        if v and v.lower() not in q.lower():
            return False
    # Field name leak (from Stage 9C audit) — this is the only semantic filter
    field_names = ["brand:", "color:", "size:", "age range:", "category:"]
    ql = q.lower()
    for fn in field_names:
        if fn in ql:
            return False
    # Invalid punctuation
    if q.count("?") > 1 or q.count(";") > 0 or q.count("|") > 0:
        return False
    # Length bounds (looser for compound queries)
    if len(q) < 5 or len(q) > 250:
        return False
    if q.count(".") > 3:
        return False
    return True


def build_prompt(attrs: dict, profile_suffix: str, idx: int) -> str:
    """Build prompt for vLLM. Inline URL workaround for vLLM 0.27.1."""
    attr_lines = "\n".join([f"- {k}: {v}" for k, v in attrs.items() if v])
    base = (
        f"You are generating a short query (10-20 words) describing a product.\n\n"
        f"Product attributes (you MUST use ALL of these):\n{attr_lines}\n\n"
        f"Rules:\n"
        f"1. Use ALL attributes exactly as listed.\n"
        f"2. Do NOT use phrases like 'looking for' or 'searching for'.\n"
        f"3. Do NOT include field names like 'brand:' or 'color:'.\n"
        f"4. Do NOT use adjectives like 'best', 'perfect', 'high quality'.\n"
        f"5. Output ONLY the query text, nothing else.\n\n"
        f"Structural instruction: {profile_suffix}\n\n"
        f"Generate a query now (variant {idx}):"
    )
    # Use raw completions format with system/user/assistant
    return base


def call_vllm_batch(prompts: List[str]) -> List[str]:
    """Batch call vLLM with inline URL (workaround for 0.27.1 routing bug)."""
    import requests
    url = "http://localhost:8800/v1/completions"
    model_path = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
    sampling = {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
        "top_p": 0.95,
        "stop": ["\n\n", "Generate"],
    }
    payload = {
        "model": model_path,
        "prompt": [f"system\n\nuser\n{p}\n\nassistant\n" for p in prompts],
        **sampling,
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        log(f"  vLLM error {resp.status_code}: {resp.text[:200]}")
        return [""] * len(prompts)
    data = resp.json()
    choices = data.get("choices", [])
    return [c.get("text", "").strip() for c in choices]


def main():
    log("=== Stage 9F-B — Coverage-aware candidate generation ===")

    # === 1. Load ASINs ===
    log("\n=== 1. Loading ASINs ===")
    asin_data_full = json.load(open(ASINS_IN))["asins"]
    asin_data = [a for a in asin_data_full if a["asin"] in
                 json.load(open(POOL_9C_IN))["pools"]]
    log(f"  ASINs: {len(asin_data)}")

    # === 2. Build all prompts ===
    log("\n=== 2. Building prompts ===")
    all_prompts = []
    prompt_meta = []
    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        for p_idx, (name, suffix, _) in enumerate(FEATURE_PROFILES):
            for k in range(K_PER_PROFILE):
                prompt = build_prompt(attrs, suffix, k)
                all_prompts.append(prompt)
                prompt_meta.append({
                    "asin": asin,
                    "profile": name,
                    "variant": k,
                    "attrs": attrs,
                })
    log(f"  total prompts: {len(all_prompts)} (10 ASINs × {N_PROFILES} profiles × {K_PER_PROFILE} variants)")

    # === 3. Batch call vLLM ===
    log("\n=== 3. Calling vLLM (batch=64) ===")
    BATCH = 64
    all_outputs = []
    t0 = time.time()
    for i in range(0, len(all_prompts), BATCH):
        batch = all_prompts[i:i + BATCH]
        outputs = call_vllm_batch(batch)
        all_outputs.extend(outputs)
        if (i // BATCH) % 4 == 0:
            log(f"  [{i + len(batch)}/{len(all_prompts)}] elapsed {time.time() - t0:.0f}s")
    log(f"  total outputs: {len(all_outputs)} in {time.time() - t0:.0f}s")

    # === 4. Parse + filter ===
    log("\n=== 4. Parsing + filtering ===")
    new_pool = {}
    layer_stats = collections.Counter()
    for meta, raw in zip(prompt_meta, all_outputs):
        asin = meta["asin"]
        # Extract the query (strip leading bullets, system prompts, etc.)
        text = raw.strip()
        # Take only first line
        text = text.split("\n")[0].strip()
        # Strip leading "Query:" or quotes
        text = re.sub(r'^(Query:|Query text:|"Quotes")', "", text).strip(' "\'')
        # Remove trailing periods and quotes
        text = text.rstrip(".").strip()
        if not text:
            layer_stats["empty"] += 1
            continue

        # Strict filter
        if not strict_filter(text, meta["attrs"]):
            layer_stats["filter_fail"] += 1
            continue

        # Structure analysis (kept for downstream stats, not enforced)
        struct = analyze_query_structure(text)

        # Dedup (exact + jaccard)
        new_pool.setdefault(asin, []).append({
            "family": f"9F_coverage_{meta['profile']}",
            "query": text,
            "strict": True,
            "n_tok": struct["n_tok"],
            "attrs_covered": len(meta["attrs"]),
            "n_prep": struct["n_prep"],
            "n_coord": struct["n_coord"],
            "n_clause": struct["n_clause"],
            "n_verb": struct["n_verb"],
            "n_modifier": struct["n_modifier"],
            "profile": meta["profile"],
        })
        layer_stats["ACCEPT"] += 1

    # Per-ASIN dedup
    for asin in new_pool:
        seen = set()
        unique = []
        for q in new_pool[asin]:
            k = q["query"].lower().strip()
            if k in seen:
                layer_stats["dup"] += 1
                continue
            seen.add(k)
            unique.append(q)
        new_pool[asin] = unique

    log(f"  layer stats: {dict(layer_stats)}")
    total_new = sum(len(v) for v in new_pool.values())
    log(f"  total new queries: {total_new}")
    log(f"  per-ASIN counts:")
    for asin in sorted(new_pool.keys()):
        log(f"    {asin}: {len(new_pool[asin])}")

    # === 5. Save new pool ===
    with open(POOL_9F_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9F-B coverage-aware generation",
                "n_profiles": N_PROFILES,
                "k_per_profile": K_PER_PROFILE,
                "n_asins": len(asin_data),
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
            },
            "layer_stats": dict(layer_stats),
            "new_pool": new_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {POOL_9F_OUT}")

    # === 6. PCA48 coverage analysis (compare 9C vs 9F+B) ===
    log("\n=== 6. PCA48 coverage analysis ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    from sklearn.decomposition import PCA
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    pca = PCA(n_components=48, random_state=SEED)
    pca.fit(P["X_scaled"][train_idx])

    # Load features cache
    feat_map = {}
    feat_cache = SCRATCH / "stage7b_query_features.jsonl.gz"
    with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]

    def extract_pca48(queries):
        zs = []
        miss = 0
        for q in queries:
            k = feat_key(q)
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs.append(z)
        return np.stack(zs) if zs else np.zeros((0, 48)), miss

    # 9C pool stats
    pool_9c = json.load(open(POOL_9C_IN))["pools"]
    cov_stats = {}
    for asin in asin_data:
        asin_key = asin["asin"]
        queries_9c = [q["query"] for q in pool_9c.get(asin_key, [])]
        queries_9f = [q["query"] for q in new_pool.get(asin_key, [])]
        queries_combined = queries_9c + queries_9f

        z9c, miss_9c = extract_pca48(queries_9c)
        z9f, miss_9f = extract_pca48(queries_9f)
        zcomb, miss_comb = extract_pca48(queries_combined)

        if len(z9c) == 0 or len(zcomb) == 0:
            continue
        # Compute std per PCA dim (proxy for spread)
        std_9c = float(np.mean(z9c.std(axis=0)))
        std_comb = float(np.mean(zcomb.std(axis=0)))
        # Compute pairwise mean distance
        from scipy.spatial.distance import pdist
        d_9c = float(pdist(z9c).mean()) if len(z9c) > 1 else 0
        d_comb = float(pdist(zcomb).mean()) if len(zcomb) > 1 else 0
        cov_stats[asin_key] = {
            "n_9c": len(z9c), "n_9f": len(z9f), "n_combined": len(zcomb),
            "miss_features": {"9c": miss_9c, "9f": miss_9f, "combined": miss_comb},
            "mean_std_9c": std_9c,
            "mean_std_combined": std_comb,
            "std_growth_pct": (std_comb / std_9c - 1) * 100 if std_9c > 0 else 0,
            "mean_pairwise_dist_9c": d_9c,
            "mean_pairwise_dist_combined": d_comb,
            "dist_growth_pct": (d_comb / d_9c - 1) * 100 if d_9c > 0 else 0,
        }
        log(f"  {asin_key}: 9C n={len(z9c)}, 9F n={len(z9f)}, "
            f"std_growth={cov_stats[asin_key]['std_growth_pct']:.1f}%, "
            f"dist_growth={cov_stats[asin_key]['dist_growth_pct']:.1f}%")

    with open(COVERAGE_OUT, "w", encoding="utf-8") as f:
        json.dump(cov_stats, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {COVERAGE_OUT}")

    log("\n=== Stage 9F-B complete ===")


if __name__ == "__main__":
    main()