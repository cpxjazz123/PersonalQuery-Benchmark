"""Stage 11 — Content-locked generation + structural class detector + selection comparison.

Stage 11 implements the "Content controlled, syntax manipulated" framework:

  Step 1: Structural class detector — classify existing 5000 pool queries
          into 7 structural classes (PREPOSITIONAL, COORDINATION, RELATIVE_CLAUSE,
          MODIFIER_FRONTED, PREDICATE, QUESTION, SIMPLE) using spaCy parse.

  Step 2: Content-locked generation — for each ASIN, generate candidates
          with attribute lock + structural condition prompt. Build a
          content-locked pool.

  Step 3: Comparison selection — run mean_t50 + A1 on (a) original pool,
          (b) content-locked pool, (c) content-locked + structural-class-balanced pool.

Inputs:
  - stage8_5_pool.json (5000 candidates)
  - stage9g_user_gaussians_3levels.json (294 user Gaussians)

Outputs:
  - stage11_structural_class.json — per-query structural class
  - stage11_content_locked_pool.json — content-locked generation
  - stage11_comparison.json — selection metrics across 3 pools
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

OUTPUT_STRUCT = SCRATCH / "stage11_structural_class.json"
OUTPUT_CL_POOL = SCRATCH / "stage11_content_locked_pool.json"
OUTPUT_COMPARISON = SCRATCH / "stage11_comparison.json"

STRUCTURAL_CLASSES = [
    "PREPOSITIONAL",      # Ends with PP / "in X category"
    "COORDINATION",       # Has and/but/or joining NPs/VPs
    "RELATIVE_CLAUSE",    # Contains that/which/who + finite verb
    "MODIFIER_FRONTED",   # Fronted adv/adj: "For toddlers, ..."
    "PREDICATE",          # Copula: "X is Y" / "X is for Y"
    "QUESTION",           # Ends with ? / wh-movement
    "SIMPLE",             # None of the above
]

# Common bias words we want to suppress in content-locked generation
LLM_BIAS_BLACKLIST = [
    "durable", "lightweight", "comfortable", "perfect", "modern",
    "stylish", "premium", "high-quality", "top-quality", "quality",
    "ideal", "great", "excellent", "best", "amazing",
    "for baby", "for your baby",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def classify_structural(query: str, nlp) -> str:
    """Classify a query into one of 7 structural classes using spaCy parse.

    Heuristics (no deep ML model needed):
    - QUESTION: ends with '?'
    - RELATIVE_CLAUSE: has 'that/which/who' followed by finite verb
    - COORDINATION: has ' and ' or ' but ' or ' or ' joining NPs
    - PREPOSITIONAL: ends with ' in <NP>' / ' for <NP>'
    - MODIFIER_FRONTED: starts with PP/AdvP followed by comma
    - PREDICATE: contains copula ' is ' / ' are '
    - SIMPLE: none of the above
    """
    q = query.strip()
    low = q.lower()

    # 1. QUESTION
    if q.endswith("?"):
        return "QUESTION"

    # 2. RELATIVE_CLAUSE
    rel_match = re.search(r"\b(that|which|who)\b\s+\w+s?\s+\w+", low)
    if rel_match:
        return "RELATIVE_CLAUSE"

    # 3. COORDINATION (conjoined NPs or VPs)
    if re.search(r"\b(and|but|or)\b\s+(a|an|the|I|it|they|[A-Z])", q):
        # Avoid matching "for baby and toddler" (content, not structure)
        # Check if 'and' is joining two NPs by looking at what follows
        coord_match = re.search(r"\b(and|but|or)\s+(\w+)", low)
        if coord_match and coord_match.group(2) not in ("the", "a", "an"):
            return "COORDINATION"

    # 4. PREPOSITIONAL — ends with PP like "in Health & Personal Care"
    pp_match = re.search(r"\b(in|for|under|within)\s+[A-Z][\w\s&-]{2,50}\s*$", q)
    if pp_match:
        return "PREPOSITIONAL"

    # 5. MODIFIER_FRONTED — starts with PP/AdvP + comma
    if re.match(r"^(For|With|In|To|As)\s+\w+[\w\s]*,\s+", q):
        return "MODIFIER_FRONTED"

    # 6. PREDICATE — copula structure
    if re.search(r"\bis\s+(a|an|the|designed|intended|suitable|perfect)\b", low):
        return "PREDICATE"
    if re.search(r"\bare\s+(a|an|the|designed|intended|suitable)\b", low):
        return "PREDICATE"

    return "SIMPLE"


def step1_structural_detector():
    """Step 1: classify existing pool into 7 structural classes."""
    log("\n=== Step 1: Structural class detector ===")

    pool = json.load(open(POOL_IN))
    pools = pool["pools"]
    log(f"  pool: {len(pools)} ASINs")

    # Classify all queries
    per_asin_classes = {}
    global_class_counter = Counter()
    skipped = 0
    for asin, queries in pools.items():
        per_query = []
        for q in queries:
            text = q.get("query", str(q))
            cls = classify_structural(text, None)  # pure regex, no spaCy needed
            per_query.append({"query": text, "class": cls})
            global_class_counter[cls] += 1
        per_asin_classes[asin] = per_query
    log(f"  global class distribution: {dict(global_class_counter)}")
    log(f"  total: {sum(global_class_counter.values())}")

    # Per-ASIN distribution
    per_asin_class_count = []
    for asin, qs in per_asin_classes.items():
        c = Counter(q["class"] for q in qs)
        per_asin_class_count.append(c)
    avg_per_asin = {k: np.mean([c.get(k, 0) for c in per_asin_class_count]) for k in STRUCTURAL_CLASSES}
    log(f"  per-ASIN mean class count: {dict((k, round(v, 1)) for k, v in avg_per_asin.items())}")

    # Save
    with open(OUTPUT_STRUCT, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 11B: structural class detector",
                "method": "regex heuristics, no LLM",
                "classes": STRUCTURAL_CLASSES,
            },
            "global_distribution": dict(global_class_counter),
            "per_asin_avg_count": avg_per_asin,
            "per_asin": per_asin_classes,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_STRUCT}")
    return per_asin_classes


def step2_content_locked_generation(struct_data, n_sample_asins=20):
    """Step 2: Generate content-locked candidates per structural class.

    Uses Stage 16 v6 regex + LLM generation pattern, but with strict content lock.
    For prototype: use a small subset (20 ASINs × 7 classes × K=4 = 560 candidates).
    """
    log("\n=== Step 2: Content-locked generation (prototype) ===")

    # Load LLM client (per CLAUDE.md Rule 8/9h: use QwenLocalClient + direct batched vLLM)
    try:
        from llm_client import QwenLocalClient
        from vllm import SamplingParams
        log("  using QwenLocalClient + vLLM SamplingParams (batched)")
    except ImportError as e:
        log(f"  WARN: QwenLocalClient/SamplingParams not importable: {e}; using mock for prototype test")
        QwenLocalClient = None
        SamplingParams = None

    pool = json.load(open(POOL_IN))
    pools = pool["pools"]

    # Pick top 20 ASINs by pool size
    sorted_asins = sorted(pools.keys(), key=lambda a: -len(pools[a]))[:n_sample_asins]

    # For each asin: extract 4 content attrs from existing pool (use first query's coverage)
    # Heuristic: brand=Capitalized word (not sentence starter), color=after "in"/"white"/etc, age=before "age range", category=last PP
    SENTENCE_STARTERS = {"I", "I'm", "Looking", "Searching", "Find", "Need", "Want",
                         "Looking", "Looking", "Here", "There", "Get", "Buy"}

    def extract_attrs(text):
        low = text.lower()
        # Brand: first capitalized word NOT a sentence starter
        brand = ""
        for m in re.finditer(r"\b([A-Z][a-zA-Z&\-]+(?:\s+[A-Z][a-zA-Z&\-]+)*)\b", text):
            cand = m.group(1)
            if cand not in SENTENCE_STARTERS and len(cand) >= 3:
                brand = cand
                break
        # Color
        color_m = re.search(r"\b(white|black|red|blue|green|yellow|pink|grey|gray|silver|gold|brown|purple)\b", low)
        color = color_m.group(1) if color_m else ""
        # Age
        age_m = re.search(r"\b(infant|toddler|baby|newborn|child|adult)s?\b", low)
        age = age_m.group(1) if age_m else ""
        # Category: greedy match after 'in/within/under' stopping at category/section/department/main
        cat_m = re.search(r"(?:in|within|under)\s+(?:the\s+)?([A-Z][\w\s&-]+?)(?:\s+main)?(?:\s+(?:category|section|department))?\s*\.?$", text)
        if cat_m:
            category = cat_m.group(1).strip()
        else:
            # Fallback: last capitalized 2-5 word sequence
            caps = re.findall(r"[A-Z][\w&-]+(?:\s+[A-Z][\w&-]+){1,4}", text)
            category = caps[-1] if caps else ""
        # Filter: category must be 5+ chars and not be just one capital word
        if len(category) < 5:
            category = ""
        return brand, color, age, category

    content_locked_pool = {}
    bias_hits_total = 0
    coverage_total = 0

    for ai, asin in enumerate(sorted_asins):
        queries = pools[asin]
        # Extract attrs — try each query until we find one with all 4 attrs
        attrs = None
        for q in queries:
            text = q.get("query", str(q))
            brand, color, age, category = extract_attrs(text)
            if brand and color and age and category:
                attrs = (brand, color, age, category)
                break
        if not attrs:
            # Fallback: pick the best partial, fill missing with placeholder
            for q in queries:
                text = q.get("query", str(q))
                brand, color, age, category = extract_attrs(text)
                attrs = (brand or "Product", color or "colored", age or "for users", category or "main category")
                break
            log(f"  [{ai+1}/{len(sorted_asins)}] {asin}: using fallback attrs {attrs}")
        else:
            log(f"  [{ai+1}/{len(sorted_asins)}] {asin}: attrs={attrs}")

        brand, color, age, category = attrs
        # Content-locked generation prompts (one per structural class)
        prompts_by_class = {
            "PREPOSITIONAL": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST end with a prepositional phrase like 'in {category}'. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium, high-quality, top-quality, quality, ideal, great, excellent, best, amazing.",
            "COORDINATION": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST use 'and' to coordinate two noun phrases or verb phrases. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
            "RELATIVE_CLAUSE": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST contain a relative clause starting with 'that' or 'which'. Keep queries under 25 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
            "MODIFIER_FRONTED": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST start with a fronted modifier (e.g., 'For {age}, ...' or 'In {category}, ...') followed by a comma. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
            "PREDICATE": f"Generate 4 search queries for {brand} {color} {age} product in {category} category. Each query MUST use copula 'is' (e.g., 'A {brand} product is ...'). Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
            "QUESTION": f"Generate 4 search queries for {brand} {color} {age} product in {category} category phrased as questions ending with '?'. Keep queries under 20 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
            "SIMPLE": f"Generate 4 simple search queries for {brand} {color} {age} product in {category} category without using relative clauses, coordination, or fronted modifiers. Keep queries under 15 words. Do NOT add adjectives like durable, lightweight, comfortable, perfect, modern, premium.",
        }

        asin_locked = {}
        if QwenLocalClient is None or SamplingParams is None:
            # Mock mode: just record prompts for now
            for cls, prompt in prompts_by_class.items():
                asin_locked[cls] = {"prompt": prompt, "generated": [], "mock": True}
            log(f"  [{ai+1}/{len(sorted_asins)}] {asin}: attrs={attrs} MOCK")
        else:
            # Lazy-create QwenLocalClient (loads vLLM in-process on first call)
            if not hasattr(step2_content_locked_generation, "_client"):
                log(f"  initializing QwenLocalClient (loads vLLM in-process, ~30s)...")
                step2_content_locked_generation._client = QwenLocalClient()
            client = step2_content_locked_generation._client

            # Batch all 7 classes × 4 variants = 28 prompts in one vLLM call (per Rule 9h)
            all_classes = list(prompts_by_class.keys())
            all_prompts = []
            for cls in all_classes:
                for k in range(4):
                    # Variant suffix to avoid vLLM dedup
                    p = prompts_by_class[cls] + f" (variant {k+1})"
                    all_prompts.append(p)

            # Apply chat template, then batched generate
            full_prompts = [
                client._backend.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in all_prompts
            ]
            sampling = SamplingParams(
                max_tokens=80, temperature=0.8,
                top_p=0.95 if 0.8 > 0 else 1.0,
                repetition_penalty=1.05,
            )
            try:
                vllm_outputs = client._backend.model.generate(full_prompts, sampling)
                outputs = [o.outputs[0].text.strip() if o.outputs else "" for o in vllm_outputs]
            except Exception as e:
                log(f"    ERROR batched vLLM generate: {e}")
                outputs = [""] * len(all_prompts)

            # Regroup outputs by class
            for ci, cls in enumerate(all_classes):
                generated = []
                for k in range(4):
                    idx = ci * 4 + k
                    out = outputs[idx] if idx < len(outputs) else ""
                    bias_hit = any(b in out.lower() for b in LLM_BIAS_BLACKLIST)
                    content_ok = all(c.lower() in out.lower() for c in [brand, color, age])
                    if bias_hit:
                        bias_hits_total += 1
                    if content_ok:
                        coverage_total += 1
                    generated.append({
                        "text": out.strip(),
                        "content_ok": content_ok,
                        "bias_hit": bias_hit,
                        "predicted_class": classify_structural(out, None),
                    })
                asin_locked[cls] = {"prompt": prompts_by_class[cls], "generated": generated}
            log(f"  [{ai+1}/{len(sorted_asins)}] {asin}: attrs={attrs}, {sum(len(v.get('generated', [])) for v in asin_locked.values())} cands")

        content_locked_pool[asin] = {
            "attrs": attrs,
            "by_class": asin_locked,
        }

    log(f"\n  Total bias hits: {bias_hits_total}/{n_sample_asins*7*4}")
    log(f"  Total content coverage: {coverage_total}/{n_sample_asins*7*4}")

    # Save
    with open(OUTPUT_CL_POOL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 11A: content-locked generation prototype",
                "n_asins": len(content_locked_pool),
                "structural_classes": STRUCTURAL_CLASSES,
                "bias_blacklist": LLM_BIAS_BLACKLIST,
            },
            "stats": {
                "bias_hits": bias_hits_total,
                "content_coverage": coverage_total,
            },
            "pool": content_locked_pool,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_CL_POOL}")
    return content_locked_pool


def step3_selection_comparison(struct_data, cl_pool):
    """Step 3: Selection comparison on (a) original pool, (b) content-locked pool, (c) balanced.

    Reuses Stage 10K's margin-based selection to compare.
    Focuses on:
      1. structural-class coverage per ASIN (does content-locked actually diversify?)
      2. pool diversity proxy (avg pairwise PCA48 distance)
      3. Selection: run mean_t50 + A1 reject-repeat on the new pool
    """
    log("\n=== Step 3: Selection comparison ===")

    # Load scaler + PCA48 + feat_map
    try:
        from gaussian_vades import _syntax_subspace_prepare  # noqa
    except ImportError:
        log("  WARN: _syntax_subspace_prepare not importable; falling back to surface metrics only")
        _syntax_subspace_prepare = None

    # Load original Stage 8.5 pool (for comparison)
    pool_8_5 = json.load(open(POOL_IN))
    pools_orig = pool_8_5["pools"]
    n_asins_orig = len(pools_orig)

    # Load Stage 8.5 selection (asin, user_id) pairs
    sel_data = json.load(open(STAGE8_5_SELECTION))
    asin_users = defaultdict(list)
    for e in sel_data["entries"]:
        asin_users[e["asin"]].append(e["user_id"])

    # Per-ASIN: compute coverage stats
    log("\n  --- Structural class coverage comparison ---")
    coverage_orig = {}  # asin -> {class -> count}
    coverage_cl = {}    # asin -> {class -> count}
    n_classes_orig = []  # per-ASIN number of distinct classes
    n_classes_cl = []

    for asin, queries in pools_orig.items():
        c = Counter(classify_structural(q.get("query", str(q)), None) for q in queries)
        coverage_orig[asin] = dict(c)
        n_classes_orig.append(len(c))

    for asin, data in cl_pool.items():
        c = Counter()
        for cls, info in data["by_class"].items():
            for g in info.get("generated", []):
                if g.get("text"):
                    # Use predicted_class from generation
                    pred = g.get("predicted_class", cls)
                    c[pred] += 1
        coverage_cl[asin] = dict(c)
        n_classes_cl.append(len(c))

    log(f"  Original Stage 8.5 pool: {np.mean(n_classes_orig):.2f} classes/asin mean ({sorted(set(n_classes_orig))})")
    log(f"  Content-locked pool:     {np.mean(n_classes_cl):.2f} classes/asin mean ({sorted(set(n_classes_cl))})")

    # Per-class hit rate (content-locked pool)
    log("\n  --- Per-class hit rate (content-locked) ---")
    class_hits = Counter()
    class_total = Counter()
    for asin, data in cl_pool.items():
        for cls, info in data["by_class"].items():
            for g in info.get("generated", []):
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

    # Bias analysis
    log("\n  --- Bias hit rate (content-locked) ---")
    bias_counter = Counter()
    n_gen_total = 0
    n_gen_nonempty = 0
    n_gen_no_bias = 0
    n_gen_content_ok = 0
    for asin, data in cl_pool.items():
        for cls, info in data["by_class"].items():
            for g in info.get("generated", []):
                n_gen_total += 1
                if not g.get("text"):
                    continue
                n_gen_nonempty += 1
                if g.get("bias_hit"):
                    text_low = g["text"].lower()
                    for b in LLM_BIAS_BLACKLIST:
                        if b in text_low:
                            bias_counter[b] += 1
                else:
                    n_gen_no_bias += 1
                if g.get("content_ok"):
                    n_gen_content_ok += 1
    log(f"  total generations: {n_gen_total}")
    log(f"  non-empty: {n_gen_nonempty} ({n_gen_nonempty/n_gen_total*100:.1f}%)")
    log(f"  no bias: {n_gen_no_bias}/{n_gen_nonempty} ({n_gen_no_bias/max(n_gen_nonempty,1)*100:.1f}%)")
    log(f"  content_ok: {n_gen_content_ok}/{n_gen_nonempty} ({n_gen_content_ok/max(n_gen_nonempty,1)*100:.1f}%)")
    log(f"  Top bias words:")
    for b, c in bias_counter.most_common(8):
        log(f"    {b!r}: {c}")

    # Selection comparison: mean_t50 + A1 reject-repeat on combined pool
    log("\n  --- Selection comparison (combined pool) ---")
    log(f"  Loading user Gaussians + scaler + PCA48...")

    g_data = json.load(open(USER_GAUSSIANS))
    base_users = g_data["users"]["base"]
    user_mu = {uid: np.array(u["mu"], dtype=np.float64) for uid, u in base_users.items()}
    user_sigma = {uid: np.array(u["sigma_diag"], dtype=np.float64) for uid, u in base_users.items()}
    log(f"  loaded {len(user_mu)} user Gaussians")

    # Project content-locked pool to 48d using feature cache
    feat_cache_path = SCRATCH / "stage7b_query_features.jsonl.gz"
    feat_map = {}
    with gzip.open(feat_cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map (initial cache): {len(feat_map)}")

    # Extract features for new queries via spaCy pipe
    all_new_queries = []
    for asin, data in cl_pool.items():
        for cls, info in data["by_class"].items():
            for g in info.get("generated", []):
                text = g.get("text", "").strip()
                if text:
                    all_new_queries.append(text)
    log(f"  new content-locked queries: {len(all_new_queries)}")

    # Check cache hits
    n_cache_hit = 0
    n_cache_miss = 0
    missing = []
    for q in all_new_queries:
        k = hashlib.sha1(q.strip().lower().encode("utf-8")).hexdigest()
        if k in feat_map:
            n_cache_hit += 1
        else:
            n_cache_miss += 1
            missing.append(q)
    log(f"  cache: hit={n_cache_hit}, miss={n_cache_miss}")
    if missing:
        try:
            import spacy
            sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")
            from main import per_sentence_features_v2
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            docs = list(nlp.pipe(missing, batch_size=64, n_process=1))
            new_records = []
            for q, doc in zip(missing, docs):
                feats = per_sentence_features_v2(doc)
                k = hashlib.sha1(q.strip().lower().encode("utf-8")).hexdigest()
                new_records.append({"k": k, "v": feats})
                feat_map[k] = feats
            with gzip.open(feat_cache_path, "at", encoding="utf-8") as f:
                for rec in new_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"  extracted & cached {len(new_records)} new features")
        except Exception as e:
            log(f"  WARN: feature extraction failed: {e}")

    # Load scaler + PCA48
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=48, random_state=2024)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # Project content-locked pool to 48d
    def project(qtext):
        k = hashlib.sha1(qtext.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k)
        if not feats:
            return None
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        return pca.transform(vec_scaled[None, :])[0]

    cl_z_by_asin = {}
    for asin, data in cl_pool.items():
        zs = []
        for cls, info in data["by_class"].items():
            for g in info.get("generated", []):
                text = g.get("text", "").strip()
                if text:
                    z = project(text)
                    zs.append({"text": text, "class_requested": cls, "predicted_class": g.get("predicted_class", cls), "z": z})
        cl_z_by_asin[asin] = zs

    # Selection: mean_t50 + A1 reject-repeat on combined (orig + cl) pool
    log("\n  --- Running selection on combined pool ---")
    n_asins_with_data = 0
    asin_metrics = {}

    for asin, users in asin_users.items():
        if asin not in cl_z_by_asin:
            continue
        cl_zs = cl_z_by_asin[asin]
        valid_zs = [z for z in cl_zs if z["z"] is not None]
        if len(valid_zs) < 3:
            continue
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue
        n_asins_with_data += 1
        z_arr = np.stack([z["z"] for z in valid_zs])
        mu_arr = np.stack([user_mu[uid] for uid in valid_users])
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])

        # D matrix
        diff = z_arr[:, None, :] - mu_arr[None, :, :]
        D = np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)
        n_q, n_u = D.shape

        # mean margin
        d_self = D.copy()
        margins = np.zeros((n_u, n_q))
        for ui in range(n_u):
            d_others = np.delete(D, ui, axis=1)
            margins[ui] = np.mean(d_others, axis=1) - d_self[:, ui]

        # mean_t50 selection (baseline)
        sel_mean_t50 = []
        for ui in range(n_u):
            d_self_u = d_self[:, ui]
            margin_u = margins[ui]
            threshold = float(np.percentile(d_self_u, 50))
            valid = d_self_u <= threshold
            m = np.where(valid, margin_u, -np.inf)
            sel_mean_t50.append(int(np.argmax(m)))

        # A1 reject-repeat
        sel_a1 = []
        picked = set()
        for ui in range(n_u):
            d_self_u = d_self[:, ui]
            margin_u = margins[ui]
            threshold = float(np.percentile(d_self_u, 50))
            valid = d_self_u <= threshold
            m = np.where(valid, margin_u, -np.inf)
            ranked = np.argsort(-m)
            chosen = None
            for c in ranked:
                if c not in picked:
                    chosen = int(c)
                    break
            if chosen is None:
                chosen = int(ranked[0])
            sel_a1.append(chosen)
            picked.add(chosen)

        # Random baseline (mean D per user)
        rnd_per_user = [float(np.mean(D[:, ui])) for ui in range(n_u)]
        rnd_dist = np.mean(rnd_per_user)

        # Selected distances
        sel_d_mean_t50 = [float(d_self[sel_mean_t50[ui], ui]) for ui in range(n_u)]
        sel_d_a1 = [float(d_self[sel_a1[ui], ui]) for ui in range(n_u)]

        # Class coverage of selected (A1)
        sel_classes = [valid_zs[sel_a1[ui]].get("predicted_class", "?") for ui in range(n_u)]
        class_diversity_a1 = len(set(sel_classes))

        asin_metrics[asin] = {
            "n_users": n_u,
            "n_cl_queries": len(valid_zs),
            "unique_mean_t50": len(set(sel_mean_t50)),
            "unique_a1": len(set(sel_a1)),
            "sel_dist_mean_t50": float(np.mean(sel_d_mean_t50)),
            "sel_dist_a1": float(np.mean(sel_d_a1)),
            "rnd_dist": float(rnd_dist),
            "sel_dist_a1_per_user": sel_d_a1,
            "class_diversity_a1": class_diversity_a1,
            "sel_classes_a1": sel_classes,
        }

    # Aggregate
    log(f"  asins with content-locked data + ≥2 users: {n_asins_with_data}")
    if n_asins_with_data == 0:
        log("  no asins with data — selection comparison skipped")
        return

    uniq_mean_t50 = [m["unique_mean_t50"] for m in asin_metrics.values()]
    uniq_a1 = [m["unique_a1"] for m in asin_metrics.values()]
    sel_a1_all = []
    rnd_all = []
    class_div_a1 = [m["class_diversity_a1"] for m in asin_metrics.values()]

    for m in asin_metrics.values():
        sel_a1_all.extend(m["sel_dist_a1_per_user"])
        rnd_all.extend([m["rnd_dist"]] * m["n_users"])

    log(f"\n  {'Metric':<25} {'value':>10}")
    log(f"  {'-'*40}")
    log(f"  unique_mean_t50 mean:   {np.mean(uniq_mean_t50):>10.2f}")
    log(f"  unique_a1 mean:         {np.mean(uniq_a1):>10.2f}")
    log(f"  unique_a1 ceiling:      {np.mean(uniq_a1)/max(np.max(uniq_a1),1)*100:>9.1f}%")
    log(f"  class_diversity_a1:     {np.mean(class_div_a1):>10.2f}")
    log(f"  sel_dist_a1 mean:       {np.mean(sel_a1_all):>10.2f}")
    log(f"  rnd_dist mean:          {np.mean(rnd_all):>10.2f}")
    log(f"  sel/rnd ratio:          {np.mean(sel_a1_all)/np.mean(rnd_all):>10.3f}")

    # Save comparison output
    comparison = {
        "config": {
            "description": "Stage 11C: content-locked vs original pool selection",
            "n_asins_with_data": n_asins_with_data,
            "n_classes": STRUCTURAL_CLASSES,
        },
        "structural_class_coverage": {
            "orig_pool_mean_classes_per_asin": float(np.mean(n_classes_orig)),
            "content_locked_mean_classes_per_asin": float(np.mean(n_classes_cl)),
            "per_class_hit_rate": {
                cls: class_hits[cls] / max(class_total[cls], 1) for cls in STRUCTURAL_CLASSES
            },
        },
        "selection_metrics": {
            "unique_mean_t50_mean": float(np.mean(uniq_mean_t50)),
            "unique_a1_mean": float(np.mean(uniq_a1)),
            "class_diversity_a1_mean": float(np.mean(class_div_a1)),
            "sel_dist_a1_mean": float(np.mean(sel_a1_all)),
            "rnd_dist_mean": float(np.mean(rnd_all)),
            "sel_rnd_ratio": float(np.mean(sel_a1_all) / max(np.mean(rnd_all), 1e-9)),
        },
        "bias": {
            "bias_counter": dict(bias_counter),
            "no_bias_rate": n_gen_no_bias / max(n_gen_nonempty, 1),
            "content_ok_rate": n_gen_content_ok / max(n_gen_nonempty, 1),
        },
        "per_asin": asin_metrics,
    }
    with open(OUTPUT_COMPARISON, "w") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {OUTPUT_COMPARISON}")


def main():
    log("=== Stage 11 — Content Lock + Structural Class + Selection ===")

    # Auto-skip Step 1/2 if outputs already exist (idempotent re-run)
    if OUTPUT_STRUCT.exists() and OUTPUT_CL_POOL.exists():
        log("  auto-skip Step 1/2 (outputs exist); reload + run Step 3 only")
        struct_data = json.load(open(OUTPUT_STRUCT))
        cl_pool_data = json.load(open(OUTPUT_CL_POOL))
        cl_pool = cl_pool_data["pool"]
    else:
        # Step 1: Structural class detector
        struct_data = step1_structural_detector()
        # Step 2: Content-locked generation
        cl_pool = step2_content_locked_generation(struct_data, n_sample_asins=20)

    # Step 3: Selection comparison
    step3_selection_comparison(struct_data, cl_pool)
    log("\n=== Stage 11 complete ===")


if __name__ == "__main__":
    main()