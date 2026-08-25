"""Stage 9B: Few-Shot Structural Diversification Pilot.

Stage 9A v1: diversity ok but template/extra-sem leakage (87.6% template, 29.1% extra-sem)
Stage 9A v2: 0% leakage but entropy collapse (1.5 mean/ASIN, all queries = attribute concat)

Stage 9B goal: few-shot structural diversification + semantic cleanliness simultaneously.

Approach:
1. Replace variant-number prompts with EXPLICIT structural family targets.
2. 8 structural families: NP-heavy, prepositional, coordination, clause, question,
   fragment, modifier-fronted, predicate-based.
3. Each query targets ONE different family; LLM varies syntax by following the target.
4. Output MUST NOT mention the family label.

New success criterion:
- QA three metrics: 0% (existing)
- per-ASIN strict pool ≥ 30
- per-ASIN structural families covered ≥ 5
- PCA48 pairwise distance not below Stage 8 baseline

Pilot: 10 ASINs × K=8 families × R=20 rounds = 1600 cands per ASIN

Output:
- stage9b_pool_pilot.json: pools with structural labels
- stage9b_diversity.json: per-ASIN family coverage + PCA48 stats

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9b_pilot.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9b_pilot.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import requests
import sys
from pathlib import Path
from typing import List

import numpy as np


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
OUT = SCRATCH / "stage9b_pool_pilot.json"
DIVERSITY_OUT = SCRATCH / "stage9b_diversity.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Pilot config ===
PILOT_ASIN_COUNT = 10
ROUNDS_PER_FAMILY = 60    # 60 cands per family per ASIN → 480 per ASIN before filter
TEMP = 1.0                # max temp for diversity
MAX_TOKENS = 40           # very short to force variation
SEED = 2024
BACKFILL_ROUNDS = 20      # per missing-family re-generate rounds

# === Quality thresholds ===
TEMPLATE_CAP_FRAC = 0.85  # very relaxed: per-family targets already ensure diversity
MAX_NEAR_DUP_JACCARD = 0.85


# === 8 structural families ===
STRUCTURAL_FAMILIES = [
    {
        "name": "np_heavy",
        "label": "noun-phrase-heavy",
        "instruction": (
            "Write a noun-phrase-heavy query (no verbs). Stack the attribute values "
            "as modifiers of a head noun. Keep it short and abstract."
        ),
        "example": "[Brand] [Color] [Size] [Item]",
    },
    {
        "name": "prepositional",
        "label": "prepositional",
        "instruction": (
            "Write a query using prepositions ('with', 'for', 'in', 'of') to relate "
            "the attribute values. Use 1-2 prepositional phrases. Abstract placeholders only."
        ),
        "example": "[Brand] [Color] for [Size] in [Item]",
    },
    {
        "name": "coordination",
        "label": "coordination",
        "instruction": (
            "Write a query using coordination (commas or 'and'). Abstract placeholders only."
        ),
        "example": "[Brand], [Color], [Size] [Item]",
    },
    {
        "name": "clause",
        "label": "subordinate-clause",
        "instruction": (
            "Write a query with a subordinate clause. Abstract placeholders only."
        ),
        "example": "[Brand] [Color] [Item] that has [Size]",
    },
    {
        "name": "question",
        "label": "question",
        "instruction": (
            "Write a short question-style search query. End with a question mark. "
            "Abstract placeholders only."
        ),
        "example": "[Brand] [Color] [Size]?",
    },
    {
        "name": "fragment",
        "label": "fragment",
        "instruction": (
            "Write a short fragment-style query (1-4 words). Drop articles and prepositions. "
            "Abstract placeholders only."
        ),
        "example": "[Brand] [Size]",
    },
    {
        "name": "modifier_fronted",
        "label": "modifier-fronted",
        "instruction": (
            "Write a query with the modifier attribute(s) BEFORE the head noun, "
            "separated by hyphens or just stacking. Abstract placeholders only."
        ),
        "example": "[Color]-[Size] [Brand] [Item]",
    },
    {
        "name": "predicate",
        "label": "predicate-based",
        "instruction": (
            "Write a query with a brief declarative/predicate structure. "
            "Abstract placeholders only."
        ),
        "example": "[Brand] [Item]: [Color], [Size]",
    },
]
N_FAMILIES = len(STRUCTURAL_FAMILIES)


# === Extra-semantic blacklist (unchanged from 9A) ===
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

# Field names that must NOT appear in the query (LLM leakage from canonical attrs)
FORBIDDEN_FIELD_NAMES = [
    "brand:", "color:", "size:", "style:", "material:", "item weight:",
    "age range", "main category:", "age range (description)", "pattern:",
    "format:", "item form:",
]


# === System prompt v9B ===
GEN_SYSTEM_TMPL_V9B = (
    "You are an Amazon shopper writing a SEARCH QUERY — the short text a person "
    "types into Amazon's search box. You must use EXACTLY the {N_INPUT} attribute "
    "values listed below verbatim (mention each value once). DO NOT add any quality, "
    "functional, aesthetic, emotional, or use-case adjective (no \"high-quality\", "
    "\"reliable\", \"durable\", \"premium\", \"modern\", \"stylish\", \"best\", "
    "\"perfect\", \"for baby\", \"newborn care\"). DO NOT include attribute FIELD "
    "NAMES — write 'Pampers' not 'Brand: Pampers'; the words 'Brand', 'Color', "
    "'Size', 'Item Weight', 'Age Range', 'Main Category', 'Material', 'Style' "
    "must NOT appear in your output.\n\n"
    "**NEVER start** with \"Looking for\", \"Searching for\", \"I am looking for\", "
    "\"I need\", \"Find me\", \"Show me\", \"Can you find\".\n\n"
    "**Structural target for THIS query**: {FAMILY_LABEL}\n"
    "{FAMILY_INSTRUCTION}\n"
    "Example shape uses abstract placeholders like [Brand], [Color], [Size], [Item]. "
    "DO NOT copy placeholder words into your output. Replace each [X] with the actual "
    "attribute value from the list below. Example: {FAMILY_EXAMPLE}\n\n"
    "**Do NOT mention the structural target, family label, or any instruction text "
    "in your output.** Output ONLY the query.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, family_idx: int) -> str:
    fam = STRUCTURAL_FAMILIES[family_idx]
    return GEN_SYSTEM_TMPL_V9B.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
        FAMILY_LABEL=fam["label"],
        FAMILY_INSTRUCTION=fam["instruction"],
        FAMILY_EXAMPLE=fam["example"],
    )


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
    bad = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:", r"^color:", r"^material:", r"^style:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad)


def has_extra_semantic(text: str) -> bool:
    text_lower = text.lower()
    for cat, patterns in EXTRA_SEMANTIC_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return True
    return False


def has_template_opening(text: str) -> bool:
    text_strip = text.strip()
    return any(re.match(t, text_strip, re.IGNORECASE) for t in TEMPLATE_OPENINGS)


def has_forbidden_field_name(text: str) -> bool:
    text_lower = text.lower()
    return any(f in text_lower for f in FORBIDDEN_FIELD_NAMES)


def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    log("=== Stage 9B: Few-Shot Structural Diversification Pilot ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pilot_asins = asin_data[:PILOT_ASIN_COUNT]
    log(f"  Pilot ASINs: {[a['asin'] for a in pilot_asins]}")
    log(f"  Config: {N_FAMILIES} families × {ROUNDS_PER_FAMILY} rounds = "
        f"{N_FAMILIES * ROUNDS_PER_FAMILY} cands per ASIN")

    # === Build prompts (round-robin per ASIN × family) ===
    log("Building prompts...")
    all_prompts = []
    asin_family_idx = []
    for entry in pilot_asins:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        for family_idx in range(N_FAMILIES):
            for round_i in range(ROUNDS_PER_FAMILY):
                all_prompts.append(make_prompt(attrs, n_input, family_idx))
                asin_family_idx.append((a, family_idx))
    log(f"  Total prompts: {len(all_prompts)}")

    # === Generate ===
    log(f"Generating via vLLM (temp={TEMP}, max_tokens={MAX_TOKENS})...")
    all_outputs = batch_generate_vllm(all_prompts)
    log(f"  Got {len(all_outputs)} outputs")

    # === Per-query quality flagging ===
    raw = []
    for (a, family_idx), out in zip(asin_family_idx, all_outputs):
        entry = next(e for e in pilot_asins if e["asin"] == a)
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        text = out.strip() if out else ""
        fam_name = STRUCTURAL_FAMILIES[family_idx]["name"]
        raw.append({
            "asin": a, "family": fam_name, "query": text,
            "n_cov": count_attrs_covered(text, attrs),
            "invalid": has_invalid_punct(text),
            "extra": has_extra_semantic(text),
            "template": has_template_opening(text),
            "field_name": has_forbidden_field_name(text),
            "n_tok": len(text.split()) if text else 0,
        })

    # === Per-ASIN filter + collect ===
    pools = collections.defaultdict(list)
    layer_stats = collections.Counter()
    family_counts_per_asin = collections.defaultdict(collections.Counter)

    for r in raw:
        asin = r["asin"]
        text = r["query"]
        if not text:
            layer_stats["empty"] += 1
            continue
        if r["n_cov"] != len(next(e for e in pilot_asins if e["asin"] == asin)["attrs_used"]):
            layer_stats["L1_reject_attrs"] += 1
            continue
        if r["invalid"]:
            layer_stats["L2_reject_punct"] += 1
            continue
        if r["extra"]:
            layer_stats["L3_reject_extra_sem"] += 1
            continue
        if r["template"]:
            layer_stats["L4_reject_template"] += 1
            continue
        if r["field_name"]:
            layer_stats["L4b_reject_field_name"] += 1
            continue
        # Dedup
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
        # Accept
        family_counts_per_asin[asin][r["family"]] += 1
        pools[asin].append({
            "family": r["family"],
            "query": text,
            "strict": True,
            "n_tok": r["n_tok"],
            "attrs_covered": r["n_cov"],
        })
        layer_stats["ACCEPT"] += 1

    # === Stats ===
    log(f"\n=== Layer filter stats ===")
    for k, v in layer_stats.most_common():
        log(f"  {k}: {v}")

    log(f"\n=== Per-ASIN pool stats ===")
    for asin, qs in pools.items():
        n = len(qs)
        if n == 0:
            continue
        fc = family_counts_per_asin[asin]
        log(f"  {asin}: n={n}, families: {dict(fc)}")

    avg_pool = sum(len(qs) for qs in pools.values()) / max(1, len(pools))
    log(f"  Mean pool size: {avg_pool:.1f}")

    # === Per-family minimum quota backfill ===
    # For each ASIN, identify families with 0 strict; re-generate with higher temp
    # + variant hint; add to pool (allow ≥1 per missing family).
    log(f"\n=== Per-family quota backfill ===")
    backfill_prompts = []
    backfill_meta = []  # (asin, family_idx)
    for entry in pilot_asins:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        covered = set(family_counts_per_asin[a].keys())
        missing = [i for i in range(N_FAMILIES) if STRUCTURAL_FAMILIES[i]["name"] not in covered]
        if not missing:
            continue
        for family_idx in missing:
            for r in range(BACKFILL_ROUNDS):
                p = make_prompt(attrs, n_input, family_idx)
                # Append variant hint as suffix (won't be echoed back into query if we keep it inside the system prompt)
                backfill_prompts.append(p)
                backfill_meta.append((a, family_idx))

    if backfill_prompts:
        log(f"  Backfilling {len(backfill_prompts)} prompts (missing families)...")
        backfill_outputs = batch_generate_vllm(backfill_prompts, temp=1.0, max_tokens=40)
        n_backfill_added = 0
        for (a, family_idx), out in zip(backfill_meta, backfill_outputs):
            entry = next(e for e in pilot_asins if e["asin"] == a)
            attrs = entry["attrs_used"]
            text = (out or "").strip()
            if not text:
                continue
            if count_attrs_covered(text, attrs) != len(attrs):
                continue
            if has_invalid_punct(text):
                continue
            if has_extra_semantic(text):
                continue
            if has_template_opening(text):
                continue
            if has_forbidden_field_name(text):
                continue
            # Dedup vs existing
            existing = pools[a]
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
                continue
            fam_name = STRUCTURAL_FAMILIES[family_idx]["name"]
            pools[a].append({
                "family": fam_name,
                "query": text,
                "strict": True,
                "n_tok": len(text.split()),
                "attrs_covered": len(attrs),
            })
            family_counts_per_asin[a][fam_name] += 1
            n_backfill_added += 1
        log(f"  Backfill added {n_backfill_added} queries")
        avg_pool = sum(len(qs) for qs in pools.values()) / max(1, len(pools))
        log(f"  Post-backfill mean pool size: {avg_pool:.1f}")
    else:
        log(f"  No missing families — skipping backfill")

    # === Family coverage metric ===
    log(f"\n=== Family coverage per ASIN ===")
    n_families_target = 5
    asins_meeting_target = 0
    for asin, qs in pools.items():
        if not qs:
            continue
        covered_families = set(q["family"] for q in qs)
        meets = len(covered_families) >= n_families_target
        if meets:
            asins_meeting_target += 1
        log(f"  {asin}: {len(covered_families)}/{N_FAMILIES} families, "
            f"target ≥{n_families_target}: {'PASS' if meets else 'FAIL'}")

    log(f"  ASINs meeting family ≥{n_families_target}: {asins_meeting_target}/{len(pools)}")

    # === Re-audit QA ===
    n_extra = 0
    n_template = 0
    n_field = 0
    n_total = 0
    for asin, qs in pools.items():
        for q in qs:
            n_total += 1
            if has_extra_semantic(q["query"]):
                n_extra += 1
            if has_template_opening(q["query"]):
                n_template += 1
            if has_forbidden_field_name(q["query"]):
                n_field += 1
    log(f"\n=== Final QA (post-filter re-audit) ===")
    log(f"  total strict: {n_total}")
    log(f"  with extra-sem: {n_extra} ({n_extra/max(1,n_total)*100:.1f}%)")
    log(f"  with template opening: {n_template} ({n_template/max(1,n_total)*100:.1f}%)")
    log(f"  with field-name leak: {n_field} ({n_field/max(1,n_total)*100:.1f}%)")

    n_dup = 0; n_pairs = 0
    for asin, qs in pools.items():
        toks_list = [tokenize(q["query"]) for q in qs]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                n_pairs += 1
                if qs[i]["query"] == qs[j]["query"] or jaccard(toks_list[i], toks_list[j]) >= MAX_NEAR_DUP_JACCARD:
                    n_dup += 1
    log(f"  near-dup pairs: {n_dup}/{n_pairs} = {n_dup/max(1,n_pairs)*100:.1f}%")

    # === PCA48 diversity check ===
    log(f"\n=== PCA48 diversity check ===")
    # Collect all unique 9B queries, batch-extract 182d features via spaCy
    all_9b_queries = []
    seen_q = set()
    for asin, qs in pools.items():
        for q in qs:
            t = q["query"].strip().lower()
            if t and t not in seen_q:
                seen_q.add(t)
                all_9b_queries.append(t)
    log(f"  Unique 9B queries: {len(all_9b_queries)}")

    import spacy
    nlp = spacy.load("en_core_web_sm")
    from main import per_sentence_features_v2  # noqa
    log(f"  Extracting 182d features via spaCy pipe...")
    feats_9b = {}
    for i, doc in enumerate(nlp.pipe(all_9b_queries, batch_size=64)):
        for sent in doc.sents:
            sf = per_sentence_features_v2(sent)
            if sf:
                feats_9b[all_9b_queries[i]] = sf
                break
    log(f"  Feature extractions: {len(feats_9b)}")

    # Load PCA48 once (frozen)
    from gaussian_vades import _syntax_subspace_prepare
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 loaded (1× only)")

    def project(qtext: str):
        sf = feats_9b.get(qtext.strip().lower())
        if not sf:
            return None
        vec = np.array([sf.get(n, 0.0) for n in fnames], dtype=np.float64)
        return pca.transform(scaler.transform(vec[None, :]))[0]

    diversity_stats = {"per_asin": {}, "global": {}}
    all_dists_9b = []
    for asin, qs in pools.items():
        if len(qs) < 2:
            continue
        zs = []
        miss = 0
        for q in qs:
            z = project(q["query"])
            if z is None:
                miss += 1
                continue
            zs.append(z)
        if len(zs) < 2:
            continue
        zs = np.array(zs)
        dists = pdist(zs)
        all_dists_9b.extend(dists.tolist())
        diversity_stats["per_asin"][asin] = {
            "n_pool": len(zs),
            "n_features_missing": miss,
            "mean_dist": float(dists.mean()),
            "median_dist": float(np.median(dists)),
            "std_dist": float(dists.std()),
            "n_families_covered": len(set(q["family"] for q in qs)),
        }
        log(f"  {asin}: n={len(zs)}, mean_dist={dists.mean():.2f}, "
            f"families={len(set(q['family'] for q in qs))}")

    if all_dists_9b:
        all_dists_9b = np.array(all_dists_9b)
        diversity_stats["global"] = {
            "mean_pairwise_dist": float(all_dists_9b.mean()),
            "median_pairwise_dist": float(np.median(all_dists_9b)),
            "std_pairwise_dist": float(all_dists_9b.std()),
            "n_pairs": int(len(all_dists_9b)),
        }
        log(f"  Global mean pairwise dist: {all_dists_9b.mean():.2f}")

    # === Save ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9B pilot: 10 ASINs × 8 families × 25 rounds = 200 cands/ASIN",
                "PILOT_ASIN_COUNT": PILOT_ASIN_COUNT,
                "N_FAMILIES": N_FAMILIES,
                "ROUNDS_PER_FAMILY": ROUNDS_PER_FAMILY,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "FAMILIES": [f["name"] for f in STRUCTURAL_FAMILIES],
            },
            "layer_stats": dict(layer_stats),
            "pool_sizes": {a: len(qs) for a, qs in pools.items()},
            "family_counts": {a: dict(fc) for a, fc in family_counts_per_asin.items()},
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")

    with open(DIVERSITY_OUT, "w", encoding="utf-8") as f:
        json.dump(diversity_stats, f, ensure_ascii=False, indent=2)
    log(f"wrote → {DIVERSITY_OUT}")


if __name__ == "__main__":
    main()