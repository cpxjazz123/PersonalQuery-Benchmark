"""Stage 9C: Syntactic Composition Repair — Pilot on 10 ASINs.

Stage 9A v2 (commit 7e0afa8): 0% QA but entropy collapse (1.5 pool, attribute concatenation)
Stage 9B v5 (commit 392b5d4): 0% QA + 4-family real diversity, BUT attribute-stack
        "Pampers White Toddler Health & Personal Care item" too flat.

Stage 9C goal: clean semantics + RICH grammatical composition (no flat attribute stack).

Approach:
1. Prompt: explicit anti-stack rule + enforce grammatical connectors
2. 8 families rewritten as GRONATICAL RELATIONS, not shape names:
   - Prepositional: must use `for/with/in/under`
   - Coordination: must use `and/while/as well as` (with commas)
   - Relative clause: must include `that/which`
   - Subordinate clause: must have main + subordinate verb
   - Predicate-based: complete predicate `is intended for / designed for`
   - Modifier-fronted: attributes pre-modify head noun
   - Question: natural interrogative
   - Fragment: short but with at least 1 connector (no pure noun stack)
3. New QA metric: attribute-stack rate (target <10%)
4. New L7 connector-presence filter: each query must satisfy ≥1 of:
   - preposition, coordination, finite verb, clause, modifier
5. Stricter max_tokens=80 to allow more complex structures

Pilot: 10 ASINs × 8 families × 60 rounds + backfill

Output:
- stage9c_pool_pilot.json: pool with grammatical-relation labels
- stage9c_qa.json: per-family connector + stack rate
"""

from __future__ import annotations

import collections
import json
import re
import requests
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
OUT = SCRATCH / "stage9c_pool_pilot.json"
QA_OUT = SCRATCH / "stage9c_qa.json"

VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/ar57_scratch/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Pilot config ===
PILOT_ASIN_COUNT = 10
ROUNDS_PER_FAMILY = 60    # 60 cands per family per ASIN
BACKFILL_ROUNDS = 25      # re-gen for missing families
TEMP = 1.0
MAX_TOKENS = 80           # longer to allow complex structures
SEED = 2024

# === Quality thresholds ===
MAX_NEAR_DUP_JACCARD = 0.85
ATTRIBUTE_STACK_RATE_TARGET = 0.10   # <10% queries may be pure attr stack
MIN_CONNECTORS_PER_QUERY = 1          # ≥1 of: prep/coord/verb/clause/modifier


# === 8 GRAMMATICAL RELATION families ===
# Each specifies exact connector rules
GRAMMATICAL_FAMILIES = [
    {
        "name": "prepositional",
        "label": "prepositional-phrase",
        "instruction": (
            "Use prepositions to relate the attributes. MUST include at least 2 "
            "different prepositions from this set: 'for', 'with', 'in', 'under', "
            "'by', 'of'. Example: 'Pampers item in white for toddlers, under "
            "Health & Personal Care.'"
        ),
        "example": "[Brand] item in [Color] for [Age] under [Category]",
        "required_prep_min": 2,
    },
    {
        "name": "coordination",
        "label": "coordination",
        "instruction": (
            "Connect the attributes with coordination. MUST use commas AND/OR "
            "the conjunction 'and' or 'while' or 'as well as' to coordinate at "
            "least 2 clauses/phrases. Example: 'Pampers in white, and designed "
            "for toddlers as well as Health & Personal Care.'"
        ),
        "example": "[Brand] in [Color], and for [Age] as well as [Category]",
        "required_coord_min": 2,
    },
    {
        "name": "relative_clause",
        "label": "relative-clause",
        "instruction": (
            "MUST include a relative clause using 'that' or 'which'. Example: "
            "'A Pampers product that comes in white, which is designed for "
            "toddlers in Health & Personal Care.'"
        ),
        "example": "[Brand] [Item] that has [Color], which fits [Age] in [Category]",
        "required_relcl_min": 1,
    },
    {
        "name": "subordinate_clause",
        "label": "subordinate-clause",
        "instruction": (
            "MUST have a main clause and a subordinate clause connected by "
            "'because', 'since', 'although', 'when', 'if', or 'while'. Example: "
            "'I want a Pampers product in white because toddlers need Health & "
            "Personal Care items.'"
        ),
        "example": "[Brand] [Color] works when [Age] needs [Category] care",
        "required_advcl_min": 1,
    },
    {
        "name": "predicate_based",
        "label": "predicate-based",
        "instruction": (
            "MUST use a complete predicate structure with 'is', 'are', 'has', "
            "'offers', 'designed for', 'intended for', or 'suitable for'. "
            "Example: 'Pampers is a white product designed for toddlers in "
            "Health & Personal Care.'"
        ),
        "example": "[Brand] is [Color] [Item] designed for [Age] in [Category]",
        "required_predicate_min": 1,
    },
    {
        "name": "modifier_fronted",
        "label": "modifier-fronted",
        "instruction": (
            "Front one or two attributes as modifiers before the head noun, "
            "joined with hyphens or stacked adjectives. Example: 'White-toddler "
            "Pampers item in Health & Personal Care.'"
        ),
        "example": "[Color]-[Age] [Brand] [Item] in [Category]",
        "required_modifier_min": 1,
    },
    {
        "name": "question",
        "label": "question",
        "instruction": (
            "MUST be a natural interrogative ending with '?', using a question "
            "word like 'which', 'what', 'where', 'is there', or just an inverted "
            "subject-verb structure. Example: 'Which Pampers product in white "
            "is suitable for toddlers in Health & Personal Care?'"
        ),
        "example": "Which [Brand] [Color] suits [Age] in [Category]?",
        "required_question_min": 1,
    },
    {
        "name": "fragment_with_connector",
        "label": "fragment-with-connector",
        "instruction": (
            "Short fragment (4-8 words) but MUST include at least 1 grammatical "
            "connector (preposition, conjunction, or verb). Pure noun stacking "
            "is FORBIDDEN. Example: 'Pampers for white toddlers' or 'Pampers in "
            "Health & Personal Care.'"
        ),
        "example": "[Brand] for [Age] in [Category]",
        "required_connector_min": 1,
    },
]
N_FAMILIES = len(GRAMMATICAL_FAMILIES)


# === Extra-semantic blacklist (unchanged) ===
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

# === Forbidden field names (LLM leakage) ===
FORBIDDEN_FIELD_NAMES = [
    "brand:", "color:", "size:", "style:", "material:", "item weight:",
    "age range", "main category:", "age range (description)", "pattern:",
    "format:", "item form:",
]


# === System prompt v9C: anti-stack + connector enforcement ===
GEN_SYSTEM_TMPL_V9C = (
    "You are an Amazon shopper writing a SEARCH QUERY — the short text a person "
    "types into Amazon's search box. You must use EXACTLY the {N_INPUT} attribute "
    "values listed below verbatim (mention each value once).\n\n"
    "**CRITICAL: Do NOT simply concatenate the attribute values into a flat noun "
    "phrase**. Strings like 'Pampers White Toddler Health & Personal Care item' "
    "are FORBIDDEN. You MUST connect the attributes using grammatical relations "
    "(prepositions, conjunctions, verbs, relative clauses, modifier attachment) "
    "to form a natural short sentence.\n\n"
    "**Semantics stay exactly the same as the attribute list**. No adding "
    "quality / functional / aesthetic / emotional / use-case adjectives (no "
    "\"high-quality\", \"reliable\", \"modern\", \"stylish\", \"perfect\", \"for "
    "baby\"). DO NOT include attribute FIELD NAMES (no 'Brand:', 'Color:' etc). "
    "Words 'Brand', 'Color', 'Size', 'Age Range', 'Main Category', 'Material', "
    "'Style' must NOT appear.\n\n"
    "**NEVER start** with \"Looking for\", \"Searching for\", \"I am looking "
    "for\", \"I need\", \"Find me\", \"Show me\", \"Can you find\".\n\n"
    "**Grammatical target for THIS query**: {FAMILY_LABEL}\n"
    "{FAMILY_INSTRUCTION}\n"
    "Example shape uses abstract placeholders like [Brand], [Color], [Age], "
    "[Category]. DO NOT copy placeholder words into your output — replace each "
    "[X] with the actual attribute value. Example: {FAMILY_EXAMPLE}\n\n"
    "**Do NOT mention the grammatical target, family label, or any instruction "
    "text in your output.** Output ONLY the query.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, family_idx: int) -> str:
    fam = GRAMMATICAL_FAMILIES[family_idx]
    return GEN_SYSTEM_TMPL_V9C.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
        FAMILY_LABEL=fam["label"],
        FAMILY_INSTRUCTION=fam["instruction"],
        FAMILY_EXAMPLE=fam["example"],
    )


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    """vLLM batched generation using inline URL (works around vLLM 0.27.1 routing bug
    where requests.post(MODULE_URL, ...) intermittently returns 404)."""
    outputs = []
    full_prompts = [f"system\n{p}\nuser\n\nassistant\n" for p in prompts]
    bs = 32
    url = "http://localhost:8800/v1/completions"
    for i in range(0, len(prompts), bs):
        chunk = full_prompts[i: i + bs]
        try:
            resp = requests.post(
                url,
                json={
                    "model": "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
                    "prompt": chunk,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95 if temp > 0 else 1.0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            for choice in resp.json()["choices"]:
                outputs.append(choice["text"].strip())
        except Exception as e:
            log(f"  batch {i} error: {e!r}")
            outputs.extend([""] * len(chunk))
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


# === New: syntactic connector detection via spaCy ===
def analyze_connectors(text: str, nlp) -> dict:
    """Parse query and return connector presence + structure counts.

    Strict counts: function words only, not punctuation/symbols (& / , / -).
    - n_prep: real prepositions (for/with/in/under/of/by/at/on)
    - n_coord: real coordinating conjunctions (and/while/but/or/because/since)
    - n_clause: relative/subordinate clauses
    - n_modifier: adjective/adverbial modifiers
    - n_verb: finite verbs (not aux)
    - n_det: determiners (a/an/the)
    """
    if not text or not text.strip():
        return {"n_prep": 0, "n_coord": 0, "n_clause": 0, "n_modifier": 0,
                "n_verb": 0, "n_det": 0, "is_question": False, "n_tokens": 0,
                "has_connector": False, "pos_seq": []}
    doc = nlp(text)
    n_prep = 0
    n_coord = 0
    n_clause = 0
    n_modifier = 0
    n_verb = 0
    n_det = 0
    pos_seq = []
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        tokens.append(tok)
        pos_seq.append(tok.pos_)
        # Prep: only real preposition words (text level)
        if tok.pos_ == "ADP" or tok.dep_ == "prep":
            if tok.text.lower() in ("for", "with", "in", "under", "of", "by",
                                     "at", "on", "from", "to", "about",
                                     "into", "onto", "upon", "within", "without"):
                n_prep += 1
        # Coord: real conjunction words (not & symbol)
        if tok.pos_ == "CCONJ" or tok.pos_ == "SCONJ":
            if tok.text.lower() in ("and", "while", "but", "or", "because",
                                     "since", "although", "when", "if", "so",
                                     "though", "whereas", "as"):
                n_coord += 1
        # Clause
        if tok.dep_ in ("acl", "relcl", "advcl", "ccomp", "xcomp"):
            n_clause += 1
        # Modifier
        if tok.dep_ in ("amod", "advmod", "nmod", "appos", "nummod"):
            n_modifier += 1
        # Finite verb (not aux)
        if tok.pos_ == "VERB" and tok.dep_ not in ("aux", "auxpass"):
            n_verb += 1
        # Determiner
        if tok.pos_ == "DET":
            n_det += 1
    is_question = text.rstrip().endswith("?")
    has_connector = (n_prep + n_coord + n_clause + n_verb + n_det) >= MIN_CONNECTORS_PER_QUERY
    return {
        "n_prep": n_prep,
        "n_coord": n_coord,
        "n_clause": n_clause,
        "n_modifier": n_modifier,
        "n_verb": n_verb,
        "n_det": n_det,
        "is_question": is_question,
        "n_tokens": len(tokens),
        "has_connector": has_connector,
        "pos_seq": pos_seq,
    }


def is_attribute_stack(text: str, attrs: dict, nlp) -> bool:
    """Detect: query is mostly contiguous attribute values with no grammatical connector.

    Strict heuristic: must have NO real grammatical connector (function word).
    "Pampers White Toddler Health & Personal Care" is stack (no prep/verb/det/coord word).
    "Pampers item in white for toddlers" is NOT stack (has prep "in", "for").
    "Pampers White and Toddler Health" is NOT stack (coord word "and" present).

    Function-word detection:
    - Has at least 1 preposition (case/prep excluding "&", punctuation)
    - OR has at least 1 conjunction word (and/while/but/or — text-level, not just spaCy dep)
    - OR has finite verb (root verb, not aux)
    - OR has determiner (DET)
    - OR has at least 1 relative/subordinate marker (that/which/because/since/although)
    - OR is a question (?)

    Plus: high attribute coverage + pos sequence mostly content words (NOUN/PROPN/ADJ)
    """
    if not text:
        return True
    text_lower = text.lower().strip()
    attrs_covered = sum(1 for v in attrs.values() if v and str(v).lower() in text_lower)
    if attrs_covered < max(1, int(0.8 * len(attrs))):
        return False  # attribute stack requires high coverage
    # Is question?
    if text_lower.endswith("?"):
        return False
    # Function-word check (text-level, robust to spaCy mis-parses)
    has_function_word = False
    function_markers = [
        # prepositions
        r"\bin\b", r"\bfor\b", r"\bwith\b", r"\bunder\b", r"\bof\b", r"\bby\b",
        r"\bat\b", r"\bon\b", r"\bfrom\b", r"\bto\b", r"\babout\b",
        # conjunctions (words, not symbols)
        r"\band\b", r"\bwhile\b", r"\bbut\b", r"\bor\b", r"\bbecause\b",
        r"\bsince\b", r"\balthough\b", r"\bwhen\b", r"\bif\b", r"\bso\b",
        # relative / subordinate markers
        r"\bthat\b", r"\bwhich\b", r"\bwho\b", r"\bwhom\b",
        # determiners (a, an, the) — even single article breaks stack
        r"\ba\b", r"\ban\b", r"\bthe\b",
    ]
    for pat in function_markers:
        if re.search(pat, text_lower):
            has_function_word = True
            break
    if has_function_word:
        return False
    # No function word at all → check pos sequence is mostly content
    doc = nlp(text)
    content_pos = 0
    total = 0
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        total += 1
        if tok.pos_ in ("NOUN", "PROPN", "ADJ", "NUM", "X"):
            content_pos += 1
    if total == 0:
        return True
    content_frac = content_pos / total
    return content_frac >= 0.85


def main():
    import spacy
    nlp = spacy.load("en_core_web_sm")

    log("=== Stage 9C: Syntactic Composition Repair Pilot ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pilot_asins = asin_data[:PILOT_ASIN_COUNT]
    log(f"  Pilot ASINs: {[a['asin'] for a in pilot_asins]}")
    log(f"  Config: {N_FAMILIES} families × {ROUNDS_PER_FAMILY} rounds = "
        f"{N_FAMILIES * ROUNDS_PER_FAMILY} cands per ASIN")

    # === Build prompts ===
    log("Building prompts...")
    all_prompts = []
    asin_family_idx = []
    for entry in pilot_asins:
        a = entry["asin"]
        n_input = len(entry["attrs_used"])
        for family_idx in range(N_FAMILIES):
            for round_i in range(ROUNDS_PER_FAMILY):
                all_prompts.append(make_prompt(entry["attrs_used"], n_input, family_idx))
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
        text = (out or "").strip()
        fam_name = GRAMMATICAL_FAMILIES[family_idx]["name"]
        raw.append({
            "asin": a,
            "family": fam_name,
            "query": text,
            "n_cov": count_attrs_covered(text, attrs),
            "invalid": has_invalid_punct(text),
            "extra": has_extra_semantic(text),
            "template": has_template_opening(text),
            "field_name": has_forbidden_field_name(text),
            "n_tok": len(text.split()) if text else 0,
        })

    # === Per-ASIN filter ===
    pools = collections.defaultdict(list)
    layer_stats = collections.Counter()
    family_counts_per_asin = collections.defaultdict(collections.Counter)

    # Pre-compute connector analysis for each raw query (spaCy pipe for speed)
    log("Running spaCy connector analysis on all queries...")
    raw_texts = [r["query"] for r in raw]
    connector_map = {}
    stack_map = {}
    attrs_lookup = {e["asin"]: e["attrs_used"] for e in pilot_asins}
    for i, doc in enumerate(nlp.pipe(raw_texts, batch_size=128)):
        text = raw_texts[i]
        asin = raw[i]["asin"]
        attrs = attrs_lookup[asin]
        # Re-parse to capture doc
        conn = analyze_connectors(text, nlp)
        connector_map[i] = conn
        stack_map[i] = is_attribute_stack(text, attrs, nlp)
        # Update the raw entry for later backfill
        raw[i]["_conn"] = conn
        raw[i]["_is_stack"] = stack_map[i]
    log(f"  Connector analysis done")

    for i, r in enumerate(raw):
        asin = r["asin"]
        text = r["query"]
        if not text:
            layer_stats["empty"] += 1
            continue
        attrs = attrs_lookup[asin]
        n_input = len(attrs)
        if r["n_cov"] != n_input:
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
        # L5: dedup
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
        # L6: connector-presence (NEW)
        if not r["_conn"]["has_connector"]:
            layer_stats["L6_reject_no_connector"] += 1
            continue
        # L7: anti-stack (NEW)
        if r["_is_stack"]:
            layer_stats["L7_reject_attr_stack"] += 1
            continue
        # Accept
        family_counts_per_asin[asin][r["family"]] += 1
        pools[asin].append({
            "family": r["family"],
            "query": text,
            "strict": True,
            "n_tok": r["n_tok"],
            "attrs_covered": r["n_cov"],
            "n_prep": r["_conn"]["n_prep"],
            "n_coord": r["_conn"]["n_coord"],
            "n_clause": r["_conn"]["n_clause"],
            "n_modifier": r["_conn"]["n_modifier"],
            "n_verb": r["_conn"]["n_verb"],
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

    # === Per-family quota backfill ===
    log(f"\n=== Per-family quota backfill ===")
    backfill_prompts = []
    backfill_meta = []
    for entry in pilot_asins:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        covered = set(family_counts_per_asin[a].keys())
        missing = [i for i in range(N_FAMILIES) if GRAMMATICAL_FAMILIES[i]["name"] not in covered]
        if not missing:
            continue
        for family_idx in missing:
            for r in range(BACKFILL_ROUNDS):
                backfill_prompts.append(make_prompt(attrs, n_input, family_idx))
                backfill_meta.append((a, family_idx))
    log(f"  Backfilling {len(backfill_prompts)} prompts (missing families)...")
    if backfill_prompts:
        backfill_outputs = batch_generate_vllm(backfill_prompts, temp=1.0, max_tokens=MAX_TOKENS)
        n_added = 0
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
            conn = analyze_connectors(text, nlp)
            if not conn["has_connector"]:
                continue
            if is_attribute_stack(text, attrs, nlp):
                continue
            fam_name = GRAMMATICAL_FAMILIES[family_idx]["name"]
            pools[a].append({
                "family": fam_name,
                "query": text,
                "strict": True,
                "n_tok": len(text.split()),
                "attrs_covered": len(attrs),
                "n_prep": conn["n_prep"],
                "n_coord": conn["n_coord"],
                "n_clause": conn["n_clause"],
                "n_modifier": conn["n_modifier"],
                "n_verb": conn["n_verb"],
            })
            family_counts_per_asin[a][fam_name] += 1
            n_added += 1
        log(f"  Backfill added {n_added} queries")
        avg_pool = sum(len(qs) for qs in pools.values()) / max(1, len(pools))
        log(f"  Post-backfill mean pool size: {avg_pool:.1f}")

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

    # === Final QA: extra-sem / template / field / attribute-stack rate ===
    log(f"\n=== Final QA (post-filter re-audit) ===")
    n_total = sum(len(qs) for qs in pools.values())
    n_extra = 0
    n_template = 0
    n_field = 0
    n_stack = 0
    n_no_connector = 0
    for asin, qs in pools.items():
        attrs = attrs_lookup[asin]
        for q in qs:
            text = q["query"]
            if has_extra_semantic(text):
                n_extra += 1
            if has_template_opening(text):
                n_template += 1
            if has_forbidden_field_name(text):
                n_field += 1
            if is_attribute_stack(text, attrs, nlp):
                n_stack += 1
            conn = analyze_connectors(text, nlp)
            if not conn["has_connector"]:
                n_no_connector += 1

    log(f"  total strict: {n_total}")
    log(f"  with extra-sem: {n_extra} ({n_extra/max(1,n_total)*100:.1f}%)")
    log(f"  with template opening: {n_template} ({n_template/max(1,n_total)*100:.1f}%)")
    log(f"  with field-name leak: {n_field} ({n_field/max(1,n_total)*100:.1f}%)")
    log(f"  **attribute-stack rate: {n_stack} ({n_stack/max(1,n_total)*100:.1f}%) [target <{ATTRIBUTE_STACK_RATE_TARGET*100:.0f}%]**")
    log(f"  no connector: {n_no_connector} ({n_no_connector/max(1,n_total)*100:.1f}%)")

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

    # === Per-family connector stats ===
    log(f"\n=== Per-family connector analysis ===")
    fam_stats = collections.defaultdict(lambda: {"count": 0, "n_prep_total": 0, "n_coord_total": 0,
                                                  "n_clause_total": 0, "n_modifier_total": 0, "n_verb_total": 0})
    for asin, qs in pools.items():
        for q in qs:
            f = q["family"]
            fam_stats[f]["count"] += 1
            fam_stats[f]["n_prep_total"] += q.get("n_prep", 0)
            fam_stats[f]["n_coord_total"] += q.get("n_coord", 0)
            fam_stats[f]["n_clause_total"] += q.get("n_clause", 0)
            fam_stats[f]["n_modifier_total"] += q.get("n_modifier", 0)
            fam_stats[f]["n_verb_total"] += q.get("n_verb", 0)
    for f, s in sorted(fam_stats.items()):
        n = max(1, s["count"])
        log(f"  {f}: n={s['count']}, prep/query={s['n_prep_total']/n:.2f}, "
            f"coord/query={s['n_coord_total']/n:.2f}, clause/query={s['n_clause_total']/n:.2f}, "
            f"modifier/query={s['n_modifier_total']/n:.2f}, verb/query={s['n_verb_total']/n:.2f}")

    # === PCA48 diversity check ===
    log(f"\n=== PCA48 diversity check ===")
    all_9c_queries = []
    seen_q = set()
    for asin, qs in pools.items():
        for q in qs:
            t = q["query"].strip().lower()
            if t and t not in seen_q:
                seen_q.add(t)
                all_9c_queries.append(t)
    log(f"  Unique 9C queries: {len(all_9c_queries)}")

    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    from gaussian_vades import _syntax_subspace_prepare
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist
    from main import per_sentence_features_v2  # noqa
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P["X_scaled"][train_idx])

    feats_9c = {}
    for i, doc in enumerate(nlp.pipe(all_9c_queries, batch_size=64)):
        for sent in doc.sents:
            sf = per_sentence_features_v2(sent)
            if sf:
                feats_9c[all_9c_queries[i]] = sf
                break
    log(f"  Feature extractions: {len(feats_9c)}")

    diversity_stats = {"per_asin": {}, "global": {}}
    all_dists = []
    for asin, qs in pools.items():
        if len(qs) < 2:
            continue
        zs = []
        miss = 0
        for q in qs:
            t = q["query"].strip().lower()
            sf = feats_9c.get(t)
            if not sf:
                miss += 1
                continue
            v = np.array([sf.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(v[None, :]))[0]
            zs.append(z)
        if len(zs) < 2:
            continue
        zs = np.array(zs)
        dists = pdist(zs)
        all_dists.extend(dists.tolist())
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

    if all_dists:
        all_dists = np.array(all_dists)
        diversity_stats["global"] = {
            "mean_pairwise_dist": float(all_dists.mean()),
            "median_pairwise_dist": float(np.median(all_dists)),
            "std_pairwise_dist": float(all_dists.std()),
            "n_pairs": int(len(all_dists)),
        }
        log(f"  Global mean pairwise dist: {all_dists.mean():.2f}")

    # === Save ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 9C: syntactic composition repair, anti-stack + connector-presence QA",
                "PILOT_ASIN_COUNT": PILOT_ASIN_COUNT,
                "N_FAMILIES": N_FAMILIES,
                "ROUNDS_PER_FAMILY": ROUNDS_PER_FAMILY,
                "BACKFILL_ROUNDS": BACKFILL_ROUNDS,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "ATTRIBUTE_STACK_RATE_TARGET": ATTRIBUTE_STACK_RATE_TARGET,
                "FAMILIES": [f["name"] for f in GRAMMATICAL_FAMILIES],
            },
            "layer_stats": dict(layer_stats),
            "pool_sizes": {a: len(qs) for a, qs in pools.items()},
            "family_counts": {a: dict(fc) for a, fc in family_counts_per_asin.items()},
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")

    with open(QA_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "stack_rate": n_stack / max(1, n_total),
            "no_connector_rate": n_no_connector / max(1, n_total),
            "extra_sem_rate": n_extra / max(1, n_total),
            "template_rate": n_template / max(1, n_total),
            "field_name_rate": n_field / max(1, n_total),
            "near_dup_rate": n_dup / max(1, n_pairs),
            "n_total": n_total,
            "n_pairs": n_pairs,
            "fam_stats": {f: dict(s) for f, s in fam_stats.items()},
            "diversity_stats": diversity_stats,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {QA_OUT}")


if __name__ == "__main__":
    main()