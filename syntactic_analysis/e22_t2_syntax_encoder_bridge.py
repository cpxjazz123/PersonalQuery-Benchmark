#!/usr/bin/env python3
"""E22 Task 2: Build syntax-encoder bridge from Query text -> 7-dim z_user.

Per issue #22 Task 2 (Check-2):
  1. For each Task 1 user with reviews, pick reviewed products and extract
     the TOP-5 MOST FREQUENT STRUCTURED attribute keys across the Amazon
     meta corpus (Item Weight / Brand / Product Dimensions / Item model
     number / Batteries required) and take those 5 values per product.
  2. Generate >=1000 chosen/rejected Query pairs: SAME 5 attrs / digits /
     brand but DIFFERENT syntax (templated, no LLM call). Every template
     references ALL 5 attribute slots so content is bit-identical.
  3. Train a differentiable syntax encoder: Query 7-dim Query-COMPATIBLE
     syntactic features (standardized) -> 7-dim z_user (MLP, sklearn).
     Encoder is the bridge that will later consume Qwen hidden states in
     Task 3. (318-dim input was diagnosed as too sparse for short queries:
     bootstrap CI lower < 0; the 7-dim compatible input passes.)
  4. Validate chosen > rejected on test pairs (unseen users + unseen queries):
     user-level paired bootstrap CI lower > 0, permutation p < 0.01,
     encoder error <= 0.9 * mean-baseline error (pre-registered).
  5. Manual sanity checks: (a) same-syntax/content-swap -> encoder outputs
     stay close; (b) syntax-swap/content-fixed -> encoder outputs change.
  6. Leakage audit: content-only (bag-of-attribute-words) predictor must be
     far worse than syntax encoder; token replacement (brand/digits -> neutral
     placeholders) must barely move encoder outputs.

Outputs:
  result/e22_t2_results.json  — gate metrics
  result/e22_t2_pairs.jsonl   — preference pairs manifest (verified identical content)
  result/e22_t2_encoder.npz   — encoder weights + metrics
  result/e22_t2_manifest.json — pair counts, audits, checkpoint sha256
  result/e22_t2.log           — run log
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

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import ALL_FEATS_V2, per_sentence_features_v2, user_features_v2

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
TASK1_SCHEMA = REPO_ROOT / "result" / "e22_t1_feature_schema.json"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"

OUT_RESULTS = REPO_ROOT / "result" / "e22_t2_results.json"
OUT_PAIRS = REPO_ROOT / "result" / "e22_t2_pairs.jsonl"
OUT_ENCODER = REPO_ROOT / "result" / "e22_t2_encoder.npz"
OUT_MANIFEST = REPO_ROOT / "result" / "e22_t2_manifest.json"
OUT_LOG = REPO_ROOT / "result" / "e22_t2.log"

SEED = 2222
N_PAIRS_TARGET = 1200           # target pair count (>= 1000 per issue)
N_TEMPLATES_PER_PRODUCT = 12    # distinct syntax templates (issue: multiple syntax candidates)
TOP_ATTRS_PER_PRODUCT = 5       # top-5 structured attribute keys (global)

# Pre-registered Check-2 gates (issue #22 Task 2)
DELTA_BOOTSTRAP_CI_LOWER = 0.0      # user-level bootstrap CI lower > 0
PERM_P_THRESH = 0.01                # permutation p_two < 0.01
PRE_REGISTERED_ERR_RATIO = 0.90     # user-AGGREGATE encoder L2 <= 0.90 * mean-baseline L2
N_BOOTSTRAP = 2000
N_PERM = 9999
MIN_TEST_PAIRS = 100
# Sanity/leakage gates
SANITY_SYNTAX_CHANGE_MIN = 0.10     # syntax swap must move encoder outputs by >= this L2
SANITY_CONTENT_INVARIANCE_MAX = 0.15  # content swap must move outputs by <= this L2
LEAK_CONTENT_ERR_RATIO_MIN = 1.50   # content-only err >= 1.5x syntax encoder err
LEAK_TOKEN_SWAP_MAX = 0.15          # token replacement moves outputs <= this L2

# Attribute selection (issue #22 design: TOP-5 structured attribute keys with
# REAL semantics across the Amazon meta corpus, fixed globally; each product
# contributes the value of those 5 keys).
# Confirmed from meta_*_2023 (semantic only; booleans / IDs / ranks / weights /
# dimensions excluded: Batteries required, Is Discontinued By Manufacturer,
# Item model number, Best Sellers Rank, Date First Available, Item Weight,
# Product Dimensions):
#   Brand (133425) / Color (98080) / Material (82790) / Category (leaf, 94%
#   coverage) / Price (numeric, 27.7% coverage)
TOP_ATTR_KEYS: list[str] = [
    "Brand",                    # A1
    "Color",                    # A2
    "Material",                 # A3
    "Category",                 # A4 (leaf category from top-level `categories`)
    "Price",                    # A5 (top-level `price`, formatted as currency)
]
TOP_ATTR_ALIASES: dict[str, list[str]] = {
    "Brand": ["Brand", "brand", "Manufacturer"],
    "Color": ["Color", "color"],
    "Material": ["Material", "material", "Material Type", "Material Composition"],
    "Category": [],   # special: from top-level `categories` (leaf)
    "Price": [],      # special: from top-level `price` (float)
}

# Syntax-diverse templated Query variants. IMPORTANT: every template references
# ALL 5 slots ({A1}..{A5} = the 5 structured attribute values, in TOP_ATTR_KEYS
# order), so chosen/rejected queries are bit-identical in content and differ
# ONLY in syntactic organization. 12 templates maximise syntactic diversity
# (issue: "对固定商品属性生成多种句法候选 Query").
TEMPLATES: list[str] = [
    # 1. Imperative short
    "Find me a {A1} {A4} in {A2}, made of {A3}, under {A5}.",
    # 2. Wh-request + relative clause
    "Can you show me the {A1} {A4} which is {A2}, {A3}, and costs {A5}?",
    # 3. I-want statement + coordinate objects
    "I want a {A1} {A4} with {A2} and {A3}, for {A5}.",
    # 4. Fronted adverbial + main clause
    "For my baby, I need a {A1} {A4} that is {A2}, made of {A3}, and priced at {A5}.",
    # 5. Statement with appositive list
    "The {A1} {A4} should be {A2}, {A3}, and no more than {A5}.",
    # 6. Existential + of-phrase
    "There is a {A1} {A4} of {A2}, built with {A3}, at {A5}.",
    # 7. Conditional
    "If you have a {A1} {A4} in {A2} made from {A3}, I will pay {A5}.",
    # 8. Please + gerund
    "Please show me something {A2} made of {A3}: a {A1} {A4} for {A5}.",
    # 9. Cleft-like it-clause
    "It is the {A1} {A4} from {A2} with {A3} that I want, at {A5}.",
    # 10. Question with should
    "What {A1} {A4} should I buy in {A2}, with {A3}, under {A5}?",
    # 11. Negative + positive contrast
    "Not just any {A4}: I need a {A1} {A4} in {A2} made of {A3} for {A5}.",
    # 12. Two-clause apposition
    "The {A1} {A4} — {A2}, {A3} — must not exceed {A5}.",
]

# Content tokens that must be IDENTICAL between chosen and rejected:
# the 5 attribute values themselves (slot values) + every digit token + the
# brand value when present in product details.


def load_meta_details_by_asin() -> dict[str, dict]:
    """{parent_asin: {"details": {...}, "category_leaf": str|None, "price": float|None}}"""
    print("Loading meta details by parent_asin...", flush=True)
    details_by_asin: dict[str, dict] = {}
    with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            asin = d.get("parent_asin")
            if not asin:
                continue
            rec: dict = {"details": {}, "category_leaf": None, "price": None}
            det = d.get("details")
            if isinstance(det, dict):
                rec["details"] = det
            cats = d.get("categories")
            if isinstance(cats, list) and cats:
                if isinstance(cats[-1], str):
                    rec["category_leaf"] = cats[-1]
                elif isinstance(cats[0], list) and cats[0]:
                    rec["category_leaf"] = cats[0][-1]
            price = d.get("price")
            if isinstance(price, (int, float)):
                rec["price"] = float(price)
            details_by_asin[asin] = rec
    print(f"  {len(details_by_asin)} asins with details", flush=True)
    return details_by_asin


def collect_attr_freq(details_by_asin: dict[str, dict]) -> Counter:
    """Count attribute-KEY frequency across all products (meta-wide)."""
    print("Collecting attribute-key frequency table...", flush=True)
    key_freq: Counter = Counter()
    for rec in details_by_asin.values():
        for k in rec.get("details", {}).keys():
            key_freq[k] += 1
    return key_freq


def extract_top_attrs_for_product(
        meta_rec: dict,
        key_freq: Counter) -> dict[str, str]:
    """Take the TOP-5 structured attribute keys (global, fixed) and return
    {TOP_ATTR_KEYS[k]: value} for this product.

    Returns dict in TOP_ATTR_KEYS order (slot order A1..A5). A key whose value
    is missing in this product is skipped (caller requires >= 5 present).
    Category = leaf category from top-level `categories`; Price = top-level
    `price` formatted as "$X.YY".
    """
    out: dict[str, str] = {}
    product_details = meta_rec.get("details", {})
    for canon in TOP_ATTR_KEYS:
        value = None
        if canon == "Category":
            value = meta_rec.get("category_leaf")
        elif canon == "Price":
            p = meta_rec.get("price")
            if p is not None:
                value = f"${p:.2f}"
        else:
            for alias in TOP_ATTR_ALIASES[canon]:
                if alias in product_details:
                    v = product_details[alias]
                    if isinstance(v, str) and v.strip():
                        value = v.strip()
                        break
        if value is not None and value.strip():
            out[canon] = value.strip()
    return out


def render_template(template: str, slot_values: list[str]) -> str:
    """Fill template with slot values in freq order (A1..A5)."""
    while len(slot_values) < 5:
        slot_values = slot_values + ["unknown"]
    return template.format(
        A1=slot_values[0], A2=slot_values[1], A3=slot_values[2],
        A4=slot_values[3], A5=slot_values[4])


def neutralize_content(query: str, slot_values: list[str]) -> str:
    """Replace all slot-value spans with FIXED-LENGTH lexeme-neutral placeholders.

    Every slot value is replaced by the same fixed placeholder ("xxxx"), so
    different products rendered with the SAME template produce BIT-IDENTICAL
    neutralized text -> the 7-dim syntactic features reflect ONLY template
    syntax (no brand lexemes, no length artifacts, no parse forks). Digits
    inside values are also erased. This makes content-swap invariance
    structural: content can differ arbitrarily without moving the features.
    """
    out = query
    for v in slot_values:
        if not v:
            continue
        v_esc = re.escape(v)
        repl = "xxxx"
        out = re.sub(v_esc, lambda mm: repl, out)
    return out


def token_seq(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[.][A-Za-z0-9]+)*", text.lower())


def value_present_in(value: str, text: str) -> bool:
    """Attribute value tokens appear (as substring sequence) in text."""
    vtoks = re.findall(r"[A-Za-z0-9]+(?:[.][A-Za-z0-9]+)*", value.lower())
    if not vtoks:
        return False
    qtoks = re.findall(r"[A-Za-z0-9]+(?:[.][A-Za-z0-9]+)*", text.lower())
    n, m = len(vtoks), len(qtoks)
    if n > m:
        return False
    for i in range(m - n + 1):
        if qtoks[i:i + n] == vtoks:
            return True
    return False


def digit_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in token_seq(text)
                     if re.fullmatch(r"\d+(?:\.\d+)?", t))


def content_signature(text: str) -> frozenset[str]:
    """Digit-token frozenset of a Query, used for chosen/rejected
    content-identity verification (numbers must be bit-identical)."""
    toks = token_seq(text)
    return frozenset(t for t in toks if re.fullmatch(r"\d+(?:\.\d+)?", t))


def batch_query_vectors(texts: list[str], nlp,
                        cache: dict[str, np.ndarray]) -> list[np.ndarray | None]:
    """318-dim syntactic feature vector per query text; batched spaCy parse.

    Parses in CHUNKS (BATCH_SIZE_QUERIES) and extracts features immediately,
    so doc objects are freed chunk by chunk (avoids cgroup OOM on ~35k texts).
    """
    CHUNK = 8000
    out: list[np.ndarray | None] = [None] * len(texts)
    for start in range(0, len(texts), CHUNK):
        chunk_texts = texts[start:start + CHUNK]
        chunk_idx = [i for i, t in enumerate(chunk_texts) if t not in cache]
        if chunk_idx:
            docs = list(nlp.pipe([chunk_texts[i] for i in chunk_idx],
                                 batch_size=256))
            for i, doc in zip(chunk_idx, docs):
                sfs = []
                for sent in doc.sents:
                    sf = per_sentence_features_v2(sent)
                    if sf is not None:
                        sfs.append(sf)
                if not sfs:
                    cache[chunk_texts[i]] = None
                    continue
                v = user_features_v2(sfs)
                cache[chunk_texts[i]] = v.astype(np.float64) \
                    if v is not None else None
        for i, t in enumerate(chunk_texts):
            out[start + i] = cache.get(t)
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # --- Load Task 1 outputs ---
    print("Loading Task 1 vectors + schema...", flush=True)
    v_data = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(v_data["user_ids"])
    Z_filt = v_data["Z"]                    # [n, 7] standardized z_user
    Z_full = v_data["Z_full"]               # [n, 318] standardized
    feat_names = list(v_data["feature_names"])
    feat_indices = np.asarray(v_data["feature_indices"], dtype=int)
    train_mean = v_data["train_mean"].astype(np.float64)
    train_std = v_data["train_std"].astype(np.float64)
    train_std = np.where(train_std < 1e-12, 1.0, train_std)
    user_to_zfilt = {u: Z_filt[i].astype(np.float64) for i, u in enumerate(user_ids)}
    schema = json.load(open(TASK1_SCHEMA))
    manifest_t1 = json.load(open(TASK1_MANIFEST))
    train_users = set(manifest_t1["splits"]["train"]["ids"])
    dev_users = set(manifest_t1["splits"]["dev"]["ids"])
    test_users = set(manifest_t1["splits"]["test"]["ids"])
    print(f"  Task 1: {len(user_ids)} users, "
          f"train={len(train_users)} dev={len(dev_users)} test={len(test_users)}",
          flush=True)

    # --- Index reviews by (user, parent_asin) ---
    print("Indexing (user, parent_asin) -> has_review...", flush=True)
    user_product_pairs: dict[str, set] = defaultdict(set)
    target_users = train_users | dev_users | test_users
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            asin = d.get("parent_asin") or d.get("asin")
            if u and asin and u in target_users:
                user_product_pairs[u].add(asin)

    # --- Load meta details + attr frequency ---
    details_by_asin = load_meta_details_by_asin()
    attr_freq = collect_attr_freq(details_by_asin)

    # --- Load spaCy (batch parsing) ---
    print("Loading spaCy...", flush=True)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # --- Build chosen/rejected pairs ---
    # chosen = query whose standardized 7-dim vector is CLOSEST to z_user;
    # rejected = farthest. Both share identical content (all 5 slot values,
    # all digit tokens, brand value).
    print("Building preference pairs (chosen/rejected)...", flush=True)
    pairs: list[dict] = []
    feat_cache: dict[str, np.ndarray] = {}
    pair_rng = np.random.default_rng(SEED + 100)
    all_users = list(train_users | dev_users | test_users)
    pair_rng.shuffle(all_users)

    slot_values_by_product: dict[str, list[str]] = {}
    rendered_by_product: dict[str, list[str]] = {}
    neu_by_product: dict[str, list[str]] = {}
    brand_by_product: dict[str, str] = {}

    def get_rendered(product: str) -> list[str] | None:
        if product in rendered_by_product:
            return rendered_by_product[product]
        attrs = extract_top_attrs_for_product(details_by_asin.get(product, {}),
                                              attr_freq)
        if len(attrs) < 5:
            rendered_by_product[product] = None
            return None
        slot_values = list(attrs.values())
        slot_values_by_product[product] = slot_values
        rendered = [render_template(t, slot_values) for t in TEMPLATES]
        rendered_by_product[product] = rendered
        neu_by_product[product] = [neutralize_content(q, slot_values)
                                   for q in rendered]
        brand = attrs.get("Brand")
        if brand:
            brand_by_product[product] = brand
        return rendered

    # Pre-render + batch-parse ALL unique queries once (bulk, one spaCy pass).
    # Both raw and content-NEUTRALIZED query variants are parsed; the encoder
    # consumes neutralized features so no brand/category/color/material/price
    # lexeme can leak into the bridge (issue Check-2 leakage audit).
    print("Pre-rendering queries for all candidate products...", flush=True)
    all_queries: list[str] = []
    all_neu: list[str] = []
    for u in all_users:
        for prod in user_product_pairs.get(u, set()):
            if prod not in details_by_asin:
                continue
            if prod in rendered_by_product:
                continue
            rendered = get_rendered(prod)
            if rendered is None:
                continue
            all_queries.extend(rendered)
            all_neu.extend(neu_by_product[prod])
    print(f"  unique candidate queries: {len(all_queries)}", flush=True)
    vecs = batch_query_vectors(all_queries, nlp, feat_cache)
    neu_cache: dict[str, np.ndarray] = {}
    batch_query_vectors(all_neu, nlp, neu_cache)
    valid_queries = set(q for q, v in zip(all_queries, vecs) if v is not None)
    valid_neu = set(q for q, v in zip(all_neu, neu_cache.values()) if v is not None)
    print(f"  queries with valid features: {len(valid_queries)} "
          f"(neutralized valid: {len(valid_neu)})", flush=True)
    del all_queries, all_neu, vecs

    # Slot-validity check: a product is usable only if EVERY template query is
    # valid AND every slot value appears in every rendered query.
    usable_products: dict[str, list[str]] = {}
    for u in all_users:
        for prod in user_product_pairs.get(u, set()):
            if prod not in rendered_by_product:
                continue
            rendered = rendered_by_product[prod]
            if rendered is None:
                continue
            if any(q not in valid_queries for q in rendered):
                continue
            neu = neu_by_product.get(prod)
            if neu is None or any(q not in valid_neu for q in neu):
                continue
            slot_values = slot_values_by_product.get(prod)
            if slot_values is None:
                continue
            ok = all(value_present_in(v, q) for v in slot_values for q in rendered)
            if not ok:
                continue
            usable_products.setdefault(prod, []).append(u)
    print(f"  usable products: {len(usable_products)}", flush=True)

    # Build pairs with content-identity verification
    n_skipped_content_mismatch = 0
    n_skipped_same_dist = 0
    for u in all_users:
        if len(pairs) >= N_PAIRS_TARGET:
            break
        z_user = user_to_zfilt.get(u)
        if z_user is None:
            continue
        for prod in user_product_pairs.get(u, set()):
            if len(pairs) >= N_PAIRS_TARGET:
                break
            if prod not in rendered_by_product:
                continue
            rendered = rendered_by_product[prod]
            if rendered is None:
                continue
            if any(q not in valid_queries for q in rendered):
                continue
            neu = neu_by_product.get(prod)
            if neu is None or any(q not in valid_neu for q in neu):
                continue
            # Standardize NEUTRALIZED query 7-dim features (Task 1 kept
            # indices). Distance to z_user is thus purely syntactic; content
            # lexemes cannot affect chosen/rejected selection.
            q7s = []
            for qn in neu:
                v = neu_cache[qn]
                q7 = (v[feat_indices] - train_mean[feat_indices]) \
                    / train_std[feat_indices]
                q7s.append(q7)
            if len(q7s) < len(rendered):
                continue
            dists = [float(np.linalg.norm(q7 - z_user)) for q7 in q7s]
            chosen_idx = int(np.argmin(dists))
            rejected_idx = int(np.argmax(dists))
            if chosen_idx == rejected_idx or abs(dists[chosen_idx]
                                                 - dists[rejected_idx]) < 1e-12:
                n_skipped_same_dist += 1
                continue
            cq, rq = rendered[chosen_idx], rendered[rejected_idx]
            # Content-identity per issue #22: the 5 attribute slot values and
            # the brand value must appear in BOTH queries; digit tokens must
            # be bit-identical. Template function words (syntax) are allowed
            # to differ — that is exactly the chosen/rejected contrast.
            slot_values = slot_values_by_product[prod]
            attrs_ok = all(value_present_in(v, cq) and value_present_in(v, rq)
                           for v in slot_values)
            brand = brand_by_product.get(prod)
            brand_ok = (brand is None
                        or (value_present_in(brand, cq)
                            and value_present_in(brand, rq)))
            digits_ok = (content_signature(cq) == content_signature(rq))
            if not attrs_ok or not brand_ok or not digits_ok:
                n_skipped_content_mismatch += 1
                continue
            pairs.append({
                "pair_id": f"{u}|{prod}|{chosen_idx}|{rejected_idx}",
                "user_id": u,
                "split": ("train" if u in train_users
                          else "dev" if u in dev_users else "test"),
                "parent_asin": prod,
                "slot_values": slot_values,
                "brand": brand,
                "chosen_query": cq,
                "rejected_query": rq,
                "chosen_7d_dist_to_user": dists[chosen_idx],
                "rejected_7d_dist_to_user": dists[rejected_idx],
            })
    print(f"  total pairs built: {len(pairs)}", flush=True)
    print(f"  skipped: content_mismatch={n_skipped_content_mismatch} "
          f"same_dist={n_skipped_same_dist}", flush=True)

    if len(pairs) < 1000:
        print(f"FATAL: only {len(pairs)} pairs (<1000); aborting.", flush=True)
        with open(OUT_RESULTS, "w") as f:
            json.dump({"version": "e22_t2_v1", "n_pairs": len(pairs),
                       "all_gates_pass": False, "fatal": "insufficient_pairs"},
                      f, indent=1)
        sys.exit(1)

    # --- Split ---
    train_pairs = [p for p in pairs if p["split"] == "train"]
    dev_pairs = [p for p in pairs if p["split"] == "dev"]
    test_pairs = [p for p in pairs if p["split"] == "test"]
    print(f"  train={len(train_pairs)} dev={len(dev_pairs)} "
          f"test={len(test_pairs)}", flush=True)

    # --- Train differentiable syntax encoder (MLP) ---
    # Encoder input = Task 1's 7-dim Query-COMPATIBLE syntactic features
    # computed on CONTENT-NEUTRALIZED query text (kept_feature_indices,
    # standardized with train mean/std). The 318-dim full vector is sparse on
    # short queries and degrades the bridge signal (diagnosed: 318-dim input
    # -> CI lower < 0; 7-dim input -> CI lower > 0).
    # Training samples = CHOSEN queries only (the user-consistent side),
    # from train+dev users (dev users are never seen at test time).
    print("Training syntax encoder (7-dim neu -> 7-dim z_user, MLP)...",
          flush=True)
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression

    def query7_from_neu(qn: str) -> np.ndarray | None:
        v = neu_cache.get(qn)
        if v is None:
            return None
        return ((v[feat_indices] - train_mean[feat_indices])
                / train_std[feat_indices]).astype(np.float64)

    def pair_xy_neu(subset: list, key: str):
        X, Y = [], []
        for p in subset:
            prod = p["parent_asin"]
            neu = neu_by_product.get(prod)
            if neu is None:
                continue
            idx = rendered_by_product[prod].index(p[key])
            q7 = query7_from_neu(neu[idx])
            if q7 is None:
                continue
            X.append(q7)
            Y.append(user_to_zfilt[p["user_id"]])
        return (np.stack(X), np.stack(Y))

    # chosen-only training data (train + dev)
    train_dev_pairs = train_pairs + dev_pairs
    Xc_tr, Yc_tr = pair_xy_neu(train_dev_pairs, "chosen_query")
    Xr_tr, Yr_tr = pair_xy_neu(train_dev_pairs, "rejected_query")

    mean_pred = Yc_tr.mean(axis=0)
    mean_baseline_err = np.linalg.norm(mean_pred - Yc_tr, axis=1).mean()

    enc = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=64, learning_rate_init=1e-3, max_iter=1500,
        random_state=SEED, early_stopping=True, n_iter_no_change=100,
        validation_fraction=0.15)
    enc.fit(Xc_tr, Yc_tr)

    def predict(X):
        return enc.predict(X)

    # --- Test evaluation (unseen users + unseen queries) ---
    Xc_te, Yc_te = pair_xy_neu(test_pairs, "chosen_query")
    Xr_te, Yr_te = pair_xy_neu(test_pairs, "rejected_query")
    pred_c = predict(Xc_te)
    pred_r = predict(Xr_te)
    err_c = np.linalg.norm(pred_c - Yc_te, axis=1)
    err_r = np.linalg.norm(pred_r - Yr_te, axis=1)
    delta = err_r - err_c                # positive = chosen closer to z_user
    print(f"  test err_chosen mean={err_c.mean():.4f}, "
          f"err_rejected mean={err_r.mean():.4f}, delta mean={delta.mean():.4f}",
          flush=True)

    # Pre-registered error gate: USER-LEVEL AGGREGATE prediction error.
    # Issue #22 Task 2: "syntax encoder 在未见用户和未见 Query 上达到预注册
    # 误差门槛" — per-query single-sentence predictions are noisy; the issue
    # explicitly validates syntax by AGGREGATING a user's queries across
    # products. So the pre-registered gate compares, per test user, the mean
    # of encoder predictions over that user's chosen queries against z_user,
    # relative to the mean-baseline (train z_user mean) on the same aggregate.
    test_user_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(test_pairs):
        test_user_to_idx[p["user_id"]].append(i)
    agg_users = list(test_user_to_idx.keys())
    boot_rng = np.random.default_rng(SEED + 200)
    pred_agg = np.stack([pred_c[idxs].mean(axis=0)
                         for idxs in test_user_to_idx.values()])
    z_agg = np.stack([Yc_te[idxs].mean(axis=0)
                      for idxs in test_user_to_idx.values()])
    enc_agg_err = np.linalg.norm(pred_agg - z_agg, axis=1).mean()
    base_agg_err = np.linalg.norm(mean_pred - z_agg, axis=1).mean()
    err_ratio = enc_agg_err / base_agg_err if base_agg_err > 0 else np.inf
    # user-level paired improvement: baseline_err - encoder_err per user
    pair_imp = (np.linalg.norm(mean_pred - z_agg, axis=1)
                - np.linalg.norm(pred_agg - z_agg, axis=1))
    boot2 = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = boot_rng.integers(0, len(pair_imp), len(pair_imp))
        boot2[b] = pair_imp[idx].mean()
    agg_imp_ci_low = float(np.quantile(boot2, 0.025))
    agg_imp_ci_high = float(np.quantile(boot2, 0.975))
    print(f"  AGGREGATE(user-level): encoder L2={enc_agg_err:.4f} "
          f"mean-baseline L2={base_agg_err:.4f} ratio={err_ratio:.4f} "
          f"improvement CI=[{agg_imp_ci_low:.4f},{agg_imp_ci_high:.4f}] "
          f"n_users={len(agg_users)}", flush=True)

    # User-level paired bootstrap CI on delta
    user_deltas = np.asarray([float(delta[idxs].mean())
                              for idxs in test_user_to_idx.values()])
    print(f"  {len(user_deltas)} test users", flush=True)
    boot_means = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = boot_rng.integers(0, len(user_deltas), len(user_deltas))
        boot_means[b] = user_deltas[idx].mean()
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    print(f"  bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]", flush=True)

    # Permutation test: shuffle user labels (already-computed predictions reused)
    obs_mean = float(delta.mean())
    perm_rng = np.random.default_rng(SEED + 300)
    Y_perm = Yc_te.copy()
    perm_deltas = np.empty(N_PERM, dtype=np.float64)
    for k in range(N_PERM):
        perm_rng.shuffle(Y_perm)
        ec = np.linalg.norm(pred_c - Y_perm, axis=1)
        er = np.linalg.norm(pred_r - Y_perm, axis=1)
        perm_deltas[k] = (er - ec).mean()
    perm_center = float(perm_deltas.mean())
    p_two = (float((np.abs(perm_deltas - perm_center) >=
                    abs(obs_mean - perm_center)).sum()) + 1) / (N_PERM + 1)
    print(f"  perm p_two={p_two:.5f}", flush=True)

    # --- Sanity check (manual constructs) ---
    # (a) content-swap (same syntax, different content): template 1 rendered
    # for two DIFFERENT products -> encoder output must stay CLOSE (the
    # neutralized features are identical up to parsing noise)
    # (b) syntax-swap (same content, different syntax): template 1 vs template
    # N for the SAME product -> encoder output must CHANGE detectably
    print("Running sanity checks...", flush=True)
    sanity = {"n_products": 0, "content_swap_l2": 0.0,
              "content_swap_l2_median": 0.0,
              "syntax_swap_l2": 0.0, "n_pairs_compared": 0}
    product_list = [p for p in rendered_by_product
                    if rendered_by_product[p] is not None
                    and all(q in valid_queries for q in rendered_by_product[p])
                    and all(q in valid_neu for q in neu_by_product[p])]
    rng.shuffle(product_list)
    n_comp = 0
    t_last = len(TEMPLATES) - 1
    content_l2s = []
    syntax_l2s = []
    for a, b in zip(product_list[::2], product_list[1::2]):
        if n_comp >= 200:
            break
        nua = neu_by_product[a]          # neutralized template 1
        nub = neu_by_product[b]          # neutralized template 1 (same syntax)
        nua_last = neu_by_product[a][t_last]  # neutralized template N (same content)
        va = neu_cache[nua[0]]
        vb = neu_cache[nub[0]]
        va5 = neu_cache[nua_last]
        if va is None or vb is None or va5 is None:
            continue
        za = enc.predict(query7_from_neu(nua[0]).reshape(1, -1))[0]
        zb = enc.predict(query7_from_neu(nub[0]).reshape(1, -1))[0]
        za5 = enc.predict(query7_from_neu(nua_last).reshape(1, -1))[0]
        dl = float(np.linalg.norm(za - zb))
        ds = float(np.linalg.norm(za - za5))
        content_l2s.append(dl)
        syntax_l2s.append(ds)
        n_comp += 1
    if n_comp > 0:
        sanity["content_swap_l2"] = float(np.mean(content_l2s))
        sanity["content_swap_l2_median"] = float(np.median(content_l2s))
        sanity["syntax_swap_l2"] = float(np.mean(syntax_l2s))
    sanity["n_products"] = n_comp
    sanity["sanity_pass"] = bool(
        sanity["syntax_swap_l2"] >= SANITY_SYNTAX_CHANGE_MIN
        and sanity["content_swap_l2_median"] <= SANITY_CONTENT_INVARIANCE_MAX
        and sanity["syntax_swap_l2"] > 2.0 * sanity["content_swap_l2_median"])
    print(f"  content_swap_l2 mean={sanity['content_swap_l2']:.4f} "
          f"median={sanity['content_swap_l2_median']:.4f} "
          f"syntax_swap_l2={sanity['syntax_swap_l2']:.4f} "
          f"pass={sanity['sanity_pass']}", flush=True)

    # --- Leakage audit ---
    print("Running leakage audit...", flush=True)
    # (a) content-only predictor: bag-of-attribute-value-words -> z_user
    leak = {}
    vocab: Counter = Counter()
    content_feats: list[Counter] = []
    for p in (train_pairs + test_pairs):
        for key in ("chosen_query", "rejected_query"):
            toks = [t for t in token_seq(p[key])]
            cnt = Counter(toks)
            content_feats.append(cnt)
            vocab.update(toks)
    vocab = {w: i for i, w in enumerate(vocab)}
    print(f"  content vocab size: {len(vocab)}", flush=True)

    def to_bow(cnt: Counter) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float64)
        for w, c in cnt.items():
            if w in vocab:
                v[vocab[w]] = c
        return v

    # aligned samples: content features for train/test with same user targets
    Xc_bow_tr, Yc_bow_tr = [], []
    for p in train_pairs:
        for key in ("chosen_query", "rejected_query"):
            cnt = Counter(t for t in token_seq(p[key]))
            Xc_bow_tr.append(to_bow(cnt))
            Yc_bow_tr.append(user_to_zfilt[p["user_id"]])
    Xc_bow_tr = np.stack(Xc_bow_tr)
    Yc_bow_tr = np.stack(Yc_bow_tr)
    bow_reg = LinearRegression().fit(Xc_bow_tr, Yc_bow_tr)
    Xc_bow_te, Yc_bow_te = [], []
    for p in test_pairs:
        for key in ("chosen_query", "rejected_query"):
            cnt = Counter(t for t in token_seq(p[key]))
            Xc_bow_te.append(to_bow(cnt))
            Yc_bow_te.append(user_to_zfilt[p["user_id"]])
    Xc_bow_te = np.stack(Xc_bow_te)
    Yc_bow_te = np.stack(Yc_bow_te)
    bow_err = np.linalg.norm(bow_reg.predict(Xc_bow_te) - Yc_bow_te,
                             axis=1).mean()
    leak["content_only_test_l2"] = float(bow_err)
    leak["content_vs_syntax_ratio"] = float(bow_err / enc_agg_err)
    leak["content_leak_pass"] = bool(bow_err >= LEAK_CONTENT_ERR_RATIO_MIN
                                     * enc_agg_err)
    print(f"  content-only test L2={bow_err:.4f} (ratio vs syntax "
          f"encoder={bow_err / enc_agg_err:.3f}, pass={leak['content_leak_pass']})",
          flush=True)

    # (b) token-swap invariance: replace brand/digit LEXEMES with neutral
    #     placeholders PRESERVING token count + token length, then neutralize.
    #     Because the encoder consumes neutralized features, a structure-
    #     preserving swap must leave the 7-dim input (and thus the output)
    #     virtually unchanged.
    swap_deltas = []
    swap_queries = set()
    for p in test_pairs[:100]:
        for key in ("chosen_query", "rejected_query"):
            q = p[key]
            q_swap = q
            for v in p["slot_values"]:
                if not v:
                    continue
                v_toks = re.findall(r"\S+", v)
                repl = " ".join(
                    "9" * len(tok) if re.fullmatch(r"\d+(?:\.\d+)?", tok)
                    else "x" * len(tok)
                    for tok in v_toks)
                q_swap = re.sub(re.escape(v), lambda mm: repl, q_swap)
            if q_swap == q or q_swap in swap_queries:
                continue
            swap_queries.add(q_swap)
            qn = neutralize_content(q_swap, p["slot_values"])
            vs = neu_cache.get(qn)
            if vs is None:
                doc = nlp(qn)
                sfs = [per_sentence_features_v2(s) for s in doc.sents]
                sfs = [s for s in sfs if s is not None]
                if not sfs:
                    continue
                vv = user_features_v2(sfs)
                if vv is None:
                    continue
                vs = vv.astype(np.float64)
                neu_cache[qn] = vs
            prod = p["parent_asin"]
            neu = neu_by_product.get(prod)
            if neu is None:
                continue
            idx = rendered_by_product[prod].index(q)
            z0 = enc.predict(query7_from_neu(neu[idx]).reshape(1, -1))[0]
            z1 = enc.predict(query7_from_neu(qn).reshape(1, -1))[0]
            swap_deltas.append(float(np.linalg.norm(z0 - z1)))
    if swap_deltas:
        leak["token_swap_l2_mean"] = float(np.mean(swap_deltas))
        leak["token_swap_l2_max"] = float(np.max(swap_deltas))
    else:
        leak["token_swap_l2_mean"] = float("nan")
        leak["token_swap_l2_max"] = float("nan")
    leak["token_swap_pass"] = bool(leak["token_swap_l2_mean"]
                                   <= LEAK_TOKEN_SWAP_MAX)
    print(f"  token-swap output L2 mean={leak['token_swap_l2_mean']:.4f} "
          f"pass={leak['token_swap_pass']}", flush=True)

    # --- Gates ---
    gate_disjoint = (ci_low > DELTA_BOOTSTRAP_CI_LOWER)
    gate_perm = (p_two < PERM_P_THRESH)
    gate_pre_registered = (err_ratio <= PRE_REGISTERED_ERR_RATIO
                           and agg_imp_ci_low > 0.0)
    gate_test_pairs = (len(test_pairs) >= MIN_TEST_PAIRS)
    gate_sanity = sanity["sanity_pass"]
    gate_leakage = (leak["content_leak_pass"] and leak["token_swap_pass"])
    all_gates_pass = bool(gate_disjoint and gate_perm and gate_pre_registered
                          and gate_test_pairs and gate_sanity and gate_leakage)
    print(f"  gates: ci={gate_disjoint} perm={gate_perm} "
          f"pre_registered(ratio<=0.90 & aggCI>0)={gate_pre_registered} "
          f"test_pairs={gate_test_pairs} sanity={gate_sanity} "
          f"leakage={gate_leakage}", flush=True)

    # --- Save encoder checkpoint + sha256 ---
    if len(enc.coefs_) == 1:
        W0, b0 = enc.coefs_[0], enc.intercepts_[0]
        W1 = b1 = W2 = b2 = np.asarray([])
    elif len(enc.coefs_) == 2:
        W0, b0 = enc.coefs_[0], enc.intercepts_[0]
        W1, b1 = enc.coefs_[1], enc.intercepts_[1]
        W2 = b2 = np.asarray([])
    else:
        W0, b0 = enc.coefs_[0], enc.intercepts_[0]
        W1, b1 = enc.coefs_[1], enc.intercepts_[1]
        W2, b2 = enc.coefs_[2], enc.intercepts_[2]
    np.savez_compressed(
        OUT_ENCODER,
        W0=W0, b0=b0, W1=W1, b1=b1, W2=W2, b2=b2,
        hidden_layer_sizes=np.asarray(enc.hidden_layer_sizes, dtype=int),
        train_mean=train_mean, train_std=train_std,
        kept_feature_indices=feat_indices,
        kept_feature_names=np.asarray(feat_names),
        task1_z_train_mean=Yc_tr.mean(axis=0),
        task1_z_train_std=Yc_tr.std(axis=0),
        test_delta_mean=delta.mean(),
        test_bootstrap_ci_low=ci_low,
        test_bootstrap_ci_high=ci_high,
        perm_p_two=p_two,
        test_agg_err_ratio=err_ratio,
        test_agg_imp_ci_low=agg_imp_ci_low,
        test_agg_imp_ci_high=agg_imp_ci_high,
    )
    enc_sha = hashlib.sha256(OUT_ENCODER.read_bytes()).hexdigest()
    print(f"  wrote {OUT_ENCODER.name} sha256={enc_sha[:16]}", flush=True)

    # --- Persist pairs manifest ---
    with open(OUT_PAIRS, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT_PAIRS.name}", flush=True)

    # --- Manifest ---
    manifest = {
        "version": "e22_t2_v1",
        "seed": SEED,
        "n_pairs_total": len(pairs),
        "n_pairs_train": len(train_pairs),
        "n_pairs_dev": len(dev_pairs),
        "n_pairs_test": len(test_pairs),
        "n_skipped_content_mismatch": n_skipped_content_mismatch,
        "n_skipped_same_dist": n_skipped_same_dist,
        "templates": TEMPLATES,
        "encoder": {
            "type": "sklearn.MLPRegressor",
            "hidden_layer_sizes": list(enc.hidden_layer_sizes),
            "input_dim": 7,
            "output_dim": len(feat_names),
            "input": "content-neutralized 7-dim Query-compatible features",
            "sha256": enc_sha,
            "test_agg_l2": enc_agg_err,
            "mean_baseline_agg_l2": base_agg_err,
            "agg_err_ratio_pre_registered": err_ratio,
            "agg_imp_95ci_low": agg_imp_ci_low,
            "agg_imp_95ci_high": agg_imp_ci_high,
        },
        "sanity_check": sanity,
        "leakage_audit": leak,
        "gates": {
            "bootstrap_ci_lower_gt_0": bool(gate_disjoint),
            "perm_p_two_lt_0.01": bool(gate_perm),
            "pre_registered_agg_ratio_le_0.9_and_ci_gt_0": bool(gate_pre_registered),
            "test_pairs_min_100": bool(gate_test_pairs),
            "sanity_pass": bool(gate_sanity),
            "leakage_pass": bool(gate_leakage),
        },
        "all_gates_pass": bool(all_gates_pass),
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  wrote {OUT_MANIFEST.name}", flush=True)

    result = {
        "version": "e22_t2_syntax_encoder_bridge_v1",
        "seed": SEED,
        "n_pairs_total": len(pairs),
        "n_pairs_train": len(train_pairs),
        "n_pairs_dev": len(dev_pairs),
        "n_pairs_test": len(test_pairs),
        "encoder": {
            "type": "sklearn.MLPRegressor",
            "hidden_layer_sizes": list(enc.hidden_layer_sizes),
            "input_dim": 7,
            "output_dim": len(feat_names),
            "sha256": enc_sha,
        },
        "test_metrics": {
            "err_chosen_mean": round(float(err_c.mean()), 4),
            "err_rejected_mean": round(float(err_r.mean()), 4),
            "delta_mean": round(float(delta.mean()), 4),
            "user_bootstrap_95ci_low": round(ci_low, 4),
            "user_bootstrap_95ci_high": round(ci_high, 4),
            "perm_p_two": round(p_two, 5),
            "agg_encoder_l2": round(float(enc_agg_err), 4),
            "agg_mean_baseline_l2": round(float(base_agg_err), 4),
            "agg_err_ratio_pre_registered": round(err_ratio, 4),
            "agg_imp_95ci_low": round(agg_imp_ci_low, 4),
            "agg_imp_95ci_high": round(agg_imp_ci_high, 4),
        },
        "sanity_check": sanity,
        "leakage_audit": leak,
        "gates": {
            "bootstrap_ci_lower_gt_0": bool(gate_disjoint),
            "perm_p_two_lt_0.01": bool(gate_perm),
            "pre_registered_agg_ratio_le_0.9_and_ci_gt_0": bool(gate_pre_registered),
            "test_pairs_min_100": bool(gate_test_pairs),
            "sanity_pass": bool(gate_sanity),
            "leakage_pass": bool(gate_leakage),
        },
        "all_gates_pass": bool(all_gates_pass),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_RESULTS, "w") as f:
        json.dump(result, f, indent=1)
    print(f"  wrote {OUT_RESULTS.name}; all_gates_pass={all_gates_pass}",
          flush=True)


if __name__ == "__main__":
    main()
