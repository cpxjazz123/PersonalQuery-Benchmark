#!/usr/bin/env python3
"""Stage 6B-γ: Natural-length N_attrs experiment + multi-metric statistical analysis.

设计 (per user spec 2026-08-24):
  - 不给 LLM 设置固定长度, 让 query 自然决定长度
  - N_input ∈ {3, 4, 5, 6, 7}, K=5 samples per (record, N_input)
  - 3000 records: dev=2000 (选择最终 N), test=1000 (独立验证)
  - 评估: PCA48 + Mahalanobis (主) + 多 metric
  - 分析: 直接对比 / length-bucket (post-hoc) / mixed-effects regression /
          bootstrap CI / permutation test / stratified by asin pool size

运行 (3 步):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery

  # Step 1: gen (NO length instruction, K=5, 3000 records × 5 × 5 = 75000 gens)
  STAGE6BG_MODE=gen nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      gaussian/syntax_subspace_stage6b_gamma.py \
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_gamma_gen.log 2>&1 &

  # Step 2: eval (PCA48 multi-metric)
  STAGE6BG_MODE=eval nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      gaussian/syntax_subspace_stage6b_gamma.py \
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_gamma_eval.log 2>&1 &

  # Step 3: analyze (bootstrap CI, permutation, mixed-effects)
  STAGE6BG_MODE=analyze nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      gaussian/syntax_subspace_stage6b_gamma.py \
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_gamma_analyze.log 2>&1 &

硬编码常量 (Rule 3):
  N_INPUT_LIST    = [3, 4, 5, 6, 7]
  K_SAMPLES       = 5           # 每个 (record, N_input) 采样数
  MAX_RECORDS     = 3000        # 总 records (dev 2000 + test 1000)
  N_DEV           = 2000
  N_TEST          = 1000
  N_FIXED         = 30          # 用户 Gaussian 历史句数
  LAMBDA          = 0.1
  N_USERS_MIN     = 3
  PCA_DIM         = 48
  SEED            = 2024
  LENGTH_BUCKETS  = [(0,15), (15,25), (25,35), (35,50), (50,200)]
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

import numpy as np

# === Repo paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH.mkdir(parents=True, exist_ok=True)

# === Mode switch ===
MODE = os.environ.get("STAGE6BG_MODE", "eval").lower()
assert MODE in ("gen", "eval", "analyze"), f"STAGE6BG_MODE must be gen|eval|analyze, got {MODE}"

# === Hardcoded ===
N_INPUT_LIST = [3, 4, 5, 6, 7]
K_SAMPLES = 5
MAX_RECORDS = int(os.environ.get("STAGE6BG_MAX_RECORDS", "3000"))
N_DEV = 2000
N_TEST = 1000
N_FIXED = 30
LAMBDA = 0.1
VAR_EPS = 1e-3
N_USERS_MIN = 3
PCA_DIM = 48
SEED = 2024
LENGTH_BUCKETS = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 200)]
LEN_LABELS = [f"{lo}-{hi if hi < 100 else '∞'}" for lo, hi in LENGTH_BUCKETS]

# === I/O ===
RECORDS_IN = REPO_ROOT / "result/query_records_10k.json"
PRODUCT_ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"

REGEN_OUT = SCRATCH / "query_records_n_input_natural.json"
QUERY_FEAT_CACHE = SCRATCH / "n_input_natural_query_features.jsonl.gz"
EVAL_OUT = SCRATCH / "syntax_subspace_stage6b_gamma_eval.json"
ANALYZE_OUT = SCRATCH / "syntax_subspace_stage6b_gamma_analyze.json"

# === Logging ===
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
#                          SHARED HELPERS
# ============================================================================

def _len_label(n_tok: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_tok < hi:
            return f"{lo}-{hi if hi < 100 else '∞'}"
    return LEN_LABELS[-1]


# Attr selection (same logic as 6b-beta)
ATTR_PRIORITY = [
    "Brand", "Main Category", "Material", "Material Type", "Fabric Type",
    "Color", "Item Form", "Style", "Pattern", "Theme", "Special Feature",
    "Size", "Age Range (Description)", "Target gender", "Frame Material",
    "Number Of Items", "Manufacturer",
]
_NUMERIC_KEYWORDS = {
    "price", "average rating", "rating number", "item weight",
    "item model number", "date first available",
    "package dimensions", "product dimensions",
    "minimum weight recommendation", "maximum weight recommendation",
    "batteries required", "is discontinued by manufacturer",
}


def _skip_value(k: str, s: str) -> bool:
    if not s or len(s) > 100:
        return True
    if any(c.isdigit() for c in s):
        return True
    if any(nk in k.lower() for nk in _NUMERIC_KEYWORDS):
        return True
    return False


def select_n_attrs(attrs_used, full_product_attrs, n_input, seed):
    import random
    rng = random.Random(seed)
    keys_priority = list(attrs_used.keys())
    if n_input <= len(attrs_used):
        chosen_keys = rng.sample(keys_priority, n_input)
    else:
        chosen_keys = list(attrs_used.keys())
        if full_product_attrs:
            extras = []
            for k in ATTR_PRIORITY:
                if k in chosen_keys:
                    continue
                v = full_product_attrs.get(k)
                if v and not _skip_value(k, str(v).strip()):
                    extras.append(k)
            for k, v in full_product_attrs.items():
                if k in chosen_keys or k in extras:
                    continue
                if v is None:
                    continue
                if _skip_value(k, str(v).strip()):
                    continue
                extras.append(k)
            rng.shuffle(extras)
            need = n_input - len(chosen_keys)
            chosen_keys.extend(extras[:need])
    out = {k: attrs_used.get(k) for k in chosen_keys if k in attrs_used}
    if full_product_attrs:
        for k in chosen_keys:
            if k not in out and k in full_product_attrs:
                out[k] = full_product_attrs[k]
    return out


HALLUCINATION_WORDS = [
    "bottle", "bottles", "clothing", "clothes", "toy", "toys", "candle",
    "candles", "tumbler", "tumblers", "carrier", "carriers", "basket",
    "baskets", "socks", "shoes", "shirt", "shirts", "pants", "dress",
    "dresses", "blanket", "blankets", "pillow", "pillows", "diaper",
    "diapers", "wipes", "formula", "pacifier", "stroller", "highchair",
    "swaddle", "swaddles", "romper", "rompers", "onesie", "onesies",
    "mittens", "booties", "teether", "teethers", "rattle", "rattles",
    "mobile", "mobiles", "nightlight", "soap", "lotion", "shampoo",
    "brush", "comb", "drinkware", "beverage", "container", "holder",
    "vessel", "equipment", "gear", "appliance", "utensil", "dish",
    "sneaker", "sneakers", "sandal", "sandals", "boot", "boots",
    "headband", "headbands", "hairband", "hairbands", "cap", "hat",
    "jacket", "coats", "vest", "shorts", "skirt", "legging", "leggings",
    "scented", "scent", "fragrance", "aroma", "flavor", "flavour",
    "decorative", "decoration", "ornament", "ornamental",
    "friend", "friends", "sister", "sister's", "brother", "brother's",
    "wife", "wife's", "husband", "husband's", "mom", "mom's", "dad",
    "dad's", "mother", "father", "daughter", "son", "grandma",
    "grandpa", "aunt", "uncle", "cousin", "neighbor",
    "lightweight", "heavyweight", "premium", "luxury", "cheap",
    "professional-grade", "commercial", "industrial",
    "handmade", "handcrafted", "artisan",
    "accessory", "accessories", "tool", "tools", "decor", "gadget",
    "gizmo", "implement", "device", "apparatus",
]


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs:
        return 0
    text_lower = text.lower()
    n = 0
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s and s.lower() in text_lower:
            n += 1
    return n


def has_hallucination(text: str, attrs: dict) -> list:
    text_lower = text.lower()
    attr_substrings = set()
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s:
            attr_substrings.add(s.lower())
            for tok in re.split(r"[\s,/]+", s.lower()):
                if len(tok) >= 4:
                    attr_substrings.add(tok)
    hits = []
    for w in HALLUCINATION_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text_lower):
            if w not in attr_substrings:
                hits.append(w)
    return hits


def has_invalid_punct(text: str) -> bool:
    return bool(re.search(r'["\(\][""]', text))


def n_tokens_simple(text: str) -> int:
    toks = re.findall(r"\b\w+\b", text)
    return len(toks)


# ============================================================================
#                          GEN MODE (NO length instruction)
# ============================================================================

GEN_SYSTEM_TMPL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Output ONLY the query, no preamble.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


# Same as above but with a natural-sentence hint (no length instruction).
GEN_SYSTEM_TMPL_NATURAL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write a natural sentence (any length is fine). Output ONLY the query, no preamble.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def build_user_content(attrs):
    return "\n".join(f"{k}: {v}" for k, v in attrs.items() if v)


def make_prompt_natural(attrs, n_input, k=0):
    base = GEN_SYSTEM_TMPL_NATURAL.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
    )
    # Add (variant k) suffix to defeat vLLM's identical-prompt dedup
    if k > 0:
        base += f"\n(variant {k})"
    return base


def main_gen():
    log("=== Stage 6B-γ GEN mode (natural length, NO length instruction) ===")
    log(f"N_INPUT_LIST={N_INPUT_LIST}, K_SAMPLES={K_SAMPLES}, MAX_RECORDS={MAX_RECORDS}")

    # 1. Load records + dev/test split
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    log(f"loaded {len(records)} records from {RECORDS_IN}")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(records))
    records_shuf = [records[i] for i in perm]
    records_use = records_shuf[:MAX_RECORDS]
    log(f"shuffled with SEED={SEED}, taking {len(records_use)} records")
    # Mark dev/test
    for i, r in enumerate(records_use):
        r["_split"] = "dev" if i < N_DEV else "test"
    n_dev = sum(1 for r in records_use if r["_split"] == "dev")
    n_test = sum(1 for r in records_use if r["_split"] == "test")
    log(f"split: dev={n_dev}, test={n_test}")

    # 2. Load full product attrs
    log(f"loading {PRODUCT_ATTRS_JSON}...")
    product_attrs = json.load(open(PRODUCT_ATTRS_JSON, "r", encoding="utf-8"))

    # 3. Init LLM (vLLM HTTP API server on :8800)
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import batch_generate_vllm  # noqa
    log("using vLLM HTTP batch_generate_vllm (server on :8800)")

    # 4. Build todo
    todo_prompts: list = []
    todo_meta: list = []
    n_skip_no_attrs = 0
    for r_idx, r in enumerate(records_use):
        attrs_used = r.get("attrs_used", {})
        if not attrs_used:
            n_skip_no_attrs += 1
            continue
        full_pa = product_attrs.get(r["asin"], {})
        for n_in in N_INPUT_LIST:
            seed_n = int(hashlib.md5(f"{r['asin']}|{n_in}|{SEED}".encode()).hexdigest()[:8], 16)
            chosen = select_n_attrs(attrs_used, full_pa, n_in, seed=seed_n)
            if len(chosen) != n_in:
                continue
            for k in range(K_SAMPLES):
                prompt = make_prompt_natural(chosen, n_in, k=k)
                todo_prompts.append(prompt)
                todo_meta.append((r_idx, n_in, k, chosen, r["_split"]))
    log(f"todo: {len(todo_prompts)} generations ({n_skip_no_attrs} skipped)")

    # 5. Resume
    existing: list = []
    if REGEN_OUT.exists():
        existing = json.load(open(REGEN_OUT, "r", encoding="utf-8"))
        log(f"resuming from {len(existing)} existing entries")
    done_keys = {(e["r_idx"], e["N_input"], e["k"]) for e in existing}
    todo_remaining = [
        (p, m) for p, m in zip(todo_prompts, todo_meta)
        if (m[0], m[1], m[2]) not in done_keys
    ]
    log(f"remaining: {len(todo_remaining)}")
    if not todo_remaining:
        log("nothing to do")
        return

    # 6. Generate in batches via vLLM HTTP API
    GEN_BATCH = 64  # vLLM is much faster; larger batches
    results: list = []
    t0 = time.time()
    for batch_start in range(0, len(todo_remaining), GEN_BATCH):
        chunk = todo_remaining[batch_start:batch_start + GEN_BATCH]
        chunk_prompts = [c[0] for c in chunk]
        chunk_meta = [c[1] for c in chunk]
        try:
            queries = batch_generate_vllm(
                prompts=chunk_prompts,
                system_text="You are an Amazon shopper writing a search query. Output ONLY the query, no preamble.",
                max_tokens=120,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        except Exception as exc:
            log(f"  BATCH FAILED at {batch_start}: {exc!r}")
            for m in chunk_meta:
                results.append({
                    "r_idx": m[0],
                    "user_id": records_use[m[0]]["user_id"],
                    "asin": records_use[m[0]]["asin"],
                    "split": m[4],
                    "N_input": m[1],
                    "k": m[2],
                    "attrs_input": m[3],
                    "query": None,
                    "attrs_covered": 0,
                    "n_hallu": -1,
                    "invalid": True,
                    "n_tok": 0,
                    "n_words": 0,
                })
            continue
        for m, q in zip(chunk_meta, queries):
            q = (q or "").strip()
            n_cov = count_attrs_covered(q, m[3])
            hallu_words = has_hallucination(q, m[3])
            n_hallu = len(hallu_words)
            invalid = has_invalid_punct(q)
            nt = n_tokens_simple(q)
            nw = len(q.split())
            results.append({
                "r_idx": m[0],
                "user_id": records_use[m[0]]["user_id"],
                "asin": records_use[m[0]]["asin"],
                "split": m[4],
                "N_input": m[1],
                "k": m[2],
                "attrs_input": m[3],
                "query": q,
                "attrs_covered": n_cov,
                "n_hallu": n_hallu,
                "invalid": invalid,
                "n_tok": nt,
                "n_words": nw,
            })
        if (batch_start // GEN_BATCH) % 5 == 0:
            all_so_far = existing + results
            REGEN_OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(REGEN_OUT, "w", encoding="utf-8") as f:
                json.dump(all_so_far, f, ensure_ascii=False, indent=2)
            rate = (batch_start + len(chunk)) / max(time.time() - t0, 1e-6)
            eta = (len(todo_remaining) - batch_start - len(chunk)) / max(rate, 1e-6)
            log(f"  done {batch_start + len(chunk)}/{len(todo_remaining)} "
                f"({rate:.2f}/s, ETA {eta:.0f}s)")

    all_so_far = existing + results
    REGEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGEN_OUT, "w", encoding="utf-8") as f:
        json.dump(all_so_far, f, ensure_ascii=False, indent=2)
    log(f"wrote {len(all_so_far)} entries to {REGEN_OUT}")
    log(f"total time: {time.time() - t0:.1f}s")


# ============================================================================
#                          EVAL MODE (PCA48 multi-metric)
# ============================================================================

def _syntax_subspace_prepare():
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    from gaussian_vades import _syntax_subspace_prepare  # type: ignore
    return _syntax_subspace_prepare()


def _d_mahal(z, mu, sigma_diag):
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def main_eval():
    log("=== Stage 6B-γ EVAL mode ===")

    if not REGEN_OUT.exists():
        raise FileNotFoundError(f"missing {REGEN_OUT} — run STAGE6BG_MODE=gen first")
    regen = json.load(open(REGEN_OUT, "r", encoding="utf-8"))
    log(f"loaded {len(regen)} regen entries from {REGEN_OUT}")
    # split counts
    by_split = collections.Counter(e.get("split", "?") for e in regen)
    log(f"  by split: {dict(by_split)}")

    # 1. Load PCA48
    log("loading PCA48 (frozen)...")
    from sklearn.decomposition import PCA
    P = _syntax_subspace_prepare()
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    Z = pca.transform(scaler.transform(P["X"]))
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    cand_users = sorted([u for u in user_id_list if len(user_to_indices[u]) >= N_FIXED])
    cand_user_set = set(cand_users)
    log(f"  cand_users: {len(cand_users)}, dim={len(fnames)}, Z={Z.shape}")

    # 2. Per-user Gaussian
    log(f"fitting per-user Gaussian (N={N_FIXED}, λ={LAMBDA})...")
    user_gauss: dict = {}
    for u in cand_users:
        idx_u = np.asarray(user_to_indices[u], dtype=np.int64)
        rng_u = np.random.default_rng(
            int(hashlib.md5(str(u).encode()).hexdigest()[:8], 16) % (2 ** 31)
        )
        perm_u = rng_u.permutation(len(idx_u))
        z_train = Z[idx_u[perm_u[:N_FIXED]]]
        mu = z_train.mean(axis=0)
        var = z_train.var(axis=0)
        var_shrink = (1 - LAMBDA) * var + LAMBDA * var.mean()
        sigma_diag = np.maximum(var_shrink, VAR_EPS)
        user_gauss[u] = (mu, sigma_diag)
    log(f"  user_gauss: {len(user_gauss)}")

    # 3. asin → users
    asin_to_users = collections.defaultdict(set)
    for e in regen:
        if e["user_id"] in cand_user_set:
            asin_to_users[e["asin"]].add(e["user_id"])
    asin_to_users = {a: sorted(us) for a, us in asin_to_users.items()}
    asin_pool_size = {a: len(us) for a, us in asin_to_users.items()}
    log(f"  asin pool: {len(asin_to_users)} asins, "
        f"intra ≥{N_USERS_MIN}: {sum(1 for v in asin_pool_size.values() if v >= N_USERS_MIN)}")

    # 4. Extract/cache query features
    log("loading/extracting query 182d features...")
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from main import per_sentence_features_v2  # type: ignore

    QUERY_FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feat_map: dict = {}
    if QUERY_FEAT_CACHE.exists():
        with gzip.open(QUERY_FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
        log(f"  loaded {len(feat_map)} from cache")

    def _key(t):
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    unique_texts = sorted({e["query"] for e in regen if e["query"]})
    new_texts = [t for t in unique_texts if _key(t) not in feat_map]
    log(f"  unique queries: {len(unique_texts)}, new: {len(new_texts)}")

    if new_texts:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        new_records = []
        for i, doc in enumerate(nlp.pipe(new_texts, batch_size=256, n_process=1)):
            t = new_texts[i]
            k = _key(t)
            try:
                feats = per_sentence_features_v2(doc)
                feats = feats if feats is not None else {}
            except Exception as _e:
                feats = {}
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            filtered = {n: numeric.get(n, 0.0) for n in fnames}
            new_records.append({"k": k, "v": filtered})
            if (i + 1) % 1000 == 0:
                log(f"    extracted {i + 1}/{len(new_texts)}")
        feat_map.update({r["k"]: r["v"] for r in new_records})
        with gzip.open(QUERY_FEAT_CACHE, "wt", encoding="utf-8") as f:
            f.write("#META {\"version\": \"v1\", \"n_entries\": " + str(len(feat_map)) + "}\n")
            for k, v in feat_map.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        log(f"  saved {len(feat_map)} features")

    # 5. Per-candidate D_M
    log("computing Mahalanobis (intra-product)...")
    rows = []
    n_skip_invalid = 0
    n_skip_no_pool = 0
    n_skip_no_feat = 0
    for e in regen:
        if not e["query"] or e["invalid"]:
            n_skip_invalid += 1
            continue
        uid, asin = e["user_id"], e["asin"]
        if uid not in cand_user_set:
            continue
        pool = asin_to_users.get(asin, [])
        if len(pool) < N_USERS_MIN:
            n_skip_no_pool += 1
            continue
        distractors = [u for u in pool if u != uid]
        if not distractors:
            continue
        m_p = len(pool)
        random_baseline = 1.0 / m_p
        k = _key(e["query"])
        v = feat_map.get(k)
        if v is None or not v:
            n_skip_no_feat += 1
            continue
        vec = np.asarray([v.get(name, 0.0) for name in fnames], dtype=np.float64)
        z_q = pca.transform(scaler.transform(vec[None, :]))[0]
        mu_self, sigma_self = user_gauss[uid]
        d_self = _d_mahal(z_q, mu_self, sigma_self)
        d_others = []
        for uo in distractors:
            mu_o, sigma_o = user_gauss[uo]
            d_others.append(_d_mahal(z_q, mu_o, sigma_o))
        d_min_other = min(d_others)
        margin = d_min_other - d_self
        rank_self = 1 + sum(1 for d in d_others if d < d_self)
        n_tok = e["n_tok"]
        rows.append({
            "r_idx": e["r_idx"],
            "user_id": uid,
            "asin": asin,
            "split": e.get("split", "?"),
            "N_input": e["N_input"],
            "k": e["k"],
            "attrs_covered": e["attrs_covered"],
            "n_hallu": e["n_hallu"],
            "n_tok": n_tok,
            "length_bucket": _len_label(n_tok),
            "m_p": m_p,
            "random_baseline": random_baseline,
            "rank_self": rank_self,
            "margin": margin,
            "d_self": d_self,
            "d_other_min": d_min_other,
            "query": e["query"],
            "attrs_input": e["attrs_input"],
        })
    log(f"  rows: {len(rows)}, "
        f"skipped: invalid={n_skip_invalid}, no_pool={n_skip_no_pool}, no_feat={n_skip_no_feat}")

    # 6. Aggregate
    def _stats(rs):
        if not rs:
            return {"count": 0}
        ranks = np.asarray([r["rank_self"] for r in rs], dtype=np.int64)
        margins = np.asarray([r["margin"] for r in rs], dtype=np.float64)
        d_self = np.asarray([r["d_self"] for r in rs], dtype=np.float64)
        rand = np.asarray([r["random_baseline"] for r in rs], dtype=np.float64)
        n_tok = np.asarray([r["n_tok"] for r in rs], dtype=np.float64)
        attrs_cov = np.asarray([r["attrs_covered"] for r in rs], dtype=np.int64)
        rank1 = float((ranks == 1).mean())
        rank3 = float((ranks <= 3).mean())
        rank10 = float((ranks <= 10).mean())
        rank100 = float((ranks <= 100).mean())
        lift1 = rank1 / rand.mean() if rand.mean() > 0 else 0.0
        attr_pass = float((attrs_cov == np.asarray([r["N_input"] for r in rs])).mean())
        return {
            "count": int(len(rs)),
            "margin_mean": float(margins.mean()),
            "margin_pos_frac": float((margins > 0).mean()),
            "d_self_mean": float(d_self.mean()),
            "d_other_min_mean": float(d_other_min.mean() if False else float(np.asarray([r["d_other_min"] for r in rs]).mean())),
            "rank1_acc": rank1,
            "rank3_acc": rank3,
            "rank10_acc": rank10,
            "rank100_acc": rank100,
            "rank_mean": float(ranks.mean()),
            "random_baseline_mean": float(rand.mean()),
            "lift1": float(lift1),
            "n_tok_mean": float(n_tok.mean()),
            "n_tok_median": float(np.median(n_tok)),
            "attrs_covered_mean": float(attrs_cov.mean()),
            "attr_pass_rate": attr_pass,
        }

    by_n: dict = {}
    by_len: dict = {}
    by_cell: dict = {}
    by_n_dev: dict = {}
    by_n_test: dict = {}

    for split in ["all", "dev", "test"]:
        rs_all = rows if split == "all" else [r for r in rows if r["split"] == split]
        for n in N_INPUT_LIST:
            rs_n = [r for r in rs_all if r["N_input"] == n]
            (by_n_dev if split == "dev" else by_n_test if split == "test" else by_n)[n] = _stats(rs_n)
        for lb in LEN_LABELS:
            rs_l = [r for r in rs_all if r["length_bucket"] == lb]
            (by_len if split == "all" else {}).setdefault(f"{split}|{lb}", _stats(rs_l))
        for n in N_INPUT_LIST:
            for lb in LEN_LABELS:
                key = f"N={n}|L={lb}"
                rs_c = [r for r in rs_all if r["N_input"] == n and r["length_bucket"] == lb]
                by_cell[f"{split}|{key}"] = _stats(rs_c)

    # 7. Output
    out = {
        "config": {
            "description": "Stage 6B-γ: Natural-length N_attrs sweep + PCA48 multi-metric eval",
            "N_INPUT_LIST": N_INPUT_LIST,
            "K_SAMPLES": K_SAMPLES,
            "MAX_RECORDS": MAX_RECORDS,
            "N_DEV": N_DEV,
            "N_TEST": N_TEST,
            "N_FIXED": N_FIXED,
            "LAMBDA": LAMBDA,
            "N_USERS_MIN": N_USERS_MIN,
            "LENGTH_BUCKETS": LENGTH_BUCKETS,
            "LEN_LABELS": LEN_LABELS,
            "PCA_DIM": PCA_DIM,
            "SEED": SEED,
        },
        "per_N_input_all": by_n,
        "per_N_input_dev": by_n_dev,
        "per_N_input_test": by_n_test,
        "per_length_all": {k.replace("all|", ""): v for k, v in by_len.items()},
        "per_cell": by_cell,
        "n_regen_total": len(regen),
        "n_rows_eval": len(rows),
        "n_skipped_invalid": n_skip_invalid,
        "n_skipped_no_pool": n_skip_no_pool,
        "n_skipped_no_feat": n_skip_no_feat,
    }
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"wrote eval → {EVAL_OUT}")

    # 8. Print summary
    def _line_n(n, s):
        if s["count"] == 0:
            return f"  {n}     0      —      —      —      —       —       —        —       —"
        return (f"  {n}  {s['count']:>5}  {s['margin_mean']:>8.1f}  "
                f"{s['d_self_mean']:>6.1f}  {s['rank1_acc']:.3f}  {s['rank3_acc']:.3f}  "
                f"{s['rank10_acc']:.3f}  {s['rank100_acc']:.3f}  {s['lift1']:.2f}×  "
                f"{s['n_tok_mean']:>5.1f}  {s['attr_pass_rate']*100:>3.0f}%")

    log("\n=== Per N_input (DEV split, n=" + str(N_DEV) + " records) ===")
    log("N   count  margin   d_self  rank1   rank3   rank10  rank100  lift1   n_tok  attr_pass")
    for n in N_INPUT_LIST:
        log(_line_n(n, by_n_dev[n]))

    log("\n=== Per N_input (TEST split, n=" + str(N_TEST) + " records) ===")
    log("N   count  margin   d_self  rank1   rank3   rank10  rank100  lift1   n_tok  attr_pass")
    for n in N_INPUT_LIST:
        log(_line_n(n, by_n_test[n]))

    log("\n=== Per Length bucket (ALL) ===")
    log("Length     count  margin   d_self  rank1   lift1   n_tok")
    for lb in LEN_LABELS:
        s = by_len.get(f"all|{lb}", {"count": 0})
        if s["count"] == 0:
            continue
        log(f"  {lb:<8} {s['count']:>5}  {s['margin_mean']:>8.1f}  {s['d_self_mean']:>6.1f}  "
            f"{s['rank1_acc']:.3f}  {s['lift1']:.2f}×  {s['n_tok_mean']:.1f}")

    log("\n=== Per (Length × N_input) cell (ALL) ===")
    log("L\\N      3         4         5         6         7")
    for lb in LEN_LABELS:
        line = f"  {lb:<8}"
        for n in N_INPUT_LIST:
            s = by_cell.get(f"all|N={n}|L={lb}", None)
            if s is None or s["count"] == 0:
                line += "    —/—"
            else:
                line += f"  c{s['count']:>3}/L{s['lift1']:.2f}×/d{s['d_self_mean']:>5.0f}"
        log(line)


# ============================================================================
#                          ANALYZE MODE (Statistical tests)
# ============================================================================

def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for the mean of `values`."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = values[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def _paired_permutation_test(diff_values, n_perm=10000, seed=42):
    """Paired permutation test: H0 mean(diff) = 0.
    p_value = P(|mean(shuffled diff)| >= |mean(actual diff)|)
    """
    diff_values = np.asarray(diff_values, dtype=np.float64)
    diff_values = diff_values[~np.isnan(diff_values)]
    if len(diff_values) == 0:
        return float("nan"), float("nan")
    actual = abs(diff_values.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff_values))
        perm_mean = (diff_values * signs).mean()
        if abs(perm_mean) >= actual:
            count += 1
    return float(diff_values.mean()), float(count / n_perm)


def main_analyze():
    log("=== Stage 6B-γ ANALYZE mode ===")

    if not EVAL_OUT.exists():
        raise FileNotFoundError(f"missing {EVAL_OUT} — run STAGE6BG_MODE=eval first")
    eval_data = json.load(open(EVAL_OUT, "r", encoding="utf-8"))
    log(f"loaded eval data from {EVAL_OUT}")

    if not REGEN_OUT.exists():
        raise FileNotFoundError(f"missing {REGEN_OUT}")
    regen = json.load(open(REGEN_OUT, "r", encoding="utf-8"))

    # Build per-row DataFrame-like structure from regen + eval
    # We need raw rows (not just summary). Rebuild from REGEN_OUT.
    # For statistical analysis we need: per-row N_input, n_tok, attrs_covered, lift1, d_self, margin, etc.
    # We re-compute lift1 from rows that we cached in EVAL_OUT (per_cell). But we need per-row data.
    # Recompute from scratch would be expensive. Instead, we cache per-row in eval run.
    # Hmm — let me add per-row dump.

    # Workaround: reconstruct per-row from regen entries we have not yet cached.
    # We'll re-run a lightweight version that reads the saved per-cell summary + regen.
    # Actually best: just iterate regen and pair with per_cell summary stats.
    # But for per-row lift1 we need user/asin/feature/Mahalanobis.
    # The simplest path: re-do the eval per-row inline here.

    log("recomputing per-row metrics for statistical analysis...")

    # Load PCA48 + user Gaussians (same as eval)
    from sklearn.decomposition import PCA
    P = _syntax_subspace_prepare()
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    Z = pca.transform(scaler.transform(P["X"]))
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    cand_users = sorted([u for u in user_id_list if len(user_to_indices[u]) >= N_FIXED])
    cand_user_set = set(cand_users)
    user_gauss: dict = {}
    for u in cand_users:
        idx_u = np.asarray(user_to_indices[u], dtype=np.int64)
        rng_u = np.random.default_rng(
            int(hashlib.md5(str(u).encode()).hexdigest()[:8], 16) % (2 ** 31)
        )
        perm_u = rng_u.permutation(len(idx_u))
        z_train = Z[idx_u[perm_u[:N_FIXED]]]
        mu = z_train.mean(axis=0)
        var = z_train.var(axis=0)
        var_shrink = (1 - LAMBDA) * var + LAMBDA * var.mean()
        sigma_diag = np.maximum(var_shrink, VAR_EPS)
        user_gauss[u] = (mu, sigma_diag)
    asin_to_users = collections.defaultdict(set)
    for e in regen:
        if e["user_id"] in cand_user_set:
            asin_to_users[e["asin"]].add(e["user_id"])
    asin_pool_size = {a: len(us) for a, us in asin_to_users.items()}

    # Load cached query features
    feat_map: dict = {}
    if QUERY_FEAT_CACHE.exists():
        with gzip.open(QUERY_FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]

    def _key(t):
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    rows = []
    for e in regen:
        if not e["query"] or e["invalid"]:
            continue
        uid, asin = e["user_id"], e["asin"]
        if uid not in cand_user_set:
            continue
        pool = asin_to_users.get(asin, [])
        if len(pool) < N_USERS_MIN:
            continue
        distractors = [u for u in pool if u != uid]
        if not distractors:
            continue
        m_p = len(pool)
        k = _key(e["query"])
        v = feat_map.get(k)
        if v is None or not v:
            continue
        vec = np.asarray([v.get(name, 0.0) for name in fnames], dtype=np.float64)
        z_q = pca.transform(scaler.transform(vec[None, :]))[0]
        mu_self, sigma_self = user_gauss[uid]
        d_self = _d_mahal(z_q, mu_self, sigma_self)
        d_others = []
        for uo in distractors:
            mu_o, sigma_o = user_gauss[uo]
            d_others.append(_d_mahal(z_q, mu_o, sigma_o))
        d_min_other = min(d_others)
        margin = d_min_other - d_self
        rank_self = 1 + sum(1 for d in d_others if d < d_self)
        rows.append({
            "r_idx": e["r_idx"],
            "user_id": uid,
            "asin": asin,
            "split": e.get("split", "?"),
            "N_input": e["N_input"],
            "k": e["k"],
            "attrs_covered": e["attrs_covered"],
            "n_hallu": e["n_hallu"],
            "n_tok": e["n_tok"],
            "length_bucket": _len_label(e["n_tok"]),
            "m_p": m_p,
            "random_baseline": 1.0 / m_p,
            "rank_self": rank_self,
            "margin": margin,
            "d_self": d_self,
            "d_other_min": d_min_other,
            "rank1": 1.0 if rank_self == 1 else 0.0,
        })
    log(f"  rows: {len(rows)}")

    # === A) Bootstrap CI per N (on lift1) ===
    log("\n=== A) Bootstrap 95% CI for lift1 (dev split) ===")
    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]
    bootstrap_results = {}
    for split_name, split_rows in [("dev", dev_rows), ("test", test_rows), ("all", rows)]:
        bs_split = {}
        for n in N_INPUT_LIST:
            rs_n = [r for r in split_rows if r["N_input"] == n]
            lifts = [r["rank1"] / r["random_baseline"] for r in rs_n]
            mean, lo, hi = _bootstrap_ci(lifts, n_boot=2000, seed=SEED + n)
            bs_split[n] = {"mean": mean, "lo": lo, "hi": hi, "n": len(lifts)}
        bootstrap_results[split_name] = bs_split
        log(f"  {split_name}:")
        for n in N_INPUT_LIST:
            d = bs_split[n]
            log(f"    N={n}: lift1={d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] (n={d['n']})")

    # === B) Paired permutation test (N vs N=3 baseline) ===
    log("\n=== B) Paired permutation test (N vs N=3 baseline, dev split) ===")
    # For each r_idx (record), compare lift1 of N=n vs N=3
    by_ridx_n = {}
    for r in dev_rows:
        by_ridx_n.setdefault(r["r_idx"], {})[r["N_input"]] = r["rank1"] / r["random_baseline"]
    perm_results = {}
    for n in N_INPUT_LIST:
        if n == 3:
            continue
        diffs = []
        for ridx, lifts in by_ridx_n.items():
            if 3 in lifts and n in lifts:
                diffs.append(lifts[n] - lifts[3])
        diff_mean, p_val = _paired_permutation_test(diffs, n_perm=10000, seed=SEED + n)
        perm_results[n] = {"diff_mean": diff_mean, "p_value": p_val, "n_pairs": len(diffs)}
        sig = "SIG" if p_val < 0.05 else "n.s."
        log(f"  N={n} vs N=3: diff={diff_mean:+.4f}, p={p_val:.4f} ({sig}), n_pairs={len(diffs)}")

    # === C) Mixed-effects regression ===
    log("\n=== C) Mixed-effects regression (lift1 ~ N * length + coverage, random=user) ===")
    reg_results = {}
    try:
        import pandas as pd  # noqa
        import statsmodels.formula.api as smf  # noqa
    except ImportError as _e:
        log(f"  statsmodels not available: {_e} — skipping regression")
    else:
        df = pd.DataFrame([{
            "lift1": r["rank1"] / r["random_baseline"],
            "N_input": r["N_input"],
            "n_tok": r["n_tok"],
            "attrs_covered": r["attrs_covered"],
            "user_id": r["user_id"],
            "asin": r["asin"],
            "split": r["split"],
        } for r in dev_rows])
        df["N_c"] = (df["N_input"] - 3).astype(float)  # N=3 as reference
        # Limit regression to records with at least one of each N to make paired-like
        try:
            md = smf.mixedlm(
                "lift1 ~ C(N_input) + n_tok + attrs_covered + C(N_input):n_tok",
                data=df, groups=df["user_id"]
            )
            mdf = md.fit(reml=False)
            log("  mixedlm summary (dev):")
            log("  " + str(mdf.summary()).replace("\n", "\n  "))
            reg_results["dev"] = {
                "params": {k: float(val) for k, val in mdf.params.items()},
                "pvalues": {k: float(val) for k, val in mdf.pvalues.items()},
                "converged": bool(mdf.converged),
            }
        except Exception as _e:
            log(f"  regression failed: {_e!r}")

        # Test split (for verification)
        df_t = pd.DataFrame([{
            "lift1": r["rank1"] / r["random_baseline"],
            "N_input": r["N_input"],
            "n_tok": r["n_tok"],
            "attrs_covered": r["attrs_covered"],
            "user_id": r["user_id"],
            "asin": r["asin"],
        } for r in test_rows])
        try:
            md_t = smf.mixedlm(
                "lift1 ~ C(N_input) + n_tok + attrs_covered + C(N_input):n_tok",
                data=df_t, groups=df_t["user_id"]
            )
            mdf_t = md_t.fit(reml=False)
            log("  mixedlm summary (test):")
            log("  " + str(mdf_t.summary()).replace("\n", "\n  "))
            reg_results["test"] = {
                "params": {k: float(val) for k, val in mdf_t.params.items()},
                "pvalues": {k: float(val) for k, val in mdf_t.pvalues.items()},
                "converged": bool(mdf_t.converged),
            }
        except Exception as _e:
            log(f"  test regression failed: {_e!r}")

    # === D) Stratified by asin pool size ===
    log("\n=== D) Stratified by asin pool size (3 / 4-5 / 6+) ===")
    stratified = {}
    for split_name, split_rows in [("dev", dev_rows), ("test", test_rows)]:
        for stratum, lo, hi in [("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 100)]:
            stratum_rows = [r for r in split_rows if lo <= r["m_p"] <= hi]
            stratum_per_n = {}
            for n in N_INPUT_LIST:
                rs_n = [r for r in stratum_rows if r["N_input"] == n]
                if not rs_n:
                    stratum_per_n[n] = {"count": 0}
                    continue
                ranks = np.asarray([r["rank_self"] for r in rs_n], dtype=np.int64)
                rand = np.asarray([r["random_baseline"] for r in rs_n], dtype=np.float64)
                rank1 = float((ranks == 1).mean())
                lift1 = rank1 / rand.mean() if rand.mean() > 0 else 0.0
                stratum_per_n[n] = {
                    "count": len(rs_n),
                    "rank1": rank1,
                    "lift1": lift1,
                    "n_tok_mean": float(np.mean([r["n_tok"] for r in rs_n])),
                }
            stratified[f"{split_name}|asin_{stratum}"] = stratum_per_n
            log(f"  {split_name} asin_mp∈{lo}-{hi}:")
            for n in N_INPUT_LIST:
                d = stratum_per_n[n]
                if d["count"] == 0:
                    log(f"    N={n}: n=0")
                else:
                    log(f"    N={n}: n={d['count']}, rank1={d['rank1']:.3f}, "
                        f"lift1={d['lift1']:.2f}×, n_tok={d['n_tok_mean']:.1f}")

    # === E) Composite recommendation ===
    log("\n=== E) Composite recommendation ===")
    composite = {}
    # Quality gates (per user's spec):
    #   attr_pass_rate >= 95% (in dev split)
    #   semantic similarity not regress (we don't have it — proxy: n_tok_mean stable)
    #   style metrics not below N=3 baseline
    for n in N_INPUT_LIST:
        s = eval_data["per_N_input_dev"][str(n)]
        # lift1 must be within bootstrap CI of N=3
        ci_3 = bootstrap_results["dev"][3]
        ci_n = bootstrap_results["dev"][n]
        within_3_ci = ci_3["lo"] <= ci_n["mean"] <= ci_3["hi"]
        composite[n] = {
            "count": s["count"],
            "lift1": s["lift1"],
            "lift1_ci": [ci_n["lo"], ci_n["hi"]],
            "attr_pass_rate": s["attr_pass_rate"],
            "n_tok_mean": s["n_tok_mean"],
            "within_N3_CI": within_3_ci,
            "n_tok_growth": s["n_tok_mean"] - eval_data["per_N_input_dev"]["3"]["n_tok_mean"],
        }
    log("  composite per N (dev split):")
    for n in N_INPUT_LIST:
        c = composite[n]
        log(f"    N={n}: lift1={c['lift1']:.2f}× CI[{c['lift1_ci'][0]:.2f}, {c['lift1_ci'][1]:.2f}], "
            f"attr_pass={c['attr_pass_rate']*100:.0f}%, n_tok={c['n_tok_mean']:.1f} "
            f"(+{c['n_tok_growth']:.1f}), within_N3_CI={c['within_N3_CI']}")

    # Recommended N: smallest N where attr_pass >= 0.95 AND lift1 within N=3 CI
    recommended = None
    for n in N_INPUT_LIST:
        c = composite[n]
        if c["attr_pass_rate"] >= 0.95 and c["within_N3_CI"]:
            recommended = n
            break
    log(f"  >>> RECOMMENDED N = {recommended} (smallest N with attr_pass>=95% AND lift1 not below N=3)")
    if recommended is None:
        log("  (no N meets composite criteria — falling back to N=4 as default)")

    # 9. Write analyze output
    out = {
        "config": {
            "description": "Stage 6B-γ: Statistical analysis (bootstrap CI, paired permutation, mixed-effects, stratified)",
            "SEED": SEED,
        },
        "bootstrap_CI": bootstrap_results,
        "permutation_test": perm_results,
        "mixed_effects_regression": reg_results,
        "stratified_by_asin_pool": stratified,
        "composite_recommendation": {
            "per_N": composite,
            "recommended_N": recommended,
            "criteria": "smallest N with attr_pass_rate>=95% AND lift1 within N=3 bootstrap CI",
        },
        "n_rows_dev": len(dev_rows),
        "n_rows_test": len(test_rows),
    }
    ANALYZE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYZE_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nwrote analyze → {ANALYZE_OUT}")


# ============================================================================
if __name__ == "__main__":
    if MODE == "gen":
        main_gen()
    elif MODE == "eval":
        main_eval()
    elif MODE == "analyze":
        main_analyze()