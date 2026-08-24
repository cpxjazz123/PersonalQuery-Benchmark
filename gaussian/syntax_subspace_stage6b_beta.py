#!/usr/bin/env python3
"""Stage 6B-β: Controlled N_input LLM regen + PCA48 length-bucket eval.

设计:
  - Stage 6 / 6B-α 揭示 N_attrs_covered bucket 受到 length / complexity confound。
  - 6B-β 重新生成 queries, 控制 N_input ∈ {3, 4, 5, 6, 7} + 长度指令 20-30 词 (≈25-35 token)
  - 同时报告 lift@1 (vs random baseline) + (length × N_input) cell, 验证:
      "在控制 length 后, N_attrs 是否仍影响用户句法匹配?"

运行 (两步, 都需要跑):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  STAGE6BB_MODE=gen  nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      gaussian/syntax_subspace_stage6b_beta.py \
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_beta_gen.log 2>&1 &

  STAGE6BB_MODE=eval nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      gaussian/syntax_subspace_stage6b_beta.py \
      > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_beta_eval.log 2>&1 &

硬编码常量 (Rule 3):
  N_INPUT_LIST    = [3, 4, 5, 6, 7]
  TARGET_WORDS    = (20, 30)   # ≈ 25-35 tokens
  K_SAMPLES       = 2          # 每个 (record, N_input) 采样数
  MAX_RECORDS     = 2000       # 上限, 控制 LLM 成本 (≈ 2000 × 5 × 2 = 20k generations)
  N_FIXED         = 30         # 用户 Gaussian 历史句数, 与 Stage 6 一致
  LAMBDA          = 0.1        # diag shrinkage
  N_USERS_MIN     = 3          # intra-product pool 最小用户数
  SEED            = 42         # attr subsample / sample shuffle
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

# === Repo paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH.mkdir(parents=True, exist_ok=True)

# === Mode switch (env-var only, no CLI) ===
MODE = os.environ.get("STAGE6BB_MODE", "eval").lower()
assert MODE in ("gen", "eval"), f"STAGE6BB_MODE must be gen|eval, got {MODE}"

# === Hardcoded constants (Rule 3) ===
N_INPUT_LIST = [3, 4, 5, 6, 7]
TARGET_WORDS_LO = 20
TARGET_WORDS_HI = 30
K_SAMPLES = 2
MAX_RECORDS = int(os.environ.get("STAGE6BB_MAX_RECORDS", "2000"))
N_FIXED = 30
LAMBDA = 0.1
VAR_EPS = 1e-3
N_USERS_MIN = 3
PCA_DIM = 48
SEED = 42

LENGTH_BUCKETS = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 200)]
LEN_LABELS = [f"{lo}-{hi if hi < 100 else '∞'}" for lo, hi in LENGTH_BUCKETS]

# === I/O paths ===
RECORDS_IN = REPO_ROOT / "result/query_records_10k.json"
PRODUCT_ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"
STRICT_FILE = REPO_ROOT / "result/query_records_with_query_inject_strict_10k.json"

REGEN_OUT = SCRATCH / "query_records_n_input_sweep.json"           # gen output
QUERY_FEAT_CACHE = SCRATCH / "n_input_query_features.jsonl.gz"     # 182d feat cache
EVAL_OUT = SCRATCH / "syntax_subspace_stage6b_beta_eval.json"      # eval output

# === Logging ===
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
#                            GEN MODE
# ============================================================================

# === System prompt: control N_input + length ===
# 与 generate_strict_nohallu.py 风格一致, 但:
#   1) 显式要求 "use EXACTLY N attributes listed"
#   2) 显式长度窗口 20-30 words (≈ 25-35 tokens)
GEN_SYSTEM_TMPL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write ONE natural sentence of {W_LO}-{W_HI} words with varied syntax — clauses, "
    "coordination, prepositional phrases, relative clauses. Output ONLY the "
    "query, no preamble.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def build_user_content(attrs: dict) -> str:
    lines = []
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(f"{k}: {s}")
    return "\n".join(lines)


def make_prompt(attrs: dict, n_input: int) -> str:
    return GEN_SYSTEM_TMPL.format(
        N_INPUT=n_input,
        W_LO=TARGET_WORDS_LO,
        W_HI=TARGET_WORDS_HI,
        ATTRIBUTES=build_user_content(attrs),
    )


# === Attr selection ===
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


def select_n_attrs(
    attrs_used: dict,
    full_product_attrs: dict | None,
    n_input: int,
    seed: int,
) -> dict:
    """Select exactly n_input attrs. If n_input <= len(attrs_used), subsample.
    Else merge attrs_used + extras from full_product_attrs.
    """
    rng = __import__("random").Random(seed)
    keys_priority = list(attrs_used.keys())

    if n_input <= len(attrs_used):
        # Subsample
        chosen_keys = rng.sample(keys_priority, n_input)
    else:
        # Take all attrs_used + extras from full_product_attrs
        chosen_keys = list(attrs_used.keys())
        if full_product_attrs:
            extras = []
            # First priority extras
            for k in ATTR_PRIORITY:
                if k in chosen_keys:
                    continue
                v = full_product_attrs.get(k)
                if v and not _skip_value(k, str(v).strip()):
                    extras.append(k)
            # Then any other
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


# === Hallucination detection (copy from generate_strict_nohallu) ===
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


def score_candidate(text: str, attrs: dict) -> tuple:
    n_cov = count_attrs_covered(text, attrs)
    n_hallu = len(has_hallucination(text, attrs))
    invalid = 1 if has_invalid_punct(text) else 0
    return (n_cov, -n_hallu, -invalid)


def n_tokens_simple(text: str) -> int:
    """简易 token 计数 (空格分词 + 标点分割), 不用 NLTK 避免依赖。"""
    toks = re.findall(r"\b\w+\b", text)
    return len(toks)


def main_gen() -> None:
    log(f"=== Stage 6B-β GEN mode ===")
    log(f"N_INPUT_LIST={N_INPUT_LIST}, K_SAMPLES={K_SAMPLES}, MAX_RECORDS={MAX_RECORDS}")

    # 1. Load records
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    log(f"loaded {len(records)} records from {RECORDS_IN}")
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
        log(f"capped to {len(records)} records")

    # 2. Load full product attrs (for N_input > 5)
    log(f"loading {PRODUCT_ATTRS_JSON} for extras...")
    product_attrs = json.load(open(PRODUCT_ATTRS_JSON, "r", encoding="utf-8"))
    log(f"  {len(product_attrs)} products")

    # 3. Init QwenLocalClient (transformers backend, no vLLM needed)
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client  # noqa: E402
    client = create_qwen_local_client(with_vllm=False)
    log("QwenLocalClient ready (transformers backend)")

    # 4. Build todo
    todo_prompts: list = []
    todo_meta: list = []  # (record_idx, n_input, k)
    n_skip_no_attrs = 0
    for r_idx, r in enumerate(records):
        uid, asin = r["user_id"], r["asin"]
        attrs_used = r.get("attrs_used", {})
        if not attrs_used:
            n_skip_no_attrs += 1
            continue
        full_pa = product_attrs.get(asin, {})
        for n_in in N_INPUT_LIST:
            seed_n = int(hashlib.md5(f"{asin}|{n_in}|{SEED}".encode()).hexdigest()[:8], 16)
            chosen = select_n_attrs(attrs_used, full_pa, n_in, seed=seed_n)
            if len(chosen) != n_in:
                log(f"  WARN: {asin} N_input={n_in} only got {len(chosen)} attrs; skipping")
                continue
            prompt = make_prompt(chosen, n_in)
            for k in range(K_SAMPLES):
                todo_prompts.append(prompt)
                todo_meta.append((r_idx, n_in, k, chosen))
    log(f"todo: {len(todo_prompts)} generations, {n_skip_no_attrs} no-attrs skipped")

    # 5. Resume support
    if REGEN_OUT.exists():
        with open(REGEN_OUT, "r", encoding="utf-8") as f:
            existing = json.load(f)
        log(f"loaded existing regen file: {len(existing)} entries (will append)")
    else:
        existing = []

    done_keys = {(e["record_idx"], e["N_input"], e["k"]) for e in existing}
    todo_remaining = [
        (i, p, m) for i, (p, m) in enumerate(zip(todo_prompts, todo_meta))
        if (m[0], m[1], m[2]) not in done_keys
    ]
    log(f"remaining: {len(todo_remaining)} (already done: {len(done_keys)})")
    if not todo_remaining:
        log("nothing to do")
        return

    # 6. Generate in batches via generate_with_hidden_injection (alpha=0, no injection)
    #    injection_per_row = [None] * batch  → no style steering, pure LM gen
    GEN_BATCH = 24
    results: list = []
    t0 = time.time()
    for batch_start in range(0, len(todo_remaining), GEN_BATCH):
        chunk = todo_remaining[batch_start:batch_start + GEN_BATCH]
        chunk_prompts = [c[1] for c in chunk]
        chunk_meta = [c[2] for c in chunk]
        injection_none = [None] * len(chunk_prompts)
        try:
            queries = client.generate_with_hidden_injection(
                system_text=(
                    "You are an Amazon shopper writing a search query. "
                    "Output ONLY the query, no preamble."
                ),
                user_texts=chunk_prompts,
                injection_per_row=injection_none,
                injection_layers=[16],
                injection_alpha=0.0,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                batch_size=GEN_BATCH,
                max_input_length=384,
                mask_cjk=True,
            )
        except Exception as exc:
            log(f"  BATCH FAILED at {batch_start}: {exc!r}")
            for m in chunk_meta:
                results.append({
                    "record_idx": m[0],
                    "N_input": m[1],
                    "k": m[2],
                    "attrs_input": m[3],
                    "query": None,
                    "attrs_covered": 0,
                    "n_hallu": -1,
                    "invalid": True,
                    "n_tok": 0,
                })
            continue
        for m, q in zip(chunk_meta, queries):
            q = (q or "").strip()
            n_cov = count_attrs_covered(q, m[3])
            hallu_words = has_hallucination(q, m[3])
            n_hallu = len(hallu_words)
            invalid = has_invalid_punct(q)
            nt = n_tokens_simple(q)
            results.append({
                "record_idx": m[0],
                "N_input": m[1],
                "k": m[2],
                "attrs_input": m[3],
                "query": q,
                "attrs_covered": n_cov,
                "n_hallu": n_hallu,
                "invalid": invalid,
                "n_tok": nt,
            })
        # incremental write
        if (batch_start // GEN_BATCH) % 5 == 0:
            all_so_far = existing + results
            REGEN_OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(REGEN_OUT, "w", encoding="utf-8") as f:
                json.dump(all_so_far, f, ensure_ascii=False, indent=2)
            rate = (batch_start + len(chunk)) / max(time.time() - t0, 1e-6)
            eta = (len(todo_remaining) - batch_start - len(chunk)) / max(rate, 1e-6)
            log(f"  done {batch_start + len(chunk)}/{len(todo_remaining)} "
                f"({rate:.2f}/s, ETA {eta:.0f}s)")

    # 7. Final write
    all_so_far = existing + results
    REGEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGEN_OUT, "w", encoding="utf-8") as f:
        json.dump(all_so_far, f, ensure_ascii=False, indent=2)
    log(f"wrote {len(all_so_far)} entries to {REGEN_OUT}")
    log(f"total time: {time.time() - t0:.1f}s")


# ============================================================================
#                            EVAL MODE
# ============================================================================

def _syntax_subspace_prepare():
    """复用 gaussian_vades._syntax_subspace_prepare(): 加载 10k 用户句法 cache."""
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    from gaussian_vades import _syntax_subspace_prepare  # type: ignore
    return _syntax_subspace_prepare()


def main_eval() -> None:
    log("=== Stage 6B-β EVAL mode ===")

    # 1. Load regen queries
    if not REGEN_OUT.exists():
        raise FileNotFoundError(f"missing {REGEN_OUT} — run STAGE6BB_MODE=gen first")
    regen = json.load(open(REGEN_OUT, "r", encoding="utf-8"))
    log(f"loaded {len(regen)} regen entries from {REGEN_OUT}")

    # Map record_idx → original record (for user_id, asin)
    records_orig = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    log(f"loaded {len(records_orig)} original records")

    # 2. Load PCA48 (frozen from Stage 5B)
    log("loading PCA48 (frozen from Stage 5B)...")
    from sklearn.decomposition import PCA  # noqa: E402
    P = _syntax_subspace_prepare()
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    X = P["X"]
    Z = pca.transform(scaler.transform(X))
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    log(f"Z shape {Z.shape}, {len(user_id_list)} users, feature dim = {len(fnames)}")

    cand_users = sorted([u for u in user_id_list if len(user_to_indices[u]) >= N_FIXED])
    cand_user_set = set(cand_users)
    log(f"cand_users (≥{N_FIXED}): {len(cand_users)}")

    # 3. asin → user pool (intra-product distractor set)
    asin_to_users = collections.defaultdict(set)
    for r in regen:
        rec = records_orig[r["record_idx"]]
        uid = rec["user_id"]
        asin = rec["asin"]
        if uid in cand_user_set:
            asin_to_users[asin].add(uid)
    asin_to_users = {a: sorted(us) for a, us in asin_to_users.items()}
    asin_pool_size = {a: len(us) for a, us in asin_to_users.items()}
    log(f"asin pool: {len(asin_to_users)} asins, "
        f"intra ≥{N_USERS_MIN}: {sum(1 for v in asin_pool_size.values() if v >= N_USERS_MIN)}")

    # 4. Per-user Gaussian (diag, λ=0.1)
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
    log(f"user_gauss ready: {len(user_gauss)} users")

    def _d_mahal(z, mu, sigma_diag):
        diff = z - mu
        return float((diff * diff / sigma_diag).sum())

    # 5. Cache query features (spaCy pipe, batched)
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

    def _key(t: str) -> str:
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    unique_texts = sorted({e["query"] for e in regen if e["query"]})
    new_texts = [t for t in unique_texts if _key(t) not in feat_map]
    log(f"  unique queries: {len(unique_texts)}, new to extract: {len(new_texts)}")

    if new_texts:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm")
        BATCH = 256
        new_records = []
        n_done = 0
        for i, doc in enumerate(nlp.pipe(new_texts, batch_size=BATCH, n_process=1)):
            t = new_texts[i]
            k = _key(t)
            try:
                feats = per_sentence_features_v2(doc)
                feats = feats if feats is not None else {}
            except Exception as _e:
                log(f"  feat extract fail ({t[:30]}...): {_e}")
                feats = {}
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            filtered = {n: numeric.get(n, 0.0) for n in fnames}
            new_records.append({"k": k, "v": filtered})
            n_done += 1
            if n_done % 500 == 0:
                log(f"    extracted {n_done}/{len(new_texts)}")
        feat_map.update({r["k"]: r["v"] for r in new_records})
        with gzip.open(QUERY_FEAT_CACHE, "wt", encoding="utf-8") as f:
            f.write("#META {\"version\": \"v1\", \"n_entries\": " + str(len(feat_map)) + "}\n")
            for k, v in feat_map.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        log(f"  query feature cache saved: {len(feat_map)} entries")

    # 6. Per-candidate: D_M(q, u_self) vs D_M(q, u_distractor)
    log("computing Mahalanobis distances (intra-product)...")
    rows = []
    n_skipped_no_pool = 0
    n_skipped_no_feat = 0
    n_skipped_invalid = 0

    for e in regen:
        if not e["query"] or e["invalid"]:
            n_skipped_invalid += 1
            continue
        rec = records_orig[e["record_idx"]]
        uid, asin = rec["user_id"], rec["asin"]
        if uid not in cand_user_set:
            continue
        pool = asin_to_users.get(asin, [])
        if len(pool) < N_USERS_MIN:
            n_skipped_no_pool += 1
            continue
        distractors = [u for u in pool if u != uid]
        if not distractors:
            continue
        m_p = len(pool)
        random_baseline = 1.0 / m_p
        k = _key(e["query"])
        v = feat_map.get(k)
        if v is None or not v:
            n_skipped_no_feat += 1
            continue
        # Pick first available
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
            "record_idx": e["record_idx"],
            "user_id": uid,
            "asin": asin,
            "N_input": e["N_input"],
            "k": e["k"],
            "attrs_covered": e["attrs_covered"],
            "n_hallu": e["n_hallu"],
            "n_tok": e["n_tok"],
            "length_bucket": _len_label(e["n_tok"]),
            "m_p": m_p,
            "random_baseline": random_baseline,
            "rank_self": rank_self,
            "margin": margin,
            "d_self": d_self,
            "d_other_min": d_min_other,
        })
    log(f"  rows: {len(rows)}, "
        f"skipped: no_pool={n_skipped_no_pool}, no_feat={n_skipped_no_feat}, invalid={n_skipped_invalid}")

    # 7. Aggregate
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
        lift1 = rank1 / rand.mean() if rand.mean() > 0 else 0.0
        return {
            "count": int(len(rs)),
            "margin_mean": float(margins.mean()),
            "margin_pos_frac": float((margins > 0).mean()),
            "d_self_mean": float(d_self.mean()),
            "rank1_acc": rank1,
            "rank3_acc": rank3,
            "rank_mean": float(ranks.mean()),
            "random_baseline_mean": float(rand.mean()),
            "lift1": float(lift1),
            "n_tok_mean": float(n_tok.mean()),
            "attrs_covered_mean": float(attrs_cov.mean()),
            "attrs_covered_eq_N": float((attrs_cov == np.asarray([r["N_input"] for r in rs])).mean()),
        }

    per_n: dict = {}
    for n in N_INPUT_LIST:
        per_n[n] = _stats([r for r in rows if r["N_input"] == n])

    per_len: dict = {}
    for lb in LEN_LABELS:
        per_len[lb] = _stats([r for r in rows if r["length_bucket"] == lb])

    per_cell: dict = {}
    for n in N_INPUT_LIST:
        for lb in LEN_LABELS:
            key = f"N={n}|L={lb}"
            rs = [r for r in rows if r["N_input"] == n and r["length_bucket"] == lb]
            per_cell[key] = _stats(rs)

    # Per (length × N_input) cell where attrs_covered matches N_input exactly
    per_cell_strict: dict = {}
    for n in N_INPUT_LIST:
        for lb in LEN_LABELS:
            key = f"N={n}|L={lb}"
            rs = [
                r for r in rows
                if r["N_input"] == n
                and r["length_bucket"] == lb
                and r["attrs_covered"] == n
            ]
            per_cell_strict[key] = _stats(rs)

    # 8. Output
    out = {
        "config": {
            "description": "Stage 6B-β: Controlled N_input LLM regen + length-bucketed PCA48 eval",
            "N_INPUT_LIST": N_INPUT_LIST,
            "TARGET_WORDS": [TARGET_WORDS_LO, TARGET_WORDS_HI],
            "K_SAMPLES": K_SAMPLES,
            "MAX_RECORDS": MAX_RECORDS,
            "N_FIXED": N_FIXED,
            "LAMBDA": LAMBDA,
            "N_USERS_MIN": N_USERS_MIN,
            "LENGTH_BUCKETS": LENGTH_BUCKETS,
            "LEN_LABELS": LEN_LABELS,
            "PCA_DIM": PCA_DIM,
            "SEED": SEED,
        },
        "per_N_input": per_n,
        "per_length": per_len,
        "per_cell": per_cell,
        "per_cell_strict": per_cell_strict,
        "n_regen_total": len(regen),
        "n_records_orig": len(records_orig),
        "n_rows_eval": len(rows),
        "n_skipped_invalid": n_skipped_invalid,
        "n_skipped_no_pool": n_skipped_no_pool,
        "n_skipped_no_feat": n_skipped_no_feat,
    }
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"wrote eval results to {EVAL_OUT}")

    # 9. Print summary
    log("\n=== Per N_input (cross-length) ===")
    log("N_in  count  margin_mean  margin_pos%  d_self   rank1   rank3   lift1   n_tok  attrs_cov")
    for n in N_INPUT_LIST:
        s = per_n[n]
        if s["count"] == 0:
            log(f"  {n}    0     —            —          —        —       —       —       —      —")
            continue
        log(f"  {n}   {s['count']:>5}  {s['margin_mean']:>10.3f}   "
            f"{s['margin_pos_frac']*100:>6.1f}%   {s['d_self_mean']:>6.1f}   "
            f"{s['rank1_acc']:.3f}   {s['rank3_acc']:.3f}   {s['lift1']:.2f}×   "
            f"{s['n_tok_mean']:.1f}  {s['attrs_covered_mean']:.2f} "
            f"(eq_N={s['attrs_covered_eq_N']*100:.0f}%)")

    log("\n=== Per Length bucket (cross-N) ===")
    log("Length     count  margin_mean  margin_pos%  d_self   rank1   lift1   n_tok")
    for lb in LEN_LABELS:
        s = per_len[lb]
        if s["count"] == 0:
            log(f"  {lb:<8}    0     —            —          —        —       —       —")
            continue
        log(f"  {lb:<8} {s['count']:>5}  {s['margin_mean']:>10.3f}   "
            f"{s['margin_pos_frac']*100:>6.1f}%   {s['d_self_mean']:>6.1f}   "
            f"{s['rank1_acc']:.3f}   {s['lift1']:.2f}×   {s['n_tok_mean']:.1f}")

    log("\n=== Per (Length × N_input) cell (key evidence) ===")
    log("L\\N      3         4         5         6         7")
    for lb in LEN_LABELS:
        line = f"  {lb:<8}"
        for n in N_INPUT_LIST:
            s = per_cell.get(f"N={n}|L={lb}", None)
            if s is None or s["count"] == 0:
                line += f"   —/—/—/—/—"
            else:
                line += f"  c{s['count']:>4}/r1{s['rank1_acc']:.2f}/L{s['lift1']:.2f}×/d{s['d_self_mean']:>5.0f}"
        log(line)


def _len_label(n_tok: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_tok < hi:
            return f"{lo}-{hi if hi < 100 else '∞'}"
    return LEN_LABELS[-1]


# === numpy / imports for both modes ===
import numpy as np  # noqa: E402


if __name__ == "__main__":
    if MODE == "gen":
        main_gen()
    else:
        main_eval()