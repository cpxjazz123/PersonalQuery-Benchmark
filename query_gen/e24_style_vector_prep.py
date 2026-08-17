#!/usr/bin/env python3
"""E24 Phase A — 数据准备（一次性，预跑后 Phase B/C/D 复用）。

步骤:
  1. 加载现有 scale_hidden_L{16,24,26}.npz + scale_rewrites.jsonl + heldout_*
  2. L27 编码（dev pool 现只有 L16/24/26，缺 L27）
  3. X=3 短句池：重扫 39 scale 用户（MIN_SENT_WORDS=3），编码 L24/26/27
  4. 新 50 用户池：从 Baby_Products_2023.jsonl.gz 单次扫，挑不在原 100 池、
     ≥200 句 ≥5 词；构建 ~200 句 construction + 8 句 held-out + rewrites +
     L24/26/27 hidden
  5. 污染特征：join meta 文件 parent_asin → categories[-1] + Brand/Color/Material

输出（全部 append-only / 不覆盖现有）：
  result/e23_style_vector/scale_hidden_L27.npz                (新)
  result/e23_style_vector/scale_hidden_L{24,26,27}_short.npz  (新)
  result/e24_style_vector/test_pool_50u.jsonl                 (新)
  result/e24_style_vector/test_rewrites.jsonl                 (新)
  result/e24_style_vector/test_hidden_L{24,26,27}.npz         (新)
  result/e24_style_vector/contamination_features.json         (新)

执行: nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
        query_gen/e24_style_vector_prep.py \\
        > result/e24_style_vector/prep.log 2>&1 &
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
sys.path.insert(0, str(REPO_ROOT))

from query_gen_main import REVIEWS, MODEL_PATH, DTYPE, DEVICE  # noqa: E402
# extract_clause_features_single_query 不能在 vllm init 前 import——
# 二者共存触发 vllm V1 EngineCore fork 时 CUDA library 状态冲突
# （"Cannot re-initialize CUDA in forked subprocess"）。改为函数内 lazy import。
from llm_client import create_qwen_local_client  # noqa: E402


def _load_spacy_model():
    """Lazy import 避免与 vllm EngineCore fork 冲突。"""
    import sys
    if "syntactic_analysis" not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from extract_clause_features_single_query import load_spacy_model
    return load_spacy_model()

OUT_E23 = REPO_ROOT / "result" / "e23_style_vector"
OUT_E24 = Path("/home/wlia0047/hj82_scratch2/wenyu/e24_style_vector")
OUT_E24.mkdir(parents=True, exist_ok=True)

META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"

# dev pool caches (existing e23)
VEC_NPZ = OUT_E23 / "style_vectors_100u.npz"
HO_JSON = OUT_E23 / "style_vectors_heldout_validity.json"
HO_HIDDEN = OUT_E23 / "heldout_hidden.npz"
HO_REW = OUT_E23 / "heldout_rewrites.jsonl"
SC_REW = OUT_E23 / "scale_rewrites.jsonl"
SC_HIDDEN = {16: OUT_E23 / "scale_hidden_L16.npz",
             24: OUT_E23 / "scale_hidden_L24.npz",
             26: OUT_E23 / "scale_hidden_L26.npz"}

# new artifacts
SC_HIDDEN_L27 = OUT_E23 / "scale_hidden_L27.npz"
SC_HIDDEN_SHORT = {24: OUT_E23 / "scale_hidden_L24_short.npz",
                   26: OUT_E23 / "scale_hidden_L26_short.npz",
                   27: OUT_E23 / "scale_hidden_L27_short.npz"}

TEST_POOL = OUT_E24 / "test_pool_50u.jsonl"
TEST_REW = OUT_E24 / "test_rewrites.jsonl"
TEST_HIDDEN = {24: OUT_E24 / "test_hidden_L24.npz",
               26: OUT_E24 / "test_hidden_L26.npz",
               27: OUT_E24 / "test_hidden_L27.npz"}
CONTAM = OUT_E24 / "contamination_features.json"

# grid
LAYERS_DEV = [24, 26, 27]                # for dev grid
LAYERS_TEST = [24, 26, 27]               # for test grid (same)

# sentence pool (dev)
MIN_SENT_WORDS = 3                        # include 3-4 word sents for X=3 cell
MAX_SENT_WORDS = 60
DEV_K_MAX = 200                           # max sents/user to keep for dev

# new user pool
N_NEW_USERS = 50
MIN_CONSTR_PER_USER = 200                 # need ≥200 sents ≥MIN_SENT_WORDS=5
N_HO_PER_USER = 8
HO_RATIO = 0.04                           # 8 / (200+8) ≈ 4%
MIN_TOTAL_PER_USER = MIN_CONSTR_PER_USER + N_HO_PER_USER
MIN_SENT_WORDS_TEST = 5
REV_CAP_PER_USER = 500
N_SENTS_PER_USER = MIN_CONSTR_PER_USER
MAX_NEW_USERS_SCAN = 300                  # pick top N_NEW_USERS among top 300

# rewrite
REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a "
    "plain, neutral, matter-of-fact style. Keep the exact same meaning and "
    "all facts, but remove personal tone, slang, exclamations, and emotional "
    "words. Output ONLY the rewritten sentence, nothing else."
)
REWRITE_TEMPERATURE = 0.3
REWRITE_MAX_NEW = 128
REWRITE_BATCH = 32

# hidden state encoding
HIDDEN_BATCH = 32
HIDDEN_MAX_LEN = 160

# contamination
BRAND_MIN_COUNT = 3                       # min product-word mentions/user to be flagged

SEED = 7777


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached_rewrites(path: Path) -> dict[str, str]:
    """Load rewrites from jsonl cache (sentence -> neutral rewrite)."""
    out = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                out[d["sentence"]] = d["rewrite"]
            except Exception:
                continue
    return out


def load_hidden_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    c = np.load(path, allow_pickle=True)
    return {str(t): np.asarray(v) for t, v in zip(c["texts"], c["vecs"])}


def save_hidden_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    items = sorted(cache.items())
    np.savez_compressed(
        path,
        texts=np.asarray([k for k, _ in items]),
        vecs=np.stack([np.asarray(x, dtype=np.float16) for _, x in items]))


def append_rewrites(path: Path, items: list[tuple[str, str]]) -> None:
    """Append (sentence, rewrite) tuples; never overwrite existing entries."""
    if not items:
        return
    with open(path, "a", encoding="utf-8") as f:
        for s, t in items:
            f.write(json.dumps({"sentence": s, "rewrite": t}) + "\n")


# ============================================================================
# Step 1+2: L27 encoding of existing dev scale pool
# ============================================================================
def step_l27_dev(client) -> None:
    if SC_HIDDEN_L27.exists():
        c27 = load_hidden_cache(SC_HIDDEN_L27)
        log(f"step_l27_dev: existing cache has {len(c27)} texts")
    else:
        c27 = {}
    # union of all scale texts (from L24 cache)
    c24 = load_hidden_cache(SC_HIDDEN[24])
    all_texts = sorted(c24.keys())
    todo = [t for t in all_texts if t not in c27]
    log(f"step_l27_dev: {len(c27)} cached, {len(todo)} to encode at L27")
    if not todo:
        log("step_l27_dev: nothing to do")
        return
    # encode at L27 only
    for st in range(0, len(todo), HIDDEN_BATCH):
        chunk = todo[st:st + HIDDEN_BATCH]
        v = client.get_hidden_states(chunk, [27])[27]   # [B, H]
        for t, row in zip(chunk, v):
            c27[t] = row
        log(f"  L27 dev {min(st + HIDDEN_BATCH, len(todo))}/{len(todo)}")
    save_hidden_cache(SC_HIDDEN_L27, c27)
    log(f"step_l27_dev: DONE, saved {len(c27)} to {SC_HIDDEN_L27}")


# ============================================================================
# Step 3: X=3 short sentence pool for dev 39 users
# ============================================================================
def step_short_dev(client) -> None:
    # 39 scale users
    v = np.load(VEC_NPZ, allow_pickle=True)
    dev_users = [str(u) for u in v["users"]]
    construct_1000 = set(str(s) for s in v["sentences"])
    ho_valid = json.load(open(HO_JSON))
    ho_per_user = {u: ho_valid["heldout_sentences_per_user"].get(u, [])
                   for u in dev_users}
    log(f"step_short_dev: {len(dev_users)} dev users")

    # load existing caches (L24)
    c24 = load_hidden_cache(SC_HIDDEN[24])
    c26 = load_hidden_cache(SC_HIDDEN[26])
    caches = {24: c24, 26: c26, 27: load_hidden_cache(SC_HIDDEN_L27)}

    # load existing short caches if any
    short_caches = {l: load_hidden_cache(p) for l, p in SC_HIDDEN_SHORT.items()}

    # scan reviews for these users, collect 3-4 word sentences
    reviews_buf = {u: [] for u in dev_users}
    nlp = _load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in reviews_buf or len(reviews_buf[u]) >= REV_CAP_PER_USER:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews_buf[u].append(t)
    log(f"step_short_dev: reviews collected for {sum(1 for u in dev_users if reviews_buf[u])}/{len(dev_users)} users")

    # per user: extract 3-4 word sentences
    short_sents = {}
    for u in dev_users:
        sents = []
        for doc in nlp.pipe(reviews_buf[u], batch_size=64):
            for s in doc.sents:
                txt = s.text.strip()
                words = [t for t in s if not t.is_space]
                nw = len(words)
                if (txt and txt not in construct_1000
                        and 3 <= nw <= 4):
                    sents.append(txt)
        short_sents[u] = list(dict.fromkeys(sents))[:200]   # cap 200
    total = sum(len(v) for v in short_sents.values())
    log(f"step_short_dev: short sentences collected: {total} total")

    # encode
    all_texts = sorted({s for ss in short_sents.values() for s in ss})
    todo = [t for t in all_texts
            if (t not in short_caches[24]
                or t not in short_caches[26]
                or t not in short_caches[27])]
    log(f"step_short_dev: {len(todo)} new short sents to encode")
    for st in range(0, len(todo), HIDDEN_BATCH):
        chunk = todo[st:st + HIDDEN_BATCH]
        v_map = client.get_hidden_states(chunk, LAYERS_DEV)   # {L24:..., L26:..., L27:...}
        for t, row in zip(chunk, v_map[24]):
            short_caches[24][t] = row
        for t, row in zip(chunk, v_map[26]):
            short_caches[26][t] = row
        for t, row in zip(chunk, v_map[27]):
            short_caches[27][t] = row
        log(f"  short dev {min(st + HIDDEN_BATCH, len(todo))}/{len(todo)}")
    for l, p in SC_HIDDEN_SHORT.items():
        save_hidden_cache(p, short_caches[l])
        log(f"  saved {p} with {len(short_caches[l])} texts")

    # save short_sents as JSON for downstream
    short_json = OUT_E24 / "short_pool_dev.json"
    with open(short_json, "w") as f:
        json.dump({u: short_sents[u] for u in dev_users}, f)
    log(f"step_short_dev: DONE, wrote {short_json}")


# ============================================================================
# Step 4: New 50 users pool — per-user review list + spaCy batched sents
# ============================================================================
def step_new_users(_client_unused) -> None:
    """收集 50 个新用户（不在原 100 池）的 review 列表 + 切句结果。

    输出 test_pool_50u.jsonl，每行一个用户：
      - user_id
      - construction_sents (200 句, 5-60 词, 去重, RNG-shuffle 后取前 200)
      - heldout_sents (8 句, RNG-shuffle 后前 8)
      - sent_to_review: {sent: {rating, parent_asin}}  用于污染分析
      - reviews: [{text, rating, parent_asin}, ...]  完整 review 列表
    切句全部走 nlp.pipe(..., batch_size=64)，禁止单条 nlp(text)（Rule 5c）。
    rewrites 与 hidden states 由后续 step_rewrites_new / step_hidden_new 单独跑，
    这里只做数据准备。
    """
    if TEST_POOL.exists():
        with open(TEST_POOL) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        if existing:
            log(f"step_new_users: existing test pool has {len(existing)} users")
            return

    nlp = _load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    existing_users = set(str(u) for u in
                         np.load(VEC_NPZ, allow_pickle=True)["users"])
    log(f"step_new_users: existing {len(existing_users)} users to exclude")

    # ---- 单次扫 reviews：按 user 收 review 列表（cap REV_CAP_PER_USER） ----
    per_user_reviews: dict[str, list[tuple[str, int, str]]] = {}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if not u or u in existing_users:
                continue
            lst = per_user_reviews.setdefault(u, [])
            if len(lst) >= REV_CAP_PER_USER:
                continue
            t = (d.get("text") or "").strip()
            if not t:
                continue
            lst.append((t, d.get("rating"), d.get("parent_asin")))
    log(f"step_new_users: scanned {len(per_user_reviews)} candidate users")

    # ---- 取 review 数最多的前 MAX_NEW_USERS_SCAN 个用户做切句 ----
    users_to_parse = sorted(
        per_user_reviews.keys(),
        key=lambda u: -len(per_user_reviews[u]))[:MAX_NEW_USERS_SCAN]
    log(f"step_new_users: top {len(users_to_parse)} users queued for parsing")

    # ---- batched nlp.pipe 切句 ----
    user_sents: dict[str, list[tuple[str, int, int]]] = {}  # sent, review_idx, nw
    for i in range(0, len(users_to_parse), 64):
        chunk_users = users_to_parse[i:i + 64]
        flat_texts: list[str] = []
        flat_meta: list[tuple[str, int]] = []   # (user_id, review_idx)
        for u in chunk_users:
            for ri, (t, _r, _p) in enumerate(per_user_reviews[u]):
                flat_texts.append(t)
                flat_meta.append((u, ri))
        for doc, (u, ri) in zip(
                nlp.pipe(flat_texts, batch_size=64), flat_meta):
            for s in doc.sents:
                txt = s.text.strip()
                words = [tok for tok in s if not tok.is_space]
                nw = len(words)
                if txt and MIN_SENT_WORDS_TEST <= nw <= MAX_SENT_WORDS:
                    user_sents.setdefault(u, []).append((txt, ri, nw))
        if (i // 64) % 20 == 0:
            log(f"  parsed {i + len(chunk_users)}/"
                f"{len(users_to_parse)} users, "
                f"{sum(1 for v in user_sents.values() if len(v) >= MIN_TOTAL_PER_USER)}"
                " qualifying so far")

    # ---- 按句数排序，取前 N_NEW_USERS 个用户 ----
    candidates = [(u, len(s)) for u, s in user_sents.items()
                  if len(s) >= MIN_TOTAL_PER_USER]
    candidates.sort(key=lambda x: -x[1])
    picked = [u for u, _ in candidates[:N_NEW_USERS]]
    log(f"step_new_users: picked {len(picked)} new users "
        f"(top by sent count, min={MIN_TOTAL_PER_USER})")

    # ---- 切 held-out / construction；构造 review 关联 metadata ----
    rng = random.Random(SEED)
    out_lines: list[dict] = []
    for u in picked:
        sents_unique = list(dict.fromkeys([s for s, _, _ in user_sents[u]]))
        rng.shuffle(sents_unique)
        ho = sents_unique[:N_HO_PER_USER]
        constr = sents_unique[
            N_HO_PER_USER:N_HO_PER_USER + N_SENTS_PER_USER]
        # sent -> {rating, parent_asin}（去重：相同 sent 只取首个 review 的元数据）
        sent_to_meta: dict[str, dict] = {}
        for s, ri, _nw in user_sents[u]:
            if s in sent_to_meta:
                continue
            _t, rating, pasin = per_user_reviews[u][ri]
            sent_to_meta[s] = {"rating": rating, "parent_asin": pasin}
        all_reviews = [
            {"text": t, "rating": r, "parent_asin": p}
            for t, r, p in per_user_reviews[u]]
        out_lines.append({
            "user_id": u,
            "construction_sents": constr,
            "heldout_sents": ho,
            "sent_to_review": sent_to_meta,
            "reviews": all_reviews,
        })

    with open(TEST_POOL, "w") as f:
        for d in out_lines:
            f.write(json.dumps(d) + "\n")
    log(f"step_new_users: wrote {TEST_POOL} with {len(out_lines)} users "
        f"(constr={N_SENTS_PER_USER}, ho={N_HO_PER_USER})")


# ============================================================================
# Step 4b: Rewrite new user sentences via vllm (call_with_cache)
# ============================================================================
def step_rewrites_new(client) -> None:
    """对 50 个新用户的 construction + held-out 句跑中性改写，结果写入
    test_rewrites.jsonl（append-only）；已 cached 的不重跑。

    使用 transformers 直跑（与 _HiddenBackend 共享已加载的 7B 模型），
    避免 vllm + transformers 同时加载 OOM（GPU 39 GiB 不够两份 14 GiB 权重）。
    """
    if not TEST_POOL.exists():
        log("step_rewrites_new: TEST_POOL 缺失，跳过")
        return

    cached = cached_rewrites(TEST_REW)
    log(f"step_rewrites_new: {len(cached)} cached rewrites")

    with open(TEST_POOL) as f:
        users = [json.loads(l) for l in f if l.strip()]
    todo_sents: list[str] = []
    for u in users:
        for s in u["construction_sents"] + u["heldout_sents"]:
            if s not in cached:
                todo_sents.append(s)
    todo_sents = sorted(set(todo_sents))
    log(f"step_rewrites_new: {len(todo_sents)} sentences to rewrite")

    if not todo_sents:
        log("step_rewrites_new: nothing to do")
        return

    # 用 _HiddenBackend 已加载的 transformers 模型做 batch rewrite
    hb = client._hidden_backend
    tok = hb.tokenizer
    model = hb.model
    import torch
    new_pairs: list[tuple[str, str]] = []
    for st in range(0, len(todo_sents), REWRITE_BATCH):
        chunk = todo_sents[st:st + REWRITE_BATCH]
        encs = [tok.apply_chat_template(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": s}],
            tokenize=True, add_generation_prompt=True) for s in chunk]
        max_l = max(len(e) for e in encs)
        ids = torch.tensor([e + [tok.pad_token_id] * (max_l - len(e))
                            for e in encs], dtype=torch.long,
                           device=hb.device)
        attn = torch.tensor([[1] * len(e) + [0] * (max_l - len(e))
                             for e in encs], dtype=torch.long,
                            device=hb.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=REWRITE_MAX_NEW,
                do_sample=True, temperature=REWRITE_TEMPERATURE, top_p=0.95,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True)
        rewrites: list[str] = []
        for b, e in enumerate(encs):
            row = gen[b, len(e):]
            if tok.eos_token_id in row:
                row = row[:row.tolist().index(tok.eos_token_id)]
            t = tok.decode(row, skip_special_tokens=True).strip()
            t = t.split("")[0].strip()
            rewrites.append(t)
        for s, r in zip(chunk, rewrites):
            new_pairs.append((s, r))
            cached[s] = r
        append_rewrites(TEST_REW, list(zip(chunk, rewrites)))
        log(f"  rewrite {min(st + REWRITE_BATCH, len(todo_sents))}/"
            f"{len(todo_sents)}")
    bad = sum(1 for r in cached.values() if len(r.split()) < 2)
    log(f"step_rewrites_new: DONE, total={len(cached)} cached, "
        f"empty/short={bad}")


# ============================================================================
# Step 4c: Hidden state encoding for new users (L24/26/27) via transformers
# ============================================================================
def step_hidden_new(client) -> None:
    """对 50 个新用户的 construction 句 + 改写 跑 L24/L26/L27 hidden state
    提取，结果写入 test_hidden_L{24,26,27}.npz（append-only 缓存语义：
    已有 texts 不重算）。"""
    if not TEST_POOL.exists():
        log("step_hidden_new: TEST_POOL 缺失，跳过")
        return

    cached = {l: load_hidden_cache(p) for l, p in TEST_HIDDEN.items()}
    log(f"step_hidden_new: caches L24={len(cached[24])} "
        f"L26={len(cached[26])} L27={len(cached[27])}")

    with open(TEST_POOL) as f:
        users = [json.loads(l) for l in f if l.strip()]
    rew = cached_rewrites(TEST_REW)

    all_texts: set[str] = set()
    for u in users:
        all_texts.update(u["construction_sents"])
        all_texts.update(u["heldout_sents"])
        for s in u["construction_sents"] + u["heldout_sents"]:
            r = rew.get(s)
            if r:
                all_texts.add(r)
    all_texts_sorted = sorted(all_texts)
    todo = [t for t in all_texts_sorted
            if (t not in cached[24]
                or t not in cached[26]
                or t not in cached[27])]
    log(f"step_hidden_new: {len(all_texts_sorted)} texts total, "
        f"{len(todo)} to encode at L24/26/27")

    for st in range(0, len(todo), HIDDEN_BATCH):
        chunk = todo[st:st + HIDDEN_BATCH]
        v_map = client.get_hidden_states(chunk, LAYERS_TEST)
        for t, row in zip(chunk, v_map[24]):
            cached[24][t] = row
        for t, row in zip(chunk, v_map[26]):
            cached[26][t] = row
        for t, row in zip(chunk, v_map[27]):
            cached[27][t] = row
        log(f"  hidden {min(st + HIDDEN_BATCH, len(todo))}/{len(todo)}")
    for l, p in TEST_HIDDEN.items():
        save_hidden_cache(p, cached[l])
        log(f"  saved {p} with {len(cached[l])} texts")


# ============================================================================
# Step 5: Contamination features (meta join)
# ============================================================================
def step_contamination() -> None:
    if CONTAM.exists():
        log(f"step_contamination: {CONTAM} already exists, skipping")
        return
    # scan meta file: parent_asin -> {category, brand, color, material}
    meta = {}
    log("step_contamination: scanning meta file...")
    with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pasin = d.get("parent_asin")
            if not pasin:
                continue
            cats = d.get("categories") or []
            cat_leaf = cats[-1] if cats else None
            details = d.get("details") or {}
            meta[pasin] = {
                "category": cat_leaf,
                "brand": details.get("Brand"),
                "color": details.get("Color"),
                "material": details.get("Material"),
            }
    log(f"step_contamination: meta entries = {len(meta)}")

    # per-sentence contamination: map (user, sent) -> {category, rating, brand}
    # TEST_POOL v2 schema: reviews 是 list of {text, rating, parent_asin}
    out = {"meta_count": len(meta), "per_user": {}}
    with open(TEST_POOL) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            u = d["user_id"]
            sent_features: dict[str, list[dict]] = {}
            for review in d["reviews"]:
                txt = review["text"]
                rating = review.get("rating")
                pasin = review.get("parent_asin")
                m = meta.get(pasin, {})
                sent_features.setdefault(txt, []).append({
                    "rating": rating, "parent_asin": pasin,
                    "category": m.get("category"),
                    "brand": m.get("brand"),
                    "color": m.get("color"),
                    "material": m.get("material"),
                })
            out["per_user"][u] = sent_features
    with open(CONTAM, "w") as f:
        json.dump(out, f)
    log(f"step_contamination: wrote {CONTAM} "
        f"({sum(len(v) for v in out['per_user'].values())} sent features)")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    log("phase A start; using transformers for hidden_states + rewrites "
        "(单一模型避免 vllm+transformers OOM，GPU 39 GiB 不够两份 7B 权重)")

    # with_vllm=False: 同一 transformers 模型既做 hidden state 又做 rewrite，
    # 避免 vllm EngineCore (~35 GiB) + transformers (~14 GiB) OOM
    client = create_qwen_local_client(with_vllm=False)

    step_l27_dev(client)
    step_short_dev(client)
    step_new_users(client)
    step_rewrites_new(client)
    step_hidden_new(client)
    step_contamination()

    log(f"DONE total {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()