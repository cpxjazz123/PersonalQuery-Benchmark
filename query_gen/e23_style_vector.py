#!/usr/bin/env python3
"""E23 P0 — StyleVector extraction for 100 random users.

Paper protocol (StyleVector): for each user sentence, generate a
STYLE-AGNOSTIC (neutral) rewrite with a general LLM; run both through the
same LLM; the difference of their hidden activations is the style direction.

Per user:
    Δ_i = meanpool_hidden(user_sent_i, layer l)
          − meanpool_hidden(neutral_rewrite_i, layer l)
    s_u = aggregate over i:
          - mean_diff : mean(Δ_i)                      (primary)
          - logreg    : normalized linear weight w of user-vs-neutral
                        logistic classifier
          - pca       : first principal component of the Δ matrix

Outputs (result/e23_style_vector/):
    rewrites.jsonl            — sentence -> neutral rewrite (resume-able)
    hidden_cache.npz          — sentence text -> mean-pooled layer vectors
    style_vectors_100u.npz    — per user / layer / method style vectors
    style_vectors_100u.json   — manifest + sanity metrics

All config hardcoded (no CLI args). Run: python query_gen/e23_style_vector.py
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import DTYPE, DEVICE, MODEL_PATH, REVIEWS  # noqa: E402
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

OUT_DIR = REPO_ROOT / "result" / "e23_style_vector"
REWRITES_JSONL = OUT_DIR / "rewrites.jsonl"
HIDDEN_CACHE = OUT_DIR / "hidden_cache.npz"
VEC_NPZ = OUT_DIR / "style_vectors_100u.npz"
VEC_JSON = OUT_DIR / "style_vectors_100u.json"

# ------------------- hardcoded config -------------------
SEED = 7777
N_USERS = 100
N_SENTS_PER_USER = 10
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60
REV_CAP_PER_USER = 100          # max review texts kept per candidate user
CAND_BUFFER = 240               # sample more candidates than needed
REWRITE_BATCH = 16
MAX_NEW = 128
REWRITE_TEMPERATURE = 0.3
HIDDEN_BATCH = 32
LAYERS = [8, 12, 16, 20, 24, 26, 27]   # 0-indexed, Qwen2-7B has 28 layers
METHODS = ["mean_diff", "logreg", "pca"]
NORM_METRIC = "l2"

REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a "
    "plain, neutral, matter-of-fact style. Keep the exact same meaning and "
    "all facts, but remove personal tone, slang, exclamations, and emotional "
    "words. Output ONLY the rewritten sentence, nothing else."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_review_counts() -> dict[str, int]:
    """Single gz pass: user_id -> review count."""
    counts: dict[str, int] = {}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u:
                counts[u] = counts.get(u, 0) + 1
    return counts


def load_reviews_for(users: set[str]) -> dict[str, list[str]]:
    """Second gz pass: collect up to REV_CAP_PER_USER review texts per user."""
    out: dict[str, list[str]] = {u: [] for u in users}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in users or len(out[u]) >= REV_CAP_PER_USER:
                continue
            t = (d.get("text") or "").strip()
            if t:
                out[u].append(t)
    return {u: ts for u, ts in out.items() if ts}


def split_sentences(reviews_by_user: dict[str, list[str]], nlp) -> dict[str, list[str]]:
    """Batch spaCy sentence split; keep 5..60-word sentences."""
    sents_by_user: dict[str, list[str]] = {}
    for u, texts in reviews_by_user.items():
        sents: list[str] = []
        for doc in nlp.pipe(texts, batch_size=64):
            for s in doc.sents:
                words = [t for t in s if not t.is_space]
                n = len(words)
                if MIN_SENT_WORDS <= n <= MAX_SENT_WORDS:
                    sents.append(s.text.strip())
        sents_by_user[u] = sents
    return sents_by_user


# ------------------- rewrite (neutral, style-agnostic) -------------------


def load_rewrites() -> dict[str, str]:
    if not REWRITES_JSONL.exists():
        return {}
    out: dict[str, str] = {}
    with open(REWRITES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            out[d["sentence"]] = d["rewrite"]
    return out


def append_rewrites(rows: list[dict]) -> None:
    REWRITES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(REWRITES_JSONL, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@torch.no_grad()
def batch_generate(model, tok, prompts: list[str], max_new: int,
                   temperature: float, batch: int) -> list[str]:
    """Right-padded batched generation (rule: batch decode)."""
    out: list[str] = []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        encs = [tok.apply_chat_template(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": s}],
            tokenize=True, add_generation_prompt=True) for s in chunk]
        max_l = max(len(e) for e in encs)
        ids = torch.tensor([e + [tok.pad_token_id] * (max_l - len(e))
                            for e in encs], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[1] * len(e) + [0] * (max_l - len(e))
                             for e in encs], dtype=torch.long, device=DEVICE)
        gen = model.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=max_new,
            do_sample=temperature > 0, temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            use_cache=True)
        new_ids = gen[:, max_l:]
        for row in new_ids:
            text = tok.decode(row, skip_special_tokens=True).strip()
            text = text.split("<|im_end|>")[0].strip()
            out.append(text)
        log(f"  rewrite {min(start + batch, len(prompts))}/{len(prompts)}")
    return out


def rewrite_all(model, tok, sentences: list[str]) -> dict[str, str]:
    cached = load_rewrites()
    todo = [s for s in sentences if s not in cached]
    log(f"rewrites: {len(cached)} cached, {len(todo)} to generate")
    if todo:
        texts = batch_generate(model, tok, todo, MAX_NEW,
                               REWRITE_TEMPERATURE, REWRITE_BATCH)
        rows, new_cache = [], {}
        for s, t in zip(todo, texts):
            rows.append({"sentence": s, "rewrite": t})
            new_cache[s] = t
        append_rewrites(rows)
        cached.update(new_cache)
    bad = sum(1 for t in cached.values() if len(t.split()) < 2)
    log(f"rewrite done: total={len(cached)}, empty/short={bad}")
    return cached


# ------------------- hidden activations -------------------


def load_hidden_cache() -> tuple[dict[str, np.ndarray], int]:
    if not HIDDEN_CACHE.exists():
        return {}, 0
    c = np.load(HIDDEN_CACHE, allow_pickle=True)
    texts = [str(t) for t in c["texts"]]
    vecs = c["vecs"]
    return {t: vecs[i] for i, t in enumerate(texts)}, len(texts)


def save_hidden_cache(cache: dict[str, np.ndarray]) -> None:
    items = sorted(cache.items())
    np.savez_compressed(
        HIDDEN_CACHE,
        texts=np.asarray([k for k, _ in items]),
        vecs=np.stack([v for _, v in items]))


@torch.no_grad()
def hidden_for_texts(model, tok, texts: list[str],
                     layers: list[int]) -> np.ndarray:
    """Mean-pooled hidden states at `layers` for a batch of texts.

    Returns [len(texts), len(layers), H] fp16. Mean-pool over non-pad tokens.
    """
    encs = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    max_l = max(len(e) for e in encs)
    ids = torch.tensor([e + [tok.pad_token_id] * (max_l - len(e))
                        for e in encs], dtype=torch.long, device=DEVICE)
    attn = torch.tensor([[1] * len(e) + [0] * (max_l - len(e))
                         for e in encs], dtype=torch.long, device=DEVICE)
    out = model(input_ids=ids, attention_mask=attn,
                use_cache=False, output_hidden_states=True)
    lidx = list(layers)
    hs = torch.stack([out.hidden_states[i] for i in lidx])  # [L,B,T,H]
    mask_f = attn.unsqueeze(0).unsqueeze(-1).to(DTYPE)      # [1,B,T,1]
    pooled = (hs * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1)
    return pooled.permute(1, 0, 2).float().cpu().numpy()    # [B,L,H]


def collect_hidden(model, tok, texts: list[str], layers: list[int]) -> dict[str, np.ndarray]:
    cache, n_cached = load_hidden_cache()
    todo = [t for t in texts if t not in cache]
    log(f"hidden: {n_cached} cached, {len(todo)} to compute")
    for start in range(0, len(todo), HIDDEN_BATCH):
        chunk = todo[start:start + HIDDEN_BATCH]
        vecs = hidden_for_texts(model, tok, chunk, layers)
        for t, v in zip(chunk, vecs):
            cache[t] = v
        log(f"  hidden {min(start + HIDDEN_BATCH, len(todo))}/{len(todo)}")
    save_hidden_cache(cache)
    return cache


# ------------------- style vectors -------------------


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def style_vectors_for_user(user_acts: np.ndarray, neutral_acts: np.ndarray,
                           n_layers: int) -> dict[str, np.ndarray]:
    """user_acts/neutral_acts: [N, L, H]. Returns method -> [L, H]."""
    D = user_acts - neutral_acts                       # [N, L, H]
    mean_diff = D.mean(axis=0)                          # [L, H]
    out = {"mean_diff": mean_diff}
    # logistic regression per layer (user=+1, neutral=-1) -> w direction
    W = np.zeros((n_layers, user_acts.shape[-1]), dtype=np.float64)
    for li in range(n_layers):
        X = np.concatenate([user_acts[:, li], neutral_acts[:, li]], axis=0)
        y = np.concatenate([np.ones(len(user_acts)),
                            np.zeros(len(neutral_acts))], axis=0)
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=500)
        clf.fit(X, y)
        W[li] = clf.coef_[0]
    out["logreg"] = W
    # PCA first component of the Δ matrix per layer
    P = np.zeros_like(mean_diff)
    for li in range(n_layers):
        pca = PCA(n_components=1)
        pca.fit(D[:, li])
        P[li] = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    out["pca"] = P
    return out


def sanity_metrics(user_acts_list: list[np.ndarray],
                   neutral_acts_list: list[np.ndarray],
                   sv: dict[str, np.ndarray], layers: list[int]) -> dict:
    """user_acts_list: list of per-user [n_i, L, H]; sv: method -> [U, L, H]."""
    U = len(user_acts_list)
    L = len(layers)
    H = user_acts_list[0].shape[-1]
    met: dict = {}
    for li, l in enumerate(layers):
        cu = np.concatenate([a[:, li] for a in user_acts_list])
        cn = np.concatenate([a[:, li] for a in neutral_acts_list])
        cu = cu / np.maximum(np.linalg.norm(cu, axis=1, keepdims=True), 1e-9)
        cn = cn / np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-9)
        pair_cos = (cu * cn).sum(axis=1)
        dn = np.concatenate([a[:, li] - b[:, li]
                             for a, b in zip(user_acts_list,
                                             neutral_acts_list)])
        # pairwise user style-vector distinctness (mean_diff)
        svl = sv["mean_diff"][:, li]
        svl = svl / np.maximum(np.linalg.norm(svl, axis=1, keepdims=True), 1e-9)
        S = svl @ svl.T
        off = S[~np.eye(U, dtype=bool)]
        met[f"layer{l}"] = {
            "user_neutral_cosine_mean": round(float(pair_cos.mean()), 4),
            "user_neutral_cosine_std": round(float(pair_cos.std()), 4),
            "delta_norm_mean": round(float(np.linalg.norm(
                dn, axis=-1).mean()), 3),
            "stylevec_norm_mean": round(float(np.linalg.norm(
                sv["mean_diff"][:, li], axis=-1).mean()), 3),
            "stylevec_pairwise_cosine_mean": round(float(off.mean()), 4),
            "stylevec_pair_cos_lt_0.9": round(
                float((off < 0.9).mean()), 4),
            "stylevec_pair_cos_lt_0.5": round(
                float((off < 0.5).mean()), 4),
        }
    return met


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"model: {MODEL_PATH}")
    log("pass1: counting reviews per user...")
    counts = load_review_counts()
    cands = sorted(u for u, c in counts.items() if c >= 30)
    log(f"users with >=30 reviews: {len(cands)}")
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands = cands[:CAND_BUFFER]

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    log("pass2: loading review texts for candidates...")
    reviews = load_reviews_for(set(cands))
    log(f"candidates with reviews: {len(reviews)}")
    sents_by_user = split_sentences(reviews, nlp)
    del reviews

    eligible = sorted(u for u in sents_by_user
                      if len(sents_by_user[u]) >= N_SENTS_PER_USER)
    log(f"eligible users (>= {N_SENTS_PER_USER} sentences): {len(eligible)}")
    rng.shuffle(eligible)
    users = eligible[:N_USERS]
    if len(users) < N_USERS:
        raise RuntimeError(f"only {len(users)} eligible users "
                           f"(need {N_USERS})")
    user_sents = {u: sents_by_user[u][:N_SENTS_PER_USER] for u in users}
    all_sents = [s for ss in user_sents.values() for s in ss]
    n_per_user = [len(user_sents[u]) for u in users]
    log(f"users={len(users)}, sentences total={len(all_sents)}, "
        f"per-user counts={n_per_user}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers
    log(f"layers total={n_layers}, extracting {LAYERS}")

    rewrites = rewrite_all(model, tok, all_sents)

    texts_user = all_sents
    texts_neutral = [rewrites[s] for s in all_sents]
    hidden = collect_hidden(model, tok,
                            sorted(set(texts_user + texts_neutral)),
                            LAYERS)

    # per-user arrays: list of [n_i, L, H] (variable n_i)
    user_acts = np.stack([hidden[s] for s in texts_user])
    neutral_acts = np.stack([hidden[s] for s in texts_neutral])
    U = len(users)
    Hdim = user_acts.shape[-1]
    all_ua = [user_acts[o:o + n] for o, n in
              zip(np.cumsum([0] + n_per_user[:-1]), n_per_user)]
    all_na = [neutral_acts[o:o + n] for o, n in
              zip(np.cumsum([0] + n_per_user[:-1]), n_per_user)]

    log("computing style vectors (mean_diff / logreg / pca)...")
    sv: dict[str, np.ndarray] = {m: np.zeros((U, len(LAYERS), Hdim),
                                             dtype=np.float32)
                                 for m in METHODS}
    for ui, u in enumerate(users):
        per = style_vectors_for_user(all_ua[ui], all_na[ui], len(LAYERS))
        for m in METHODS:
            sv[m][ui] = per[m].astype(np.float32)
        if ui % 20 == 0 or ui == U - 1:
            log(f"  user {ui + 1}/{U}")
    sv_norm = {m: np.stack([np.stack([normalize(v) for v in per])
                            for per in sv[m]]) for m in METHODS}
    log("computing sanity metrics...")
    metrics = sanity_metrics(all_ua, all_na, sv, LAYERS)

    np.savez_compressed(
        VEC_NPZ,
        users=np.asarray(users),
        sentences=np.asarray(texts_user),
        layers=np.asarray(LAYERS),
        methods=np.asarray(METHODS),
        n_per_user=np.asarray(n_per_user),
        mean_diff=sv["mean_diff"],
        logreg=sv["logreg"],
        pca=sv["pca"],
        mean_diff_norm=sv_norm["mean_diff"],
        logreg_norm=sv_norm["logreg"],
        pca_norm=sv_norm["pca"])

    examples = [{"user": s, "neutral": rewrites[s]}
                for s in all_sents[:10]]
    result = {
        "version": "e23_style_vector_v1",
        "seed": SEED,
        "model": Path(MODEL_PATH).name,
        "pooling": "mean-pool over sentence tokens",
        "layers": LAYERS,
        "methods": METHODS,
        "n_users": U,
        "n_sents_per_user_max": N_SENTS_PER_USER,
        "n_sents_per_user": n_per_user,
        "n_sentences_total": len(all_sents),
        "sentence_words": [MIN_SENT_WORDS, MAX_SENT_WORDS],
        "rewrite_prompt": REWRITE_SYSTEM,
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "rewrite_max_new": MAX_NEW,
        "layer_metrics": metrics,
        "rewrite_examples": examples,
        "vector_file": str(VEC_NPZ),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(VEC_JSON, "w") as f:
        json.dump(result, f, indent=1)
    log(f"DONE — style vectors written to {VEC_NPZ}")
    log(f"  layer metrics (layer27): "
        f"{json.dumps(metrics.get('layer27', {}))}")
    log(f"runtime {result['runtime_sec']}s")


if __name__ == "__main__":
    main()
