#!/usr/bin/env python3
"""Phase 15.4: Per-user exemplar conditioned generation.

For each user, select 3-5 most distinctive sentences, strip product-specific terms,
use as few-shot exemplars in generation. Generate K=8 queries per user.

Distinctiveness selection:
- Score each sentence by average distance from other users (in PCA-residual space)
- Pick top-3 most distinctive (different from average population)

Product-stripping:
- Remove brand names, model numbers, sizes from sentences
"""
from __future__ import annotations
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"
IN_GEN = OUT_DIR / "phase14_q10l_generated.jsonl"
IN_USER_REVIEWS = OUT_DIR / "phase14_q10l_user_reviews.pkl"
IN_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"

OUT_GEN = OUT_DIR / "phase15_4_generated.jsonl"

N_EXEMPLARS = 3
K = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 64
MIN_SENT_LEN = 5
MAX_SENT_LEN = 30
MAX_EXEMPLAR_WORDS = 25  # truncate exemplars to this many words

# Product-stripping patterns (English baby products)
PRODUCT_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+[A-Z0-9]{3,}\b',  # Brand+Model
    r'\b\d+(?:\.\d+)?\s*(?:oz|lb|kg|ml|g|kg|lbs|inch|inches|cm|mm)\b',  # sizes
    r'\$\d+(?:\.\d+)?',  # prices
    r'\b(?:pack of|count of)\s*\d+\b',
    r'#\d+',
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip_product(text):
    out = text
    for pat in PRODUCT_PATTERNS:
        out = re.sub(pat, '', out, flags=re.IGNORECASE)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def main() -> None:
    log("=" * 70)
    log(f"Phase 15.4: Per-user exemplar generation (n_exemplars={N_EXEMPLARS}, K={K})")
    log("=" * 70)

    log("[1] Loading pairs + reviews ...")
    pairs = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    with IN_USER_REVIEWS.open("rb") as f:
        user_reviews = pickle.load(f)

    log(f"  pairs: {len(pairs)}, user reviews: {len(user_reviews):,}")

    log("[2] Loading user hiddens for distinctiveness scoring ...")
    npz_u = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids = list(npz_u["user_ids"])
    user_hiddens = npz_u["hiddens"].astype(np.float32)

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, [8, 14, 18, 22, 26], :].mean(axis=0)

    LAYER_FOCUS = 26
    layer_idx = [8, 14, 18, 22, 26].index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    user_mean_resid = user_residuals.mean(axis=1)  # (n_users, 3584)
    log(f"  user_mean_resid: {user_mean_resid.shape}")

    # Project to 100-d PCA for distinctiveness scoring (lightweight)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, svd_solver='randomized', random_state=42)
    pca.fit(user_mean_resid)
    user_pca = (user_mean_resid - pca.mean_) @ pca.components_.T
    log(f"  user_pca: {user_pca.shape}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    log("[3] Selecting 3-5 most distinctive sentences per user ...")
    user_exemplars = {}

    for pair in pairs:
        uid = pair["user_id"]
        reviews = user_reviews.get(uid, [])
        if not reviews:
            continue

        # Collect all candidate sentences
        cands = []
        for r in reviews:
            text = r["text"]
            for sent in split_sentences(text):
                words = sent.split()
                if MIN_SENT_LEN <= len(words) <= MAX_SENT_LEN:
                    cands.append(sent)

        if len(cands) < N_EXEMPLARS:
            user_exemplars[uid] = cands[:N_EXEMPLARS]
            continue

        # Score distinctiveness: how far each sentence's style is from population mean
        # For each sentence, encode with PCA-style simple average of length + opener heuristics
        # Quick proxy: use sentence length + punctuation density
        sent_features = []
        for sent in cands:
            words = sent.split()
            avg_word_len = sum(len(w) for w in words) / max(1, len(words))
            n_commas = sent.count(',')
            n_ands = sent.lower().count(' and ')
            n_underscore = sum(1 for w in words if w.startswith('_'))
            sent_features.append([len(words), avg_word_len, n_commas, n_ands, n_underscore])
        sent_features = np.array(sent_features)
        # Distance from population mean
        pop_mean = sent_features.mean(axis=0)
        pop_std = sent_features.std(axis=0) + 1e-9
        z_scores = np.abs((sent_features - pop_mean) / pop_std).sum(axis=1)

        # Top-N most distinctive (z-score highest = most different)
        top_idx = np.argsort(-z_scores)[:N_EXEMPLARS]
        exemplars = [cands[i] for i in top_idx]
        user_exemplars[uid] = exemplars

    n_with = sum(1 for uid, v in user_exemplars.items() if len(v) == N_EXEMPLARS)
    log(f"  users with {N_EXEMPLARS} exemplars: {n_with}/{len(pairs)}")

    log("[4] Build per-user few-shot prompts ...")
    PROMPT_TEMPLATE = (
        "You are helping a user write a shopping search query in their personal style. "
        "Here are some examples of how this user writes:\n\n"
        "{exemplars}\n\n"
        "Given the product attributes below, write a natural search query that matches "
        "this user's style. Output ONLY the query.\n\n"
        "Attributes: {attrs}\n\n"
        "Search query:"
    )

    prompts_with_meta = []
    for pair in pairs:
        uid = pair["user_id"]
        attrs_str = ", ".join(f"{k}: {v}" for k, v in pair["attrs"].items() if v)
        exem = user_exemplars.get(uid, [])
        if not exem:
            # Fallback to base prompt
            exem_str = "(no examples)"
        else:
            stripped = [strip_product(e) for e in exem]
            # Truncate
            stripped = [" ".join(s.split()[:MAX_EXEMPLAR_WORDS]) for s in stripped]
            exem_str = "\n".join(f"- {e}" for e in stripped)

        prompt = PROMPT_TEMPLATE.format(exemplars=exem_str, attrs=attrs_str)
        for k in range(K):
            suffix = f" (variant {k + 1}/{K})"
            prompts_with_meta.append({
                "user_id": uid,
                "asin": pair["asin"],
                "attrs": pair["attrs"],
                "attrs_str": attrs_str,
                "k": k,
                "prompt": prompt + suffix,
                "exemplars": exem,
            })

    log(f"  total prompts: {len(prompts_with_meta)}")

    log("[5] Apply chat template + vLLM generation ...")
    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=True)

    full_prompts = [
        client._backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_with_meta
    ]

    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    t0 = time.time()
    BATCH = 256
    outputs = []
    for i in range(0, len(full_prompts), BATCH):
        batch_prompts = full_prompts[i: i + BATCH]
        out = client._backend.model.generate(batch_prompts, sampling, use_tqdm=False)
        outputs.extend(out)
        if (i // BATCH) % 5 == 0:
            elapsed = time.time() - t0
            done = min(i + BATCH, len(full_prompts))
            rate = done / elapsed
            eta = (len(full_prompts) - done) / rate
            log(f"  gen {done}/{len(full_prompts)} ({rate:.1f} q/s, ETA {eta:.0f}s)")
    log(f"  done in {time.time() - t0:.1f}s")

    log("[6] Write jsonl ...")
    def append_missing_attrs(q, attrs_str):
        q_low = q.lower()
        missing = []
        for kv in attrs_str.split(","):
            kv = kv.strip()
            if ":" not in kv:
                continue
            _, v = kv.split(":", 1)
            v = v.strip()
            if v and v.lower() not in q_low:
                missing.append(v)
        if missing:
            q = q.rstrip(".") + ", " + ", ".join(missing) + "."
        return q

    with OUT_GEN.open("w") as f:
        for meta, out in zip(prompts_with_meta, outputs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            q_post = append_missing_attrs(text, meta["attrs_str"])
            rec = {
                "user_id": meta["user_id"],
                "asin": meta["asin"],
                "attrs": meta["attrs"],
                "attrs_str": meta["attrs_str"],
                "condition": "Q_PerUserExemplar",
                "k": meta["k"],
                "q_styled": text,
                "q_final_post": q_post,
                "exemplars": meta["exemplars"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GEN}")

    log("=" * 70)
    log("PHASE 15.4 GENERATION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()