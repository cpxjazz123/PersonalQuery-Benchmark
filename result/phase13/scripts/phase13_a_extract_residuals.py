#!/usr/bin/env python3
"""Phase 13.A: Extract Qwen-space per-sentence residuals r_i^ℓ = h_ℓ(s_i) - h_ℓ(n_i).

Per E23 StyleVector protocol but on the full 876-user phase10 train set (vs e23's 100u).
For each user u and each of 10 sampled review sentences s_i:
  1. LLM-rewrite s_i to neutral n_i  (style-agnostic, "Write neutral shopping query" style)
  2. Compute h_ℓ(s_i) and h_ℓ(n_i) at every Qwen2-7B layer ℓ (0..27)
  3. Residual: r_i^ℓ = h_ℓ(s_i) - h_ℓ(n_i)
  4. Aggregate: per-user mean residual μ_u^ℓ = mean_i r_i^ℓ

Output (for Phase 13.B Gaussian fit + Phase 13.C injection):
  - phase13_a_per_sentence_residuals_qwen.npz
      user_ids    : [876]   user order
      sent_uids   : [n_total=8760] per-sentence user_id
      sent_idx    : [n_total] per-sentence within-user index (0..9)
      residuals   : [n_total, 28, 3584] fp16  (subtract neutral from user)
      mean_resid  : [876, 28, 3584] fp32  (mean over 10 per user; null users 0)
      user_sents  : [n_total]  (the original sentence text; for diagnostic)
  - phase13_a_meta.json
  - phase13_a_rewrites_cache.jsonl   (sentence -> neutral rewrite, resume-able)

Reused code patterns:
  - e23_style_vector.py::batch_generate / rewrite_all / hidden_for_texts / save_hidden_cache
  - llm_client.py::QwenLocalClient.get_hidden_states (transformers backend)
  - vLLM batched generate (phase11_g_diverse_z_v2.py::batch_generate) for rewrites
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
RAW_SENTENCES_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"
PHASE10_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_RESIDUALS = OUT_DIR / "phase13_a_per_sentence_residuals_qwen.npz"
OUT_META = OUT_DIR / "phase13_a_meta.json"
OUT_REWRITES = OUT_DIR / "phase13_a_rewrites_cache.jsonl"
OUT_HIDDEN_USER = OUT_DIR / "phase13_a_user_hiddens.npz"
OUT_HIDDEN_NEUTRAL = OUT_DIR / "phase13_a_neutral_hiddens.npz"

# === Hardcoded config ===
N_USERS = 876
N_SENTS_PER_USER = 10
QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
N_LAYERS = 28
HIDDEN_DIM = 3584
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60
REWRITE_BATCH = 64            # vLLM batched
REWRITE_MAX_NEW = 80
REWRITE_TEMP = 0.3
HIDDEN_BATCH = 32             # transformers get_hidden_states batch
HIDDEN_MAX_LENGTH = 160
RANDOM_SEED = 42

REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a "
    "plain, neutral, matter-of-fact style. Keep the exact same meaning and "
    "all facts, but remove personal tone, slang, exclamations, and emotional "
    "words. Output ONLY the rewritten sentence, nothing else."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rewrites() -> dict[str, str]:
    if not OUT_REWRITES.exists():
        return {}
    out: dict[str, str] = {}
    with open(OUT_REWRITES, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("sentence") and d.get("rewrite"):
                    out[d["sentence"]] = d["rewrite"]
            except Exception:
                continue
    return out


def append_rewrites(rows: list[dict]) -> None:
    with open(OUT_REWRITES, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_hidden_cache(path: str) -> dict[str, np.ndarray]:
    if not Path(path).exists():
        return {}
    c = np.load(path, allow_pickle=True)
    texts = [str(t) for t in c["texts"]]
    vecs = c["vecs"]
    return {t: vecs[i] for i, t in enumerate(texts)}


def save_hidden_cache(path: str, cache: dict[str, np.ndarray]) -> None:
    items = sorted(cache.items())
    np.savez_compressed(
        path,
        texts=np.asarray([k for k, _ in items]),
        vecs=np.stack([v for _, v in items]),
    )


def batch_generate_neutral(client, prompts: list[str], max_new: int, batch: int) -> list[str]:
    """Batched vLLM generation (Phase 11.G v2 pattern)."""
    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=max_new, temperature=REWRITE_TEMP,
        top_p=0.95 if REWRITE_TEMP > 0 else 1.0,
    )
    full_prompts = [
        client._backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": s}],
            tokenize=False, add_generation_prompt=True,
        ) for s in prompts
    ]
    outputs = client._backend.model.generate(full_prompts, sampling)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]


def main():
    log("=" * 70)
    log("Phase 13.A: Extract Qwen-space Style Residuals (876u × 10 sents × 28 layers)")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # === [1] Load 876 user_ids ===
    log("[1] Loading phase10 users ...")
    user_ids: list[str] = []
    with PHASE10_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user_ids.append(obj["user_id"])
    user_ids = sorted(set(user_ids))[:N_USERS]
    log(f"  users: {len(user_ids)}")

    # === [2] Load sentences, take 10 per user (skip holdout) ===
    log("[2] Loading sentences, taking 10 per user (skip holdout) ...")
    user_to_sents: dict[str, list[str]] = {u: [] for u in user_ids}
    with RAW_SENTENCES_FILE.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            uid = d.get("user_id")
            if uid not in user_to_sents:
                continue
            if d.get("is_holdout", False):
                continue
            txt = (d.get("sentence_text") or "").strip()
            wc = d.get("word_count", len(txt.split()))
            if not txt or wc < MIN_SENT_WORDS or wc > MAX_SENT_WORDS:
                continue
            user_to_sents[uid].append(txt)

    n_per_user = [len(user_to_sents[u]) for u in user_ids]
    log(f"  per-user sentence counts: min={min(n_per_user)}, max={max(n_per_user)}, mean={sum(n_per_user)/len(n_per_user):.1f}")
    eligible = [u for u in user_ids if len(user_to_sents[u]) >= N_SENTS_PER_USER]
    if len(eligible) < N_USERS:
        log(f"  WARNING: only {len(eligible)} users have >= {N_SENTS_PER_USER} sentences; will use what we have")

    user_ids = eligible
    # Take first N sents per user
    user_sents_dict = {u: user_to_sents[u][:N_SENTS_PER_USER] for u in user_ids}
    all_user_sents = [s for u in user_ids for s in user_sents_dict[u]]
    sent_uids = [u for u in user_ids for _ in range(N_SENTS_PER_USER)]
    n_total = len(all_user_sents)
    log(f"  eligible users: {len(user_ids)}, total sentences: {n_total}")

    # === [3] Generate neutral rewrites (vLLM batched, cached) ===
    log("[3] Generating neutral rewrites via vLLM batched ...")
    os.environ.setdefault("QWEN_GPU_MEMORY_UTILIZATION", "0.5")
    os.environ.setdefault("QWEN_MODEL_PATH", QWEN_MODEL_PATH)
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=True)
    log(f"  Qwen vLLM loaded")

    rewrites = load_rewrites()
    todo = [s for s in all_user_sents if s not in rewrites]
    log(f"  rewrites: {len(rewrites)} cached, {len(todo)} to generate")

    if todo:
        t0 = time.time()
        # Batched vLLM generate
        new_rows: list[dict] = []
        for start in range(0, len(todo), REWRITE_BATCH):
            chunk = todo[start:start + REWRITE_BATCH]
            try:
                texts = batch_generate_neutral(client, chunk, REWRITE_MAX_NEW, REWRITE_BATCH)
            except Exception as e:
                log(f"  ERROR at {start}: {e}")
                raise
            for s, t in zip(chunk, texts):
                new_rows.append({"sentence": s, "rewrite": t})
                rewrites[s] = t
            if (start // REWRITE_BATCH) % 20 == 0 or start + REWRITE_BATCH >= len(todo):
                log(f"  rewrites {min(start + REWRITE_BATCH, len(todo))}/{len(todo)} ({time.time()-t0:.1f}s)")
        append_rewrites(new_rows)
        log(f"  rewrites done: {len(rewrites)} total in {time.time()-t0:.1f}s")

    # === [4] Hidden states (all 28 layers) ===
    log(f"[4] Computing hidden states at all {N_LAYERS} layers ...")
    user_h_cache = load_hidden_cache(str(OUT_HIDDEN_USER))
    neutral_h_cache = load_hidden_cache(str(OUT_HIDDEN_NEUTRAL))
    log(f"  user cache: {len(user_h_cache)}, neutral cache: {len(neutral_h_cache)}")

    all_layers = list(range(N_LAYERS))

    # Filter out empty rewrites
    valid_user_sents = [s for s in all_user_sents if s in rewrites and len(rewrites[s].split()) >= 2]
    log(f"  valid sentences (with rewrite): {len(valid_user_sents)} / {n_total}")

    # Build (user, within_idx) for each valid sentence
    valid_uid_per_sent: list[str] = []
    valid_idx_per_sent: list[int] = []
    user_first10_idx: dict[str, int] = {u: 0 for u in user_ids}
    for u in user_ids:
        cnt = 0
        for s in user_sents_dict[u]:
            if s in valid_user_sents:
                valid_uid_per_sent.append(u)
                valid_idx_per_sent.append(cnt)
                cnt += 1
                if cnt >= N_SENTS_PER_USER:
                    break
    valid_neutral_sents = [rewrites[s] for s in valid_user_sents]
    log(f"  actual valid: {len(valid_user_sents)}, sample first: '{valid_user_sents[0][:50]}...' -> '{valid_neutral_sents[0][:50]}...'")

    todo_user = [s for s in valid_user_sents if s not in user_h_cache]
    todo_neutral = [s for s in valid_neutral_sents if s not in neutral_h_cache]
    log(f"  hidden: user {len(todo_user)} to compute, neutral {len(todo_neutral)} to compute")

    if todo_user:
        log("  computing user hiddens (batched) ...")
        t0 = time.time()
        for st in range(0, len(todo_user), HIDDEN_BATCH):
            chunk = todo_user[st:st + HIDDEN_BATCH]
            hdict = client.get_hidden_states(chunk, layers=all_layers, batch_size=HIDDEN_BATCH, max_length=HIDDEN_MAX_LENGTH)
            # hdict[layer] is [B, H]; stack to [B, L, H]
            arr = np.stack([hdict[l] for l in all_layers], axis=1).astype(np.float16)  # [B, L, H]
            for s, v in zip(chunk, arr):
                user_h_cache[s] = v
            if (st // HIDDEN_BATCH) % 50 == 0:
                log(f"    user hidden {min(st + HIDDEN_BATCH, len(todo_user))}/{len(todo_user)} ({time.time()-t0:.1f}s)")
        save_hidden_cache(str(OUT_HIDDEN_USER), user_h_cache)
        log(f"  user hidden done in {time.time()-t0:.1f}s, cache size: {len(user_h_cache)}")

    if todo_neutral:
        log("  computing neutral hiddens (batched) ...")
        t0 = time.time()
        for st in range(0, len(todo_neutral), HIDDEN_BATCH):
            chunk = todo_neutral[st:st + HIDDEN_BATCH]
            hdict = client.get_hidden_states(chunk, layers=all_layers, batch_size=HIDDEN_BATCH, max_length=HIDDEN_MAX_LENGTH)
            arr = np.stack([hdict[l] for l in all_layers], axis=1).astype(np.float16)  # [B, L, H]
            for s, v in zip(chunk, arr):
                neutral_h_cache[s] = v
            if (st // HIDDEN_BATCH) % 50 == 0:
                log(f"    neutral hidden {min(st + HIDDEN_BATCH, len(todo_neutral))}/{len(todo_neutral)} ({time.time()-t0:.1f}s)")
        save_hidden_cache(str(OUT_HIDDEN_NEUTRAL), neutral_h_cache)
        log(f"  neutral hidden done in {time.time()-t0:.1f}s, cache size: {len(neutral_h_cache)}")

    # === [5] Compute residuals and save ===
    log("[5] Computing per-sentence residuals r_i^ℓ = h_ℓ(user) - h_ℓ(neutral) ...")
    n_valid = len(valid_user_sents)
    # Allocate fp16: n_valid × 28 × 3584
    residuals = np.zeros((n_valid, N_LAYERS, HIDDEN_DIM), dtype=np.float16)
    for i, s in enumerate(valid_user_sents):
        n = rewrites[s]
        u_vec = user_h_cache[s]   # [L, H] fp16
        n_vec = neutral_h_cache[n]  # [L, H] fp16
        residuals[i] = u_vec.astype(np.float16) - n_vec.astype(np.float16)
        if (i + 1) % 500 == 0:
            log(f"  residuals {i + 1}/{n_valid}")

    # Per-user mean residual
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    mean_resid = np.zeros((len(user_ids), N_LAYERS, HIDDEN_DIM), dtype=np.float32)
    counts = np.zeros(len(user_ids), dtype=np.int32)
    for i, uid in enumerate(valid_uid_per_sent):
        j = uid_to_idx[uid]
        mean_resid[j] += residuals[i].astype(np.float32)
        counts[j] += 1
    valid_users_mask = counts > 0
    mean_resid[valid_users_mask] /= counts[valid_users_mask, None, None]

    log(f"  valid users: {int(valid_users_mask.sum())}/{len(user_ids)}")
    log(f"  residuals: {residuals.shape}, mean_resid: {mean_resid.shape}")
    log(f"  per-user residual L2 norm (mean over layers/users): "
        f"{np.linalg.norm(mean_resid[valid_users_mask], axis=-1).mean():.4f}")

    # === [6] Save ===
    log(f"[6] Saving residuals to {OUT_RESIDUALS} ...")
    np.savez_compressed(
        OUT_RESIDUALS,
        user_ids=np.asarray(user_ids, dtype=object),
        sent_uids=np.asarray(valid_uid_per_sent, dtype=object),
        sent_within_idx=np.asarray(valid_idx_per_sent, dtype=np.int32),
        residuals=residuals,
        mean_resid=mean_resid,
        user_sents=np.asarray(valid_user_sents, dtype=object),
        neutral_sents=np.asarray(valid_neutral_sents, dtype=object),
    )
    log(f"  saved: residuals {residuals.shape} fp16 (~{residuals.nbytes / 1e9:.2f} GB)")

    meta = {
        "phase": "13.A",
        "model": Path(QWEN_MODEL_PATH).name,
        "n_users": len(user_ids),
        "n_users_with_residuals": int(valid_users_mask.sum()),
        "n_total_sentences": n_valid,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "rewrite_system": REWRITE_SYSTEM,
        "rewrite_temp": REWRITE_TEMP,
        "rewrite_max_new": REWRITE_MAX_NEW,
        "rewrite_batch": REWRITE_BATCH,
        "hidden_batch": HIDDEN_BATCH,
        "min_sent_words": MIN_SENT_WORDS,
        "max_sent_words": MAX_SENT_WORDS,
        "residual_mean_l2": float(np.linalg.norm(mean_resid[valid_users_mask], axis=-1).mean()),
        "user_sentence_counts": {u: int(c) for u, c in zip(user_ids, counts.tolist())},
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta: {OUT_META}")

    log("=" * 70)
    log("PHASE 13.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()