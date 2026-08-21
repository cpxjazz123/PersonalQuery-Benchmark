#!/usr/bin/env python3
"""Phase 14.F-paired: per-sentence paired neutral residual rerank.

Goal: refine Phase 14.F by replacing global neutral (mean over 2976 neutral rewrites)
with per-sentence paired neutral rewrites (LLM-rewritten for each user sentence +
each candidate query).

Pipeline:
  1. Load 198 user_ids + user_hiddens cache (198 × 30 × 5 × 3584 mean-pool)
  2. Re-extract user sentences (5940 = 198 × 30) from raw reviews (same logic as phase14_b)
  3. LLM rewrite each user sentence → neutral (cache JSONL; reuse phase13_a_rewrites_cache if available)
  4. Extract paired neutral hidden at 5 layers (5940 × 5 × 3584)
  5. Compute paired residual = user_hidden - paired_neutral_hidden
  6. Fit per-user per-layer Gaussian (paired)
  7. Pooled Maha (paired)
  8. Load 720 candidates from Phase 14.B jsonl
  9. LLM rewrite 720 candidates → neutral (separate cache)
 10. Extract cand neutral hidden (720 × 5 × 3584)
 11. Compute cand paired residual = cand_emb - cand_neutral_emb
 12. Per (pair, cond, layer) pooled Maha + per-user log-lik rerank
 13. Aggregate + paired bootstrap
 14. Verdict vs Phase 14.F (global neutral) baseline

Key cache files (all in OUT_DIR):
  - phase14_f_paired_user_sents.jsonl        # {uid, sent_idx, sentence}
  - phase14_f_paired_user_neutral_cache.jsonl  # {sentence, rewrite}
  - phase14_f_paired_user_neutral_hiddens.npz  # user_ids, sent_idx, layers, hiddens [5940, 5, 3584]
  - phase14_f_paired_cand_neutral_cache.jsonl  # {q_styled, rewrite}
  - phase14_f_paired_cand_neutral_hiddens.npy  # [720, 5, 3584]
  - phase14_f_paired_cand_residuals.npy         # [720, 5, 3584] cand - paired_neutral
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_REWRITES_CACHE = OUT_DIR / "phase13_a_rewrites_cache.jsonl"  # reused for user sentence rewrites
RAW_REVIEWS_GZ = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz")

OUT_USER_SENTS = OUT_DIR / "phase14_f_paired_user_sents.jsonl"
OUT_USER_NEUTRAL_CACHE = OUT_DIR / "phase14_f_paired_user_neutral_cache.jsonl"
OUT_USER_NEUTRAL_HIDDENS = OUT_DIR / "phase14_f_paired_user_neutral_hiddens.npz"
OUT_CAND_NEUTRAL_CACHE = OUT_DIR / "phase14_f_paired_cand_neutral_cache.jsonl"
OUT_CAND_NEUTRAL_HIDDENS = OUT_DIR / "phase14_f_paired_cand_neutral_hiddens.npy"
OUT_CAND_RESID = OUT_DIR / "phase14_f_paired_cand_residuals.npy"

OUT_EVAL = OUT_DIR / "phase14_f_paired_qwen_residual_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_f_paired_qwen_residual_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_f_paired_qwen_residual_rerank_meta.json"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
LAYERS_5 = [8, 14, 18, 22, 26]
N_FIT = 30
HIDDEN = 3584
MIN_WORDS = 5
MAX_WORDS = 60
LW_SHRINKAGE_PER_USER = 0.1
MIN_VAR = 1e-4
N_BOOTSTRAP = 2000
SEED = 42

# Rewrite LLM settings (matching phase13_a)
REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a plain, neutral, "
    "matter-of-fact style. Keep the exact same meaning and all facts, but remove personal tone, "
    "slang, exclamations, and emotional words. Output ONLY the rewritten sentence, nothing else."
)
REWRITE_MAX_NEW = 80
REWRITE_TEMP = 0.3
REWRITE_BATCH = 64


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED):
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def get_qwen_client(with_vllm: bool = False):
    """Get local Qwen client (transformers-only for rewrite + hidden states)."""
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    return create_qwen_local_client(with_vllm=with_vllm)


def batch_rewrite_transformers(client, sentences: list[str], batch_size: int = REWRITE_BATCH,
                              max_new: int = REWRITE_MAX_NEW, temp: float = REWRITE_TEMP) -> list[str]:
    """Batched transformers generate rewrite. Returns list of neutral rewrites."""
    import torch
    model = client._hidden_backend.model
    tokenizer = client._hidden_backend.tokenizer

    msgs_list = []
    for s in sentences:
        msgs = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Sentence: {s}\nNeutral rewrite:"},
        ]
        msgs_list.append(msgs)
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs_list
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    do_sample = temp > 0
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=do_sample, temperature=temp if do_sample else 1.0,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Trim prompt prefix
    input_len = enc["input_ids"].shape[1]
    new_tokens = gen[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def encode_texts_at_layers(client, texts: list[str], layers_0idx: list[int],
                           batch_size: int = 32, max_length: int = 256):
    """Mean-pool hidden states at specified layers (0-indexed hidden_states tuple)."""
    hdict = client.get_hidden_states(texts, layers=layers_0idx, batch_size=batch_size, max_length=max_length)
    return np.stack([hdict[l] for l in layers_0idx], axis=1).astype(np.float32)


def fit_user_gaussians(user_residuals: np.ndarray, lw_shrink: float = LW_SHRINKAGE_PER_USER,
                       min_var: float = MIN_VAR):
    """Per-user per-layer Gaussian fit with LW shrinkage.

    user_residuals: [n_users, n_samples, n_layers, H]
    Returns: mu [n_users, n_layers, H], var [n_users, n_layers, H]
    """
    mu = user_residuals.mean(axis=1)  # over sentences (axis=1)
    var = user_residuals.var(axis=1)
    if lw_shrink > 0:
        pooled_var = var.mean(axis=0, keepdims=True)
        var_shrunk = (1 - lw_shrink) * var + lw_shrink * pooled_var
    else:
        var_shrunk = var
    var_shrunk = np.maximum(var_shrunk, min_var)
    return mu.astype(np.float32), var_shrunk.astype(np.float32)


def pooled_maha_distance(cand_residuals, user_mu, pooled_var):
    """Pooled Mahalanobis distance per layer. cand_residuals: [K, L, H], user_mu: [U, L, H], pooled_var: [L, H]."""
    n_layers = cand_residuals.shape[1]
    inv_var = 1.0 / (pooled_var + 1e-6)
    distances = np.zeros((cand_residuals.shape[0], user_mu.shape[0], n_layers), dtype=np.float32)
    for li in range(n_layers):
        delta = cand_residuals[:, li, :][:, None, :] - user_mu[:, li, :][None, :, :]
        distances[:, :, li] = (delta * delta * inv_var[li][None, None, :]).sum(axis=-1)
    return distances


def per_user_loglik(cand_residuals, user_mu, user_var):
    """Per-user log-likelihood (summed across layers)."""
    n_layers = cand_residuals.shape[1]
    log_lik = np.zeros((cand_residuals.shape[0], user_mu.shape[0]), dtype=np.float32)
    for li in range(n_layers):
        delta2 = (cand_residuals[:, li, :][:, None, :] - user_mu[:, li, :][None, :, :]) ** 2
        term = delta2 / user_var[:, li, :][None, :, :] + np.log(user_var[:, li, :][None, :, :])
        log_lik -= 0.5 * term.sum(axis=-1)
    return log_lik


def extract_user_sentences_for_pairs(target_uids: list[str]) -> dict[str, list[str]]:
    """Replay same logic as phase14_b.extract_user_sentences."""
    user_sents: dict[str, list[str]] = {uid: [] for uid in target_uids}
    n_lines = 0
    target_set = set(target_uids)
    with gzip.open(RAW_REVIEWS_GZ, "rt") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("user_id") or r.get("reviewerID")
            if uid not in target_set:
                continue
            text = (r.get("text") or r.get("reviewText") or "").strip()
            if not text:
                continue
            sents = re.split(r"[.!?;。！？；]+", text)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                w = len(s.split())
                if MIN_WORDS <= w <= MAX_WORDS:
                    if len(user_sents[uid]) < N_FIT:
                        user_sents[uid].append(s)
    return user_sents


def main() -> None:
    log("=" * 70)
    log("Phase 14.F-paired: per-sentence paired neutral residual rerank")
    log("=" * 70)

    # === [1] Load Phase 14.B user hiddens + extract user sentences ===
    log(f"[1] Loading {IN_USER_HIDDENS.name} ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    log(f"  users: {len(user_ids_cache)}, hiddens shape: {user_hiddens.shape}")

    log("[2] Extracting user sentences (replay phase14_b logic) ...")
    if OUT_USER_SENTS.exists():
        log(f"  loading cached {OUT_USER_SENTS.name} ...")
        user_sents: dict[str, list[str]] = {}
        with OUT_USER_SENTS.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                user_sents.setdefault(r["user_id"], []).append(r["sentence"])
        log(f"  loaded: {sum(len(v) for v in user_sents.values())} sentences for {len(user_sents)} users")
    else:
        t0 = time.time()
        user_sents = extract_user_sentences_for_pairs(user_ids_cache)
        log(f"  extracted in {time.time()-t0:.1f}s")
        # Save
        with OUT_USER_SENTS.open("w") as f:
            for uid in user_ids_cache:
                for sent in user_sents.get(uid, [])[:N_FIT]:
                    f.write(json.dumps({"user_id": uid, "sentence": sent}, ensure_ascii=False) + "\n")
        log(f"  saved → {OUT_USER_SENTS}")

    # === [3] LLM rewrite user sentences → neutral (cache via JSONL) ===
    log("[3] LLM rewriting user sentences (with phase13_a cache reuse) ...")
    # Load phase13_a rewrites cache for reuse
    rewrites_cache: dict[str, str] = {}
    if IN_REWRITES_CACHE.exists():
        with IN_REWRITES_CACHE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rewrites_cache[r["sentence"]] = r["rewrite"]
        log(f"  loaded {len(rewrites_cache)} phase13_a rewrites for reuse")

    # Collect unique sentences
    all_user_sents: list[tuple[str, int, str]] = []  # (uid, sent_idx, sentence)
    for uid in user_ids_cache:
        for si, s in enumerate(user_sents.get(uid, [])[:N_FIT]):
            all_user_sents.append((uid, si, s))

    todo_sents: list[str] = []
    todo_indices: list[int] = []
    paired_rewrites: dict[str, str] = {}  # sentence → rewrite
    for idx, (_, _, s) in enumerate(all_user_sents):
        if s in rewrites_cache:
            paired_rewrites[s] = rewrites_cache[s]
        elif s not in paired_rewrites:
            todo_sents.append(s)
            todo_indices.append(idx)

    if OUT_USER_NEUTRAL_CACHE.exists():
        log(f"  loading cached {OUT_USER_NEUTRAL_CACHE.name} ...")
        with OUT_USER_NEUTRAL_CACHE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                paired_rewrites[r["sentence"]] = r["rewrite"]
        # Re-check todo after loading
        todo_sents = []
        todo_indices = []
        for idx, (_, _, s) in enumerate(all_user_sents):
            if s not in paired_rewrites:
                todo_sents.append(s)
                todo_indices.append(idx)

    log(f"  total: {len(all_user_sents)}, reuse: {len(all_user_sents) - len(todo_sents)}, "
        f"todo rewrite: {len(todo_sents)}")

    # Load transformers client once for reuse across [3]/[4]/[7]
    client = get_qwen_client(with_vllm=False)

    if todo_sents:
        log("  rewrites todo with transformers client ...")
        t0 = time.time()
        for st in range(0, len(todo_sents), REWRITE_BATCH):
            chunk = todo_sents[st:st + REWRITE_BATCH]
            try:
                rewrites = batch_rewrite_transformers(client, chunk)
            except Exception as e:
                log(f"  ERROR at {st}: {e}")
                raise
            for s, r in zip(chunk, rewrites):
                paired_rewrites[s] = r
            if st % (REWRITE_BATCH * 5) == 0 or st + REWRITE_BATCH >= len(todo_sents):
                log(f"    rewrites {min(st + REWRITE_BATCH, len(todo_sents))}/{len(todo_sents)} ({time.time()-t0:.1f}s)")
        log(f"  rewrites done in {time.time()-t0:.1f}s")
        # Save cache (append all)
        with OUT_USER_NEUTRAL_CACHE.open("w") as f:
            for s, r in paired_rewrites.items():
                f.write(json.dumps({"sentence": s, "rewrite": r}, ensure_ascii=False) + "\n")
        log(f"  saved → {OUT_USER_NEUTRAL_CACHE}")

    # === [4] Extract paired neutral hidden at 5 layers ===
    log("[4] Extracting paired neutral hidden at 5 layers ...")
    if OUT_USER_NEUTRAL_HIDDENS.exists():
        log(f"  loading cached {OUT_USER_NEUTRAL_HIDDENS.name} ...")
        npz = np.load(OUT_USER_NEUTRAL_HIDDENS, allow_pickle=True)
        cached_uids = list(npz["user_ids"])
        cached_sidx = list(npz["sent_idx"])
        cached_layers = list(npz["layers"])
        cached_hiddens = npz["hiddens"]  # [N_total, 5, 3584]
        log(f"  cached: {cached_hiddens.shape}, layers={cached_layers}")
        if (cached_layers == LAYERS_5 and len(cached_uids) == len(all_user_sents)):
            neutral_hiddens_user = cached_hiddens.astype(np.float32)
            log(f"  fully cached, skip")
        else:
            raise RuntimeError(f"cache mismatch: layers {cached_layers} vs {LAYERS_5}, "
                               f"n {len(cached_uids)} vs {len(all_user_sents)}")
    else:
        # Build neutral sentences in order
        neutral_sents = [paired_rewrites[s] for (_, _, s) in all_user_sents]
        log(f"  encoding {len(neutral_sents)} neutral sentences at 5 layers ...")
        t0 = time.time()
        neutral_hiddens_user = encode_texts_at_layers(client, neutral_sents, LAYERS_5,
                                                       batch_size=32, max_length=256)
        log(f"  encoded in {time.time()-t0:.1f}s, shape={neutral_hiddens_user.shape}")
        np.savez(OUT_USER_NEUTRAL_HIDDENS,
                 user_ids=np.array([uid for (uid, _, _) in all_user_sents]),
                 sent_idx=np.array([si for (_, si, _) in all_user_sents]),
                 layers=np.array(LAYERS_5),
                 hiddens=neutral_hiddens_user.astype(np.float16))
        log(f"  saved → {OUT_USER_NEUTRAL_HIDDENS}")

    # === [5] Compute paired residuals + fit Gaussian ===
    log("[5] Computing paired residuals + fitting Gaussian ...")
    # Reshape user_hiddens to (n_users, n_samples, n_layers, H) and pair with neutral
    # user_hiddens is (198, 30, 5, 3584) mean-pool per user, sent_idx 0..29, layer 0..4
    # neutral_hiddens_user is (5940, 5, 3584) per sentence, layer 0..4
    # Pair: (uid, si) → neutral hiddens for same (uid, si)
    neutral_by_pair = {}
    for (uid, si, _), h in zip(all_user_sents, neutral_hiddens_user):
        neutral_by_pair[(uid, si)] = h

    user_residuals_paired = np.zeros_like(user_hiddens)
    n_filled = 0
    for ui, uid in enumerate(user_ids_cache):
        for si in range(N_FIT):
            h_neutral = neutral_by_pair.get((uid, si))
            if h_neutral is None:
                raise RuntimeError(f"missing neutral for ({uid}, {si})")
            user_residuals_paired[ui, si] = user_hiddens[ui, si] - h_neutral
            n_filled += 1
    log(f"  user_residuals_paired shape: {user_residuals_paired.shape}, mean norm per layer: "
        f"{np.linalg.norm(user_residuals_paired.mean(axis=(0,1)), axis=-1)}")

    user_mu_paired, user_var_paired = fit_user_gaussians(user_residuals_paired)
    pooled_var_paired = user_var_paired.mean(axis=0)  # [5, 3584]
    log(f"  user_mu_paired shape: {user_mu_paired.shape}")
    log(f"  pooled_var_paired mean per layer: {pooled_var_paired.mean(axis=-1)}")

    # === [6] Load candidates ===
    log(f"[6] Loading candidates from {IN_JSONL.name} ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    log(f"  candidates: {len(candidates)}")

    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [7] LLM rewrite candidates + extract cand neutral hidden ===
    log("[7] LLM rewriting 720 candidates → neutral ...")
    cand_qs = [c.get("q_styled") or "" for c in candidates]

    if OUT_CAND_NEUTRAL_HIDDENS.exists():
        log(f"  loading cached {OUT_CAND_NEUTRAL_HIDDENS.name} ...")
        cand_neutral_hiddens = np.load(OUT_CAND_NEUTRAL_HIDDENS)
        if cand_neutral_hiddens.shape[0] != len(candidates):
            raise RuntimeError(f"cache shape mismatch: {cand_neutral_hiddens.shape}")
    else:
        cand_rewrites: dict[str, str] = {}
        if OUT_CAND_NEUTRAL_CACHE.exists():
            with OUT_CAND_NEUTRAL_CACHE.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    cand_rewrites[r["q_styled"]] = r["rewrite"]
        todo_cands = [q for q in cand_qs if q not in cand_rewrites]
        log(f"  unique cands: {len(set(cand_qs))}, todo rewrite: {len(todo_cands)}, reuse: {len(set(cand_qs)) - len(todo_cands)}")

        if todo_cands:
            log("  reusing transformers client (already loaded) for cand rewrite ...")
            t0 = time.time()
            for st in range(0, len(todo_cands), REWRITE_BATCH):
                chunk = todo_cands[st:st + REWRITE_BATCH]
                rewrites = batch_rewrite_transformers(client, chunk)
                for q, r in zip(chunk, rewrites):
                    cand_rewrites[q] = r
                if st % (REWRITE_BATCH * 5) == 0 or st + REWRITE_BATCH >= len(todo_cands):
                    log(f"    cand rewrites {min(st + REWRITE_BATCH, len(todo_cands))}/{len(todo_cands)} ({time.time()-t0:.1f}s)")
            with OUT_CAND_NEUTRAL_CACHE.open("w") as f:
                for q, r in cand_rewrites.items():
                    f.write(json.dumps({"q_styled": q, "rewrite": r}, ensure_ascii=False) + "\n")
            log(f"  cand rewrites done in {time.time()-t0:.1f}s, saved → {OUT_CAND_NEUTRAL_CACHE}")

        # Now encode cand neutral rewrites
        cand_neutral_sents = [cand_rewrites[q] for q in cand_qs]
        log(f"  encoding {len(cand_neutral_sents)} cand neutral hiddens ...")
        t0 = time.time()
        cand_neutral_hiddens = encode_texts_at_layers(client, cand_neutral_sents, LAYERS_5,
                                                       batch_size=32, max_length=256)
        log(f"  encoded in {time.time()-t0:.1f}s, shape={cand_neutral_hiddens.shape}")
        np.save(OUT_CAND_NEUTRAL_HIDDENS, cand_neutral_hiddens)
        log(f"  saved → {OUT_CAND_NEUTRAL_HIDDENS}")

    # === [8] Compute cand paired residuals ===
    log("[8] Computing cand paired residuals (cand_embs - cand_neutral) ...")
    # Load cand embs from Phase 14.F cache
    cand_embs_path = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"
    if cand_embs_path.exists():
        cand_embs_global_resid = np.load(cand_embs_path).astype(np.float32)  # [720, 5, 3584]
        log(f"  loaded cand global residuals from {cand_embs_path.name}: {cand_embs_global_resid.shape}")
    else:
        # Recompute cand_embs (without subtracting global_neutral)
        log("  recomputing cand_embs (no global neutral subtraction) ...")
        client = get_qwen_client(with_vllm=False)
        cand_embs = encode_texts_at_layers(client, cand_qs, LAYERS_5, batch_size=32, max_length=256)
        cand_embs_global_resid = cand_embs  # identity here
        log(f"  cand_embs shape: {cand_embs.shape}")

    # But wait — Phase 14.F cand_residuals was cand_embs - global_neutral.
    # For paired, we need cand_embs (raw) - cand_neutral_hiddens.
    # If we only have cand_residuals (= cand_embs - global_neutral), need to recover cand_embs.
    # Easier: just re-encode here.
    log("  re-encoding cand_qs (raw, no global neutral subtraction) ...")
    client = get_qwen_client(with_vllm=False)
    cand_embs_raw = encode_texts_at_layers(client, cand_qs, LAYERS_5, batch_size=32, max_length=256)
    log(f"  cand_embs_raw shape: {cand_embs_raw.shape}")

    cand_residuals_paired = cand_embs_raw - cand_neutral_hiddens  # [720, 5, 3584]
    log(f"  cand_residuals_paired mean norm per layer: {np.linalg.norm(cand_residuals_paired.mean(axis=0), axis=-1)}")
    np.save(OUT_CAND_RESID, cand_residuals_paired)
    log(f"  saved → {OUT_CAND_RESID}")

    # === [9] Per-(pair, cond, layer) rerank ===
    log("[9] Per-(pair, cond, layer) rerank (pooled Maha + per-user log-lik, paired residuals) ...")
    pair_keys = sorted({(c["user_id"], c["asin"]) for c in candidates})

    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}
    pair_user_idx = [(uid, asin, uid_to_useridx.get(uid)) for (uid, asin) in pair_keys]

    per_pair_per_cond_layer_maha: dict[tuple[str, str, str, int], int] = {}
    per_pair_per_cond_layer_loglik: dict[tuple[str, str, int], int] = {}

    n_skipped = 0
    for (uid, asin, ui) in pair_user_idx:
        if ui is None:
            n_skipped += 1
            continue
        for cond, cand_indices in cand_by_pair_cond[(uid, asin)].items():
            local_resid = cand_residuals_paired[cand_indices]
            maha_d = pooled_maha_distance(local_resid, user_mu_paired, pooled_var_paired)
            loglik = per_user_loglik(local_resid, user_mu_paired, user_var_paired)
            for li in range(len(LAYERS_5)):
                layer_d = maha_d[:, :, li]
                target_d = layer_d[:, ui:ui + 1]
                best_rank = int(np.min((layer_d < target_d).sum(axis=1)))
                per_pair_per_cond_layer_maha[(uid, asin, cond, LAYERS_5[li])] = best_rank
            target_loglik = loglik[:, ui:ui + 1]
            best_rank_ll = int(np.min((loglik < target_loglik).sum(axis=1)))
            per_pair_per_cond_layer_loglik[(uid, asin, cond, -1)] = best_rank_ll
    log(f"  done; skipped: {n_skipped}")

    # === [10] Aggregate ===
    log("[10] Aggregating per (cond, layer, metric) ...")
    pair_keys_valid = [(uid, asin) for (uid, asin, ui) in pair_user_idx if ui is not None]

    per_cond_layer_eval: dict[tuple[str, str, int], dict] = {}
    for cond in CONDITIONS:
        for layer in LAYERS_5:
            ranks = []
            for (uid, asin) in pair_keys_valid:
                r = per_pair_per_cond_layer_maha.get((uid, asin, cond, layer))
                if r is not None:
                    ranks.append(r)
            arr = np.array(ranks)
            mean, ci = bootstrap_ci(ranks)
            per_cond_layer_eval[(cond, "maha", layer)] = {
                "n": len(ranks),
                "rank1_coverage": float((arr == 0).sum()) / max(1, len(arr)),
                "top10_coverage": float((arr < 10).sum()) / max(1, len(arr)),
                "top100_coverage": float((arr < 100).sum()) / max(1, len(arr)),
                "mean_best_rank": mean,
                "mean_best_rank_ci95": list(ci),
            }
            log(f"  maha {cond} layer={layer}: rank1={(arr==0).sum()}/{len(arr)}, "
                f"top100={(arr<100).sum()}/{len(arr)}, mean={mean:.1f} CI [{ci[0]:.1f}, {ci[1]:.1f}]")
        ranks_ll = []
        for (uid, asin) in pair_keys_valid:
            r = per_pair_per_cond_layer_loglik.get((uid, asin, cond, -1))
            if r is not None:
                ranks_ll.append(r)
        arr_ll = np.array(ranks_ll)
        mean_ll, ci_ll = bootstrap_ci(ranks_ll)
        per_cond_layer_eval[(cond, "loglik", -1)] = {
            "n": len(ranks_ll),
            "rank1_coverage": float((arr_ll == 0).sum()) / max(1, len(arr_ll)),
            "top10_coverage": float((arr_ll < 10).sum()) / max(1, len(arr_ll)),
            "top100_coverage": float((arr_ll < 100).sum()) / max(1, len(arr_ll)),
            "mean_best_rank": mean_ll,
            "mean_best_rank_ci95": list(ci_ll),
        }
        log(f"  loglik {cond} (sum layers): rank1={(arr_ll==0).sum()}/{len(arr_ll)}, "
            f"top100={(arr_ll<100).sum()}/{len(arr_ll)}, mean={mean_ll:.1f}")

    # === [11] Paired bootstrap diff (paired vs Phase 14.F global) ===
    log("[11] Paired bootstrap diff vs Phase 14.F (global neutral) baseline ...")
    # Load Phase 14.F eval
    f_eval_path = OUT_DIR / "phase14_f_qwen_residual_rerank_eval.json"
    f_per_pair_path = OUT_DIR / "phase14_f_qwen_residual_rerank_per_pair.jsonl"
    if f_per_pair_path.exists():
        f_per_pair = {}
        with f_per_pair_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                f_per_pair[(r["user_id"], r["asin"])] = r
        # Compare mean ranks per cond × layer
        diffs_vs_global: dict[str, dict] = {}
        for cond in CONDITIONS:
            for layer in LAYERS_5:
                diffs = []
                for (uid, asin) in pair_keys_valid:
                    r_paired = per_pair_per_cond_layer_maha.get((uid, asin, cond, layer))
                    f_row = f_per_pair.get((uid, asin))
                    if r_paired is not None and f_row is not None:
                        r_global = f_row.get(f"{cond}_maha_layer{layer}")
                        if r_global is not None:
                            diffs.append(int(r_global) - int(r_paired))  # positive = paired better
                if diffs:
                    mean_d, ci_d = bootstrap_ci(diffs)
                    diffs_vs_global[f"{cond}_layer{layer}"] = {
                        "mean_diff": mean_d,
                        "ci95": list(ci_d),
                        "ci_excludes_0": ci_d[0] > 0,
                        "n": len(diffs),
                    }
                    log(f"  {cond} layer={layer}: paired_vs_global diff={mean_d:+.1f} "
                        f"CI [{ci_d[0]:+.1f}, {ci_d[1]:+.1f}] "
                        f"({'excl 0' if ci_d[0] > 0 else 'incl 0'})")
    else:
        log(f"  Phase 14.F per_pair not found at {f_per_pair_path}, skip diff")
        diffs_vs_global = {}

    # === [12] Save ===
    out = {
        "phase": "14.F-paired",
        "rerank_space": "Qwen mean-pool paired residual (sentence_hidden - LLM-rewritten_paired_neutral_hidden) per layer",
        "layers": LAYERS_5,
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys_valid),
        "n_users": len(user_ids_cache),
        "lw_shrinkage_user": LW_SHRINKAGE_PER_USER,
        "min_var": MIN_VAR,
        "per_cond_layer_eval": {
            f"{cond}__{metric}__layer{layer}": per_cond_layer_eval[(cond, metric, layer)]
            for (cond, metric, layer) in per_cond_layer_eval.keys()
        },
        "paired_vs_global_diffs": diffs_vs_global,
        "comparison_baseline_phase14_f_global": {
            "A14_a1.0_maha_layer26_top100": "86.7%",
            "A14_a1.0_maha_layer26_mean_rank": 47.2,
            "A22_a0.5_maha_layer26_top100": "90.0%",
            "A22_a0.5_maha_layer26_mean_rank": 50.3,
            "D_off_maha_layer26_top100": "83.3%",
            "D_off_maha_layer26_mean_rank": 55.8,
        },
        "decision_logic": {
            "paired > global": "diff CI > 0 (paired neutral residual rerank > global neutral)",
            "best_cond_layer_paired": "report best cond × layer per metric",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in pair_keys_valid:
            row = {"user_id": uid, "asin": asin}
            for cond in CONDITIONS:
                for layer in LAYERS_5:
                    row[f"{cond}_maha_layer{layer}"] = per_pair_per_cond_layer_maha.get((uid, asin, cond, layer))
                row[f"{cond}_loglik_sum"] = per_pair_per_cond_layer_loglik.get((uid, asin, cond, -1))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.F-paired",
        "in_jsonl": str(IN_JSONL),
        "in_user_hiddens": str(IN_USER_HIDDENS),
        "in_rewrites_cache": str(IN_REWRITES_CACHE),
        "out_user_sents": str(OUT_USER_SENTS),
        "out_user_neutral_cache": str(OUT_USER_NEUTRAL_CACHE),
        "out_user_neutral_hiddens": str(OUT_USER_NEUTRAL_HIDDENS),
        "out_cand_neutral_cache": str(OUT_CAND_NEUTRAL_CACHE),
        "out_cand_neutral_hiddens": str(OUT_CAND_NEUTRAL_HIDDENS),
        "out_cand_residuals": str(OUT_CAND_RESID),
        "out_eval": str(OUT_EVAL),
        "out_per_pair": str(OUT_PER_PAIR),
        "n_pairs": len(pair_keys_valid),
        "n_users_cache": len(user_ids_cache),
        "layers": LAYERS_5,
        "lw_shrinkage_user": LW_SHRINKAGE_PER_USER,
        "min_var": MIN_VAR,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.F-PAIRED COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()