#!/usr/bin/env python3
"""E23 P0 — scaling sweep: how many sentences/words per user make the style
vector signal effective?

Motivation: the held-out validation showed weak signal (top1 2-3% vs chance
1%) with only 10 construction sentences per user. Each user actually has
~187 usable sentences (>=5 words). Hypothesis: more sentences -> mean_diff
direction becomes a reliable estimate of the user's real style.

Protocol:
  - fixed held-out set per user: 8 sentences (reuse heldout_rewrites/hidden)
  - construction set per user: up to 200 NEW sentences (excluding construction
    set of 1000 and the held-out 8), rewritten + encoded at layers {16,24,26}
  - user subset: those with >= 200 construction sentences (so every k in the
    grid is evaluated on the SAME users -> same chance level)
  - grid: k = {10,20,40,80,120,150,200} x min_words = {5,10,15}
    mean_diff built from first k sentences (each >= min_words)
  - metrics on held-out: identity retrieval top1, own-vs-other cosine gap,
    aggregate (8-sent average) retrieval top1

Outputs: result/e23_style_vector/scale_{rewrites,hidden,L}.npz (caches) +
  scale_validity.json (metrics)
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import DTYPE, DEVICE, MODEL_PATH, REVIEWS  # noqa: E402
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

OUT_DIR = REPO_ROOT / "result" / "e23_style_vector"
VEC_NPZ = OUT_DIR / "style_vectors_100u.npz"
HO_SENT_JSON = OUT_DIR / "style_vectors_heldout_validity.json"
SCALE_REWRITES = OUT_DIR / "scale_rewrites.jsonl"
SCALE_HIDDEN_L16 = OUT_DIR / "scale_hidden_L16.npz"
SCALE_HIDDEN_L24 = OUT_DIR / "scale_hidden_L24.npz"
SCALE_HIDDEN_L26 = OUT_DIR / "scale_hidden_L26.npz"
SCALE_JSON = OUT_DIR / "scale_validity.json"
HO_HIDDEN_FILE = OUT_DIR / "heldout_hidden.npz"

SEED = 7777
K_GRID = [10, 20, 40, 80, 120, 150, 200]
MIN_WORDS_GRID = [5, 10, 15]
K_MAX = max(K_GRID)
N_HO = 8                       # held-out sentences reused per user
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60
REV_CAP = 500
REWRITE_BATCH = 32
MAX_NEW = 128
REWRITE_TEMPERATURE = 0.3
HIDDEN_BATCH = 32
LAYERS = [16, 24, 26]          # strongest layers from held-out validation

REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a "
    "plain, neutral, matter-of-fact style. Keep the exact same meaning and "
    "all facts, but remove personal tone, slang, exclamations, and emotional "
    "words. Output ONLY the rewritten sentence, nothing else."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v = np.load(VEC_NPZ, allow_pickle=True)
    users_all = [str(u) for u in v["users"]]
    construct_1000 = set(str(s) for s in v["sentences"])
    ho_valid = json.load(open(HO_SENT_JSON))
    ho_per_user = {u: ho_valid["heldout_sentences_per_user"].get(u, [])
                   for u in users_all}
    U_all = len(users_all)
    log(f"users={U_all}, K_MAX={K_MAX}, layers={LAYERS}")

    # ---- collect construction sentences (exclude 1000-set + held-out) ----
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    reviews: dict[str, list[str]] = {u: [] for u in users_all}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in reviews or len(reviews[u]) >= REV_CAP:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews[u].append(t)
    log("reviews collected")
    exclude = set(construct_1000)
    for u in users_all:
        exclude.update(ho_per_user[u])
    constr_sents: dict[str, list[str]] = {}
    constr_wc: dict[str, list[int]] = {}     # word count per sentence
    for u in users_all:
        sents, wcs = [], []
        for doc in nlp.pipe(reviews[u], batch_size=64):
            for s in doc.sents:
                txt = s.text.strip()
                words = [t for t in s if not t.is_space]
                nw = len(words)
                if txt and txt not in exclude and MIN_SENT_WORDS <= nw \
                        <= MAX_SENT_WORDS:
                    sents.append(txt)
                    wcs.append(nw)
        constr_sents[u] = sents[:K_MAX]
        constr_wc[u] = wcs[:K_MAX]
    n_ok = sum(len(constr_sents[u]) >= K_MAX for u in users_all)
    log(f"construction sentences collected: "
        f"users with >= {K_MAX}: {n_ok}/{U_all}")

    # ---- rewrites (cached, append-only) ----
    cached = {}
    if SCALE_REWRITES.exists():
        with open(SCALE_REWRITES, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cached[d["sentence"]] = d["rewrite"]
    # held-out rewrites live in the heldout file (loaded for completeness)
    ho_rewrites = {}
    if (OUT_DIR / "heldout_rewrites.jsonl").exists():
        with open(OUT_DIR / "heldout_rewrites.jsonl", "r",
                  encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ho_rewrites[d["sentence"]] = d["rewrite"]
    cached.update(ho_rewrites)
    all_s = sorted({s for ss in constr_sents.values() for s in ss})
    todo = [s for s in all_s if s not in cached]
    log(f"scale rewrites: {len(cached)} cached, {len(todo)} to generate")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def batch_rewrite(texts: list[str]) -> list[str]:
        outs = []
        for st in range(0, len(texts), REWRITE_BATCH):
            chunk = texts[st:st + REWRITE_BATCH]
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
                input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                do_sample=True, temperature=REWRITE_TEMPERATURE, top_p=0.95,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True)
            for b, e in enumerate(encs):
                row = gen[b, len(e):]
                if tok.eos_token_id in row:
                    row = row[:row.tolist().index(tok.eos_token_id)]
                t = tok.decode(row, skip_special_tokens=True).strip()
                t = t.split("<|im_end|>")[0].strip()
                outs.append(t)
            log(f"  rewrite {min(st + REWRITE_BATCH, len(texts))}/{len(texts)}")
        return outs

    if todo:
        texts = batch_rewrite(todo)
        with open(SCALE_REWRITES, "a", encoding="utf-8") as f:
            for s, t in zip(todo, texts):
                f.write(json.dumps({"sentence": s, "rewrite": t}) + "\n")
                cached[s] = t
    bad = sum(1 for t in cached.values() if len(t.split()) < 2)
    log(f"scale rewrites done: total={len(cached)}, empty/short={bad}")

    # ---- hidden states at selected layers (cache per layer file) ----
    @torch.no_grad()
    def pooled_layers(texts: list[str]) -> np.ndarray:
        outs = []
        for st in range(0, len(texts), HIDDEN_BATCH):
            enc = tok(texts[st:st + HIDDEN_BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=160).to(DEVICE)
            o = model(**enc, use_cache=False, output_hidden_states=True)
            hs = torch.stack([o.hidden_states[l] for l in LAYERS])
            m = enc["attention_mask"].unsqueeze(0).unsqueeze(-1).to(hs.dtype)
            p = (hs * m).sum(2) / m.sum(2).clamp_min(1)
            outs.append(p.permute(1, 0, 2).float().cpu().numpy())
        return np.concatenate(outs, axis=0)          # [T, len(LAYERS), H]

    def load_hidden_cache(path) -> dict[str, np.ndarray]:
        if not path.exists():
            return {}
        c = np.load(path, allow_pickle=True)
        return {str(t): vec for t, vec in zip(c["texts"], c["vecs"])}

    def save_hidden_cache(path, cache: dict[str, np.ndarray]) -> None:
        items = sorted(cache.items())
        np.savez_compressed(
            path,
            texts=np.asarray([k for k, _ in items]),
            vecs=np.stack([np.asarray(x, dtype=np.float16) for _, x in items]))

    # encode: build all-text list = user sents + neutral rewrites
    cache_paths = {16: SCALE_HIDDEN_L16, 24: SCALE_HIDDEN_L24,
                   26: SCALE_HIDDEN_L26}
    caches = {l: load_hidden_cache(p) for l, p in cache_paths.items()}
    all_texts = sorted({s for ss in constr_sents.values() for s in ss}
                       | {cached[s] for ss in constr_sents.values()
                          for s in ss if s in cached})
    todo_h = [t for t in all_texts if t not in caches[LAYERS[0]]]
    log(f"scale hidden: {len(all_texts) - len(todo_h)} cached, "
        f"{len(todo_h)} to compute")
    for st in range(0, len(todo_h), HIDDEN_BATCH):
        chunk = todo_h[st:st + HIDDEN_BATCH]
        vecs = pooled_layers(chunk)                    # [B, 3, H]
        for t, row in zip(chunk, vecs):
            for li, l in enumerate(LAYERS):
                caches[l][t] = row[li]
        log(f"  hidden {min(st + HIDDEN_BATCH, len(todo_h))}/{len(todo_h)}")
    for l, p in cache_paths.items():
        save_hidden_cache(p, caches[l])

    # ---- load held-out activations from the existing heldout cache ----
    # heldout_hidden.npz stores all 7 layers [T, 7, H]; pick LAYERS cols
    ho_cache_all = load_hidden_cache(HO_HIDDEN_FILE)
    li_all = {l: [8, 12, 16, 20, 24, 26, 27].index(l) for l in LAYERS}
    ho_texts = set(ho_cache_all)
    log(f"held-out cache: {len(ho_texts)} texts")

    # ---- user subset: >= K_MAX construction sentences AND >= N_HO held-out ----
    keep_users = [u for u in users_all
                  if len(constr_sents[u]) >= K_MAX
                  and len([s for s in ho_per_user[u] if s in ho_texts
                           and cached.get(s) in ho_texts]) >= N_HO]
    U = len(keep_users)
    log(f"users with >= {K_MAX} construction sents + {N_HO} held-out: {U}")
    if U < 10:
        raise RuntimeError("too few qualifying users")

    # held-out deltas [U, N_HO, 3, H] (from heldout cache, 7-layer -> cols)
    def ho_vec(txt: str, l: int) -> np.ndarray:
        return np.asarray(ho_cache_all[txt])[li_all[l]]

    ho_delta = []
    for u in keep_users:
        seg = [s for s in ho_per_user[u]
               if s in ho_texts and cached.get(s) in ho_texts][:N_HO]
        ua3 = np.stack([np.stack([ho_vec(s, l) for l in LAYERS])
                        for s in seg])
        na3 = np.stack([np.stack([ho_vec(cached[s], l) for l in LAYERS])
                        for s in seg])
        ho_delta.append(ua3 - na3)
    ho_delta = np.stack(ho_delta)                      # [U, N_HO, 3, H]
    log(f"held-out deltas: {ho_delta.shape}")

    # construction deltas per user at full K_MAX: [U, K_MAX, 3, H]
    constr_delta = []
    for u in keep_users:
        seg = constr_sents[u][:K_MAX]
        ua = np.stack([np.stack([caches[l][s] for l in LAYERS]) for s in seg])
        na = np.stack([np.stack([caches[l][cached[s]] for l in LAYERS])
                       for s in seg])
        constr_delta.append(ua - na)
    constr_delta = np.stack(constr_delta)              # [U, K_MAX, 3, H]
    log(f"construction deltas: {constr_delta.shape}")

    # ---- metrics for each (k, min_words) ----
    # k must leave enough users (>=20) for a stable chance level
    def unit(a, axis=-1):
        return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-9)

    def n_users_with(k: int) -> int:
        return sum(1 for u in keep_users if len(constr_sents[u]) >= k)

    k_effective = [k for k in K_GRID if n_users_with(k) >= 20]
    log(f"k grid usable (>=20 users each): {k_effective}")

    results = {}
    for mw in MIN_WORDS_GRID:
        for k in k_effective:
            key = f"k{k}_mw{mw}"
            row = {"k": k, "min_words": mw}
            # per-user: first k sentences with word count >= mw
            # (keep the same user set across k -> same chance level)
            us = [ui for ui, u in enumerate(keep_users)
                  if sum(w >= mw for w in constr_wc[u][:k]) >= 3]
            n_us = len(us)
            row["n_users"] = n_us
            for li, l in enumerate(LAYERS):
                if n_us < 10:
                    row[f"L{l}"] = {"n_users": n_us}
                    continue
                S = np.zeros((n_us, constr_delta.shape[-1]))
                for j, ui in enumerate(us):
                    idx = [i for i in range(k) if constr_wc[keep_users[ui]][i] >= mw]
                    S[j] = constr_delta[ui, idx, li].mean(axis=0)
                s = unit(S)                              # [U, H]
                top1 = top5 = n = 0
                own_c = []
                other_c = []
                for j, ui in enumerate(us):
                    for i in range(N_HO):
                        dq = unit(ho_delta[ui, i, li])
                        sims = dq @ s.T
                        order = np.argsort(-sims)
                        rank = int(np.where(order == j)[0][0]) + 1
                        top1 += rank == 1
                        top5 += rank <= 5
                        n += 1
                        own_c.append(float(dq @ s[j]))
                        other_c.append(float(
                            np.mean(dq @ np.delete(s, j, axis=0).T)))
                dagg = unit(ho_delta[us, :, li].mean(axis=1))
                agg_ranks = [int(np.where(np.argsort(-r) == j)[0][0]) + 1
                             for j, r in enumerate(dagg @ unit(S).T)]
                agg_top1 = np.mean([rk == 1 for rk in agg_ranks])
                row[f"L{l}"] = {
                    "top1": round(float(top1 / n), 4),
                    "top5": round(float(top5 / n), 4),
                    "chance": round(1.0 / n_us, 4),
                    "own_cos": round(float(np.mean(own_c)), 4),
                    "other_cos": round(float(np.mean(other_c)), 4),
                    "own_minus_other": round(
                        float(np.mean(own_c) - np.mean(other_c)), 4),
                    "agg_top1": round(float(agg_top1), 4),
                }
            results[key] = row
    log("metrics computed")

    # ---- summary table ----
    log("=== top1 per k (L24, mw=5) ===")
    for k in k_effective:
        r = results[f"k{k}_mw5"]
        log(f"k={k:<4} n={r['n_users']} top1={r['L24']['top1']:.3f} "
            f"own-other={r['L24']['own_minus_other']:+.3f} "
            f"agg_top1={r['L24']['agg_top1']:.3f}")
    log("=== top1 per min_words (k=200, L24) ===")
    for mw in MIN_WORDS_GRID:
        r = results[f"k{K_MAX}_mw{mw}"]
        log(f"mw={mw:<3} n={r['n_users']} top1={r['L24']['top1']:.3f} "
            f"own-other={r['L24']['own_minus_other']:+.3f}")

    out = {
        "version": "e23_scale_validity_v1",
        "seed": SEED,
        "n_users": U,
        "k_grid": K_GRID,
        "min_words_grid": MIN_WORDS_GRID,
        "layers": LAYERS,
        "n_heldout": N_HO,
        "k_effective": k_effective,
        "results": results,
        "users": keep_users,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(SCALE_JSON, "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE — wrote {SCALE_JSON} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
