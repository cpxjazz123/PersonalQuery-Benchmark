#!/usr/bin/env python3
"""E23 P0 — HELD-OUT validity of style vectors.

Question: is s_u a genuine, user-specific activation direction that
generalizes to sentences NEVER used to build it?

Protocol:
  for each of the 100 users with a style vector s_u:
    1. collect HELD-OUT sentences from reviews NOT in the construction set
       (construction set = the 1000 sentences stored in style_vectors_100u.npz)
    2. rewrite them to neutral (same protocol as construction)
    3. encode user + neutral held-out sentences -> layer activations
    4. Δ_heldout = act_user − act_neutral
  tests (per layer):
    A. identity retrieval: rank users by cosine(Δ_heldout, s_v);
       top1/top5 vs chance (1/100) — the core generalization test
    B. own-direction vs other-direction: cosine(Δ_heldout, s_u) vs
       cosine(Δ_heldout, s_v≠u) — user's own vector aligns better
    C. permutation null: retrieval under shuffled user→s mapping (~chance)
    D. cross-method comparison (mean_diff / logreg / pca) on held-out retrieval
    E. sign-consistency: frac of held-out pairs with cos(Δ_i, s_u) > 0

Hardcoded config, no CLI args. Run: python query_gen/e23_style_vector_heldout.py
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
REWRITES_JSONL = OUT_DIR / "rewrites.jsonl"
HIDDEN_CACHE = OUT_DIR / "hidden_cache.npz"
HO_REWRITES = OUT_DIR / "heldout_rewrites.jsonl"
HO_HIDDEN = OUT_DIR / "heldout_hidden.npz"
HO_JSON = OUT_DIR / "style_vectors_heldout_validity.json"

SEED = 7777
N_HO_PER_USER = 8            # held-out sentences per user
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60
REV_CAP = 300
REWRITE_BATCH = 16
MAX_NEW = 128
REWRITE_TEMPERATURE = 0.3
HIDDEN_BATCH = 32
LAYERS = [8, 12, 16, 20, 24, 26, 27]
METHODS = ["mean_diff", "logreg", "pca"]

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
    users = [str(u) for u in v["users"]]
    layers_arr = [int(l) for l in v["layers"]]
    construct_sents = set(str(s) for s in v["sentences"])
    sv = {m: v[m] for m in METHODS}          # [U, L, H] raw
    U = len(users)
    log(f"users={U}, layers={LAYERS}, construction sentences={len(construct_sents)}")

    # ---- collect held-out sentences (not in construction set) ----
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    reviews: dict[str, list[str]] = {u: [] for u in users}
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
    ho_sents: dict[str, list[str]] = {}
    for u in users:
        sents = []
        for doc in nlp.pipe(reviews[u], batch_size=64):
            for s in doc.sents:
                txt = s.text.strip()
                words = [t for t in s if not t.is_space]
                if txt and txt not in construct_sents and txt not in sents \
                        and MIN_SENT_WORDS <= len(words) <= MAX_SENT_WORDS:
                    sents.append(txt)
        ho_sents[u] = sents[:N_HO_PER_USER]
    n_ho = sum(len(s) for s in ho_sents.values())
    log(f"held-out sentences: {n_ho} "
        f"(per-user min={min(len(s) for s in ho_sents.values())}, "
        f"max={max(len(s) for s in ho_sents.values())})")

    # ---- rewrites for held-out sentences ----
    cached = {}
    if HO_REWRITES.exists():
        with open(HO_REWRITES, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cached[d["sentence"]] = d["rewrite"]
    all_ho = sorted({s for ss in ho_sents.values() for s in ss})
    todo = [s for s in all_ho if s not in cached]
    log(f"held-out rewrites: {len(cached)} cached, {len(todo)} to generate")

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
        with open(HO_REWRITES, "a", encoding="utf-8") as f:
            for s, t in zip(todo, texts):
                f.write(json.dumps({"sentence": s, "rewrite": t}) + "\n")
                cached[s] = t
    bad = sum(1 for t in cached.values() if len(t.split()) < 2)
    log(f"held-out rewrites done: total={len(cached)}, empty/short={bad}")

    @torch.no_grad()
    def pooled_hidden(texts: list[str]) -> np.ndarray:
        outs = []
        for st in range(0, len(texts), HIDDEN_BATCH):
            enc = tok(texts[st:st + HIDDEN_BATCH], return_tensors="pt",
                      padding=True, truncation=True, max_length=160).to(DEVICE)
            o = model(**enc, use_cache=False, output_hidden_states=True)
            lidx = [layers_arr.index(l) for l in LAYERS]
            hs = torch.stack([o.hidden_states[i] for i in lidx])
            m = enc["attention_mask"].unsqueeze(0).unsqueeze(-1).to(hs.dtype)
            p = (hs * m).sum(2) / m.sum(2).clamp_min(1)
            outs.append(p.permute(1, 0, 2).float().cpu().numpy())
        return np.concatenate(outs, axis=0)          # [T, L, H]

    # encode held-out user + neutral sentences
    ho_cache: dict[str, np.ndarray] = {}
    if HO_HIDDEN.exists():
        c = np.load(HO_HIDDEN, allow_pickle=True)
        ho_cache = {str(t): vec for t, vec in zip(c["texts"], c["vecs"])}
    need = sorted({s for ss in ho_sents.values() for s in ss}
                  | {cached[s] for ss in ho_sents.values() for s in ss})
    todo_h = [t for t in need if t not in ho_cache]
    log(f"held-out hidden: {len(ho_cache)} cached, {len(todo_h)} to compute")
    for st in range(0, len(todo_h), HIDDEN_BATCH):
        chunk = todo_h[st:st + HIDDEN_BATCH]
        vecs = pooled_hidden(chunk)
        for t, vec in zip(chunk, vecs):
            ho_cache[t] = vec
        log(f"  hidden {min(st + HIDDEN_BATCH, len(todo_h))}/{len(todo_h)}")
    np.savez_compressed(
        HO_HIDDEN,
        texts=np.asarray(list(ho_cache.keys())),
        vecs=np.stack([ho_cache[t] for t in ho_cache]))
    log(f"held-out hidden saved ({len(ho_cache)})")

    # ---- per-user held-out deltas: list of [n_i, L, H] ----
    ho_delta: list[np.ndarray] = []
    for u in users:
        seg = [s for s in ho_sents[u] if s in ho_cache and cached.get(s)
               and cached[s] in ho_cache]
        ua = np.stack([ho_cache[s] for s in seg])
        na = np.stack([ho_cache[cached[s]] for s in seg])
        ho_delta.append(ua - na)
    counts = [len(d) for d in ho_delta]
    log(f"held-out pairs: min={min(counts)} max={max(counts)} "
        f"total={sum(counts)}")

    # ---- A. identity retrieval: rank users by cos(Δ_heldout, s_v) ----
    def unit(a, axis=-1):
        return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-9)

    retr: dict[str, dict] = {}
    for li, l in enumerate(LAYERS):
        su = unit(sv["mean_diff"][:, li])                # [U, H]
        top1 = top5 = n = 0
        for ui, u in enumerate(users):
            for i in range(len(ho_delta[ui])):
                dq = unit(ho_delta[ui][i, li])
                sims = dq @ su.T
                order = np.argsort(-sims)
                rank = int(np.where(order == ui)[0][0]) + 1
                top1 += rank == 1
                top5 += rank <= 5
                n += 1
        retr[f"layer{l}"] = {
            "top1": round(float(top1 / n), 4),
            "top5": round(float(top5 / n), 4),
            "chance_top1": round(1.0 / U, 4),
            "n_pairs": n,
        }

    # ---- B. own-direction vs other-direction alignment ----
    align: dict[str, dict] = {}
    for li, l in enumerate(LAYERS):
        su = unit(sv["mean_diff"][:, li])
        own, other = [], []
        for ui, u in enumerate(users):
            for i in range(len(ho_delta[ui])):
                dq = unit(ho_delta[ui][i, li])
                own.append(float(dq @ su[ui]))
                others = np.delete(su, ui, axis=0)
                other.append(float(np.mean(dq @ others.T)))
        align[f"layer{l}"] = {
            "own_cos_mean": round(float(np.mean(own)), 4),
            "other_cos_mean": round(float(np.mean(other)), 4),
            "own_minus_other": round(float(np.mean(own) - np.mean(other)), 4),
            "frac_own_gt_other": round(float(
                np.mean([o > th for o, th in zip(own, other)])), 4),
        }

    # ---- C. permutation null ----
    null: dict[str, dict] = {}
    rng = random.Random(SEED + 1)
    for li, l in enumerate(LAYERS):
        su = unit(sv["mean_diff"][:, li])
        perms = [rng.sample(range(U), U) for _ in range(20)]
        accs = []
        for perm in perms:
            top1 = n = 0
            for ui, u in enumerate(users):
                for i in range(len(ho_delta[ui])):
                    dq = unit(ho_delta[ui][i, li])
                    sims = dq @ su.T
                    order = np.argsort(-sims)
                    rank = int(np.where(order == perm[ui])[0][0]) + 1
                    top1 += rank == 1
                    n += 1
            accs.append(top1 / n)
        null[f"layer{l}"] = {
            "permuted_top1_mean": round(float(np.mean(accs)), 4),
            "permuted_top1_std": round(float(np.std(accs)), 4),
        }

    # ---- D. cross-method held-out retrieval ----
    meth: dict[str, dict] = {}
    for m in METHODS:
        for li, l in enumerate(LAYERS):
            su = unit(sv[m][:, li])
            top1 = n = 0
            for ui, u in enumerate(users):
                for i in range(len(ho_delta[ui])):
                    dq = unit(ho_delta[ui][i, li])
                    sims = dq @ su.T
                    rank = int(np.where(np.argsort(-sims) == ui)[0][0]) + 1
                    top1 += rank == 1
                    n += 1
            meth.setdefault(m, {})[f"layer{l}"] = round(float(top1 / n), 4)

    # ---- E. sign consistency ----
    sign: dict[str, dict] = {}
    for li, l in enumerate(LAYERS):
        su = unit(sv["mean_diff"][:, li])
        pos = n = 0
        for ui, u in enumerate(users):
            for i in range(len(ho_delta[ui])):
                dq = unit(ho_delta[ui][i, li])
                pos += (dq @ su[ui]) > 0
                n += 1
        sign[f"layer{l}"] = {"frac_cos_gt_0": round(float(pos / n), 4)}

    # ---- F. aggregate retrieval: average k held-out deltas, then rank ----
    # simulates the personalization use case (k observed sentences -> s_u)
    agg: dict[str, dict] = {}
    for k in [2, 4, 8]:
        for li, l in enumerate(LAYERS):
            su = unit(sv["mean_diff"][:, li])
            top1 = n = 0
            for ui, u in enumerate(users):
                if len(ho_delta[ui]) < k:
                    continue
                dq = unit(ho_delta[ui][:k, li].mean(axis=0))
                sims = dq @ su.T
                rank = int(np.where(np.argsort(-sims) == ui)[0][0]) + 1
                top1 += rank == 1
                n += 1
            agg.setdefault(f"k{k}", {})[f"layer{l}"] = round(float(top1 / n), 4)

    result = {
        "version": "e23_style_vector_heldout_v1",
        "seed": SEED,
        "n_users": U,
        "n_heldout_pairs": sum(counts),
        "construction_set_excluded": True,
        "A_identity_retrieval_heldout": retr,
        "B_own_vs_other_alignment": align,
        "C_permutation_null": null,
        "D_cross_method_top1": meth,
        "E_sign_consistency": sign,
        "F_aggregate_retrieval_top1": agg,
        "heldout_sentences_per_user": {u: [s for s in ho_sents[u][:counts[ui]]]
                                       for ui, u in enumerate(users)
                                       if counts[ui] > 0},
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(HO_JSON, "w") as f:
        json.dump(result, f, indent=1)
    log("=== summary (held-out) ===")
    for l in LAYERS:
        a = retr[f"layer{l}"]
        b = align[f"layer{l}"]
        c = null[f"layer{l}"]
        e = sign[f"layer{l}"]
        log(f"L{l}: top1={a['top1']:.3f} top5={a['top5']:.3f} "
            f"(chance {a['chance_top1']:.3f}, perm-null {c['permuted_top1_mean']:.3f}) | "
            f"own={b['own_cos_mean']:.3f} other={b['other_cos_mean']:.3f} "
            f"(Δ={b['own_minus_other']:+.3f}, frac_own>other={b['frac_own_gt_other']:.3f}) | "
            f"sign>0 {e['frac_cos_gt_0']:.3f}")
    log("cross-method top1:")
    for m in METHODS:
        log(f"  {m}: " + " ".join(f"L{l}={meth[m][f'layer{l}']:.3f}"
                                   for l in LAYERS))
    log(f"DONE — wrote {HO_JSON} ({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
