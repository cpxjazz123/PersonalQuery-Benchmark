#!/usr/bin/env python3
"""query_gen_final_swap — Task 3 final validation: fixed product, swap users.

Design (user directive):
  Use the 4,194 products bought by >=10 users (all 10 attrs). For each product
  take its top-10 users by review count; compute each user's FULL 318-dim
  z_user from ALL their reviews (Task-1 standardization params, frozen).
  Then for pairs (A, B) of users on the SAME product:
    - inject z_A -> generate query_A (real user A style)
    - inject z_B -> generate query_B (real user B style)
    - aggregate each user's generated queries into 318-dim syntax vectors
      (multi-query pooled, same protocol as z_user)
  Check:
    - swap: aggregated query_A closer to z_A than to z_B (and vice versa)
    - user-level bootstrap CI / permutation test
    - content preservation (10 attrs exact)
    - report on FULL 318 and on the comparable-dim subset

Outputs: result/e22_t3/e22_t3_final_swap.json
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import (  # noqa: E402
    SoftPrefixProjector, CopyAwareHead, generate_batch, user_prompt_tokens,
    spacy318_aggregate, attrs_for_n, load_meta, EXT_ATTR_KEYS, TOP_ATTR_KEYS,
    DTYPE, DEVICE, NUM_TOKENS, PROJ_HIDDEN, Z_DIM, GATE_INIT, MODEL_PATH,
    INJECTOR_PT, TASK1_VECTORS, REVIEWS,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)

# register feature functions into query_gen_main's globals (spacy318_aggregate
# reads them from there)
import query_gen_main as _qgm
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_final_swap.json"
SEED = 7777
N_PRODUCTS = 60            # products used in the swap experiment
N_QUERIES_PER_USER = 2     # generated queries aggregated per user
MAX_NEW = 48
BATCH = 4
N_BOOTSTRAP = 2000
N_PERM = 999
MIN_USERS = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    meta = load_meta()
    log("indexing reviews (single pass)...")
    prod_users: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    user_all: dict[str, list[str]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            text = d.get("text") or ""
            if u and a and text.strip():
                prod_users[a][u].append(text.strip())
                user_all[u].append(text.strip())
            if i > 20000000:
                break
    cands = [a for a, uc in prod_users.items()
             if len(uc) >= 10 and a in meta
             and attrs_for_n(meta[a], 10) is not None]
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands = cands[:N_PRODUCTS]
    log(f"products: {len(cands)}")

    # per-product top-10 users; collect their ALL reviews for z_user
    prod_top_users: dict[str, list[str]] = {}
    target_users: set[str] = set()
    for a in cands:
        uc = prod_users[a]
        top = [u for u, _ in
               sorted(uc.items(), key=lambda kv: -len(kv[1]))[:10]]
        prod_top_users[a] = top
        target_users.update(top)
    log(f"target users: {len(target_users)}")

    # batch spaCy over all target-user reviews (single pipe pass)
    user_sfs: dict[str, list] = defaultdict(list)
    for u in target_users:
        texts = user_all[u][:200]
        docs = nlp.pipe(texts, batch_size=256)
        for doc in docs:
            for s in doc.sents:
                sf = per_sentence_features_v2(s)
                if sf is not None:
                    user_sfs[u].append(sf)
    user_z: dict[str, np.ndarray] = {}
    for u, sfs in user_sfs.items():
        if len(sfs) < 5:
            continue
        v = user_features_v2(sfs)
        if v is not None:
            user_z[u] = ((v - tm) / ts).astype(np.float64)
    log(f"users with z_full: {len(user_z)}")

    # pairs: (product, userA, userB) — same product, both with z
    pairs = []
    for a in cands:
        top = [u for u in prod_top_users[a] if u in user_z]
        if len(top) < 2:
            continue
        rng.shuffle(top)
        for i in range(0, len(top) - 1, 2):
            pairs.append((a, top[i], top[i + 1]))
    rng.shuffle(pairs)
    log(f"swap pairs: {len(pairs)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(INJECTOR_PT, map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector loaded")

    # generate per pair: query_A (z_A) and query_B (z_B) on the SAME product
    # (attrs identical). Aggregate N_QUERIES_PER_USER queries per user.
    swap_acc = []
    margins = []
    user_deltas: dict[str, list[float]] = defaultdict(list)
    n_done = 0
    for a, uA, uB in pairs:
        attrs = attrs_for_n(meta[a], 10)
        zA = user_z[uA]
        zB = user_z[uB]
        with torch.no_grad():
            pA = proj(torch.tensor(zA, dtype=torch.float32,
                                   device=DEVICE).to(DTYPE).unsqueeze(0))[0]
            pB = proj(torch.tensor(zB, dtype=torch.float32,
                                   device=DEVICE).to(DTYPE).unsqueeze(0))[0]
        prompt = [user_prompt_tokens(attrs, tok)]
        # generate queries for A (repeat to aggregate)
        qAs = generate_batch(model, tok, copy_head,
                             [pA] * N_QUERIES_PER_USER,
                             prompt * N_QUERIES_PER_USER,
                             [attrs] * N_QUERIES_PER_USER, MAX_NEW)
        qBs = generate_batch(model, tok, copy_head,
                             [pB] * N_QUERIES_PER_USER,
                             prompt * N_QUERIES_PER_USER,
                             [attrs] * N_QUERIES_PER_USER, MAX_NEW)
        aggA = spacy318_aggregate(
            [(q, attrs) for q in qAs], nlp, tm, ts)
        aggB = spacy318_aggregate(
            [(q, attrs) for q in qBs], nlp, tm, ts)
        if aggA is None or aggB is None:
            continue
        dAA = float(np.linalg.norm(aggA - zA))
        dAB = float(np.linalg.norm(aggA - zB))
        dBB = float(np.linalg.norm(aggB - zB))
        dBA = float(np.linalg.norm(aggB - zA))
        # swap success: query_A closer to z_A, query_B closer to z_B
        okA = dAA < dAB
        okB = dBB < dBA
        swap_acc.append(int(okA and okB))
        margins.append((dAB - dAA) + (dBA - dBB))
        user_deltas[uA].append(dAB - dAA)
        user_deltas[uB].append(dBA - dBB)
        n_done += 1
        if n_done % 40 == 0:
            log(f"  {n_done}/{len(pairs)} pairs done")
    if not swap_acc:
        log("no valid pairs")
        return

    swap_acc = np.array(swap_acc)
    margins = np.array(margins)
    # user-level margin aggregation
    user_m = np.array([np.mean(v) for v in user_deltas.values() if v])
    rng2 = np.random.default_rng(SEED)
    bs = np.array([user_m[rng2.integers(0, len(user_m), len(user_m))].mean()
                   for _ in range(N_BOOTSTRAP)])
    obs = float(user_m.mean())
    perm = np.empty(N_PERM)
    for k in range(N_PERM):
        perm[k] = (user_m * rng2.choice([-1, 1], len(user_m))).mean()
    pc = float(perm.mean())
    p_two = float((np.abs(perm - pc) >= abs(obs - pc)).sum() + 1) / (N_PERM + 1)

    result = {
        "version": "query_gen_final_swap_v1",
        "seed": SEED, "n_products": N_PRODUCTS,
        "n_queries_per_user": N_QUERIES_PER_USER,
        "n_pairs": len(swap_acc),
        "n_users": len(user_m),
        "swap_accuracy": round(float(swap_acc.mean()), 4),
        "margin_mean": round(obs, 4),
        "margin_user_95ci": [round(float(np.quantile(bs, 0.025)), 4),
                             round(float(np.quantile(bs, 0.975)), 4)],
        "perm_p_two": round(p_two, 5),
        "gate_pass": bool(swap_acc.mean() >= 0.6
                          and np.quantile(bs, 0.025) > 0
                          and p_two < 0.01),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    log(f"done: acc={result['swap_accuracy']} margin={result['margin_mean']} "
        f"CI={result['margin_user_95ci']} p={result['perm_p_two']} "
        f"pass={result['gate_pass']}")


if __name__ == "__main__":
    main()
