#!/usr/bin/env python3
"""E22 Task 3 Step 3b: SCALE GRID evaluation (FULL 318-dim diagnostics).

User directive: evaluate different Query LENGTHS x Query COUNTS; find the
minimal generation scale that stably carries user syntax info.

Per (max_new, n_queries) cell, aggregating N test users x products:
  1. observability   — # 318 dims non-zero in the per-user aggregate (of the
                       generated queries)
  2. stability       — repeated-generation (rep 0 vs 1) aggregate 318-vec
                       closeness (cosine) for the same user
  3. fidelity        — generated aggregate vs user-review z_full 318 distance,
                       relative to the random-user baseline (d_other - d_own)
  4. controllability — paired swap: real-z query closer to own z than
                       shuffled-z query on the SAME product (aggregated)
  5. content         — 5 attrs / digits / brand exact rates

Outputs: result/e22_t3/e22_t3_grid_eval.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from e22_t3_check3 import neutralize_content  # noqa: E402
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
GEN = REPO_ROOT / "result" / "e22_t3" / "e22_t3_grid_generations.jsonl"
OUT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_grid_eval.json"

SEED = 7777
N_BOOTSTRAP = 2000
MIN_USERS = 20
CONTENT_EXACT_MIN = 0.99


def spacy318_full(text: str, attrs: dict, nlp, tm, ts) -> np.ndarray | None:
    qn = neutralize_content(text, attrs)
    doc = nlp(qn)
    sfs = [per_sentence_features_v2(s) for s in doc.sents]
    sfs = [s for s in sfs if s is not None]
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    return ((v - tm) / ts).astype(np.float64)


def digit_tokens(text: str) -> frozenset[str]:
    import re
    return frozenset(re.findall(r"\d+(?:\.\d+)?", text))


def content_exact(attrs: dict, q: str) -> dict:
    ql = q.lower()
    ok = {k: str(v).lower() in ql for k, v in attrs.items()}
    digits_ok = digit_tokens(q) == digit_tokens(" ".join(str(v) for v in attrs.values()))
    return {"all_5": all(ok.values()), "digits": digits_ok}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = set(mt["splits"]["test"]["ids"])

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    gens = [json.loads(l) for l in open(GEN)]
    print(f"generations: {len(gens)}", flush=True)

    # group by (max_new, control, rep, user)
    by_cell: dict[tuple, list] = defaultdict(list)
    for g in gens:
        v = spacy318_full(g["query"], g["attrs"], nlp, tm, ts)
        if v is None:
            continue
        by_cell[(g["max_new"], g["control"], g["rep"], g["user_id"])].append(v)
    print(f"cells with vectors: {len(by_cell)}", flush=True)

    results: dict[str, dict] = {}
    for mn in sorted({g["max_new"] for g in gens}):
        for nq in (2, 4, 8):
            cell = f"len{mn}_q{nq}"
            # ---- build per-user aggregates by sampling nq queries from the
            # user's available products (with replacement across reps) ----
            users = sorted({g["user_id"] for g in gens if g["max_new"] == mn})
            obs = []
            stab = []
            fid = []
            swap_acc = []
            cont = {"n": 0, "all_5": 0, "digits": 0}
            for u in users:
                real_pool = by_cell.get((mn, "real-z", 0, u), []) + \
                    by_cell.get((mn, "real-z", 1, u), [])
                if len(real_pool) < nq:
                    continue
                # observability: mean fraction of non-zero dims over sampled
                # aggregates
                agg0 = np.mean(real_pool[:nq], axis=0)
                obs.append(float((agg0 != 0).mean()))
                # stability: rep0-aggregate vs rep1-aggregate cosine
                r0 = by_cell.get((mn, "real-z", 0, u), [])
                r1 = by_cell.get((mn, "real-z", 1, u), [])
                if len(r0) >= nq and len(r1) >= nq:
                    a0 = np.mean(r0[:nq], axis=0)
                    a1 = np.mean(r1[:nq], axis=0)
                    stab.append(float(np.dot(a0, a1) /
                                      (np.linalg.norm(a0) * np.linalg.norm(a1) + 1e-9)))
                # fidelity: aggregate vs own z, vs mean-other z
                z_own = user_to_zfull[u]
                d_own = float(np.linalg.norm(agg0 - z_own))
                others = np.stack([user_to_zfull[o] for o in test_users
                                   if o != u and o in user_to_zfull])
                d_other = float(np.linalg.norm(agg0[None] - others, axis=1).mean())
                fid.append(d_other - d_own)
                # controllability: paired swap on same product (rep0 real vs
                # shuffled on same cell)
                shuf_pool = by_cell.get((mn, "shuffled-z", 0, u), []) + \
                    by_cell.get((mn, "shuffled-z", 1, u), [])
                if shuf_pool:
                    d_shuf = float(np.linalg.norm(
                        np.mean(shuf_pool[:nq], axis=0) - z_own))
                    swap_acc.append(int(d_own < d_shuf))
                # content (use real-z rep0 raw queries)
                for g in gens:
                    if (g["max_new"] == mn and g["control"] == "real-z"
                            and g["rep"] == 0 and g["user_id"] == u):
                        ce = content_exact(g["attrs"], g["query"])
                        cont["n"] += 1
                        cont["all_5"] += int(ce["all_5"])
                        cont["digits"] += int(ce["digits"])
                        break
            if len(users) < MIN_USERS:
                results[cell] = {"n_users": len(users), "run": False}
                continue
            obs = np.array(obs)
            stab = np.array(stab)
            fid = np.array(fid)
            swap_acc = np.array(swap_acc) if swap_acc else np.array([0.0])
            bs = np.array([fid[rng.integers(0, len(fid), len(fid))].mean()
                           for _ in range(N_BOOTSTRAP)])
            results[cell] = {
                "run": True,
                "n_users": len(users),
                "observable_dims_frac_mean": round(float(obs.mean()), 4),
                "observable_dims_count": round(float(obs.mean() * 318), 1),
                "stability_cosine_mean": round(float(stab.mean()), 4),
                "fidelity_margin_mean": round(float(fid.mean()), 4),
                "fidelity_margin_95ci": [round(float(np.quantile(bs, 0.025)), 4),
                                         round(float(np.quantile(bs, 0.975)), 4)],
                "swap_accuracy": round(float(swap_acc.mean()), 4),
                "content_exact_5": round(cont["all_5"] / max(1, cont["n"]), 4),
                "content_digits": round(cont["digits"] / max(1, cont["n"]), 4),
                "content_n": cont["n"],
            }
            print(f"  {cell}: obs={results[cell]['observable_dims_count']} "
                  f"stab={results[cell]['stability_cosine_mean']:.3f} "
                  f"fid={results[cell]['fidelity_margin_mean']:.3f} "
                  f"swap={results[cell]['swap_accuracy']:.3f} "
                  f"cont5={results[cell]['content_exact_5']:.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump({"version": "e22_t3_grid_eval_v1", "seed": SEED,
                   "results": results,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=1)
    print(f"wrote {OUT.name}; runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
