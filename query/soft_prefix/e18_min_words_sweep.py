#!/usr/bin/env python3
"""E18: min-words-per-user threshold sweep for style-vector stability.

Question: what is the smallest per-user review word-count threshold T such
that the user syntactic style vector (20 features + 10 opener histogram,
same contract as E17 Step 1) shows significant split-half stability
(user-clustered bootstrap 95% CI lower bound > 0 AND permutation p < 0.01)?

Protocol:
  pass 1: stream 2023 Baby reviews, count total_words per user.
  pool   : users with total_words >= T_MAX, sample up to 500.
  pass 2: stream again, extract pooled users' review texts (cap ~3000 words
          per user), sentence-split + syntactic features via one spaCy pass.
  sweep  : for each threshold T in the grid, keep users with total_words>=T,
           take sentences covering the first T words, split half A/B
           (alternating), build vectors, user-clustered bootstrap +
           label-permutation test (same protocol as E17 Step 1).
  output : threshold table with n_users, mean delta-z, 95% CI, p, passed.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features_from_doc  # noqa: E402

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e18_min_words_sweep_fine.json"
SEED = 42
T_MAX = 5000          # pool users with >= this many review words
POOL_SIZE = 500
CAP_WORDS = 3000      # max words per user extracted for parsing
THRESHOLDS = [40, 60, 80, 100, 120, 150, 200]
N_PERM = 999
N_BOOT = 999
FEATS = FEATURES20


def build_vec(feat_list: list[np.ndarray], openers: list[int]) -> np.ndarray:
    fmean = np.mean(feat_list, axis=0) if feat_list else np.zeros(len(FEATS), dtype=np.float32)
    n = float(np.linalg.norm(fmean))
    if n > 1e-12:
        fmean = fmean / n
    ohist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
    if openers:
        for c in openers:
            ohist[c] += 1
        ohist = ohist / len(openers)
    return np.concatenate([fmean, ohist]).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # pass 1: word counts
    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            uid = d.get("user_id")
            t = d.get("text") or ""
            if uid:
                wc[uid] += len(t.split())
    print(f"users scanned: {len(wc)}", flush=True)
    pool = [u for u, w in wc.items() if w >= T_MAX]
    rng.shuffle(pool)
    pool = pool[:POOL_SIZE]
    print(f"pool users (words>={T_MAX}): {len(pool)}", flush=True)

    # pass 2: extract pooled users' review texts (cap words)
    texts: dict[str, list[str]] = defaultdict(list)
    poolset = set(pool)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            uid = d.get("user_id")
            if uid not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if not t:
                continue
            texts[uid].append(t)
    # truncate per user to CAP_WORDS
    for u in pool:
        acc, kept = 0, []
        for t in texts[u]:
            acc += len(t.split())
            kept.append(t)
            if acc >= CAP_WORDS:
                break
        texts[u] = kept
    print(f"users with texts: {len(texts)}", flush=True)

    # one spaCy pass: sentences -> features + openers, per user, in order
    nlp = load_spacy_model()
    flat = [(u, t[:2000]) for u in pool for t in texts[u]]
    sfeats: dict[str, list[np.ndarray]] = defaultdict(list)
    sopeners: dict[str, list[int]] = defaultdict(list)
    sent_words: dict[str, list[int]] = defaultdict(list)
    for (u, t), doc in zip(flat, nlp.pipe([t for _, t in flat], batch_size=256)):
        for sent in doc.sents:
            sent_words[u].append(len([tk for tk in sent if not tk.is_space]))
            toks = [tk for tk in sent if not tk.is_punct and not tk.is_space]
            if len(toks) < 3:
                continue
            try:
                ex = extract_clause_features_from_doc(sent, sent.text)
                sfeats[u].append(np.asarray([float(ex.get(k, 0.0)) for k in FEATS], dtype=np.float32))
            except Exception:
                continue
            sopeners[u].append(OPENER_CLASSES.index(opener_class_of(toks[0].text)))
    print(f"users with sentences: {len(sfeats)}", flush=True)

    def fisher_z(r: float) -> float:
        r = max(min(r, 0.9999), -0.9999)
        return 0.5 * float(np.log((1 + r) / (1 - r)))

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    results = []
    for T in THRESHOLDS:
        users = [u for u in pool if wc[u] >= T]
        vecA, vecB = {}, {}
        for u in users:
            sf, op, sw = sfeats.get(u, []), sopeners.get(u, []), sent_words.get(u, [])
            # window = sentences covering the first T words
            acc, n_sent = 0, 0
            for w in sw:
                acc += w
                n_sent += 1
                if acc >= T:
                    break
            if n_sent < 6:  # need >=3 sentences per half
                continue
            sf = sf[:n_sent]
            op = op[:n_sent]
            ia = slice(0, len(sf), 2)
            ib = slice(1, len(sf), 2)
            vecA[u] = build_vec(sf[ia], [op[i] for i in range(len(op)) if i % 2 == 0])
            vecB[u] = build_vec(sf[ib], [op[i] for i in range(len(op)) if i % 2 == 1])
        stable = sorted(vecA)
        if len(stable) < 30:
            results.append({"T": T, "n_users": len(stable), "passed": False,
                            "note": "too few stable users"})
            print(f"T={T}: only {len(stable)} users, skipped", flush=True)
            continue
        same = {u: fisher_z(rho(vecA[u], vecB[u])) for u in stable}
        diff = {}
        for u in stable:
            others = rng.choice([v for v in stable if v != u], min(3, len(stable) - 1), replace=False)
            diff[u] = [fisher_z(rho(vecA[u], vecB[o])) for o in others]
        obs = float(np.mean(list(same.values())) - np.mean([d for ds in diff.values() for d in ds]))

        rngb = np.random.default_rng(0)
        boot = []
        for _ in range(N_BOOT):
            ids = rngb.choice(stable, size=len(stable), replace=True)
            boot.append(np.mean([same[u] for u in ids]) - np.mean([d for u in ids for d in diff[u]]))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        poolz = np.concatenate([np.array([same[u] for u in stable]),
                                np.array([d for ds in diff.values() for d in ds])])
        n_same = len(stable)
        cnt = 0
        for _ in range(N_PERM):
            pv = rngb.permutation(poolz)
            if pv[:n_same].mean() - pv[n_same:].mean() >= obs:
                cnt += 1
        p = (cnt + 1) / (N_PERM + 1)
        passed = bool(lo > 0 and p < 0.01)
        results.append({"T": T, "n_users": len(stable), "mean_delta_z": round(obs, 4),
                        "ci95": [round(lo, 4), round(hi, 4)], "p": round(p, 4), "passed": passed})
        print(f"T={T}: users={len(stable)} dZ={obs:.3f} CI=[{lo:.3f},{hi:.3f}] p={p:.4f} passed={passed}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"question": "min per-user review words for significant style-vector stability",
               "protocol": "E17 Step-1 split-half stability (20 feats + 10 openers, alternating halves, "
                           "window = first T words; user-clustered bootstrap 95% CI lower > 0 and "
                           "label-permutation p < 0.01)",
               "data": str(REVIEWS), "seed": SEED, "pool_size": POOL_SIZE, "cap_words": CAP_WORDS,
               "results": results,
               "min_passed": next((r for r in results if r.get("passed")), None)},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
