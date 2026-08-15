#!/usr/bin/env python3
"""E18 stratified-sent: min-sentences-per-user threshold, selecting users by
their TRUE total review sentence count (bin around T), not windows.

Protocol (fixes selection bias of the windowed sweep):
  pass 1: total_words per user (streaming).
  bins  : [(T_lo, T_hi)] around candidate thresholds; sample up to 350
          users per bin (users with fewest words get priority to keep the
          'minimal amount' question honest within the bin).
  pass 2: one spaCy pass over all sampled users' full review texts.
  split : A/B halves by alternating sentence index (per-user order kept);
          same protocol: user-clustered bootstrap 95% CI + label permutation.
  split by review: not feasible for 40-80-word users (1 review); noted.
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
OUT = REPO_ROOT / "result" / "e18_min_sent_stratified.json"
SEED = 42
N_PER_BIN = 350
BINS = [(4, 8), (8, 12), (12, 20), (20, 32), (32, 50), (50, 80), (80, 120)]
MIN_SENTS_PER_HALF = 3
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

    # pass 1
    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if u:
                wc[u] += len(t.split())
    import re
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    ns: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = (d.get("text") or "").strip()
            if u and t:
                ns[u] += len([x for x in _SENT_SPLIT.split(t) if x.strip()])
    print(f"users scanned: {len(wc)} (words) + {len(ns)} (rough sents)", flush=True)

    # select users per bin (true total words within bin); fill smallest first
    bin_users: dict[tuple, list[str]] = {}
    for (lo, hi) in BINS:
        cands = [u for u, n in ns.items() if lo <= n < hi]
        rng.shuffle(cands)
        bin_users[(lo, hi)] = cands[:N_PER_BIN]
        print(f"bin [{lo},{hi}) sents: candidates={len(cands)} sampled={len(cands[:N_PER_BIN])}", flush=True)

    pool = sorted({u for v in bin_users.values() for u in v})
    poolset = set(pool)
    texts: dict[str, list[str]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if t:
                texts[u].append(t)
    print(f"pool users: {len(pool)} with texts", flush=True)

    # one spaCy pass
    nlp = load_spacy_model()
    flat = [(u, t[:2000]) for u in pool for t in texts[u]]
    sfeats: dict[str, list[np.ndarray]] = defaultdict(list)
    sopeners: dict[str, list[int]] = defaultdict(list)
    for (u, t), doc in zip(flat, nlp.pipe([t for _, t in flat], batch_size=256)):
        for sent in doc.sents:
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
    for (lo, hi) in BINS:
        T = (lo + hi) // 2
        users = [u for u in bin_users[(lo, hi)] if len(sfeats.get(u, [])) >= 2 * MIN_SENTS_PER_HALF]
        vecA, vecB = {}, {}
        for u in users:
            sf, op = sfeats[u], sopeners[u]
            ia = slice(0, len(sf), 2)
            ib = slice(1, len(sf), 2)
            vecA[u] = build_vec(sf[ia], [op[i] for i in range(len(op)) if i % 2 == 0])
            vecB[u] = build_vec(sf[ib], [op[i] for i in range(len(op)) if i % 2 == 1])
        stable = sorted(vecA)
        if len(stable) < 30:
            results.append({"T_center": T, "bin": [lo, hi], "n_users": len(stable),
                            "passed": False, "note": "too few users"})
            print(f"T~{T}: only {len(stable)} users, skipped", flush=True)
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
        lo_, hi_ = np.percentile(boot, [2.5, 97.5])
        poolz = np.concatenate([np.array([same[u] for u in stable]),
                                np.array([d for ds in diff.values() for d in ds])])
        n_same = len(stable)
        cnt = 0
        for _ in range(N_PERM):
            pv = rngb.permutation(poolz)
            if pv[:n_same].mean() - pv[n_same:].mean() >= obs:
                cnt += 1
        p = (cnt + 1) / (N_PERM + 1)
        passed = bool(lo_ > 0 and p < 0.01)
        results.append({"T_center": T, "bin": [lo, hi], "n_users": len(stable),
                        "mean_delta_z": round(obs, 4), "ci95": [round(lo_, 4), round(hi_, 4)],
                        "p": round(p, 4), "passed": passed})
        print(f"T~{T} (bin {lo}-{hi}): users={len(stable)} dZ={obs:.3f} CI=[{lo_:.3f},{hi_:.3f}] "
              f"p={p:.4f} passed={passed}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"question": "min TRUE per-user review SENTENCES for significant style-vector stability",
               "protocol": "stratified by true total words (bin [0.8T,1.2T]); all review text used; "
                           "alternating-sentence split-half; user-clustered bootstrap + label permutation",
               "data": str(REVIEWS), "seed": SEED, "n_per_bin": N_PER_BIN, "results": results,
               "min_passed": next((r for r in results if r.get("passed")), None)},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
