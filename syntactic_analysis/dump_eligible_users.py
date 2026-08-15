#!/usr/bin/env python3
"""Memory-light replay: dump eligible user IDs for v9/v10/v11.

Avoid per-user set of (asin, text) dedup keys (was blowing to ~16GB).
Since duplicates don't change asin-set cardinality or sentence counts,
skip dedup entirely. Eligible rule for v11 was:
  word>=100, distinct_asins>=2, total_sents>=30, n_tok>=5 per sent.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from parse_sentences_to_features import (
    CACHE_META_VERSION, load_features_cache, sent_key, split_sents,
)

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
CACHE_PATH = REPO_ROOT / "result" / "cache" / "per_sentence_features.jsonl.gz"
MIN_WORDS = 100
MIN_ASINS = 2
N_USERS_POOL = 5000

VERSIONS = [
    ("v9", 43),
    ("v10", 43),
    ("v11", 99),
]


def main() -> None:
    print("Loading sentence cache...", flush=True)
    sent_cache = load_features_cache(CACHE_PATH)
    print(f"  cache entries: {len(sent_cache)} (version={CACHE_META_VERSION})",
          flush=True)

    print("Pass 1: word counts + per-user asin set...", flush=True)
    wc: dict[str, int] = {}
    user_asins: dict[str, set] = {}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if not u:
                continue
            wc[u] = wc.get(u, 0) + len(t.split())
            t_strip = t.strip()
            if not t_strip:
                continue
            asin = d.get("parent_asin") or d.get("asin")
            user_asins.setdefault(u, set()).add(asin)
    print(f"  users: {len(wc)}; users with >=2 asins: "
          f"{sum(1 for s in user_asins.values() if len(s) >= MIN_ASINS)}",
          flush=True)

    # Pre-filter users (the only ones we track for sent counts).
    prefilter = {u for u, s in user_asins.items()
                 if len(s) >= MIN_ASINS and wc.get(u, 0) >= MIN_WORDS}
    print(f"  prefilter set size: {len(prefilter)}", flush=True)

    print("Pass 2: per-user sentence counts via cache...", flush=True)
    user_total_sents: dict[str, int] = {}
    user_asin_sents: dict[str, set] = {}
    n_lines = 0
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = (d.get("text") or "").strip()
            if not u or not t:
                continue
            if u not in prefilter:
                continue
            asin = d.get("parent_asin") or d.get("asin")
            for s in split_sents(t):
                sf = sent_cache.get(sent_key(s))
                if sf is None:
                    continue
                if sf["n_tok"] < 5:
                    continue
                user_total_sents[u] = user_total_sents.get(u, 0) + 1
                user_asin_sents.setdefault(u, set()).add(asin)
    print(f"  pass2 done ({n_lines} lines scanned)", flush=True)

    for label, cand_seed in VERSIONS:
        rng = np.random.default_rng(cand_seed)
        cands = [u for u, w in wc.items() if w >= MIN_WORDS]
        rng.shuffle(cands)
        cands = cands[:N_USERS_POOL * 4]

        eligible: list[str] = []
        for u in cands:
            if user_total_sents.get(u, 0) < 30:
                continue
            if len(user_asin_sents.get(u, set())) < 2:
                continue
            eligible.append(u)
        out_path = REPO_ROOT / "result" / f"e21_{label}_eligible_users.json"
        with open(out_path, "w") as f:
            json.dump({
                "label": label,
                "seed": cand_seed,
                "eligible_count": len(eligible),
                "eligible_users": sorted(eligible),
            }, f, indent=1)
        print(f"  {label} (seed={cand_seed}): eligible={len(eligible)} → "
              f"{out_path.name}", flush=True)


if __name__ == "__main__":
    main()