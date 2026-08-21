#!/usr/bin/env python3
"""Phase 14.Q9-2: First-person narrative generation.

Generate queries with first-person prompt + exemplar selection favoring
sentences with "I/my/we/our" pronouns (matches narrative review style).

Conditions:
  - Q_NarrativeFirstPerson: prompt asks for personal shopping note + 4 first-person exemplars
  - Q_NarrativeFirstPerson_s1..s4: 4 different exemplar subsets (median fp / +5 fp / -3 fp / random)

Total: 30 pairs × 5 conds × K=8 = 1200 queries.
"""
from __future__ import annotations
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_10 = OUT_DIR / "phase10_pairs_1000.jsonl"
IN_USER_SENTS = Path("/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl")

OUT_USER_CONDS = OUT_DIR / "phase14_q9_user_conditions.json"
OUT_GEN_PROMPTS = OUT_DIR / "phase14_q9_generation_prompts.jsonl"
OUT_META = OUT_DIR / "phase14_q9_prep_meta.json"

MAX_SENT_LEN = 30
MIN_SENT_LEN = 5
N_EXEMPLARS = 4
N_CONDS = 5  # 1 base + 4 subsets
SEED = 42

# First-person pronouns (expanded for narrative detection)
FP_PATTERN = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|me|myself|we|we're|we've|we'd|our|us)\b", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_person_score(text: str) -> int:
    """Count first-person pronouns (capped at 5)."""
    return min(5, len(FP_PATTERN.findall(text)))


def select_exemplars_fp(user_id, user_sents, n=N_EXEMPLARS):
    """Return 4 different exemplar subsets, all favoring first-person sentences.

    Strategies:
      - s0 (default for Q_NarrativeFirstPerson): top-n by fp_score, ties broken by length
      - s1: median-fp (closest to median fp_score)
      - s2: top-n by fp_score + longest
      - s3: top-n by fp_score + shortest
      - s4: random from fp-positive candidates
    """
    sents = user_sents.get(user_id, [])
    if not sents:
        return [[] for _ in range(N_CONDS)]

    candidates = []
    for sent, n_words, _asin in sents:
        s = sent.strip()
        if MIN_SENT_LEN <= n_words <= MAX_SENT_LEN:
            fp = first_person_score(s)
            candidates.append((n_words, fp, s))

    if len(candidates) < n:
        candidates = [(n_words, first_person_score(s), s) for s, n_words, _ in sents if n_words > 0]
    if len(candidates) < n:
        return [[c[2] for c in candidates] for _ in range(N_CONDS)]

    # Score: prefer fp > 0; among those, by length diversity
    fp_pos = [c for c in candidates if c[1] > 0]
    if len(fp_pos) < n:
        # Fall back to top-n by fp_score then by length
        candidates.sort(key=lambda x: (-x[1], abs(x[0] - 12)))
    else:
        candidates = fp_pos

    # Subset 0 (default): top-n by fp_score
    c0 = sorted(candidates, key=lambda x: (-x[1], abs(x[0] - 12)))
    subset0 = [c[2] for c in c0[:n]]

    # Subset 1: median-fp, longest
    c1 = sorted(candidates, key=lambda x: (-x[1], -x[0]))
    subset1 = [c[2] for c in c1[:n]]

    # Subset 2: top-fp + shortest
    c2 = sorted(candidates, key=lambda x: (-x[1], x[0]))
    subset2 = [c[2] for c in c2[:n]]

    # Subset 3: random from fp-positive
    rng = random.Random(hash(user_id) & 0xFFFFFFFF)
    cand_pool = list(candidates)
    rng.shuffle(cand_pool)
    subset3 = [c[2] for c in cand_pool[:n]]

    # Subset 4: alternating by fp-score rank
    c4 = sorted(candidates, key=lambda x: (-x[1], abs(x[0] - 12)))
    subset4 = [c[2] for c in c4[0::2][:n]]

    return [subset0, subset1, subset2, subset3, subset4]


def build_prompt(attrs_str: str, exemplars: list, cond_idx: int) -> str:
    """Build first-person narrative prompt."""
    exem_str = "\n".join([f"Example {i+1}: {e}" for i, e in enumerate(exemplars)])

    if cond_idx == 0:
        instructions = (
            "Write a short personal shopping note from this user's perspective. "
            "Use first-person voice ('I want...', 'for my kids...', 'I love...') "
            "and include every product attribute. "
            "Match the user's expression style from the examples but do not copy them."
        )
    else:
        instructions = (
            "Write a short personal shopping note from this user's perspective. "
            "Use first-person voice ('I want...', 'for my kids...', 'I love...') "
            "with a personal context (who the product is for, why they need it). "
            "Include every product attribute. "
            "Match the user's expression style from the examples but do not copy them."
        )

    return (
        f"Product attributes:\n{attrs_str}\n\n"
        f"User's typical review style (first-person narrative):\n{exem_str}\n\n"
        f"Task:\n{instructions}"
    )


def main() -> None:
    log("=" * 70)
    log(f"Phase 14.Q9-2: First-person narrative generation ({N_CONDS} conds × K=8)")
    log("=" * 70)

    log("[1] Loading pairs ...")
    p10_pairs = []
    with IN_PAIRS_10.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p10_pairs.append(json.loads(line))

    attrs_map = {}
    for r in p10_pairs:
        attrs_map[(r["user_id"], r["asin"])] = r.get("attrs", {})

    log("[2] Loading phase14_b pairs ...")
    b_pairs_set = set()
    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            b_pairs_set.add((r["user_id"], r["asin"]))
    log(f"  pairs: {len(b_pairs_set)}")

    log("[3] Loading user_sents ...")
    with IN_USER_SENTS.open("rb") as f:
        user_sents = pickle.load(f)
    log(f"  users: {len(user_sents)}")

    log(f"[4] Building first-person exemplar subsets for {len(b_pairs_set)} pairs ...")
    per_pair_conds = {}
    n_with_fp = 0
    for (uid, asin) in sorted(b_pairs_set):
        subsets = select_exemplars_fp(uid, user_sents, N_EXEMPLARS)
        # Count how many exemplars are first-person
        n_fp_in_subset = sum(1 for e in subsets[0] if first_person_score(e) > 0)
        if n_fp_in_subset > 0:
            n_with_fp += 1
        per_pair_conds[(uid, asin)] = {
            "exemplars_subsets": subsets,
            "n_fp_in_default_subset": n_fp_in_subset,
        }

    log(f"  pairs with >=1 first-person exemplar: {n_with_fp}/{len(per_pair_conds)}")

    log("[5] Building generation prompts (5 conds × 30 pairs) ...")
    all_prompts = []
    pair_keys = sorted(per_pair_conds.keys())

    for (uid, asin) in pair_keys:
        rec = per_pair_conds[(uid, asin)]
        attrs = attrs_map.get((uid, asin), {})
        attrs_str = "\n".join([f"{k}: {v}" for k, v in attrs.items()])

        for cond_idx, exemplars in enumerate(rec["exemplars_subsets"]):
            if not exemplars:
                continue
            cond_label = "Q_NarrativeFirstPerson" if cond_idx == 0 else f"Q_NarrativeFirstPerson_s{cond_idx}"
            prompt = build_prompt(attrs_str, exemplars, cond_idx)
            all_prompts.append({
                "user_id": uid,
                "asin": asin,
                "condition": cond_label,
                "prompt": prompt,
            })

    log(f"  total prompts: {len(all_prompts)}")

    log("\n=== Sample inspection ===")
    sample_key = pair_keys[0]
    sample_rec = per_pair_conds[sample_key]
    log(f"Pair: {sample_key[0][:10]}... / {sample_key[1]}")
    log(f"  default subset ({sample_rec['n_fp_in_default_subset']} first-person):")
    for e in sample_rec["exemplars_subsets"][0][:2]:
        log(f"    - {repr(e)[:100]}")

    per_pair_dict = {
        f"{uid}|{asin}": v for (uid, asin), v in per_pair_conds.items()
    }
    OUT_USER_CONDS.write_text(json.dumps(per_pair_dict, indent=2, ensure_ascii=False))
    log(f"\n  user_conds → {OUT_USER_CONDS}")

    with OUT_GEN_PROMPTS.open("w") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"  gen_prompts → {OUT_GEN_PROMPTS} ({len(all_prompts)} prompts)")

    meta = {
        "phase": "14.Q9-2",
        "method": f"First-person narrative generation ({N_CONDS} conds × K=8)",
        "n_pairs": len(pair_keys),
        "n_conds": N_CONDS,
        "n_exemplars_per_subset": N_EXEMPLARS,
        "subset_strategies": [
            "default-top-fp",
            "top-fp-longest",
            "top-fp-shortest",
            "random-fp",
            "alt-fp-rank",
        ],
        "n_pairs_with_fp": n_with_fp,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q9-2 PREP COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()