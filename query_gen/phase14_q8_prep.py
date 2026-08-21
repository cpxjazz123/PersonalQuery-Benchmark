#!/usr/bin/env python3
"""Phase 14.Q-8: 8 exemplar subsets (4 from Q-7 + 4 new strategies).

New strategies:
  - s5: longest 4 (max word count)
  - s6: shortest 4 (min word count)
  - s7: alternating odd (1st, 3rd, 5th, 7th closest to median)
  - s8: alternating even (2nd, 4th, 6th, 8th closest to median)

Total: 30 pairs × 8 subsets × K=8 = 1920 queries.
"""
from __future__ import annotations
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_10 = OUT_DIR / "phase10_pairs_1000.jsonl"
IN_USER_SENTS = Path("/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl")

OUT_USER_CONDS = OUT_DIR / "phase14_q8_user_conditions.json"
OUT_GEN_PROMPTS = OUT_DIR / "phase14_q8_generation_prompts.jsonl"
OUT_META = OUT_DIR / "phase14_q8_prep_meta.json"

FEATURE_20_NAMES = [
    "clause_rate", "acl_rate", "advcl_rate", "ccomp_rate", "relcl_rate",
    "modifier_density", "coordination_density", "mean_dep_distance", "depth_variance",
    "passive_rate", "interrogative_rate", "opener_noun", "opener_pron", "opener_verb",
    "senttype_complex", "nest_max", "nest_mean", "nest_ge2", "depth_eq3", "punct_period",
]

CONDITION_TEMPLATES = {
    "clause_rate":        ("use simpler sentences with fewer clauses", "use richer clause structures with subordinate clauses"),
    "acl_rate":           ("avoid adverbial clauses", "include adverbial clauses (when/because/although)"),
    "advcl_rate":         ("avoid adverbial clauses", "include adverbial clauses (when/because/although)"),
    "ccomp_rate":         ("avoid clausal complements", "use clausal complements (that/whether)"),
    "relcl_rate":         ("avoid relative clauses", "use relative clauses (which/that/who)"),
    "modifier_density":   ("use fewer adjectives and adverbs", "use more descriptive adjectives and adverbs"),
    "coordination_density":("avoid coordinating conjunctions", "use more coordinating conjunctions (and/but/or)"),
    "mean_dep_distance":  ("keep words close together", "use longer dependency chains"),
    "depth_variance":     ("use uniform sentence depth", "vary sentence depth more"),
    "passive_rate":       ("prefer active voice", "use passive voice constructions"),
    "interrogative_rate": ("write declarative statements", "use interrogative questions"),
    "opener_noun":        ("open with pronouns or verbs", "open sentences with nouns"),
    "opener_pron":        ("avoid opening with pronouns", "open sentences with pronouns"),
    "opener_verb":        ("avoid opening with verbs", "open sentences with verbs"),
    "senttype_complex":   ("use simple sentences", "use complex sentences"),
    "nest_max":           ("keep nesting shallow", "use deeper clause nesting"),
    "nest_mean":          ("keep nesting shallow", "use deeper clause nesting"),
    "nest_ge2":           ("keep nesting shallow", "use deeper clause nesting"),
    "depth_eq3":          ("avoid deep syntactic structures", "use deeper syntactic structures"),
    "punct_period":       ("avoid periods, use shorter forms", "end sentences with periods"),
}

N_TOP_CONDITIONS = 5
Z_THRESHOLD = 0.5
MAX_SENT_LEN = 30
MIN_SENT_LEN = 5
N_EXEMPLARS = 4
N_SUBSETS = 8
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_20d(desc: str) -> list[float]:
    m = re.search(r"20-d syntactic mean: \[([\d.,\s\-eE]+)\]", desc)
    if not m:
        return []
    nums = [float(x.strip()) for x in m.group(1).split(",")]
    return nums[:20]


def build_dynamic_conditions(vec20, pop_mean, pop_std):
    if len(vec20) != 20:
        return []
    z_scores = (np.array(vec20) - pop_mean) / (pop_std + 1e-9)
    abs_z = np.abs(z_scores)
    top_idx = np.argsort(-abs_z)[:N_TOP_CONDITIONS]
    conditions = []
    for idx in top_idx:
        z = z_scores[idx]
        if abs(z) < Z_THRESHOLD:
            continue
        feat_name = FEATURE_20_NAMES[idx]
        low_text, high_text = CONDITION_TEMPLATES[feat_name]
        if z > 0:
            conditions.append(f"  - {high_text}")
        else:
            conditions.append(f"  - {low_text}")
    return conditions


def select_exemplars_8subsets(user_id, user_sents, n=N_EXEMPLARS):
    """Return 8 different 4-exemplar subsets.

    Strategies:
      - s1: closest to median
      - s2: closest to median+5
      - s3: closest to median-3
      - s4: random (seed=user_id)
      - s5: longest 4 (max word count)
      - s6: shortest 4 (min word count)
      - s7: alternating odd (1st, 3rd, 5th, 7th closest to median)
      - s8: alternating even (2nd, 4th, 6th, 8th closest to median)
    """
    sents = user_sents.get(user_id, [])
    if not sents:
        return [[] for _ in range(8)]

    candidates = []
    for sent, n_words, _asin in sents:
        s = sent.strip()
        if MIN_SENT_LEN <= n_words <= MAX_SENT_LEN:
            candidates.append((n_words, s))
    if len(candidates) < n:
        candidates = [(n_words, s) for s, n_words, _ in sents if n_words > 0]

    if len(candidates) < n:
        return [[c[1] for c in candidates] for _ in range(8)]

    word_counts = sorted(c[0] for c in candidates)
    median_wc = word_counts[len(word_counts) // 2]

    # s1: closest to median
    cs1 = sorted(candidates, key=lambda x: abs(x[0] - median_wc))
    subset1 = [c[1] for c in cs1[:n]]

    # s2: closest to median+5
    cs2 = sorted(candidates, key=lambda x: abs(x[0] - (median_wc + 5)))
    subset2 = [c[1] for c in cs2[:n]]

    # s3: closest to median-3
    cs3 = sorted(candidates, key=lambda x: abs(x[0] - (median_wc - 3)))
    subset3 = [c[1] for c in cs3[:n]]

    # s4: random
    rng = random.Random(hash(user_id) & 0xFFFFFFFF)
    cand_pool = list(candidates)
    rng.shuffle(cand_pool)
    subset4 = [c[1] for c in cand_pool[:n]]

    # s5: longest 4
    cs5 = sorted(candidates, key=lambda x: -x[0])
    subset5 = [c[1] for c in cs5[:n]]

    # s6: shortest 4
    cs6 = sorted(candidates, key=lambda x: x[0])
    subset6 = [c[1] for c in cs6[:n]]

    # s7: alternating odd (1st, 3rd, 5th, 7th closest to median)
    cs7 = sorted(candidates, key=lambda x: abs(x[0] - median_wc))
    subset7 = [c[1] for c in cs7[0::2][:n]]  # 0, 2, 4, 6 indices

    # s8: alternating even (2nd, 4th, 6th, 8th closest to median)
    cs8 = sorted(candidates, key=lambda x: abs(x[0] - median_wc))
    subset8 = [c[1] for c in cs8[1::2][:n]]  # 1, 3, 5, 7 indices

    return [subset1, subset2, subset3, subset4, subset5, subset6, subset7, subset8]


def main() -> None:
    log("=" * 70)
    log(f"Phase 14.Q-8: 8 exemplar subsets per user")
    log("=" * 70)

    log("[1] Loading phase10_pairs_1000 ...")
    p10_pairs = []
    with IN_PAIRS_10.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p10_pairs.append(json.loads(line))
    log(f"  total: {len(p10_pairs)}")

    p10_map = {}
    for r in p10_pairs:
        key = (r["user_id"], r["asin"])
        p10_map[key] = r.get("target_style_desc", "")

    vec_20_all = []
    for r in p10_pairs:
        vec = parse_20d(r.get("target_style_desc", ""))
        if len(vec) == 20:
            vec_20_all.append(vec)
    vec_20_all_np = np.array(vec_20_all)
    pop_mean = vec_20_all_np.mean(axis=0)
    pop_std = vec_20_all_np.std(axis=0)

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

    log(f"[4] Building per-pair conditions + 8 exemplar subsets ...")
    per_pair_conds = {}
    for (uid, asin) in sorted(b_pairs_set):
        desc = p10_map.get((uid, asin), "")
        vec = parse_20d(desc)
        if len(vec) != 20:
            continue
        conds = build_dynamic_conditions(vec, pop_mean, pop_std)
        subsets = select_exemplars_8subsets(uid, user_sents, N_EXEMPLARS)

        per_pair_conds[(uid, asin)] = {
            "vec_20": vec,
            "conditions_text": conds,
            "exemplars_subsets": subsets,
        }

    n_with_conds = sum(1 for v in per_pair_conds.values() if v["conditions_text"])
    n_with_exem = sum(1 for v in per_pair_conds.values() if v["exemplars_subsets"][0])
    log(f"  pairs with conditions: {n_with_conds}/{len(per_pair_conds)}")
    log(f"  pairs with exemplars (s1): {n_with_exem}/{len(per_pair_conds)}")

    log("[5] Building generation prompts (8 conds × 30 pairs) ...")
    all_prompts = []
    pair_keys = sorted(per_pair_conds.keys())

    attrs_map = {}
    for r in p10_pairs:
        key = (r["user_id"], r["asin"])
        attrs_map[key] = r.get("attrs", {})

    for (uid, asin) in pair_keys:
        rec = per_pair_conds[(uid, asin)]
        attrs = attrs_map.get((uid, asin), {})
        attrs_str = "\n".join([f"{k}: {v}" for k, v in attrs.items()])
        conds_text = "\n".join(rec["conditions_text"]) if rec["conditions_text"] else "  - (no specific conditions)"

        for subset_idx, exemplars in enumerate(rec["exemplars_subsets"]):
            if not exemplars:
                continue
            exem_str = "\n".join([f"Example {i+1}: {e}" for i, e in enumerate(exemplars)])
            prompt = (
                f"Product attributes:\n{attrs_str}\n\n"
                f"Style examples from this user (subset {subset_idx + 1}):\n{exem_str}\n\n"
                f"Style conditions to match this user's typical syntax:\n{conds_text}\n\n"
                f"Write a short shopping query that includes every attribute. "
                f"Match the user's expression style from the examples but do not copy them."
            )
            all_prompts.append({
                "user_id": uid,
                "asin": asin,
                "condition": f"Q_DynamicExemplar4_s{subset_idx + 1}",
                "prompt": prompt,
            })

    log(f"  total prompts: {len(all_prompts)}")

    log("\n=== Sample inspection ===")
    sample_key = pair_keys[0]
    sample_rec = per_pair_conds[sample_key]
    log(f"Pair: {sample_key[0][:10]}... / {sample_key[1]}")
    for si, exem in enumerate(sample_rec["exemplars_subsets"]):
        log(f"  subset {si+1} ({len(exem)} exemplars):")
        for e in exem[:2]:
            log(f"    - {repr(e)[:80]}")

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
        "phase": "14.Q-8",
        "method": f"8 different exemplar subsets × {N_EXEMPLARS} exemplars",
        "n_pairs": len(pair_keys),
        "n_subsets": N_SUBSETS,
        "n_exemplars_per_subset": N_EXEMPLARS,
        "subset_strategies": [
            "median", "median+5", "median-3", "random",
            "longest", "shortest", "alt odd", "alt even",
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-8 PREP COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
