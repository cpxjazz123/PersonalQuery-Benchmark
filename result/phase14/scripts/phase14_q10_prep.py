#!/usr/bin/env python3
"""Phase 14.Q10: Enriched user-conditioned prompt generation.

Combine multiple diagnostic insights into single prompt:
  - First-person narrative voice
  - Length match (user avg sentence length)
  - Top-3 distinguishing style features (20d z-score)
  - 4 first-person exemplars (median length)

Conditions:
  - Q_EnrichedUserProfile: full enriched prompt (3 subsets)
  - Q_EnrichedUserProfile_s1: top-fp + longest
  - Q_EnrichedUserProfile_s2: top-fp + shortest
  - Q_EnrichedUserProfile_s3: random fp

Total: 30 pairs × 4 conds × K=8 = 960 queries.
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

OUT_USER_CONDS = OUT_DIR / "phase14_q10_user_conditions.json"
OUT_GEN_PROMPTS = OUT_DIR / "phase14_q10_generation_prompts.jsonl"
OUT_META = OUT_DIR / "phase14_q10_prep_meta.json"

FEATURE_20_NAMES = [
    "clause_rate", "acl_rate", "advcl_rate", "ccomp_rate", "relcl_rate",
    "modifier_density", "coordination_density", "mean_dep_distance", "depth_variance",
    "passive_rate", "interrogative_rate", "opener_noun", "opener_pron", "opener_verb",
    "senttype_complex", "nest_max", "nest_mean", "nest_ge2", "depth_eq3", "punct_period",
]

FEATURE_NATURAL = {
    "clause_rate": ("use simpler sentences with fewer clauses", "use richer clause structures with multiple subordinate clauses"),
    "acl_rate": ("avoid adverbial clauses", "use adverbial clauses (when/because/although)"),
    "advcl_rate": ("avoid adverbial clauses", "use adverbial clauses (when/because/although)"),
    "ccomp_rate": ("avoid clausal complements", "use clausal complements (that/whether)"),
    "relcl_rate": ("avoid relative clauses", "use relative clauses (which/that/who)"),
    "modifier_density": ("use fewer adjectives and adverbs", "use descriptive adjectives and adverbs"),
    "coordination_density": ("avoid coordinating conjunctions", "use coordinating conjunctions (and/but/or)"),
    "mean_dep_distance": ("keep words close together", "use longer dependency chains"),
    "depth_variance": ("use uniform sentence depth", "vary sentence depth more"),
    "passive_rate": ("prefer active voice", "use passive voice constructions"),
    "interrogative_rate": ("write declarative statements", "use interrogative questions"),
    "opener_noun": ("open with pronouns or verbs", "open sentences with nouns"),
    "opener_pron": ("avoid opening with pronouns", "open sentences with pronouns"),
    "opener_verb": ("avoid opening with verbs", "open sentences with verbs"),
    "senttype_complex": ("use simple sentences", "use complex sentences"),
    "nest_max": ("keep nesting shallow", "use deeper clause nesting"),
    "nest_mean": ("keep nesting shallow", "use deeper clause nesting"),
    "nest_ge2": ("keep nesting shallow", "use deeper clause nesting"),
    "depth_eq3": ("avoid deep syntactic structures", "use deeper syntactic structures"),
    "punct_period": ("avoid periods, use shorter forms", "end sentences with periods"),
}

Z_THRESHOLD = 0.5
N_TOP_FEATURES = 3
MAX_SENT_LEN = 30
MIN_SENT_LEN = 5
N_EXEMPLARS = 4
N_CONDS = 4
SEED = 42

FP_PATTERN = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|me|myself|we|we're|we've|we'd|our|us)\b", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_20d(desc: str) -> list[float]:
    m = re.search(r"20-d syntactic mean: \[([\d.,\s\-eE]+)\]", desc)
    if not m:
        return []
    nums = [float(x.strip()) for x in m.group(1).split(",")]
    return nums[:20]


def first_person_score(text: str) -> int:
    return min(5, len(FP_PATTERN.findall(text)))


def get_user_avg_len(user_id: str, user_sents: dict) -> float:
    """Get average sentence word count for the user."""
    sents = user_sents.get(user_id, [])
    if not sents:
        return 12.0
    lens = [n_words for _, n_words, _ in sents if MIN_SENT_LEN <= n_words <= MAX_SENT_LEN]
    if not lens:
        return 12.0
    return float(np.median(lens))


def get_top_features(vec20, pop_mean, pop_std, n=N_TOP_FEATURES, threshold=Z_THRESHOLD):
    if len(vec20) != 20:
        return []
    z_scores = (np.array(vec20) - pop_mean) / (pop_std + 1e-9)
    abs_z = np.abs(z_scores)
    top_idx = np.argsort(-abs_z)[:n]
    features = []
    for idx in top_idx:
        z = z_scores[idx]
        if abs(z) < threshold:
            continue
        feat_name = FEATURE_20_NAMES[idx]
        low_text, high_text = FEATURE_NATURAL[feat_name]
        direction = high_text if z > 0 else low_text
        magnitude = "strongly" if abs(z) > 1.0 else "moderately"
        features.append((feat_name, direction, magnitude))
    return features[:n]


def select_exemplars_fp(user_id, user_sents, n=N_EXEMPLARS):
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

    fp_pos = [c for c in candidates if c[1] > 0]
    if len(fp_pos) < n:
        candidates.sort(key=lambda x: (-x[1], abs(x[0] - 12)))
    else:
        candidates = fp_pos

    # Subset 0 (default): top-fp
    c0 = sorted(candidates, key=lambda x: (-x[1], abs(x[0] - 12)))
    subset0 = [c[2] for c in c0[:n]]

    # Subset 1: top-fp + longest
    c1 = sorted(candidates, key=lambda x: (-x[1], -x[0]))
    subset1 = [c[2] for c in c1[:n]]

    # Subset 2: top-fp + shortest
    c2 = sorted(candidates, key=lambda x: (-x[1], x[0]))
    subset2 = [c[2] for c in c2[:n]]

    # Subset 3: random fp
    rng = random.Random(hash(user_id) & 0xFFFFFFFF)
    cand_pool = list(candidates)
    rng.shuffle(cand_pool)
    subset3 = [c[2] for c in cand_pool[:n]]

    return [subset0, subset1, subset2, subset3]


def build_enriched_prompt(attrs_str, exemplars, features, avg_len):
    """Build enriched user-conditioned prompt combining FP + length + features."""
    exem_str = "\n".join([f"Example {i+1}: {e}" for i, e in enumerate(exemplars)])
    if features:
        feat_lines = []
        for i, (feat_name, direction, magnitude) in enumerate(features):
            feat_lines.append(f"  {i+1}. {direction} ({magnitude} characteristic)")
        feat_str = "\n".join(feat_lines)
    else:
        feat_str = "  (no specific distinguishing features)"

    len_target = int(round(avg_len))
    return (
        f"Product attributes:\n{attrs_str}\n\n"
        f"User's typical review style (first-person narrative):\n{exem_str}\n\n"
        f"This user's distinguishing style features:\n{feat_str}\n\n"
        f"Task: Write a personal shopping note (about {len_target} words, 1-2 sentences) "
        f"from this user's perspective. Use first-person voice ('I want...', 'for my kids...', 'I love...'). "
        f"Match the user's distinguishing style features above. Include every product attribute. "
        f"Match the user's expression style from the examples but do not copy them."
    )


def main() -> None:
    log("=" * 70)
    log(f"Phase 14.Q10: Enriched user-conditioned prompt generation ({N_CONDS} conds × K=8)")
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
    style_desc_map = {}
    for r in p10_pairs:
        attrs_map[(r["user_id"], r["asin"])] = r.get("attrs", {})
        style_desc_map[(r["user_id"], r["asin"])] = r.get("target_style_desc", "")

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

    log("[4] Computing population mean/std from p10 ...")
    vec_20_all = []
    for r in p10_pairs:
        vec = parse_20d(r.get("target_style_desc", ""))
        if len(vec) == 20:
            vec_20_all.append(vec)
    vec_20_all_np = np.array(vec_20_all)
    pop_mean = vec_20_all_np.mean(axis=0)
    pop_std = vec_20_all_np.std(axis=0)

    log(f"[5] Building enriched conds for {len(b_pairs_set)} pairs ...")
    per_pair_conds = {}
    for (uid, asin) in sorted(b_pairs_set):
        desc = style_desc_map.get((uid, asin), "")
        vec = parse_20d(desc)
        features = get_top_features(vec, pop_mean, pop_std)
        subsets = select_exemplars_fp(uid, user_sents, N_EXEMPLARS)
        avg_len = get_user_avg_len(uid, user_sents)

        per_pair_conds[(uid, asin)] = {
            "features": features,
            "exemplars_subsets": subsets,
            "avg_len": avg_len,
        }

    log(f"  pairs with features: {sum(1 for v in per_pair_conds.values() if v['features'])}/{len(per_pair_conds)}")
    log(f"  pairs with >=1 fp exemplar: {sum(1 for v in per_pair_conds.values() if v['exemplars_subsets'][0] and any(first_person_score(e) > 0 for e in v['exemplars_subsets'][0]))}/{len(per_pair_conds)}")

    log("[6] Building generation prompts ...")
    all_prompts = []
    pair_keys = sorted(per_pair_conds.keys())

    for (uid, asin) in pair_keys:
        rec = per_pair_conds[(uid, asin)]
        attrs = attrs_map.get((uid, asin), {})
        attrs_str = "\n".join([f"{k}: {v}" for k, v in attrs.items()])

        for cond_idx, exemplars in enumerate(rec["exemplars_subsets"]):
            if not exemplars:
                continue
            cond_label = "Q_EnrichedUserProfile" if cond_idx == 0 else f"Q_EnrichedUserProfile_s{cond_idx}"
            prompt = build_enriched_prompt(attrs_str, exemplars, rec["features"], rec["avg_len"])
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
    log(f"  features: {sample_rec['features']}")
    log(f"  avg_len: {sample_rec['avg_len']:.1f}")
    log(f"  default exemplar ({sum(1 for e in sample_rec['exemplars_subsets'][0] if first_person_score(e) > 0)} fp):")
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
        "phase": "14.Q10",
        "method": f"Enriched user-conditioned prompt (FP + length + features + exemplars) ({N_CONDS} conds × K=8)",
        "n_pairs": len(pair_keys),
        "n_conds": N_CONDS,
        "n_top_features": N_TOP_FEATURES,
        "z_threshold": Z_THRESHOLD,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q10 PREP COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()