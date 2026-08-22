#!/usr/bin/env python3
"""Phase 14.Q-1: Build per-user dynamic conditions + exemplars for 30 dev pairs.

For each (user, asin) pair in phase14_b (30 pairs):
  1. Parse 20-d syntactic vector from phase10_pairs_1000.target_style_desc
  2. Compute z-scores against population mean (from 876 pairs)
  3. Map top-5 most distinctive dims to text conditions
  4. Select 2 exemplars from user_sents_cache2 (most distinctive sentences)

Outputs:
  - phase14_q_user_conditions.json (per-pair conditions + exemplars)
  - phase14_q_generation_prompts.jsonl (ready-to-use prompts for K=8 generation)
"""
from __future__ import annotations

import json
import pickle
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

OUT_USER_CONDS = OUT_DIR / "phase14_q_user_conditions.json"
OUT_GEN_PROMPTS = OUT_DIR / "phase14_q_generation_prompts.jsonl"
OUT_META = OUT_DIR / "phase14_q_prep_meta.json"

# 20-d feature names (heuristic mapping based on e22_t3 features)
# Order: roughly top-20 most-discriminating syntactic features after
# dropping punctuation and length-correlated features
FEATURE_20_NAMES = [
    "clause_rate",        # 0: fraction of words in subordinate clauses
    "acl_rate",           # 1: adverbial clause rate
    "advcl_rate",         # 2: adverbial clause rate (variant)
    "ccomp_rate",         # 3: clausal complement rate
    "relcl_rate",         # 4: relative clause rate
    "modifier_density",   # 5: adjective/adverb density
    "coordination_density",  # 6: coordinating conjunction density
    "mean_dep_distance",  # 7: avg dependency distance
    "depth_variance",     # 8: variance of parse depth
    "passive_rate",       # 9: passive voice rate
    "interrogative_rate", # 10: question rate
    "opener_noun",        # 11: sentence opener is noun
    "opener_pron",        # 12: sentence opener is pronoun
    "opener_verb",        # 13: sentence opener is verb
    "senttype_complex",   # 14: complex sentence type rate
    "nest_max",           # 15: max clause nesting depth
    "nest_mean",          # 16: mean clause nesting depth
    "nest_ge2",           # 17: fraction of clauses with depth >= 2
    "depth_eq3",          # 18: fraction of tokens at depth 3
    "punct_period",       # 19: period rate (less in conversational)
]

# Map feature name -> (low_z_text_condition, high_z_text_condition)
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

N_EXEMPLARS = 2
N_TOP_CONDITIONS = 5
Z_THRESHOLD = 0.5  # only flag dimensions with |z| > 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_20d(desc: str) -> list[float]:
    """Parse 20-d mean from target_style_desc string."""
    m = re.search(r"20-d syntactic mean: \[([\d.,\s\-eE]+)\]", desc)
    if not m:
        return []
    nums = [float(x.strip()) for x in m.group(1).split(",")]
    return nums[:20]


def build_dynamic_conditions(vec20: list[float], pop_mean: np.ndarray, pop_std: np.ndarray) -> list[str]:
    """Build text conditions from z-scores."""
    if len(vec20) != 20:
        return []
    z_scores = (np.array(vec20) - pop_mean) / (pop_std + 1e-9)
    # Sort by |z|, take top 5
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


def select_exemplars(user_id: str, user_sents: dict, n: int = N_EXEMPLARS) -> list[str]:
    """Select n representative sentences from user's history.

    Strategy:
      - Filter sentences with 5-30 words (typical query length)
      - Skip very short (< 5 words) and very long (> 30 words) sentences
      - Pick 2 with sentence length closest to median
    """
    sents = user_sents.get(user_id, [])
    if not sents:
        return []

    candidates = []
    for sent, n_words, _asin in sents:
        s = sent.strip()
        if 5 <= n_words <= 30:
            candidates.append((n_words, s))
    if not candidates:
        # Fallback: take any 2 sentences with words > 0
        candidates = [(n_words, s) for s, n_words, _ in sents if n_words > 0]

    if len(candidates) < n:
        return [c[1] for c in candidates]

    # Sort by word count, pick median ± 1
    word_counts = sorted(c[0] for c in candidates)
    median_idx = len(word_counts) // 2
    median_wc = word_counts[median_idx]

    # Pick 2 candidates closest to median word count
    candidates.sort(key=lambda x: abs(x[0] - median_wc))
    return [c[1] for c in candidates[:n]]


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q-1: Build per-user dynamic conditions + exemplars")
    log("=" * 70)

    # === [1] Load phase10 pairs (876) ===
    log("[1] Loading phase10_pairs_1000 ...")
    p10_pairs = []
    with IN_PAIRS_10.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p10_pairs.append(r)
    log(f"  total p10 pairs: {len(p10_pairs)}")

    # Build (uid, asin) → target_style_desc map
    p10_map = {}
    for r in p10_pairs:
        key = (r["user_id"], r["asin"])
        p10_map[key] = r.get("target_style_desc", "")

    # Parse all 20-d vectors
    vec_20_all = []
    for r in p10_pairs:
        vec = parse_20d(r.get("target_style_desc", ""))
        if len(vec) == 20:
            vec_20_all.append(vec)
    log(f"  parsed 20-d: {len(vec_20_all)}")

    vec_20_all_np = np.array(vec_20_all)
    pop_mean = vec_20_all_np.mean(axis=0)
    pop_std = vec_20_all_np.std(axis=0)
    log(f"  pop_mean[:5]: {pop_mean[:5].round(2).tolist()}")
    log(f"  pop_std[:5]: {pop_std[:5].round(2).tolist()}")

    # === [2] Load phase14_b pairs (30) ===
    log("[2] Loading phase14_b pairs ...")
    b_pairs_set = set()
    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            b_pairs_set.add((r["user_id"], r["asin"]))
    log(f"  phase14_b pairs: {len(b_pairs_set)}")

    # === [3] Load user sentences ===
    log("[3] Loading user_sents cache ...")
    with IN_USER_SENTS.open("rb") as f:
        user_sents = pickle.load(f)
    log(f"  user_sents cache: {len(user_sents)} users")

    # === [4] Build per-pair conditions + exemplars ===
    log("[4] Building per-pair conditions + exemplars ...")
    per_pair_conds = {}
    for (uid, asin) in sorted(b_pairs_set):
        desc = p10_map.get((uid, asin), "")
        vec = parse_20d(desc)
        if len(vec) != 20:
            log(f"  WARN: cannot parse 20-d for ({uid}, {asin})")
            continue
        conds = build_dynamic_conditions(vec, pop_mean, pop_std)
        exem = select_exemplars(uid, user_sents, N_EXEMPLARS)

        # Compute z-scores for later use
        z_scores = (np.array(vec) - pop_mean) / (pop_std + 1e-9)

        per_pair_conds[(uid, asin)] = {
            "vec_20": vec,
            "z_scores": z_scores.tolist(),
            "conditions_text": conds,
            "exemplars": exem,
        }

    n_with_conds = sum(1 for v in per_pair_conds.values() if v["conditions_text"])
    n_with_exem = sum(1 for v in per_pair_conds.values() if v["exemplars"])
    log(f"  pairs with conditions: {n_with_conds}/{len(per_pair_conds)}")
    log(f"  pairs with exemplars: {n_with_exem}/{len(per_pair_conds)}")

    # === [5] Build generation prompts (2 conds × K=8) ===
    log("[5] Building generation prompts (Q-Dynamic + Q-Dynamic+Exemplar) ...")
    prompts_dyn = []
    prompts_dyn_exem = []
    pair_keys = sorted(per_pair_conds.keys())

    attrs_map = {}
    for r in p10_pairs:
        key = (r["user_id"], r["asin"])
        attrs_map[key] = r.get("attrs", {})

    for (uid, asin) in pair_keys:
        rec = per_pair_conds[(uid, asin)]
        attrs = attrs_map.get((uid, asin), {})
        attrs_str = "\n".join([f"{k}: {v}" for k, v in attrs.items()])

        # Cond Q-Dynamic (text conditions only)
        conds_text = "\n".join(rec["conditions_text"]) if rec["conditions_text"] else "  - (no specific conditions)"
        prompt_dyn = (
            f"Product attributes:\n{attrs_str}\n\n"
            f"Style conditions to match this user's typical syntax:\n{conds_text}\n\n"
            f"Write a short shopping query that includes every attribute."
        )
        prompts_dyn.append({
            "user_id": uid, "asin": asin,
            "condition": "Q_Dynamic",
            "prompt": prompt_dyn,
        })

        # Cond Q-Dynamic + Exemplars
        if rec["exemplars"]:
            exem_str = "\n".join([f"Example {i+1}: {e}" for i, e in enumerate(rec["exemplars"])])
            prompt_dyn_exem = (
                f"Product attributes:\n{attrs_str}\n\n"
                f"Style examples from this user:\n{exem_str}\n\n"
                f"Style conditions to match this user's typical syntax:\n{conds_text}\n\n"
                f"Write a short shopping query that includes every attribute. "
                f"Match the user's expression style from the examples but do not copy them."
            )
        else:
            prompt_dyn_exem = prompt_dyn  # fallback

        prompts_dyn_exem.append({
            "user_id": uid, "asin": asin,
            "condition": "Q_DynamicExemplar",
            "prompt": prompt_dyn_exem,
        })

    log(f"  Q-Dynamic prompts: {len(prompts_dyn)}")
    log(f"  Q-DynamicExemplar prompts: {len(prompts_dyn_exem)}")

    # === [6] Sample inspection ===
    log("\n=== Sample inspection ===")
    sample_key = pair_keys[0]
    sample_rec = per_pair_conds[sample_key]
    log(f"Pair: {sample_key[0][:10]}... / {sample_key[1]}")
    log(f"  z-scores (top 5): {sorted(zip(sample_rec['z_scores'], FEATURE_20_NAMES), key=lambda x: -abs(x[0]))[:5]}")
    log(f"  conditions:\n{sample_rec['conditions_text']}")
    log(f"  exemplars:")
    for e in sample_rec["exemplars"]:
        log(f"    - {repr(e)[:120]}")
    log(f"\n  Q-Dynamic prompt:\n{prompts_dyn[0]['prompt'][:300]}...")
    log(f"\n  Q-DynamicExemplar prompt:\n{prompts_dyn_exem[0]['prompt'][:400]}...")

    # === [7] Save ===
    # Save per-pair conds as dict (key: uid+asin)
    per_pair_dict = {
        f"{uid}|{asin}": v for (uid, asin), v in per_pair_conds.items()
    }
    OUT_USER_CONDS.write_text(json.dumps(per_pair_dict, indent=2, ensure_ascii=False))
    log(f"\n  user_conds → {OUT_USER_CONDS}")

    # Save generation prompts
    with OUT_GEN_PROMPTS.open("w") as f:
        for p in prompts_dyn:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        for p in prompts_dyn_exem:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"  gen_prompts → {OUT_GEN_PROMPTS} ({len(prompts_dyn) + len(prompts_dyn_exem)} prompts)")

    meta = {
        "phase": "14.Q-1",
        "method": "Per-user dynamic conditions + exemplars prep",
        "n_pairs": len(pair_keys),
        "n_conditions_per_user": N_TOP_CONDITIONS,
        "z_threshold": Z_THRESHOLD,
        "n_exemplars": N_EXEMPLARS,
        "feature_names": FEATURE_20_NAMES,
        "condition_templates": CONDITION_TEMPLATES,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-1 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()