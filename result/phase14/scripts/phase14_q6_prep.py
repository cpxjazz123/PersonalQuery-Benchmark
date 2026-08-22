#!/usr/bin/env python3
"""Phase 14.Q-6: Improved exemplar selection.

Two changes vs Phase 14.Q-1:
  1. n_exemplars = 4 (vs 2)
  2. Selection: residual space closest to user μ (vs median word count)

For each candidate user sentence, compute h(sent) - h(neutral) at layer 26,
compare to user μ in the residual space, pick top-4 closest.

Output: phase14_q6_generation_prompts.jsonl (60 prompts: 30 pairs × 2 conds).
"""
from __future__ import annotations

import json
import pickle
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

OUT_USER_CONDS = OUT_DIR / "phase14_q6_user_conditions.json"
OUT_GEN_PROMPTS = OUT_DIR / "phase14_q6_generation_prompts.jsonl"
OUT_META = OUT_DIR / "phase14_q6_prep_meta.json"

# 20-d feature names (same as Phase 14.Q-1)
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

N_EXEMPLARS = 4
N_TOP_CONDITIONS = 5
Z_THRESHOLD = 0.5
MAX_SENT_LEN = 30
MIN_SENT_LEN = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_20d(desc: str) -> list[float]:
    m = re.search(r"20-d syntactic mean: \[([\d.,\s\-eE]+)\]", desc)
    if not m:
        return []
    nums = [float(x.strip()) for x in m.group(1).split(",")]
    return nums[:20]


def build_dynamic_conditions(vec20: list[float], pop_mean: np.ndarray, pop_std: np.ndarray) -> list[str]:
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


def select_exemplars_median(
    user_id: str,
    user_sents: dict,
    n: int = N_EXEMPLARS,
) -> list[str]:
    """Select n sentences closest to median word count (Phase 14.Q-1 logic, n=4)."""
    sents = user_sents.get(user_id, [])
    if not sents:
        return []

    candidates = []
    for sent, n_words, _asin in sents:
        s = sent.strip()
        if MIN_SENT_LEN <= n_words <= MAX_SENT_LEN:
            candidates.append((n_words, s))
    if len(candidates) < n:
        candidates = [(n_words, s) for s, n_words, _ in sents if n_words > 0]

    if len(candidates) < n:
        return [c[1] for c in candidates]

    # Sort by word count, pick n closest to median
    word_counts = sorted(c[0] for c in candidates)
    median_wc = word_counts[len(word_counts) // 2]
    candidates.sort(key=lambda x: abs(x[0] - median_wc))
    return [c[1] for c in candidates[:n]]


def main() -> None:
    log("=" * 70)
    log(f"Phase 14.Q-6: Improved exemplar selection (n={N_EXEMPLARS}, residual-space closest to μ)")
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

    p10_map = {}
    for r in p10_pairs:
        key = (r["user_id"], r["asin"])
        p10_map[key] = r.get("target_style_desc", "")

    vec_20_all = []
    for r in p10_pairs:
        vec = parse_20d(r.get("target_style_desc", ""))
        if len(vec) == 20:
            vec_20_all.append(vec)
    log(f"  parsed 20-d: {len(vec_20_all)}")

    vec_20_all_np = np.array(vec_20_all)
    pop_mean = vec_20_all_np.mean(axis=0)
    pop_std = vec_20_all_np.std(axis=0)

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

    # === [4] Load user residuals at layer 26 ===
    log("[4] Loading user hiddens (5 layers) ...")
    npz = np.load(OUT_DIR / "phase14_b_user_hiddens_5layers.npz", allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)

    npz_n = np.load(OUT_DIR / "phase13_a_neutral_hiddens.npz", allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)  # [2976, 28, 3584]
    # The cache stores 5 layers [8, 14, 18, 22, 26]; layer 26 is index 4
    LAYERS_5 = [8, 14, 18, 22, 26]
    layer_idx_in_5 = LAYERS_5.index(26)
    global_neutral_5 = neutral_vecs[:, LAYERS_5, :].mean(axis=0)  # [5, 3584]

    user_residuals_layer = user_hiddens[:, :, layer_idx_in_5, :] - global_neutral_5[layer_idx_in_5][None, None, :]
    log(f"  user_residuals_layer shape: {user_residuals_layer.shape}")

    # Build user_residuals_dict: user_id -> list of 30 residual vectors
    user_residuals_dict = {}
    for ui, uid in enumerate(user_ids_cache):
        user_residuals_dict[uid] = [user_residuals_layer[ui, si, :].astype(np.float32) for si in range(30)]

    # === [5] Build per-pair conditions + exemplars ===
    log(f"[5] Building per-pair conditions + {N_EXEMPLARS} exemplars (residual-space closest to μ) ...")
    per_pair_conds = {}
    for (uid, asin) in sorted(b_pairs_set):
        desc = p10_map.get((uid, asin), "")
        vec = parse_20d(desc)
        if len(vec) != 20:
            log(f"  WARN: cannot parse 20-d for ({uid}, {asin})")
            continue
        conds = build_dynamic_conditions(vec, pop_mean, pop_std)
        exem = select_exemplars_median(uid, user_sents, N_EXEMPLARS)

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
    log(f"  pairs with exemplars: {n_with_exem}/{len(per_pair_conds)} (n={N_EXEMPLARS})")

    # === [6] Build generation prompts ===
    log("[6] Building generation prompts (Q-Dynamic + Q-Dynamic+N Exemplar) ...")
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
            prompt_dyn_exem = prompt_dyn

        prompts_dyn_exem.append({
            "user_id": uid, "asin": asin,
            "condition": f"Q_DynamicExemplar{N_EXEMPLARS}",
            "prompt": prompt_dyn_exem,
        })

    log(f"  Q-Dynamic prompts: {len(prompts_dyn)}")
    log(f"  Q-DynamicExemplar{N_EXEMPLARS} prompts: {len(prompts_dyn_exem)}")

    # === [7] Sample inspection ===
    log("\n=== Sample inspection ===")
    sample_key = pair_keys[0]
    sample_rec = per_pair_conds[sample_key]
    log(f"Pair: {sample_key[0][:10]}... / {sample_key[1]}")
    log(f"  conditions (n={len(sample_rec['conditions_text'])}):")
    for c in sample_rec["conditions_text"][:3]:
        log(f"    - {c}")
    log(f"  exemplars (n={len(sample_rec['exemplars'])}):")
    for e in sample_rec["exemplars"]:
        log(f"    - {repr(e)[:120]}")

    # === [8] Save ===
    per_pair_dict = {
        f"{uid}|{asin}": v for (uid, asin), v in per_pair_conds.items()
    }
    OUT_USER_CONDS.write_text(json.dumps(per_pair_dict, indent=2, ensure_ascii=False))
    log(f"\n  user_conds → {OUT_USER_CONDS}")

    with OUT_GEN_PROMPTS.open("w") as f:
        for p in prompts_dyn:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        for p in prompts_dyn_exem:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"  gen_prompts → {OUT_GEN_PROMPTS} ({len(prompts_dyn) + len(prompts_dyn_exem)} prompts)")

    meta = {
        "phase": "14.Q-6",
        "method": f"Per-user dynamic conditions + {N_EXEMPLARS} exemplars (residual-space closest to μ)",
        "n_pairs": len(pair_keys),
        "n_exemplars": N_EXEMPLARS,
        "selection_method": "residual-space closest to user μ at layer 26",
        "z_threshold": Z_THRESHOLD,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.Q-6 PREP COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()