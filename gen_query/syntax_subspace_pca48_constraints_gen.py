#!/usr/bin/env python3
"""Phase 5 — Explicit Syntax-Conditioned Generation.

用户指令 2026-08-30:
Phase 4 (4 / 4.B / 4.C) 否定了 exemplar 路线. 下一步测 explicit syntax
constraints 是否能控制 PCA48 location.

**核心因果链** (要验证):
    PCA48 target → interpretable F3 syntax constraints → LLM prompt → query d_self
    如果 d_self 跟随 constraint 改变 → causal chain 成立.
    如果不改变 → LLM controllability / syntax representation gap.

**Reverse-Map 算法** (per user):
    1. mu_u (48d) 用户 PCA48 中心
    2. mu_global (48d) cohort 用户平均 (computed once from all user_gaussians)
    3. delta_mu = mu_u - mu_global (48d)
    4. delta_f3 = delta_mu @ pca_components (103d)  -- 反投影到 F3 raw space
    5. 取 top-K (K=5) |delta_f3[i]| 最大的 features → 写进 prompt

**4 conditions**:
    A. control          — 不加 syntax constraint (Phase 4.B N=1 prompt without exemplar)
    B. random_constraints — 随机选 5 个 F3 features → NL instruction
    C. target_constraints — 选 target user top-5 deviating features → NL
    D. anti_target     — 选 NEGATIVE-deviating top-5 features (push AWAY)

**Leakage control**: 不引用任何 user history sentence.
**所有其他变量锁定**: same 50 ASINs × 167 pairs × K=4, same attrs, same
5-layer strict filter, same N_EXEMPLARS=0.

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

# Import helpers (re-use prototype pilot's batch_generate_vllm + filter helpers)
_PROTO = REPO_ROOT / "gen_query" / "syntax_subspace_prototype_pool_regen.py"
_spec = importlib.util.spec_from_file_location("proto_regen", _PROTO)
_proto = importlib.util.module_from_spec(_spec)
sys.modules["proto_regen"] = _proto
_spec.loader.exec_module(_proto)
batch_generate_vllm = _proto.batch_generate_vllm
build_user_content = _proto.build_user_content
log = _proto.log
get_top_n_attrs = _proto.get_top_n_attrs
count_attrs_covered = _proto.count_attrs_covered
has_invalid_punct = _proto.has_invalid_punct
has_first_person = _proto.has_first_person
has_emoji = _proto.has_emoji
has_self_talk = _proto.has_self_talk
is_query_too_long = _proto.is_query_too_long
n_tokens_simple = _proto.n_tokens_simple

PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
COHORT_ASINS = SCRATCH / "stage8_5_asins_exemplar_count_sweep.json"
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"

# === Locked config (Phase 4 same) ===
PILOT_N_ASIN = 50
N_SAMPLE_USERS = 20
RANDOM_SEED_ASIN = 42
RANDOM_SEED_USERS = 0
K_PER_PAIR = 4
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5
N_EXEMPLARS = 0  # LOCKED — isolate syntax constraint effect
TOP_K_CONSTRAINTS = 5

CONDITIONS = ["control", "random", "target", "anti_target"]

# =============================================================================
# F3 feature → natural language instruction
# =============================================================================
# Extended from syntax_subspace_prototype_pool_regen.py:147 (pattern_to_guidance)
# Each entry: (formatter_fn, requires_value: bool, direction_handling: str)
#   direction_handling: "high" means instruction makes the feature higher;
#                      "low" means lower; "binary" is on/off.

def fmt_n_tok(value: float) -> str:
    return f"Write exactly {int(round(value))} tokens."


def fmt_n_clause(value: float) -> str:
    n = int(round(value))
    if n <= 0:
        return "Use a single simple clause (no subordinate clauses)."
    if n == 1:
        return "Use exactly 1 subordinate clause (because/while/when/which/that)."
    return f"Use {n} subordinate clauses in the sentence."


def fmt_clause_type(value: float, label: str, cue: str) -> str:
    return f"Include a {label} ({cue})."


def fmt_n_coord(value: float) -> str:
    if value < 0.3:
        return "Do NOT use coordinating conjunctions (no and/but/or)."
    if value > 1.5:
        return "Use 2-3 coordinated phrases (connect with and/but/or)."
    return "Use 1 coordinated phrase (connect with and/but/or)."


def fmt_n_mod(value: float) -> str:
    if value < 2.0:
        return "Use few modifiers (keep the noun phrase simple)."
    if value > 6.0:
        return "Add many modifiers (multiple adjectives, noun modifiers, possessives)."
    return "Use a moderate number of modifiers."


def fmt_passive(value: float) -> str:
    if value < 0.1:
        return "Use active voice (subject performs the action)."
    return "Use passive voice (object receives the action; use 'was/were + past participle')."


def fmt_pos(value: float, pos_tag: str, nl: str) -> str:
    if value < 0.5:
        return f"Use few {nl}."
    if value > 4.0:
        return f"Use many {nl}."
    return f"Use 1-2 {nl}."


def fmt_max_depth(value: float) -> str:
    if value <= 3:
        return "Use a flat parse tree (subject + verb + object)."
    if value <= 5:
        return "Use a moderate-depth parse tree (allow one level of nesting)."
    return "Use a deeply nested parse tree (multi-level subordinate clauses)."


def fmt_nest_max(value: float) -> str:
    return fmt_max_depth(value)


def fmt_depth_var(value: float) -> str:
    if value < 1.0:
        return "Use uniform tree depth (consistent clause structure)."
    return "Use varied tree depth (mix simple and complex clauses)."


def fmt_main_pattern(value: float, label: str) -> str:
    # value is 0 or 1 — only emit when 1
    if value < 0.5:
        return None  # skip silent features
    return f"Target sentence pattern: {label}."


def fmt_punct_period(value: float) -> str:
    if value > 0.5:
        return "End with a period (declarative)."
    return None


# Mapping: F3 feature → (formatter_fn, value-key-in-feature-dict, direction-handling)
F3_TO_INSTRUCTION = {
    "n_tok":          (lambda v: fmt_n_tok(v),                                        "value", "high"),
    "n_clause":       (lambda v: fmt_n_clause(v),                                      "value", "high"),
    "n_advcl":        (lambda v: fmt_clause_type(v, "adverbial clause", "because/while/when"), "value", "binary"),
    "n_relcl":        (lambda v: fmt_clause_type(v, "relative clause", "which/that/who"),     "value", "binary"),
    "n_ccomp":        (lambda v: fmt_clause_type(v, "complement clause", "what/that/how"),     "value", "binary"),
    "n_coord":        (lambda v: fmt_n_coord(v),                                       "value", "high"),
    "n_mod":          (lambda v: fmt_n_mod(v),                                         "value", "high"),
    "dep_auxpass":    (lambda v: fmt_passive(v),                                       "value", "binary"),
    "pos_ADV":        (lambda v: fmt_pos(v, "ADV", "adverbs"),                         "value", "high"),
    "pos_SCONJ":      (lambda v: fmt_pos(v, "SCONJ", "subordinating conjunctions like because/although/when"), "value", "high"),
    "pos_VERB":       (lambda v: fmt_pos(v, "VERB", "verbs"),                          "value", "high"),
    "pos_ADJ":        (lambda v: fmt_pos(v, "ADJ", "adjectives"),                      "value", "high"),
    "max_depth":      (lambda v: fmt_max_depth(v),                                    "value", "high"),
    "nest_max":       (lambda v: fmt_nest_max(v),                                      "value", "high"),
    "depth_var":      (lambda v: fmt_depth_var(v),                                     "value", "high"),
    # main_* patterns — emit only when value > 0.5
    "main_SV":        (lambda v: fmt_main_pattern(v, "Subject-Verb (SV)"),            "value", "binary"),
    "main_SVO":       (lambda v: fmt_main_pattern(v, "Subject-Verb-Object (SVO)"),     "value", "binary"),
    "main_SVC":       (lambda v: fmt_main_pattern(v, "Subject-Verb-Complement (SVC)"), "value", "binary"),
    "main_SVA":       (lambda v: fmt_main_pattern(v, "Subject-Verb-Adverbial (SVA)"), "value", "binary"),
}


def feature_to_instruction(fname: str, delta_value: float) -> str | None:
    """Convert (F3 feature name, signed delta) → instruction string."""
    if fname not in F3_TO_INSTRUCTION:
        return None
    fn, _, handling = F3_TO_INSTRUCTION[fname]
    if handling == "binary":
        # Use sign of delta to decide emit/skip
        if abs(delta_value) < 0.05:
            return None
        # Higher delta → instruction includes feature; negative → opposite
        if delta_value > 0:
            return fn(1.0)
        else:
            # For binary features, negative delta means "lower frequency of this feature"
            # Provide opposite instruction
            if fname == "dep_auxpass":
                return fmt_passive(0.0)  # active voice
            if fname.startswith("n_advcl"):
                return "Do NOT include adverbial clauses (no because/while/when)."
            if fname.startswith("n_relcl"):
                return "Do NOT include relative clauses (no which/that/who)."
            if fname.startswith("n_ccomp"):
                return "Do NOT include complement clauses (no what/that/how)."
            if fname.startswith("main_"):
                return None  # silent for main_ negative
            return None
    else:  # "high"
        # Always emit with target value (sign tells magnitude, but for instructions we use direction)
        # For positive delta → high value; for negative delta → low value
        target_value = max(1.0, abs(delta_value))  # at least 1
        if delta_value < 0:
            # Negative delta means this feature should be low in the user's query
            # Provide opposite instruction
            if fname == "n_tok":
                return "Keep the sentence short (under 12 tokens)."
            if fname == "n_clause":
                return "Use a single simple clause (no subordinate clauses)."
            if fname == "n_coord":
                return fmt_n_coord(0.0)
            if fname == "n_mod":
                return fmt_n_mod(1.0)
            if fname == "max_depth" or fname == "nest_max":
                return fmt_max_depth(2.0)
            if fname == "depth_var":
                return fmt_depth_var(0.5)
            if fname.startswith("pos_"):
                tag = fname.split("_", 1)[1]
                if tag == "ADV":
                    return "Use few adverbs."
                if tag == "SCONJ":
                    return "Avoid subordinating conjunctions (no because/although/when)."
                if tag == "VERB":
                    return "Use few verbs."
                if tag == "ADJ":
                    return "Use few adjectives."
                return None
            return None
        return fn(target_value)


def build_constraints_block(user_delta_f3: list[tuple[str, float]]) -> str:
    """Build NL constraint block from top-K (fname, delta_value) pairs."""
    lines = ["Syntactic constraints for this query:"]
    for fname, delta in user_delta_f3:
        ins = feature_to_instruction(fname, delta)
        if ins:
            lines.append(f"- {ins}")
    if len(lines) == 1:
        lines.append("- Use natural flowing English.")
    return "\n".join(lines)


# =============================================================================
# Prompt template
# =============================================================================
GEN_SYSTEM_TMPL_CONSTRAINTS = (
    "You are a real customer searching for a product. "
    "Write a single natural search query as if you are a shopper looking for "
    "this exact product. The query MUST mention ALL of these product attributes:\n"
    "{ATTRIBUTES}\n\n"
    "{CONSTRAINTS}\n\n"
    "Rules:\n"
    "- First-person voice: use 'I', 'my', 'I'm looking for', 'I want', etc.\n"
    "- One natural sentence (NOT a bulleted list).\n"
    "- Do NOT start with 'Here:', 'Brand:', 'Attribute:' or any prefix.\n"
    "- Do NOT use emoji.\n"
    "- Do NOT talk to the AI (no 'can someone help', no 'thanks!').\n"
    "- Maximum 60 tokens.\n"
    "Write only the query, no commentary."
)


def make_constraint_prompt(attrs: dict, constraints_block: str) -> str:
    return GEN_SYSTEM_TMPL_CONSTRAINTS.format(
        ATTRIBUTES=build_user_content(attrs),
        CONSTRAINTS=constraints_block,
    )


# =============================================================================
# Cohort + Gaussian loading
# =============================================================================
def get_pairs_and_attrs() -> list[dict]:
    pattrs = json.load(open(PATTRS_PATH))
    cohort = json.load(open(COHORT_ASINS))
    rng_asin = random.Random(RANDOM_SEED_ASIN)
    asin_data_sorted = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng_asin.shuffle(asin_data_sorted)
    asin_data_pilot = asin_data_sorted[:PILOT_N_ASIN]
    pairs = []
    for entry in asin_data_pilot:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng = random.Random(RANDOM_SEED_USERS)
        rng.shuffle(users)
        for u in users:
            pa = pattrs.get(entry["asin"])
            if pa is None:
                continue
            top_n = get_top_n_attrs(pa, N_INPUT)
            if len(top_n) < N_INPUT:
                continue
            pairs.append({"user_id": u, "asin": entry["asin"], "attrs": top_n})
    return pairs


def load_gaussians_for_cohort(pairs: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load Gaussian + scaler + PCA + cohort mean of mu.

    Returns:
      gauss_users: {uid → {mu, sigma_diag}}
      scaler_mean (103,), scaler_scale (103,)
      pca_components (48, 103), pca_mean (103,)
      fnames_f3 (list of 103 F3 feature names)
      mu_global (48,) — cohort mean of mu across all users in cohort
    """
    gdoc = json.load(open(GAUSS_PATH))
    cohort_users = {pd["user_id"] for pd in pairs}
    users = {u: g for u, g in gdoc["users"].items() if u in cohort_users}
    mus = np.array([users[u]["mu"] for u in users], dtype=np.float64)
    mu_global = mus.mean(axis=0)  # (48,)
    return users, mu_global, gdoc


def per_user_target_features(users: dict, mu_global: np.ndarray,
                              gdoc: dict) -> dict:
    """For each user u in cohort, compute top-K (fname, delta_f3) pairs.

    delta_f3 = (mu_u - mu_global) @ pca_components  →  (103,)
    Top-K by |delta_f3|.
    """
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)  # (48, 103)
    fnames_f3 = gdoc["fnames_f3"]

    out = {}
    for uid, g in users.items():
        mu_u = np.asarray(g["mu"], dtype=np.float64)  # (48,)
        delta_mu = mu_u - mu_global
        delta_f3 = delta_mu @ pca_components  # (103,)
        # Rank by |delta_f3|
        abs_delta = np.abs(delta_f3)
        top_idx = np.argsort(-abs_delta)[:TOP_K_CONSTRAINTS]
        top = [(fnames_f3[i], float(delta_f3[i])) for i in top_idx]
        out[uid] = top
    return out


# =============================================================================
# Per-condition prompt builder
# =============================================================================
def build_constraints_for(pd: dict, condition: str,
                          user_target_feats: dict, rng: random.Random) -> str:
    u = pd["user_id"]
    if condition == "control":
        return "(No additional syntax constraints — write naturally.)"
    if condition == "random":
        # Random 5 F3 features
        all_fnames = list(F3_TO_INSTRUCTION.keys())
        picks = rng.sample(all_fnames, k=TOP_K_CONSTRAINTS)
        # Random sign to simulate "high" vs "low" — use abs() because random
        return build_constraints_block([(fn, abs(rng.gauss(0, 1.0))) for fn in picks])
    if condition == "target":
        feats = user_target_feats.get(u, [])
        return build_constraints_block(feats)
    if condition == "anti_target":
        feats = user_target_feats.get(u, [])
        # Flip sign of every delta
        flipped = [(fn, -delta) for fn, delta in feats]
        return build_constraints_block(flipped)
    raise ValueError(condition)


def run_condition(condition: str, pairs: list[dict],
                  user_target_feats: dict, rng_seed: int) -> dict:
    log(f"\n=== Condition: {condition} ===")
    pair_data = []
    for pd in pairs:
        rng = random.Random(hash((pd["user_id"], pd["asin"], condition)) & 0xffffffff)
        constraints_block = build_constraints_for(pd, condition, user_target_feats, rng)
        pair_data.append({**pd, "constraints_block": constraints_block})

    all_prompts, pair_k = [], []
    for pd in pair_data:
        prompt = make_constraint_prompt(pd["attrs"], pd["constraints_block"])
        for k in range(K_PER_PAIR):
            all_prompts.append(prompt)
            pair_k.append((pd["user_id"], pd["asin"], k))
    n_prompts = len(all_prompts)
    log(f"  total prompts: {n_prompts} = {len(pair_data)} pairs × {K_PER_PAIR}")

    t0 = time.time()
    outputs = batch_generate_vllm(all_prompts)
    log(f"  vLLM done in {time.time()-t0:.0f}s")

    pools = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full":0, "invalid":0, "first_p":0, "emoji":0, "self_talk":0, "too_long":0}
    pair_records = []
    for (u, a, k), out_text in zip(pair_k, outputs):
        attrs_dict = next(pd["attrs"] for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
        constraints = next(pd["constraints_block"] for pd in pair_data if pd["user_id"] == u and pd["asin"] == a)
        text = out_text.strip() if out_text else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs_dict)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        strict = (n_cov == N_INPUT) and (not invalid) and first_p and (not emoji) and (not self_talk) and (not too_long)
        if n_cov == N_INPUT: filter_counts["cov_full"] += 1
        if invalid: filter_counts["invalid"] += 1
        if first_p: filter_counts["first_p"] += 1
        if emoji: filter_counts["emoji"] += 1
        if self_talk: filter_counts["self_talk"] += 1
        if too_long: filter_counts["too_long"] += 1
        if strict:
            n_total_strict += 1
        pools[a].append({"user_id":u, "k":k, "query":text, "strict":strict,
                         "attrs_covered":n_cov, "invalid":invalid,
                         "first_person":first_p, "emoji":emoji,
                         "self_talk":self_talk, "too_long":too_long,
                         "n_tok":n_tokens_simple(text)})
        pair_records.append({
            "user_id": u, "asin": a, "k": k,
            "constraints_block": constraints,
            "query": text,
            "strict": strict,
        })

    log(f"  filter: {filter_counts}")
    log(f"  total strict: {n_total_strict}")

    out_path = SCRATCH / f"pool_constraints_{condition}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"description": f"Phase 5 — {condition}",
                       "N_EXEMPLARS": N_EXEMPLARS, "K_PER_PAIR": K_PER_PAIR,
                       "TOP_K_CONSTRAINTS": TOP_K_CONSTRAINTS},
            "filter_counts": filter_counts,
            "n_total_strict": n_total_strict,
            "pools": pools,
            "pair_records": pair_records,
        }, f, ensure_ascii=False)
    log(f"  wrote → {out_path.name}")
    return {"condition": condition, "n_pairs": len(pair_data), "n_strict": n_total_strict,
            "pool_path": str(out_path)}


def main():
    log("=== Phase 5 — Explicit Syntax-Conditioned Generation ===")
    pairs = get_pairs_and_attrs()
    log(f"  pilot pairs: {len(pairs)}")

    users_gauss, mu_global, gdoc = load_gaussians_for_cohort(pairs)
    log(f"  cohort Gaussians: {len(users_gauss)} (mu_global norm={np.linalg.norm(mu_global):.3f})")

    # Compute top-K target features per user
    log("\n--- Computing target F3 features per user (PCA48 → F3 reverse-map) ---")
    user_target_feats = per_user_target_features(users_gauss, mu_global, gdoc)
    sample_u = next(iter(user_target_feats))
    log(f"  sample user {sample_u} top-{TOP_K_CONSTRAINTS}:")
    for fn, dv in user_target_feats[sample_u]:
        log(f"    {fn:20s}  delta={dv:+.3f}")

    # Generate for each condition
    log("\n--- Generating queries per condition ---")
    cond_results = {}
    for cond in CONDITIONS:
        cond_results[cond] = run_condition(cond, pairs, user_target_feats, RANDOM_SEED_ASIN)

    # Save the per-user top-K features (for audit reuse)
    feat_path = SCRATCH / "user_target_f3_features.json"
    with open(feat_path, "w") as f:
        json.dump({u: [{"fname": fn, "delta_f3": dv} for fn, dv in feats]
                   for u, feats in user_target_feats.items()}, f, ensure_ascii=False)
    log(f"\n  wrote → {feat_path.name}")

    summary = {
        "config": {
            "PILOT_N_ASIN": PILOT_N_ASIN, "N_SAMPLE_USERS": N_SAMPLE_USERS,
            "K_PER_PAIR": K_PER_PAIR, "N_EXEMPLARS": N_EXEMPLARS,
            "TOP_K_CONSTRAINTS": TOP_K_CONSTRAINTS, "TEMP": TEMP,
            "leakage_control": "no exemplar; no user history; constraints derived from cohort PCA48",
            "conditions": CONDITIONS,
        },
        "results": cond_results,
        "next_step": "Phase 5 audit: Stage 4 strict alignment per condition, "
                     "compute per-pair transfer target-vs-anti_target (causal test).",
    }
    out = REPO_ROOT / "result/gen_query/pca48_constraints_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out}")


if __name__ == "__main__":
    main()