#!/usr/bin/env python3
"""Phase 10.12.C: Dynamic User-Specific Syntactic Control.

用户洞察 (2026-08-20):
  不要所有用户都使用相同的 12 个句法条件, 应该为每个用户生成专属的
  3-5 个句法控制条件, 基于 μ_u - μ_pop 的稳定偏离方向.

流程:
  μ_u (raw 20d) - μ_pop → r_u
  → split-half stability filter (sign consistent)
  → top-K features by |r_u|
  → user-specific natural language hints
  → LLM free generation (per pair × per control × seed)
  → hard-copy attrs
  → Mahalanobis max-margin rerank (VADES 20d, 同 Phase 10.12.B)
  → ControlHit + Win rate + Rank + Gap

对比目标: Phase 10.12.B (12 全局条件, max_margin) win_rate=67.24%.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_CANDIDATES = OUT_DIR / "phase10_12_c_candidates.jsonl"
OUT_EVAL = OUT_DIR / "phase10_12_c_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase10_12_c_per_pair.jsonl"
OUT_USER_CONTROLS = OUT_DIR / "phase10_12_c_user_controls.jsonl"

SEED = 42
N_BOOTSTRAP = 5000
N_WRONG_USERS_PER_PAIR = 5

# === 控制选择参数 ===
N_CONTROLS_PER_USER = 3  # 每个用户 top-3 features
N_SEEDS_PER_CONTROL = 2  # 每个 control 2 个种子
MIN_R_U_THRESHOLD = 0.3  # min |r_u| 才算"显著偏离"
# Split-half stability: 仅保留 sign(r_h1) == sign(r_h2) 的特征 (binary)

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"


# === 20d feature → 自然语言 hint ===
# direction: "high" 偏高于总体, "low" 偏低于总体
FEATURE_HINTS = {
    "max_dependency_depth": {
        "high": "use deep nested clause structures",
        "low": "keep clauses shallow and flat",
    },
    "mean_dependency_depth": {
        "high": "prefer complex multi-clause structures",
        "low": "prefer simple flat sentence structures",
    },
    "dependency_tree_height": {
        "high": "build tall hierarchical sentence structures",
        "low": "keep syntactic structures flat",
    },
    "depth_variance": {
        "high": "mix deep and shallow clauses across sentences",
        "low": "keep uniform clause depth throughout",
    },
    "acl_count": {
        "high": "use more adjectival clauses (e.g. 'which is made of...')",
        "low": "avoid adjectival clauses",
    },
    "relcl_count": {
        "high": "include relative clauses (with which/that/who)",
        "low": "avoid relative clauses",
    },
    "ccomp_count": {
        "high": "use clausal complements (e.g. 'I think that ...')",
        "low": "avoid clausal complements",
    },
    "xcomp_count": {
        "high": "use open clausal complements (e.g. 'want to buy')",
        "low": "avoid open clausal complements",
    },
    "advcl_count": {
        "high": "use adverbial clauses (when/because/while)",
        "low": "avoid adverbial clauses",
    },
    "clause_nesting_depth": {
        "high": "use deeply nested clauses",
        "low": "keep clauses at top level",
    },
    "mean_dependency_distance": {
        "high": "use long-range dependencies between words",
        "low": "keep word dependencies short and local",
    },
    "max_dependency_distance": {
        "high": "allow very long dependencies",
        "low": "keep all dependencies local",
    },
    "long_dependency_ratio": {
        "high": "use long-distance dependencies frequently",
        "low": "keep all dependencies short",
    },
    "amod_count": {
        "high": "use many adjectival modifiers (e.g. 'soft cotton', 'big red')",
        "low": "minimize adjectival modifiers",
    },
    "advmod_count": {
        "high": "use many adverbial modifiers (e.g. 'very', 'really')",
        "low": "minimize adverb modifiers",
    },
    "nmod_count": {
        "high": "use many nominal modifiers (possessives, prepositional)",
        "low": "minimize nominal modifiers",
    },
    "compound_count": {
        "high": "use compound words and noun phrases",
        "low": "avoid compounds",
    },
    "modifier_density": {
        "high": "use rich modifiers throughout the sentence",
        "low": "keep modifiers minimal",
    },
    "coordination_count": {
        "high": "coordinate many elements with and/but/or",
        "low": "avoid coordination",
    },
    "max_branching_factor": {
        "high": "use wide-branching structures",
        "low": "keep branching narrow",
    },
}


def load_user_profiles():
    """Load VADES user_mu/user_logvar + user_id mapping (用于 Mahalanobis distance)."""
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    return profiles, user_id_to_idx, user_mu, user_logvar


def load_sentence_features_by_user():
    """Group sentences by user_id, return {user_id: list of 20d feature dicts}."""
    by_user = defaultdict(list)
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            by_user[r["user_id"]].append(r["features"])
    return by_user


def compute_population_stats(by_user: Dict[str, List[Dict]]) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute μ_pop, σ_pop (across users), plus per-user μ_u_raw.

    Returns:
        feature_names: list of 20 feature names
        pop_mean: 20d array
        pop_std: 20d array
        user_mu_raw: dict user_id -> 20d array (raw feature mean)
    """
    feature_names = None
    user_mu_raw = {}
    for uid, feats_list in by_user.items():
        if feature_names is None:
            feature_names = list(feats_list[0].keys())
        arr = np.array([[float(f[k]) for k in feature_names] for f in feats_list], dtype=np.float32)
        user_mu_raw[uid] = arr.mean(axis=0)

    user_arr = np.stack(list(user_mu_raw.values()), axis=0)
    pop_mean = user_arr.mean(axis=0)
    pop_std = user_arr.std(axis=0) + 1e-9
    return feature_names, pop_mean, pop_std, user_mu_raw


def compute_user_r_u_with_stability(
    by_user: Dict[str, List[Dict]],
    pop_mean: np.ndarray,
    pop_std: np.ndarray,
    seed: int = SEED,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute r_u (full) and split-half sign consistency per user.

    Args:
        by_user: {user_id: list of feature dicts}
        pop_mean, pop_std: 20d population statistics
        seed: split-half random seed

    Returns:
        r_u_full: {user_id: 20d array of (μ_u - μ_pop) / σ_pop}
        stability: {user_id: 20d array, +1 / -1 / 0 (sign consistent + direction)}
    """
    rng = np.random.default_rng(seed)
    r_u_full = {}
    stability = {}

    for uid, feats_list in by_user.items():
        feature_names = list(feats_list[0].keys())
        arr = np.array([[float(f[k]) for k in feature_names] for f in feats_list], dtype=np.float32)

        # Full mean
        mu_full = arr.mean(axis=0)
        r_full = (mu_full - pop_mean) / pop_std

        # Split-half (random 50/50, 至少 2 sentences/half 才计算)
        n = arr.shape[0]
        if n < 2:
            r_u_full[uid] = r_full
            stability[uid] = np.zeros_like(r_full)
            continue

        idx = np.arange(n)
        rng.shuffle(idx)
        half = max(1, n // 2)
        h1 = arr[idx[:half]]
        h2 = arr[idx[half:half*2]]  # 第二半
        r_h1 = (h1.mean(axis=0) - pop_mean) / pop_std
        r_h2 = (h2.mean(axis=0) - pop_mean) / pop_std

        # Sign consistency
        sign_r = np.sign(r_full)
        sign_h1 = np.sign(r_h1)
        sign_h2 = np.sign(r_h2)
        # consistency = +1 if both halves same sign as full, -1 if both halves opposite, 0 otherwise
        match = (sign_h1 == sign_r) & (sign_h2 == sign_r)
        match_opposite = (sign_h1 == -sign_r) & (sign_h2 == -sign_r)
        sign_consistency = np.where(match, sign_r, np.where(match_opposite, -sign_r, 0))

        r_u_full[uid] = r_full
        stability[uid] = sign_consistency

    return r_u_full, stability


def select_user_controls(
    user_id: str,
    r_u: np.ndarray,
    stability: np.ndarray,
    feature_names: List[str],
    top_k: int = N_CONTROLS_PER_USER,
    min_r_u: float = MIN_R_U_THRESHOLD,
) -> List[Dict]:
    """Select top-K controls for user based on stable |r_u|.

    Returns:
        list of {feat_name, feat_idx, direction, magnitude, stability_sign}
    """
    # Score: |r_u| where stability matches sign of r_u (positive stability)
    sign_r = np.sign(r_u)
    positive_stability = (stability == sign_r) & (np.abs(r_u) > min_r_u)

    scores = np.where(positive_stability, np.abs(r_u), 0.0)

    # Top-k indices
    top_indices = np.argsort(-scores)[:top_k]

    controls = []
    for idx in top_indices:
        if scores[idx] < min_r_u:
            break
        direction = "high" if r_u[idx] > 0 else "low"
        controls.append({
            "feat_idx": int(idx),
            "feat_name": feature_names[idx],
            "direction": direction,
            "magnitude": float(np.abs(r_u[idx])),
            "stability_sign": int(stability[idx]),
        })

    return controls


def build_prompt(attrs_5: Dict[str, str], control_text: str) -> str:
    """Build prompt with user-specific control hint (one feature at a time)."""
    main_attrs = []
    for k in ["Brand", "Color", "Material", "Item Weight", "Product Dimensions"]:
        if k in attrs_5 and attrs_5[k]:
            main_attrs.append(f"{k}: {attrs_5[k]}")

    attr_str = "\n".join(main_attrs[:3])

    prompt = f"""Write ONE short shopping query (1 sentence, ≤25 words) that mentions every attribute below.

Product attributes:
{attr_str}

STYLE REQUIREMENTS:
Your personal writing style tends to {control_text}.
Use your own wording and do not follow a fixed sentence template.
Write naturally, as you would for a personal shopping search.

Output ONLY the query itself on a single line. No prefix, no explanation, no bullet points. Start directly with the query.
"""
    return prompt


def needs_hard_copy(query: str, attrs: dict) -> bool:
    q_lower = query.lower()
    for k, v in attrs.items():
        if v and str(v).lower() not in q_lower:
            return True
    return False


def hard_copy(query: str, attrs: dict) -> str:
    q_lower = query.lower()
    missing = []
    for k, v in attrs.items():
        if v and str(v).lower() not in q_lower:
            missing.append(f"{v}")
    if missing:
        if not query.endswith(("?", ".", "!")):
            query = query.rstrip() + "."
        return query + " Looking for " + ", ".join(missing) + "."
    return query


def bootstrap_ci(values_per_item: Dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    items = list(values_per_item.keys())
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0)
    obs_vals = np.array([values_per_item[k] for k in items])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals = obs_vals[sample_idx]
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high)


def main():
    log = lambda m: print(f"[phase10-12.C] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.12.C: Dynamic User-Specific Syntactic Control")
    log("=" * 70)

    # === Load data ===
    log("[1] Loading user profiles + sentences + pairs ...")
    profiles, user_id_to_idx, user_mu, user_logvar = load_user_profiles()
    n_vades_users = len(user_id_to_idx)
    log(f"  VADES users: {n_vades_users}")

    by_user = load_sentence_features_by_user()
    log(f"  sentence users: {len(by_user)}")

    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    # === Population stats + per-user μ_u_raw ===
    log("[2] Computing population stats (across users) ...")
    feature_names, pop_mean, pop_std, user_mu_raw = compute_population_stats(by_user)
    log(f"  features: {feature_names}")
    log(f"  pop_mean sample: {pop_mean[:5].tolist()}")
    log(f"  pop_std sample:  {pop_std[:5].tolist()}")

    # === r_u + stability ===
    log("[3] Computing r_u + split-half stability ...")
    r_u_full, stability = compute_user_r_u_with_stability(by_user, pop_mean, pop_std)
    log(f"  r_u computed for {len(r_u_full)} users")

    # === Select top-3 controls per user ===
    log("[4] Selecting top-3 controls per user ...")
    user_controls = {}
    n_no_control = 0
    for uid in r_u_full.keys():
        ctrls = select_user_controls(uid, r_u_full[uid], stability[uid], feature_names)
        if len(ctrls) == 0:
            n_no_control += 1
        user_controls[uid] = ctrls

    n_with_1 = sum(1 for c in user_controls.values() if len(c) == 1)
    n_with_2 = sum(1 for c in user_controls.values() if len(c) == 2)
    n_with_3 = sum(1 for c in user_controls.values() if len(c) == 3)
    log(f"  controls: 1={n_with_1}, 2={n_with_2}, 3={n_with_3}, 0={n_no_control}")

    # Save user controls
    log(f"[5] Saving user controls ...")
    with OUT_USER_CONTROLS.open("w") as f:
        for uid, ctrls in user_controls.items():
            f.write(json.dumps({
                "user_id": uid,
                "controls": ctrls,
                "r_u": r_u_full[uid].tolist(),
                "stability": stability[uid].tolist(),
            }, ensure_ascii=False) + "\n")
    log(f"  → {OUT_USER_CONTROLS}")

    # === Build prompts (per pair × per control × per seed) ===
    log("[6] Building prompts ...")
    valid_pairs = []
    skipped_no_user = 0
    skipped_no_main_attrs = 0
    for p in pairs:
        attrs = p.get("attrs_5", {}) or p.get("attrs", {})
        if not all(k in attrs and attrs[k] for k in ["Brand", "Color", "Material"]):
            skipped_no_main_attrs += 1
            continue
        if p["user_id"] not in user_controls:
            skipped_no_user += 1
            continue
        if len(user_controls[p["user_id"]]) == 0:
            skipped_no_user += 1  # user has no significant deviation
            continue
        valid_pairs.append(p)
    log(f"  valid pairs: {len(valid_pairs)}, skipped no_attrs={skipped_no_main_attrs}, skipped no_user={skipped_no_user}")

    all_prompts = []
    n_total_ctrl = 0
    for pair_idx, p in enumerate(valid_pairs):
        attrs = p.get("attrs_5", {}) or p.get("attrs", {})
        ctrls = user_controls[p["user_id"]]
        for ctrl in ctrls:
            n_total_ctrl += 1
            feat_name = ctrl["feat_name"]
            direction = ctrl["direction"]
            hint = FEATURE_HINTS[feat_name][direction]
            # Personal phrasing
            if direction == "high":
                personal = f"{hint} (more than most users)"
            else:
                personal = f"{hint} (less than most users)"
            for seed_idx in range(N_SEEDS_PER_CONTROL):
                prompt = build_prompt(attrs, personal)
                all_prompts.append((pair_idx, ctrl, seed_idx, prompt))

    total = len(all_prompts)
    log(f"  total prompts: {total} (= {len(valid_pairs)} pairs × avg {n_total_ctrl/max(1,len(valid_pairs)):.2f} ctrls/pair × {N_SEEDS_PER_CONTROL} seeds)")

    # === Generate via vllm direct batch ===
    log(f"[7] Generating {total} candidates via vllm ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client()

    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=120,
        temperature=0.9,
        top_p=0.95,
        stop=["\n\n", "Product attributes:", "STYLE REQUIREMENTS:"],
    )

    backend = client._backend
    if backend is None:
        raise RuntimeError("vllm backend not initialized")

    prompts_only = [p[3] for p in all_prompts]
    t0 = time.time()
    outputs = backend.model.generate(prompts_only, sampling)
    elapsed = time.time() - t0
    log(f"  vllm returned {len(outputs)} in {elapsed:.1f}s, rate={len(outputs)/elapsed:.1f}/s")

    # === Build candidates + hard-copy ===
    log("[8] Building candidates + hard-copy attrs ...")
    candidates = []
    n_failed = 0
    n_with_append = 0
    for i, out in enumerate(outputs):
        pair_idx, ctrl, seed_idx, _ = all_prompts[i]
        pair = valid_pairs[pair_idx]
        if not out.outputs:
            result = ""
            n_failed += 1
        else:
            result = out.outputs[0].text.strip()
            if not result:
                n_failed += 1

        attrs_3 = {k: (pair.get("attrs_5", {}) or pair.get("attrs", {})).get(k, "") for k in ["Brand", "Color", "Material"]}
        needed_append = False
        if result and needs_hard_copy(result, attrs_3):
            result = hard_copy(result, attrs_3)
            needed_append = True
            n_with_append += 1

        candidates.append({
            "user_id": pair["user_id"],
            "asin": pair["asin"],
            "ctrl_feat": ctrl["feat_name"],
            "ctrl_direction": ctrl["direction"],
            "ctrl_magnitude": ctrl["magnitude"],
            "ctrl_stability": ctrl["stability_sign"],
            "seed_idx": seed_idx,
            "candidate_query": result,
            "attrs": pair.get("attrs", {}),
            "attrs_5": pair.get("attrs_5") or pair.get("attrs", {}),
            "needed_append": needed_append,
        })

    log(f"  candidates: {len(candidates)}, failed: {n_failed}, hard-copy: {n_with_append}")

    # Save candidates
    log(f"[9] Saving candidates to {OUT_CANDIDATES} ...")
    with OUT_CANDIDATES.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # === Extract candidate features for distance ===
    log(f"[10] Extracting 20d features for {len(candidates)} candidates ...")
    from extract_clause_features_single_query import extract_clause_features
    t0 = time.time()
    cand_feats_raw = np.zeros((len(candidates), len(feature_names)), dtype=np.float64)
    n_feat_failed = 0
    for ci, c in enumerate(candidates):
        try:
            feat_dict = extract_clause_features(c["candidate_query"])
            cand_feats_raw[ci] = np.array([float(feat_dict[n]) for n in feature_names], dtype=np.float32)
        except Exception:
            n_feat_failed += 1
    log(f"  done in {time.time()-t0:.1f}s, failed: {n_feat_failed}")

    # === Mahalanobis distance (same as Phase 10.12.B) ===
    log("[11] Computing Mahalanobis distance matrix (VADES 20d) ...")
    # Normalize candidate features using population stats
    cand_feats_norm = (cand_feats_raw - pop_mean) / pop_std

    # Normalize user_mu by SAME mean/std (per Phase 10.12.B logic)
    user_mu_norm = (user_mu - pop_mean) / pop_std

    inv_var = np.exp(-user_logvar)
    cand_norm_sq = (cand_feats_norm ** 2) @ inv_var.T
    user_norm_sq = (user_mu_norm ** 2 * inv_var).sum(axis=1)
    weighted_user = inv_var * user_mu_norm
    cross = cand_feats_norm @ weighted_user.T
    dist_matrix = cand_norm_sq + user_norm_sq[None, :] - 2 * cross
    log(f"  dist_matrix: {dist_matrix.shape}")

    n_users = user_mu_norm.shape[0]

    # Group candidates by pair
    cand_by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    log(f"  pairs w/ candidates: {len(cand_by_pair)}")

    # Pre-sample wrong users per pair
    rng = np.random.default_rng(SEED)
    wrong_users_per_pair = {}
    for (uid, asin) in cand_by_pair.keys():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        mask = np.ones(n_users, dtype=bool)
        mask[target_idx] = False
        wrong_idx = rng.choice(np.arange(n_users)[mask], size=N_WRONG_USERS_PER_PAIR, replace=False)
        wrong_users_per_pair[(uid, asin)] = wrong_idx

    # === Per-pair: 3 rerank strategies + cross-user metrics ===
    log("[12] Computing per-pair metrics for 3 rerank strategies ...")
    results = {
        "A_min_d_target": {"gap": {}, "wr": {}, "rank": {}},
        "B_max_margin": {"gap": {}, "wr": {}, "rank": {}},
        "C_weighted": {"gap": {}, "wr": {}, "rank": {}},
    }
    control_hit_per_pair = {}

    t0 = time.time()
    for (uid, asin), cand_indices in cand_by_pair.items():
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            continue
        wrong_idx = wrong_users_per_pair.get((uid, asin))
        if wrong_idx is None:
            continue
        dist_pair = dist_matrix[cand_indices]
        d_target = dist_pair[:, target_idx]
        d_wrong = dist_pair[:, wrong_idx]

        # Margin
        masked = dist_pair.copy()
        masked[:, target_idx] = np.inf
        d_nearest_other = masked.min(axis=1)
        margin = d_nearest_other - d_target

        for strategy, score_fn in [
            ("A_min_d_target", lambda: np.argmin(d_target)),
            ("B_max_margin", lambda: np.argmax(margin)),
            ("C_weighted", lambda: np.argmin(-margin + d_target)),
        ]:
            pick_idx = score_fn()
            pick_d_target = float(d_target[pick_idx])
            pick_d_wrong = float(d_wrong[pick_idx].mean())
            results[strategy]["gap"][(uid, asin)] = pick_d_target - pick_d_wrong
            results[strategy]["wr"][(uid, asin)] = float(pick_d_target < pick_d_wrong)
            ranks = np.argsort(np.argsort(dist_pair[pick_idx]))
            results[strategy]["rank"][(uid, asin)] = float(ranks[target_idx]) / n_users

        # ControlHit: 选出的候选是否真的反映了用户控制方向
        # 每个 control feature 都有预期方向 (high/low)
        # 候选的对应特征与 pop_mean 比较, 看是否同方向
        ctrls = user_controls[uid]
        # 选 B_max_margin 的候选评估 (与主策略一致)
        pick_idx = int(np.argmax(margin))
        cand_feats_pick = cand_feats_raw[cand_indices[pick_idx]]
        hits = 0
        total_ctrls = 0
        for ctrl in ctrls:
            feat_idx = ctrl["feat_idx"]
            expected_dir = ctrl["direction"]  # "high" or "low"
            actual_dir = "high" if cand_feats_pick[feat_idx] > pop_mean[feat_idx] else "low"
            if actual_dir == expected_dir:
                hits += 1
            total_ctrls += 1
        control_hit_per_pair[(uid, asin)] = hits / max(1, total_ctrls)

    log(f"  per-pair done in {time.time()-t0:.1f}s")

    # === Aggregate ===
    log("[13] Aggregating ...")
    summary = {}
    for strategy in results:
        m_gap, ci_gap = bootstrap_ci(results[strategy]["gap"])
        m_wr, ci_wr = bootstrap_ci(results[strategy]["wr"])
        m_tr, ci_tr = bootstrap_ci(results[strategy]["rank"])
        summary[strategy] = {
            "n_pairs": len(results[strategy]["gap"]),
            "gap_mean": m_gap,
            "gap_ci_95": list(ci_gap),
            "win_rate_mean": m_wr,
            "win_rate_ci_95": list(ci_wr),
            "target_rank_pct_mean": m_tr,
            "target_rank_pct_ci_95": list(ci_tr),
            "gap_pass": bool(ci_gap[0] > 0),
            "wr_pass_60pct": bool(ci_wr[0] > 0.6),
            "rank_pass_50pct": bool(ci_tr[1] < 0.5),
            "all_pass": bool(ci_gap[0] > 0 and ci_wr[0] > 0.6 and ci_tr[1] < 0.5),
        }
        log(f"\n  Strategy {strategy}:")
        log(f"    Gap: mean={m_gap:.4f}, CI=[{ci_gap[0]:.4f},{ci_gap[1]:.4f}] {'PASS' if ci_gap[0]>0 else 'FAIL'}")
        log(f"    Win rate: mean={m_wr:.4f}, CI=[{ci_wr[0]:.4f},{ci_wr[1]:.4f}] {'PASS' if ci_wr[0]>0.6 else 'FAIL'}")
        log(f"    Target rank: mean={m_tr:.4f}, CI=[{ci_tr[0]:.4f},{ci_tr[1]:.4f}] {'PASS' if ci_tr[1]<0.5 else 'FAIL'}")
        log(f"    ALL PASS: {summary[strategy]['all_pass']}")

    # ControlHit aggregate
    m_ch, ci_ch = bootstrap_ci(control_hit_per_pair)
    log(f"\n  ControlHit (per pair, mean over ctrls):")
    log(f"    mean={m_ch:.4f}, CI=[{ci_ch[0]:.4f},{ci_ch[1]:.4f}]")

    # === Save eval ===
    log("\n[14] Saving eval ...")
    eval_dict = {
        "n_pairs": summary["A_min_d_target"]["n_pairs"],
        "n_candidates_per_pair_avg": len(candidates) // max(1, summary["A_min_d_target"]["n_pairs"]),
        "feature_space": "VADES 20d",
        "n_wrong_users_per_pair": N_WRONG_USERS_PER_PAIR,
        "n_controls_per_user_target": N_CONTROLS_PER_USER,
        "n_seeds_per_control": N_SEEDS_PER_CONTROL,
        "min_r_u_threshold": MIN_R_U_THRESHOLD,
        "summary_per_strategy": summary,
        "control_hit": {
            "mean": m_ch,
            "ci_95": list(ci_ch),
        },
        "comparison": {
            "Phase10_10_7_baseline_318d_win_rate": 0.5635,
            "Phase10_10_8_vades_20d_win_rate": 0.6041,
            "Phase10_12_B_max_margin_win_rate": 0.6724,
            "Phase10_12_B_rank": 0.4297,
        },
        "note": "Phase 10.12.C: Dynamic user-specific syntactic control. "
                "Each user's top-3 features by stable |r_u| drive 3 distinct control prompts. "
                "6 candidates/pair (3 controls x 2 seeds). "
                "Compared to Phase 10.12.B (12 global conditions, 24 candidates/pair).",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    # Per-pair (only strategy B + control_hit)
    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in results["B_max_margin"]["gap"].keys():
            row = {
                "user_id": uid,
                "asin": asin,
                "A_gap": results["A_min_d_target"]["gap"][(uid, asin)],
                "A_wr": results["A_min_d_target"]["wr"][(uid, asin)],
                "A_rank": results["A_min_d_target"]["rank"][(uid, asin)],
                "B_gap": results["B_max_margin"]["gap"][(uid, asin)],
                "B_wr": results["B_max_margin"]["wr"][(uid, asin)],
                "B_rank": results["B_max_margin"]["rank"][(uid, asin)],
                "C_gap": results["C_weighted"]["gap"][(uid, asin)],
                "C_wr": results["C_weighted"]["wr"][(uid, asin)],
                "C_rank": results["C_weighted"]["rank"][(uid, asin)],
                "control_hit": control_hit_per_pair[(uid, asin)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 10.12.C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()