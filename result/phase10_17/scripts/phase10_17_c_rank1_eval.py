#!/usr/bin/env python3
"""Phase 10.16.C: 318d feature extraction + Rank-1 evaluation for 4-control generations.

Reuses the Mahalanobis + Ledoit-Wolf framework from Phase 10.15 to score the
soft-prefix-generated candidates (real-z / shuffled-z / true-zero-z / injection-off).

Goal: For each control, compute the rank-1 coverage (% of pairs whose best cand
ranks target user #1). Compare 4 controls:
  - real-z       : should beat baseline (injection-off) if soft prefix works
  - shuffled-z   : control — should NOT beat real-z (sanity check)
  - true-zero-z  : control — should ≈ injection-off (no signal)
  - injection-off: baseline (Phase 10.15-style)
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
GENERATIONS_FILE = OUT_DIR / "phase10_17_b_generations_4control.jsonl"
CAND_FEATS_OUT = OUT_DIR / "phase10_17_c_candidates_318d.npy"
CAND_BY_PAIR_OUT = OUT_DIR / "phase10_16_c_cand_by_pair.json"
EVAL_OUT = OUT_DIR / "phase10_17_c_rank1_eval.json"

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"

CONTROLS = ["real-z", "shuffled-z", "true-zero-z", "injection-off"]
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 10.17.C: 4-control rank-1 coverage evaluation (TinyStyler)")
    log("=" * 70)

    # === Load candidates ===
    log("[1] Loading candidates ...")
    candidates = []
    with GENERATIONS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")
    log(f"  by control:")
    ctr_count = defaultdict(int)
    for c in candidates:
        ctr_count[c["control"]] += 1
    for k, v in sorted(ctr_count.items()):
        log(f"    {k}: {v}")

    # === Load profiles + sentence features ===
    log("[2] Loading profiles + sentence features ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(user_id_to_idx)
    log(f"  VADES users: {n_users}")

    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    log(f"  effective dims: {n_dims_eff}")

    user_mu_318d = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for si in range(len(sent_users)):
        ui = user_id_to_idx.get(sent_users[si])
        if ui is None:
            continue
        user_mu_318d[ui] += sent_feats_keep[si]
        counts[ui] += 1
    valid_user_mask = counts > 0
    user_mu_318d[valid_user_mask] /= counts[valid_user_mask, None]
    log(f"  valid users: {valid_user_mask.sum()}/{n_users}")

    # === Ledoit-Wolf ===
    log("[3] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = lw.shrinkage_
    log(f"  shrinkage: {shrinkage:.4f}")

    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)

    # === spaCy batch feature extraction for all candidates ===
    log("[4] spaCy batch feature extraction for candidates ...")
    import spacy
    from extract_syntactic_features import per_sentence_features_v2, user_features_v2

    nlp = spacy.load("en_core_web_sm")
    try:
        from e22_t2_syntax_encoder_bridge import neutralize_content
        USE_NEUTRALIZE = True
    except ImportError:
        USE_NEUTRALIZE = False
    log(f"  USE_NEUTRALIZE: {USE_NEUTRALIZE}")

    cand_texts = [c.get("candidate_query") or "" for c in candidates]
    if USE_NEUTRALIZE:
        cand_neu = []
        for c, txt in zip(candidates, cand_texts):
            attrs = c.get("attrs") or {}
            cand_neu.append(neutralize_content(txt, attrs))
        docs = list(nlp.pipe(cand_neu, batch_size=128))
    else:
        docs = list(nlp.pipe(cand_texts, batch_size=128))
    cand_feats = np.zeros((len(candidates), 318), dtype=np.float64)
    n_skip = 0
    for i, doc in enumerate(docs):
        sfs = [per_sentence_features_v2(s) for s in doc.sents]
        sfs = [s for s in sfs if s is not None]
        if not sfs:
            n_skip += 1
            continue
        v = user_features_v2(sfs)
        if v is None:
            n_skip += 1
            continue
        cand_feats[i, :] = v
    log(f"  cand_feats: {cand_feats.shape}, skipped: {n_skip}")

    # Filter to keep_dims (matching user_mu_318d)
    cand_feats = cand_feats[:, keep_dims]
    np.save(CAND_FEATS_OUT, cand_feats)
    log(f"  saved → {CAND_FEATS_OUT}")

    # === Per-pair × per-control evaluation ===
    log("[5] Per-pair × per-control rank-1 coverage ...")
    # Group candidates by (user, asin, control)
    cand_by_pair_ctrl = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        cand_by_pair_ctrl[(c["user_id"], c["asin"])][c["control"]].append(ci)

    cand_by_pair_pairs = list(cand_by_pair_ctrl.keys())
    log(f"  pairs: {len(cand_by_pair_pairs)}")

    results = {}
    for ctrl in CONTROLS:
        log(f"\n--- control: {ctrl} ---")
        n_rank1 = 0
        n_total_pairs = 0
        per_pair_best_rank = {}

        for pi, ((uid, asin), ctrls) in enumerate(cand_by_pair_ctrl.items()):
            cand_indices = ctrls.get(ctrl, [])
            if not cand_indices:
                continue
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue
            n_total_pairs += 1

            cands_feats_local = cand_feats[cand_indices]
            cross = cands_feats_local @ inv_cov @ user_mu_318d.T
            quad_cand = np.einsum('ij,jk,ik->i', cands_feats_local, inv_cov, cands_feats_local)
            dist_local = quad_cand[:, None] + quad_mu[None, :] - 2 * cross

            d_target = dist_local[:, target_idx]
            target_d = d_target[:, None]
            target_rank = (dist_local < target_d).sum(axis=1)
            best_rank = int(target_rank.min())  # best = lowest rank achieved
            per_pair_best_rank[(uid, asin)] = best_rank
            if best_rank == 0:
                n_rank1 += 1

        coverage = n_rank1 / max(n_total_pairs, 1)
        log(f"  rank-1 coverage: {n_rank1}/{n_total_pairs} = {coverage:.4f}")
        mean_best_rank = float(np.mean(list(per_pair_best_rank.values()))) if per_pair_best_rank else 0.0
        log(f"  mean best rank: {mean_best_rank:.1f}")
        results[ctrl] = {
            "rank1_coverage": coverage,
            "rank1_count": n_rank1,
            "n_pairs": n_total_pairs,
            "mean_best_rank": mean_best_rank,
            "per_pair_best_rank": per_pair_best_rank,
        }
        if pi % 50 == 0:
            log(f"  progress: {pi}/{len(cand_by_pair_pairs)}")

    # === Aggregate comparison ===
    log("\n[6] Aggregate comparison (TinyStyler: K=8 alpha=2.0) ...")
    log(f"  Control          Rank-1 Cov   Mean Best Rank")
    log(f"  -------------------------------------------")
    base = results.get("injection-off", {}).get("rank1_coverage", 0)
    for ctrl in CONTROLS:
        cov = results[ctrl]["rank1_coverage"]
        mbr = results[ctrl]["mean_best_rank"]
        delta = cov - base
        log(f"  {ctrl:16s}  {cov:.4f}     {mbr:.1f}    (Δ vs off: {delta:+.4f})")

    # Save eval (without per_pair_best_rank to keep json small)
    eval_save = {}
    for ctrl in CONTROLS:
        eval_save[ctrl] = {
            "rank1_coverage": results[ctrl]["rank1_coverage"],
            "rank1_count": results[ctrl]["rank1_count"],
            "n_pairs": results[ctrl]["n_pairs"],
            "mean_best_rank": results[ctrl]["mean_best_rank"],
        }

    # Paired bootstrap CI: real-z vs shuffled-z
    realz = results["real-z"]["per_pair_best_rank"]
    shufz = results["shuffled-z"]["per_pair_best_rank"]
    off = results["injection-off"]["per_pair_best_rank"]

    common_keys = sorted(set(realz) & set(shufz) & set(off))
    if common_keys:
        rng = np.random.default_rng(SEED)
        n_b = 2000
        delta_real_vs_shuf = []
        delta_real_vs_off = []
        idx = np.arange(len(common_keys))
        for _ in range(n_b):
            sample = rng.choice(idx, size=len(idx), replace=True)
            ranks_real = np.array([realz[common_keys[i]] for i in sample])
            ranks_shuf = np.array([shufz[common_keys[i]] for i in sample])
            ranks_off = np.array([off[common_keys[i]] for i in sample])
            delta_real_vs_shuf.append(float((ranks_real == 0).mean() - (ranks_shuf == 0).mean()))
            delta_real_vs_off.append(float((ranks_real == 0).mean() - (ranks_off == 0).mean()))
        delta_real_vs_shuf = np.array(delta_real_vs_shuf)
        delta_real_vs_off = np.array(delta_real_vs_off)
        eval_save["paired_bootstrap"] = {
            "real_vs_shuffled_mean_delta": float(delta_real_vs_shuf.mean()),
            "real_vs_shuffled_ci95": [float(np.percentile(delta_real_vs_shuf, 2.5)),
                                       float(np.percentile(delta_real_vs_shuf, 97.5))],
            "real_vs_off_mean_delta": float(delta_real_vs_off.mean()),
            "real_vs_off_ci95": [float(np.percentile(delta_real_vs_off, 2.5)),
                                  float(np.percentile(delta_real_vs_off, 97.5))],
            "n_paired_pairs": len(common_keys),
        }
        log(f"\n  Paired bootstrap (real-z vs shuffled-z):")
        log(f"    Δ mean = {delta_real_vs_shuf.mean():+.4f}, "
            f"95% CI [{delta_real_vs_shuf.mean() - 1.96*delta_real_vs_shuf.std():+.4f}, "
            f"{delta_real_vs_shuf.mean() + 1.96*delta_real_vs_shuf.std():+.4f}]")
        log(f"  Paired bootstrap (real-z vs off):")
        log(f"    Δ mean = {delta_real_vs_off.mean():+.4f}")

    EVAL_OUT.write_text(json.dumps(eval_save, ensure_ascii=False, indent=2))
    log(f"  saved → {EVAL_OUT}")

    log("=" * 70)
    log("PHASE 10.17.C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()