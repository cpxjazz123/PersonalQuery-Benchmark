#!/usr/bin/env python3
"""Phase 13.E: Rank-1 evaluation on Phase 13.D candidates, per condition.

For each of 4 conditions (A_sampled, B_mean, C_shuffled, D_off):
  1. Filter 960 candidates by condition
  2. Extract 318d features (spaCy batched)
  3. VADES 318d Mahalanobis rerank (reuse Phase 10.15 user_mu + LW covariance)
  4. Compute per-pair rank-1 coverage, mean best rank

Decision:
  GO         : A_sampled rank-1 > 1.26% (Phase 10.15) CI excludes 0
  PARTIAL-GO : A_sampled rank-1 ~ 1.26% but top-100 > 17% (Phase 10.19)
  NO-GO      : A_sampled rank-1 ~ 0%, pivot to contrastive RAG mainline
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
IN_CANDIDATES = OUT_DIR / "phase13_d_e2e_queries.jsonl"
OUT_FEATS = OUT_DIR / "phase13_e_candidates_318d.npy"
OUT_EVAL = OUT_DIR / "phase13_e_rank1_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase13_e_per_pair.jsonl"

CACHE_DIR = OUT_DIR / "phase10_10_6_cache"
SENT_FEATS_CACHE = CACHE_DIR / "sentence_318d.npy"
SENT_USERS_CACHE = CACHE_DIR / "sentence_318d_users.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"

SEED = 42
N_BOOTSTRAP = 2000
CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values_per_item: dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
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
    return obs_mean, (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def main():
    log("=" * 70)
    log("Phase 13.E: Rank-1 evaluation per condition (A/B/C/D)")
    log("=" * 70)

    # === Load 13.D candidates ===
    log("[1] Loading 13.D candidates ...")
    candidates = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  total candidates: {len(candidates)}")
    log(f"  conditions: {set(c['condition'] for c in candidates)}")

    # === Load user profiles + sentence features (cached 318d) ===
    log("[2] Loading user profiles + cached 318d features ...")
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

    # === Ledoit-Wolf covariance ===
    log("[3] Ledoit-Wolf covariance ...")
    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu_318d[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    log(f"  shrinkage: {shrinkage:.4f}")

    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    quad_mu = np.einsum('ij,jk,ik->i', user_mu_318d, inv_cov, user_mu_318d)
    log(f"  inv_cov: {inv_cov.shape}")

    # === spaCy batch feature extraction ===
    log("[4] spaCy batch feature extraction for 13.D candidates ...")
    import spacy
    from extract_syntactic_features import per_sentence_features_v2, user_features_v2
    nlp = spacy.load("en_core_web_sm")
    try:
        from e22_t2_syntax_encoder_bridge import neutralize_content
        USE_NEUTRALIZE = True
    except ImportError:
        USE_NEUTRALIZE = False
    log(f"  USE_NEUTRALIZE: {USE_NEUTRALIZE}")

    cand_texts = [c.get("q_final_post") or c.get("q_styled") or "" for c in candidates]
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
    cand_feats_keep = cand_feats[:, keep_dims]
    np.save(OUT_FEATS, cand_feats_keep)
    log(f"  saved → {OUT_FEATS}")

    # === Per-condition rank-1 evaluation ===
    log("[5] Per-condition evaluation ...")
    per_cond_eval: dict[str, dict] = {}
    per_pair_all: list[dict] = []

    for cond in CONDITIONS:
        log(f"  --- condition {cond} ---")
        cond_idx = [i for i, c in enumerate(candidates) if c["condition"] == cond]
        log(f"    candidates: {len(cond_idx)}")

        cand_by_pair = defaultdict(list)
        for ci in cond_idx:
            c = candidates[ci]
            cand_by_pair[(c["user_id"], c["asin"])].append(ci)

        n_rank1 = 0
        n_rank3 = 0
        n_rank5 = 0
        n_rank10 = 0
        n_margin_pos = 0
        n_pass_all = 0
        pair_data_cond = {}

        for (uid, asin), cand_indices in cand_by_pair.items():
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue
            cands_feats_local = cand_feats_keep[cand_indices]
            n_cands = len(cand_indices)

            cross = cands_feats_local @ inv_cov @ user_mu_318d.T
            quad_cand = np.einsum('ij,jk,ik->i', cands_feats_local, inv_cov, cands_feats_local)
            dist_local = quad_cand[:, None] + quad_mu[None, :] - 2 * cross
            d_target = dist_local[:, target_idx]

            target_d = d_target[:, None]
            target_rank = (dist_local < target_d).sum(axis=1)

            rng = np.random.default_rng(SEED)
            mask = np.ones(n_users, dtype=bool)
            mask[target_idx] = False
            wrong_idx = rng.choice(np.arange(n_users)[mask], size=5, replace=False)
            d_wrong = dist_local[:, wrong_idx]
            d_wrong_mean = d_wrong.mean(axis=1)
            margin = d_wrong_mean - d_target
            best_idx = int(np.argmax(margin))
            best_rank = int(target_rank[best_idx])
            best_margin = float(margin[best_idx])

            any_rank1 = int((target_rank == 0).any())
            n_rank1 += any_rank1
            n_rank3 += int((target_rank < 3).any())
            n_rank5 += int((target_rank < 5).any())
            n_rank10 += int((target_rank < 10).any())
            if best_margin > 0:
                n_margin_pos += 1
            if best_rank == 0 and best_margin > 0:
                n_pass_all += 1

            rank1_mask = (target_rank == 0)
            if rank1_mask.any():
                margin_among_rank1 = np.where(rank1_mask, margin, -np.inf)
                rank1_pick = int(np.argmax(margin_among_rank1))
                rank1_pick_margin = float(margin[rank1_pick])
            else:
                rank1_pick = None
                rank1_pick_margin = None

            pair_data_cond[(uid, asin)] = {
                "n_cands": n_cands,
                "best_idx": best_idx,
                "best_rank": best_rank,
                "best_margin": best_margin,
                "any_rank1": any_rank1,
                "any_rank3": int((target_rank < 3).any()),
                "rank1_pick_idx": rank1_pick,
                "rank1_pick_margin": rank1_pick_margin,
            }
            per_pair_all.append({
                "condition": cond,
                "user_id": uid,
                "asin": asin,
                "n_cands": n_cands,
                "best_rank": best_rank,
                "best_margin": best_margin,
                "any_rank1": any_rank1,
                "any_rank3": int((target_rank < 3).any()),
                "rank1_pick_idx": rank1_pick,
                "rank1_pick_margin": rank1_pick_margin,
            })
            del dist_local

        n_total = len(pair_data_cond)
        per_pair_best_rank = {k: v["best_rank"] for k, v in pair_data_cond.items()}
        per_pair_best_margin = {k: v["best_margin"] for k, v in pair_data_cond.items()}
        per_pair_any_rank1 = {k: v["any_rank1"] for k, v in pair_data_cond.items()}
        mean_rank, ci_rank = bootstrap_ci(per_pair_best_rank)
        mean_margin, ci_margin = bootstrap_ci(per_pair_best_margin)
        mean_any_rank1, ci_any_rank1 = bootstrap_ci(per_pair_any_rank1)

        per_cond_eval[cond] = {
            "n_pairs": n_total,
            "rank1_coverage": n_rank1 / n_total,
            "rank3_coverage": n_rank3 / n_total,
            "rank5_coverage": n_rank5 / n_total,
            "rank10_coverage": n_rank10 / n_total,
            "best_margin_positive": n_margin_pos / n_total,
            "best_rank1_and_margin_positive": n_pass_all / n_total,
            "mean_best_rank": mean_rank,
            "mean_best_rank_ci_95": list(ci_rank),
            "mean_best_margin": mean_margin,
            "mean_best_margin_ci_95": list(ci_margin),
            "mean_any_rank1": mean_any_rank1,
            "mean_any_rank1_ci_95": list(ci_any_rank1),
        }
        log(f"    rank-1: {n_rank1}/{n_total} ({100*n_rank1/n_total:.1f}%)")
        log(f"    top-10: {n_rank10}/{n_total} ({100*n_rank10/n_total:.1f}%)")
        log(f"    mean rank: {mean_rank:.1f} [{ci_rank[0]:.1f}, {ci_rank[1]:.1f}]")

    # === Save eval ===
    log(f"[6] Saving eval to {OUT_EVAL} ...")
    eval_dict = {
        "phase": "13.E",
        "n_candidates_per_condition": len([c for c in candidates if c["condition"] == "A_sampled"]),
        "n_conditions": len(CONDITIONS),
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "distance_method": "Mahalanobis + Ledoit-Wolf shrinkage",
        "shrinkage_intensity": shrinkage,
        "per_condition": per_cond_eval,
        "comparison": {
            "Phase10_15_876pairs_96cands_rank1": 0.012557077625570776,
            "Phase10_15_top10": 0.09246575342465753,
            "Phase10_19_30pairs_contrastive_rag_top100": "5/30 = 17%",
            "Phase11_D_30pairs_20cands_rank1_pct": "0%",
            "Phase11_G_v2_30pairs_20cands_rank1_pct": "0%",
        },
        "decision_logic": {
            "GO": "A_sampled rank-1 > 1.26% (Phase 10.15) CI excludes 0",
            "PARTIAL-GO": "A_sampled rank-1 ~ 1.26% but top-100 > 17% (Phase 10.19)",
            "NO-GO": "A_sampled rank-1 ~ 0%, pivot to contrastive RAG",
        },
        "verdict": None,  # filled below
    }

    a_eval = per_cond_eval["A_sampled"]
    if a_eval["rank1_coverage"] > 0.025:
        verdict = "GO"
    elif a_eval["rank10_coverage"] > 0.17:
        verdict = "PARTIAL-GO"
    else:
        verdict = "NO-GO"
    eval_dict["verdict"] = verdict
    log(f"\n  FINAL VERDICT: {verdict}")
    log(f"    A_sampled rank-1: {100*a_eval['rank1_coverage']:.1f}% (target > 1.26%)")
    log(f"    A_sampled top-10: {100*a_eval['rank10_coverage']:.1f}% (target > 17%)")

    OUT_EVAL.write_text(json.dumps(eval_dict, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in per_pair_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 13.E COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()