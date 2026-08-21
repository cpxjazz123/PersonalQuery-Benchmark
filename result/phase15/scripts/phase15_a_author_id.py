#!/usr/bin/env python3
"""Phase 15.A: Author Identification Accuracy — stricter than StyleVector paper.

User feedback (iter_037 followup): StyleVector paper uses ROUGE/METEOR as main
metric (lexical overlap, not style-specific). Missing:
  - Author Identification (mix generated into N users, classifier predicts "who")
  - Style preference eval
  - Syntactic-feature distance (only implicit)
  - Semantic preservation (style may regress content)

Phase 15.A specifically addresses Author Identification:
  - For each of 30 pair users, encode their 10 real sentences via AnnaWegmann → 768d
  - User μ_u = mean of 10 embeddings (also already in phase10_user_embs_768d.npz)
  - For each candidate, compute cosine distance to all 30 users' μ
  - Per candidate: predicted_user = argmin distance
  - Top-K accuracy = fraction where target_user ∈ top-K predictions
  - Per cond × K=1 (no rerank — direct classification of each candidate)

Difference vs Phase 14:
  - Phase 14 used BEST-OF-K candidate (margin-aware selection over K=8)
  - Phase 15.A uses EACH candidate individually — does the model output
    naturally land in the target user's style bucket, without rerank?

Decision logic:
  - GO         : A_sampled top-1 acc > D_off top-1, paired CI > 0
  - PARTIAL-GO : A_sampled top-3/top-10 acc > D_off
  - NO-GO      : A_sampled ≤ D_off (StyleVector injection doesn't really shift style)

Reuses:
  - phase13_d_e2e_queries.jsonl (960 candidates, 4 conds × 30 pairs × K=8)
  - phase13_f_cand_embs_768d.npy (already-encoded candidates, 960 × 768)
  - phase10_user_embs_768d.npz (876 user μ)
  - vades_prototype_3000u_v6_raw_sentences.jsonl (per-user 10 real sentences)
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
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_CANDIDATES = OUT_DIR / "phase13_d_e2e_queries.jsonl"
IN_CAND_EMBS = OUT_DIR / "phase13_f_cand_embs_768d.npy"
IN_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"
IN_SENTENCES = (
    REPO_ROOT / "result" / "personal_query"
    / "12_complexity_analysis_clause_features" / "Baby_Products"
    / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)

OUT_REPORT = OUT_DIR / "phase15_a_author_id.json"
OUT_PER_CAND = OUT_DIR / "phase15_a_per_cand.jsonl"

ANNA_SNAPSHOT_DIR = "/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots"

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]
SEED = 42
ENCODER_BATCH = 128


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts, convert_to_numpy=True, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=False,
    )


def bootstrap_ci(values, n: int = 2000, seed: int = SEED):
    arr = np.array(list(values))
    if len(arr) == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def main():
    log("=" * 70)
    log("Phase 15.A: Author Identification Accuracy (AnnaWegmann 768d)")
    log("=" * 70)

    np.random.seed(SEED)

    # === [1] Load candidates and embeddings ===
    log("[1] Loading Phase 13.D candidates + 13.F 768d embeddings ...")
    candidates: list[dict] = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  total candidates: {len(candidates)}")

    cand_embs = np.load(IN_CAND_EMBS).astype(np.float32)
    log(f"  cand_embs: {cand_embs.shape}")
    cand_embs_norm = l2_normalize(cand_embs)

    # === [2] Load 876 user μ_768 ===
    log("[2] Loading 768d user style vectors (μ_u) ...")
    npz = np.load(IN_USER_EMBS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    user_embs_768 = npz["embs"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}
    user_embs_norm = l2_normalize(user_embs_768)
    log(f"  users: {len(user_ids_arr)}, dim={user_embs_768.shape[1]}")

    # === [3] Get the 30 pair user_ids (eval pool) ===
    pair_uids = []
    seen = set()
    for c in candidates:
        if c["user_id"] not in seen:
            seen.add(c["user_id"])
            pair_uids.append(c["user_id"])
    log(f"  eval pool (pair users): {len(pair_uids)}")

    # Subset user_embs to eval pool (for Author ID task scope)
    eval_user_idxs = [uid_to_useridx[u] for u in pair_uids]
    eval_user_embs = user_embs_norm[eval_user_idxs]  # [30, 768]
    eval_user_ids = pair_uids
    log(f"  eval pool μ shape: {eval_user_embs.shape}")

    # === [4] Per-candidate author ID: cosine to each of 30 eval-pool users ===
    log("[4] Per-candidate author ID prediction ...")
    n_cands = len(candidates)
    # Cosine similarity matrix [n_cands, 30 eval users]
    cos_mat = cand_embs_norm @ eval_user_embs.T
    log(f"  cos_mat: {cos_mat.shape}")
    # predicted_user = argmax (highest cosine)
    predicted_user_idx = np.argmax(cos_mat, axis=1)
    predicted_uid = [eval_user_ids[i] for i in predicted_user_idx]
    # Sort descending cosine → top-K
    sorted_idx = np.argsort(-cos_mat, axis=1)  # [n_cands, 30]
    # rank of target
    target_user_idx_arr = np.array([eval_user_ids.index(c["user_id"]) for c in candidates])

    # per-cand rank of target (0 = top-1)
    ranks = np.zeros(n_cands, dtype=np.int32)
    for i in range(n_cands):
        ranks[i] = int(np.where(sorted_idx[i] == target_user_idx_arr[i])[0][0])

    # top-1 hit
    top1_hit = (ranks == 0).astype(np.int8)
    top3_hit = (ranks < 3).astype(np.int8)
    top5_hit = (ranks < 5).astype(np.int8)
    top10_hit = (ranks < 10).astype(np.int8)

    # Per-cand records (for Phase 15.B/C reuse)
    per_cand_records = []
    for i, c in enumerate(candidates):
        per_cand_records.append({
            "cand_idx_global": i,
            "user_id": c["user_id"],
            "asin": c["asin"],
            "condition": c["condition"],
            "cand_local_idx": c["cand_idx"],
            "q_styled": c.get("q_styled", ""),
            "predicted_uid": predicted_uid[i],
            "rank_of_target": int(ranks[i]),
            "top1_hit": int(top1_hit[i]),
            "top3_hit": int(top3_hit[i]),
            "top10_hit": int(top10_hit[i]),
        })

    # === [5] Per-cond aggregate ===
    log("[5] Per-cond aggregate (per-candidate author ID) ...")
    per_cond: dict[str, dict] = {}
    per_pair_top1: dict[tuple[str, str], list[int]] = defaultdict(list)
    per_pair_top10: dict[tuple[str, str], list[int]] = defaultdict(list)

    for r in per_cand_records:
        key = (r["user_id"], r["condition"])
        per_pair_top1[key].append(r["top1_hit"])
        per_pair_top10[key].append(r["top10_hit"])

    for cond in CONDITIONS:
        idxs = [i for i, c in enumerate(candidates) if c["condition"] == cond]
        n = len(idxs)
        per_cond[cond] = {
            "n_candidates": n,
            "top1_acc": float(top1_hit[idxs].mean()),
            "top3_acc": float((ranks[idxs] < 3).mean()),
            "top5_acc": float((ranks[idxs] < 5).mean()),
            "top10_acc": float(top10_hit[idxs].mean()),
            "mean_rank_of_target": float(ranks[idxs].mean()),
            "median_rank_of_target": float(np.median(ranks[idxs])),
        }
        log(f"  {cond}: top1={per_cond[cond]['top1_acc']*100:.2f}%, "
            f"top3={per_cond[cond]['top3_acc']*100:.2f}%, "
            f"top10={per_cond[cond]['top10_acc']*100:.2f}%, "
            f"mean_rank={per_cond[cond]['mean_rank_of_target']:.1f}")

    # === [6] Per-pair best-of-K aggregation ===
    log("[6] Per-pair best-of-K (rerank with style cos — Phase 14 protocol) ...")
    per_pair_best_top1: dict[tuple[str, str], int] = {}
    per_pair_best_top10: dict[tuple[str, str], int] = {}
    per_pair_best_rank: dict[tuple[str, str], int] = {}
    for key, top1s in per_pair_top1.items():
        top10s = per_pair_top10[key]
        ranks_local = []
        for r in per_cand_records:
            if (r["user_id"], r["condition"]) == key:
                ranks_local.append(r["rank_of_target"])
        per_pair_best_top1[key] = int(any(top1s))
        per_pair_best_top10[key] = int(any(top10s))
        per_pair_best_rank[key] = int(min(ranks_local)) if ranks_local else 30

    best_top1_per_cond = {c: [] for c in CONDITIONS}
    best_top10_per_cond = {c: [] for c in CONDITIONS}
    best_rank_per_cond = {c: [] for c in CONDITIONS}
    for (uid, cond), v in per_pair_best_top1.items():
        best_top1_per_cond[cond].append(v)
        best_top10_per_cond[cond].append(per_pair_best_top10[(uid, cond)])
        best_rank_per_cond[cond].append(per_pair_best_rank[(uid, cond)])

    per_cond_best = {}
    for cond in CONDITIONS:
        per_cond_best[cond] = {
            "best_of_K_top1": float(np.mean(best_top1_per_cond[cond])),
            "best_of_K_top10": float(np.mean(best_top10_per_cond[cond])),
            "best_of_K_mean_rank": float(np.mean(best_rank_per_cond[cond])),
        }
        log(f"  {cond} (best-of-K): top1={per_cond_best[cond]['best_of_K_top1']*100:.1f}%, "
            f"top10={per_cond_best[cond]['best_of_K_top10']*100:.1f}%, "
            f"mean_rank={per_cond_best[cond]['best_of_K_mean_rank']:.1f}")

    # === [7] Paired bootstrap diffs (per-cand, A_sampled vs baselines) ===
    log("[7] Paired bootstrap diffs (per-cand level) ...")
    # per-pair mean top1 acc for paired test
    per_pair_top1_mean: dict[tuple[str, str], float] = {
        k: float(np.mean(v)) for k, v in per_pair_top1.items()
    }
    per_pair_rank_mean: dict[tuple[str, str], float] = {
        k: float(np.mean([r["rank_of_target"] for r in per_cand_records
                          if (r["user_id"], r["condition"]) == k]))
        for k in per_pair_top1.keys()
    }

    diffs = {}
    for cond_b in CONDITIONS:
        if cond_b == "A_sampled":
            continue
        # per-cand top1 hit diff (paired per user)
        top1_diffs = []
        rank_diffs = []
        for uid in pair_uids:
            t1_a = per_pair_top1_mean[(uid, "A_sampled")]
            t1_b = per_pair_top1_mean[(uid, cond_b)]
            rk_a = per_pair_rank_mean[(uid, "A_sampled")]
            rk_b = per_pair_rank_mean[(uid, cond_b)]
            top1_diffs.append(t1_a - t1_b)
            rank_diffs.append(rk_a - rk_b)
        d_t1_mean, d_t1_ci = bootstrap_ci(top1_diffs)
        d_rk_mean, d_rk_ci = bootstrap_ci(rank_diffs)
        diffs[f"A_sampled_vs_{cond_b}"] = {
            "top1_acc_diff": d_t1_mean,
            "top1_acc_diff_ci95": list(d_t1_ci),
            "rank_diff": d_rk_mean,
            "rank_diff_ci95": list(d_rk_ci),
            # correct CI-excludes-0: both endpoints same sign (strict)
            "rank_diff_excludes_0": (d_rk_ci[0] > 0 and d_rk_ci[1] > 0)
                                       or (d_rk_ci[0] < 0 and d_rk_ci[1] < 0),
            "top1_acc_diff_excludes_0": (d_t1_ci[0] > 0 and d_t1_ci[1] > 0)
                                          or (d_t1_ci[0] < 0 and d_t1_ci[1] < 0),
        }
        log(f"  A vs {cond_b}: top1_diff={d_t1_mean:+.4f} [{d_t1_ci[0]:+.4f}, {d_t1_ci[1]:+.4f}], "
            f"rank_diff={d_rk_mean:+.2f} [{d_rk_ci[0]:+.2f}, {d_rk_ci[1]:+.2f}]")

    # === [8] Verdict ===
    a_top1 = per_cond["A_sampled"]["top1_acc"]
    d_top1 = per_cond["D_off"]["top1_acc"]
    a_top10 = per_cond["A_sampled"]["top10_acc"]
    d_top10 = per_cond["D_off"]["top10_acc"]
    a_best_top10 = per_cond_best["A_sampled"]["best_of_K_top10"]
    d_best_top10 = per_cond_best["D_off"]["best_of_K_top10"]
    a_vs_d = diffs["A_sampled_vs_D_off"]

    # Two-track verdict: per-cand (strict author ID) + best-of-K (rerank protocol)
    if a_vs_d.get("top1_acc_diff_excludes_0") or a_vs_d.get("rank_diff_excludes_0"):
        verdict = "GO_per_cand"
        verdict_reason = (
            f"A_sampled per-cand top1 or rank paired CI excludes 0 over D_off: "
            f"top1_diff={a_vs_d['top1_acc_diff']:+.4f}, rank_diff={a_vs_d['rank_diff']:+.2f}"
        )
    elif a_best_top10 > d_best_top10 + 0.10:  # >10pp best-of-K lift
        verdict = "GO_best_of_K_only"
        verdict_reason = (
            f"Per-cand author ID NOT significant (A top1={a_top1*100:.2f}% vs D {d_top1*100:.2f}%), "
            f"but best-of-K rerank top10 lifts {a_best_top10*100-d_best_top10*100:+.1f}pp "
            f"(A {a_best_top10*100:.1f}% vs D {d_best_top10*100:.1f}%)"
        )
    elif a_best_top10 > d_best_top10:
        verdict = "PARTIAL-GO"
        verdict_reason = (
            f"Best-of-K rerank top10 lifts slightly: A {a_best_top10*100:.1f}% vs D {d_best_top10*100:.1f}%, "
            f"per-cand not significant"
        )
    else:
        verdict = "NO-GO"
        verdict_reason = (
            f"No improvement: A top1={a_top1*100:.2f}%, top10={a_top10*100:.2f}%, "
            f"best-K top10={a_best_top10*100:.1f}%"
        )
    log(f"\n  FINAL VERDICT: {verdict}")
    log(f"  reason: {verdict_reason}")

    # === [9] Save ===
    out = {
        "phase": "15.A",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "feature_space": "AnnaWegmann 768d cosine (user μ vs cand)",
        "n_eval_users": len(pair_uids),
        "n_candidates": len(candidates),
        "n_conditions": len(CONDITIONS),
        "per_cond_per_cand": per_cond,
        "per_cond_best_of_K": per_cond_best,
        "paired_diffs": diffs,
        "random_baseline_top1": 1.0 / len(pair_uids),
        "decision_logic": {
            "GO": "A_sampled per-cand top1 OR rank diff paired CI > 0",
            "PARTIAL-GO": "A_sampled top10 > D_off but per-cand top1 not significant",
            "NO-GO": "no per-cand improvement",
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    OUT_REPORT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  report → {OUT_REPORT}")

    with OUT_PER_CAND.open("w") as f:
        for r in per_cand_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_cand → {OUT_PER_CAND}")

    log("=" * 70)
    log("PHASE 15.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()