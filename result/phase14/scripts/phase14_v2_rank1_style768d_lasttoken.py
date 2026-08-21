#!/usr/bin/env python3
"""Phase 14: Hybrid pipeline rank-1 eval in 768d AnnaWegmann style space.

User pivot from Phase 13.F: style space (NOT 318d syntactic) is the right eval
for StyleVector. Phase 13.D A_sampled had best-of-K style margin +0.098 vs D_off
in 768d style space. Phase 14 measures whether best-of-K candidate gets target
user ranked #1 among all 876 users in 768d space.

Pipeline:
  1. Load 960 candidates (13.D) + 768d encodings (13.F)
  2. Pooled Mahalanobis covariance on 876 user μ_768 (LW shrinkage)
  3. Per-(pair, cond): best-of-K rank-1 coverage using:
     - cosine distance to all 876 users
     - Mahalanobis distance to all 876 users
  4. Aggregate per cond: rank-1, top-10, top-100, mean_best_rank
  5. Paired bootstrap diff (A_sampled vs D_off/B_mean/C_shuffled)
  6. Verdict

Decision logic:
  GO         : A_sampled 768d-cosine rank-1 > 1.26% (Phase 10.15) CI excludes 0
                OR A_sampled top-100 > 17% (Phase 10.19)
  PARTIAL-GO : best-K margin lift in 13.F (+0.098 CI excludes 0) but rank-1 unchanged
  NO-GO      : no significant lift in style space
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_CANDIDATES = OUT_DIR / "phase13_d_v2_e2e_queries_lasttoken.jsonl"
IN_CAND_EMBS = OUT_DIR / "phase13_f_v2_cand_embs_768d_lasttoken.npy"
IN_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"

OUT_EVAL = OUT_DIR / "phase14_v2_rank1_eval_lasttoken.json"
OUT_PER_PAIR = OUT_DIR / "phase14_v2_per_pair_lasttoken.jsonl"
OUT_META = OUT_DIR / "phase14_v2_meta_lasttoken.json"

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]
SEED = 42
N_BOOTSTRAP = 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, tuple[float, float]]:
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
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
    log("Phase 14: Hybrid StyleVector + 768d style rerank rank-1 eval")
    log("=" * 70)

    # === [1] Load candidates + embs ===
    log("[1] Loading candidates + 768d encodings + user embs ...")
    candidates: list[dict] = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  candidates: {len(candidates)}")

    cand_embs = np.load(IN_CAND_EMBS).astype(np.float32)
    log(f"  cand_embs: {cand_embs.shape}")

    npz = np.load(IN_USER_EMBS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    user_embs = npz["embs"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}
    n_users = len(user_ids_arr)
    log(f"  user_embs: {user_embs.shape}, n_users={n_users}")

    if cand_embs.shape[0] != len(candidates):
        raise ValueError(
            f"cand_embs rows {cand_embs.shape[0]} != len(candidates) {len(candidates)}; "
            "13.F encoding order mismatch with 13.D jsonl"
        )

    # === [2] Pooled Mahalanobis covariance on user μ_768 ===
    log("[2] Pooled Mahalanobis cov (LW shrinkage) on 876 user μ_768 ...")
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(user_embs)
    cov_shrunk = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    log(f"  shrinkage: {shrinkage:.4f}, eigvals min/max: {eigvals.min():.4f}/{eigvals.max():.4f}")
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(cov_shrunk.shape[0]))
    quad_user = np.einsum('ij,jk,ik->i', user_embs, inv_cov, user_embs)

    # === [3] Cosine matrix precompute ===
    user_embs_norm = l2_normalize(user_embs)
    cand_embs_norm = l2_normalize(cand_embs)

    # === [4] Group by (uid, asin, cond) ===
    log("[4] Grouping candidates by (uid, asin, cond) ...")
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)
    log(f"  pairs: {len(cand_by_pair_cond)}")

    # === [5] Per-pair-per-cond rank-1 ===
    log("[5] Per-pair rerank (cosine + Mahalanobis) ...")
    per_pair_per_cond_cos: dict[tuple[str, str, str], int] = {}
    per_pair_per_cond_maha: dict[tuple[str, str, str], int] = {}

    n_skipped = 0
    for (uid, asin), conds_local in cand_by_pair_cond.items():
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            n_skipped += 1
            continue
        for cond, cand_indices in conds_local.items():
            local_embs = cand_embs[cand_indices]
            local_norm = cand_embs_norm[cand_indices]

            # Cosine distance to all 876 users
            cos_d = 1.0 - local_norm @ user_embs_norm.T  # (K, 876)
            target_d = cos_d[:, target_idx:target_idx + 1]
            cos_rank_per_cand = (cos_d < target_d).sum(axis=1)
            best_cos_rank = int(np.min(cos_rank_per_cand))

            # Mahalanobis distance to all 876 users
            quad_cand = np.einsum('ij,jk,ik->i', local_embs, inv_cov, local_embs)
            cross = local_embs @ inv_cov @ user_embs.T
            maha_d = quad_cand[:, None] + quad_user[None, :] - 2 * cross
            target_d_maha = maha_d[:, target_idx:target_idx + 1]
            maha_rank_per_cand = (maha_d < target_d_maha).sum(axis=1)
            best_maha_rank = int(np.min(maha_rank_per_cand))

            per_pair_per_cond_cos[(uid, asin, cond)] = best_cos_rank
            per_pair_per_cond_maha[(uid, asin, cond)] = best_maha_rank
    log(f"  done; skipped pairs: {n_skipped}")

    # === [6] Aggregate per condition ===
    log("[6] Per-condition aggregate ...")
    pair_keys = sorted({(uid, asin) for (uid, asin, _) in per_pair_per_cond_cos.keys()})
    log(f"  pairs (target in 876): {len(pair_keys)}")

    per_cond_eval: dict[str, dict] = {}
    for cond in CONDITIONS:
        ranks_cos = []
        ranks_maha = []
        for (uid, asin) in pair_keys:
            r_cos = per_pair_per_cond_cos.get((uid, asin, cond))
            r_maha = per_pair_per_cond_maha.get((uid, asin, cond))
            if r_cos is not None:
                ranks_cos.append(r_cos)
            if r_maha is not None:
                ranks_maha.append(r_maha)
        ranks_cos_arr = np.array(ranks_cos)
        ranks_maha_arr = np.array(ranks_maha)

        mean_cos, ci_cos = bootstrap_ci(ranks_cos)
        mean_maha, ci_maha = bootstrap_ci(ranks_maha)
        per_cond_eval[cond] = {
            "n_pairs": len(ranks_cos),
            "cosine": {
                "rank1_coverage": float((ranks_cos_arr == 0).sum()) / max(1, len(ranks_cos_arr)),
                "top10_coverage": float((ranks_cos_arr < 10).sum()) / max(1, len(ranks_cos_arr)),
                "top100_coverage": float((ranks_cos_arr < 100).sum()) / max(1, len(ranks_cos_arr)),
                "mean_best_rank": mean_cos,
                "mean_best_rank_ci95": list(ci_cos),
            },
            "maha": {
                "rank1_coverage": float((ranks_maha_arr == 0).sum()) / max(1, len(ranks_maha_arr)),
                "top10_coverage": float((ranks_maha_arr < 10).sum()) / max(1, len(ranks_maha_arr)),
                "top100_coverage": float((ranks_maha_arr < 100).sum()) / max(1, len(ranks_maha_arr)),
                "mean_best_rank": mean_maha,
                "mean_best_rank_ci95": list(ci_maha),
            },
        }
        log(
            f"  {cond}: n={len(ranks_cos)}, "
            f"cosine rank1={per_cond_eval[cond]['cosine']['rank1_coverage']*100:.1f}%, "
            f"top10={per_cond_eval[cond]['cosine']['top10_coverage']*100:.1f}%, "
            f"top100={per_cond_eval[cond]['cosine']['top100_coverage']*100:.1f}%, "
            f"mean_rank={mean_cos:.1f}; "
            f"maha rank1={per_cond_eval[cond]['maha']['rank1_coverage']*100:.1f}%, "
            f"top10={per_cond_eval[cond]['maha']['top10_coverage']*100:.1f}%, "
            f"top100={per_cond_eval[cond]['maha']['top100_coverage']*100:.1f}%, "
            f"mean_rank={mean_maha:.1f}"
        )

    # === [7] Paired bootstrap diff (A_sampled vs each baseline) ===
    log("[7] Paired bootstrap diff (A_sampled vs each baseline) ...")
    diffs: dict[str, dict] = {}
    for cond_b in CONDITIONS:
        if cond_b == "A_sampled":
            continue
        # cosine rank diff (positive = A better)
        diffs_cos = []
        diffs_maha = []
        for (uid, asin) in pair_keys:
            r_a_cos = per_pair_per_cond_cos.get((uid, asin, "A_sampled"))
            r_b_cos = per_pair_per_cond_cos.get((uid, asin, cond_b))
            r_a_maha = per_pair_per_cond_maha.get((uid, asin, "A_sampled"))
            r_b_maha = per_pair_per_cond_maha.get((uid, asin, cond_b))
            if r_a_cos is not None and r_b_cos is not None:
                diffs_cos.append(int(r_b_cos) - int(r_a_cos))  # positive = A better
            if r_a_maha is not None and r_b_maha is not None:
                diffs_maha.append(int(r_b_maha) - int(r_a_maha))
        mean_d_cos, ci_d_cos = bootstrap_ci(diffs_cos)
        mean_d_maha, ci_d_maha = bootstrap_ci(diffs_maha)
        diffs[f"A_sampled_vs_{cond_b}"] = {
            "cosine_rank_diff": {
                "mean": mean_d_cos,
                "ci95": list(ci_d_cos),
                "ci_excludes_0": ci_d_cos[0] > 0,
            },
            "maha_rank_diff": {
                "mean": mean_d_maha,
                "ci95": list(ci_d_maha),
                "ci_excludes_0": ci_d_maha[0] > 0,
            },
        }
        log(
            f"  A_sampled vs {cond_b}: "
            f"cosine_rank_diff={mean_d_cos:+.1f} CI {ci_d_cos[0]:+.1f}/{ci_d_cos[1]:+.1f} "
            f"({'excl 0' if ci_d_cos[0] > 0 else 'incl 0'}); "
            f"maha_rank_diff={mean_d_maha:+.1f} CI {ci_d_maha[0]:+.1f}/{ci_d_maha[1]:+.1f} "
            f"({'excl 0' if ci_d_maha[0] > 0 else 'incl 0'})"
        )

    # === [8] Verdict ===
    log("[8] Verdict ...")
    a_eval = per_cond_eval["A_sampled"]
    a_vs_d = diffs["A_sampled_vs_D_off"]
    a_vs_b = diffs["A_sampled_vs_B_mean"]

    # Use cosine as primary metric (matches Phase 13.F)
    if a_eval["cosine"]["rank1_coverage"] > 0.0126 and a_vs_d["cosine_rank_diff"]["ci_excludes_0"]:
        verdict = "GO"
        reason = f"A_sampled cosine rank-1 = {a_eval['cosine']['rank1_coverage']*100:.1f}% > 1.26% with paired CI > 0"
    elif a_eval["cosine"]["top100_coverage"] > 0.17:
        verdict = "GO"
        reason = f"A_sampled cosine top-100 = {a_eval['cosine']['top100_coverage']*100:.1f}% > 17%"
    elif a_vs_d["cosine_rank_diff"]["ci_excludes_0"]:
        verdict = "PARTIAL-GO"
        reason = f"A_sampled cosine best-rank lift over D_off CI > 0 but rank-1 unchanged at {a_eval['cosine']['rank1_coverage']*100:.1f}%"
    else:
        verdict = "NO-GO"
        reason = (
            f"No significant rank lift in 768d cosine: "
            f"rank-1 = {a_eval['cosine']['rank1_coverage']*100:.1f}%, "
            f"top-100 = {a_eval['cosine']['top100_coverage']*100:.1f}%, "
            f"A vs D rank_diff CI {a_vs_d['cosine_rank_diff']['ci95'][0]:+.1f}/{a_vs_d['cosine_rank_diff']['ci95'][1]:+.1f}"
        )
    log(f"  VERDICT: {verdict}")
    log(f"  REASON: {reason}")

    # === [9] Save ===
    out = {
        "phase": "14",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "k_per_cond": 8,
        "cov_pooled_method": "Ledoit-Wolf shrinkage",
        "cov_shrinkage": shrinkage,
        "eigval_min": float(eigvals.min()),
        "eigval_max": float(eigvals.max()),
        "per_cond_eval": per_cond_eval,
        "paired_diffs": diffs,
        "comparison": {
            "Phase10_15_318d_Maha_rank1_876pairs_96cands": 0.01256,
            "Phase10_19_318d_Maha_top100_30pairs": "5/30 = 17%",
            "Phase13_E_318d_Maha_rank1_30pairs": "0% (NO-GO)",
            "Phase13_F_768d_cosine_best_K_margin_A_sampled_vs_D_off": "+0.098 CI [0.016, 0.190]",
        },
        "decision_logic": {
            "GO": "A_sampled cosine rank-1 > 1.26% with paired CI > 0 OR top-100 > 17%",
            "PARTIAL-GO": "A_sampled best-rank lift CI > 0 but rank-1 unchanged",
            "NO-GO": "no significant rank lift in 768d style space",
        },
        "verdict": verdict,
        "verdict_reason": reason,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in pair_keys:
            row = {"user_id": uid, "asin": asin}
            for cond in CONDITIONS:
                row[f"{cond}_cosine_rank"] = per_pair_per_cond_cos.get((uid, asin, cond))
                row[f"{cond}_maha_rank"] = per_pair_per_cond_maha.get((uid, asin, cond))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "cov_pooled_method": "Ledoit-Wolf shrinkage",
        "cov_shrinkage": shrinkage,
        "verdict": verdict,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()