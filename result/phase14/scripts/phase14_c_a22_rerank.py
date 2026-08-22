#!/usr/bin/env python3
"""Phase 14.C: A22_a0.5 + 768d style rerank rank-1 eval on Phase 14.B's 30 pairs.

Goal: test if layer-22 α=0.5 (sweet spot from Phase 14.B) beats Phase 14 hybrid
(A14_a1.0) on the SAME 30 pairs.

Pipeline:
  1. Filter Phase 14.B jsonl to A22_a0.5 + A14_a1.0 + D_off (3 conds × 30 × 8 = 720)
  2. Encode q_styled → 768d via AnnaWegmann
  3. Load 876 user μ_768 + pooled Maha (LW shrinkage)
  4. Per-(pair, cond) best-of-K rerank (cosine + Maha)
  5. Aggregate per cond: rank-1, top-10, top-100, mean rank
  6. Paired bootstrap: A22_a0.5 vs A14_a1.0 vs D_off
  7. Verdict

Within-experiment fair comparison: all 3 share the SAME 30 (user, asin) pairs.
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

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"

OUT_EVAL = OUT_DIR / "phase14_c_a22_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_c_a22_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_c_a22_rerank_meta.json"
OUT_EMB_CANDS = OUT_DIR / "phase14_c_a22_cand_embs_768d.npy"

ANNA_SNAPSHOT_DIR = "/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]  # A22 = sweet spot, A14 = Phase 14 hybrid, D_off = no injection
SEED = 42
N_BOOTSTRAP = 2000
ENCODER_BATCH = 128


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED):
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


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts, convert_to_numpy=True, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=False,
    )


def main() -> None:
    log("=" * 70)
    log("Phase 14.C: A22_a0.5 + 768d style rerank vs Phase 14 hybrid (A14_a1.0)")
    log("=" * 70)

    # === [1] Load + filter Phase 14.B jsonl ===
    log("[1] Loading Phase 14.B jsonl + filtering to A22_a0.5 / A14_a1.0 / D_off ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    log(f"  filtered candidates: {len(candidates)} (expected {30 * 3 * 8} = 720)")
    conds_present = sorted({c["condition"] for c in candidates})
    log(f"  conditions: {conds_present}")
    n_pairs = len({(c["user_id"], c["asin"]) for c in candidates})
    log(f"  pairs: {n_pairs} (expected 30)")

    # === [2] Load user 768d ===
    log("[2] Loading 768d user embs ...")
    npz = np.load(IN_USER_EMBS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    user_embs = npz["embs"].astype(np.float32) if "embs" in npz.files else npz["vecs"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}
    n_users = len(user_ids_arr)
    log(f"  users: {n_users}, emb_dim={user_embs.shape[1]}")

    # === [3] Load AnnaWegmann encoder ===
    log("[3] Loading AnnaWegmann/Style-Embedding ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import glob
    snapshots = sorted(glob.glob(os.path.join(ANNA_SNAPSHOT_DIR, "*")))
    if not snapshots:
        raise RuntimeError(f"No AnnaWegmann snapshots in {ANNA_SNAPSHOT_DIR}")
    snapshot = snapshots[-1]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(snapshot, device="cuda:0")
    log(f"  using snapshot: {snapshot}, dim={model.get_sentence_embedding_dimension()}")

    # === [4] Encode candidates ===
    log("[4] Encoding q_styled → 768d ...")
    cand_texts = [c.get("q_styled") or "" for c in candidates]
    t0 = time.time()
    cand_embs = encode_texts(model, cand_texts, ENCODER_BATCH).astype(np.float32)
    log(f"  encoded {len(cand_texts)} in {time.time()-t0:.1f}s, shape={cand_embs.shape}")
    np.save(OUT_EMB_CANDS, cand_embs)
    log(f"  saved → {OUT_EMB_CANDS}")

    # === [5] Pooled Mahalanobis (LW shrinkage) on user μ_768 ===
    log("[5] Pooled Mahalanobis cov (LW shrinkage) on 876 user μ_768 ...")
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(user_embs)
    cov_shrunk = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    log(f"  shrinkage: {shrinkage:.4f}, eigvals min/max: {eigvals.min():.4f}/{eigvals.max():.4f}")
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(cov_shrunk.shape[0]))
    quad_user = np.einsum('ij,jk,ik->i', user_embs, inv_cov, user_embs)
    log(f"  inv_cov shape: {inv_cov.shape}")

    # === [6] Cosine matrix precompute ===
    user_embs_norm = l2_normalize(user_embs)
    cand_embs_norm = l2_normalize(cand_embs)

    # === [7] Group by (uid, asin, cond) ===
    log("[7] Grouping candidates by (uid, asin, cond) ...")
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)
    log(f"  pairs: {len(cand_by_pair_cond)}")

    # === [8] Per-pair-per-cond rerank ===
    log("[8] Per-pair rerank (cosine + Mahalanobis) ...")
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

            cos_d = 1.0 - local_norm @ user_embs_norm.T
            target_d = cos_d[:, target_idx:target_idx + 1]
            cos_rank_per_cand = (cos_d < target_d).sum(axis=1)
            best_cos_rank = int(np.min(cos_rank_per_cand))

            quad_cand = np.einsum('ij,jk,ik->i', local_embs, inv_cov, local_embs)
            cross = local_embs @ inv_cov @ user_embs.T
            maha_d = quad_cand[:, None] + quad_user[None, :] - 2 * cross
            target_d_maha = maha_d[:, target_idx:target_idx + 1]
            maha_rank_per_cand = (maha_d < target_d_maha).sum(axis=1)
            best_maha_rank = int(np.min(maha_rank_per_cand))

            per_pair_per_cond_cos[(uid, asin, cond)] = best_cos_rank
            per_pair_per_cond_maha[(uid, asin, cond)] = best_maha_rank
    log(f"  done; skipped pairs: {n_skipped}")

    # === [9] Aggregate per condition ===
    log("[9] Per-condition aggregate ...")
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
            f"cos rank1={per_cond_eval[cond]['cosine']['rank1_coverage']*100:.1f}%, "
            f"top10={per_cond_eval[cond]['cosine']['top10_coverage']*100:.1f}%, "
            f"top100={per_cond_eval[cond]['cosine']['top100_coverage']*100:.1f}%, "
            f"mean_rank={mean_cos:.1f}; "
            f"maha rank1={per_cond_eval[cond]['maha']['rank1_coverage']*100:.1f}%, "
            f"top100={per_cond_eval[cond]['maha']['top100_coverage']*100:.1f}%, "
            f"mean_rank={mean_maha:.1f}"
        )

    # === [10] Paired bootstrap diff (A22_a0.5 vs A14_a1.0 vs D_off) ===
    log("[10] Paired bootstrap diff (A22_a0.5 vs each baseline) ...")
    diffs: dict[str, dict] = {}
    for cond_b in CONDITIONS:
        if cond_b == "A22_a0.5":
            continue
        diffs_cos = []
        diffs_maha = []
        for (uid, asin) in pair_keys:
            r_a_cos = per_pair_per_cond_cos.get((uid, asin, "A22_a0.5"))
            r_b_cos = per_pair_per_cond_cos.get((uid, asin, cond_b))
            r_a_maha = per_pair_per_cond_maha.get((uid, asin, "A22_a0.5"))
            r_b_maha = per_pair_per_cond_maha.get((uid, asin, cond_b))
            if r_a_cos is not None and r_b_cos is not None:
                diffs_cos.append(int(r_b_cos) - int(r_a_cos))
            if r_a_maha is not None and r_b_maha is not None:
                diffs_maha.append(int(r_b_maha) - int(r_a_maha))
        mean_d_cos, ci_d_cos = bootstrap_ci(diffs_cos)
        mean_d_maha, ci_d_maha = bootstrap_ci(diffs_maha)
        diffs[f"A22_a0.5_vs_{cond_b}"] = {
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
            f"  A22_a0.5 vs {cond_b}: "
            f"cosine_rank_diff={mean_d_cos:+.1f} CI {ci_d_cos[0]:+.1f}/{ci_d_cos[1]:+.1f} "
            f"({'excl 0' if ci_d_cos[0] > 0 else 'incl 0'}); "
            f"maha_rank_diff={mean_d_maha:+.1f} CI {ci_d_maha[0]:+.1f}/{ci_d_maha[1]:+.1f} "
            f"({'excl 0' if ci_d_maha[0] > 0 else 'incl 0'})"
        )

    # === [11] Save ===
    out = {
        "phase": "14.C",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "snapshot": snapshot,
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "k_per_cond": 8,
        "cov_pooled_method": "Ledoit-Wolf shrinkage",
        "cov_shrinkage": shrinkage,
        "eigval_min": float(eigvals.min()),
        "eigval_max": float(eigvals.max()),
        "per_cond_eval": per_cond_eval,
        "paired_diffs": diffs,
        "comparison": {
            "Phase14_A_sampled_A14_a1.0_cosine_top100_30pairs": "56.67%",
            "Phase14_A_sampled_A14_a1.0_cosine_rank1_30pairs": "0%",
        },
        "note": (
            "Phase 14.C uses Phase 14.B's 30 pairs (different from Phase 14 hybrid's 30 pairs). "
            "Only 9 pairs overlap. Within-experiment fair: A22_a0.5 vs A14_a1.0 (Phase 14 hybrid equivalent) "
            "vs D_off on identical 30 pairs."
        ),
        "decision_logic": {
            "A22_a0.5 > A14_a1.0": "rank diff CI > 0 in cosine (A22 ranks target user higher than A14)",
            "A22_a0.5 > D_off": "rank diff CI > 0 (injection lifts over baseline)",
        },
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
        "phase": "14.C",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "snapshot": snapshot,
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "cov_shrinkage": shrinkage,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()