#!/usr/bin/env python3
"""Phase 13.F: Style-space (AnnaWegmann 768d) re-evaluation of Phase 13.D candidates.

User feedback: 318d syntactic Mahalanobis is the WRONG evaluation space for ACL
StyleVector. ACL StyleVector measures style via AnnaWegmann-style embeddings,
NOT syntactic features. Re-evaluate 13.D outputs in 768d style space.

Pipeline:
  1. Load 960 candidates from phase13_d_v2_e2e_queries_lasttoken.jsonl
  2. Encode each candidate's q_styled (or q_final_post) via AnnaWegmann → 768d
  3. For each candidate compute:
       cos_to_target   = cos(cand_emb, mu_target_user_768d)
       cos_to_off_mean = mean cos(cand_emb, mu_off_users_768d) over 20 random other users
       margin          = cos_to_target - cos_to_off_mean
  4. Per-pair-per-cond: best-of-K margin, mean-of-K margin
  5. Per-cond aggregate (30 pairs):
       A_sampled mean margin vs D_off paired bootstrap
  6. Verdict:
       GO         : A_sampled margin - D_off margin > 0 with CI excludes 0
       PARTIAL-GO : A_sampled > B_mean (sampling helps vs fixed mean)
       NO-GO      : no significant lift

Reuses:
  - phase10_user_embs_768d.npz (876 user_mu_768)
  - AnnaWegmann/Style-Embedding
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

IN_CANDIDATES = OUT_DIR / "phase13_d_v2_e2e_queries_lasttoken.jsonl"
IN_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"
OUT_EMB_CANDS = OUT_DIR / "phase13_f_v2_cand_embs_768d_lasttoken.npy"
OUT_EVAL = OUT_DIR / "phase13_f_v2_style_eval_lasttoken.json"
OUT_PER_PAIR = OUT_DIR / "phase13_f_v2_per_pair_lasttoken.jsonl"
OUT_META = OUT_DIR / "phase13_f_v2_meta_lasttoken.json"

ANNA_SNAPSHOT_DIR = "/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots"

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]
SEED = 42
N_OFF_USERS = 20
ENCODER_BATCH = 128
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-row L2 normalize."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts via sentence_transformers, no normalize (we handle it)."""
    return model.encode(
        texts, convert_to_numpy=True, batch_size=batch_size,
        show_progress_bar=False, normalize_embeddings=False,
    )


def main():
    log("=" * 70)
    log("Phase 13.F: Style-space (AnnaWegmann 768d) re-evaluation")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)

    # === [1] Load 13.D candidates ===
    log("[1] Loading 13.D candidates ...")
    candidates: list[dict] = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  total candidates: {len(candidates)}")
    log(f"  conditions: {sorted({c['condition'] for c in candidates})}")

    # === [2] Load user mean 768d style vectors ===
    log("[2] Loading 768d user style vectors ...")
    npz = np.load(IN_USER_EMBS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    user_embs_768 = npz["embs"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}
    log(f"  users: {len(user_ids_arr)}, emb_dim={user_embs_768.shape[1]}")
    user_embs_768_norm = l2_normalize(user_embs_768)

    # === [3] Load AnnaWegmann encoder ===
    log(f"[3] Loading AnnaWegmann/Style-Embedding ...")
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
    log("[4] Encoding 960 candidate queries ...")
    cand_texts = [c.get("q_styled") or "" for c in candidates]
    t0 = time.time()
    cand_embs = encode_texts(model, cand_texts, ENCODER_BATCH)
    log(f"  encoded {len(cand_texts)} in {time.time()-t0:.1f}s, shape={cand_embs.shape}")
    np.save(OUT_EMB_CANDS, cand_embs)
    log(f"  saved → {OUT_EMB_CANDS}")
    cand_embs_norm = l2_normalize(cand_embs.astype(np.float32))

    # === [5] Per-cand cosines ===
    log("[5] Computing per-cand cos_to_target and cos_to_off_mean ...")
    rng = np.random.default_rng(SEED)
    n_cands = len(candidates)
    cos_to_target_arr = np.zeros(n_cands, dtype=np.float64)
    cos_to_off_mean_arr = np.zeros(n_cands, dtype=np.float64)

    # pre-sample one fixed off-user-set per pair (so same off-users across conds)
    pair_to_off_idx = {}
    pairs_seen = []
    for ci, c in enumerate(candidates):
        uid = c["user_id"]
        if uid not in pair_to_off_idx:
            pairs_seen.append(uid)
            target_uidx = uid_to_useridx.get(uid)
            if target_uidx is None:
                raise ValueError(f"User {uid} not in user_embs cache")
            mask = np.ones(len(user_ids_arr), dtype=bool)
            mask[target_uidx] = False
            off_idx = rng.choice(np.arange(len(user_ids_arr))[mask], size=N_OFF_USERS, replace=False)
            pair_to_off_idx[uid] = (target_uidx, off_idx)
        target_uidx, off_idx = pair_to_off_idx[uid]
        v = cand_embs_norm[ci]
        cos_to_target_arr[ci] = float(np.dot(v, user_embs_768_norm[target_uidx]))
        cos_to_off_mean_arr[ci] = float(np.mean(np.dot(v, user_embs_768_norm[off_idx].T)))

    margin_arr = cos_to_target_arr - cos_to_off_mean_arr
    log(f"  margin stats: mean={margin_arr.mean():.4f}, std={margin_arr.std():.4f}, "
        f"min={margin_arr.min():.4f}, max={margin_arr.max():.4f}")

    # === [6] Per-(pair, cond) best-of-K and mean-of-K margin ===
    log("[6] Aggregating per-(pair, cond) ...")
    per_pair_per_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
    best_per_pair_per_cond: dict[tuple[str, str], float] = {}

    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["condition"])
        per_pair_per_cond[key].append(float(margin_arr[ci]))

    for key, vals in per_pair_per_cond.items():
        best_per_pair_per_cond[key] = float(np.max(vals))

    # build per-cond paired array (30 pairs each, same product across conds)
    cond_pairs = {cond: [] for cond in CONDITIONS}
    cond_pairs_best = {cond: [] for cond in CONDITIONS}
    pair_uids_in_order = []
    seen = set()
    for c in candidates:
        uid = c["user_id"]
        if uid not in seen:
            seen.add(uid)
            pair_uids_in_order.append(uid)

    for cond in CONDITIONS:
        for uid in pair_uids_in_order:
            vals = per_pair_per_cond.get((uid, cond), [])
            if vals:
                cond_pairs[cond].append(float(np.mean(vals)))
                cond_pairs_best[cond].append(best_per_pair_per_cond[(uid, cond)])

    log(f"  pairs: {len(pair_uids_in_order)}, per-cond aggregate sizes: "
        f"{ {c: len(v) for c, v in cond_pairs.items()} }")

    # === [7] Bootstrap CI per cond ===
    log("[7] Bootstrap CI per cond (n=2000) ...")

    def bootstrap_ci(values: list[float], n: int = 2000, seed: int = SEED) -> tuple[float, tuple[float, float]]:
        if not values:
            return 0.0, (0.0, 0.0)
        arr = np.array(values)
        rng = np.random.default_rng(seed)
        n_obs = len(arr)
        boot_means = []
        for _ in range(n):
            idx = rng.choice(n_obs, size=n_obs, replace=True)
            boot_means.append(float(arr[idx].mean()))
        bm = np.array(boot_means)
        return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))

    per_cond_eval: dict[str, dict] = {}
    for cond in CONDITIONS:
        m_mean, m_ci = bootstrap_ci(cond_pairs[cond])
        b_mean, b_ci = bootstrap_ci(cond_pairs_best[cond])
        per_cond_eval[cond] = {
            "mean_of_K_margin_mean": m_mean,
            "mean_of_K_margin_ci95": list(m_ci),
            "best_of_K_margin_mean": b_mean,
            "best_of_K_margin_ci95": list(b_ci),
            "n_pairs": len(cond_pairs[cond]),
        }
        log(f"  {cond}: mean_margin={m_mean:.4f} [{m_ci[0]:.4f},{m_ci[1]:.4f}], "
            f"best_margin={b_mean:.4f} [{b_ci[0]:.4f},{b_ci[1]:.4f}]")

    # === [8] Paired bootstrap: A_sampled vs each baseline ===
    log("[8] Paired bootstrap diffs ...")
    diffs_results = {}
    for cond_b in CONDITIONS:
        if cond_b == "A_sampled":
            continue
        diffs_arr_mean = np.array(cond_pairs["A_sampled"]) - np.array(cond_pairs[cond_b])
        diffs_arr_best = np.array(cond_pairs_best["A_sampled"]) - np.array(cond_pairs_best[cond_b])
        d_mean, d_ci_mean = bootstrap_ci(diffs_arr_mean.tolist())
        d_best, d_ci_best = bootstrap_ci(diffs_arr_best.tolist())
        diffs_results[f"A_sampled_vs_{cond_b}"] = {
            "mean_margin_diff": d_mean,
            "mean_margin_diff_ci95": list(d_ci_mean),
            "best_margin_diff": d_best,
            "best_margin_diff_ci95": list(d_ci_best),
            "ci_excludes_0_mean": d_ci_mean[0] > 0,
            "ci_excludes_0_best": d_ci_best[0] > 0,
        }
        log(f"  A_sampled vs {cond_b}: diff_mean={d_mean:.4f} [{d_ci_mean[0]:.4f},{d_ci_mean[1]:.4f}], "
            f"diff_best={d_best:.4f} [{d_ci_best[0]:.4f},{d_ci_best[1]:.4f}]")

    # === [9] Per-cand breakdown by cond (raw cos target / off stats) ===
    per_cond_raw: dict[str, dict] = {}
    for cond in CONDITIONS:
        idxs = [i for i, c in enumerate(candidates) if c["condition"] == cond]
        if not idxs:
            continue
        per_cond_raw[cond] = {
            "n_cands": len(idxs),
            "mean_cos_target": float(np.mean(cos_to_target_arr[idxs])),
            "mean_cos_off": float(np.mean(cos_to_off_mean_arr[idxs])),
            "mean_margin": float(np.mean(margin_arr[idxs])),
        }

    # === [10] Verdict ===
    a_eval = per_cond_eval["A_sampled"]
    d_eval = per_cond_eval["D_off"]
    a_vs_d = diffs_results["A_sampled_vs_D_off"]
    a_vs_b = diffs_results.get("A_sampled_vs_B_mean", {})

    if a_vs_d.get("ci_excludes_0_mean") or a_vs_d.get("ci_excludes_0_best"):
        verdict = "GO"
        verdict_reason = (
            f"A_sampled margin lifts over D_off (paired CI excludes 0): "
            f"diff_mean={a_vs_d['mean_margin_diff']:.4f}, diff_best={a_vs_d['best_margin_diff']:.4f}"
        )
    elif a_vs_b.get("ci_excludes_0_mean") or a_vs_b.get("ci_excludes_0_best"):
        verdict = "PARTIAL-GO"
        verdict_reason = (
            f"A_sampled margin lifts over B_mean (sampling helps vs fixed mean) "
            f"but NOT over D_off: diff_mean_vs_B={a_vs_b.get('mean_margin_diff', 0):.4f}"
        )
    else:
        verdict = "NO-GO"
        verdict_reason = (
            f"No margin lift in style space either: "
            f"A_sampled vs D_off diff_mean={a_vs_d.get('mean_margin_diff', 0):.4f} "
            f"CI [{a_vs_d.get('mean_margin_diff_ci95', [0, 0])[0]:.4f}, "
            f"{a_vs_d.get('mean_margin_diff_ci95', [0, 0])[1]:.4f}]"
        )
    log(f"\n  FINAL VERDICT: {verdict}")
    log(f"  reason: {verdict_reason}")

    # === [11] Save ===
    out = {
        "phase": "13.F",
        "encoder": "AnnaWegmann/Style-Embedding",
        "snapshot": snapshot,
        "feature_space": "768d AnnaWegmann style (cosine)",
        "n_candidates": len(candidates),
        "n_conditions": len(CONDITIONS),
        "n_pairs": len(pair_uids_in_order),
        "n_off_users": N_OFF_USERS,
        "per_cond_raw_cos": per_cond_raw,
        "per_cond_eval": per_cond_eval,
        "paired_diffs": diffs_results,
        "decision_logic": {
            "GO": "A_sampled vs D_off paired CI > 0 (mean or best)",
            "PARTIAL-GO": "A_sampled vs B_mean CI > 0 but vs D_off does not",
            "NO-GO": "no significant lift",
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for uid in pair_uids_in_order:
            for cond in CONDITIONS:
                vals = per_pair_per_cond.get((uid, cond), [])
                if not vals:
                    continue
                row = {
                    "user_id": uid,
                    "condition": cond,
                    "mean_margin": float(np.mean(vals)),
                    "best_margin": float(np.max(vals)),
                    "n_cands": len(vals),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "13.F style-space re-eval",
        "encoder": "AnnaWegmann/Style-Embedding",
        "snapshot": snapshot,
        "n_candidates": len(candidates),
        "n_pairs": len(pair_uids_in_order),
        "n_off_users": N_OFF_USERS,
        "verdict": verdict,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 13.F COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()