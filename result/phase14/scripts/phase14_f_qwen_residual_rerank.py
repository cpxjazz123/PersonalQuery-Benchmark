#!/usr/bin/env python3
"""Phase 14.F: Rerank in Qwen mean-pool residual space (matching StyleVector injection space).

Goal: test if rerank metric in Qwen residual space (sentence_hidden - neutral_hidden)
beats Phase 14.C pooled Maha in 768d AnnaWegmann space.

Key design (per user feedback):
  - StyleVector is injected in Qwen hidden space (layer L: 14 / 22)
  - Rerank should also live in Qwen hidden space, not 768d
  - residual = sentence_hidden - neutral_hidden (Qwen mean-pool per layer)

Pipeline:
  1. Load Phase 14.B jsonl (720 records: A22_a0.5 / A14_a1.0 / D_off × 30 pairs × K=8)
  2. Load Phase 14.B user hiddens (198 × 30 × 5 layers × 3584d mean-pool)
  3. Load Phase 13.A neutral hiddens (2976 × 28 × 3584d mean-pool per text)
     - global neutral ref = mean over 2976 at each layer
  4. Per-user residuals at each of 5 layers (30 samples × 3584d)
     - fit Gaussian: residual_mu[u, layer], residual_sigma_diag[u, layer]
     - LW shrinkage to regularize σ²
  5. Pooled Maha (per-layer LW shrunk on per-user μ_residual)
  6. Encode 720 q_styled through Qwen at 5 layers (mean-pool) → (720, 5, 3584)
  7. Per-cand residual_query = cand_embs - global_neutral_ref → (720, 5, 3584)
  8. Per (pair, cond, layer):
      - For each user u in 198: pooled Maha(residual_query, μ_residual_u) + per-user log-lik
      - Rank target = #{u | D(cand_i) < D(target_user)}
      - best-of-K = min rank across 8 cands per layer per metric
  9. Aggregate per (cond, layer, metric): rank-1, top-100, mean rank
 10. Paired bootstrap diff: A22_a0.5 vs A14_a1.0 vs D_off
 11. Verdict vs Phase 14.C pooled Maha baseline (768d)

Compared metrics:
  - pooled_maha_qwen  : pooled Maha in Qwen residual space (primary)
  - pooled_maha_768d  : Phase 14.C baseline (recomputed for fair pair-bootstrap)
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
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"

OUT_EVAL = OUT_DIR / "phase14_f_qwen_residual_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_f_qwen_residual_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_f_qwen_residual_rerank_meta.json"
OUT_CAND_RESID = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
LAYERS_5 = [8, 14, 18, 22, 26]  # Phase 14.B cache layers (consistent with StyleVector injection layers)
N_USERS = 198                    # Phase 14.B cache size
N_NEUTRAL = 2976                 # Phase 13.A cache size
HIDDEN = 3584                    # Qwen2-7B hidden dim
K_PER_PAIR = 8
SEED = 42
N_BOOTSTRAP = 2000
LW_SHRINKAGE_PER_USER = 0.1      # Per-user σ² shrinkage towards mean variance (regularize)
MIN_VAR = 1e-4                   # Floor on variance (avoid degenerate sigma)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def get_qwen_local_client():
    """Get local Qwen client via project llm_client.py (with_vllm=False: only hidden states)."""
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    return create_qwen_local_client(with_vllm=False)


def encode_candidates_at_layers(
    client, texts: list[str], layers_0idx: list[int], batch_size: int = 32, max_length: int = 256,
):
    """Encode texts via Qwen at specified layers (0-indexed), mean-pool over tokens.

    Uses llm_client.QwenLocalClient.get_hidden_states (canonical path).
    Returns: [N, len(layers), H]  float32
    """
    hdict = client.get_hidden_states(texts, layers=layers_0idx, batch_size=batch_size, max_length=max_length)
    # hdict[layer] is [N, H]; stack to [N, len(layers), H]
    out = np.stack([hdict[l] for l in layers_0idx], axis=1).astype(np.float32)
    return out


def fit_user_gaussians(user_residuals: np.ndarray, lw_shrink: float = LW_SHRINKAGE_PER_USER, min_var: float = MIN_VAR):
    """Per-user per-layer Gaussian fit with LW shrinkage.

    Args:
      user_residuals: [n_users, n_samples, n_layers, H]  (float32)
        axis 0 = users, axis 1 = sentences, axis 2 = layers, axis 3 = hidden
      lw_shrink: shrinkage towards per-layer pooled mean variance
      min_var: floor on variance

    Returns:
      mu: [n_users, n_layers, H]      (mean over sentences)
      sigma_diag: [n_users, n_layers, H]  (variance over sentences, with shrinkage)
    """
    # user_residuals shape: (n_users=198, n_samples=30, n_layers=5, H=3584)
    # Mean over sentences (axis=1) → (n_users, n_layers, H)
    mu = user_residuals.mean(axis=1)  # [n_users, n_layers, H]
    var = user_residuals.var(axis=1)  # [n_users, n_layers, H] variance over sentences

    if lw_shrink > 0:
        # Pooled variance per layer (mean over users)
        pooled_var = var.mean(axis=0, keepdims=True)  # [1, n_layers, H]
        var_shrunk = (1 - lw_shrink) * var + lw_shrink * pooled_var
    else:
        var_shrunk = var
    var_shrunk = np.maximum(var_shrunk, min_var)
    return mu.astype(np.float32), var_shrunk.astype(np.float32)


def pooled_maha_distance(
    cand_residuals: np.ndarray,    # [N_cand, n_layers, H]
    user_mu: np.ndarray,           # [n_users, n_layers, H]
    user_var: np.ndarray,          # [n_users, n_layers, H]  (variance, already shrunk)
    pooled_var: np.ndarray,        # [n_layers, H]           (pooled variance per layer)
    pooled_shrink: float = 0.1,    # shrinkage of pooled cov towards identity
):
    """Pooled Mahalanobis distance in Qwen residual space, per-layer.

    For each (cand, layer, user), compute:
      D_pooled = (cand - mu_u)^T @ inv_Sigma_pooled @ (cand - mu_u)
    then sum over layers (or evaluate per-layer — return per-layer for layer-wise eval).

    Returns:
      distances: [N_cand, n_users, n_layers]  per-layer distance
    """
    n_layers = cand_residuals.shape[1]
    n_users = user_mu.shape[0]

    # Pooled covariance = LW shrunk on user mean (approx diagonal)
    # For tractability use pooled_var (diagonal)
    inv_var_pooled = 1.0 / (pooled_var + 1e-6)  # [n_layers, H]

    distances = np.zeros((cand_residuals.shape[0], n_users, n_layers), dtype=np.float32)
    for li in range(n_layers):
        # delta: [N_cand, n_users, H]
        delta = cand_residuals[:, li, :][:, None, :] - user_mu[:, li, :][None, :, :]
        # weighted L2: [N_cand, n_users, H] @ inv_var [H]
        weighted = delta * inv_var_pooled[li][None, None, :]
        # per-layer distance: sum over H
        distances[:, :, li] = (delta * weighted).sum(axis=-1)  # [N_cand, n_users]
    return distances


def per_user_loglik(
    cand_residuals: np.ndarray,    # [N_cand, n_layers, H]
    user_mu: np.ndarray,           # [n_users, n_layers, H]
    user_var: np.ndarray,          # [n_users, n_layers, H]
):
    """Per-user diagonal Gaussian log-likelihood.

    log_lik(cand | u) = -0.5 * Σ_layer Σ_d [ (cand - mu)² / var + log var + log(2π) ]

    Returns:
      log_lik: [N_cand, n_users]
    """
    n_layers = cand_residuals.shape[1]
    log_lik = np.zeros((cand_residuals.shape[0], user_mu.shape[0]), dtype=np.float32)
    for li in range(n_layers):
        delta2 = (cand_residuals[:, li, :][:, None, :] - user_mu[:, li, :][None, :, :]) ** 2
        term = delta2 / user_var[:, li, :][None, :, :] + np.log(user_var[:, li, :][None, :, :])
        log_lik -= 0.5 * term.sum(axis=-1)
    return log_lik


def main() -> None:
    log("=" * 70)
    log("Phase 14.F: Qwen residual-space rerank (StyleVector injection space)")
    log("=" * 70)

    # === [1] Load candidates ===
    log(f"[1] Loading {IN_JSONL.name} + filtering to {CONDITIONS} ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    log(f"  filtered candidates: {len(candidates)} (expected 30 × 3 × 8 = 720)")
    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
    log(f"  pairs: {len(pairs)}")

    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load Phase 14.B user hiddens ===
    log(f"[2] Loading {IN_USER_HIDDENS.name} ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)  # [198, 30, 5, 3584]
    layers_cache = list(npz["layers"])
    assert layers_cache == LAYERS_5, f"layer mismatch: {layers_cache} vs {LAYERS_5}"
    log(f"  users: {len(user_ids_cache)}, hiddens shape: {user_hiddens.shape}, layers: {layers_cache}")

    # Map phase14 user_id -> cache index
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}
    pair_user_idx = []
    skipped = 0
    for (uid, asin) in pairs:
        if uid in uid_to_useridx:
            pair_user_idx.append(uid_to_useridx[uid])
        else:
            pair_user_idx.append(None)
            skipped += 1
    log(f"  pair user_idx: {sum(1 for x in pair_user_idx if x is not None)}/{len(pairs)} (skipped {skipped})")

    # === [3] Load Phase 13.A neutral hiddens ===
    log(f"[3] Loading {IN_NEUTRAL_HIDDENS.name} ...")
    npz = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz["vecs"].astype(np.float32)  # [2976, 28, 3584]
    # Layer convention: phase13_a used range(28) -> [0..27] direct hidden_states tuple indices.
    # phase14_b cache uses layers [8, 14, 18, 22, 26] as direct hidden_states tuple indices.
    # So neutral access matches phase14_b indexing: neutral_vecs[:, layer, :] for layer in [8,14,18,22,26].
    layer_idx_in_neutral = LAYERS_5  # no offset, direct indexing
    neutral_layer = neutral_vecs[:, layer_idx_in_neutral, :]  # [2976, 5, 3584]
    global_neutral = neutral_layer.mean(axis=0)  # [5, 3584]
    log(f"  neutral_layer shape: {neutral_layer.shape}, global_neutral shape: {global_neutral.shape}")
    log(f"  global_neutral mean norm per layer: {np.linalg.norm(global_neutral, axis=-1)}")

    # === [4] Compute per-user residuals + fit Gaussian ===
    log("[4] Computing per-user residuals (sentence_hidden - global_neutral) ...")
    # user_hiddens: [198, 30, 5, 3584], global_neutral: [5, 3584]
    user_residuals = user_hiddens - global_neutral[None, None, :, :]  # [198, 30, 5, 3584]
    log(f"  user_residuals shape: {user_residuals.shape}, mean norm per layer: {np.linalg.norm(user_residuals.mean(axis=(0,1)), axis=-1)}")

    log(f"  Fitting per-user Gaussian (LW shrinkage={LW_SHRINKAGE_PER_USER}, min_var={MIN_VAR}) ...")
    user_mu, user_var = fit_user_gaussians(user_residuals, LW_SHRINKAGE_PER_USER, MIN_VAR)  # [198, 5, 3584]
    pooled_var = user_var.mean(axis=0)  # [5, 3584]
    log(f"  user_mu shape: {user_mu.shape}, user_var shape: {user_var.shape}")
    log(f"  pooled_var mean per layer: {pooled_var.mean(axis=-1)}")

    # === [5] Encode q_styled through Qwen at 5 layers (mean-pool) ===
    cand_path = OUT_CAND_RESID
    if cand_path.exists():
        log(f"[5] Loading cached residuals from {cand_path} ...")
        cand_residuals = np.load(cand_path)  # [720, 5, 3584]
        if cand_residuals.shape[0] != len(candidates):
            raise ValueError(f"cached cand_residuals shape {cand_residuals.shape} mismatch with candidates {len(candidates)}")
    else:
        log("[5] Loading Qwen local client (with_vllm=False for hidden states only) ...")
        client = get_qwen_local_client()
        cand_texts = [c.get("q_styled") or "" for c in candidates]
        log(f"  Encoding {len(cand_texts)} candidates at 5 layers (direct hidden_states idx: {LAYERS_5}) ...")
        # Convention: get_hidden_states(layers=L) returns hidden_states[L] directly (no offset).
        # This matches phase14_b cache indexing (layers = [8, 14, 18, 22, 26]).
        cand_embs = encode_candidates_at_layers(
            client, cand_texts, layers_0idx=LAYERS_5, batch_size=32, max_length=256,
        )
        log(f"  cand_embs shape: {cand_embs.shape}")
        # Subtract global neutral
        cand_residuals = cand_embs - global_neutral[None, :, :]  # [720, 5, 3584]
        np.save(cand_path, cand_residuals)
        log(f"  saved cand_residuals → {cand_path}")
    log(f"  cand_residuals shape: {cand_residuals.shape}, mean norm per layer: {np.linalg.norm(cand_residuals.mean(axis=0), axis=-1)}")

    # === [6] Per-(pair, cond, layer) rerank ===
    log("[6] Per-(pair, cond, layer) rerank (pooled Maha + per-user log-lik) ...")
    # Store: per_pair_per_cond_layer_maha_rank[(uid, asin, cond, layer)] = best-of-K rank
    # Store: per_pair_per_cond_layer_loglik_rank[(uid, asin, cond, layer)] = best-of-K rank
    per_pair_per_cond_layer_maha: dict[tuple[str, str, str, int], int] = {}
    per_pair_per_cond_layer_loglik: dict[tuple[str, str, int], int] = {}  # Note: 4-tuple

    n_layers = len(LAYERS_5)
    n_skipped_pair = 0

    for (uid, asin), conds_local in cand_by_pair_cond.items():
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            n_skipped_pair += 1
            continue
        for cond, cand_indices in conds_local.items():
            local_resid = cand_residuals[cand_indices]  # [K, 5, 3584]
            # Pooled Maha: [K, n_users, 5] per-layer distance
            maha_d = pooled_maha_distance(local_resid, user_mu, user_var, pooled_var, pooled_shrink=0.1)  # [K, n_users, 5]
            # Per-user log-lik: [K, n_users] summed across layers
            loglik = per_user_loglik(local_resid, user_mu, user_var)  # [K, n_users]

            for li in range(n_layers):
                layer_d = maha_d[:, :, li]  # [K, n_users]
                target_d = layer_d[:, target_idx: target_idx + 1]  # [K, 1]
                rank_per_cand = (layer_d < target_d).sum(axis=1)  # [K]
                best_rank = int(np.min(rank_per_cand))
                per_pair_per_cond_layer_maha[(uid, asin, cond, LAYERS_5[li])] = best_rank

            # Log-lik (summed across layers, single best-of-K)
            target_loglik = loglik[:, target_idx: target_idx + 1]  # [K, 1]
            rank_per_cand_ll = (loglik < target_loglik).sum(axis=1)  # [K]
            best_rank_ll = int(np.min(rank_per_cand_ll))
            per_pair_per_cond_layer_loglik[(uid, asin, cond, -1)] = best_rank_ll  # -1 = aggregated

    log(f"  done; skipped pairs: {n_skipped_pair}")

    # === [7] Aggregate per (cond, layer) per metric ===
    log("[7] Aggregating per (cond, layer, metric) ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)
    log(f"  valid pairs: {len(pair_keys)}")

    per_cond_layer_eval: dict[tuple[str, str, int], dict] = {}  # (cond, metric, layer)

    for cond in CONDITIONS:
        for li, layer in enumerate(LAYERS_5):
            ranks = []
            for (uid, asin) in pair_keys:
                r = per_pair_per_cond_layer_maha.get((uid, asin, cond, layer))
                if r is not None:
                    ranks.append(r)
            arr = np.array(ranks)
            mean, ci = bootstrap_ci(ranks)
            per_cond_layer_eval[(cond, "maha", layer)] = {
                "n": len(ranks),
                "rank1_coverage": float((arr == 0).sum()) / max(1, len(arr)),
                "top10_coverage": float((arr < 10).sum()) / max(1, len(arr)),
                "top100_coverage": float((arr < 100).sum()) / max(1, len(arr)),
                "mean_best_rank": mean,
                "mean_best_rank_ci95": list(ci),
            }
            log(
                f"  maha {cond} layer={layer}: "
                f"rank1={(arr == 0).sum()}/{len(arr)}, "
                f"top100={(arr < 100).sum()}/{len(arr)}, "
                f"mean={mean:.1f} CI [{ci[0]:.1f}, {ci[1]:.1f}]"
            )
        # Log-lik (aggregated)
        ranks_ll = []
        for (uid, asin) in pair_keys:
            r = per_pair_per_cond_layer_loglik.get((uid, asin, cond, -1))
            if r is not None:
                ranks_ll.append(r)
        arr_ll = np.array(ranks_ll)
        mean_ll, ci_ll = bootstrap_ci(ranks_ll)
        per_cond_layer_eval[(cond, "loglik", -1)] = {
            "n": len(ranks_ll),
            "rank1_coverage": float((arr_ll == 0).sum()) / max(1, len(arr_ll)),
            "top10_coverage": float((arr_ll < 10).sum()) / max(1, len(arr_ll)),
            "top100_coverage": float((arr_ll < 100).sum()) / max(1, len(arr_ll)),
            "mean_best_rank": mean_ll,
            "mean_best_rank_ci95": list(ci_ll),
        }
        log(
            f"  loglik {cond} (sum layers): "
            f"rank1={(arr_ll == 0).sum()}/{len(arr_ll)}, "
            f"top100={(arr_ll < 100).sum()}/{len(arr_ll)}, "
            f"mean={mean_ll:.1f} CI [{ci_ll[0]:.1f}, {ci_ll[1]:.1f}]"
        )

    # === [8] Paired bootstrap diff (A22 vs each baseline, per layer) ===
    log("[8] Paired bootstrap diff (A22_a0.5 vs A14_a1.0 vs D_off) per layer (maha) ...")
    diffs: dict[str, dict] = {}
    for layer in LAYERS_5:
        for cond_b in CONDITIONS:
            if cond_b == "A22_a0.5":
                continue
            diffs_layer = []
            for (uid, asin) in pair_keys:
                r_a = per_pair_per_cond_layer_maha.get((uid, asin, "A22_a0.5", layer))
                r_b = per_pair_per_cond_layer_maha.get((uid, asin, cond_b, layer))
                if r_a is not None and r_b is not None:
                    diffs_layer.append(int(r_b) - int(r_a))  # positive = A22 ranks higher (smaller rank)
            mean_d, ci_d = bootstrap_ci(diffs_layer)
            diffs[f"A22_a0.5_vs_{cond_b}_layer{layer}"] = {
                "metric": "maha",
                "mean_diff": mean_d,
                "ci95": list(ci_d),
                "ci_excludes_0": ci_d[0] > 0,
                "n": len(diffs_layer),
            }
            log(
                f"  A22 vs {cond_b} layer={layer}: "
                f"diff={mean_d:+.1f} CI [{ci_d[0]:+.1f}, {ci_d[1]:+.1f}] "
                f"({'excl 0' if ci_d[0] > 0 else 'incl 0'})"
            )

    # === [9] Save ===
    out = {
        "phase": "14.F",
        "rerank_space": "Qwen mean-pool residual (sentence_hidden - global_neutral_hidden) per layer",
        "layers": LAYERS_5,
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys),
        "n_users": len(user_ids_cache),
        "n_neutral_refs": N_NEUTRAL,
        "lw_shrinkage_user": LW_SHRINKAGE_PER_USER,
        "min_var": MIN_VAR,
        "per_cond_layer_eval": {
            f"{cond}__{metric}__layer{layer}": per_cond_layer_eval[(cond, metric, layer)]
            for (cond, metric, layer) in per_cond_layer_eval.keys()
        },
        "paired_diffs": diffs,
        "comparison_baseline_phase14_c_768d_maha": {
            "A22_a0.5_maha_768d_top100": "43.3%",
            "A14_a1.0_maha_768d_top100": "53.3%",
            "D_off_maha_768d_top100": "20% (estimated)",
        },
        "decision_logic": {
            "A22_a0.5 > A14_a1.0": "rank diff CI > 0 (Qwen residual rerank > 768d AnnaWegmann rerank)",
            "A22_a0.5 > D_off": "rank diff CI > 0 (residual rerank captures injection signal)",
            "best_layer_per_cond": "report layer with best mean rank per cond",
        },
        "interpretation_notes": (
            "Phase 14.F rerank lives in Qwen residual space matching StyleVector injection space. "
            "Layer 22 should favor A22_a0.5 (direct injection), Layer 14 should favor A14_a1.0. "
            "If A22 wins on layer 22 with CI excludes 0, StyleVector injection space is genuinely effective."
        ),
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in pair_keys:
            row = {"user_id": uid, "asin": asin}
            for cond in CONDITIONS:
                for li, layer in enumerate(LAYERS_5):
                    row[f"{cond}_maha_layer{layer}"] = per_pair_per_cond_layer_maha.get((uid, asin, cond, layer))
                row[f"{cond}_loglik_sum"] = per_pair_per_cond_layer_loglik.get((uid, asin, cond, -1))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.F",
        "in_jsonl": str(IN_JSONL),
        "in_user_hiddens": str(IN_USER_HIDDENS),
        "in_neutral_hiddens": str(IN_NEUTRAL_HIDDENS),
        "out_eval": str(OUT_EVAL),
        "out_per_pair": str(OUT_PER_PAIR),
        "out_cand_residuals": str(OUT_CAND_RESID),
        "n_pairs": len(pair_keys),
        "n_users_cache": len(user_ids_cache),
        "n_neutral_refs": N_NEUTRAL,
        "layers": LAYERS_5,
        "lw_shrinkage_user": LW_SHRINKAGE_PER_USER,
        "min_var": MIN_VAR,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.F COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()