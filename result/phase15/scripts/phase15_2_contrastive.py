#!/usr/bin/env python3
"""Phase 15.2: Supervised contrastive style space.

Train a linear projection from Qwen layer 26 residuals to a learned 128-d style space.
Loss: same-user sentences = positive, different users = negative.
Hard negatives: similar PCA-projected users (top-K most similar other users in train pool).

Then rerank using the projected space + Maha distance.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"
IN_GEN = OUT_DIR / "phase14_q10l_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_q10l_cand_residuals_qwen.npy"

OUT_EMB = OUT_DIR / "phase15_2_user_emb_contrastive.npy"
OUT_PROJ = OUT_DIR / "phase15_2_projection.npy"
OUT_EVAL = OUT_DIR / "phase15_2_lopo_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase15_2_lopo_per_pair.jsonl"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
HIDDEN = 3584
PROJ_DIM = 128
TEMPERATURE = 0.07
MARGIN = 1.0
EPOCHS = 30
BATCH_USERS = 32
LR = 0.005
N_TEST_USERS = 500
SPLITS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=42):
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


def aggregate(ranks):
    if not ranks:
        return None
    arr = np.array(ranks)
    mean, ci = bootstrap_ci(ranks)
    return {
        "n": len(ranks),
        "rank1": int((arr == 0).sum()),
        "rank1_pct": float((arr == 0).sum()) / max(1, len(arr)),
        "rank1_ci95": list(bootstrap_ci([1.0 if r == 0 else 0.0 for r in ranks])[1]),
        "top10": int((arr < 10).sum()),
        "top10_pct": float((arr < 10).sum()) / max(1, len(arr)),
        "top100": int((arr < 100).sum()),
        "top100_pct": float((arr < 100).sum()) / max(1, len(arr)),
        "mrr": float((1.0 / (arr + 1)).mean()),
        "mrr_ci95": list(bootstrap_ci([1.0 / (r + 1) for r in ranks])[1]),
        "mean_rank": mean,
        "mean_rank_ci95": list(ci),
    }


def train_contrastive_projection(user_residuals, train_uids, epochs=EPOCHS, batch_users=BATCH_USERS, lr=LR,
                                  proj_dim=PROJ_DIM, hidden=HIDDEN, temperature=TEMPERATURE, margin=MARGIN, seed=42):
    """Train a linear projection W: hidden → proj_dim.

    Loss: for each (user, sent) pair, ensure proj(sent) is close to other sents of same user,
    and far from sents of different users.

    Uses supervised contrastive style (NT-Xent-like).
    """
    rng = np.random.default_rng(seed)
    n_users = len(train_uids)
    n_sents = user_residuals.shape[1]
    # Flatten: (n_users * n_sents, hidden)
    flat = user_residuals[train_uids].reshape(-1, hidden).astype(np.float32)
    # L2 normalize inputs for stable training
    flat_norm = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-9)

    # Init projection: random normal, scaled
    W = rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, proj_dim)).astype(np.float32) * 0.1
    b = np.zeros(proj_dim, dtype=np.float32)

    losses = []
    n_total = n_users * n_sents
    # Build label map
    sent_user_ids = np.repeat(np.arange(n_users), n_sents)  # sent_user_ids[i] = user_idx for sent i

    for epoch in range(epochs):
        # Shuffle user order
        user_perm = rng.permutation(n_users)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_users, batch_users):
            batch_users_idx = user_perm[batch_start: batch_start + batch_users]
            batch_sents_idx = np.concatenate([np.arange(u * n_sents, (u + 1) * n_sents) for u in batch_users_idx])
            batch_user_labels = sent_user_ids[batch_sents_idx]

            # Forward
            Z = flat_norm[batch_sents_idx] @ W + b  # (B*n_sents, proj_dim)
            Z_norm = Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-9)

            # Similarity matrix
            sim = Z_norm @ Z_norm.T  # (B*n_sents, B*n_sents)

            # Supervised contrastive loss:
            # For each anchor i, positive are j where batch_user_labels[j] == batch_user_labels[i] and j != i
            # Negatives are j where batch_user_labels[j] != batch_user_labels[i]
            B = sim.shape[0]
            labels_eq = (batch_user_labels[:, None] == batch_user_labels[None, :]).astype(np.float32)
            labels_eq -= np.eye(B, dtype=np.float32)  # exclude self
            n_pos = labels_eq.sum(axis=1) + 1e-9

            # NT-Xent-like: numerator = exp(sim / T) * mask_pos, denom = sum exp(sim / T)
            sim_scaled = sim / temperature
            exp_sim = np.exp(sim_scaled - sim_scaled.max(axis=1, keepdims=True))
            pos_term = (exp_sim * labels_eq).sum(axis=1) / n_pos
            neg_term = exp_sim.sum(axis=1) - (exp_sim * labels_eq).sum(axis=1) - exp_sim.diagonal()
            loss_per_anchor = -np.log(pos_term / (pos_term + neg_term) + 1e-9)
            loss = loss_per_anchor.mean()
            epoch_loss += loss
            n_batches += 1

            # Backward (manual)
            # Approximate gradient: use the loss structure to compute dL/dZ
            # Simpler: compute analytic gradient
            softmax_denom = pos_term + neg_term + 1e-9
            # dL/d(pos_term) = -1/(pos_term * softmax_denom) for each anchor
            # dL/d(neg_term) = +1/(softmax_denom) for each anchor (but weight by how much neg contributes)
            # For simplicity, use weighted softmax over negatives only
            grad_Z = np.zeros_like(Z_norm)

            # Re-compute exp for backward
            for i in range(B):
                if n_pos[i] < 1:
                    continue
                pos_mask = labels_eq[i]
                # Anchor derivative: dL/dZ[i]
                # We use simplified: anchor i should be similar to positives, dissimilar to all
                exp_row = exp_sim[i]
                pos_sum = (exp_row * pos_mask).sum()
                total_sum = exp_row.sum()
                # grad on similarity = (-1/(total_sum)) * pos_mask + (1/(total_sum)) * (1 - pos_mask - self)
                grad_sim = (-pos_mask / (total_sum + 1e-9)) + ((1 - pos_mask - np.eye(1, B, i, dtype=np.float32)[0]) / (total_sum + 1e-9))
                grad_sim /= temperature
                # Convert to Z grad: dL/dZ[i] = sum_j grad_sim[j] * Z_norm[j]  (since sim[i,j] = Z_norm[i] . Z_norm[j])
                grad_Z[i] = (grad_sim[:, None] * Z_norm).sum(axis=0)

            # grad_Z = grad_Z_norm = grad_Z_normalized (since we used Z_norm in sim)
            # dZ_norm/dZ = (I - Z_norm Z_norm.T) / |Z| -- but Z_norm already normalized, so use identity approx
            # Apply learning rate
            grad_W = (flat_norm[batch_sents_idx].T @ grad_Z) / B
            grad_b = grad_Z.mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b

        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            log(f"    epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return W, b, losses


def main() -> None:
    log("=" * 70)
    log("Phase 15.2: Supervised contrastive style space + multi-split LOPO")
    log("=" * 70)

    log("[1] Loading data ...")
    cand_records = []
    with IN_GEN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records.append(json.loads(line))
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(cand_records):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    pairs = sorted(cand_by_pair.keys())

    npz_u = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids = list(npz_u["user_ids"])
    user_hiddens = npz_u["hiddens"].astype(np.float32)
    n_users_total = user_hiddens.shape[0]

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals = np.load(IN_CAND_RESID).astype(np.float32)[:, layer_idx, :]
    log(f"  user_residuals: {user_residuals.shape}, cand_residuals: {cand_residuals.shape}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    log(f"[2] Multi-split training + LOPO ({len(SPLITS)} splits) ...")
    all_splits_summary = []
    all_per_pair = []

    for split_id in SPLITS:
        log(f"\n--- Split {split_id} ---")
        t0 = time.time()
        rng = np.random.default_rng(split_id)
        all_indices = np.arange(n_users_total)
        test_indices = rng.choice(all_indices, size=N_TEST_USERS, replace=False)
        test_uid_set = set(int(i) for i in test_indices)
        train_indices = np.array([i for i in range(n_users_total) if i not in test_uid_set])

        log(f"  train users: {len(train_indices)}, test users: {len(test_uid_set)}")

        # Train on TRAIN users only
        train_uids_local = np.arange(len(train_indices))  # local indices for training
        W, b, losses = train_contrastive_projection(
            user_residuals, train_indices,
            epochs=EPOCHS, batch_users=BATCH_USERS, lr=LR,
        )
        log(f"  trained projection: W={W.shape}, b={b.shape}")

        # Project all user residuals + cand residuals
        # Use mean pool across sentences for each user
        user_mean_resid = user_residuals.mean(axis=1)  # (n_users, hidden)
        proj_user = user_mean_resid @ W + b  # (n_users, proj_dim)
        proj_user_norm = proj_user / (np.linalg.norm(proj_user, axis=-1, keepdims=True) + 1e-9)

        proj_cand = cand_residuals @ W + b  # (n_cands, proj_dim)
        proj_cand_norm = proj_cand / (np.linalg.norm(proj_cand, axis=-1, keepdims=True) + 1e-9)

        # Cosine distance as score (lower = more similar)
        # Maha-like: weighted by inverse variance per dim
        # For simplicity: cosine distance
        # Score = 1 - cosine_sim
        # LOPO
        test_pairs = [(uid, asin) for (uid, asin) in pairs if uid_to_idx.get(uid) in test_uid_set]
        log(f"  test pairs: {len(test_pairs)}")

        split_records = []
        for (uid, asin) in test_pairs:
            target_idx = uid_to_idx[uid]
            cand_idxs = cand_by_pair[(uid, asin)]
            if not cand_idxs:
                continue
            cand_emb = proj_cand_norm[cand_idxs]  # (n_cands, proj_dim)
            # Cosine distance: 1 - sim
            sim = cand_emb @ proj_user_norm.T  # (n_cands, n_users)
            dist = 1 - sim
            # Smaller dist = better. Rank: count of users with smaller dist
            target_dist = dist[:, target_idx: target_idx + 1]
            rank_per_cand = (dist < target_dist).sum(axis=1)
            best_rank = int(np.min(rank_per_cand))

            split_records.append({
                "split": split_id,
                "user_id": uid,
                "asin": asin,
                "best_rank": best_rank,
                "n_cands": len(cand_idxs),
            })

        agg = aggregate([r["best_rank"] for r in split_records])
        log(f"  rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.2f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.2f}%), "
            f"MRR={agg['mrr']:.4f}, mean_rank={agg['mean_rank']:.2f}")

        all_splits_summary.append({
            "split": split_id,
            "n_test_users": len(test_uid_set),
            "n_test_pairs": len(test_pairs),
            **agg,
        })
        all_per_pair.extend(split_records)
        log(f"  elapsed: {time.time() - t0:.1f}s")

    # Aggregate
    all_ranks = [r["best_rank"] for r in all_per_pair]
    cross_split_agg = aggregate(all_ranks)
    log(f"\n=== CROSS-SPLIT AGGREGATE ({len(all_ranks)} pairs × {len(SPLITS)} splits) ===")
    log(f"  rank1={cross_split_agg['rank1']}/{cross_split_agg['n']} ({cross_split_agg['rank1_pct']*100:.2f}%) CI95={cross_split_agg['rank1_ci95']}")
    log(f"  top10={cross_split_agg['top10']}/{cross_split_agg['n']} ({cross_split_agg['top10_pct']*100:.2f}%)")
    log(f"  top100={cross_split_agg['top100']}/{cross_split_agg['n']} ({cross_split_agg['top100_pct']*100:.2f}%)")
    log(f"  MRR={cross_split_agg['mrr']:.4f} CI95={cross_split_agg['mrr_ci95']}")
    log(f"  mean_rank={cross_split_agg['mean_rank']:.2f} CI95={cross_split_agg['mean_rank_ci95']}")

    random_rank1 = 1.0 / n_users_total
    log(f"  random baseline rank1={random_rank1*100:.3f}% (1/{n_users_total})")

    out = {
        "phase": "15.2",
        "method": f"Supervised contrastive projection ({PROJ_DIM}d) + cosine LOPO",
        "n_users_total": n_users_total,
        "n_test_per_split": N_TEST_USERS,
        "splits": SPLITS,
        "proj_dim": PROJ_DIM,
        "epochs": EPOCHS,
        "temperature": TEMPERATURE,
        "random_baseline_rank1": random_rank1,
        "cross_split_summary": cross_split_agg,
        "per_split_summary": all_splits_summary,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in all_per_pair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 15.2 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()