#!/usr/bin/env python3
"""Phase 10.16.A: Build 318d user vectors for all 876 phase10 users.

Strategy:
  1. Load e22_t1 train_mean / train_std (computed from e22_t1's training set)
  2. Load VADES 318d sentence features (raw, un-normalized)
  3. For each phase10 user: compute mean over their sentences, then z-score
     normalize using e22_t1's train_mean/train_std
  4. Save phase10_user_vectors_318d.npz with same schema as e22_t1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_VECTORS = OUT_DIR / "phase10_user_vectors_318d.npy"
OUT_USER_IDS = OUT_DIR / "phase10_user_vectors_318d_user_ids.json"
OUT_META = OUT_DIR / "phase10_user_vectors_318d_meta.json"

# Sources
E22_T1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
E22_T1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
SENT_FEATS = OUT_DIR / "phase10_10_6_cache" / "sentence_318d.npy"
SENT_USERS = OUT_DIR / "phase10_10_6_cache" / "sentence_318d_users.json"

VADES_PROFILES = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"

PHASE10_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"


def main():
    log = lambda m: print(f"[phase10-16.A] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.16.A: Build 318d user vectors for 876 phase10 users")
    log("=" * 70)

    # === Load e22_t1 normalization stats ===
    log("[1] Loading e22_t1 normalization stats ...")
    e22 = np.load(E22_T1_VECTORS)
    train_mean = e22["train_mean"].astype(np.float64)  # (318,)
    train_std = e22["train_std"].astype(np.float64)  # (318,)
    log(f"  train_mean: {train_mean.shape}, train_std: {train_std.shape}")
    log(f"  e22_t1 user count: {len(e22['user_ids'])}")

    # === Load VADES sentence 318d features (raw, un-normalized) ===
    log("[2] Loading VADES 318d sentence features ...")
    sent_feats = np.load(SENT_FEATS)
    sent_users = json.loads(SENT_USERS.read_text())
    log(f"  sent_feats: {sent_feats.shape}")
    log(f"  sent_users: {len(sent_users)}")

    if sent_feats.shape[1] != 318:
        raise ValueError(f"Expected 318 dim sentence feats, got {sent_feats.shape[1]}")

    # === Load VADES profiles for user_id mapping ===
    log("[3] Loading VADES profiles ...")
    profiles = []
    with VADES_PROFILES.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(profiles)
    log(f"  VADES users: {n_users}")

    # === Load phase10 pairs ===
    log("[4] Loading phase10 pairs ...")
    pairs = []
    with PHASE10_PAIRS.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    phase10_users_set = set(p["user_id"] for p in pairs)
    phase10_users_list = sorted(phase10_users_set)
    log(f"  phase10 users: {len(phase10_users_set)}")

    # === Compute raw user_mu for each phase10 user (mean of sentence features) ===
    log("[5] Computing raw user_mu for each phase10 user ...")
    # user_mu_map: user_id -> (318,) raw mean
    user_mu_map = {}
    sentence_counts = {}
    for si in range(len(sent_users)):
        uid = sent_users[si]
        if uid not in phase10_users_set:
            continue
        if uid not in user_mu_map:
            user_mu_map[uid] = np.zeros(318, dtype=np.float64)
            sentence_counts[uid] = 0
        user_mu_map[uid] += sent_feats[si]
        sentence_counts[uid] += 1

    for uid in user_mu_map:
        user_mu_map[uid] /= sentence_counts[uid]

    log(f"  raw user_mu computed for {len(user_mu_map)} users")

    # === Z-score normalize using e22_t1 train_mean/train_std ===
    log("[6] Z-score normalizing using e22_t1 train_mean/train_std ...")
    # train_std may have zeros (zero-variance dims); avoid div-by-zero
    safe_std = np.where(train_std > 1e-9, train_std, 1.0)

    phase10_user_ids = []
    phase10_user_vectors = []
    skipped = []
    for uid in phase10_users_list:
        if uid not in user_mu_map:
            skipped.append(uid)
            continue
        raw = user_mu_map[uid]
        normed = (raw - train_mean) / safe_std
        # Clip extreme values
        normed = np.clip(normed, -10.0, 10.0)
        phase10_user_ids.append(uid)
        phase10_user_vectors.append(normed)

    phase10_user_vectors = np.array(phase10_user_vectors, dtype=np.float32)
    log(f"  normalized vectors: {phase10_user_vectors.shape}")
    log(f"  skipped (no sentences): {len(skipped)}")

    # Check overlap with e22_t1 to verify consistency
    e22_user_ids = set(str(u) for u in e22["user_ids"])
    overlap_users = [u for u in phase10_user_ids if u in e22_user_ids]
    if overlap_users:
        log(f"  overlap with e22_t1: {len(overlap_users)} users")
        # Compare first overlap user
        u0 = overlap_users[0]
        i0 = phase10_user_ids.index(u0)
        our_vec = phase10_user_vectors[i0]
        j0 = list(e22["user_ids"]).index(u0)
        e22_vec = e22["Z_full"][j0]
        cos_sim = float(np.dot(our_vec, e22_vec) / (np.linalg.norm(our_vec) * np.linalg.norm(e22_vec) + 1e-9))
        log(f"    first overlap user {u0}: cosine similarity (ours vs e22_t1) = {cos_sim:.4f}")
        # If low, our normalization differs from e22_t1's actual computation

    # === Save ===
    log("[7] Saving ...")
    np.save(OUT_VECTORS, phase10_user_vectors)
    OUT_USER_IDS.write_text(json.dumps(phase10_user_ids))
    log(f"  vectors → {OUT_VECTORS}")
    log(f"  user_ids → {OUT_USER_IDS}")

    # Meta
    meta = {
        "n_users": len(phase10_user_ids),
        "skipped_no_sentences": len(skipped),
        "vector_dim": 318,
        "normalization_source": "e22_t1_user_vectors.npz train_mean/train_std",
        "feature_space": "318d length-invariant (ALL_FEATS_V2)",
        "method": "per-user mean over sentences → z-score using e22_t1 train stats",
        "overlap_with_e22_t1": len(overlap_users),
        "skipped_user_ids": skipped[:20],  # first 20
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 10.16.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()