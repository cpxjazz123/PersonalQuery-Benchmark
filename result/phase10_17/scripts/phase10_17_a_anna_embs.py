#!/usr/bin/env python3
"""Phase 10.17.A: Extract 768d AnnaWegmann style embeddings for 876 phase10 users.

Reuses AnnaWegmann/Style-Embedding (sentence_transformers, 768d). For each of
the 876 phase10 users, collect all train sentences (exclude is_holdout=True)
from vades_prototype_3000u_v6_raw_sentences.jsonl (43770 sentences, 2918 users),
mean-pool per user -> 768d vector.

Why this is fast:
- sentence_transformers .encode() with batch_size=128 handles 43k sentences in
  ~30s on a GPU.
- After encoding, per-user mean-pool is just a dict aggregation.

Output: /home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_user_embs_768d.npz
        keys: user_ids (876,), embs (876, 768)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
RAW_SENTENCES_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"

PHASE10_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"

ANNA_SNAPSHOT_DIR = "/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots"

ENCODER_BATCH = 128


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 10.17.A: 768d AnnaWegmann style embeddings for 876 phase10 users")
    log("=" * 70)

    # === Load phase10 user_ids ===
    log("[1] Loading phase10 pairs (876 users) ...")
    phase10_users = set()
    with PHASE10_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            phase10_users.add(obj["user_id"])
    phase10_users = sorted(phase10_users)
    log(f"  phase10 users: {len(phase10_users)}")

    # === Load raw sentences (filter to phase10 users, exclude holdout) ===
    log("[2] Loading raw sentences (filter phase10 users, exclude holdout) ...")
    user_to_texts: dict[str, list[str]] = {u: [] for u in phase10_users}
    n_total = 0
    n_kept = 0
    with RAW_SENTENCES_FILE.open() as f:
        for line in f:
            n_total += 1
            obj = json.loads(line)
            uid = obj["user_id"]
            if uid not in user_to_texts:
                continue
            if obj.get("is_holdout", False):
                continue
            txt = obj.get("sentence_text", "").strip()
            if not txt:
                continue
            user_to_texts[uid].append(txt)
            n_kept += 1
    log(f"  total sentences: {n_total}, kept: {n_kept}")
    log(f"  users with >=1 train sentence: {sum(1 for v in user_to_texts.values() if v)}")
    n_missing = sum(1 for v in user_to_texts.values() if not v)
    if n_missing > 0:
        log(f"  WARNING: {n_missing} users have 0 train sentences (will use zero vector)")

    # === Load AnnaWegmann Style-Embedding ===
    log(f"[3] Loading AnnaWegmann/Style-Embedding from {ANNA_SNAPSHOT_DIR} ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import glob
    snapshots = sorted(glob.glob(os.path.join(ANNA_SNAPSHOT_DIR, "*")))
    if not snapshots:
        raise RuntimeError(f"No AnnaWegmann snapshots found in {ANNA_SNAPSHOT_DIR}")
    snapshot = snapshots[-1]
    log(f"  using snapshot: {snapshot}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(snapshot, device="cuda:0")
    log(f"  loaded, dim={model.get_sentence_embedding_dimension()}")

    # === Batch encode all texts ===
    log(f"[4] Encoding all texts (batch_size={ENCODER_BATCH}) ...")
    # Build flat list + index map
    flat_texts: list[str] = []
    flat_uids: list[str] = []
    for uid in phase10_users:
        texts = user_to_texts[uid]
        for t in texts:
            flat_texts.append(t)
            flat_uids.append(uid)
    log(f"  flat texts: {len(flat_texts)}")

    t0 = time.time()
    embs = model.encode(
        flat_texts, convert_to_numpy=True, batch_size=ENCODER_BATCH,
        show_progress_bar=False, normalize_embeddings=False
    )
    log(f"  encoded {len(flat_texts)} texts in {time.time()-t0:.1f}s, shape={embs.shape}")

    # === Per-user mean pool ===
    log("[5] Per-user mean pooling ...")
    dim = embs.shape[1]
    user_embs = np.zeros((len(phase10_users), dim), dtype=np.float32)
    counts = np.zeros(len(phase10_users), dtype=np.int32)
    uid_to_idx = {u: i for i, u in enumerate(phase10_users)}
    for txt_emb, uid in zip(embs, flat_uids):
        idx = uid_to_idx[uid]
        user_embs[idx] += txt_emb
        counts[idx] += 1
    valid = counts > 0
    user_embs[valid] /= counts[valid, None]
    log(f"  valid users: {int(valid.sum())}/{len(phase10_users)}")
    log(f"  mean count per user: {counts[valid].mean():.1f}, min/max: {counts[valid].min()}/{counts[valid].max()}")

    # === Sanity: identification on train sentences ===
    log("[6] Sanity: split-half cosine on user vectors ...")
    rng = np.random.default_rng(42)
    cos_sims = []
    for i, uid in enumerate(phase10_users):
        if not user_to_texts[uid]:
            continue
        texts = user_to_texts[uid]
        if len(texts) < 4:
            continue
        embs_local = model.encode(texts, convert_to_numpy=True, batch_size=ENCODER_BATCH,
                                  show_progress_bar=False, normalize_embeddings=False)
        for _ in range(min(10, len(texts) // 2)):
            perm = rng.permutation(len(embs_local))
            half = len(embs_local) // 2
            a = embs_local[perm[:half]].mean(0)
            b = embs_local[perm[half:]].mean(0)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            cos_sims.append(cos)
    cos_arr = np.array(cos_sims)
    log(f"  split-half cos (mean): {cos_arr.mean():.4f} ± {cos_arr.std():.4f} (n={len(cos_arr)})")

    # === Save ===
    log(f"[7] Saving to {OUT_EMBS} ...")
    np.savez(
        OUT_EMBS,
        user_ids=np.array(phase10_users, dtype=object),
        embs=user_embs,
        counts=counts,
    )
    log(f"  saved: user_ids={len(phase10_users)}, embs.shape={user_embs.shape}")

    # === Meta ===
    meta = {
        "encoder": "AnnaWegmann/Style-Embedding",
        "snapshot": snapshot,
        "n_users": len(phase10_users),
        "emb_dim": int(dim),
        "n_sentences_used": int(n_kept),
        "mean_count_per_user": float(counts[valid].mean()),
        "split_half_cos_mean": float(cos_arr.mean()) if len(cos_arr) > 0 else None,
        "split_half_cos_std": float(cos_arr.std()) if len(cos_arr) > 0 else None,
        "n_split_half_trials": int(len(cos_arr)),
    }
    META_OUT = OUT_DIR / "phase10_user_embs_768d_meta.json"
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta: {META_OUT}")

    log("=" * 70)
    log("PHASE 10.17.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()