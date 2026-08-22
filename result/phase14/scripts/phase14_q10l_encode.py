#!/usr/bin/env python3
"""Phase 14.Q10-L encode: 2041 users × 30 sentences + 16328 candidates via Qwen."""
from __future__ import annotations
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_GEN = OUT_DIR / "phase14_q10l_generated.jsonl"
IN_USER_REVIEWS = OUT_DIR / "phase14_q10l_user_reviews.pkl"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"

OUT_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
OUT_CAND_RESID = OUT_DIR / "phase14_q10l_cand_residuals_qwen.npy"

LAYERS_5 = [8, 14, 18, 22, 26]
HIDDEN = 3584
N_SENTS = 30
MIN_SENT_LEN = 5
MAX_SENT_LEN = 30
BATCH_SIZE = 64


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q10-L encode: 2041 users × 30 sents + 16328 candidates")
    log("=" * 70)

    log("[1] Loading pairs + user reviews ...")
    pairs = []
    with (OUT_DIR / "phase14_q10l_pairs.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    with IN_USER_REVIEWS.open("rb") as f:
        user_reviews = pickle.load(f)
    log(f"  user reviews: {len(user_reviews):,} users")

    log("[2] Preparing user sentences ...")
    user_sents = {}
    for pair in pairs:
        uid = pair["user_id"]
        reviews = user_reviews.get(uid, [])
        sents = []
        for r in reviews:
            text = r["text"]
            if not text:
                continue
            # split into sentences (very rough: split on . ! ?)
            import re
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            for p in parts:
                words = p.split()
                if MIN_SENT_LEN <= len(words) <= MAX_SENT_LEN:
                    sents.append(p)
                if len(sents) >= N_SENTS:
                    break
            if len(sents) >= N_SENTS:
                break
        if len(sents) >= N_SENTS:
            user_sents[uid] = sents[:N_SENTS]
        else:
            # pad with repeats if too few
            while len(sents) < N_SENTS and sents:
                sents.append(sents[0])
            user_sents[uid] = sents
    log(f"  users with {N_SENTS} sents: {sum(1 for v in user_sents.values() if len(v) == N_SENTS)}/{len(pairs)}")

    log("[3] Loading Qwen client (transformers) ...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    log(f"[4] Encoding {len(user_sents)} users × {N_SENTS} sents ...")
    user_ids = []
    all_user_texts = []
    for uid in sorted(user_sents.keys()):
        user_ids.append(uid)
        all_user_texts.extend(user_sents[uid])
    log(f"  total user sentences: {len(all_user_texts)}")

    t0 = time.time()
    hdict_user = client.get_hidden_states(
        all_user_texts,
        layers=LAYERS_5,
        batch_size=BATCH_SIZE,
        max_length=128,
    )
    elapsed = time.time() - t0
    log(f"  encoded {len(all_user_texts)} sents in {elapsed:.1f}s ({len(all_user_texts)/elapsed:.1f} sent/s)")

    # reshape: (n_users, N_SENTS, 5, 3584)
    n_users = len(user_ids)
    user_hiddens_raw = np.stack([hdict_user[l] for l in LAYERS_5], axis=1).astype(np.float32)
    log(f"  raw shape: {user_hiddens_raw.shape}")
    user_hiddens = user_hiddens_raw.reshape(n_users, N_SENTS, len(LAYERS_5), HIDDEN)
    log(f"  user_hiddens shape: {user_hiddens.shape}")

    np.savez(OUT_USER_HIDDENS, user_ids=np.array(user_ids), hiddens=user_hiddens)
    log(f"  → {OUT_USER_HIDDENS}")

    log("[5] Encoding candidates ...")
    cand_records = []
    with IN_GEN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records.append(json.loads(line))
    log(f"  candidates: {len(cand_records)}")

    cand_texts = [r.get("q_styled") or "" for r in cand_records]
    t0 = time.time()
    hdict_cand = client.get_hidden_states(
        cand_texts,
        layers=LAYERS_5,
        batch_size=BATCH_SIZE,
        max_length=128,
    )
    log(f"  encoded {len(cand_texts)} cands in {time.time() - t0:.1f}s")

    cand_hiddens_all = np.stack([hdict_cand[l] for l in LAYERS_5], axis=1).astype(np.float32)
    log(f"  cand_hiddens_all shape: {cand_hiddens_all.shape}")

    log("[6] Loading global neutral ...")
    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)
    log(f"  global_neutral shape: {global_neutral.shape}")

    cand_residuals = cand_hiddens_all - global_neutral[None, :, :]
    log(f"  cand_residuals shape: {cand_residuals.shape}")

    np.save(OUT_CAND_RESID, cand_residuals)
    log(f"  → {OUT_CAND_RESID}")

    log("=" * 70)
    log("PHASE 14.Q10-L ENCODE COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()