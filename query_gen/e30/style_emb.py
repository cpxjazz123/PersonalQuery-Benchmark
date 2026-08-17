"""E30 Step 2 — build per-user 768-dim AnnaWegmann style embeddings.

For each candidate user (across all picked products), encode their qualifying
sents (≥X words) with AnnaWegmann/Style-Embedding, then mean(L2_norm)→L2_norm.

Writes /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_style_embs.npz
"""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

USER_SENTS_PKL = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl')
PICKED_JSON = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_picked_products.json')
OUT_NPZ = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_style_embs.npz')

X = 8           # min words per sentence
MAX_SENTS = 40  # cap per user
EMB_BATCH = 128
MODEL_NAME = 'AnnaWegmann/Style-Embedding'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("loading picked products...")
    with open(PICKED_JSON) as f:
        picked = json.load(f)
    all_users = set()
    for p in picked['products']:
        all_users.update(p['candidate_users'])
    log(f"unique users across picked products: {len(all_users):,}")

    log("loading user_sents_cache2.pkl...")
    t0 = time.time()
    with open(USER_SENTS_PKL, 'rb') as f:
        user_sents = pickle.load(f)
    log(f"loaded {len(user_sents):,} users in {time.time()-t0:.0f}s")

    log("collecting qualifying sents per user...")
    t0 = time.time()
    user_qual_sents = {}
    for u in all_users:
        sents = user_sents.get(u, [])
        qual = [s for s, wc, a in sents if wc >= X][:MAX_SENTS]
        if len(qual) >= 5:  # need at least 5 sents to have a stable vector
            user_qual_sents[u] = qual
    log(f"users with ≥5 qualifying sents: {len(user_qual_sents):,} in {time.time()-t0:.0f}s")

    log(f"loading {MODEL_NAME} on GPU...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)

    uids = sorted(user_qual_sents.keys())
    log(f"encoding {len(uids):,} users × up to {MAX_SENTS} sents...")
    t0 = time.time()
    flat_sents = []
    flat_uid_idx = []
    for i, u in enumerate(uids):
        for s in user_qual_sents[u]:
            flat_sents.append(s)
            flat_uid_idx.append(i)
    log(f"flat sents: {len(flat_sents):,}")

    emb_t0 = time.time()
    flat_embs = model.encode(flat_sents, batch_size=EMB_BATCH,
                              show_progress_bar=False,
                              convert_to_numpy=True)
    log(f"encoded all flat sents in {time.time()-emb_t0:.0f}s, shape={flat_embs.shape}")

    log("aggregating per-user mean(L2_norm(e)) then L2_norm...")
    n_users = len(uids)
    agg = np.zeros((n_users, flat_embs.shape[1]), dtype=np.float32)
    counts = np.zeros(n_users, dtype=np.int32)
    for i, idx in enumerate(flat_uid_idx):
        agg[idx] += flat_embs[i]
        counts[idx] += 1
    # mean
    agg /= np.maximum(counts[:, None], 1)
    # L2 normalize per user
    norms = np.linalg.norm(agg, axis=1, keepdims=True) + 1e-12
    emb = (agg / norms).astype(np.float32)
    log(f"style embeddings shape: {emb.shape}, norm range [{emb.shape[0]:,}, 768]")

    np.savez_compressed(OUT_NPZ, user_ids=np.array(uids), embs=emb)
    log(f"saved {OUT_NPZ}")


if __name__ == '__main__':
    main()