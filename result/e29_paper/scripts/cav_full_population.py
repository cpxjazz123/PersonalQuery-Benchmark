"""CAV on full Baby Products population, sample 100 users per (X, Y) cell."""
import json
import gzip
import re
import time
import pickle
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz'
SENT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r'\s+')
GRID_X = [3, 5, 8, 15, 20, 30, 40, 50]
GRID_Y = [20, 40, 80, 200]
N_SAMPLE = 100
N_PERM = 100  # reduced from 200 for speed
CACHE_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache.pkl')
RESULT_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/cav_full_pop.json')

OUT_LOG = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/cav_full_run.log')


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


# ============ Step 1: Build user sentences cache ============
if CACHE_PATH.exists():
    log(f"loading cache from {CACHE_PATH}...")
    t0 = time.time()
    with open(CACHE_PATH, 'rb') as f:
        user_sents = pickle.load(f)
    log(f"loaded {len(user_sents):,} users in {time.time()-t0:.0f}s")
else:
    log("building user cache from reviews...")
    user_sents = {}
    n_rec = 0
    t0 = time.time()
    with gzip.open(FILE, 'rt') as f:
        for line in f:
            n_rec += 1
            if n_rec % 1000000 == 0:
                log(f"  {n_rec/1e6:.1f}M records, {len(user_sents):,} users")
            o = json.loads(line)
            u = o['user_id']
            text = o.get('text', '') or ''
            if not text:
                continue
            if u not in user_sents:
                user_sents[u] = []
            for s in SENT_RE.split(text):
                s = s.strip()
                if not s:
                    continue
                wc = len(WORD_RE.split(s)) - 1
                user_sents[u].append((s, wc))
    log(f"built cache: {len(user_sents):,} users in {time.time()-t0:.0f}s")
    log(f"saving to {CACHE_PATH}...")
    t0 = time.time()
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(user_sents, f, protocol=4)
    log(f"saved in {time.time()-t0:.0f}s")


# ============ CAV function (E29-compatible) ============
def cav_per_sent(embs, uid, rng, n_perm=N_PERM):
    """CAV accuracy + permutation p-value. Returns (acc, perm_p, auc, own_other)."""
    n = len(embs)
    if n < 10:
        return 0.5, 1.0, 0.5, 0.0
    user_to_idx = defaultdict(list)
    for i, u in enumerate(uid):
        user_to_idx[u].append(i)
    user_ids = list(user_to_idx.keys())
    if len(user_ids) < 2:
        return 0.5, 1.0, 0.5, 0.0
    pos_idx_all = []
    neg_idx_all = []
    for i, u in enumerate(uid):
        same = [j for j in user_to_idx[u] if j != i]
        if not same:
            pos_idx_all.append(-1)
            neg_idx_all.append(-1)
            continue
        pos_idx_all.append(int(rng.choice(same)))
        other_users = [ou for ou in user_ids if ou != u]
        ou = rng.choice(other_users)
        neg_idx_all.append(int(rng.choice(user_to_idx[ou])))
    pos_arr = np.array(pos_idx_all)
    neg_arr = np.array(neg_idx_all)
    valid = pos_arr >= 0
    if valid.sum() < 10:
        return 0.5, 1.0, 0.5, 0.0
    a = embs[valid]
    p = embs[pos_arr[valid]]
    neg = embs[neg_arr[valid]]
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    p_n = p / np.linalg.norm(p, axis=1, keepdims=True)
    n_n = neg / np.linalg.norm(neg, axis=1, keepdims=True)
    sim_p = (a_n * p_n).sum(axis=1)
    sim_n = (a_n * n_n).sum(axis=1)
    total = len(sim_p)
    acc = float((sim_p > sim_n).sum()) / total
    auc = float(((sim_p > sim_n).sum() + 0.5 * (sim_p == sim_n).sum()) / total)
    own_other = float((sim_p - sim_n).mean())
    # Permutation test
    perm_acc = np.zeros(n_perm)
    valid_idx_arr = np.where(valid)[0]
    n_v = len(valid_idx_arr)
    user_to_idx_arr = [np.array(user_to_idx[u]) for u in user_ids]
    sizes = np.array([len(arr) for arr in user_to_idx_arr])
    cum = np.concatenate([[0], np.cumsum(sizes)])
    starts = cum[:-1]
    n_users = len(user_ids)
    anchor_uid = np.array([user_ids.index(uid[j]) for j in valid_idx_arr])
    flat_to_orig = np.concatenate(user_to_idx_arr)
    BATCH = 50
    chunk = 5000
    for p_start in range(0, n_perm, BATCH):
        p_end = min(p_start + BATCH, n_perm)
        b = p_end - p_start
        rand_u = rng.integers(0, n_users, size=(b, n_v))
        rand_u = np.where(rand_u == anchor_uid[None, :],
                          (rand_u + 1) % n_users, rand_u)
        rand_j = rng.integers(0, sizes[rand_u])
        flat_pos = starts[rand_u] + rand_j
        orig_pos = flat_to_orig[flat_pos.ravel()].reshape(b, n_v)
        win = np.zeros(b)
        for c_start in range(0, n_v, chunk):
            c_end = min(c_start + chunk, n_v)
            a_chunk = a_n[c_start:c_end]
            sim_n_chunk = sim_n[c_start:c_end]
            orig_chunk = orig_pos[:, c_start:c_end]
            p_emb = embs[orig_chunk]
            p_emb_n = p_emb / np.linalg.norm(p_emb, axis=2, keepdims=True)
            sim_p_chunk = (a_chunk[None, :, :] * p_emb_n).sum(axis=2)
            win += (sim_p_chunk > sim_n_chunk[None, :]).sum(axis=1)
        perm_acc[p_start:p_end] = win / total
    perm_p = float(((perm_acc >= acc).sum() + 1) / (n_perm + 1))
    return acc, perm_p, auc, own_other


# ============ Step 2: Per-cell CAV ============
log("loading AnnaWegmann/Style-Embedding model...")
import torch
from sentence_transformers import SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"device: {device}")
model = SentenceTransformer("AnnaWegmann/Style-Embedding", device=device)

rng = np.random.default_rng(42)
results = {}

for X in GRID_X:
    for Y in GRID_Y:
        cell_key = f"{X}_{Y}"
        # Eligible users
        t0 = time.time()
        eligible = [u for u, sents in user_sents.items()
                    if sum(1 for s, wc in sents if wc >= X) >= Y]
        n_elig = len(eligible)
        if n_elig < 2:
            log(f"X={X} Y={Y}: SKIP (only {n_elig} eligible users)")
            results[cell_key] = {
                "X": X, "Y": Y, "n_eligible": n_elig, "n_sampled": 0,
                "acc": None, "perm_p": None, "auc": None, "own_other": None,
            }
            continue
        # Sample up to 100 users
        n_sample = min(N_SAMPLE, n_elig)
        sampled = rng.choice(eligible, size=n_sample, replace=False).tolist()
        # Collect sentences
        all_sents = []
        all_uid = []
        for u in sampled:
            u_sents = [(s, wc) for s, wc in user_sents[u] if wc >= X]
            if len(u_sents) > Y:
                idx = rng.choice(len(u_sents), size=Y, replace=False)
                u_sents = [u_sents[i] for i in idx]
            for s, _ in u_sents:
                all_sents.append(s)
                all_uid.append(u)
        # Embed
        t_embed = time.time()
        embs = model.encode(all_sents, convert_to_tensor=True,
                            show_progress_bar=False, batch_size=128).cpu().numpy()
        embed_t = time.time() - t_embed
        # CAV
        t_cav = time.time()
        acc, perm_p, auc, oo = cav_per_sent(embs, all_uid, rng, n_perm=N_PERM)
        cav_t = time.time() - t_cav
        cell_t = time.time() - t0
        results[cell_key] = {
            "X": X, "Y": Y, "n_eligible": n_elig, "n_sampled": n_sample,
            "n_sents": len(all_sents),
            "acc": float(acc), "perm_p": float(perm_p),
            "auc": float(auc), "own_other": float(oo),
            "embed_time_s": round(embed_t, 1),
            "cav_time_s": round(cav_t, 1),
            "total_time_s": round(cell_t, 1),
        }
        log(f"X={X:>3} Y={Y:>3} | n_user={n_sample} n_sent={len(all_sents)} "
            f"acc={acc:.3f} perm_p={perm_p:.3g} oo={oo:+.3f} "
            f"t={cell_t:.0f}s (embed={embed_t:.0f}s, cav={cav_t:.0f}s)")
        # Save incrementally
        with open(RESULT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

log(f"\nDONE. saved {RESULT_PATH}")
