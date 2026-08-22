"""User-level style vector comparison with split-half validation.

Methodology:
  For each user u:
    1. Collect Y qualifying sentences (each >= X words)
    2. Split into two halves, ideally from different products
    3. Embed each sent with AnnaWegmann/Style-Embedding
    4. Aggregate: z_u^(k) = mean(L2_norm(e)) for k in {1,2}, then L2 normalize again
  Per cell:
    self_sim   = mean over users of cos(z_u^(1), z_u^(2))
    cross_sim  = mean over user pairs (u,v), u != v, of cos(z_u, z_v)
    gap        = self_sim - cross_sim
    AUC        = P(sim_self > sim_cross) across all valid pairs
    bootstrap CI on gap
    permutation p-value (shuffle user labels)
"""
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
N_SAMPLE = 1000
N_PERM = 100
N_BOOT = 500
CACHE_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl')
RESULT_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_level_cav_1000.json')
OUT_LOG = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_level_cav_1000_run.log')


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


# ============ Step 1: Build user (sent, word_count, asin) cache ============
if CACHE_PATH.exists():
    log(f"loading cache from {CACHE_PATH}...")
    t0 = time.time()
    with open(CACHE_PATH, 'rb') as f:
        user_sents = pickle.load(f)
    log(f"loaded {len(user_sents):,} users in {time.time()-t0:.0f}s")
else:
    log("building user cache (sent, wc, asin)...")
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
            asin = o.get('parent_asin', '') or ''
            if not text:
                continue
            if u not in user_sents:
                user_sents[u] = []
            for s in SENT_RE.split(text):
                s = s.strip()
                if not s:
                    continue
                wc = len(WORD_RE.split(s)) - 1
                user_sents[u].append((s, wc, asin))
    log(f"built cache: {len(user_sents):,} users in {time.time()-t0:.0f}s")
    log(f"saving to {CACHE_PATH}...")
    t0 = time.time()
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(user_sents, f, protocol=4)
    log(f"saved in {time.time()-t0:.0f}s")


# ============ User-level CAV computation ============
def l2_norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def user_level_cav(z1, z2, rng, n_perm=N_PERM, n_boot=N_BOOT):
    """z1, z2: (n_users, dim). Self sim = cos(z1[u], z2[u]). Cross sim = cos(z1[u], z2[v])."""
    n_users = len(z1)
    if n_users < 2:
        return {"self_sim": None, "cross_sim": None, "gap": None,
                "auc": None, "acc": None, "boot_ci": None, "perm_p": None,
                "n_users": n_users}
    # Cosine sim matrix
    s1 = l2_norm(z1)
    s2 = l2_norm(z2)
    sim_matrix = s1 @ s2.T  # (n, n)
    # Self
    self_sims = np.array([sim_matrix[u, u] for u in range(n_users)])
    # Cross (off-diagonal)
    cross_pairs = []
    for u in range(n_users):
        for v in range(n_users):
            if u != v:
                cross_pairs.append(sim_matrix[u, v])
    cross_sims = np.array(cross_pairs)
    self_mean = float(self_sims.mean())
    cross_mean = float(cross_sims.mean())
    gap = self_mean - cross_mean
    # AUC: how often self_sim > cross_sim
    # For each user, fraction of cross-users with lower sim
    auc_list = []
    for u in range(n_users):
        cross_for_u = np.array([sim_matrix[u, v] for v in range(n_users) if v != u])
        gt = (cross_for_u < self_sims[u]).sum()
        eq = (cross_for_u == self_sims[u]).sum()
        auc_list.append((gt + 0.5 * eq) / len(cross_for_u))
    auc = float(np.mean(auc_list))
    # Accuracy: same as AUC but stricter
    acc = float((self_sims[:, None] > np.tril(sim_matrix, -1)).sum() /
                max(1, len(np.tril_indices(n_users, -1)[0])))
    # Bootstrap CI on gap
    boot_gaps = []
    for _ in range(n_boot):
        bs = rng.integers(0, n_users, size=n_users)
        bs_self = self_sims[bs]
        bs_cross = cross_sims[rng.integers(0, len(cross_sims), size=len(cross_sims))]
        boot_gaps.append(bs_self.mean() - bs_cross.mean())
    boot_gaps = np.array(boot_gaps)
    ci_lo = float(np.percentile(boot_gaps, 2.5))
    ci_hi = float(np.percentile(boot_gaps, 97.5))
    # Permutation p-value: shuffle user labels in z2
    perm_gaps = np.zeros(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n_users)
        ps2 = s2[perm]
        pmatrix = s1 @ ps2.T
        p_self = np.array([pmatrix[u, u] for u in range(n_users)]).mean()
        p_cross = np.array([pmatrix[u, v] for u in range(n_users)
                            for v in range(n_users) if u != v]).mean()
        perm_gaps[k] = p_self - p_cross
    perm_p = float(((perm_gaps >= gap).sum() + 1) / (n_perm + 1))
    return {
        "self_sim": self_mean,
        "cross_sim": cross_mean,
        "gap": float(gap),
        "auc": auc,
        "acc": acc,
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "perm_p": perm_p,
        "n_users": n_users,
    }


# ============ Step 2: Per-cell User-level CAV ============
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
        t0 = time.time()
        # Eligible users: ≥Y sents with ≥X words
        eligible = [u for u, sents in user_sents.items()
                    if sum(1 for s, wc, a in sents if wc >= X) >= Y]
        n_elig = len(eligible)
        if n_elig < N_SAMPLE:
            log(f"X={X} Y={Y}: SKIP ({n_elig} eligible)")
            results[cell_key] = {
                "X": X, "Y": Y, "n_eligible": n_elig, "n_sampled": 0,
                "self_sim": None, "cross_sim": None, "gap": None,
                "auc": None, "perm_p": None,
            }
            continue
        # Sample up to 100 users
        n_sample = min(N_SAMPLE, n_elig)
        sampled = rng.choice(eligible, size=n_sample, replace=False).tolist()
        # For each user: collect Y sents, split into halves
        sents_half1 = []  # sent
        uids_half1 = []
        sents_half2 = []
        uids_half2 = []
        for u in sampled:
            u_sents = [(s, wc, a) for s, wc, a in user_sents[u] if wc >= X]
            if len(u_sents) > Y:
                idx = rng.choice(len(u_sents), size=Y, replace=False)
                u_sents = [u_sents[i] for i in idx]
            # Split-half: random shuffle, first half / second half
            n = len(u_sents)
            perm = rng.permutation(n)
            h1 = perm[:n // 2]
            h2 = perm[n // 2:]
            for i in h1:
                sents_half1.append(u_sents[i][0])
                uids_half1.append(u)
            for i in h2:
                sents_half2.append(u_sents[i][0])
                uids_half2.append(u)
        # Embed both halves
        t_embed = time.time()
        e1 = model.encode(sents_half1, convert_to_tensor=True,
                          show_progress_bar=False, batch_size=128).cpu().numpy()
        e2 = model.encode(sents_half2, convert_to_tensor=True,
                          show_progress_bar=False, batch_size=128).cpu().numpy()
        embed_t = time.time() - t_embed
        # Aggregate per user
        z1_dict = defaultdict(list)
        z2_dict = defaultdict(list)
        for i, u in enumerate(uids_half1):
            z1_dict[u].append(e1[i])
        for i, u in enumerate(uids_half2):
            z2_dict[u].append(e2[i])
        ordered_users = list(z1_dict.keys())
        z1 = np.array([np.mean(z1_dict[u], axis=0) for u in ordered_users])
        z2 = np.array([np.mean(z2_dict[u], axis=0) for u in ordered_users])
        z1 = l2_norm(z1)
        z2 = l2_norm(z2)
        # User-level CAV
        t_cav = time.time()
        cav = user_level_cav(z1, z2, rng, n_perm=N_PERM, n_boot=N_BOOT)
        cav_t = time.time() - t_cav
        cell_t = time.time() - t0
        results[cell_key] = {
            "X": X, "Y": Y,
            "n_eligible": n_elig, "n_sampled": n_sample,
            "n_sents_h1": len(sents_half1), "n_sents_h2": len(sents_half2),
            **cav,
            "embed_time_s": round(embed_t, 1),
            "cav_time_s": round(cav_t, 1),
            "total_time_s": round(cell_t, 1),
        }
        log(f"X={X:>3} Y={Y:>3} | n_user={n_sample} n_sent={len(sents_half1)}/{len(sents_half2)} "
            f"self={cav['self_sim']:.3f} cross={cav['cross_sim']:.3f} "
            f"gap={cav['gap']:+.3f} AUC={cav['auc']:.3f} perm_p={cav['perm_p']:.3g} "
            f"t={cell_t:.0f}s (embed={embed_t:.0f}s, cav={cav_t:.0f}s)")
        with open(RESULT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

log(f"\nDONE. saved {RESULT_PATH}")