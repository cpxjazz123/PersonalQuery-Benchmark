"""User-level CAV with N=10000, GPU-accelerated permutation test."""
import json
import gzip
import re
import time
import pickle
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

FILE = '/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz'
SENT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r'\s+')
GRID_X = [3, 5, 8, 15, 20]
GRID_Y = [20, 40]
N_SAMPLE = 10000
N_PERM = 100
N_BOOT = 500
CACHE_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_sents_cache2.pkl')
RESULT_PATH = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_level_cav_10000.json')
OUT_LOG = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper/user_level_cav_10000_run.log')


def log(msg):
    msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(msg, flush=True)
    with open(OUT_LOG, 'a') as f:
        f.write(msg + '\n')


# ============ Step 1: Load cache ============
log(f"loading cache from {CACHE_PATH}...")
t0 = time.time()
with open(CACHE_PATH, 'rb') as f:
    user_sents = pickle.load(f)
log(f"loaded {len(user_sents):,} users in {time.time()-t0:.0f}s")


# ============ GPU-accelerated CAV ============
def user_level_cav_gpu(z1_np, z2_np, rng, n_perm=N_PERM, n_boot=N_BOOT, device='cuda'):
    """GPU version: z1, z2 are numpy (n, 768)."""
    n_users = len(z1_np)
    if n_users < 2:
        return None
    z1 = torch.from_numpy(z1_np).to(device).float()
    z2 = torch.from_numpy(z2_np).to(device).float()
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + 1e-12)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + 1e-12)
    sim = z1 @ z2.T
    self_sims = sim.diag()
    mask = ~torch.eye(n_users, dtype=torch.bool, device=device)
    cross_sims = sim[mask]
    self_mean = float(self_sims.mean())
    cross_mean = float(cross_sims.mean())
    gap = self_mean - cross_mean
    # AUC (vectorized on GPU): for each user, fraction of cross-sims < self
    # shape (n_users, n_users); set diagonal to -inf
    sim_for_auc = sim.clone()
    sim_for_auc.fill_diagonal_(-float('inf'))
    gt = (sim_for_auc < self_sims.unsqueeze(1)).float().sum(dim=1)
    eq = (sim_for_auc == self_sims.unsqueeze(1)).float().sum(dim=1)
    auc = float(((gt + 0.5 * eq) / (n_users - 1)).mean())
    # acc: stricter accuracy using tril (lower triangle excluding diag)
    tril_mask = torch.tril(torch.ones(n_users, n_users, dtype=torch.bool, device=device), diagonal=-1)
    acc = float(((self_sims.unsqueeze(1) > sim) & tril_mask).float().sum() / max(1, tril_mask.sum().item()))
    boot_gaps = torch.zeros(n_boot, device=device)
    n_cross = len(cross_sims)
    for k in range(n_boot):
        bs_self_idx = torch.randint(0, n_users, (n_users,), device=device)
        bs_cross_idx = torch.randint(0, n_cross, (n_cross,), device=device)
        bs_self_sampled = self_sims[bs_self_idx]
        bs_cross_sampled = cross_sims[bs_cross_idx]
        boot_gaps[k] = bs_self_sampled.mean() - bs_cross_sampled.mean()
    boot_gaps_np = boot_gaps.cpu().numpy()
    ci_lo = float(np.percentile(boot_gaps_np, 2.5))
    ci_hi = float(np.percentile(boot_gaps_np, 97.5))
    # GPU permutation: each perm is a (n, n) matmul
    perm_gaps = torch.zeros(n_perm, device=device)
    eye_mask = ~torch.eye(n_users, dtype=torch.bool, device=device)
    for k in range(n_perm):
        perm = torch.randperm(n_users, device=device)
        perm_sim = z1 @ z2[perm].T
        p_self = perm_sim.diag().mean()
        p_cross = perm_sim[eye_mask].mean()
        perm_gaps[k] = p_self - p_cross
    perm_gaps_np = perm_gaps.cpu().numpy()
    perm_p = float(((perm_gaps_np >= gap).sum() + 1) / (n_perm + 1))
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


# ============ Step 2: Per-cell CAV ============
log("loading AnnaWegmann/Style-Embedding model...")
from sentence_transformers import SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"device: {device}")
model = SentenceTransformer("AnnaWegmann/Style-Embedding", device=device)

rng = np.random.default_rng(42)
results = {}

# Pre-compute per-user sX counts for each X once
log("pre-computing per-user sX counts for all X...")
x_counts = {X: [] for X in GRID_X}
all_uids = list(user_sents.keys())
for u in all_uids:
    sents = user_sents[u]
    # Count per X
    cnt_per_x = [0] * len(GRID_X)
    for s, wc, a in sents:
        for i, X in enumerate(GRID_X):
            if wc >= X:
                cnt_per_x[i] += 1
    for i, X in enumerate(GRID_X):
        x_counts[X].append(cnt_per_x[i])
x_counts = {X: dict(zip(all_uids, cnts)) for X, cnts in x_counts.items()}
log("pre-compute done")

for X in GRID_X:
    for Y in GRID_Y:
        cell_key = f"{X}_{Y}"
        t0 = time.time()
        eligible = [u for u, c in x_counts[X].items() if c >= Y]
        n_elig = len(eligible)
        if n_elig < N_SAMPLE:
            log(f"X={X} Y={Y}: SKIP ({n_elig} eligible, need {N_SAMPLE})")
            results[cell_key] = {
                "X": X, "Y": Y, "n_eligible": n_elig, "n_sampled": 0,
                "self_sim": None, "cross_sim": None, "gap": None,
                "auc": None, "perm_p": None,
            }
            continue
        n_sample = min(N_SAMPLE, n_elig)
        sampled = rng.choice(eligible, size=n_sample, replace=False).tolist()
        sents_half1 = []
        uids_half1 = []
        sents_half2 = []
        uids_half2 = []
        for u in sampled:
            u_sents = [(s, wc, a) for s, wc, a in user_sents[u] if wc >= X]
            if len(u_sents) > Y:
                idx = rng.choice(len(u_sents), size=Y, replace=False)
                u_sents = [u_sents[i] for i in idx]
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
        t_embed = time.time()
        e1 = model.encode(sents_half1, convert_to_tensor=True,
                          show_progress_bar=False, batch_size=128).cpu().numpy()
        e2 = model.encode(sents_half2, convert_to_tensor=True,
                          show_progress_bar=False, batch_size=128).cpu().numpy()
        embed_t = time.time() - t_embed
        z1_dict = defaultdict(list)
        z2_dict = defaultdict(list)
        for i, u in enumerate(uids_half1):
            z1_dict[u].append(e1[i])
        for i, u in enumerate(uids_half2):
            z2_dict[u].append(e2[i])
        ordered_users = list(z1_dict.keys())
        z1 = np.array([np.mean(z1_dict[u], axis=0) for u in ordered_users])
        z2 = np.array([np.mean(z2_dict[u], axis=0) for u in ordered_users])
        t_cav = time.time()
        cav = user_level_cav_gpu(z1, z2, rng, n_perm=N_PERM, n_boot=N_BOOT, device=device)
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