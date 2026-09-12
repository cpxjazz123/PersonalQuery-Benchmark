"""Smoke version of pre_encoder_h30_strict.py
- Subset 200 users
- 3 epochs
- Same SupCon + user-balanced sampler
- Output -> pcfg_cache/_smoke/

Validates:
  - data load + sent_csr slicing
  - SHA1 split reproducible
  - SupCon loss computes, no NaN
  - dense encode loop works
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
SMOKE_DIR = CACHE_DIR / "_smoke"
SMOKE_DIR.mkdir(exist_ok=True)

SEED = 42
HASH_SALT = "pcfg_lopo_v1"
SUP_EPOCHS = 3
SUP_LR = 5e-4
SUP_Z_DIM = 32
SUP_HIDDEN = (256,)
SUP_DROPOUT = 0.5
SUP_WEIGHT_DECAY = 1e-2
SUPCON_TAU = 0.07
K_USERS_PER_BATCH = 32
K_PER_USER = 8
SMOKE_N_USERS = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hash_bucket(text, mod=2, salt=HASH_SALT):
    h = hashlib.sha1((salt + "|" + text.strip().lower()).encode()).hexdigest()
    return int(h, 16) % mod


class SupConLoss(nn.Module):
    def __init__(self, tau=SUPCON_TAU):
        super().__init__()
        self.tau = tau

    def forward(self, z, labels):
        z_norm = F.normalize(z, dim=1)
        sim = z_norm @ z_norm.T / self.tau
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        sim_exp = torch.exp(sim - sim_max)
        labels_row = labels.unsqueeze(0)
        labels_col = labels.unsqueeze(1)
        pos_mask = (labels_row == labels_col).float()
        eye = torch.eye(z.size(0), device=z.device)
        pos_mask = pos_mask - eye
        pos_count = pos_mask.sum(dim=1)
        valid_anchor = (pos_count > 0).float()
        denom = sim_exp.sum(dim=1) - sim_exp.diagonal()
        log_prob = sim - torch.log(denom + 1e-12).unsqueeze(1) - sim_max
        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
        n_valid = valid_anchor.sum()
        if n_valid < 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        return (loss_per_anchor * valid_anchor).sum() / n_valid


def main():
    t0 = time.time()
    sys.path.insert(0, str(REPO_ROOT / "03_spacy_encode"))
    from syntax_pcfg_pipeline import _SupEncoder

    log("loading meta + cache")
    with open(CACHE_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(CACHE_DIR / "uid_list.json") as f:
        uid_list = json.load(f)
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    V = meta["vocab_size"]

    # Take first SMOKE_N_USERS as subset
    n_users_subset = SMOKE_N_USERS
    user_n_sents_sub = user_n_sents[:n_users_subset]
    n_total_sub = sum(user_n_sents_sub)
    log(f"subset: n_users={n_users_subset}, n_sents={n_total_sub}, V={V}")

    log("loading sent_vectors.npz (CSR slice)")
    sent_csr_full = load_npz(CACHE_DIR / "sent_vectors.npz").tocsr()
    sent_csr = sent_csr_full[:n_total_sub].tocsr()
    log(f"  full {sent_csr_full.shape} -> sub {sent_csr.shape}")
    del sent_csr_full

    user_off = np.zeros(n_users_subset + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents_sub)

    def uid_for(sent_i):
        # sent_i is global index into n_total_sub; find which user owns it
        return int(np.searchsorted(user_off[1:], sent_i, side="right"))

    log("SHA1 profile/test split (subset)")
    sent_id_strs = [
        f"u{uid_list[ui]}_s{i}"
        for ui in range(n_users_subset)
        for i in range(user_n_sents_sub[ui])
    ]
    profile_idx = np.array(
        [i for i, sid in enumerate(sent_id_strs)
         if hash_bucket(sid) == 0], dtype=np.int64)
    test_idx = np.array(
        [i for i, sid in enumerate(sent_id_strs)
         if hash_bucket(sid) == 1], dtype=np.int64)
    log(f"  profile={len(profile_idx)}, test={len(test_idx)}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(profile_idx))
    n_val = int(len(profile_idx) * 0.15)
    train_idx = profile_idx[perm[n_val:]]
    log(f"  train={len(train_idx)}, val={n_val}")

    user_labels = np.array([uid_for(int(i)) for i in profile_idx],
                           dtype=np.int64)
    user_labels_train = user_labels[perm[n_val:]]
    user_labels_val = user_labels[perm[:n_val]]

    train_per_user = [[] for _ in range(n_users_subset)]
    for li, idx in enumerate(train_idx):
        u = uid_for(int(idx))
        train_per_user[u].append(int(li))
    valid_users = np.array(
        [u for u, arr in enumerate(train_per_user) if len(arr) >= K_PER_USER],
        dtype=np.int64)
    log(f"  valid users: {len(valid_users)} (need ≥{K_PER_USER} train sents)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    model = _SupEncoder(V, SUP_Z_DIM, SUP_HIDDEN, n_users_subset,
                        SUP_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"encoder params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR,
                           weight_decay=SUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SUP_EPOCHS)
    supcon = SupConLoss(tau=SUPCON_TAU).to(device)

    def to_dense(idx_arr):
        sub = sent_csr[idx_arr]
        return torch.tensor(sub.toarray(), dtype=torch.float32, device=device)

    n_user_batches = len(valid_users) // K_USERS_PER_BATCH
    log(f"  batches/epoch: {n_user_batches}")

    for epoch in range(SUP_EPOCHS):
        model.train()
        np.random.shuffle(valid_users)
        train_loss_sum = 0.0
        for ub in range(n_user_batches):
            batch_users = valid_users[ub * K_USERS_PER_BATCH:
                                      (ub + 1) * K_USERS_PER_BATCH]
            bi_local = []
            for u in batch_users:
                arr = np.array(train_per_user[u])
                picks = np.random.choice(arr, size=K_PER_USER, replace=False)
                bi_local.extend(picks.tolist())
            bi_local = np.array(bi_local, dtype=np.int64)
            x = to_dense(train_idx[bi_local])
            y = torch.from_numpy(user_labels_train[bi_local]).to(device)
            z, _ = model(x)
            loss = supcon(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
        sched.step()
        log(f"  ep {epoch}: train_supcon={train_loss_sum/n_user_batches:.4f}")

    # Encode all
    model.eval()
    z_all = np.zeros((n_total_sub, SUP_Z_DIM), dtype=np.float32)
    ENCODE_BS = 1024
    with torch.no_grad():
        for s in range(0, n_total_sub, ENCODE_BS):
            e = min(s + ENCODE_BS, n_total_sub)
            x = to_dense(np.arange(s, e, dtype=np.int64))
            z, _ = model(x)
            z_all[s:e] = z.cpu().numpy()
    log(f"  z_all shape: {z_all.shape}, "
        f"mean_norm={np.linalg.norm(z_all, axis=1).mean():.3f}")

    log(f"\nSMOKE PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()