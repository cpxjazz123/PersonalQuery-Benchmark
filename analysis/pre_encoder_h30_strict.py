"""Phase L8.25 — SupCon-based strict encoder for 64,996-user cohort.

WHY (per user 2026-09-11):
    stage_strict() with CE failed to converge on 64,996-class CE
    (val_acc=0.1%, no progress across epochs). Switch to
    Supervised Contrastive (SupCon) loss with user-balanced
    batch sampler (L8.21 lesson) so the encoder optimizes
    user-separation directly, not 64996-class classification.

SCOPE:
    - Single file (per Rule 18).
    - Lives under analysis/ (analysis infrastructure, not
      changing 03_spacy_encode/ per Rule 14).
    - Reuses _SupEncoder (imported from 03_spacy_encode).
    - Same SHA1 50/50 profile/test split + 15% val (matches
      stage_strict() protocol so downstream analysis is comparable).

OUTPUTS (replaces stage_strict() outputs):
    pcfg_cache/strict_embeddings.npz (z_profile + z_test)
    pcfg_cache/supervised_embeddings.npy (z_all)
    pcfg_cache/strict_encoder.pt
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")

# Reuse constants from stage_strict (kept identical so protocol matches)
SEED = 42
HASH_SALT = "pcfg_lopo_v1"
SUP_EPOCHS = 20
SUP_BATCH_SIZE = 256
SUP_LR = 5e-4
SUP_Z_DIM = 32
SUP_HIDDEN = (256,)
SUP_DROPOUT = 0.5
SUP_WEIGHT_DECAY = 1e-2
STRICT_VAL_FRAC = 0.15
SUPCON_TAU = 0.07
# User-balanced batch sampler (L8.21 validated)
K_USERS_PER_BATCH = 32
K_PER_USER = 8
# Effective batch size = 32 * 8 = 256

# Smoke config (Rule 18/20: smoke 内嵌到主脚本,SMOKE=True 跑快速 sanity check)
SMOKE = False  # Rule 18: smoke 内嵌,SMOKE=True 跑 sanity check
SMOKE_N_USERS = 200
SMOKE_EPOCHS = 3
SMOKE_DIR = CACHE_DIR / "_smoke"


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

    log("loading cache")
    with open(CACHE_DIR / "meta.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_total_sents = meta["n_total_sents"]
    V = meta["vocab_size"]
    log(f"  n_users={n_users}, n_sents={n_total_sents}, V={V}")

    log("loading sent_vectors.npz (dense path needs full V=31847)")
    t_load = time.time()
    sent_csr = load_npz(CACHE_DIR / "sent_vectors.npz").tocsr()
    log(f"  loaded {sent_csr.shape} in {time.time()-t_load:.1f}s")

    # Per-user sents layout: n_users × ~58 = n_total_sents
    with open(CACHE_DIR / "uid_list.json") as f:
        uid_list = json.load(f)
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)

    # SMOKE subset: take first N users + slice sent_csr accordingly
    if SMOKE:
        SMOKE_DIR.mkdir(exist_ok=True)
        n_users_orig = n_users
        n_users = SMOKE_N_USERS
        user_n_sents = user_n_sents[:SMOKE_N_USERS]
        n_total_sents = int(sum(user_n_sents))
        sent_csr = sent_csr[:n_total_sents].tocsr()
        log(f"  SMOKE subset: users {n_users_orig} -> {n_users}, "
            f"sents -> {n_total_sents}, sent_csr -> {sent_csr.shape}")

    user_off_per_user = np.zeros(n_users + 1, dtype=np.int64)
    user_off_per_user[1:] = np.cumsum(user_n_sents)

    # SHA1 50/50 profile/test split (matches stage_strict protocol)
    log("SHA1 profile/test split")
    sent_id_strs = [
        f"u{uid_list[ui]}_s{i}"
        for ui in range(n_users)
        for i in range(user_n_sents[ui])
    ]
    profile_idx = np.array(
        [i for i, sid in enumerate(sent_id_strs)
         if hash_bucket(sid) == 0], dtype=np.int64)
    test_idx = np.array(
        [i for i, sid in enumerate(sent_id_strs)
         if hash_bucket(sid) == 1], dtype=np.int64)
    log(f"  profile={len(profile_idx)}, test={len(test_idx)}")

    # Per-user profile/train/val split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(profile_idx))
    n_val = int(len(profile_idx) * STRICT_VAL_FRAC)
    val_idx = profile_idx[perm[:n_val]]
    train_idx = profile_idx[perm[n_val:]]
    log(f"  profile_train={len(train_idx)}, profile_val={len(val_idx)}")

    # User labels per profile_idx
    def uid_for(sent_i):
        # sent_i is global index into n_total_sents; find which user owns it
        return int(np.searchsorted(user_off_per_user[1:], sent_i,
                                    side="right"))
    user_labels = np.array([uid_for(int(i)) for i in profile_idx],
                            dtype=np.int64)
    user_labels_train = user_labels[perm[n_val:]]
    user_labels_val = user_labels[perm[:n_val]]

    # User-balanced batch sampler for SupCon
    train_per_user = [[] for _ in range(n_users)]
    for li, idx in enumerate(train_idx):
        ui = int(idx)
        u = uid_for(ui)
        train_per_user[u].append(int(li))  # store local-into-train_idx position
    valid_users = np.array(
        [u for u, arr in enumerate(train_per_user) if len(arr) >= K_PER_USER],
        dtype=np.int64)
    log(f"  user-balanced sampler: {len(valid_users)} users, "
        f"{K_USERS_PER_BATCH}×{K_PER_USER}={K_USERS_PER_BATCH * K_PER_USER}/batch")

    # We need dense x per sent → use sparse @ batch sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    model = _SupEncoder(V, SUP_Z_DIM, SUP_HIDDEN, n_users, SUP_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"encoder params: {n_params/1e6:.2f}M")

    sup_epochs_effective = SMOKE_EPOCHS if SMOKE else SUP_EPOCHS
    if SMOKE:
        log(f"  SMOKE epochs: {sup_epochs_effective} (override SUP_EPOCHS={SUP_EPOCHS})")

    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR,
                           weight_decay=SUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sup_epochs_effective)
    supcon = SupConLoss(tau=SUPCON_TAU).to(device)

    def batch_to_dense(local_idx_arr, source_idx_arr):
        # local_idx_arr: indices into profile_idx (or train_idx)
        # source_idx_arr: the underlying array (profile_idx or train_idx)
        g_idx = source_idx_arr[local_idx_arr]
        sub = sent_csr[g_idx]
        return torch.tensor(sub.toarray(), dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_count = 0
    PATIENCE = 8
    n_user_batches = len(valid_users) // K_USERS_PER_BATCH

    for epoch in range(sup_epochs_effective):
        model.train()
        np.random.shuffle(valid_users)
        train_loss_sum = 0.0
        n_batches = 0
        for ub in range(n_user_batches):
            batch_users = valid_users[ub * K_USERS_PER_BATCH:
                                      (ub + 1) * K_USERS_PER_BATCH]
            bi_local = []
            for u in batch_users:
                arr = np.array(train_per_user[u])
                picks = np.random.choice(arr, size=K_PER_USER, replace=False)
                bi_local.extend(picks.tolist())
            bi_local = np.array(bi_local, dtype=np.int64)
            x = batch_to_dense(bi_local, train_idx)
            y = torch.from_numpy(user_labels_train[bi_local]).to(device)
            z, _ = model(x)
            loss = supcon(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1
        sched.step()
        train_loss_avg = train_loss_sum / max(n_batches, 1)

        # Validation: SupCon loss on val_idx (chunked to avoid OOM)
        model.eval()
        with torch.no_grad():
            val_local = perm[:n_val]
            val_loss_sum = 0.0
            val_n = 0
            VAL_BS = 1024
            for s in range(0, len(val_local), VAL_BS):
                e = min(s + VAL_BS, len(val_local))
                x_v = batch_to_dense(val_local[s:e], profile_idx)
                y_v = torch.from_numpy(user_labels_val[s:e]).to(device)
                z_v, _ = model(x_v)
                l_v = supcon(z_v, y_v)
                val_loss_sum += l_v.item() * (e - s)
                val_n += (e - s)
            val_loss = val_loss_sum / max(val_n, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        log(f"  ep {epoch:>3d}/{SUP_EPOCHS}: "
            f"train_supcon={train_loss_avg:.4f} val_supcon={val_loss:.4f} "
            f"pat={patience_count}/{PATIENCE}")
        if patience_count >= PATIENCE:
            log(f"  early stop @ ep{epoch} (best ep{best_epoch} val={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Encode all sents
    log("encoding all sents (batched dense)")
    model.eval()
    z_all = np.zeros((n_total_sents, SUP_Z_DIM), dtype=np.float32)
    ENCODE_BS = 1024
    t_enc = time.time()
    with torch.no_grad():
        for s in range(0, n_total_sents, ENCODE_BS):
            e = min(s + ENCODE_BS, n_total_sents)
            sub = sent_csr[s:e].toarray().astype(np.float32)
            x = torch.from_numpy(sub).to(device)
            z, _ = model(x)
            z_all[s:e] = z.cpu().numpy()
    log(f"  encoded {n_total_sents} sents in {time.time()-t_enc:.1f}s")

    z_profile = z_all[profile_idx]
    z_test = z_all[test_idx]
    out_dir = SMOKE_DIR if SMOKE else CACHE_DIR
    np.savez(out_dir / "strict_embeddings.npz",
             z_profile=z_profile, z_test=z_test,
             profile_idx=profile_idx, test_idx=test_idx)
    log(f"  wrote strict_embeddings.npz "
        f"(z_profile {z_profile.shape}, z_test {z_test.shape})")
    np.save(out_dir / "supervised_embeddings.npy", z_all)
    log(f"  wrote supervised_embeddings.npy {z_all.shape}")
    torch.save({
        "state_dict": model.state_dict(),
        "V": V,
        "n_users": n_users,
        "z_dim": SUP_Z_DIM,
        "hidden": SUP_HIDDEN,
        "dropout": SUP_DROPOUT,
        "objective": "supcon",
        "tau": SUPCON_TAU,
        "best_epoch": best_epoch,
        "best_val_supcon": best_val,
        "smoke": SMOKE,
    }, out_dir / "strict_encoder.pt")
    log(f"  wrote strict_encoder.pt (best ep{best_epoch} val={best_val:.4f})")

    if SMOKE:
        mean_norm = float(np.linalg.norm(z_all, axis=1).mean())
        log(f"  SMOKE z_all shape={z_all.shape} mean_norm={mean_norm:.3f}")
        if z_all.shape != (n_total_sents, SUP_Z_DIM):
            raise ValueError(f"SMOKE shape mismatch: {z_all.shape} != "
                             f"({n_total_sents}, {SUP_Z_DIM})")
        if not np.isfinite(z_all).all():
            raise ValueError("SMOKE z_all contains NaN/Inf")
        log("SMOKE PASSED")

    log(f"\nALL DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()