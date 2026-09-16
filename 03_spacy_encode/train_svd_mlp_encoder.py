#!/usr/bin/env python3
"""Stage 03b — 2-step 用户风格 encoder: TruncatedSVD → MLP(contrastive)。

输入:
  - pcfg_cache/sent_vectors.npz (CSR, 4.34M x V=27647, sparse rule-count)
  - pcfg_cache/strict3_embeddings.npz (profile/val/test 切分 + uid_list)

流程:
  1) 用 profile_idx 切出 profile 子稀疏矩阵 Xp (3.47M x V)
  2) TruncatedSVD: V=27647 -> 256, 在 Xp 上 fit_transform
  3) 256 -> 128 -> 64 MLP, 用 user-id 对比学习训练
  4) freeze MLP, 对所有 n_sents=4.34M 句子编码 -> z_all (4.34M, 64)
  5) 输出与 strict3_embeddings.npz 兼容的 schema (但 z_dim=64)

输出:
  - pcfg_cache/svd_components.npz        (V, 256) SVD V components
  - pcfg_cache/svd_mlp_encoder.pt        (MLP 权重 + config)
  - pcfg_cache/svd_mlp_embeddings.npz    {z_profile, z_val, z_test, z_dim=64}

使用:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        03_spacy_encode/train_svd_mlp_encoder.py \
        > /home/wlia0047/hj82_scratch2/wenyu/svd_mlp_train.log 2>&1 &
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import TruncatedSVD

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
SENT_VECTORS_NPZ = CACHE_DIR / "sent_vectors.npz"
STRICT3_NPZ = CACHE_DIR / "strict3_embeddings.npz"

OUT_SVD = CACHE_DIR / "svd_components.npz"
OUT_MLP_PT = CACHE_DIR / "svd_mlp_encoder.pt"
OUT_EMB_NPZ = CACHE_DIR / "svd_mlp_embeddings.npz"

# ---- 硬编码配置 ----
SVD_DIM = 256
MLP_HIDDEN = 128
MLP_OUT = 64
N_EPOCHS = 8
BATCH_SENTS = 4096
N_NEG = 64
LR = 3e-3
TEMPERATURE = 0.1
SVD_N_ITER = 7
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SVD_SAMPLE_N = 2_000_000
MLP_SAMPLE_N = 1_500_000
ROW_NORMALIZE = True
BATCHES_PER_EPOCH = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sparse() -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    log(f"loading {SENT_VECTORS_NPZ}")
    X = sp.load_npz(SENT_VECTORS_NPZ).astype(np.float32)
    log(f"  X: {X.shape} nnz={X.nnz} density={X.nnz/(X.shape[0]*X.shape[1]):.5f}")
    strict3 = np.load(STRICT3_NPZ, allow_pickle=False)
    uid_list = [str(u) for u in strict3["uid_list"]]
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = [int(n) for n in json.load(f)]
    uid_per_row = np.repeat(
        np.arange(len(uid_list), dtype=np.int64), user_n_sents
    )
    assert uid_per_row.shape[0] == X.shape[0]
    return X, uid_per_row, uid_list


def fit_svd(X: sp.csr_matrix, sample_n: int) -> tuple[np.ndarray, np.ndarray]:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    if sample_n < X.shape[0]:
        idx = rng.choice(X.shape[0], size=sample_n, replace=False)
        Xs = X[idx]
        log(f"  SVD fit on subsample: {Xs.shape}")
    else:
        Xs = X
    if ROW_NORMALIZE:
        t0 = time.time()
        norms = np.asarray(sp.linalg.norm(Xs, axis=1)).ravel()
        norms[norms == 0] = 1.0
        Xs = sp.diags(1.0 / norms) @ Xs
        log(f"    normalize done {time.time() - t0:.1f}s")
    svd = TruncatedSVD(
        n_components=SVD_DIM, n_iter=SVD_N_ITER,
        algorithm="randomized", random_state=SEED,
    )
    t0 = time.time()
    svd.fit(Xs)
    log(f"  SVD fit: components={svd.components_.shape} "
        f"evr_sum={svd.explained_variance_ratio_.sum():.4f} "
        f"elapsed={time.time() - t0:.1f}s")
    return svd.components_.astype(np.float32), svd.explained_variance_ratio_.astype(np.float32)


def project_to_svd(X: sp.csr_matrix, Vt: np.ndarray,
                   batch_rows: int = 500_000) -> np.ndarray:
    N = X.shape[0]
    out = np.empty((N, SVD_DIM), dtype=np.float32)
    t0 = time.time()
    if ROW_NORMALIZE:
        norms = np.asarray(sp.linalg.norm(X, axis=1)).ravel()
        norms[norms == 0] = 1.0
    for s in range(0, N, batch_rows):
        e = min(s + batch_rows, N)
        chunk = X[s:e]
        if ROW_NORMALIZE:
            chunk = sp.diags(1.0 / norms[s:e]) @ chunk
        out[s:e] = (chunk @ Vt.T).astype(np.float32)
    log(f"  full projection: {out.shape}  total={time.time() - t0:.1f}s")
    return out


class StyleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def info_nce_loss(anchor, positive, negative, temperature):
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / temperature
    neg_sim = torch.einsum("bd,bnd->bn", anchor, negative) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=-1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def encode_all(mlp: StyleMLP, z_svd_all: np.ndarray,
               batch_size: int = 16_384) -> np.ndarray:
    mlp.eval()
    out = np.empty((z_svd_all.shape[0], MLP_OUT), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, z_svd_all.shape[0], batch_size):
            e = min(s + batch_size, z_svd_all.shape[0])
            x = torch.from_numpy(z_svd_all[s:e]).to(DEVICE)
            out[s:e] = mlp(x).cpu().numpy()
    return out


def main() -> None:
    set_seed(SEED)
    log(f"device={DEVICE}  SVD_DIM={SVD_DIM}  MLP_OUT={MLP_OUT}")

    X, uid_per_row, uid_list = load_sparse()

    log("=== Step 1: TruncatedSVD on profile rows ===")
    t0 = time.time()
    Vt, evr = fit_svd(X, sample_n=SVD_SAMPLE_N)
    np.savez_compressed(
        OUT_SVD, Vt=Vt, explained_variance_ratio=evr,
        svd_dim=SVD_DIM, row_normalize=ROW_NORMALIZE,
    )
    log(f"  saved {OUT_SVD}")

    log("=== Step 2: project all 4.34M rows through SVD ===")
    z_svd_all = project_to_svd(X, Vt)
    del X
    log(f"  z_svd_all: {z_svd_all.shape}  total={time.time()-t0:.1f}s")

    log("=== Step 3: train contrastive MLP ===")
    strict3 = np.load(STRICT3_NPZ, allow_pickle=False)
    profile_idx = np.asarray(strict3["profile_idx"], dtype=np.int64)
    val_idx = np.asarray(strict3["val_idx"], dtype=np.int64)
    test_idx = np.asarray(strict3["test_idx"], dtype=np.int64)

    z_svd_profile = z_svd_all[profile_idx]
    uid_profile = uid_per_row[profile_idx]
    log(f"  z_svd_profile: {z_svd_profile.shape}  uid unique={len(np.unique(uid_profile))}")

    mlp = StyleMLP(SVD_DIM, MLP_HIDDEN, MLP_OUT).to(DEVICE)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)

    for epoch in range(N_EPOCHS):
        if MLP_SAMPLE_N < z_svd_profile.shape[0]:
            sample_idx = rng.choice(
                z_svd_profile.shape[0], size=MLP_SAMPLE_N, replace=False
            )
            zp_sub = z_svd_profile[sample_idx]
            uid_sub = uid_profile[sample_idx]
        else:
            zp_sub = z_svd_profile
            uid_sub = uid_profile
        log(f"  epoch {epoch+1}/{N_EPOCHS}  sample_z={zp_sub.shape}")

        # 按 uid 聚合
        uid_to_rows_sub: dict[int, np.ndarray] = {}
        order = np.argsort(uid_sub, kind="stable")
        sorted_uids = uid_sub[order]
        offsets = np.concatenate(
            ([0], np.cumsum(np.bincount(sorted_uids, minlength=len(uid_list))))
        )
        for u in range(len(uid_list)):
            if offsets[u + 1] > offsets[u]:
                uid_to_rows_sub[u] = order[offsets[u]:offsets[u + 1]]
        valid = np.array(
            [u for u, rs in uid_to_rows_sub.items() if len(rs) >= 2],
            dtype=np.int64,
        )
        log(f"    trainable users: {len(valid)}")

        # === 全向量化: flat row table + schedule ===
        valid_arr = valid
        row_per_uid = uid_to_rows_sub
        flat_offsets = np.concatenate(
            ([0], np.cumsum([len(row_per_uid[int(u)]) for u in valid_arr]))
        )
        total_rows = int(flat_offsets[-1])
        flat_rows = np.empty(total_rows, dtype=np.int64)
        for i, u in enumerate(valid_arr):
            flat_rows[flat_offsets[i]:flat_offsets[i + 1]] = row_per_uid[int(u)]
        row_lens_valid = np.diff(flat_offsets).astype(np.int64)

        # schedule: 整 epoch 的 anchor/neg uid (一次生成, 整 epoch 复用)
        schedule_anchor_uids = rng.integers(
            0, len(valid_arr), size=(BATCHES_PER_EPOCH, BATCH_SENTS)
        ).astype(np.int64)
        schedule_neg_uids = rng.integers(
            0, len(valid_arr),
            size=(BATCHES_PER_EPOCH, BATCH_SENTS, N_NEG),
        ).astype(np.int64)
        eq_mask = (schedule_neg_uids == schedule_anchor_uids[:, :, None])
        schedule_neg_uids = np.where(
            eq_mask, (schedule_neg_uids + 1) % len(valid_arr), schedule_neg_uids
        )

        epoch_loss = 0.0
        n_batches = 0
        t_ep = time.time()

        for bi in range(BATCHES_PER_EPOCH):
            anchor_uids = schedule_anchor_uids[bi]                # (B,)
            neg_uid_idx = schedule_neg_uids[bi]                   # (B, N_NEG)

            # anchor/positive: 同 uid 抽 2 个不同 row (vectorized)
            u_lens = row_lens_valid[anchor_uids]
            seg_offsets = flat_offsets[anchor_uids]
            u_lens_safe = np.maximum(u_lens, 1)
            r1 = rng.integers(0, u_lens_safe)
            r2 = (r1 + 1 + rng.integers(0, np.maximum(u_lens - 1, 1))) % u_lens_safe
            anchor_rows = flat_rows[seg_offsets + r1]
            positive_rows = flat_rows[seg_offsets + r2]

            # neg rows
            neg_u_lens = row_lens_valid[neg_uid_idx]
            neg_seg_offsets = flat_offsets[neg_uid_idx]
            neg_u_lens_safe = np.maximum(neg_u_lens, 1)
            neg_r = rng.integers(0, neg_u_lens_safe)
            neg_rows = flat_rows[neg_seg_offsets + neg_r]

            z_anchor = torch.from_numpy(zp_sub[anchor_rows]).to(DEVICE)
            z_pos = torch.from_numpy(zp_sub[positive_rows]).to(DEVICE)
            z_neg_svd = torch.from_numpy(
                zp_sub[neg_rows.reshape(-1)]
            ).to(DEVICE).view(BATCH_SENTS, N_NEG, SVD_DIM)

            anchor_emb = mlp(z_anchor)
            pos_emb = mlp(z_pos)
            neg_emb = mlp(z_neg_svd.view(-1, SVD_DIM)).view(
                BATCH_SENTS, N_NEG, MLP_OUT
            )

            loss = info_nce_loss(anchor_emb, pos_emb, neg_emb, TEMPERATURE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        log(f"    avg loss={epoch_loss/max(n_batches,1):.4f}  "
            f"elapsed={time.time()-t_ep:.1f}s")

    log("=== Step 4: encode all rows through MLP ===")
    z_all = encode_all(mlp, z_svd_all)
    log(f"  z_all: {z_all.shape}")

    log("=== Step 5: save svd_mlp_embeddings.npz ===")
    z_profile = z_all[profile_idx]
    z_val = z_all[val_idx]
    z_test = z_all[test_idx]
    np.savez_compressed(
        OUT_EMB_NPZ,
        z_profile=z_profile.astype(np.float32),
        z_val=z_val.astype(np.float32),
        z_test=z_test.astype(np.float32),
        profile_idx=profile_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        uid_list=np.asarray(uid_list, dtype=object),
        z_dim=np.int64(MLP_OUT),
        svd_dim=np.int64(SVD_DIM),
        cohort_fingerprint=strict3["cohort_fingerprint"],
        uid_layout_fingerprint=strict3["uid_layout_fingerprint"],
        vocab_fingerprint=strict3["vocab_fingerprint"],
    )
    log(f"  wrote {OUT_EMB_NPZ}")

    torch.save({
        "state_dict": mlp.state_dict(),
        "config": {
            "in_dim": SVD_DIM, "hidden": MLP_HIDDEN, "out_dim": MLP_OUT,
            "n_epochs": N_EPOCHS, "temperature": TEMPERATURE, "lr": LR,
        },
    }, OUT_MLP_PT)
    log(f"  wrote {OUT_MLP_PT}")
    log(f"DONE  total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
